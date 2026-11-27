# core/deconvolution.py
"""
core/deconvolution.py - Motor de Deconvolución Óptica Regularizada (Richardson-Lucy / PSF Estimator)
en AstroLoom.

Implementa:
1. Modelado matemático de PSF sintética (Moffat 2D astronómico y Gaussiano 2D) con soporte de
   FWHM subpíxel, elipticidad (relación de aspecto) y ángulo de posición.
2. Estimador automático de PSF a partir del análisis estadístico de estrellas aisladas no saturadas.
3. Algoritmo iterativo de Richardson-Lucy (R-L) en espacio de Luminancia perceptual (ITU-R BT.709)
   para preservar la fidelidad cromática espectral.
4. Supresión estricta de artefactos:
   - Protección antianillos (*Local Deringing Support*) para erradicar halos oscuros ("donuts").
   - Amortiguación de ruido en el fondo (*Bayesian Noise Damping*) basada en dispersión MAD.
5. Ámbitos selectivos de aplicación (Imagen completa, Solo Estrellas o Solo Fondo/Nebulosa).
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable
import astroloom_rs

HAS_RUST = True


def generate_moffat_psf(
    fwhm: float = 3.0,
    beta: float = 4.5,
    aspect_ratio: float = 1.0,
    angle_deg: float = 0.0,
    kernel_size: Optional[int] = None
) -> np.ndarray:
    """Genera un kernel de PSF bidimensional según el perfil astrofísico de Moffat (Rust)."""
    return astroloom_rs.generate_moffat_psf_rs(
        float(fwhm), float(beta), float(aspect_ratio), float(angle_deg),
        int(kernel_size) if kernel_size is not None else None
    )


def generate_gaussian_psf(
    fwhm: float = 3.0,
    aspect_ratio: float = 1.0,
    angle_deg: float = 0.0,
    kernel_size: Optional[int] = None
) -> np.ndarray:
    """Genera un kernel de PSF bidimensional gaussiano (Rust)."""
    return astroloom_rs.generate_gaussian_psf_rs(
        float(fwhm), float(aspect_ratio), float(angle_deg),
        int(kernel_size) if kernel_size is not None else None
    )


def _measure_star_fwhm_1d(profile: np.ndarray) -> float:
    """Calcula el ancho a media altura (FWHM) por interpolación lineal fraccional en 1D."""
    peak = float(profile.max())
    bg = min(float(profile[0]), float(profile[-1]))
    amp = peak - bg
    if amp < 0.015:
        return 0.0

    hm = bg + amp / 2.0
    above = np.where(profile >= hm)[0]
    if len(above) < 1 or above[0] == 0 or above[-1] == len(profile) - 1:
        return 0.0

    i_l = above[0]
    denom_l = max(1e-5, profile[i_l] - profile[i_l - 1])
    x_l = (i_l - 1) + (hm - profile[i_l - 1]) / denom_l

    i_r = above[-1]
    denom_r = max(1e-5, profile[i_r] - profile[i_r + 1])
    x_r = i_r + (profile[i_r] - hm) / denom_r

    fwhm = x_r - x_l
    return float(fwhm) if fwhm > 0.5 else 0.0


def estimate_psf_from_stars(
    image_f32: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    max_stars: int = 60
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detecta automáticamente estrellas aisladas no saturadas y estima la PSF empírica del tren óptico.
    
    Devuelve: (psf_kernel_moffat, stats_dict).
    """
    if image_f32.ndim == 3:
        # Luminancia perceptual BT.709
        gray = 0.2126 * image_f32[:, :, 0] + 0.7152 * image_f32[:, :, 1] + 0.0722 * image_f32[:, :, 2]
    else:
        gray = image_f32.copy()

    h, w = gray.shape[:2]

    # Máscara morfológica Top-Hat para aislamiento de estrellas
    k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_th)

    # Filtrar candidatos: ni saturadas (>0.95) ni ruido insignificante (<0.01)
    dilated = cv2.dilate(tophat, k_th)
    peaks = (tophat == dilated) & (tophat > 0.01) & (gray < 0.95)
    if sky_mask is not None:
        sm = sky_mask if sky_mask.shape[:2] == (h, w) else cv2.resize(sky_mask, (w, h))
        peaks &= (sm > 0.5)

    ys, xs = np.where(peaks)
    if len(xs) == 0:
        # Fallback a PSF estándar
        default_psf = generate_moffat_psf(fwhm=3.0, beta=4.5)
        return default_psf, {"fwhm": 3.0, "aspect_ratio": 1.0, "angle_deg": 0.0, "stars_sampled": 0, "fallback": True}

    # Ordenar candidatos por brillo de pico decreciente
    intensities = gray[ys, xs]
    order = np.argsort(-intensities)
    xs = xs[order]
    ys = ys[order]

    fwhm_list: List[float] = []
    ratio_list: List[float] = []

    rad = 10
    for x0, y0 in zip(xs, ys):
        if x0 < rad or x0 >= w - rad or y0 < rad or y0 >= h - rad:
            continue

        patch = gray[y0 - rad:y0 + rad + 1, x0 - rad:x0 + rad + 1]

        # Comprobar aislamiento: el borde debe ser cielo de fondo oscuro
        border_vals = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
        bg_med = float(np.median(border_vals))
        peak_val = float(patch[rad, rad])
        if (peak_val - bg_med) < 0.015:
            continue

        fx = _measure_star_fwhm_1d(patch[rad, :])
        fy = _measure_star_fwhm_1d(patch[:, rad])

        if 1.4 <= fx <= 8.5 and 1.4 <= fy <= 8.5:
            mean_f = float(np.sqrt(fx * fy))
            ratio = float(max(fx, fy) / max(0.5, min(fx, fy)))
            fwhm_list.append(mean_f)
            ratio_list.append(ratio)

            if len(fwhm_list) >= max_stars:
                break

    if len(fwhm_list) >= 4:
        est_fwhm = float(np.median(fwhm_list))
        est_ratio = float(np.median(ratio_list))
        est_ratio = max(1.0, min(est_ratio, 2.0))
        stars_sampled = len(fwhm_list)
        fallback = False
    else:
        est_fwhm = 3.0
        est_ratio = 1.0
        stars_sampled = len(fwhm_list)
        fallback = True

    fitted_psf = generate_moffat_psf(fwhm=est_fwhm, beta=4.5, aspect_ratio=est_ratio)

    stats = {
        "fwhm": round(est_fwhm, 2),
        "aspect_ratio": round(est_ratio, 2),
        "angle_deg": 0.0,
        "stars_sampled": stars_sampled,
        "fallback": fallback
    }

    return fitted_psf, stats


def richardson_lucy_deconvolve(
    image_rgb: np.ndarray,
    psf_kernel: np.ndarray,
    iterations: int = 15,
    deringing: float = 0.008,
    damping: float = 0.25,
    luminance_only: bool = True,
    star_mask: Optional[np.ndarray] = None,
    target_scope: str = "all",
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None,
    use_gpu: bool = True,
    global_sigma_noise: Optional[float] = None,
) -> np.ndarray:
    """
    Aplica deconvolución de Richardson-Lucy (R-L) con regularización bayesiana y protección antianillos.
    
    :param image_rgb: Matriz float32 [0..1] de 3 canales (RGB) o monocromo.
    :param psf_kernel: Matriz float32 de la PSF (suma = 1.0).
    :param iterations: Número de iteraciones R-L (10..35 recomendado).
    :param deringing: Factor de suelo local para evitar halos oscuros en estrellas (0.0 a 0.05).
    :param damping: Amortiguación de ruido en el fondo oscuro (0.0 = sin amortiguar, 1.0 = máxima amortiguación).
    :param luminance_only: Si es True, deconvoluciona solo la luminancia para preservar la pureza espectral.
    :param star_mask: Máscara binaria opcional de estrellas (1.0 = estrella, 0.0 = fondo).
    :param target_scope: 'all' (imagen completa), 'stars_only' (solo estrellas), 'background_only' (solo nebulosa).
    :param use_gpu: Si es True, utiliza aceleración por hardware (OpenCL / UMat) con respaldo automático en CPU.
    """
    iterations = max(1, min(int(iterations), 60))
    deringing = max(0.0, min(float(deringing), 0.1))
    damping = max(0.0, min(float(damping), 1.0))

    h_img, w_img = image_rgb.shape[:2]
    is_color = (image_rgb.ndim == 3 and image_rgb.shape[2] == 3)
    if is_color and luminance_only:
        lum = 0.2126 * image_rgb[:, :, 0] + 0.7152 * image_rgb[:, :, 1] + 0.0722 * image_rgb[:, :, 2]
        work_channel = np.clip(lum, 0.0, 1.0).astype(np.float32)
    else:
        work_channel = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    # Estimación del nivel de ruido (MAD) para amortiguación bayesiana
    if damping > 0.0:
        if global_sigma_noise is not None:
            sigma_noise = float(global_sigma_noise)
        else:
            ref_2d = work_channel if work_channel.ndim == 2 else (
                0.2126 * work_channel[:, :, 0] + 0.7152 * work_channel[:, :, 1] + 0.0722 * work_channel[:, :, 2]
            )
            med_smooth = cv2.medianBlur((ref_2d * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
            high_freq = np.abs(ref_2d - med_smooth)
            hf_sample = high_freq[::4, ::4] if high_freq.size >= 16384 else high_freq
            sigma_noise = float(1.4826 * np.median(hf_sample))
        noise_threshold_sq = float((damping * max(0.002, sigma_noise)) ** 2)
    else:
        noise_threshold_sq = 0.0

    psf_c = np.ascontiguousarray(psf_kernel, dtype=np.float32)

    def _run_rl_2d(channel_2d: np.ndarray, ch_prefix: str = "") -> np.ndarray:
        work_c = np.ascontiguousarray(channel_2d, dtype=np.float32)
        if deringing > 0.0 and np.any(work_c > 0.65):
            k_star = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            sm_c = np.ascontiguousarray(cv2.dilate((work_c > 0.65).astype(np.float32), k_star), dtype=np.float32)
            df_c = np.ascontiguousarray(work_c * (1.0 - deringing), dtype=np.float32)
        else:
            sm_c = None
            df_c = None

        step = 2 if (progress_callback or abort_flag) else iterations
        done_it = 0
        curr_est = None
        while done_it < iterations:
            if abort_flag and abort_flag():
                raise InterruptedError("Deconvolución cancelada por el usuario.")
            chunk_it = min(step, iterations - done_it)
            curr_est = astroloom_rs.richardson_lucy_deconv_rs(
                work_c,
                psf_c,
                iterations=chunk_it,
                noise_threshold_sq=float(noise_threshold_sq),
                bright_star_mask=sm_c,
                deringing_floor=df_c,
                est_initial=curr_est,
            )
            done_it += chunk_it
            if progress_callback and (done_it % 2 == 0 or done_it == iterations):
                pct = int((done_it / float(iterations)) * 100.0)
                progress_callback(pct, f"Deconvolución R-L FFT (Rust{ch_prefix}): iteración {done_it}/{iterations}")
        return curr_est

    if work_channel.ndim == 3:
        est = np.empty_like(work_channel)
        for ch, cname in enumerate((" R", " G", " B")):
            est[:, :, ch] = _run_rl_2d(work_channel[:, :, ch], cname)
    else:
        est = _run_rl_2d(work_channel)

    # Filtrado por ámbito (Solo estrellas / Solo nebulosa)
    if target_scope in ("stars_only", "background_only") and star_mask is not None:
        sm = star_mask if star_mask.shape[:2] == (h_img, w_img) else cv2.resize(star_mask, (w_img, h_img))
        if sm.ndim == 3:
            sm = sm[:, :, 0]
        sm_b = sm[..., np.newaxis] if est.ndim == 3 else sm
        if target_scope == "stars_only":
            est = (est * sm_b) + (work_channel * (1.0 - sm_b))
        else:
            est = (est * (1.0 - sm_b)) + (work_channel * sm_b)

    # Reconstrucción final con preservación estricta de color
    if is_color and luminance_only:
        return astroloom_rs.reconstruct_deconv_color_rs(
            np.ascontiguousarray(image_rgb, dtype=np.float32),
            np.ascontiguousarray(est, dtype=np.float32),
            np.ascontiguousarray(work_channel, dtype=np.float32),
            star_mask=None,
            target_scope="all",
        )
    return np.clip(est, 0.0, 1.0).astype(np.float32)
