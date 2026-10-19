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
import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Callable
from core.gpu_backend import is_gpu_enabled, is_opencl_available


def generate_moffat_psf(
    fwhm: float = 3.0,
    beta: float = 4.5,
    aspect_ratio: float = 1.0,
    angle_deg: float = 0.0,
    kernel_size: Optional[int] = None
) -> np.ndarray:
    """
    Genera un kernel de PSF bidimensional según el perfil astrofísico de Moffat.
    
    :param fwhm: Diámetro a mitad de altura en píxeles.
    :param beta: Parámetro de pendiente de alas (en astrofotografía suele estar entre 3.5 y 5.0).
    :param aspect_ratio: Relación mayor/menor para modelar deriva de guiado o coma (>= 1.0).
    :param angle_deg: Ángulo de orientación de la elipse en grados.
    :param kernel_size: Tamaño impar de la matriz (si es None, se calcula automáticamente).
    :return: Matriz float32 normalizada a suma = 1.0.
    """
    fwhm = max(1.0, float(fwhm))
    beta = max(1.1, float(beta))
    aspect_ratio = max(1.0, min(float(aspect_ratio), 3.0))

    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))

    if kernel_size is None:
        rad = int(np.ceil(fwhm * 3.5))
        rad = max(7, min(rad, 25))
        ksize = 2 * rad + 1
    else:
        ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        rad = ksize // 2

    y, x = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(np.float64)

    if angle_deg != 0.0:
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xr = x * cos_t + y * sin_t
        yr = -x * sin_t + y * cos_t
    else:
        xr, yr = x, y

    ax = alpha * np.sqrt(aspect_ratio)
    ay = alpha / np.sqrt(aspect_ratio)

    r_sq = (xr / ax) ** 2 + (yr / ay) ** 2
    psf = (1.0 + r_sq) ** (-beta)

    s = np.sum(psf)
    if s > 0:
        psf /= s
    else:
        psf = np.zeros_like(psf)
        psf[rad, rad] = 1.0

    return psf.astype(np.float32)


def generate_gaussian_psf(
    fwhm: float = 3.0,
    aspect_ratio: float = 1.0,
    angle_deg: float = 0.0,
    kernel_size: Optional[int] = None
) -> np.ndarray:
    """
    Genera un kernel de PSF bidimensional gaussiano.
    """
    fwhm = max(1.0, float(fwhm))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~ FWHM / 2.35482
    aspect_ratio = max(1.0, min(float(aspect_ratio), 3.0))

    if kernel_size is None:
        rad = int(np.ceil(sigma * 3.5))
        rad = max(7, min(rad, 25))
        ksize = 2 * rad + 1
    else:
        ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        rad = ksize // 2

    y, x = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(np.float64)

    if angle_deg != 0.0:
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xr = x * cos_t + y * sin_t
        yr = -x * sin_t + y * cos_t
    else:
        xr, yr = x, y

    sx = sigma * np.sqrt(aspect_ratio)
    sy = sigma / np.sqrt(aspect_ratio)

    psf = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    s = np.sum(psf)
    if s > 0:
        psf /= s
    else:
        psf = np.zeros_like(psf)
        psf[rad, rad] = 1.0

    return psf.astype(np.float32)


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
    use_gpu: bool = True
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
    psf_flipped = cv2.flip(psf_kernel, -1)

    # 0. Configuración de aceleración GPU (OpenCL Transparent API / UMat)
    use_opencl = use_gpu and is_gpu_enabled() and is_opencl_available()
    psf_umat = None
    psf_flip_umat = None

    # 1. Separación de canal de trabajo (Luminancia)
    is_color = (image_rgb.ndim == 3 and image_rgb.shape[2] == 3)
    if is_color and luminance_only:
        # Luminancia perceptual BT.709
        lum = 0.2126 * image_rgb[:, :, 0] + 0.7152 * image_rgb[:, :, 1] + 0.0722 * image_rgb[:, :, 2]
        lum = np.clip(lum, 0.0, 1.0).astype(np.float32)
        work_channel = lum
    else:
        work_channel = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    # 2. Estimación del nivel de ruido del sensor mediante MAD (Median Absolute Deviation)
    # en altas frecuencias para la amortiguación bayesiana
    if damping > 0.0:
        med_smooth = cv2.medianBlur((work_channel * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
        high_freq = np.abs(work_channel - med_smooth)
        sigma_noise = float(1.4826 * np.median(high_freq))
        noise_threshold_sq = float((damping * max(0.002, sigma_noise)) ** 2)
    else:
        noise_threshold_sq = 0.0

    # 3. Soporte local de Deringing (máscara de protección de núcleos estelares)
    if deringing > 0.0:
        k_star = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_star_mask = cv2.dilate((work_channel > 0.65).astype(np.float32), k_star)
        deringing_floor = work_channel * (1.0 - deringing)
    else:
        bright_star_mask = None
        deringing_floor = None

    # Inicialización de tensores persistentes en VRAM para evitar saturación del bus PCIe
    u_work = None
    u_est = None
    eps_umat = None
    ones_umat = None
    damp_const_umat = None
    u_star_mask = None
    u_inv_star_mask = None
    u_deringing_floor = None

    if use_opencl:
        try:
            psf_umat = cv2.UMat(psf_kernel)
            psf_flip_umat = cv2.UMat(psf_flipped)
            u_work = cv2.UMat(work_channel)
            u_est = cv2.UMat(work_channel)
            eps_umat = cv2.UMat(np.full_like(work_channel, 1e-6))
            ones_umat = cv2.UMat(np.ones_like(work_channel))
            if noise_threshold_sq > 0.0:
                damp_const_umat = cv2.UMat(np.full_like(work_channel, noise_threshold_sq + 1e-7))
            if bright_star_mask is not None and deringing_floor is not None:
                u_star_mask = cv2.UMat(bright_star_mask)
                u_inv_star_mask = cv2.UMat(1.0 - bright_star_mask)
                u_deringing_floor = cv2.UMat(deringing_floor)
        except Exception:
            use_opencl = False

    # 4. Bucle iterativo Richardson-Lucy
    est = work_channel.copy()

    for it in range(iterations):
        if abort_flag and abort_flag():
            raise InterruptedError("Deconvolución cancelada por el usuario.")

        if progress_callback and (it % 2 == 0 or it == iterations - 1):
            pct = int(((it + 1) / float(iterations)) * 100.0)
            progress_callback(pct, f"Deconvolución Richardson-Lucy: iteración {it + 1}/{iterations}")

        # Convolución y corrección persistente en VRAM (OpenCL UMat con fallback automático a CPU)
        if use_opencl:
            try:
                u_conv = cv2.filter2D(u_est, -1, psf_umat, borderType=cv2.BORDER_REFLECT)
                u_ratio = cv2.divide(u_work, cv2.add(u_conv, eps_umat))
                u_bproj = cv2.filter2D(u_ratio, -1, psf_flip_umat, borderType=cv2.BORDER_REFLECT)
                if noise_threshold_sq > 0.0:
                    u_delta = cv2.subtract(u_bproj, ones_umat)
                    u_delta_sq = cv2.multiply(u_delta, u_delta)
                    u_denom = cv2.add(u_delta_sq, damp_const_umat)
                    u_damp_w = cv2.divide(u_delta_sq, u_denom)
                    u_update = cv2.add(ones_umat, cv2.multiply(u_delta, u_damp_w))
                else:
                    u_update = u_bproj
                u_est = cv2.multiply(u_est, u_update)
                u_est = cv2.min(cv2.max(u_est, 0.0), 1.0)
                if u_star_mask is not None and u_deringing_floor is not None:
                    u_clamped = cv2.max(u_est, u_deringing_floor)
                    u_est = cv2.add(cv2.multiply(u_clamped, u_star_mask), cv2.multiply(u_est, u_inv_star_mask))
            except Exception:
                use_opencl = False
                est = u_est.get() if u_est is not None else work_channel.copy()

        if not use_opencl:
            conv = cv2.filter2D(est, -1, psf_kernel, borderType=cv2.BORDER_REFLECT)
            ratio = work_channel / (conv + 1e-6)
            b_proj = cv2.filter2D(ratio, -1, psf_flipped, borderType=cv2.BORDER_REFLECT)
            delta = b_proj - 1.0
            if noise_threshold_sq > 0.0:
                delta_sq = delta ** 2
                damp_weight = delta_sq / (delta_sq + noise_threshold_sq + 1e-7)
                update_factor = 1.0 + delta * damp_weight
            else:
                update_factor = b_proj
            est = np.clip(est * update_factor, 0.0, 1.0)
            if bright_star_mask is not None and deringing_floor is not None:
                est = np.where(bright_star_mask > 0.1, np.maximum(est, deringing_floor), est)

    if use_opencl:
        est = u_est.get()

    # 5. Filtrado por ámbito (Solo estrellas / Solo nebulosa) si corresponde
    if target_scope in ["stars_only", "background_only"] and star_mask is not None:
        sm = star_mask if star_mask.shape[:2] == (h_img, w_img) else cv2.resize(star_mask, (w_img, h_img))
        if sm.ndim == 3:
            sm = sm[:, :, 0]
        if target_scope == "stars_only":
            est = (est * sm) + (work_channel * (1.0 - sm))
        else:  # background_only
            est = (est * (1.0 - sm)) + (work_channel * sm)

    # 6. Reconstrucción final con preservación estricta de color
    if is_color and luminance_only:
        ratio_rgb = est / (work_channel + 1e-6)
        out_rgb = image_rgb * ratio_rgb[..., np.newaxis]
        return np.clip(out_rgb, 0.0, 1.0).astype(np.float32)
    else:
        return np.clip(est, 0.0, 1.0).astype(np.float32)
