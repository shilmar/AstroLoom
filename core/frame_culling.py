# core/frame_culling.py
"""
core/frame_culling.py - Módulo de inspección de calidad estelar y cribado inteligente de tomas (Frame Culling).

Evalúa métricas ópticas objetivas sobre cada fotograma antes del apilado:
- FWHM (Full Width at Half Maximum) / Nitidez de enfoque.
- Excentricidad / Elongación estelar (detección de deriva por viento o trípode golpeado).
- Densidad estelar y relación señal/ruido (detección de nubes o niebla).
- Puntuación de calidad normalizada (0 - 100%) y clasificación semafórica.
"""

import os
import math
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable

import cv2
import numpy as np


@dataclass
class FrameQualityResult:
    """Resultado del análisis de calidad óptica para una sola toma."""
    filepath: str
    filename: str
    fwhm: float            # FWHM en píxeles (mediana de estrellas analizadas)
    eccentricity: float    # Excentricidad estelar [0.0 = círculo perfecto, 1.0 = trazo lineal]
    star_count: int        # Número total de estrellas detectadas en el cielo
    score: float           # Puntuación global de calidad [0.0 a 100.0]
    status: str            # 'EXCELLENT', 'ACCEPTABLE', 'DEFECTIVE'
    status_label: str      # Texto amigable con icono para la GUI
    defect_reason: str     # Razón del defecto si aplica
    is_recommended: bool   # True si se recomienda incluir en el apilado
    bg_noise: float = 0.0  # Nivel de ruido del fondo


def load_fast_frame_preview(filepath: str, max_dim: int = 1600) -> Optional[np.ndarray]:
    """
    Carga rápida de imagen optimizada para análisis espectral/estelar.
    Para archivos RAW utiliza half_size=True de LibRaw (aceleración 4x-6x).
    Retorna array 2D float32 [0.0, 1.0] en escala de grises.
    """
    if not os.path.exists(filepath):
        return None

    ext = os.path.splitext(filepath)[1].lower()
    raw_exts = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raw", ".raf", ".orf", ".rw2"}

    try:
        if ext in raw_exts:
            try:
                import rawpy
                with rawpy.imread(filepath) as raw:
                    # Half-size es ultra rápido y conserva la geometría y redondez de estrellas
                    rgb = raw.postprocess(
                        half_size=True,
                        no_auto_bright=True,
                        use_camera_wb=True,
                        output_bps=8
                    )
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    return gray.astype(np.float32) / 255.0
            except Exception:
                pass

        if ext in {".fits", ".fit", ".fts"}:
            try:
                from astropy.io import fits
                with fits.open(filepath, memmap=True) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim in (2, 3):
                            data = hdu.data.astype(np.float32)
                            if data.ndim == 3:
                                data = np.mean(data[:3], axis=0)
                            p_hi = float(np.percentile(data, 99.8))
                            p_lo = float(np.percentile(data, 1.0))
                            rng = max(1e-5, p_hi - p_lo)
                            norm = np.clip((data - p_lo) / rng, 0.0, 1.0)
                            h, w = norm.shape[:2]
                            if max(h, w) > max_dim:
                                scale = max_dim / max(h, w)
                                norm = cv2.resize(norm, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                            return norm
            except Exception:
                pass

        # Formatos estándar (TIFF, PNG, JPG)
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        max_v = 65535.0 if img.dtype == np.uint16 else (1.0 if img.dtype in (np.float32, np.float64) else 255.0)
        norm = np.clip(img.astype(np.float32) / max_v, 0.0, 1.0)

        h, w = norm.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            norm = cv2.resize(norm, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return norm

    except Exception:
        return None


def extract_star_metrics(
    gray_img: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    max_stars_to_profile: int = 60
) -> Tuple[float, float, int, float]:
    """
    Analiza las propiedades geométricas de las estrellas en el área de cielo.
    
    Retorna:
        (median_fwhm, median_eccentricity, total_stars_detected, bg_noise)
    """
    h, w = gray_img.shape[:2]

    # Delimitar zona de cielo:
    if sky_mask is not None:
        if sky_mask.shape[:2] != (h, w):
            mask_resized = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = sky_mask
        sky_roi_mask = (mask_resized > 0.5)
    else:
        # Por defecto, en astropaisaje el cielo ocupa al menos el 65% superior
        sky_roi_mask = np.zeros((h, w), dtype=bool)
        sky_roi_mask[:int(h * 0.70), :] = True

    sky_pixels = gray_img[sky_roi_mask]
    if len(sky_pixels) < 500:
        return 0.0, 0.0, 0, 0.0

    # Estimación robusta del fondo usando percentil 25 y MAD
    bg_median = float(np.median(sky_pixels))
    bg_mad = float(np.median(np.abs(sky_pixels - bg_median)))
    bg_sigma = max(1e-4, 1.4826 * bg_mad)

    # Umbral de detección para núcleos estelares (5 sigmas sobre el fondo)
    threshold = bg_median + 4.5 * bg_sigma

    # Máscara binaria de picos estelares dentro del cielo
    peaks = (gray_img > threshold) & sky_roi_mask

    # Etiquetar componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8), connectivity=8
    )

    # Filtrar candidatos: descartar ruido de 1-2 px y manchas gigantes (nebulosas / luna)
    valid_stars = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= 400:
            cx, cy = centroids[i]
            # Descartar estrellas pegadas a los bordes
            if 10 <= cx < w - 10 and 10 <= cy < h - 10:
                peak_val = gray_img[int(cy), int(cx)]
                # Descartar saturadas al límite (donde se aplana el núcleo)
                if peak_val < 0.98:
                    valid_stars.append((cx, cy, area, peak_val))

    total_stars = len(valid_stars)
    if total_stars < 4:
        return 0.0, 0.0, total_stars, bg_sigma

    # Ordenar por brillo para analizar las estrellas con mejor SNR
    valid_stars.sort(key=lambda s: s[3], reverse=True)
    sample_stars = valid_stars[:max_stars_to_profile]

    fwhm_list = []
    eccentricity_list = []
    patch_radius = 6

    for cx, cy, _, _ in sample_stars:
        ix, iy = int(round(cx)), int(round(cy))
        patch = gray_img[iy - patch_radius : iy + patch_radius + 1, ix - patch_radius : ix + patch_radius + 1]
        if patch.shape != (2 * patch_radius + 1, 2 * patch_radius + 1):
            continue

        # Restar nivel de fondo local para calcular momentos puros de la estrella
        patch_clean = np.clip(patch - bg_median, 0.0, None)
        m00 = np.sum(patch_clean)
        if m00 <= 1e-4:
            continue

        # Coordenadas relativas al centro del patch
        y_grid, x_grid = np.mgrid[-patch_radius : patch_radius + 1, -patch_radius : patch_radius + 1]
        
        # Momentos de primer orden (subpíxel centroid)
        m10 = np.sum(x_grid * patch_clean)
        m01 = np.sum(y_grid * patch_clean)
        cx_sub = m10 / m00
        cy_sub = m01 / m00

        # Momentos centrales de segundo orden
        mu20 = np.sum(((x_grid - cx_sub) ** 2) * patch_clean) / m00
        mu02 = np.sum(((y_grid - cy_sub) ** 2) * patch_clean) / m00
        mu11 = np.sum(((x_grid - cx_sub) * (y_grid - cy_sub)) * patch_clean) / m00

        # Autovalores del tensor de inercia
        delta = math.sqrt(max(0.0, (mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2)))
        lambda1 = max(1e-4, ((mu20 + mu02) + delta) / 2.0)
        lambda2 = max(1e-4, ((mu20 + mu02) - delta) / 2.0)

        # Excentricidad geométrica: e = sqrt(1 - lambda2 / lambda1)
        ratio = min(1.0, max(0.0, lambda2 / lambda1))
        ecc = math.sqrt(1.0 - ratio)

        # FWHM = 2.355 * sigma_medio
        sigma_mean = math.sqrt((lambda1 + lambda2) / 2.0)
        fwhm = 2.35482 * sigma_mean

        if 0.5 <= fwhm <= 25.0:
            fwhm_list.append(fwhm)
            eccentricity_list.append(ecc)

    if not fwhm_list:
        return 0.0, 0.0, total_stars, bg_sigma

    median_fwhm = float(np.median(fwhm_list))
    median_ecc = float(np.median(eccentricity_list))

    return median_fwhm, median_ecc, total_stars, bg_sigma


def analyze_single_frame(filepath: str, sky_mask: Optional[np.ndarray] = None) -> FrameQualityResult:
    """Analiza una sola toma y calcula sus métricas y diagnóstico individual."""
    filename = os.path.basename(filepath)
    gray = load_fast_frame_preview(filepath)

    if gray is None:
        return FrameQualityResult(
            filepath=filepath,
            filename=filename,
            fwhm=0.0,
            eccentricity=1.0,
            star_count=0,
            score=0.0,
            status="DEFECTIVE",
            status_label="🔴 Error al leer",
            defect_reason="No se pudo decodificar la imagen",
            is_recommended=False
        )

    fwhm, ecc, stars, bg_noise = extract_star_metrics(gray, sky_mask=sky_mask)

    # Diagnóstico preliminar absoluto
    if stars < 10:
        return FrameQualityResult(
            filepath=filepath,
            filename=filename,
            fwhm=fwhm,
            eccentricity=ecc,
            star_count=stars,
            score=15.0,
            status="DEFECTIVE",
            status_label="🔴 Nubes / Sin estrellas",
            defect_reason="Conteo de estrellas crítico (< 10 estrellas detectadas)",
            is_recommended=False,
            bg_noise=bg_noise
        )

    if ecc >= 0.65:
        return FrameQualityResult(
            filepath=filepath,
            filename=filename,
            fwhm=fwhm,
            eccentricity=ecc,
            star_count=stars,
            score=30.0,
            status="DEFECTIVE",
            status_label="🔴 Movida (Viento / Deriva)",
            defect_reason=f"Excentricidad estelar severa ({ecc:.2f} >= 0.65)",
            is_recommended=False,
            bg_noise=bg_noise
        )

    # Base provisional (se normalizará en batch con el resto de tomas)
    base_score = 100.0 * (1.0 - 0.7 * ecc)
    status = "EXCELLENT" if ecc < 0.40 else "ACCEPTABLE"
    label = "🟢 Excelente" if status == "EXCELLENT" else "🟡 Aceptable"

    return FrameQualityResult(
        filepath=filepath,
        filename=filename,
        fwhm=fwhm,
        eccentricity=ecc,
        star_count=stars,
        score=max(10.0, min(100.0, base_score)),
        status=status,
        status_label=label,
        defect_reason="",
        is_recommended=True,
        bg_noise=bg_noise
    )


def batch_analyze_frame_quality(
    file_paths: List[str],
    sky_mask: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    max_workers: Optional[int] = None
) -> List[FrameQualityResult]:
    """
    Ejecuta el cribado y control de calidad en paralelo sobre una lista de tomas.
    Calcula una puntuación relativa consistente para todo el lote.
    """
    if not file_paths:
        return []

    if max_workers is None:
        max_workers = max(1, min(8, (os.cpu_count() or 4)))

    results: List[Optional[FrameQualityResult]] = [None] * len(file_paths)
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(analyze_single_frame, path, sky_mask): idx
            for idx, path in enumerate(file_paths)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            if cancel_check and cancel_check():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as e:
                res = FrameQualityResult(
                    filepath=file_paths[idx],
                    filename=os.path.basename(file_paths[idx]),
                    fwhm=0.0,
                    eccentricity=1.0,
                    star_count=0,
                    score=0.0,
                    status="DEFECTIVE",
                    status_label="🔴 Error",
                    defect_reason=str(e),
                    is_recommended=False
                )

            results[idx] = res
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, len(file_paths), res.filename)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return []

    # =========================================================================
    # NORMALIZACIÓN RELATIVA DE LA PUNTUACIÓN SEGÚN EL MEJOR DE LA SESIÓN
    # =========================================================================
    valid_fwhms = [r.fwhm for r in valid_results if r.fwhm > 0.5 and r.is_recommended]
    best_fwhm = min(valid_fwhms) if valid_fwhms else 3.0

    valid_stars = [r.star_count for r in valid_results if r.is_recommended]
    max_stars = max(valid_stars) if valid_stars else 100

    for r in valid_results:
        if not r.is_recommended:
            continue

        # Penalización cuadrática si la excentricidad supera 0.40
        ecc_factor = max(0.0, 1.0 - (r.eccentricity / 0.70) ** 1.8)
        
        # Penalización por FWHM respecto a la toma más nítida del lote
        fwhm_factor = min(1.0, (best_fwhm / max(r.fwhm, 1e-3)) ** 1.2)
        
        # Factor de permanencia estelar (detección de nubes parciales)
        stars_factor = min(1.0, r.star_count / max(1, 0.75 * max_stars))

        composite = 100.0 * (0.45 * ecc_factor + 0.35 * fwhm_factor + 0.20 * stars_factor)
        r.score = round(max(5.0, min(100.0, composite)), 1)

        # Re-evaluar estatus tras normalización
        if r.eccentricity >= 0.58:
            r.status = "DEFECTIVE"
            r.status_label = "🔴 Movida (Viento)"
            r.defect_reason = f"Estrellas elongadas (Excentricidad {r.eccentricity:.2f})"
            r.is_recommended = False
        elif r.fwhm > best_fwhm * 1.8 and r.fwhm > 5.0:
            r.status = "DEFECTIVE"
            r.status_label = "🔴 Desenfoque"
            r.defect_reason = f"FWHM inflado ({r.fwhm:.1f}px vs mejor {best_fwhm:.1f}px)"
            r.is_recommended = False
        elif r.star_count < 0.35 * max_stars:
            r.status = "DEFECTIVE"
            r.status_label = "🟡 Nubes"
            r.defect_reason = f"Bajo conteo de estrellas ({r.star_count} vs máx {max_stars})"
            r.is_recommended = False
        elif r.score >= 82.0:
            r.status = "EXCELLENT"
            r.status_label = "🟢 Excelente"
            r.is_recommended = True
        elif r.score >= 65.0:
            r.status = "ACCEPTABLE"
            r.status_label = "🟡 Aceptable"
            r.is_recommended = True
        else:
            r.status = "DEFECTIVE"
            r.status_label = "🔴 Calidad Baja"
            r.defect_reason = f"Puntuación insuficiente ({r.score:.1f}%)"
            r.is_recommended = False

    return valid_results
