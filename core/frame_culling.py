# core/frame_culling.py
"""
Módulo de cribado inteligente de tomas y control de calidad óptica (Frame Culling).
Analiza FWHM, excentricidad/elongación de estrellas (para detectar trepidación por viento o deriva)
y censo estelar (para detectar paso de nubes o niebla) de forma rápida y multihilo.
Diseñado especialmente para astrofotografía de paisaje (soporta gran angular y coma de esquinas).
"""

import os
import math
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable, Union

import cv2
import numpy as np


@dataclass
class FrameQualityResult:
    """Resultado del análisis de calidad óptica para una sola toma."""
    filepath: str
    filename: str
    fwhm: float            # FWHM en píxeles (mediana en zona central/óptica)
    eccentricity: float    # Excentricidad efectiva [0.0 = círculo perfecto, 1.0 = trazo lineal]
    star_count: int        # Número total de estrellas detectadas en el cielo
    score: float           # Puntuación global de calidad [0.0 a 100.0]
    status: str            # 'EXCELLENT', 'ACCEPTABLE', 'DEFECTIVE'
    status_label: str      # Texto amigable con icono para la GUI
    defect_reason: str     # Razón del defecto si aplica
    is_recommended: bool   # True si se recomienda incluir en el apilado
    bg_noise: float = 0.0  # Nivel de ruido del fondo
    center_ecc: float = 0.0 # Excentricidad en zona central / eje óptico (libre de coma)
    corner_ecc: float = 0.0 # Excentricidad en periferia / esquinas (coma óptica de la lente)
    grid_3x3_fwhm: Optional[List[List[float]]] = None # Matriz 3x3 de FWHM mediana por sector
    grid_3x3_ecc: Optional[List[List[float]]] = None  # Matriz 3x3 de Excentricidad mediana por sector
    optical_tilt_ratio: float = 0.0                   # Asimetría porcentual de inclinación (Tilt)
    optical_curvature_ratio: float = 0.0              # Ratio de curvatura de campo (esquinas vs centro)
    optical_defect: str = "OK"                        # 'OK', 'FIELD_CURVATURE', 'SENSOR_TILT', 'COMBINED', 'INSUFFICIENT_DATA'
    optical_diagnosis_msg: str = ""                   # Diagnóstico legible del defecto del equipo


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
                    h, w = gray.shape[:2]
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
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


def compute_3x3_optical_diagnosis(
    star_profiles: List[Tuple[float, float, float, float]],
    h: int,
    w: int
) -> Dict[str, Any]:
    """
    Calcula el mapa 3x3 de calidad de campo óptico (FWHM y Excentricidad)
    y clasifica anomalías del tren óptico: Curvatura de campo vs Inclinación de sensor (Tilt).
    
    star_profiles: Lista de tuplas (cx, cy, fwhm, ecc).
    h, w: Dimensiones de la imagen analizada.
    """
    fwhm_grid_vals: List[List[List[float]]] = [[[] for _ in range(3)] for _ in range(3)]
    ecc_grid_vals: List[List[List[float]]] = [[[] for _ in range(3)] for _ in range(3)]

    cell_h = max(1.0, h / 3.0)
    cell_w = max(1.0, w / 3.0)

    for cx, cy, fwhm, ecc in star_profiles:
        r = min(2, max(0, int(cy / cell_h)))
        c = min(2, max(0, int(cx / cell_w)))
        fwhm_grid_vals[r][c].append(fwhm)
        ecc_grid_vals[r][c].append(ecc)

    grid_fwhm: List[List[float]] = [[float("nan")] * 3 for _ in range(3)]
    grid_ecc: List[List[float]] = [[float("nan")] * 3 for _ in range(3)]
    grid_counts: List[List[int]] = [[0] * 3 for _ in range(3)]

    valid_cells = 0
    for r in range(3):
        for c in range(3):
            cnt = len(fwhm_grid_vals[r][c])
            grid_counts[r][c] = cnt
            if cnt >= 2:
                grid_fwhm[r][c] = round(float(np.median(fwhm_grid_vals[r][c])), 2)
                grid_ecc[r][c] = round(float(np.median(ecc_grid_vals[r][c])), 3)
                valid_cells += 1

    total_profiled = len(star_profiles)
    if total_profiled < 6 or valid_cells < 3:
        return {
            "grid_fwhm": grid_fwhm,
            "grid_ecc": grid_ecc,
            "grid_counts": grid_counts,
            "curvature_ratio": 0.0,
            "tilt_x": 0.0,
            "tilt_y": 0.0,
            "total_tilt": 0.0,
            "defect_type": "INSUFFICIENT_DATA",
            "diagnosis_msg": "Estrellas insuficientes para diagnóstico de campo"
        }

    # 1. Curvatura de campo (Centro vs 4 Esquinas)
    f_center = grid_fwhm[1][1]
    if math.isnan(f_center):
        adj_center = [grid_fwhm[r][c] for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)] if not math.isnan(grid_fwhm[r][c])]
        f_center = float(np.median(adj_center)) if adj_center else float("nan")

    corner_indices = [(0, 0), (0, 2), (2, 0), (2, 2)]
    corner_fwhms = [grid_fwhm[r][c] for r, c in corner_indices if not math.isnan(grid_fwhm[r][c])]

    if len(corner_fwhms) >= 2 and not math.isnan(f_center) and f_center > 0.5:
        med_corner = float(np.median(corner_fwhms))
        curvature_ratio = max(0.0, (med_corner - f_center) / f_center)
        corner_spread = float((max(corner_fwhms) - min(corner_fwhms)) / max(1e-3, med_corner))
    else:
        curvature_ratio = 0.0
        corner_spread = 0.0

    # 2. Inclinación de sensor / Flexión (Tilt)
    left_vals = [grid_fwhm[r][0] for r in range(3) if not math.isnan(grid_fwhm[r][0])]
    right_vals = [grid_fwhm[r][2] for r in range(3) if not math.isnan(grid_fwhm[r][2])]

    if left_vals and right_vals:
        med_l = float(np.median(left_vals))
        med_r = float(np.median(right_vals))
        tilt_x = abs(med_l - med_r) / max(0.5, min(med_l, med_r))
    else:
        tilt_x = 0.0

    top_vals = [grid_fwhm[0][c] for c in range(3) if not math.isnan(grid_fwhm[0][c])]
    bot_vals = [grid_fwhm[2][c] for c in range(3) if not math.isnan(grid_fwhm[2][c])]

    if top_vals and bot_vals:
        med_t = float(np.median(top_vals))
        med_b = float(np.median(bot_vals))
        tilt_y = abs(med_t - med_b) / max(0.5, min(med_t, med_b))
    else:
        tilt_y = 0.0

    total_tilt = float(math.hypot(tilt_x, tilt_y))

    # 3. Clasificación del diagnóstico
    TILT_THRESH = 0.18
    CURV_THRESH = 0.20

    if total_tilt >= TILT_THRESH and curvature_ratio >= CURV_THRESH:
        defect_type = "COMBINED"
        diagnosis_msg = f"Curvatura (+{curvature_ratio*100:.0f}%) e Inclinación/Tilt ({total_tilt*100:.0f}%)"
    elif total_tilt >= TILT_THRESH:
        defect_type = "SENSOR_TILT"
        axis = "horizontal (izq/der)" if tilt_x >= tilt_y else "vertical (sup/inf)"
        diagnosis_msg = f"Inclinación/Tilt mecánico {axis} ({total_tilt*100:.0f}%)"
    elif curvature_ratio >= CURV_THRESH and corner_spread <= 0.25:
        defect_type = "FIELD_CURVATURE"
        diagnosis_msg = f"Curvatura de campo simétrica (+{curvature_ratio*100:.0f}% FWHM esquinas)"
    elif curvature_ratio >= 0.25 and corner_spread > 0.25:
        defect_type = "SENSOR_TILT"
        diagnosis_msg = f"Degradación asimétrica de esquinas (posible flexión/tilt)"
    else:
        defect_type = "OK"
        diagnosis_msg = "Óptica bien corregida (campo plano y simétrico)"

    return {
        "grid_fwhm": grid_fwhm,
        "grid_ecc": grid_ecc,
        "grid_counts": grid_counts,
        "curvature_ratio": round(curvature_ratio, 3),
        "tilt_x": round(tilt_x, 3),
        "tilt_y": round(tilt_y, 3),
        "total_tilt": round(total_tilt, 3),
        "defect_type": defect_type,
        "diagnosis_msg": diagnosis_msg
    }


def extract_star_metrics(
    gray_img: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    max_stars_to_profile: int = 90,
    profile_mode: str = "wide_angle",
    return_optical_grid: bool = False
) -> Union[Tuple[float, float, int, float, float, float], Tuple[float, float, int, float, float, float, Dict[str, Any]]]:
    """
    Analiza las propiedades geométricas de las estrellas en el área de cielo.
    Distingue de forma inteligente entre la zona central (eje óptico, sensible al viento/deriva)
    y las esquinas/periferia (donde los objetivos gran angular tienen coma natural).
    
    Retorna:
        (median_fwhm, effective_ecc, total_stars_detected, bg_noise, center_ecc, corner_ecc)
        o tupla de 7 elementos con diagnóstico óptico 3x3 si return_optical_grid=True.
    """
    empty_diag = {
        "grid_fwhm": None, "grid_ecc": None, "grid_counts": None,
        "curvature_ratio": 0.0, "tilt_x": 0.0, "tilt_y": 0.0, "total_tilt": 0.0,
        "defect_type": "INSUFFICIENT_DATA", "diagnosis_msg": "Estrellas insuficientes para diagnóstico de campo"
    }

    h, w = gray_img.shape[:2]
    cx_center = w / 2.0
    cy_center = h / 2.0
    r_max = max(1.0, math.hypot(cx_center, cy_center))

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
        if return_optical_grid:
            return 0.0, 0.0, 0, 0.0, 0.0, 0.0, empty_diag
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0

    # Estimación robusta del fondo usando percentil 25 y MAD
    bg_median = float(np.median(sky_pixels))
    bg_mad = float(np.median(np.abs(sky_pixels - bg_median)))
    bg_sigma = max(1e-4, 1.4826 * bg_mad)

    # Umbral de detección para núcleos estelares (4.5 sigmas sobre el fondo)
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
            # Descartar estrellas pegadas al borde absoluto
            if 10 <= cx < w - 10 and 10 <= cy < h - 10:
                peak_val = gray_img[int(cy), int(cx)]
                # Descartar saturadas al límite (donde se aplana el núcleo)
                if peak_val < 0.98:
                    r_norm = math.hypot(cx - cx_center, cy - cy_center) / r_max
                    valid_stars.append((cx, cy, area, peak_val, r_norm))

    total_stars = len(valid_stars)
    if total_stars < 4:
        if return_optical_grid:
            return 0.0, 0.0, total_stars, bg_sigma, 0.0, 0.0, empty_diag
        return 0.0, 0.0, total_stars, bg_sigma, 0.0, 0.0

    # Estratificación: separar estrellas centrales (r < 0.60) y periféricas (r >= 0.60)
    central_candidates = [s for s in valid_stars if s[4] < 0.60]
    outer_candidates = [s for s in valid_stars if s[4] >= 0.60]

    central_candidates.sort(key=lambda s: s[3], reverse=True)
    outer_candidates.sort(key=lambda s: s[3], reverse=True)

    # Muestreo representativo dando prioridad a la zona central (inmune a coma)
    if len(central_candidates) >= 6:
        n_center = min(len(central_candidates), int(max_stars_to_profile * 0.60))
        n_outer = min(len(outer_candidates), max_stars_to_profile - n_center)
        sample_stars = central_candidates[:n_center] + outer_candidates[:n_outer]
    else:
        valid_stars.sort(key=lambda s: s[3], reverse=True)
        sample_stars = valid_stars[:max_stars_to_profile]

    center_fwhms = []
    center_eccs = []
    corner_eccs = []
    all_fwhms = []
    all_eccs = []
    star_profiles = []
    patch_radius = 6

    for cx, cy, _, _, r_norm in sample_stars:
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
            all_fwhms.append(fwhm)
            all_eccs.append(ecc)
            star_profiles.append((cx, cy, fwhm, ecc))
            if r_norm < 0.60:
                center_fwhms.append(fwhm)
                center_eccs.append(ecc)
            else:
                corner_eccs.append(ecc)

    if not all_fwhms:
        if return_optical_grid:
            return float("nan"), float("nan"), total_stars, bg_sigma, float("nan"), float("nan"), empty_diag
        return float("nan"), float("nan"), total_stars, bg_sigma, float("nan"), float("nan")

    median_center_ecc = float(np.median(center_eccs)) if center_eccs else float(np.median(all_eccs))
    median_corner_ecc = float(np.median(corner_eccs)) if corner_eccs else median_center_ecc
    median_fwhm = float(np.median(center_fwhms)) if center_fwhms else float(np.median(all_fwhms))

    # Ponderación de excentricidad según el perfil del objetivo
    if profile_mode == "wide_angle":
        # En gran angular, la trepidación/deriva se mide en el centro (las esquinas tienen coma óptica inherente)
        effective_ecc = (0.85 * median_center_ecc) + (0.15 * median_corner_ecc)
    elif profile_mode == "strict":
        effective_ecc = (0.50 * median_center_ecc) + (0.50 * median_corner_ecc)
    else:  # balanced
        effective_ecc = (0.70 * median_center_ecc) + (0.30 * median_corner_ecc)

    if return_optical_grid:
        opt_diag = compute_3x3_optical_diagnosis(star_profiles, h, w)
        return median_fwhm, effective_ecc, total_stars, bg_sigma, median_center_ecc, median_corner_ecc, opt_diag

    return median_fwhm, effective_ecc, total_stars, bg_sigma, median_center_ecc, median_corner_ecc


def analyze_single_frame(
    filepath: str, 
    sky_mask: Optional[np.ndarray] = None,
    profile_mode: str = "wide_angle"
) -> FrameQualityResult:
    """Analiza una sola toma y calcula sus métricas y diagnóstico individual preliminar."""
    filename = os.path.basename(filepath)
    gray = load_fast_frame_preview(filepath)

    if gray is None:
        return FrameQualityResult(
            filepath=filepath,
            filename=filename,
            fwhm=float("nan"),
            eccentricity=float("nan"),
            star_count=0,
            score=0.0,
            status="DEFECTIVE",
            status_label="🔴 Error al leer",
            defect_reason="No se pudo decodificar la imagen",
            is_recommended=False,
            bg_noise=float("nan"),
            center_ecc=float("nan"),
            corner_ecc=float("nan")
        )

    fwhm, ecc, stars, bg_noise, c_ecc, cor_ecc, opt_diag = extract_star_metrics(
        gray, sky_mask=sky_mask, profile_mode=profile_mode, return_optical_grid=True
    )

    # Diagnóstico preliminar absoluto para casos extremos
    if stars < 6:
        return FrameQualityResult(
            filepath=filepath,
            filename=filename,
            fwhm=fwhm,
            eccentricity=ecc,
            star_count=stars,
            score=10.0,
            status="DEFECTIVE",
            status_label="🔴 Nubes / Sin estrellas",
            defect_reason="Conteo de estrellas crítico (< 6 estrellas detectadas)",
            is_recommended=False,
            bg_noise=bg_noise,
            center_ecc=c_ecc,
            corner_ecc=cor_ecc,
            grid_3x3_fwhm=opt_diag.get("grid_fwhm"),
            grid_3x3_ecc=opt_diag.get("grid_ecc"),
            optical_tilt_ratio=opt_diag.get("total_tilt", 0.0),
            optical_curvature_ratio=opt_diag.get("curvature_ratio", 0.0),
            optical_defect=opt_diag.get("defect_type", "INSUFFICIENT_DATA"),
            optical_diagnosis_msg=opt_diag.get("diagnosis_msg", "")
        )

    # Trepidación severa en el centro (indica movimiento global de cámara)
    if c_ecc >= 0.75:
        return FrameQualityResult(
            filepath=filepath,
            filename=filename,
            fwhm=fwhm,
            eccentricity=ecc,
            star_count=stars,
            score=25.0,
            status="DEFECTIVE",
            status_label="🔴 Movida (Viento / Deriva)",
            defect_reason=f"Trepidación central severa ({c_ecc:.2f} >= 0.75)",
            is_recommended=False,
            bg_noise=bg_noise,
            center_ecc=c_ecc,
            corner_ecc=cor_ecc,
            grid_3x3_fwhm=opt_diag.get("grid_fwhm"),
            grid_3x3_ecc=opt_diag.get("grid_ecc"),
            optical_tilt_ratio=opt_diag.get("total_tilt", 0.0),
            optical_curvature_ratio=opt_diag.get("curvature_ratio", 0.0),
            optical_defect=opt_diag.get("defect_type", "OK"),
            optical_diagnosis_msg=opt_diag.get("diagnosis_msg", "")
        )

    # Base provisional normal (se recalculará en batch comparando con el lote)
    base_score = 100.0 * (1.0 - 0.55 * ecc)
    status = "EXCELLENT" if ecc < 0.45 else "ACCEPTABLE"
    label = "🟢 Excelente" if status == "EXCELLENT" else "🟡 Aceptable"

    return FrameQualityResult(
        filepath=filepath,
        filename=filename,
        fwhm=fwhm,
        eccentricity=ecc,
        star_count=stars,
        score=max(15.0, min(100.0, base_score)),
        status=status,
        status_label=label,
        defect_reason="",
        is_recommended=True,
        bg_noise=bg_noise,
        center_ecc=c_ecc,
        corner_ecc=cor_ecc,
        grid_3x3_fwhm=opt_diag.get("grid_fwhm"),
        grid_3x3_ecc=opt_diag.get("grid_ecc"),
        optical_tilt_ratio=opt_diag.get("total_tilt", 0.0),
        optical_curvature_ratio=opt_diag.get("curvature_ratio", 0.0),
        optical_defect=opt_diag.get("defect_type", "OK"),
        optical_diagnosis_msg=opt_diag.get("diagnosis_msg", "")
    )


def batch_analyze_frame_quality(
    file_paths: List[str],
    sky_mask: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    max_workers: Optional[int] = None,
    profile_mode: str = "wide_angle"
) -> List[FrameQualityResult]:
    """
    Ejecuta el cribado y control de calidad en paralelo sobre una lista de tomas.
    Calcula una puntuación relativa consistente para todo el lote usando la mediana de la sesión
    como referencia, evitando descartar falsamente fotos por aberraciones ópticas de la lente.
    """
    if not file_paths:
        return []

    if max_workers is None:
        max_workers = max(1, min(8, (os.cpu_count() or 4)))

    results: List[Optional[FrameQualityResult]] = [None] * len(file_paths)
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(analyze_single_frame, path, sky_mask, profile_mode): idx
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
                    fwhm=float("nan"),
                    eccentricity=float("nan"),
                    star_count=0,
                    score=0.0,
                    status="DEFECTIVE",
                    status_label="🔴 Error",
                    defect_reason=str(e),
                    is_recommended=False,
                    bg_noise=float("nan"),
                    center_ecc=float("nan"),
                    corner_ecc=float("nan")
                )

            results[idx] = res
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, len(file_paths), res.filename)

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return []
    return evaluate_session_quality(valid_results, profile_mode=profile_mode)


def evaluate_session_quality(
    valid_results: List[FrameQualityResult], 
    profile_mode: str = "wide_angle"
) -> List[FrameQualityResult]:
    """Evalúa de forma relativa la calidad del lote sin placeholders ni centinelas numéricos."""
    if not valid_results:
        return []

    # =========================================================================
    # LÍNEA BASE DE LA SESIÓN (Estilo SubframeSelector relativo sin placeholders)
    # =========================================================================
    valid_center_eccs = [
        r.center_ecc for r in valid_results 
        if r.star_count >= 8 and r.center_ecc is not None and not math.isnan(r.center_ecc) and r.center_ecc > 0.05
    ]
    session_median_ecc = float(np.nanmedian(valid_center_eccs)) if valid_center_eccs else 0.32

    valid_stars = [r.star_count for r in valid_results if r.star_count is not None and not math.isnan(r.star_count)]
    session_median_stars = float(np.nanmedian(valid_stars)) if valid_stars else 100.0

    valid_fwhms = [
        r.fwhm for r in valid_results 
        if r.fwhm is not None and not math.isnan(r.fwhm) and r.fwhm > 0.5 
        and (r.center_ecc is not None and not math.isnan(r.center_ecc) and r.center_ecc < 0.65)
    ]
    if len(valid_fwhms) >= 5:
        best_fwhm = float(np.nanpercentile(valid_fwhms, 15))
    elif valid_fwhms:
        best_fwhm = float(np.nanmin(valid_fwhms))
    else:
        best_fwhm = 3.0

    # Umbrales según perfil (Principio 5.4: techos duros independientes para evitar
    # que una sesión anómala/degradada expanda su propio margen de aceptación)
    MAX_USABLE_ECC = 0.70
    if profile_mode == "wide_angle":
        # Tolerante con coma de esquinas; solo descarta anomalías reales
        ecc_outlier_thresh = min(MAX_USABLE_ECC, max(0.66, session_median_ecc + 0.22))
        clouds_thresh_ratio = 0.35
        fwhm_thresh_ratio = 2.1
        min_accept_score = 40.0
        min_excel_score = 70.0
    elif profile_mode == "strict":
        ecc_outlier_thresh = min(0.62, max(0.50, session_median_ecc + 0.14))
        clouds_thresh_ratio = 0.55
        fwhm_thresh_ratio = 1.6
        min_accept_score = 60.0
        min_excel_score = 80.0
    else:  # balanced
        ecc_outlier_thresh = min(0.66, max(0.58, session_median_ecc + 0.18))
        clouds_thresh_ratio = 0.45
        fwhm_thresh_ratio = 1.8
        min_accept_score = 50.0
        min_excel_score = 75.0

    for r in valid_results:
        # Principio 5.4: Una toma declarada DEFECTIVE por un fallo físico individual
        # (ej. sin estrellas, error de lectura, o trepidación crítica unitaria) no puede
        # ser auto-perdonada por una sesión degradada.
        if not r.is_recommended and r.status == "DEFECTIVE":
            r.score = min(r.score, 25.0)
            continue

        # Penalización relativa por excentricidad respecto a la mediana del lote
        if r.center_ecc is not None and not math.isnan(r.center_ecc):
            delta_ecc = max(0.0, r.center_ecc - session_median_ecc)
            ecc_factor = max(0.0, 1.0 - (delta_ecc / 0.38) ** 1.3)
        else:
            ecc_factor = 0.0

        # Factor FWHM respecto a la mejor toma (ignora NaNs)
        if r.fwhm is not None and not math.isnan(r.fwhm) and r.fwhm > 0.5:
            fwhm_factor = min(1.0, (best_fwhm / max(r.fwhm, 1e-3)) ** 1.0)
        else:
            fwhm_factor = 0.0

        # Factor de permanencia estelar (detección de nubes parciales)
        stars_cnt = r.star_count if (r.star_count is not None and not math.isnan(r.star_count)) else 0
        stars_factor = min(1.0, stars_cnt / max(1.0, 0.80 * session_median_stars))

        composite = 100.0 * (0.45 * ecc_factor + 0.35 * fwhm_factor + 0.20 * stars_factor)
        r.score = round(max(5.0, min(100.0, composite)), 1)

        # Clasificación por defectos reales vs calidad aceptable
        if r.center_ecc is not None and not math.isnan(r.center_ecc) and r.center_ecc >= ecc_outlier_thresh:
            r.status = "DEFECTIVE"
            r.status_label = "🔴 Movida (Viento)"
            r.defect_reason = f"Trepidación anómala ({r.center_ecc:.2f} >= umbral {ecc_outlier_thresh:.2f})"
            r.is_recommended = False
        elif r.star_count < max(6, int(clouds_thresh_ratio * session_median_stars)):
            r.status = "DEFECTIVE"
            r.status_label = "🔴 Nubes"
            r.defect_reason = f"Caída de estrellas ({r.star_count} vs mediana {int(session_median_stars)})"
            r.is_recommended = False
        elif (r.fwhm is not None and not math.isnan(r.fwhm)) and (
            (r.fwhm > best_fwhm * fwhm_thresh_ratio and r.fwhm > 5.5) or r.fwhm > 16.0
        ):
            r.status = "DEFECTIVE"
            r.status_label = "🔴 Desenfoque"
            r.defect_reason = f"FWHM inflado ({r.fwhm:.1f}px vs mejor {best_fwhm:.1f}px)"
            r.is_recommended = False
        elif r.score >= min_excel_score:
            r.status = "EXCELLENT"
            r.status_label = f"🟢 {r.score:.0f}%"
            r.defect_reason = ""
            r.is_recommended = True
        elif r.score >= min_accept_score:
            r.status = "ACCEPTABLE"
            r.status_label = f"🟡 {r.score:.0f}%"
            r.defect_reason = ""
            r.is_recommended = True
        else:
            r.status = "DEFECTIVE"
            r.status_label = f"🔴 {r.score:.0f}%"
            r.defect_reason = f"Puntuación insuficiente ({r.score:.1f}% < {min_accept_score:.0f}%)"
            r.is_recommended = False

    return valid_results


def pick_best_reference_frame(
    results: List[FrameQualityResult],
    excluded_paths: Optional[set] = None
) -> Optional[FrameQualityResult]:
    """
    Selecciona el fotograma óptimo como referencia geométrica y fotométrica
    a partir de las métricas de calidad estelar (mayor score, menor FWHM y mayor censo estelar).
    """
    candidates = [r for r in results if excluded_paths is None or r.filepath not in excluded_paths]
    if not candidates:
        return None

    def sort_key(r: FrameQualityResult):
        rec_priority = 1 if r.is_recommended else 0
        score = r.score if (r.score is not None and not math.isnan(r.score)) else 0.0
        fwhm_val = -r.fwhm if (r.fwhm is not None and not math.isnan(r.fwhm) and r.fwhm > 0) else -float("inf")
        stars_val = r.star_count if (r.star_count is not None and not math.isnan(r.star_count)) else 0
        return (rec_priority, score, fwhm_val, stars_val)

    return max(candidates, key=sort_key)


def diagnose_session_optics(valid_results: List[FrameQualityResult]) -> Dict[str, Any]:
    """
    Agrega el diagnóstico óptico a nivel de sesión completa para aislar
    aberraciones sistemáticas del equipo (curvatura o tilt) del ruido y turbulencia temporal.
    
    Retorna:
    - has_systematic_defect: bool
    - defect_type: str ('OK', 'FIELD_CURVATURE', 'SENSOR_TILT', 'COMBINED', 'INSUFFICIENT_DATA')
    - summary: str (descripción legible con porcentajes y recomendación)
    - median_tilt: float
    - median_curvature: float
    - session_grid_fwhm: List[List[float]] (mapa 3x3 de FWHM mediana del lote)
    - session_grid_ecc: List[List[float]] (mapa 3x3 de Excentricidad mediana del lote)
    """
    eligible = [r for r in valid_results if r.star_count >= 8 and r.grid_3x3_fwhm is not None]
    if len(eligible) < 3:
        return {
            "has_systematic_defect": False,
            "defect_type": "INSUFFICIENT_DATA",
            "summary": "Toma de datos insuficiente para diagnóstico de sesión",
            "median_tilt": 0.0,
            "median_curvature": 0.0,
            "session_grid_fwhm": None,
            "session_grid_ecc": None,
        }

    # Cuadrícula 3x3 de sesión: mediana de cada celda a través de todas las tomas
    session_grid_fwhm: List[List[float]] = [[float("nan")] * 3 for _ in range(3)]
    session_grid_ecc: List[List[float]] = [[float("nan")] * 3 for _ in range(3)]

    for r in range(3):
        for c in range(3):
            cell_fwhms = [
                res.grid_3x3_fwhm[r][c] for res in eligible 
                if res.grid_3x3_fwhm and not math.isnan(res.grid_3x3_fwhm[r][c])
            ]
            cell_eccs = [
                res.grid_3x3_ecc[r][c] for res in eligible 
                if res.grid_3x3_ecc and not math.isnan(res.grid_3x3_ecc[r][c])
            ]
            if len(cell_fwhms) >= 3:
                session_grid_fwhm[r][c] = round(float(np.median(cell_fwhms)), 2)
            if len(cell_eccs) >= 3:
                session_grid_ecc[r][c] = round(float(np.median(cell_eccs)), 3)

    # Contar frecuencia de defectos en tomas individuales
    tilt_count = sum(1 for res in eligible if res.optical_defect in ("SENSOR_TILT", "COMBINED"))
    curv_count = sum(1 for res in eligible if res.optical_defect in ("FIELD_CURVATURE", "COMBINED"))
    total = len(eligible)

    tilt_freq = tilt_count / total
    curv_freq = curv_count / total

    valid_tilts = [r.optical_tilt_ratio for r in eligible if not math.isnan(r.optical_tilt_ratio)]
    valid_curvs = [r.optical_curvature_ratio for r in eligible if not math.isnan(r.optical_curvature_ratio)]
    med_tilt = float(np.median(valid_tilts)) if valid_tilts else 0.0
    med_curv = float(np.median(valid_curvs)) if valid_curvs else 0.0

    if tilt_freq >= 0.60 and curv_freq >= 0.60:
        summary = (
            f"Defecto sistemático combinado: Curvatura (+{med_curv*100:.0f}%) "
            f"e Inclinación/Tilt ({med_tilt*100:.0f}% en el {tilt_freq*100:.0f}% del lote). "
            f"Revisar colimación, espaciado del aplanador y holguras del enfocador."
        )
        defect_type = "COMBINED"
        has_defect = True
    elif tilt_freq >= 0.60:
        summary = (
            f"Inclinación sistemática (Tilt) detectada en el {tilt_freq*100:.0f}% de las tomas "
            f"(asimetría media: {med_tilt*100:.0f}%). "
            f"Revisar flexión mecánica, tornillos de colimación o anillo adaptador T2."
        )
        defect_type = "SENSOR_TILT"
        has_defect = True
    elif curv_freq >= 0.60:
        summary = (
            f"Curvatura de campo sistemática detectada en el {curv_freq*100:.0f}% de las tomas "
            f"(+{med_curv*100:.0f}% FWHM en esquinas). "
            f"Ajustar la distancia de backfocus entre el aplanador y el sensor."
        )
        defect_type = "FIELD_CURVATURE"
        has_defect = True
    else:
        summary = "Óptica del equipo bien corregida y equilibrada en toda la sesión"
        defect_type = "OK"
        has_defect = False

    return {
        "has_systematic_defect": has_defect,
        "defect_type": defect_type,
        "summary": summary,
        "median_tilt": round(med_tilt, 3),
        "median_curvature": round(med_curv, 3),
        "session_grid_fwhm": session_grid_fwhm,
        "session_grid_ecc": session_grid_ecc,
    }

