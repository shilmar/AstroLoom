# core/background_extraction.py
"""
Módulo de Extracción de Gradientes de Fondo por Muestreo Interactivo (DBE / RBF).
Inspirado en PixInsight DynamicBackgroundExtraction (DBE) y GraXpert.
Permite modelar y sustraer gradientes espaciales complejos de polución lumínica mediante
interpolación multiescala 2D con Radial Basis Functions (RBF), preservando estructuras astronómicas
y excluyendo el suelo terrestre mediante máscaras de horizonte.
"""
from typing import Optional, List, Tuple
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def generate_sky_sample_points(
    h: int,
    w: int,
    sky_mask: Optional[np.ndarray] = None,
    grid_cols: int = 8,
    grid_rows: int = 6,
    img_rgb: Optional[np.ndarray] = None,
    box_size: int = 25,
    margin_ratio: float = 0.04,
    max_luminance_percentile: float = 98.0
) -> List[Tuple[float, float]]:
    """
    Genera automáticamente una rejilla uniforme de puntos de muestreo en el cielo,
    garantizando la exclusión estricta de la silueta del suelo y de zonas sobreiluminadas.

    Retorna una lista de coordenadas normalizadas (nx, ny) en rango [0.0, 1.0].
    """
    # 1. Crear rejilla regular candidata
    xs = np.linspace(margin_ratio, 1.0 - margin_ratio, max(3, grid_cols), dtype=np.float32)
    ys = np.linspace(margin_ratio, 1.0 - margin_ratio, max(3, grid_rows), dtype=np.float32)

    # 2. Preparar máscara de suelo con margen de seguridad moderado (2% de la altura)
    eroded_mask = None
    if sky_mask is not None:
        sm_h, sm_w = 200, max(200, int(200 * (w / max(1, h))))
        sm = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (sm_w, sm_h), interpolation=cv2.INTER_NEAREST)
        k_h = max(3, int(sm_h * 0.02))
        k_w = max(3, int(sm_w * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        eroded_mask = cv2.erode((sm > 0.5).astype(np.uint8), kernel)

    # 3. Luminancia para descarte exclusivo de núcleos estelares saturados o nebulosas hiperbrillantes
    lum_small = None
    lum_thresh = 0.96
    if img_rgb is not None:
        lh, lw = 150, max(150, int(150 * (w / max(1, h))))
        small = cv2.resize(img_rgb, (lw, lh), interpolation=cv2.INTER_AREA)
        lum = 0.2126 * small[..., 0] + 0.7152 * small[..., 1] + 0.0722 * small[..., 2]
        lum_small = lum
        if eroded_mask is not None:
            em_resized = cv2.resize(eroded_mask, (lw, lh), interpolation=cv2.INTER_NEAREST)
            sky_lum = lum[em_resized > 0]
            if len(sky_lum) > 50:
                p_thresh = float(np.percentile(sky_lum, max_luminance_percentile))
                lum_thresh = min(0.96, max(0.85, p_thresh))
        else:
            p_thresh = float(np.percentile(lum, max_luminance_percentile))
            lum_thresh = min(0.96, max(0.85, p_thresh))

    pts: List[Tuple[float, float]] = []

    for ny in ys:
        for nx in xs:
            # Comprobación de máscara de cielo
            if eroded_mask is not None:
                sm_y = int(np.clip(ny * (eroded_mask.shape[0] - 1), 0, eroded_mask.shape[0] - 1))
                sm_x = int(np.clip(nx * (eroded_mask.shape[1] - 1), 0, eroded_mask.shape[1] - 1))
                if eroded_mask[sm_y, sm_x] == 0:
                    continue

            # Comprobación de luminancia máxima
            if lum_small is not None:
                ly = int(np.clip(ny * (lum_small.shape[0] - 1), 0, lum_small.shape[0] - 1))
                lx = int(np.clip(nx * (lum_small.shape[1] - 1), 0, lum_small.shape[1] - 1))
                if lum_small[ly, lx] > lum_thresh:
                    continue

            pts.append((float(nx), float(ny)))

    # 4. Anclas de seguridad en las esquinas superiores para prevenir divergencia en los bordes
    for cx, cy in [(margin_ratio, margin_ratio), (1.0 - margin_ratio, margin_ratio)]:
        if eroded_mask is None or eroded_mask[int(cy * (eroded_mask.shape[0] - 1)), int(cx * (eroded_mask.shape[1] - 1))] > 0:
            if not any(abs(p[0] - cx) < 0.04 and abs(p[1] - cy) < 0.04 for p in pts):
                pts.append((float(cx), float(cy)))

    return pts


def extract_background_rbf(
    img_rgb: np.ndarray,
    sample_points: List[Tuple[float, float]],
    sky_mask: Optional[np.ndarray] = None,
    box_size: int = 25,
    mode: str = "subtraction",
    kernel: str = "multiquadric",
    smoothing: float = 0.15,
    epsilon: float = 1.0,
    eval_scale: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modela el fondo del cielo mediante interpolación 2D multiescala con funciones RBF
    a partir de puntos de muestra y neutraliza el gradiente.

    Parámetros:
    - img_rgb: Imagen RGB en float32 [0.0, 1.0] (H, W, 3).
    - sample_points: Lista de coordenadas normalizadas (nx, ny) en [0.0, 1.0].
    - sky_mask: Máscara de cielo float32 [0.0, 1.0] (1=cielo, 0=suelo).
    - box_size: Tamaño de la ventana de muestreo local en píxeles.
    - mode: "subtraction" (resta de contaminación lumínica) o "division" (viñeteo/óptica).
    - kernel: Kernel RBF ("multiquadric", "thin_plate_spline", "cubic", "linear").
    - smoothing: Parámetro de regularización de curvatura (mayor suavizado previene sobreoscilaciones).
    - epsilon: Escala de forma para kernels multiquadric/gaussian.
    - eval_scale: Factor de submuestreo de la cuadrícula de interpolación para cálculo en <50ms.

    Retorna:
    - corrected_img: Imagen corregida float32 [0.0, 1.0].
    - bg_model: Superficie 2D estimada del fondo (H, W, 3).
    """
    h, w = img_rgb.shape[:2]

    # Si hay menos de 4 puntos, generar rejilla automática mínima
    valid_points = [p for p in sample_points if 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0]
    if len(valid_points) < 4:
        valid_points = generate_sky_sample_points(h, w, sky_mask=sky_mask, img_rgb=img_rgb, box_size=box_size)
        if len(valid_points) < 4:
            raise ValueError("Se requieren al menos 4 puntos de ancla en el cielo para la interpolación RBF.")

    # 1. Extracción de valores de fondo por parche con rechazo de estrellas y exclusión de suelo
    half_r = max(2, box_size // 2)
    sampled_vals: List[np.ndarray] = []
    points_kept: List[Tuple[float, float]] = []

    m_eval = None
    if sky_mask is not None:
        m_eval = np.squeeze(sky_mask).astype(np.float32)
        if m_eval.shape[:2] != (h, w):
            m_eval = cv2.resize(m_eval, (w, h), interpolation=cv2.INTER_NEAREST)

    for nx, ny in valid_points:
        px = int(np.clip(round(nx * (w - 1)), 0, w - 1))
        py = int(np.clip(round(ny * (h - 1)), 0, h - 1))

        x1 = max(0, px - half_r)
        x2 = min(w, px + half_r + 1)
        y1 = max(0, py - half_r)
        y2 = min(h, py + half_r + 1)

        patch = img_rgb[y1:y2, x1:x2]  # shape: (ph, pw, 3)

        if m_eval is not None:
            patch_m = m_eval[y1:y2, x1:x2]
            sky_px = patch[patch_m >= 0.35]
            if len(sky_px) >= 4:
                val_rgb = np.percentile(sky_px, 25, axis=0).astype(np.float32)
            else:
                # Punto situado dentro de terreno terrestre, se descarta para no alterar el modelo de cielo
                continue
        else:
            # Rechazo de estrellas: percentil 25 por canal para aislar el verdadero pedestal de fondo
            val_rgb = np.percentile(patch, 25, axis=(0, 1)).astype(np.float32)

        sampled_vals.append(val_rgb)
        points_kept.append((nx, ny))

    if len(points_kept) < 4:
        # Fallback de seguridad si el usuario puso puntos en el suelo
        points_kept = generate_sky_sample_points(h, w, sky_mask=sky_mask, img_rgb=img_rgb, box_size=box_size)
        sampled_vals = []
        for nx, ny in points_kept:
            px = int(np.clip(round(nx * (w - 1)), 0, w - 1))
            py = int(np.clip(round(ny * (h - 1)), 0, h - 1))
            x1, x2 = max(0, px - half_r), min(w, px + half_r + 1)
            y1, y2 = max(0, py - half_r), min(h, py + half_r + 1)
            patch = img_rgb[y1:y2, x1:x2]
            val_rgb = np.percentile(patch, 25, axis=(0, 1)).astype(np.float32)
            sampled_vals.append(val_rgb)

    pts_arr = np.array(points_kept, dtype=np.float64)
    vals_arr = np.array(sampled_vals, dtype=np.float64)

    eval_h = max(24, min(180, h // max(1, eval_scale)))
    eval_w = max(24, min(240, w // max(1, eval_scale)))

    # Aceleración nativa en Rust con Rayon + Zero-Copy (Drop-in Fallback)
    if HAS_RUST:
        try:
            smooth_val = max(1e-3, float(smoothing))
            eps_val = max(0.01, float(epsilon)) if kernel in ("multiquadric", "gaussian", "inverse_multiquadric") else 1.0
            bg_proxy = astroloom_rs.fit_and_eval_rbf_proxy(
                pts_arr, vals_arr, eval_h, eval_w, str(kernel), smooth_val, eps_val
            )
            bg_full = cv2.resize(bg_proxy, (w, h), interpolation=cv2.INTER_CUBIC)
            pedestal = np.median(vals_arr, axis=0).astype(np.float32)

            img_c = np.ascontiguousarray(img_rgb, dtype=np.float32)
            bg_c = np.ascontiguousarray(bg_full, dtype=np.float32)
            ped_c = np.ascontiguousarray(pedestal, dtype=np.float32)
            is_div = (mode == "division")

            sky_m = None
            if sky_mask is not None:
                sm = np.squeeze(sky_mask).astype(np.float32)
                if sm.shape != (h, w):
                    sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
                sky_m = np.ascontiguousarray(sm, dtype=np.float32)

            corrected = astroloom_rs.apply_bg_model_fused(img_c, bg_c, ped_c, is_div, sky_m)
            return corrected, bg_full
        except Exception:
            pass  # Fallback a SciPy + NumPy

    # 2. Ajuste de superficie RBF multiescala regularizada (Fallback SciPy)
    rbf_kwargs = {"smoothing": max(1e-3, float(smoothing))}
    if kernel in ("multiquadric", "gaussian", "inverse_multiquadric"):
        rbf_kwargs["epsilon"] = max(0.01, float(epsilon))

    rbf = RBFInterpolator(pts_arr, vals_arr, kernel=kernel, **rbf_kwargs)

    # 3. Evaluación rápida en rejilla espacial proxy
    eval_h = max(24, min(180, h // max(1, eval_scale)))
    eval_w = max(24, min(240, w // max(1, eval_scale)))
    gy, gx = np.mgrid[0:1:complex(0, eval_h), 0:1:complex(0, eval_w)]
    eval_coords = np.column_stack([gx.ravel(), gy.ravel()])

    bg_proxy = rbf(eval_coords).reshape((eval_h, eval_w, 3)).astype(np.float32)

    # Acotación segura del modelo para prevenir sobreoscilaciones (anti-ringing / anti-fosos negros)
    max_sample_val = np.max(vals_arr, axis=0).astype(np.float32) * 1.05
    min_sample_val = np.maximum(0.0, np.min(vals_arr, axis=0).astype(np.float32) * 0.90)
    bg_proxy = np.clip(bg_proxy, min_sample_val, max_sample_val)

    # 4. Upscaling bicúbico de alta calidad a resolución nativa
    bg_full = cv2.resize(bg_proxy, (w, h), interpolation=cv2.INTER_CUBIC)

    # 5. Pedestal medio del cielo para preservar la calibración tonal global
    pedestal = np.median(vals_arr, axis=0).astype(np.float32)

    # 6. Neutralización de gradiente (Sustracción o División)
    if mode == "division":
        corrected = (img_rgb / np.maximum(bg_full, 1e-4)) * pedestal
    else:  # subtraction
        corrected = img_rgb - bg_full + pedestal

    # 7. Preservación estricta del terreno terrestre con la máscara de cielo
    if sky_mask is not None:
        m_full = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        m_3d = np.clip(m_full, 0.0, 1.0)[..., np.newaxis]
        # El suelo (0.0) queda 100% idéntico a la entrada original; el gradiente solo actúa en el cielo
        corrected = (corrected * m_3d) + (img_rgb * (1.0 - m_3d))

    return np.clip(corrected, 0.0, 1.0).astype(np.float32), bg_full
