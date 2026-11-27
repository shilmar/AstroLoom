# core/masking.py
"""
core/masking.py - Algoritmos de segmentación y refinamiento de máscaras cielo/suelo.
Implementa segmentación híbrida mediante GrabCut asistido por luminancia y refinado guiado.
"""
from typing import Optional
import cv2
import numpy as np

try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def refine_mask_guided(
    image_rgb: np.ndarray, 
    user_scribbles: np.ndarray,
    feather_radius: int = 7,
    iterations: int = 3,
    base_mask: Optional[np.ndarray] = None,
    shift_edge: int = 0
) -> np.ndarray:
    """
    Refina la máscara cielo/suelo preservando ramas finas y siluetas complejas mediante:
    1. Análisis fotométrico entre trazos de cielo y suelo para calcular el umbral de silueta.
    2. GrabCut iterativo asistido por máscara en resolución adaptativa.
    3. Recuperación de micro-estructuras oscuras (ramas/follaje) y suavizado perimetral guiado.

    Parámetros:
        image_rgb (np.ndarray): Imagen base normalizada float32 [0.0, 1.0].
        user_scribbles (np.ndarray): Matriz uint8 (0: no marcado, 1: suelo, 2: cielo).
        feather_radius (int): Radio para el desenfoque gaussiano del perímetro (en píxeles).
        iterations (int): Número de iteraciones del algoritmo GrabCut.
        base_mask (Optional[np.ndarray]): Máscara previa opcional (ej. de IA 1-Click) para inicializar GrabCut.

    Retorna:
        np.ndarray: Máscara continua float32 [0.0, 1.0] (1.0 = cielo, 0.0 = suelo).
    """
    h, w = user_scribbles.shape[:2]
    
    has_gnd_scribble = np.any(user_scribbles == 1)
    has_sky_scribble = np.any(user_scribbles == 2)

    # Si falta trazo y no hay máscara base, retorna cielo completo
    if not (has_gnd_scribble and has_sky_scribble) and base_mask is None:
        if has_sky_scribble and not has_gnd_scribble:
            return np.ones((h, w), dtype=np.float32)
        elif has_gnd_scribble and not has_sky_scribble:
            return np.zeros((h, w), dtype=np.float32)
        return np.ones((h, w), dtype=np.float32)

    # 1. Luminancia normalizada
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(np.clip(image_rgb, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    else:
        gray = np.clip(image_rgb, 0.0, 1.0)

    # 2. Análisis fotométrico de las semillas del usuario y/o máscara base
    if has_sky_scribble and has_gnd_scribble:
        sky_samples = gray[user_scribbles == 2]
        gnd_samples = gray[user_scribbles == 1]
    elif base_mask is not None:
        bm = np.clip(base_mask.astype(np.float32), 0.0, 1.0)
        if bm.shape[:2] != (h, w):
            bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_LINEAR)
        sky_samples = gray[bm > 0.8] if np.any(bm > 0.8) else (gray[user_scribbles == 2] if has_sky_scribble else np.array([0.2]))
        gnd_samples = gray[bm < 0.2] if np.any(bm < 0.2) else (gray[user_scribbles == 1] if has_gnd_scribble else np.array([0.05]))
    else:
        sky_samples = np.array([0.2])
        gnd_samples = np.array([0.05])

    sky_min = float(np.percentile(sky_samples, 5)) if len(sky_samples) > 0 else 0.2
    gnd_max = float(np.percentile(gnd_samples, 95)) if len(gnd_samples) > 0 else 0.1

    # Umbral de separación de silueta entre suelo y cielo
    split_thresh = (sky_min * 0.4) + (gnd_max * 0.6)

    # 3. Inicializar matriz de probabilidades de GrabCut
    if base_mask is not None:
        bm = np.clip(base_mask.astype(np.float32), 0.0, 1.0)
        if bm.shape[:2] != (h, w):
            bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_LINEAR)
        gc_mask = np.where(bm > 0.5, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    else:
        gc_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)

    # Fijar semillas seguras del usuario
    if has_gnd_scribble:
        gc_mask[user_scribbles == 1] = cv2.GC_BGD  # Suelo seguro
    if has_sky_scribble:
        gc_mask[user_scribbles == 2] = cv2.GC_FGD  # Cielo seguro

    # Clasificar como probable fondo las estructuras oscuras que contrastan con el cielo
    dark_silhouette = (gray <= split_thresh) & (user_scribbles == 0)
    gc_mask[dark_silhouette] = cv2.GC_PR_BGD

    # 4. Redimensionado conservando detalle para optimizar el rendimiento de GrabCut
    max_dim = 1600.0
    scale = min(1.0, max_dim / max(h, w))
    
    if scale < 1.0:
        small_w = int(w * scale)
        small_h = int(h * scale)
        small_rgb_8u = (np.clip(cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA), 0.0, 1.0) * 255.0).astype(np.uint8)
        small_mask = cv2.resize(gc_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    else:
        small_rgb_8u = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        small_mask = gc_mask.copy()

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # GrabCut iterativo asistido
    cv2.grabCut(
        small_rgb_8u, 
        small_mask, 
        None, 
        bgd_model, 
        fgd_model, 
        iterCount=max(1, iterations), 
        mode=cv2.GC_INIT_WITH_MASK
    )

    # Probable o seguro cielo = 1.0; resto = 0.0
    refined_small = np.where(
        (small_mask == cv2.GC_FGD) | (small_mask == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)

    # Reescalar la máscara base a la resolución nativa
    if scale < 1.0:
        mask_full = cv2.resize(refined_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask_full = refined_small

    # 5. Suavizado y refinado guiado de bordes (evita desborde gaussiano ciego)
    k = max(1, feather_radius)
    if k > 1:
        try:
            from core.ai_horizon import fast_guided_filter
            mask_final = fast_guided_filter(gray, mask_full, radius=k, eps=1e-4)
        except Exception:
            if k % 2 == 0:
                k += 1
            mask_final = cv2.GaussianBlur(mask_full, (k, k), sigmaX=k / 3.0)
    else:
        mask_final = mask_full

    # Desplazamiento opcional de frontera (Shift Edge)
    if shift_edge != 0:
        mask_final = shift_mask_edge(mask_final, shift_edge)

    # Mantener las semillas manuales del usuario con peso estricto
    mask_final[user_scribbles == 2] = 1.0
    mask_final[user_scribbles == 1] = 0.0

    return np.clip(mask_final, 0.0, 1.0).astype(np.float32)


def shift_mask_edge(mask: np.ndarray, shift_px: int) -> np.ndarray:
    """
    Desplaza la frontera de la máscara hacia el cielo (shift_px > 0) o hacia el suelo (shift_px < 0).
    Permite contraer la máscara de cielo para ocultar flecos luminosos en siluetas complejas.
    """
    if shift_px == 0 or mask is None:
        return mask
    is_3d = mask.ndim == 3
    m_2d = mask[..., 0] if is_3d else mask
    k_size = 2 * abs(int(shift_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    if shift_px > 0:
        res = cv2.dilate(m_2d, kernel)
    else:
        res = cv2.erode(m_2d, kernel)
    if is_3d:
        return np.repeat(res[..., np.newaxis], mask.shape[2], axis=2)
    return res


def suppress_horizon_halos(
    image_rgb: np.ndarray,
    sky_mask: np.ndarray,
    radius: int = 3,
    strength: float = 0.8,
    suppress_dark_halos: bool = False
) -> np.ndarray:
    if strength <= 1e-4 or sky_mask is None or radius < 1:
        return image_rgb

    h, w = image_rgb.shape[:2]
    m_2d = sky_mask[..., 0] if sky_mask.ndim == 3 else sky_mask
    if m_2d.shape[:2] != (h, w):
        m_2d = cv2.resize(m_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    return astroloom_rs.suppress_horizon_halos_rs(
        np.ascontiguousarray(image_rgb, dtype=np.float32),
        np.ascontiguousarray(m_2d, dtype=np.float32),
        int(radius),
        float(strength),
        bool(suppress_dark_halos)
    )


def apply_direct_brush_to_mask(
    mask: np.ndarray,
    user_scribbles: np.ndarray,
    feather_radius: int = 3
) -> np.ndarray:
    """
    Aplica trazos de pincel directamente sobre una máscara existente (1: suelo, 2: cielo),
    con centro 100% sólido y borde suavizado local, sin necesidad de ejecutar GrabCut.
    """
    h, w = mask.shape[:2]
    out = mask.copy().astype(np.float32)
    if user_scribbles.shape[:2] != (h, w):
        user_scribbles = cv2.resize(user_scribbles, (w, h), interpolation=cv2.INTER_NEAREST)

    # Suelo seguro (1 -> 0.0)
    if np.any(user_scribbles == 1):
        brush_gnd = (user_scribbles == 1).astype(np.float32)
        if feather_radius > 1:
            k = 2 * feather_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dilated = cv2.dilate(brush_gnd, kernel)
            fringe = cv2.GaussianBlur(dilated, (k, k), feather_radius / 2.0)
            weight = np.maximum(brush_gnd, fringe)
        else:
            weight = brush_gnd
        out = out * (1.0 - weight)

    # Cielo seguro (2 -> 1.0)
    if np.any(user_scribbles == 2):
        brush_sky = (user_scribbles == 2).astype(np.float32)
        if feather_radius > 1:
            k = 2 * feather_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dilated = cv2.dilate(brush_sky, kernel)
            fringe = cv2.GaussianBlur(dilated, (k, k), feather_radius / 2.0)
            weight = np.maximum(brush_sky, fringe)
        else:
            weight = brush_sky
        out = np.clip(out + weight * (1.0 - out), 0.0, 1.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def generate_luminance_mask(
    image_rgb: np.ndarray,
    luma_min: float = 0.0,
    luma_max: float = 1.0,
    feather: float = 0.08,
    invert: bool = False
) -> np.ndarray:
    """Genera una máscara continua de rango de luminancia tonal (Rust)."""
    img_c = np.ascontiguousarray(image_rgb, dtype=np.float32)
    return astroloom_rs.generate_luminance_mask_rs(
        img_c,
        luma_min=float(luma_min),
        luma_max=float(luma_max),
        feather=float(feather),
        invert=bool(invert)
    )


def generate_color_range_mask(
    image_rgb: np.ndarray,
    target_hue_deg: float,
    hue_tolerance_deg: float = 30.0,
    min_saturation: float = 0.08,
    feather: float = 0.08,
    invert: bool = False
) -> np.ndarray:
    """Genera una máscara continua de rango cromático en el espacio HSV (Rust)."""
    if image_rgb.ndim != 3:
        return np.ones(image_rgb.shape[:2], dtype=np.float32)

    img_c = np.ascontiguousarray(image_rgb, dtype=np.float32)
    return astroloom_rs.generate_color_range_mask_rs(
        img_c,
        target_hue_deg=float(target_hue_deg),
        hue_tolerance_deg=float(hue_tolerance_deg),
        min_saturation=float(min_saturation),
        feather=float(feather),
        invert=bool(invert)
    )


def refine_mask_guided_cross(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    strength: float = 1.0,
    radius: Optional[int] = None,
    eps: float = 0.02
) -> np.ndarray:
    """
    Refinamiento de bordes de máscara por Filtro Cruzado Guiado (guided_cross / LightCraft).
    Imanta los contornos de la máscara a las aristas y microestructuras reales de la escena
    utilizando la luminancia logarítmica como guía fotométrica.

    Fórmula:
        guia = log(1.0 + 255.0 * L) / log(256.0)
        alpha_refinado = GuidedFilter(guia=guia, src=alpha, radius=sigma, eps=eps)
        alpha_final = alpha + (alpha_refinado - alpha) * sqrt(strength)
    """
    if mask is None or image_rgb is None:
        return mask
    if strength <= 1e-4:
        return mask

    is_3d = (mask.ndim == 3 and mask.shape[2] == 1)
    m_2d = mask[..., 0] if mask.ndim == 3 else mask
    m_f = np.clip(m_2d.astype(np.float32), 0.0, 1.0)
    h, w = m_f.shape[:2]

    # 1. Preparar imagen guía: calcular luminancia perceptual
    if image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
        if image_rgb.shape[:2] != (h, w):
            ref_img = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            ref_img = image_rgb

        if ref_img.dtype == np.uint8:
            ref_f = ref_img.astype(np.float32) / 255.0
        else:
            ref_f = np.clip(ref_img, 0.0, 1.0).astype(np.float32)

        lum = (0.2126 * ref_f[..., 0] + 0.7152 * ref_f[..., 1] + 0.0722 * ref_f[..., 2]).astype(np.float32)
    else:
        if image_rgb.shape[:2] != (h, w):
            ref_img = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            ref_img = image_rgb
        if ref_img.dtype == np.uint8:
            lum = ref_img.astype(np.float32) / 255.0
        else:
            lum = np.clip(ref_img, 0.0, 1.0).astype(np.float32)

    # 2. Guía en escala logarítmica normalizada en [0.0, 1.0]
    guide_log = np.log1p(255.0 * lum) / np.log(256.0)

    # 3. Radio adaptativo proporcional al eje mayor (sigma ≈ 0.04 * long_edge)
    if radius is None or radius <= 0:
        long_edge = max(h, w)
        r = max(3, int(round(0.04 * long_edge)))
    else:
        r = max(1, int(radius))

    # 4. Filtro guiado con aceleración Rust O(1) vía SAT
    g = np.ascontiguousarray(guide_log, dtype=np.float32)
    m = np.ascontiguousarray(m_f, dtype=np.float32)
    q = np.clip(astroloom_rs.fast_guided_filter_rs(g, m, radius=r, eps=float(eps)), 0.0, 1.0)

    # 5. Ponderación no lineal con raíz cuadrada de la intensidad
    k = float(np.clip(strength, 0.0, 1.0))
    alpha_final = m_f + (q - m_f) * np.sqrt(k)
    alpha_final = np.clip(alpha_final, 0.0, 1.0).astype(np.float32)

    if is_3d:
        return alpha_final[..., np.newaxis]
    return alpha_final

