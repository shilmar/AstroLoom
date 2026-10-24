# core/masking.py
"""
core/masking.py - Algoritmos de segmentación y refinamiento de máscaras cielo/suelo.
Implementa segmentación híbrida mediante GrabCut asistido por luminancia y refinado guiado.
"""
from typing import Optional
import cv2
import numpy as np


def refine_mask_guided(
    image_rgb: np.ndarray, 
    user_scribbles: np.ndarray,
    feather_radius: int = 7,
    iterations: int = 3,
    base_mask: Optional[np.ndarray] = None
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

    # 5. Suavizado y refinado guiado de bordes
    k = max(1, feather_radius)
    if k % 2 == 0:
        k += 1

    if k > 1:
        feathered = cv2.GaussianBlur(mask_full, (k, k), sigmaX=k / 3.0)
        # Mantener las semillas manuales del usuario con peso estricto
        feathered[user_scribbles == 2] = 1.0
        feathered[user_scribbles == 1] = 0.0
        mask_final = feathered
    else:
        mask_final = mask_full

    return np.clip(mask_final, 0.0, 1.0).astype(np.float32)


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


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Función de interpolación suave Hermite cúbica (smoothstep) acotada en [0, 1]."""
    diff = max(1e-6, edge1 - edge0)
    t = np.clip((x - edge0) / diff, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_luminance_mask(
    image_rgb: np.ndarray,
    luma_min: float = 0.0,
    luma_max: float = 1.0,
    feather: float = 0.08,
    invert: bool = False
) -> np.ndarray:
    """
    Genera una máscara continua de rango de luminancia tonal (Luces, Medios, Sombras).
    
    Parámetros:
        image_rgb (np.ndarray): Imagen base RGB o monocroma float32 en [0.0, 1.0].
        luma_min (float): Umbral inferior de luminancia (0.0 a 1.0).
        luma_max (float): Umbral superior de luminancia (0.0 a 1.0).
        feather (float): Ancho de la transición suave en los extremos.
        invert (bool): Si es True, invierte la máscara (1.0 - máscara).

    Retorna:
        np.ndarray: Máscara continua float32 2D con valores en [0.0, 1.0].
    """
    if image_rgb.ndim == 3:
        # Ponderación fotométrica Rec.709 para luminancia perceptual exacta
        luma = (
            0.2126 * image_rgb[..., 0] + 
            0.7152 * image_rgb[..., 1] + 
            0.0722 * image_rgb[..., 2]
        ).astype(np.float32)
    else:
        luma = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    f = max(1e-4, float(feather))
    l_min = float(min(luma_min, luma_max))
    l_max = float(max(luma_min, luma_max))

    # Transición en el umbral inferior (fade-in)
    low_ramp = _smoothstep(l_min - f, l_min, luma) if l_min > 0.0 else np.ones_like(luma)

    # Transición en el umbral superior (fade-out)
    high_ramp = (1.0 - _smoothstep(l_max, l_max + f, luma)) if l_max < 1.0 else np.ones_like(luma)

    mask = np.clip(low_ramp * high_ramp, 0.0, 1.0)

    if invert:
        mask = 1.0 - mask

    return mask.astype(np.float32)


def generate_color_range_mask(
    image_rgb: np.ndarray,
    target_hue_deg: float,
    hue_tolerance_deg: float = 30.0,
    min_saturation: float = 0.08,
    feather: float = 0.08,
    invert: bool = False
) -> np.ndarray:
    """
    Genera una máscara continua de rango cromático en el espacio HSV.
    Aísla bandas espectrales específicas (H-Alfa, OIII, polución lumínica de sodio, etc.).

    Parámetros:
        image_rgb (np.ndarray): Imagen base RGB float32 normalizada [0.0, 1.0].
        target_hue_deg (float): Matiz objetivo en grados (0.0 a 360.0).
        hue_tolerance_deg (float): Tolerancia angular alrededor del matiz objetivo.
        min_saturation (float): Umbral mínimo de saturación para evitar ruido neutro.
        feather (float): Suavizado de la transición angular y cromática.
        invert (bool): Si es True, invierte la máscara.

    Retorna:
        np.ndarray: Máscara continua float32 2D con valores en [0.0, 1.0].
    """
    if image_rgb.ndim != 3:
        return np.ones(image_rgb.shape[:2], dtype=np.float32)

    img_clipped = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(img_clipped, cv2.COLOR_RGB2HSV)
    h_channel = hsv[..., 0]  # En OpenCV float32, Hue está en [0.0, 360.0]
    s_channel = hsv[..., 1]  # Saturation en [0.0, 1.0]

    # Diferencia angular circular más corta en el círculo cromático [0, 360]
    target_h = float(target_hue_deg) % 360.0
    tol = max(1.0, float(hue_tolerance_deg))
    f = max(0.01, float(feather))

    diff = np.abs(h_channel - target_h)
    angular_dist = np.minimum(diff, 360.0 - diff)

    # Rampa de tolerancia angular
    fade_deg = f * 45.0
    hue_mask = 1.0 - _smoothstep(tol, tol + fade_deg, angular_dist)

    # Rampa de corte por saturación para excluir estrellas blancas y fondo neutro
    s_min = max(0.01, float(min_saturation))
    sat_ramp = _smoothstep(max(0.0, s_min - f * 0.15), s_min, s_channel)

    mask = np.clip(hue_mask * sat_ramp, 0.0, 1.0)

    if invert:
        mask = 1.0 - mask

    return mask.astype(np.float32)