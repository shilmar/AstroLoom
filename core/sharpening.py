# core/sharpening.py
"""
core/sharpening.py - Máscara de Enfoque con Detección de Bordes (Estilo Lightroom).
Proporciona realce selectivo de altas frecuencias espaciales (USM) con protección de ruido
en zonas homogéneas y cielo nocturno mediante gradientes de Sobel y smoothstep thresholding.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import cv2


def compute_edge_mask(
    image: np.ndarray,
    masking: float = 0.0,
    sky_mask: Optional[np.ndarray] = None,
    stars_mask: Optional[np.ndarray] = None,
    scope: str = "ground",
) -> np.ndarray:
    """
    Calcula un mapa de ponderación de bordes [0.0, 1.0] basado en gradientes espaciales.

    Parámetros:
        image: Imagen float32 RGB [H, W, 3] o monocroma [H, W] con rango [0.0, 1.0].
        masking: Umbral de máscara [0.0, 100.0]. 0.0 = máscara blanca (1.0 en todo el fotograma).
        sky_mask: Máscara opcional cielo/suelo (1.0 = cielo, 0.0 = suelo).
        stars_mask: Máscara opcional de estrellas (1.0 = estrellas, 0.0 = fondo).
        scope: Ámbito de confinamiento: 'ground' (solo suelo), 'sky_starless' (cielo sin estrellas),
               o 'global' (todo el fotograma).

    Retorna:
        Matriz bidimensional float32 [H, W] en rango [0.0, 1.0].
    """
    if image is None:
        raise ValueError("image cannot be None")

    # 1. Extracción de luminancia perceptiva ITU-R BT.709
    if image.ndim == 3 and image.shape[2] >= 3:
        luma = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        luma = image.squeeze() if image.ndim == 3 else image

    luma = np.ascontiguousarray(np.clip(luma, 0.0, 1.0), dtype=np.float32)
    h, w = luma.shape[:2]

    # 2. Detección de gradientes horizontales y verticales
    # shortcut: fixed Sobel ksize 3x3 is optimal for subpixel edges, upgrade to multiscale Scharr if gigapixel native pyramids request it
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Suavizado gaussiano subpíxel para evitar interpretar ruido térmico aislado como bordes
    mag_smooth = cv2.GaussianBlur(mag, (0, 0), sigmaX=0.8, sigmaY=0.8)

    # 3. Rampa de umbralización progresiva (Smoothstep Thresholding)
    if masking <= 1e-4:
        edge_mask = np.ones((h, w), dtype=np.float32)
    else:
        # Referencia robusta al percentil 99.5 de magnitud para escala invariante
        p99 = float(np.percentile(mag_smooth, 99.5)) if mag_smooth.size > 0 else 0.5
        m_ref = max(0.08, p99)
        threshold = (float(masking) / 100.0) * m_ref
        trans_width = max(1e-4, 0.15 * m_ref)

        t = np.clip((mag_smooth - threshold) / trans_width, 0.0, 1.0)
        edge_mask = t * t * (3.0 - 2.0 * t)

    # 4. Modulación por ámbito (Scope)
    final_mask = edge_mask.copy()
    if scope == "ground":
        if sky_mask is not None:
            sm = sky_mask[..., 0] if sky_mask.ndim == 3 else sky_mask
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            ground = 1.0 - np.clip(sm, 0.0, 1.0)
            final_mask *= ground
    elif scope == "sky_starless":
        if sky_mask is not None:
            sm = sky_mask[..., 0] if sky_mask.ndim == 3 else sky_mask
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            final_mask *= np.clip(sm, 0.0, 1.0)
        if stars_mask is not None:
            stm = stars_mask[..., 0] if stars_mask.ndim == 3 else stars_mask
            if stm.shape[:2] != (h, w):
                stm = cv2.resize(stm, (w, h), interpolation=cv2.INTER_LINEAR)
            final_mask *= (1.0 - np.clip(stm, 0.0, 1.0))
    # 'global': no-op, applies across entire frame

    return np.ascontiguousarray(np.clip(final_mask, 0.0, 1.0), dtype=np.float32)


def apply_unsharp_mask(
    image: np.ndarray,
    amount: float = 0.0,
    radius: float = 1.0,
    detail: float = 25.0,
    edge_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aplica una máscara de enfoque selectiva basada en luminancia.

    Parámetros:
        image: Imagen float32 RGB [H, W, 3] o [H, W] en [0.0, 1.0].
        amount: Intensidad del enfoque [0.0, 150.0]. 0.0 = sin cambios.
        radius: Radio espacial gaussiano en píxeles [0.5, 3.0].
        detail: Control de amortiguación de halos y ruido [0.0, 100.0].
        edge_mask: Máscara previa opcional [H, W] float32 con valores en [0.0, 1.0].

    Retorna:
        Imagen con enfoque aplicado con el mismo tipo y forma que la entrada.
    """
    if image is None or amount <= 1e-4:
        return image

    is_rgb = image.ndim == 3 and image.shape[2] >= 3
    if is_rgb:
        luma = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        luma = image.squeeze() if image.ndim == 3 else image

    luma = np.ascontiguousarray(luma, dtype=np.float32)
    h, w = luma.shape[:2]

    # Componente paso-bajo gaussiana
    sigma = max(0.1, float(radius))
    luma_blur = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
    diff = luma - luma_blur

    # Amortiguación de micro-ruido basada en el parámetro Detail
    noise_thresh = (100.0 - float(detail)) / 100.0 * 0.015
    if noise_thresh > 1e-5:
        abs_diff = np.abs(diff)
        diff = np.where(abs_diff < noise_thresh, diff * (abs_diff / noise_thresh), diff)

    strength = float(amount) / 50.0
    delta = diff * strength

    # Ponderación por máscara de bordes
    if edge_mask is not None:
        em = edge_mask[..., 0] if edge_mask.ndim == 3 else edge_mask
        if em.shape[:2] != (h, w):
            em = cv2.resize(em, (w, h), interpolation=cv2.INTER_LINEAR)
        delta *= np.clip(em, 0.0, 1.0)

    # Inyección sin alterar cromaticidad
    if is_rgb:
        delta_3d = delta[..., np.newaxis]
        out = np.clip(image + delta_3d, 0.0, 1.0)
    else:
        out = np.clip(image + delta, 0.0, 1.0)

    return np.ascontiguousarray(out, dtype=np.float32)
