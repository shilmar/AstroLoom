# core/denoise.py
"""
Módulo de Reducción de Ruido Astronómico Multietapa (Fase 1).
Implementa un motor de filtrado adaptativo de alto rendimiento en CPU:
1. Reducción de Ruido Cromático en espacio YCrCb (limpieza de parches de color en sombras).
2. Reducción de Ruido de Luminancia Bilateral con preservación estricta de microestructuras.
3. Máscara de Luminosidad Invertida (prioridad a fondos oscuros y sombras donde el SNR es mínimo).
4. Máscara de Protección de Bordes/Estructura (estilo Lightroom) para proteger estrellas y filamentos.
"""

from typing import Optional
import numpy as np
import cv2


def compute_denoise_mask(
    img_rgb: np.ndarray,
    masking: float = 0.5,
    sky_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calcula la máscara combinada de reducción de ruido [0.0, 1.0].
    - Blanco (1.0): Zonas oscuras planas donde la reducción actúa con máxima fuerza.
    - Negro (0.0): Estrellas, altas luces y filamentos de alto contraste protegidos.
    """
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        y = cv2.cvtColor(np.clip(img_rgb, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
    else:
        y = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)

    # 1. Máscara de luminancia invertida (más denoise en cielo negro y sombras profundas)
    lum_exp = 1.0 + float(masking) * 1.5
    lum_mask = np.clip(1.0 - y, 0.0, 1.0) ** lum_exp

    # 2. Detección de bordes y gradientes espaciales (protección de estrellas y nebulosas)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    edge_sensitivity = 2.0 + 18.0 * float(masking)
    edge_protection = np.clip(1.0 - grad * edge_sensitivity, 0.0, 1.0)

    combined = lum_mask * edge_protection

    if sky_mask is not None:
        h, w = y.shape[:2]
        sm = sky_mask
        if sm.shape[:2] != (h, w):
            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
        if sm.ndim == 3:
            sm = sm[..., 0]
        combined = combined * sm

    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def apply_astro_denoise(
    img_rgb: np.ndarray,
    strength: float = 0.0,
    color_strength: float = 0.5,
    masking: float = 0.5,
    sky_mask: Optional[np.ndarray] = None,
    is_draft: bool = False,
    is_full_res: bool = False
) -> np.ndarray:
    """
    Aplica reducción de ruido astronómica desacoplada (Color + Luminancia + Máscara).

    Parámetros:
    - img_rgb: Imagen en RGB float32 [0.0, 1.0].
    - strength: Fuerza de reducción de ruido de luminancia [0.0, 1.0] (grano de fondo).
    - color_strength: Fuerza de reducción de ruido cromático [0.0, 1.0] (motas de color en sombras).
    - masking: Protección de bordes y enfoque en sombras [0.0, 1.0] (estilo Lightroom).
    - sky_mask: Máscara opcional de cielo para acotar el filtrado.
    - is_draft: Optimización para proxy borrador interactivo (60 FPS).
    - is_full_res: Escala radios para fotograma completo nativo.
    """
    if strength <= 1e-4 and color_strength <= 1e-4:
        return img_rgb

    # Conversión a espacio YCrCb para desacoplar Luminancia y Croma
    safe_input = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)
    ycrcb = cv2.cvtColor(safe_input, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0]
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]

    # --- 1. REDUCCIÓN DE RUIDO CROMÁTICO (COLOR NOISE) ---
    if color_strength > 1e-4:
        scale_c = 1.8 if is_full_res else 1.0
        ksize = 5 if is_draft else int(max(3, round((3.0 + 8.0 * (color_strength ** 0.8)) * scale_c)))
        if ksize % 2 == 0:
            ksize += 1
        sigma_c = float((1.0 + 3.5 * color_strength) * scale_c)

        cr_clean = cv2.GaussianBlur(cr, (ksize, ksize), sigma_c)
        cb_clean = cv2.GaussianBlur(cb, (ksize, ksize), sigma_c)

        blend_c = float(np.clip(color_strength, 0.0, 1.0))
        ycrcb[..., 1] = cr * (1.0 - blend_c) + cr_clean * blend_c
        ycrcb[..., 2] = cb * (1.0 - blend_c) + cb_clean * blend_c

    # --- 2. REDUCCIÓN DE RUIDO DE LUMINANCIA CON MÁSCARA ADAPTATIVA ---
    if strength > 1e-4:
        # Calcular máscara combinada (sombras profundas + protección de gradientes)
        spatial_mask = compute_denoise_mask(safe_input, masking=masking, sky_mask=sky_mask)

        # Filtrado bilateral en el canal de luminancia
        d = 5 if is_draft else (9 if is_full_res else 7)
        sigma_color = float(0.03 + 0.12 * strength)
        sigma_space = float((2.0 + 5.0 * strength) * (1.8 if is_full_res else 1.0))

        y_filtered = cv2.bilateralFilter(y, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        # Modulación proporcional por píxel
        blend_y = np.clip(strength * spatial_mask, 0.0, 1.0)
        ycrcb[..., 0] = y * (1.0 - blend_y) + y_filtered * blend_y

    out_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return np.clip(out_rgb, 0.0, 1.0).astype(np.float32)


# Alias de retrocompatibilidad directa con callers existentes
apply_denoise = apply_astro_denoise
