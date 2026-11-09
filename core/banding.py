# core/banding.py
"""
Módulo de Supresión de Banding CMOS de Sensor (Banding Reduction).
Elimina el ruido estructurado por filas y columnas generado por conversores ADC y raíles
de alimentación en sensores CMOS (Sony, Canon, Nikon) en astrofotografía a ISOs altos.

Aísla el fondo del cielo protegiendo núcleos estelares y nebulosas, y separa el banding
de alta frecuencia del gradiente astronómico suave mediante filtrado paso bajo 1D.
"""

from typing import Optional, Union
import numpy as np

try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def _suppress_1d_banding_axis(
    channel_2d: np.ndarray,
    axis: int = 0,
    amount: float = 1.0,
    protect_sigma: float = 2.5,
    smooth_sigma: float = 35.0,
    sky_mask: Optional[np.ndarray] = None,
    subsample_step: int = 4
) -> np.ndarray:
    """
    Suprime bandas a lo largo de un eje (axis=0 para filas horizontales, axis=1 para columnas verticales).
    Opera en arrays 2D float32.
    """
    if amount <= 1e-4:
        return channel_2d

    # Si se corrige por columnas (axis=1), transponemos para tratar siempre el problema como por filas
    data = channel_2d if axis == 0 else channel_2d.T
    mask_in = sky_mask if (sky_mask is None or axis == 0) else sky_mask.T
    h, w = data.shape

    # 1. Estimación robusta del fondo del cielo (Mediana y MAD)
    if mask_in is not None:
        sky_pts = data[mask_in > 0.5]
        if len(sky_pts) < 100:
            sky_pts = data.ravel()
    else:
        sky_pts = data.ravel()

    # Muestreo representativo para estadísticas rápidas
    if len(sky_pts) > 50000:
        sky_sample = sky_pts[::max(1, len(sky_pts) // 50000)]
    else:
        sky_sample = sky_pts

    med_bg = float(np.median(sky_sample))
    mad_bg = float(np.median(np.abs(sky_sample - med_bg)))
    # Clamp físico de dispersión de ruido de fondo para evitar que terrenos/paisajes oscuros
    # inflen artificialmente sigma_bg (Lección 5.4 - umbrales independientes de anomalías)
    sigma_bg = min(0.020, max(1e-4, 1.4826 * mad_bg))

    # 2. Máscara de píxeles válidos de fondo (excluye estrellas, altas luces y ruido negativo extremo)
    lo_bound = med_bg - 2.5 * sigma_bg
    hi_bound = med_bg + protect_sigma * sigma_bg
    bg_mask = (data >= lo_bound) & (data <= hi_bound)
    if mask_in is not None:
        bg_mask = bg_mask & (mask_in > 0.5)

    # 3. Submuestreo ortogonal a lo largo de la fila (aceleración 4x sin pérdida de precisión de offset)
    step = max(1, min(subsample_step, w // 50))
    sub_data = data[:, ::step]
    sub_bg_mask = bg_mask[:, ::step]

    # Matriz con NaNs en píxeles que no pertenecen al fondo
    nan_grid = np.where(sub_bg_mask, sub_data, np.nan)

    # Mediana por fila ignorando NaNs
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_medians = np.nanmedian(nan_grid, axis=1).astype(np.float32)

    # 4. Manejo de filas con insuficientes píxeles de cielo (ej. suelo/montaña en astropaisajes o nebulosas densas)
    # Si una fila tiene menos del 15% de píxeles válidos de fondo, no es estadísticamente fiable y se interpola
    valid_counts = np.sum(sub_bg_mask, axis=1)
    min_valid_cols = max(10, int(sub_data.shape[1] * 0.15))
    row_medians[valid_counts < min_valid_cols] = np.nan

    nan_rows = np.isnan(row_medians)
    if np.any(nan_rows):
        valid_indices = np.where(~nan_rows)[0]
        if len(valid_indices) > 0:
            row_medians[nan_rows] = np.interp(
                np.where(nan_rows)[0],
                valid_indices,
                row_medians[valid_indices]
            )
        else:
            row_medians.fill(med_bg)

    # 5. Separación de frecuencias: gradiente suave natural (baja frec) vs banding (alta frec)
    smooth_sigma_eff = max(3.0, float(smooth_sigma))
    import scipy.ndimage
    smooth_profile = scipy.ndimage.gaussian_filter1d(
        row_medians,
        sigma=smooth_sigma_eff,
        mode="nearest"
    )

    # El patrón de banding es únicamente el residuo de alta frecuencia acotado físicamente
    # (en sensores CMOS reales el banding de fila nunca supera ~0.015 en float normalizado)
    raw_pattern = row_medians - smooth_profile
    banding_pattern = np.clip(raw_pattern, -0.015, 0.015) * float(amount)

    # 6. Sustracción del offset a lo largo de cada fila
    corrected = data - banding_pattern[:, None]

    return corrected if axis == 0 else corrected.T


def suppress_cmos_banding(
    image: np.ndarray,
    direction: str = "horizontal",
    amount: float = 1.0,
    protect_sigma: float = 2.5,
    smooth_sigma: float = 35.0,
    sky_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Aplica supresión de bandas CMOS horizontales y/o verticales sobre imágenes astronómicas.

    Parámetros:
    - image: Array numpy float32 [H, W] o [H, W, C] (lineal o revelada).
    - direction: "horizontal" (filas), "vertical" (columnas) o "both" (ambas direcciones).
    - amount: Factor de intensidad de corrección [0.0 = sin efecto, 1.0 = corrección completa].
    - protect_sigma: Umbral en desviaciones estándar sobre el fondo para excluir estrellas y nebulosas.
    - smooth_sigma: Radio del filtro paso bajo 1D (en píxeles) para no alterar el gradiente natural del cielo.
    - sky_mask: Opcional, máscara binaria de cielo para no sesgar las estadísticas con la silueta terrestre.

    Retorna:
    - Imagen corregida float32 con las mismas dimensiones que la original.
    """
    if amount <= 1e-4:
        return image

    img_f32 = image.astype(np.float32, copy=False)
    is_2d = (img_f32.ndim == 2)
    dir_lower = direction.lower()
    mask_2d = (sky_mask[..., 0] if sky_mask.ndim == 3 else sky_mask) if sky_mask is not None else None

    if HAS_RUST:
        m2d = np.ascontiguousarray(mask_2d, dtype=np.float32) if mask_2d is not None else None
        return astroloom_rs.suppress_cmos_banding_rs(
            np.ascontiguousarray(img_f32, dtype=np.float32),
            direction,
            float(amount),
            float(protect_sigma),
            float(smooth_sigma),
            m2d
        )

    if is_2d:
        out = img_f32.copy()
        if dir_lower in ("horizontal", "both"):
            out = _suppress_1d_banding_axis(
                out, axis=0, amount=amount, protect_sigma=protect_sigma,
                smooth_sigma=smooth_sigma, sky_mask=mask_2d
            )
        if dir_lower in ("vertical", "both"):
            out = _suppress_1d_banding_axis(
                out, axis=1, amount=amount, protect_sigma=protect_sigma,
                smooth_sigma=smooth_sigma, sky_mask=mask_2d
            )
        return out

    # Procesar canales RGB / multicanal de forma independiente
    c = img_f32.shape[2]
    out = np.empty_like(img_f32)
    for ch in range(c):
        ch_data = img_f32[..., ch].copy()
        if dir_lower in ("horizontal", "both"):
            ch_data = _suppress_1d_banding_axis(
                ch_data, axis=0, amount=amount, protect_sigma=protect_sigma,
                smooth_sigma=smooth_sigma, sky_mask=mask_2d
            )
        if dir_lower in ("vertical", "both"):
            ch_data = _suppress_1d_banding_axis(
                ch_data, axis=1, amount=amount, protect_sigma=protect_sigma,
                smooth_sigma=smooth_sigma, sky_mask=mask_2d
            )
        out[..., ch] = ch_data

    return out
