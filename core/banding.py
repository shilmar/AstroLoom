# core/banding.py
"""
Módulo de Supresión de Banding CMOS de Sensor (Banding Reduction, Rust-native).
Elimina el ruido estructurado por filas y columnas en sensores CMOS protegiendo estrellas y nebulosas.
"""

from typing import Optional
import numpy as np
import astroloom_rs

HAS_RUST = True


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
    """
    if amount <= 1e-4:
        return image

    img_f32 = np.ascontiguousarray(image, dtype=np.float32)
    mask_2d = (sky_mask[..., 0] if sky_mask.ndim == 3 else sky_mask) if sky_mask is not None else None
    m2d = np.ascontiguousarray(mask_2d, dtype=np.float32) if mask_2d is not None else None
    return astroloom_rs.suppress_cmos_banding_rs(
        img_f32,
        direction,
        float(amount),
        float(protect_sigma),
        float(smooth_sigma),
        m2d,
    )
