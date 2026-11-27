# core/display_transfer.py
"""
core/display_transfer.py - Motor de visualización acelerada por Look-Up Table (LUT) 1D.
Permite arrastre de deslizadores de estirado (MTF / GHS), curvas tonales y contraste a >500 FPS
mediante indexación paralela en Rust sobre matrices uint16.
"""

from typing import Optional
import numpy as np
import astroloom_rs

from core.stretch import manual_stretch
from core.ghs import compute_ghs_lut

HAS_RUST = True


def apply_display_lut16(img_u16: np.ndarray, lut_u8: np.ndarray, out_u8: np.ndarray) -> None:
    """
    Aplica una Look-Up Table (LUT) 1D de 65.536 valores uint8 sobre una imagen RGB en uint16.
    Acelerado con Rust Rayon (~0.2 ms para 2 Megapíxeles).
    """
    astroloom_rs.apply_display_lut16_rs(
        np.ascontiguousarray(img_u16, dtype=np.uint16),
        np.ascontiguousarray(lut_u8, dtype=np.uint8),
        out_u8,
    )


def compute_combined_stf_lut(
    stretch_mode: str = "mtf",
    bp: float = 0.0,
    mtf: float = 0.1,
    ghs_sp: float = 0.02,
    ghs_d: float = 5.0,
    ghs_b: float = 2.0,
    ghs_hp: float = 0.0,
    ghs_lp: float = 0.0,
    ghs_bp: float = 0.0,
    curve_lut_rgb: Optional[np.ndarray] = None,
    contrast: float = 0.0,
    contrast_pivot: float = 0.5,
    lut_size: int = 65536
) -> np.ndarray:
    """
    Compone en una sola Look-Up Table (LUT) 1D de 16 bits (65.536 valores) las etapas
    de estirado (MTF o GHS), curva tonal spline RGB y contraste.
    """
    x = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)

    if stretch_mode == "ghs":
        y = compute_ghs_lut(
            sp=ghs_sp, d=ghs_d, b=ghs_b,
            hp=ghs_hp, lp=ghs_lp, bp=ghs_bp,
            lut_size=lut_size
        )
    else:
        y = manual_stretch(x, black_point=bp, midtone=mtf)

    if curve_lut_rgb is not None and len(curve_lut_rgb) > 0:
        c_len = len(curve_lut_rgb)
        c_x = np.linspace(0.0, 1.0, c_len, dtype=np.float32)
        y = np.interp(y, c_x, curve_lut_rgb).astype(np.float32)

    if abs(contrast) > 1e-4:
        factor = float(np.tan((contrast + 1.0) * (np.pi / 4.0)))
        diff = y - contrast_pivot
        y = np.clip(contrast_pivot + diff * factor, 0.0, 1.0)

    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)
