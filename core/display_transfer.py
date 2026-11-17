# core/display_transfer.py
"""
core/display_transfer.py - Motor de visualización acelerada por Look-Up Table (LUT) 1D.
Inspirado en la tecnología Display Transfer Function / Screen Transfer Function (STF).
Permite arrastre de deslizadores de estirado (MTF / GHS), curvas tonales y contraste a más de 500-1.000 FPS
mediante indexación paralela en CPU sobre matrices uint16, manteniendo intactos los datos de 32 bits en RAM.
"""

from typing import Optional
import numpy as np

try:
    import astroloom_rs
    HAS_RUST = hasattr(astroloom_rs, "apply_display_lut16_rs")
except ImportError:
    HAS_RUST = False

HAS_NUMBA = False

from core.stretch import manual_stretch
from core.ghs import compute_ghs_lut


def apply_display_lut16(img_u16: np.ndarray, lut_u8: np.ndarray, out_u8: np.ndarray) -> None:
    """
    Aplica una Look-Up Table (LUT) 1D de 65.536 valores uint8 sobre una imagen RGB en uint16.
    Acelerado con Rust Rayon (~0.2 ms para 2 Megapíxeles, > 4.000 FPS).
    """
    if HAS_RUST and img_u16.flags.c_contiguous and out_u8.flags.c_contiguous and lut_u8.flags.c_contiguous:
        try:
            astroloom_rs.apply_display_lut16_rs(img_u16, lut_u8, out_u8)
            return
        except Exception:
            pass
    out_u8[:] = lut_u8[img_u16]


def apply_display_lut16_dual(
    sl_u16: np.ndarray,
    st_u16: np.ndarray,
    lut_sl: np.ndarray,
    lut_st: np.ndarray,
    star_intensity: float,
    out_u8: np.ndarray
) -> None:
    """
    Aplica y fusiona en una sola pasada de memoria las capas Starless y Stars indexadas por LUT.
    Acelerado con Rust Rayon (~0.8 ms para 2 Megapíxeles).
    """
    if HAS_RUST and sl_u16.flags.c_contiguous and st_u16.flags.c_contiguous and out_u8.flags.c_contiguous:
        try:
            astroloom_rs.apply_display_lut16_dual_rs(sl_u16, st_u16, lut_sl, lut_st, float(star_intensity), out_u8)
            return
        except Exception:
            pass
    v_sl = lut_sl[sl_u16].astype(np.float32)
    v_st = lut_st[st_u16].astype(np.float32) * float(star_intensity)
    out_u8[:] = np.clip(v_sl + v_st, 0, 255).astype(np.uint8)


# Alias para retrocompatibilidad absoluta
apply_display_lut16_numba = apply_display_lut16
apply_display_lut16_dual_numba = apply_display_lut16_dual


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
    
    Coste de cómputo: ~0.03 - 0.05 ms en cualquier procesador moderno.
    Retorna un array np.ndarray uint8 de longitud lut_size con valores en [0, 255].
    """
    x = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)

    # 1. Estirado Base (MTF o GHS)
    if stretch_mode == "ghs":
        y = compute_ghs_lut(
            sp=ghs_sp,
            d=ghs_d,
            b=ghs_b,
            hp=ghs_hp,
            lp=ghs_lp,
            bp=ghs_bp,
            lut_size=lut_size
        )
    else:
        y = manual_stretch(x, black_point=bp, midtone=mtf)

    # 2. Curva Tonal Spline RGB (si está activa)
    if curve_lut_rgb is not None and len(curve_lut_rgb) > 0:
        c_len = len(curve_lut_rgb)
        c_x = np.linspace(0.0, 1.0, c_len, dtype=np.float32)
        y = np.interp(y, c_x, curve_lut_rgb).astype(np.float32)

    # 3. Ajuste de Contraste Sigmoidal (si es distinto de 0)
    if abs(contrast) > 1e-4:
        factor = float(np.tan((contrast + 1.0) * (np.pi / 4.0)))
        diff = y - contrast_pivot
        y = np.clip(contrast_pivot + diff * factor, 0.0, 1.0)

    # 4. Proyección final a uint8 [0, 255]
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)
