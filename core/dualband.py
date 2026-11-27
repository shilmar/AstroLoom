# core/dualband.py
"""Dual-band OSC processing, palette mapping, and stellar restoration for AstroLoom (Rust-native)."""
from __future__ import annotations

import numpy as np
import astroloom_rs

HAS_RUST = True


def apply_dualband_palette(
    img_rgb: np.ndarray,
    palette: str = "HOO",
    boost_oiii: float = 1.0,
    desat_cores: bool = True
) -> np.ndarray:
    """Applies a dual-band false color palette to an OSC image ('HOO', 'SHO', 'Foraxx', 'Cannistra')."""
    if palette.upper() in ("NONE", "OFF", "RGB"):
        return img_rgb

    return astroloom_rs.apply_dualband_palette_rs(
        np.ascontiguousarray(img_rgb, dtype=np.float32),
        palette,
        float(boost_oiii),
        bool(desat_cores)
    )


def unclip_stars(
    img_rgb: np.ndarray,
    sat_threshold: float = 0.92,
    ring_width: int = 3,
    inner_r: int | None = None,
    outer_r: int | None = None
) -> np.ndarray:
    """Rescues burned/saturated star cores by transferring outer PSF chromatic ratios."""
    img = np.ascontiguousarray(img_rgb, dtype=np.float32)
    eff_width = max(2, outer_r - inner_r) if (inner_r is not None and outer_r is not None) else int(ring_width)
    return astroloom_rs.unclip_stars_rs(img, float(sat_threshold), int(eff_width))


def unpurple(
    img_rgb: np.ndarray,
    amount: float = 1.0,
    protect_ha: bool = True
) -> np.ndarray:
    """Suppresses magenta/purple chromatic aberration halos while preserving Ha nebulosity."""
    if amount <= 0.0:
        return img_rgb

    return astroloom_rs.unpurple_rs(
        np.ascontiguousarray(img_rgb, dtype=np.float32),
        float(amount),
        bool(protect_ha)
    )


def apply_dualband_pipeline_fused(
    img_rgb: np.ndarray,
    palette: str = "HOO",
    boost_oiii: float = 1.0,
    desat_cores: bool = True,
    unclip_amount: float = 0.0,
    unpurple_amount: float = 0.0,
    sky_mask: np.ndarray | None = None
) -> np.ndarray:
    """Fused dual-band pipeline: palette mapping + star unclip + magenta suppression in a single native pass."""
    return astroloom_rs.apply_dualband_pipeline_fused(
        np.ascontiguousarray(img_rgb, dtype=np.float32),
        palette,
        float(boost_oiii),
        bool(desat_cores),
        float(unclip_amount),
        float(unpurple_amount),
        np.ascontiguousarray(sky_mask, dtype=np.float32) if sky_mask is not None else None
    )
