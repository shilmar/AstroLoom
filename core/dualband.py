# core/dualband.py
"""Dual-band OSC processing, palette mapping, and stellar restoration for AstroLoom.

Provides:
- Ha and [OIII] unmixing from color sensor frames.
- Narrowband palettes: HOO, Synthetic SHO, Dynamic Foraxx, Cannistra.
- unclip_stars: restores true chromatic ratios to burned/saturated star cores.
- unpurple: suppresses magenta/violet chromatic aberration without clipping Ha nebulosity.
- desaturate_star_cores: neutralizes harsh narrow-band cyan/magenta tint in star centers.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def extract_ha_oiii(
    img_rgb: np.ndarray,
    crosstalk_ha_in_oiii: float = 0.04
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts Ha and [OIII] signals from an OSC RGB image.
    
    Returns (ha, oiii, bg_pedestal).
    """
    img = np.ascontiguousarray(img_rgb, dtype=np.float32)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    # Local / global background floor (5th percentile to protect deep shadows)
    bg_r = float(np.percentile(r, 5))
    bg_g = float(np.percentile(g, 5))
    bg_b = float(np.percentile(b, 5))
    bg = np.array([bg_r, bg_g, bg_b], dtype=np.float32)

    ha_raw = np.maximum(0.0, r - bg_r)
    oiii_raw = np.maximum(0.0, 0.5 * (g - bg_g) + 0.5 * (b - bg_b))

    # Mild crosstalk compensation
    if crosstalk_ha_in_oiii > 0.0:
        oiii = np.maximum(0.0, oiii_raw - crosstalk_ha_in_oiii * ha_raw)
    else:
        oiii = oiii_raw

    return ha_raw, oiii, bg


def apply_dualband_palette(
    img_rgb: np.ndarray,
    palette: str = "HOO",
    boost_oiii: float = 1.0,
    desat_cores: bool = True
) -> np.ndarray:
    """Applies a dual-band false color palette to an OSC image.
    
    Supported palettes: 'HOO', 'SHO', 'Foraxx', 'Cannistra'.
    """
    if palette.upper() in ("NONE", "OFF", "RGB"):
        return img_rgb

    if HAS_RUST:
        return astroloom_rs.apply_dualband_palette_rs(
            np.ascontiguousarray(img_rgb, dtype=np.float32),
            palette,
            float(boost_oiii),
            bool(desat_cores)
        )

    ha, oiii, bg = extract_ha_oiii(img_rgb)
    oiii_scaled = oiii * max(0.1, boost_oiii)

    palette_key = palette.strip().upper()
    h, w = ha.shape[:2]
    out = np.empty((h, w, 3), dtype=np.float32)

    if palette_key == "HOO":
        out[..., 0] = ha
        out[..., 1] = oiii_scaled
        out[..., 2] = oiii_scaled
    elif palette_key == "SHO":
        # Synthetic Hubble palette: SII synthesized from Ha
        sii = 0.45 * ha
        out[..., 0] = sii
        out[..., 1] = 0.80 * ha + 0.20 * oiii_scaled
        out[..., 2] = oiii_scaled
    elif palette_key == "FORAXX":
        # Dynamic Foraxx formulation: non-linear green modulation
        norm_ha = np.clip(ha / (np.max(ha) + 1e-6), 0.0, 1.0)
        norm_o3 = np.clip(oiii_scaled / (np.max(oiii_scaled) + 1e-6), 0.0, 1.0)
        prod = norm_ha * norm_o3
        f = np.power(prod, np.maximum(0.01, 1.0 - prod))
        g_synth = f * ha + (1.0 - f) * oiii_scaled
        out[..., 0] = ha
        out[..., 1] = g_synth
        out[..., 2] = oiii_scaled
    elif palette_key == "CANNISTRA":
        out[..., 0] = ha
        out[..., 1] = np.maximum(oiii_scaled, 0.50 * ha)
        out[..., 2] = oiii_scaled
    else:
        # Fallback to HOO
        out[..., 0] = ha
        out[..., 1] = oiii_scaled
        out[..., 2] = oiii_scaled

    # Restore neutral pedestal
    med_bg = np.mean(bg)
    out += med_bg

    if desat_cores:
        out = desaturate_star_cores(out)

    return np.ascontiguousarray(out, dtype=np.float32)


def desaturate_star_cores(
    img_rgb: np.ndarray,
    threshold: float = 0.75,
    feather: float = 0.20
) -> np.ndarray:
    """Neutralizes harsh narrow-band cyan/magenta tint in bright star cores."""
    img = np.ascontiguousarray(img_rgb, dtype=np.float32)
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    # Weight curve based on luminance
    mask = np.clip((luma - threshold) / max(1e-4, feather), 0.0, 1.0)
    if np.max(mask) < 1e-4:
        return img

    mask_3c = mask[..., np.newaxis]
    luma_3c = luma[..., np.newaxis]
    return (img * (1.0 - mask_3c)) + (luma_3c * mask_3c)


def unclip_stars(
    img_rgb: np.ndarray,
    sat_threshold: float = 0.92,
    ring_width: int = 3,
    inner_r: int | None = None,
    outer_r: int | None = None
) -> np.ndarray:
    """Rescues burned/saturated star cores by transferring outer PSF chromatic ratios.
    
    Preserves total luminance while replacing flat white clipped discs with true star color.
    """
    img = np.ascontiguousarray(img_rgb, dtype=np.float32)
    if HAS_RUST and inner_r is None and outer_r is None:
        return astroloom_rs.unclip_stars_rs(img, float(sat_threshold), int(ring_width))

    sat_mask = np.any(img >= sat_threshold, axis=-1).astype(np.uint8)
    if not np.any(sat_mask):
        return img

    eff_width = outer_r - inner_r if (inner_r is not None and outer_r is not None) else ring_width
    eff_width = max(2, eff_width)

    num_labels, labels = cv2.connectedComponents(sat_mask)
    if num_labels <= 1:
        return img

    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * eff_width + 1, 2 * eff_width + 1))
    out = img.copy()
    eps = 1e-6
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    for label_id in range(1, num_labels):
        star_mask = (labels == label_id).astype(np.uint8)
        dilated = cv2.dilate(star_mask, k_ring)
        ring_mask = (dilated > 0) & (sat_mask == 0)

        if not np.any(ring_mask):
            continue

        r_ring = img[..., 0][ring_mask]
        g_ring = img[..., 1][ring_mask]
        b_ring = img[..., 2][ring_mask]

        ratio_rg = float(np.median((r_ring + eps) / (g_ring + eps)))
        ratio_bg = float(np.median((b_ring + eps) / (g_ring + eps)))

        denom = 0.2126 * ratio_rg + 0.7152 + 0.0722 * ratio_bg + eps
        star_luma = luma[star_mask > 0]
        synth_g = star_luma / denom
        synth_r = synth_g * ratio_rg
        synth_b = synth_g * ratio_bg

        out[star_mask > 0, 0] = synth_r
        out[star_mask > 0, 1] = synth_g
        out[star_mask > 0, 2] = synth_b

    return np.ascontiguousarray(out, dtype=np.float32)


def unpurple(
    img_rgb: np.ndarray,
    amount: float = 1.0,
    protect_ha: bool = True
) -> np.ndarray:
    """Suppresses magenta/purple chromatic aberration halos while preserving Ha nebulosity.
    
    Magenta fringe occurs when both R and B exceed G around stars (B approx R >> G).
    True Ha nebulosity has R >> B and is protected.
    """
    if amount <= 0.0:
        return img_rgb

    if HAS_RUST:
        return astroloom_rs.unpurple_rs(
            np.ascontiguousarray(img_rgb, dtype=np.float32),
            float(amount),
            bool(protect_ha)
        )

    img = np.ascontiguousarray(img_rgb, dtype=np.float32)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    # Magenta fringe: both R > G and B > G
    fringe = np.maximum(0.0, np.minimum(r - g, b - g))
    if np.max(fringe) < 1e-5:
        return img

    # Ha protection: True Ha nebula has high red and low blue (r - b > 0.15)
    if protect_ha:
        ha_dominance = np.clip((r - b) / (r + 1e-4), 0.0, 1.0)
        # Suppress fringe correction where Ha dominates
        fringe = fringe * (1.0 - ha_dominance)

    amt = float(np.clip(amount, 0.0, 1.0))
    corr_r = r - amt * fringe * 0.75
    corr_b = b - amt * fringe

    out = np.stack([corr_r, g, corr_b], axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


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
    if HAS_RUST:
        return astroloom_rs.apply_dualband_pipeline_fused(
            np.ascontiguousarray(img_rgb, dtype=np.float32),
            palette,
            float(boost_oiii),
            bool(desat_cores),
            float(unclip_amount),
            float(unpurple_amount),
            np.ascontiguousarray(sky_mask, dtype=np.float32) if sky_mask is not None else None
        )
    out = apply_dualband_palette(img_rgb, palette=palette, boost_oiii=boost_oiii, desat_cores=desat_cores)
    if unclip_amount > 1e-4:
        out = unclip_stars(out)
    if unpurple_amount > 1e-4:
        out = unpurple(out, amount=unpurple_amount)
    return out

