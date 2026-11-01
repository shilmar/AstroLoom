# core/pcc.py
"""Photometric Color Calibration (PCC) against Gaia DR3 / Stellar Locus standard for AstroLoom.

Provides:
- Circular aperture photometry on unsaturated stars.
- Robust Theil-Sen regression for log(R/G) and log(B/G) vs Gaia (BP - RP).
- Calibration anchored to the solar standard G2V (BP - RP = 0.82).
- Offline empirical stellar locus fallback when astrometry metadata is unavailable.
- Non-destructive gain application with neutral background preservation.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import theilslopes


def detect_stars_photometry(
    img_rgb: np.ndarray,
    aperture_r: int = 5,
    annulus_inner_r: int = 8,
    annulus_outer_r: int = 13,
    max_stars: int = 400,
    sat_limit: float = 0.90
) -> list[dict]:
    """Measures net aperture flux for isolated unsaturated stars across R, G, B channels."""
    img = np.ascontiguousarray(img_rgb, dtype=np.float32)
    h, w = img.shape[:2]

    # Use green channel for star detection (best SNR on Bayer sensors)
    green = img[..., 1]
    blurred = cv2.GaussianBlur(green, (5, 5), 1.2)
    lap = cv2.Laplacian(blurred, cv2.CV_32F)

    bg_median = float(np.median(green))
    bg_std = float(1.4826 * np.median(np.abs(green - bg_median)))
    thresh = bg_median + 4.0 * max(1e-4, bg_std)

    # Local maxima detection
    dilated = cv2.dilate(green, np.ones((7, 7), np.uint8))
    peaks = (green == dilated) & (green > thresh) & (green < sat_limit)

    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return []

    # Sort peaks by brightness and limit count
    peak_vals = green[ys, xs]
    order = np.argsort(-peak_vals)[:max_stars * 2]
    xs = xs[order]
    ys = ys[order]

    # Pre-generate circular masks for speed
    margin = annulus_outer_r + 2
    y_grid, x_grid = np.ogrid[-margin:margin + 1, -margin:margin + 1]
    dist_sq = x_grid**2 + y_grid**2
    mask_ap = dist_sq <= (aperture_r**2)
    mask_ann = (dist_sq >= annulus_inner_r**2) & (dist_sq <= annulus_outer_r**2)

    results = []
    for x, y in zip(xs, ys):
        if x < margin or x >= w - margin or y < margin or y >= h - margin:
            continue

        patch = img[y - margin:y + margin + 1, x - margin:x + margin + 1]
        # Reject if patch contains saturated pixels
        if np.any(patch >= sat_limit):
            continue

        # Photometry per channel
        fluxes = []
        is_valid = True
        for c in range(3):
            c_patch = patch[..., c]
            ann_vals = c_patch[mask_ann]
            if len(ann_vals) == 0:
                is_valid = False
                break
            bg_c = float(np.median(ann_vals))
            net_flux = float(np.sum(c_patch[mask_ap] - bg_c))
            if net_flux <= 1e-4:
                is_valid = False
                break
            fluxes.append(net_flux)

        if is_valid and len(fluxes) == 3:
            results.append({
                "x": float(x),
                "y": float(y),
                "flux_r": fluxes[0],
                "flux_g": fluxes[1],
                "flux_b": fluxes[2],
            })
            if len(results) >= max_stars:
                break

    return results


def fit_theil_sen_pcc(
    stars: list[dict],
    bp_rp_catalog: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Fits robust Theil-Sen slopes against Gaia (BP - RP) or empirical stellar locus.
    
    Zero point is calibrated to the Sun (G2V spectral type: BP - RP = 0.82).
    Returns (k_r, k_g=1.0, k_b) multipliers.
    """
    if len(stars) < 6:
        return 1.0, 1.0, 1.0

    f_r = np.array([s["flux_r"] for s in stars], dtype=np.float64)
    f_g = np.array([s["flux_g"] for s in stars], dtype=np.float64)
    f_b = np.array([s["flux_b"] for s in stars], dtype=np.float64)

    eps = 1e-7
    log_rg = np.log10((f_r + eps) / (f_g + eps))
    log_bg = np.log10((f_b + eps) / (f_g + eps))

    # Reject non-finite values
    finite_mask = np.isfinite(log_rg) & np.isfinite(log_bg)
    log_rg = log_rg[finite_mask]
    log_bg = log_bg[finite_mask]

    if len(log_rg) < 6:
        return 1.0, 1.0, 1.0

    solar_bp_rp = 0.82

    if bp_rp_catalog is not None and len(bp_rp_catalog) == len(stars):
        x = np.asarray(bp_rp_catalog, dtype=np.float64)[finite_mask]
        # Check that we have at least 2 distinct x values to fit a slope
        if len(np.unique(np.round(x, 2))) >= 2:
            res_r = theilslopes(log_rg, x)
            res_b = theilslopes(log_bg, x)

            slope_r = getattr(res_r, 'slope', res_r[0])
            intercept_r = getattr(res_r, 'intercept', res_r[1])
            slope_b = getattr(res_b, 'slope', res_b[0])
            intercept_b = getattr(res_b, 'intercept', res_b[1])

            expected_log_rg = slope_r * solar_bp_rp + intercept_r
            expected_log_bg = slope_b * solar_bp_rp + intercept_b
        else:
            expected_log_rg = float(np.median(log_rg))
            expected_log_bg = float(np.median(log_bg))
    else:
        # Empirical Main Sequence Locus: median star population is approx solar (BP-RP ~ 0.82)
        expected_log_rg = float(np.median(log_rg))
        expected_log_bg = float(np.median(log_bg))

    k_r = float(10.0 ** (-expected_log_rg))
    k_b = float(10.0 ** (-expected_log_bg))

    # Clamp gains to sensible physical limits [0.25, 4.0] to prevent explosive tints
    k_r = float(np.clip(k_r, 0.25, 4.0))
    k_b = float(np.clip(k_b, 0.25, 4.0))

    return k_r, 1.0, k_b


def apply_pcc_gains(
    img_rgb: np.ndarray,
    k_r: float,
    k_b: float,
    preserve_neutral_bg: bool = True
) -> np.ndarray:
    """Applies PCC channel multipliers while preserving sky background neutral pedestal."""
    img = np.ascontiguousarray(img_rgb, dtype=np.float32)

    if preserve_neutral_bg:
        bg_r = float(np.percentile(img[..., 0], 10))
        bg_g = float(np.percentile(img[..., 1], 10))
        bg_b = float(np.percentile(img[..., 2], 10))
        bg_target = (bg_r + bg_g + bg_b) / 3.0

        r_signal = img[..., 0] - bg_r
        g_signal = img[..., 1] - bg_g
        b_signal = img[..., 2] - bg_b

        cal_r = r_signal * k_r + bg_target
        cal_g = g_signal + bg_target
        cal_b = b_signal * k_b + bg_target
    else:
        cal_r = img[..., 0] * k_r
        cal_g = img[..., 1]
        cal_b = img[..., 2] * k_b

    out = np.stack([cal_r, cal_g, cal_b], axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def calibrate_image_pcc(
    img_rgb: np.ndarray,
    preserve_neutral_bg: bool = True
) -> tuple[np.ndarray, dict]:
    """High-level one-shot PCC calibration function.
    
    Detects stars, fits robust physical gains, and returns (calibrated_image, stats).
    """
    stars = detect_stars_photometry(img_rgb)
    k_r, k_g, k_b = fit_theil_sen_pcc(stars)
    calibrated = apply_pcc_gains(img_rgb, k_r, k_b, preserve_neutral_bg=preserve_neutral_bg)
    stats = {
        "num_stars": len(stars),
        "k_r": k_r,
        "k_g": 1.0,
        "k_b": k_b,
    }
    return calibrated, stats
