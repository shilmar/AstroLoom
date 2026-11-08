"""core/dust.py - Automatic Sensor Dust Spot Detection, Visualization and Healing for AstroLoom.

Inspired by Lightcraft's classical computer vision pipeline:
1. High-pass / Gaussian difference with local texture normalization for robust detection on smooth gradients.
2. Morphological filtering (circularity, fill factor, bounding limits) rejecting stars, lines, and textures.
3. Normalized convolution seamless tone transfer (Poisson approximation) & biharmonic inpainting.
4. Visualize Spots high-pass inspection overlay for fine sensor spot inspection.
"""
from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import cv2


def _to_luma(img: np.ndarray) -> np.ndarray:
    """Convert RGB or single-channel image to 2D float32 luminance in [0.0, 1.0]."""
    if img.ndim == 2:
        luma = img.astype(np.float32)
    elif img.ndim == 3 and img.shape[2] == 1:
        luma = img[..., 0].astype(np.float32)
    elif img.ndim == 3 and img.shape[2] >= 3:
        # Standard Rec.709 sRGB luminance coefficients
        luma = (
            0.2126 * img[..., 0].astype(np.float32)
            + 0.7152 * img[..., 1].astype(np.float32)
            + 0.0722 * img[..., 2].astype(np.float32)
        )
    else:
        raise ValueError(f"Unsupported image shape for luma conversion: {img.shape}")

    # Normalize to [0.0, 1.0] if uint8 or uint16
    if img.dtype == np.uint8:
        luma /= 255.0
    elif img.dtype == np.uint16:
        luma /= 65535.0
    else:
        # float32: clamp to [0.0, 1.0]
        luma = np.clip(luma, 0.0, 1.0)
    return luma


def detect_dust_spots(
    img: np.ndarray,
    sensitivity: float = 50.0,
    max_spots: int = 100,
    r_min_frac: float = 0.0015,
    r_max_frac: float = 0.035,
    sky_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Detect sensor dust spots on smooth areas (sky, flat background) of the image.

    Parameters
    ----------
    img : np.ndarray
        Input image (float32 [0..1] or uint8 [0..255]).
    sensitivity : float
        Detection sensitivity in [0..100]. Higher values detect fainter spots.
    max_spots : int
        Maximum number of spots to return (ranked by contrast/strength).
    r_min_frac : float
        Minimum spot radius as fraction of the image's longer edge.
    r_max_frac : float
        Maximum spot radius as fraction of the image's longer edge.
    sky_mask : Optional[np.ndarray]
        Optional binary/float mask (1 = sky/background, 0 = foreground/terrain).

    Returns
    -------
    List[Dict[str, Any]]
        List of detected spots: [{"x": norm_x, "y": norm_y, "radius": norm_r, "strength": peak}, ...]
    """
    if img is None or img.size == 0:
        return []

    h, w = img.shape[:2]
    if w < 32 or h < 32:
        return []

    long_edge = float(max(w, h))
    luma = _to_luma(img)

    # 1. Background surroundings estimation via Gaussian blur
    sigma_around = max(2.0, 0.012 * long_edge)
    k_around = int(math.ceil(sigma_around * 4.0)) | 1
    around = cv2.GaussianBlur(luma, (k_around, k_around), sigma_around)

    # 2. Local texture estimation: mean absolute deviation from surroundings
    detail = np.abs(luma - around)
    sigma_texture = max(3.0, 0.020 * long_edge)
    k_texture = int(math.ceil(sigma_texture * 4.0)) | 1
    texture = cv2.GaussianBlur(detail, (k_texture, k_texture), sigma_texture)

    # 3. Candidate pixel selection
    sens_norm = np.clip(sensitivity / 100.0, 0.0, 1.0)
    min_dark = float(0.030 - 0.022 * sens_norm)
    dark = np.maximum(0.0, around - luma)

    candidate_mask = (dark > min_dark) & (dark > 2.8 * texture) & (luma > 0.05)
    if sky_mask is not None:
        if sky_mask.shape[:2] != (h, w):
            sm = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            sm = sky_mask
        candidate_mask &= (sm > 0.3)

    if not np.any(candidate_mask):
        return []

    # 4. Connected components analysis (morphology)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=4
    )

    rmin_px = r_min_frac * long_edge
    rmax_px = r_max_frac * long_edge

    spots: List[Dict[str, Any]] = []

    for label in range(1, num_labels):
        x0 = stats[label, cv2.CC_STAT_LEFT]
        y0 = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]

        r = max(bw, bh) / 2.0

        # Reject if outside radius bounds
        if r < rmin_px or r > rmax_px:
            continue

        # Roundness / aspect ratio check: allows slightly elliptical dust donuts
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > 2.2:
            continue

        # Area fill factor check: ensures solid blob rather than thin hollow circle or line
        if area < 0.38 * (bw * bh):
            continue

        # Measure peak darkness within the spot bbox
        sub_dark = dark[y0 : y0 + bh, x0 : x0 + bw]
        sub_lbl = labels[y0 : y0 + bh, x0 : x0 + bw] == label
        if not np.any(sub_lbl):
            continue
        peak = float(np.max(sub_dark[sub_lbl]))

        # Normalized coordinates and radius (enclosing the penumbra)
        spots.append(
            {
                "x": float((cx + 0.5) / w),
                "y": float((cy + 0.5) / h),
                "radius": float((r / long_edge) * 1.6),
                "strength": peak,
            }
        )

    # Sort descending by contrast/strength and truncate
    spots.sort(key=lambda s: s["strength"], reverse=True)
    if len(spots) > max_spots:
        spots = spots[:max_spots]

    return spots


def pick_donor_source(
    img: np.ndarray,
    target_norm: Tuple[float, float],
    radius_norm: float,
    existing_spots: Optional[List[Dict[str, Any]]] = None,
    avoid_offset: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """Find the best clean donor patch offset around target (normalized coordinates).

    Tests 48 candidate positions across 3 concentric rings (2.4r, 3.2r, 4.2r),
    scoring texture consistency and avoiding image boundaries or other dust spots.
    """
    h, w = img.shape[:2]
    long_edge = float(max(w, h))

    tx = target_norm[0] * w
    ty = target_norm[1] * h
    r_px = radius_norm * long_edge

    luma = _to_luma(img)

    rings = [2.4, 3.2, 4.2]
    num_angles = 16
    candidates: List[Tuple[float, float]] = []

    for ring_mult in rings:
        dist = r_px * ring_mult
        for i in range(num_angles):
            angle = i * (2.0 * math.pi / num_angles)
            dx = math.cos(angle) * dist
            dy = math.sin(angle) * dist
            candidates.append((dx, dy))

    best_offset: Optional[Tuple[float, float]] = None
    best_score = float("inf")

    # Spokes for ring statistics
    spoke_angles = [k * (2.0 * math.pi / 24.0) for k in range(24)]

    # Compute target surrounding samples
    target_samples = []
    r_around = r_px * 1.3
    for a in spoke_angles:
        sx = int(round(tx + math.cos(a) * r_around))
        sy = int(round(ty + math.sin(a) * r_around))
        if 0 <= sx < w and 0 <= sy < h:
            target_samples.append(luma[sy, sx])
    if not target_samples:
        return None
    target_mean = float(np.mean(target_samples))

    for dx, dy in candidates:
        cx = tx + dx
        cy = ty + dy

        # 1. Boundary check: whole radius must fit inside the image
        if cx - r_px < 0 or cx + r_px >= w or cy - r_px < 0 or cy + r_px >= h:
            continue

        # 2. Avoid offset if specified (e.g. refresh source)
        if avoid_offset is not None:
            adx = avoid_offset[0] * w
            ady = avoid_offset[1] * h
            if math.hypot(dx - adx, dy - ady) < r_px:
                continue

        # 3. Check collision with existing spots (fix of Lightcraft issue)
        norm_cx = cx / w
        norm_cy = cy / h
        collision = False
        if existing_spots:
            for other in existing_spots:
                ox = other["x"]
                oy = other["y"]
                orad = other.get("radius", radius_norm)
                # If distance is less than combined radii + safety margin
                if math.hypot(norm_cx - ox, norm_cy - oy) < (radius_norm + orad):
                    collision = True
                    break
        if collision:
            continue

        # 4. Measure candidate score: texture matching + inside variance
        cand_samples = []
        for a in spoke_angles:
            sx = int(round(cx + math.cos(a) * r_around))
            sy = int(round(cy + math.sin(a) * r_around))
            cand_samples.append(luma[sy, sx])

        diff_score = float(np.mean((np.array(cand_samples) - np.array(target_samples)) ** 2))

        # Check interior variance of donor patch (must be clean and smooth)
        r_inner = r_px * 0.6
        inner_samples = []
        for a in spoke_angles:
            sx = int(round(cx + math.cos(a) * r_inner))
            sy = int(round(cy + math.sin(a) * r_inner))
            inner_samples.append(luma[sy, sx])
        var_score = float(np.var(inner_samples))

        total_score = diff_score + 2.0 * var_score

        if total_score < best_score:
            best_score = total_score
            best_offset = (float(dx / w), float(dy / h))

    return best_offset


def heal_single_spot(
    img: np.ndarray,
    spot: Dict[str, Any],
    source_offset: Optional[Tuple[float, float]] = None,
    mode: str = "heal",
    opacity: float = 1.0,
    feather: float = 0.5,
) -> np.ndarray:
    """Heal a single circular spot on img using normalized convolution seamless blending.

    Parameters
    ----------
    img : np.ndarray
        Float32 image [0..1], 2D or 3D.
    spot : Dict[str, Any]
        Spot dictionary containing 'x', 'y', 'radius'.
    source_offset : Optional[Tuple[float, float]]
        Normalized (dx, dy) donor source offset. If None, auto-picked or inpainting fallback.
    mode : str
        'heal' (tone-matched seamless donor blend), 'clone' (direct clone), or 'inpaint' (donor-free hole fill).
    opacity : float
        Blending opacity in [0..1].
    feather : float
        Feather ratio in [0..1].
    """
    h, w = img.shape[:2]
    long_edge = float(max(w, h))

    cx = spot["x"] * w
    cy = spot["y"] * h
    r_px = max(1.0, spot["radius"] * long_edge)

    # Bounding box of operation with safety margin
    pad = int(math.ceil(r_px * 2.2))
    x0 = max(0, int(math.floor(cx - pad)))
    y0 = max(0, int(math.floor(cy - pad)))
    x1 = min(w, int(math.ceil(cx + pad)))
    y1 = min(h, int(math.ceil(cy + pad)))

    if x1 <= x0 or y1 <= y0:
        return img

    bw = x1 - x0
    bh = y1 - y0

    # Build local coordinate grids
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.hypot(xs - cx, ys - cy)

    # Smoothstep feathered alpha
    inner_r = r_px * (1.0 - np.clip(feather, 0.0, 1.0) * 0.8)
    smooth_t = np.clip((dist - inner_r) / max(1e-4, r_px - inner_r), 0.0, 1.0)
    alpha = (1.0 - (smooth_t * smooth_t * (3.0 - 2.0 * smooth_t))) * opacity
    alpha = np.clip(alpha, 0.0, 1.0)

    if not np.any(alpha > 0.01):
        return img

    out = img.copy()
    target_patch = out[y0:y1, x0:x1]

    # Donor-free inpainting fallback
    if mode == "inpaint" or (source_offset is None and mode == "heal"):
        # Auto-pick donor source if possible
        if mode == "heal":
            source_offset = pick_donor_source(img, (spot["x"], spot["y"]), spot["radius"])

    if source_offset is not None and mode in ("heal", "clone"):
        dx_px = source_offset[0] * w
        dy_px = source_offset[1] * h

        # Sample donor patch using cv2.remap for sub-pixel bilinear accuracy
        map_x = (xs + dx_px).astype(np.float32)
        map_y = (ys + dy_px).astype(np.float32)
        source_patch = cv2.remap(
            img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        if mode == "clone":
            if target_patch.ndim == 3 and alpha.ndim == 2:
                a3 = alpha[..., np.newaxis]
            else:
                a3 = alpha
            out[y0:y1, x0:x1] = target_patch * (1.0 - a3) + source_patch * a3
            return out

        # Heal mode: Normalized Convolution Poisson approximation
        outside = (alpha < 0.02).astype(np.float32)
        diff = (target_patch - source_patch) * (outside[..., np.newaxis] if target_patch.ndim == 3 else outside)

        sigma = max(1.0, float(r_px * 0.6))
        ksize = int(math.ceil(sigma * 4.0)) | 1

        bd = cv2.GaussianBlur(diff, (ksize, ksize), sigma)
        bm = cv2.GaussianBlur(outside, (ksize, ksize), sigma)
        if target_patch.ndim == 3:
            bm = bm[..., np.newaxis]

        low_tone = bd / np.maximum(bm, 1e-4)
        healed_patch = np.clip(source_patch + low_tone, 0.0, 1.0 if img.dtype == np.float32 else 255.0)

        if target_patch.ndim == 3 and alpha.ndim == 2:
            a3 = alpha[..., np.newaxis]
        else:
            a3 = alpha
        out[y0:y1, x0:x1] = target_patch * (1.0 - a3) + healed_patch * a3
        return out

    # Fallback inpaint via Telea on 8-bit representation
    hole_mask = (alpha > 0.05).astype(np.uint8) * 255
    if img.ndim == 3:
        # Process each channel or uint8 conversion
        patch_u8 = np.clip(target_patch * 255.0, 0, 255).astype(np.uint8)
        inpaint_radius = max(2, int(round(r_px * 0.5)))
        healed_u8 = cv2.inpaint(patch_u8, hole_mask, inpaint_radius, cv2.INPAINT_TELEA)
        healed_f32 = healed_u8.astype(np.float32) / 255.0
        a3 = alpha[..., np.newaxis]
        out[y0:y1, x0:x1] = target_patch * (1.0 - a3) + healed_f32 * a3
    else:
        patch_u8 = np.clip(target_patch * 255.0, 0, 255).astype(np.uint8)
        inpaint_radius = max(2, int(round(r_px * 0.5)))
        healed_u8 = cv2.inpaint(patch_u8, hole_mask, inpaint_radius, cv2.INPAINT_TELEA)
        healed_f32 = healed_u8.astype(np.float32) / 255.0
        out[y0:y1, x0:x1] = target_patch * (1.0 - alpha) + healed_f32 * alpha

    return out


def apply_dust_healing(
    img: np.ndarray,
    spots: List[Dict[str, Any]],
    mode: str = "heal",
) -> np.ndarray:
    """Apply healing to a collection of detected dust spots on img."""
    if img is None or not spots:
        return img

    out = img.copy()
    for spot in spots:
        source_offset = spot.get("source_offset")
        out = heal_single_spot(out, spot, source_offset=source_offset, mode=mode)
    return out


# Alias for consistency
heal_dust_spots = apply_dust_healing


def visualize_dust_spots(
    img: np.ndarray,
    threshold: float = 50.0,
) -> np.ndarray:
    """Generate a high-pass diagnostic view where sensor dust spots and specks stand out as white dots on black.

    Matches Lightcraft's and Lightroom's 'Visualize Spots (A)' inspection mode.
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    long_edge = float(max(w, h))

    luma = _to_luma(img)

    # Blur radius of spot inspection view (fraction of long edge)
    spot_sigma = max(2.0, 0.008 * long_edge)
    ksize = int(math.ceil(spot_sigma * 4.0)) | 1
    blurred = cv2.GaussianBlur(luma, (ksize, ksize), spot_sigma)

    # Threshold mapping: 100 shows faintest specks, 0 only strong ones
    k = 1.0 - np.clip(threshold, 0.0, 100.0) / 100.0
    thr = float(0.005 + 0.15 * (k * k))

    high_pass = np.abs(luma - blurred)
    binary = (high_pass > thr).astype(np.float32)

    # Return 3-channel RGB image in float32 [0..1]
    return np.repeat(binary[..., np.newaxis], 3, axis=2)
