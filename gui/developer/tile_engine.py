# gui/developer/tile_engine.py
"""
Fixed-grid tiled rendering engine (512x512) with thread-safe LRU TileCache,
progressive center-first async evaluation, and background prefetch ring
for AstroLoom Developer Zoom 1:1.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal

from core.orton import apply_astronomical_orton
from core.stretch import (
    apply_clarity,
    apply_dehaze,
    apply_ghs_stretch,
    descomponer_ondiculas,
    manual_stretch,
    realce_multiescala_ondiculas,
)
from gui.developer.developer_state import DeveloperState
from gui.developer.pipeline import apply_developer_pipeline

logger = logging.getLogger("AstroLoom.TileEngine")

DEFAULT_TILE_SIZE = 512


class TileCache:
    """Thread-safe LRU cache for rendered 1:1 uint8 tiles and per-tile milestones with strict RAM budgets."""

    def __init__(
        self,
        max_bytes: int = 256 * 1024 * 1024,
        milestone_max_bytes: int = 192 * 1024 * 1024,
    ):
        self._lock = threading.Lock()
        self._cache: OrderedDict[Tuple[int, int, Any], np.ndarray] = OrderedDict()
        self._current_bytes: int = 0
        self._max_bytes: int = max(4 * 1024 * 1024, int(max_bytes))
        self.hits: int = 0
        self.misses: int = 0

        self._milestones: OrderedDict[Tuple[str, int, int, Any], np.ndarray] = OrderedDict()
        self._milestone_bytes: int = 0
        self._milestone_max_bytes: int = max(4 * 1024 * 1024, int(milestone_max_bytes))
        self.milestone_hits: int = 0
        self.milestone_misses: int = 0

    def get(self, tx: int, ty: int, state_key: Any) -> Optional[np.ndarray]:
        k = (tx, ty, state_key)
        with self._lock:
            val = self._cache.get(k)
            if val is not None:
                self._cache.move_to_end(k)
                self.hits += 1
                return val
            self.misses += 1
            return None

    def contains(self, tx: int, ty: int, state_key: Any) -> bool:
        k = (tx, ty, state_key)
        with self._lock:
            return k in self._cache

    def put(self, tx: int, ty: int, state_key: Any, tile_u8: np.ndarray) -> None:
        if tile_u8 is None:
            return
        k = (tx, ty, state_key)
        nbytes = int(tile_u8.nbytes)
        if nbytes > self._max_bytes:
            return
        with self._lock:
            old = self._cache.pop(k, None)
            if old is not None:
                self._current_bytes -= int(old.nbytes)
            while self._current_bytes + nbytes > self._max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._current_bytes -= int(evicted.nbytes)
            self._cache[k] = tile_u8
            self._current_bytes += nbytes

    def get_milestone(self, stage: str, tx: int, ty: int, key: Any) -> Optional[np.ndarray]:
        if key is None:
            return None
        k = (stage, tx, ty, key)
        with self._lock:
            val = self._milestones.get(k)
            if val is not None:
                self._milestones.move_to_end(k)
                self.milestone_hits += 1
                return val
            self.milestone_misses += 1
            return None

    def put_milestone(self, stage: str, tx: int, ty: int, key: Any, arr: Optional[np.ndarray]) -> None:
        if arr is None or key is None:
            return
        k = (stage, tx, ty, key)
        nbytes = int(arr.nbytes)
        if nbytes > self._milestone_max_bytes:
            return
        with self._lock:
            old = self._milestones.pop(k, None)
            if old is not None:
                self._milestone_bytes -= int(old.nbytes)
            while self._milestone_bytes + nbytes > self._milestone_max_bytes and self._milestones:
                _, evicted = self._milestones.popitem(last=False)
                self._milestone_bytes -= int(evicted.nbytes)
            self._milestones[k] = arr
            self._milestone_bytes += nbytes

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
            self._milestones.clear()
            self._milestone_bytes = 0

    def reset_stats(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.milestone_hits = 0
            self.milestone_misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "bytes": self._current_bytes,
                "max_bytes": self._max_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "milestone_entries": len(self._milestones),
                "milestone_bytes": self._milestone_bytes,
                "milestone_max_bytes": self._milestone_max_bytes,
                "milestone_hits": self.milestone_hits,
                "milestone_misses": self.milestone_misses,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


@dataclass
class TileContext:
    """Immutable render snapshot shared safely across background tile workers."""

    full_shape: Tuple[int, int]
    image_32bit: np.ndarray
    state: DeveloperState
    state_key: Any
    image_starless: Optional[np.ndarray] = None
    image_stars: Optional[np.ndarray] = None
    current_mask: Optional[np.ndarray] = None
    view_layer_mode: int = 0
    has_image_layers: bool = False
    mask_layers: Optional[List[Any]] = None
    curve_luts: Dict[str, np.ndarray] = field(default_factory=dict)
    psf_kernel: Optional[np.ndarray] = None
    cached_starless_ai: Optional[np.ndarray] = None
    cached_ground_ai: Optional[np.ndarray] = None
    starless_ai_enabled: bool = False
    starless_params: Dict[str, Any] = field(default_factory=dict)
    global_stats: Dict[str, float] = field(default_factory=dict)
    pad_margin: int = 64
    fallback_preview_u8: Optional[np.ndarray] = None
    spatial_key: Any = None
    stars_key: Any = None
    linear_key: Any = None
    is_scrubbing: bool = False


def compute_tile_pad_margin(state: DeveloperState, starless_params: Optional[Dict[str, Any]] = None) -> int:
    """Compute spatial padding margin (64..256 px) needed for seamless tile borders."""
    active_radius = 0.0
    orton_s = float(getattr(state, "orton_strength", 0))
    if starless_params:
        orton_s = max(orton_s, float(starless_params.get("orton_strength", 0.0)) * 100.0)
    if orton_s > 1e-2:
        active_radius = max(active_radius, float(getattr(state, "orton_radius", 40)))
    if float(getattr(state, "star_diff_strength", 0)) > 1e-2:
        active_radius = max(active_radius, float(getattr(state, "star_diff_radius", 20)))

    # 3-sigma halo clamped to [64, 256] so 512x512 tiles stay L3-cache friendly
    # while eliminating border seams on large-radius Orton/StarDiff blurs.
    if active_radius > 0.0:
        return int(min(256, max(64, round(2.5 * active_radius))))
    return 64


def compute_global_pipeline_stats(
    preview_img: Optional[np.ndarray],
    preview_mask: Optional[np.ndarray],
    preview_stretched: Optional[np.ndarray],
    full_shape: Tuple[int, int],
    state: DeveloperState,
    starless_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute global scalar statistics once from the small Fit preview (~1-2 ms)
    so every 512x512 tile evaluates with identical pivots, thresholds, and radii.
    """
    h_f, w_f = full_shape
    # Cap reference min_dim at 1600 to match Preview visual scale and keep blur kernels bounded.
    stats: Dict[str, float] = {"ref_min_dim": float(min(1600, min(h_f, w_f)))}

    sp = starless_params or {}
    dehaze_v = abs(float(sp.get("dehaze_starless_val", state.dehaze_val / 100.0)))
    if dehaze_v > 1e-4 and preview_img is not None:
        dark_ch = np.min(preview_img[::2, ::2], axis=2)
        h_s, w_s = dark_ch.shape[:2]
        r = int(max(7, (min(h_s, w_s) // 25) | 1))
        if r % 2 == 0:
            r += 1
        haze_map = cv2.GaussianBlur(dark_ch, (r, r), sigmaX=r / 2.5)
        stats["airlight_p95"] = float(np.percentile(haze_map, 95))

    ond_bg = max(float(sp.get("multiescala_fondo", 0.0)), float(getattr(state, "ond_bg_val", 0)) / 100.0)
    if ond_bg > 1e-4 and preview_img is not None:
        src = preview_img[::2, ::2]
        lum = (0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]).astype(np.float32)
        _, residual = descomponer_ondiculas(np.clip(lum, 0.0, 1.0), niveles=5)
        stats["ond_pedestal"] = float(np.percentile(residual[::2, ::2], 5))

    ref_str = preview_stretched if preview_stretched is not None else preview_img
    if ref_str is not None:
        if abs(state.contrast_starless_val) > 1e-4:
            sample_s = ref_str[::4, ::4]
            if preview_mask is not None:
                m2 = preview_mask[..., 0] if preview_mask.ndim == 3 else preview_mask
                if m2.shape[:2] != ref_str.shape[:2]:
                    m2 = cv2.resize(m2, (ref_str.shape[1], ref_str.shape[0]), interpolation=cv2.INTER_LINEAR)
                sample_m = m2[::4, ::4]
                stats["sky_median"] = (
                    float(np.median(sample_s[sample_m > 0.5]))
                    if np.any(sample_m > 0.5)
                    else float(np.median(sample_s))
                )
            else:
                stats["sky_median"] = float(np.median(sample_s))

        if state.deconv_iterations > 0 and state.deconv_damping > 0:
            lum = (0.2126 * ref_str[..., 0] + 0.7152 * ref_str[..., 1] + 0.0722 * ref_str[..., 2]).astype(np.float32)
            lum = np.clip(lum, 0.0, 1.0)
            med_smooth = cv2.medianBlur((lum * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
            hf = np.abs(lum - med_smooth)[::4, ::4]
            stats["sigma_noise"] = float(1.4826 * np.median(hf))

        if getattr(state, "sharp_amount", 0) > 0 and getattr(state, "sharp_masking", 0) > 0:
            lum = (0.299 * ref_str[..., 0] + 0.587 * ref_str[..., 1] + 0.114 * ref_str[..., 2]).astype(np.float32)
            gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), sigmaX=0.8, sigmaY=0.8)
            stats["edge_p99"] = float(np.percentile(mag[::4, ::4], 99.5))

    return stats


def _crop_aligned_slice(
    src: Optional[np.ndarray],
    crop_rect: Tuple[int, int, int, int],
    full_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Crop src corresponding to crop_rect in full_shape, resizing only the cropped ROI if resolutions differ."""
    if src is None:
        return None
    x1, y1, x2, y2 = crop_rect
    h_f, w_f = full_shape
    cw, ch = x2 - x1, y2 - y1
    h_s, w_s = src.shape[:2]
    if (h_s, w_s) == (h_f, w_f):
        return np.ascontiguousarray(src[y1:y2, x1:x2], dtype=np.float32)

    sx = w_s / float(w_f)
    sy = h_s / float(h_f)
    sx1 = max(0, min(w_s - 1, int(math.floor(x1 * sx))))
    sy1 = max(0, min(h_s - 1, int(math.floor(y1 * sy))))
    sx2 = max(sx1 + 1, min(w_s, int(math.ceil(x2 * sx))))
    sy2 = max(sy1 + 1, min(h_s, int(math.ceil(y2 * sy))))
    sub = src[sy1:sy2, sx1:sx2]
    if sub.dtype == np.uint8:
        sub = sub.astype(np.float32) * (1.0 / 255.0)
    elif sub.dtype != np.float32:
        sub = sub.astype(np.float32)
    if sub.shape[:2] != (ch, cw):
        sub = cv2.resize(sub, (cw, ch), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(sub, dtype=np.float32)


def _process_starless_tile(
    starless: np.ndarray,
    mask_3d: Optional[np.ndarray],
    crop_rect: Tuple[int, int, int, int],
    ctx: TileContext,
) -> np.ndarray:
    """Thread-safe starless linear preprocessing on a single padded tile."""
    sp = ctx.starless_params
    denoise_s = float(sp.get("denoise_strength", 0.0))
    clarity_v = float(sp.get("clarity_starless_val", 0.0))
    tex_clarity_v = float(sp.get("texture_clarity_starless_val", 0.0))
    dehaze_v = float(sp.get("dehaze_starless_val", 0.0))
    ond_est = float(sp.get("multiescala_estructura", 0.0))
    ond_bg = float(sp.get("multiescala_fondo", 0.0))
    orton_s = float(sp.get("orton_strength", 0.0))

    if (
        denoise_s <= 1e-4
        and abs(clarity_v) <= 1e-4
        and abs(tex_clarity_v) <= 1e-4
        and abs(dehaze_v) <= 1e-4
        and abs(ond_est) <= 1e-4
        and ond_bg <= 1e-4
        and orton_s <= 1e-4
    ):
        return starless

    proc = starless.copy()
    m2 = mask_3d[..., 0] if mask_3d is not None else None
    gstats = ctx.global_stats
    ref_dim = int(gstats["ref_min_dim"]) if "ref_min_dim" in gstats else None

    # 1. Denoise (NAFNet AI cache or classical astro denoise)
    if denoise_s > 1e-4:
        if ctx.starless_ai_enabled and ctx.cached_starless_ai is not None:
            ai_aligned = _crop_aligned_slice(ctx.cached_starless_ai, crop_rect, ctx.full_shape)
            k = float(np.clip(denoise_s, 0.0, 1.0))
            denoised = cv2.addWeighted(ai_aligned, k, proc, 1.0 - k, 0.0)
        else:
            from core.denoise import apply_astro_denoise

            dn_color_ratio = float(sp.get("denoise_color_strength", 0.5)) / 0.5
            dn_color = min(1.0, denoise_s * dn_color_ratio)
            denoised = apply_astro_denoise(
                proc,
                strength=denoise_s,
                color_strength=dn_color,
                masking=float(sp.get("denoise_masking", 0.5)),
                sky_mask=m2,
                is_draft=False,
                is_full_res=True,
            )
        proc = (denoised * mask_3d) + (proc * (1.0 - mask_3d)) if mask_3d is not None else denoised

    # 2. Clarity, Texture Clarity, Dehaze (with global ref_dim & airlight for zero tile seams)
    if abs(clarity_v) > 1e-4:
        clarified = apply_clarity(proc, strength=clarity_v, ref_min_dim=ref_dim)
        proc = (clarified * mask_3d) + (proc * (1.0 - mask_3d)) if mask_3d is not None else clarified

    if abs(tex_clarity_v) > 1e-4:
        from core.sharpening import apply_texture_clarity

        t_clarified = apply_texture_clarity(proc, strength=tex_clarity_v)
        proc = (t_clarified * mask_3d) + (proc * (1.0 - mask_3d)) if mask_3d is not None else t_clarified

    if abs(dehaze_v) > 1e-4:
        dehazed = apply_dehaze(
            proc,
            strength=dehaze_v,
            ref_min_dim=ref_dim,
            global_airlight_p95=gstats.get("airlight_p95"),
        )
        proc = (dehazed * mask_3d) + (proc * (1.0 - mask_3d)) if mask_3d is not None else dehazed

    # 3. Multiscale wavelets
    if abs(ond_est) > 1e-4 or ond_bg > 1e-4:
        proc = realce_multiescala_ondiculas(
            proc,
            fuerza_estructura=ond_est,
            reduccion_fondo=ond_bg,
            sky_mask=m2,
            global_pedestal=gstats.get("ond_pedestal"),
        )

    # 4. Orton on starless
    if orton_s > 1e-4:
        proc = apply_astronomical_orton(
            proc,
            strength=orton_s,
            radius=float(sp.get("orton_radius", 40.0)),
            threshold=float(sp.get("orton_threshold", 0.05)),
            exponent=1.0,
            tint_type=str(sp.get("orton_tint_type", "warm_gold")),
            custom_tint=tuple(sp.get("orton_custom_tint", (1.0, 0.85, 0.55))),
            tint_strength=float(sp.get("orton_tint_strength", 0.5)),
            blend_mode=str(sp.get("orton_mode", "screen")),
            sky_mask=m2,
            is_starless=True,
            ref_min_dim=ref_dim,
        )

    return proc


def render_single_tile(
    ctx: TileContext,
    tx: int,
    ty: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    abort_flag: Optional[Callable[[], bool]] = None,
    milestone_cache: Optional[TileCache] = None,
) -> Optional[np.ndarray]:
    """
    Render a single (tx, ty) tile at native 1:1 resolution with spatial padding,
    reusing per-tile milestones (spatial, stars) when available, then crop back
    to the exact inner tile bounds and convert to uint8 RGB.
    """
    if abort_flag and abort_flag():
        return None

    h_f, w_f = ctx.full_shape
    ix1 = tx * tile_size
    iy1 = ty * tile_size
    ix2 = min(w_f, ix1 + tile_size)
    iy2 = min(h_f, iy1 + tile_size)
    if ix1 >= ix2 or iy1 >= iy2:
        return None

    pad = ctx.pad_margin
    px1 = max(0, ix1 - pad)
    py1 = max(0, iy1 - pad)
    px2 = min(w_f, ix2 + pad)
    py2 = min(h_f, iy2 + pad)
    padded_rect = (px1, py1, px2, py2)

    pre_spatial: Optional[np.ndarray] = None
    pre_stars: Optional[np.ndarray] = None
    if milestone_cache is not None and ctx.view_layer_mode != 2:
        if ctx.spatial_key is not None:
            pre_spatial = milestone_cache.get_milestone("spatial", tx, ty, ctx.spatial_key)
        if ctx.stars_key is not None:
            pre_stars = milestone_cache.get_milestone("stars", tx, ty, ctx.stars_key)

    mask_3d: Optional[np.ndarray] = None
    if ctx.current_mask is not None:
        m_src = ctx.current_mask[..., 0] if ctx.current_mask.ndim == 3 else ctx.current_mask
        m_sub = _crop_aligned_slice(m_src, padded_rect, ctx.full_shape)
        if m_sub is not None:
            if float(m_sub.max()) > 1.05:
                m_sub = m_sub * (1.0 / 255.0)
            mask_3d = np.ascontiguousarray(np.repeat(m_sub[..., np.newaxis], 3, axis=2), dtype=np.float32)

    if abort_flag and abort_flag():
        return None

    starless = _crop_aligned_slice(ctx.image_starless, padded_rect, ctx.full_shape)
    stars = None if pre_stars is not None else _crop_aligned_slice(ctx.image_stars, padded_rect, ctx.full_shape)

    # Skip heavy starless preproc completely when spatial milestone hits
    if pre_spatial is not None:
        target_img = pre_spatial
    else:
        base = _crop_aligned_slice(ctx.image_32bit, padded_rect, ctx.full_shape)
        if starless is not None and (stars is not None or pre_stars is not None):
            proc_starless = _process_starless_tile(starless, mask_3d, padded_rect, ctx)
            if ctx.has_image_layers:
                target_img = (stars if stars is not None else _crop_aligned_slice(ctx.image_stars, padded_rect, ctx.full_shape)) if ctx.view_layer_mode == 2 else proc_starless
            else:
                if ctx.view_layer_mode == 0:
                    st_mix = stars if stars is not None else _crop_aligned_slice(ctx.image_stars, padded_rect, ctx.full_shape)
                    star_int = float(ctx.state.stars_val) / 100.0
                    target_img = np.clip(proc_starless + (st_mix * star_int), 0.0, 1.0)
                elif ctx.view_layer_mode == 1:
                    target_img = proc_starless
                else:
                    target_img = stars if stars is not None else _crop_aligned_slice(ctx.image_stars, padded_rect, ctx.full_shape)
        else:
            target_img = base

    if abort_flag and abort_flag():
        return None

    ox1 = ix1 - px1
    oy1 = iy1 - py1
    ox2 = ox1 + (ix2 - ix1)
    oy2 = oy1 + (iy2 - iy1)

    if ctx.view_layer_mode == 2:
        if ctx.state.stretch_mode == "ghs":
            out_padded = apply_ghs_stretch(
                target_img,
                sp=ctx.state.ghs_sp_val,
                d=ctx.state.ghs_d_val,
                b=ctx.state.ghs_b_val,
                hp=ctx.state.ghs_hp_val,
                lp=ctx.state.ghs_lp_val,
                bp=ctx.state.ghs_bp_val,
                color_mode=ctx.state.ghs_color_mode,
            )
        else:
            out_padded = manual_stretch(target_img, black_point=ctx.state.bp_val, midtone=ctx.state.mtf_val)
    else:
        want_ms = milestone_cache is not None and (
            (pre_spatial is None and ctx.spatial_key is not None)
            or (pre_stars is None and ctx.stars_key is not None)
            or (ctx.linear_key is not None and pre_spatial is None)
        )
        res = apply_developer_pipeline(
            target_img=target_img,
            state=ctx.state,
            precomputed_mask=mask_3d,
            image_stars=stars,
            image_starless=starless,
            mask_layers=ctx.mask_layers,
            curve_luts=ctx.curve_luts,
            psf_kernel=ctx.psf_kernel,
            is_preview=True,
            is_scrubbing=ctx.is_scrubbing,
            crop_rect=padded_rect,
            full_shape=ctx.full_shape,
            cached_ground_ai=ctx.cached_ground_ai,
            precomputed_spatial_base=pre_spatial,
            precomputed_stretched_stars=pre_stars,
            return_milestones=want_ms,
            abort_flag=abort_flag,
            global_stats=ctx.global_stats,
        )
        if res is None:
            return None
        if want_ms:
            out_padded, _, out_lin, _, _, out_spt, out_stars = res
            if pre_spatial is None and out_spt is not None and ctx.spatial_key is not None:
                milestone_cache.put_milestone("spatial", tx, ty, ctx.spatial_key, out_spt)
            if pre_stars is None and out_stars is not None and ctx.stars_key is not None:
                milestone_cache.put_milestone("stars", tx, ty, ctx.stars_key, out_stars)
            if out_lin is not None and ctx.linear_key is not None:
                inner_lin = out_lin[oy1:oy2, ox1:ox2]
                inner_lin_u16 = np.clip(inner_lin * 65535.0, 0.0, 65535.0).astype(np.uint16)
                milestone_cache.put_milestone(
                    "linear_u16", tx, ty, ctx.linear_key, np.ascontiguousarray(inner_lin_u16)
                )
        else:
            out_padded = res

    if out_padded is None or (abort_flag and abort_flag()):
        return None

    # Crop back to exact inner tile bounds (discarding spatial halo)
    inner = out_padded[oy1:oy2, ox1:ox2]
    tile_u8 = cv2.convertScaleAbs(np.clip(inner, 0.0, 1.0), alpha=255.0)
    return np.ascontiguousarray(tile_u8, dtype=np.uint8)


class TileGridEngine:
    """Headless fixed-grid tile coordinator (usable in both GUI and benchmarks/tests)."""

    def __init__(self, tile_size: int = DEFAULT_TILE_SIZE, cache_max_bytes: int = 256 * 1024 * 1024):
        self.tile_size = int(tile_size)
        self.cache = TileCache(max_bytes=cache_max_bytes)

    def visible_tiles(
        self,
        cx: float,
        cy: float,
        vw: int,
        vh: int,
        full_shape: Tuple[int, int],
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
        """
        Return (sorted_tile_coords, mosaic_rect) for a viewport centered at (cx, cy).
        Tiles are sorted by distance from (cx, cy) so center tiles sharpen first.
        """
        h_f, w_f = full_shape
        ts = self.tile_size
        vw = max(64, int(vw))
        vh = max(64, int(vh))

        vx1 = max(0, int(math.floor(cx - vw / 2.0)))
        vy1 = max(0, int(math.floor(cy - vh / 2.0)))
        vx2 = min(w_f, int(math.ceil(cx + vw / 2.0)))
        vy2 = min(h_f, int(math.ceil(cy + vh / 2.0)))
        if vx2 <= vx1:
            vx1, vx2 = 0, min(w_f, ts)
        if vy2 <= vy1:
            vy1, vy2 = 0, min(h_f, ts)

        max_tx = max(0, (w_f - 1) // ts)
        max_ty = max(0, (h_f - 1) // ts)
        tx1 = max(0, min(max_tx, vx1 // ts))
        ty1 = max(0, min(max_ty, vy1 // ts))
        tx2 = max(tx1, min(max_tx, (vx2 - 1) // ts))
        ty2 = max(ty1, min(max_ty, (vy2 - 1) // ts))

        coords: List[Tuple[int, int]] = []
        for ty in range(ty1, ty2 + 1):
            for tx in range(tx1, tx2 + 1):
                coords.append((tx, ty))

        coords.sort(key=lambda t: ((t[0] + 0.5) * ts - cx) ** 2 + ((t[1] + 0.5) * ts - cy) ** 2)

        mx1 = tx1 * ts
        my1 = ty1 * ts
        mx2 = min(w_f, (tx2 + 1) * ts)
        my2 = min(h_f, (ty2 + 1) * ts)
        return coords, (mx1, my1, mx2, my2)

    def prefetch_ring(
        self,
        visible_coords: List[Tuple[int, int]],
        cx: float,
        cy: float,
        full_shape: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Return the 1-tile ring surrounding the visible bounding box, sorted by proximity."""
        if not visible_coords:
            return []
        h_f, w_f = full_shape
        ts = self.tile_size
        max_tx = max(0, (w_f - 1) // ts)
        max_ty = max(0, (h_f - 1) // ts)
        vis_set: Set[Tuple[int, int]] = set(visible_coords)

        tx_min = max(0, min(t[0] for t in visible_coords) - 1)
        tx_max = min(max_tx, max(t[0] for t in visible_coords) + 1)
        ty_min = max(0, min(t[1] for t in visible_coords) - 1)
        ty_max = min(max_ty, max(t[1] for t in visible_coords) + 1)

        ring: List[Tuple[int, int]] = []
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                if (tx, ty) not in vis_set:
                    ring.append((tx, ty))

        ring.sort(key=lambda t: ((t[0] + 0.5) * ts - cx) ** 2 + ((t[1] + 0.5) * ts - cy) ** 2)
        return ring

    def assemble_mosaic(
        self,
        tile_coords: List[Tuple[int, int]],
        mosaic_rect: Tuple[int, int, int, int],
        state_key: Any,
        full_shape: Tuple[int, int],
        fallback_preview_u8: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Assemble a uint8 RGB mosaic from cached tiles. Uncached slots are filled from
        fallback_preview_u8 (if provided) so the canvas never flashes black.
        """
        mx1, my1, mx2, my2 = mosaic_rect
        mw, mh = mx2 - mx1, my2 - my1
        ts = self.tile_size

        cached_tiles: List[Tuple[int, int, np.ndarray]] = []
        missing: List[Tuple[int, int]] = []
        for tx, ty in tile_coords:
            t = self.cache.get(tx, ty, state_key)
            if t is not None:
                cached_tiles.append((tx, ty, t))
            else:
                missing.append((tx, ty))

        if missing and fallback_preview_u8 is not None and fallback_preview_u8.size > 0:
            h_f, w_f = full_shape
            ph, pw = fallback_preview_u8.shape[:2]
            sx = pw / float(w_f)
            sy = ph / float(h_f)
            fx1 = max(0, min(pw - 1, int(math.floor(mx1 * sx))))
            fy1 = max(0, min(ph - 1, int(math.floor(my1 * sy))))
            fx2 = max(fx1 + 1, min(pw, int(math.ceil(mx2 * sx))))
            fy2 = max(fy1 + 1, min(ph, int(math.ceil(my2 * sy))))
            sub_fb = fallback_preview_u8[fy1:fy2, fx1:fx2]
            mosaic = cv2.resize(sub_fb, (mw, mh), interpolation=cv2.INTER_LINEAR)
            mosaic = np.ascontiguousarray(mosaic, dtype=np.uint8)
        else:
            mosaic = np.zeros((mh, mw, 3), dtype=np.uint8)

        for tx, ty, tile_u8 in cached_tiles:
            ox = tx * ts - mx1
            oy = ty * ts - my1
            th, tw = tile_u8.shape[:2]
            mosaic[oy : oy + th, ox : ox + tw] = tile_u8

        return mosaic, missing

    def assemble_linear_u16_mosaic(
        self,
        tile_coords: List[Tuple[int, int]],
        mosaic_rect: Tuple[int, int, int, int],
        linear_key: Any,
    ) -> Optional[np.ndarray]:
        """Assemble a uint16 linear RGB mosaic from cached per-tile linear milestones for 1:1 LUT scrubbing."""
        if not tile_coords or linear_key is None:
            return None
        mx1, my1, mx2, my2 = mosaic_rect
        mw, mh = mx2 - mx1, my2 - my1
        ts = self.tile_size

        tiles: List[Tuple[int, int, np.ndarray]] = []
        for tx, ty in tile_coords:
            t_u16 = self.cache.get_milestone("linear_u16", tx, ty, linear_key)
            if t_u16 is None:
                return None
            tiles.append((tx, ty, t_u16))

        mosaic = np.empty((mh, mw, 3), dtype=np.uint16)
        for tx, ty, t_u16 in tiles:
            ox = tx * ts - mx1
            oy = ty * ts - my1
            th, tw = t_u16.shape[:2]
            mosaic[oy : oy + th, ox : ox + tw] = t_u16
        return mosaic

    def render_viewport_sync(
        self,
        ctx: TileContext,
        cx: float,
        cy: float,
        vw: int,
        vh: int,
        max_workers: Optional[int] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], int, int]:
        """
        Render all missing tiles in a viewport using a thread pool and return
        (mosaic_u8, mosaic_rect, cache_hits, newly_rendered).
        """
        coords, mosaic_rect = self.visible_tiles(cx, cy, vw, vh, ctx.full_shape)
        mosaic, missing = self.assemble_mosaic(
            coords, mosaic_rect, ctx.state_key, ctx.full_shape, ctx.fallback_preview_u8
        )
        hits = len(coords) - len(missing)
        if not missing:
            return mosaic, mosaic_rect, hits, 0

        workers = max_workers or max(2, min(4, (os.cpu_count() or 4) // 2))
        mx1, my1, _, _ = mosaic_rect
        ts = self.tile_size

        if len(missing) == 1 or workers <= 1:
            for tx, ty in missing:
                tile_u8 = render_single_tile(ctx, tx, ty, tile_size=ts, milestone_cache=self.cache)
                if tile_u8 is not None:
                    self.cache.put(tx, ty, ctx.state_key, tile_u8)
                    ox = tx * ts - mx1
                    oy = ty * ts - my1
                    th, tw = tile_u8.shape[:2]
                    mosaic[oy : oy + th, ox : ox + tw] = tile_u8
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
                futures = [
                    (tx, ty, pool.submit(render_single_tile, ctx, tx, ty, ts, None, self.cache))
                    for tx, ty in missing
                ]
                for tx, ty, fut in futures:
                    tile_u8 = fut.result()
                    if tile_u8 is not None:
                        self.cache.put(tx, ty, ctx.state_key, tile_u8)
                        ox = tx * ts - mx1
                        oy = ty * ts - my1
                        th, tw = tile_u8.shape[:2]
                        mosaic[oy : oy + th, ox : ox + tw] = tile_u8

        return mosaic, mosaic_rect, hits, len(missing)


class _TileWorkerSignals(QObject):
    tile_done = Signal(int, int, int, object, bool)  # (gen_id, tx, ty, tile_u8, is_prefetch)


class _TileWorker(QRunnable):
    def __init__(
        self,
        gen_id: int,
        ctx: TileContext,
        tx: int,
        ty: int,
        tile_size: int,
        is_prefetch: bool,
        abort_check: Callable[[], bool],
        signals: _TileWorkerSignals,
        milestone_cache: Optional[TileCache] = None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.gen_id = gen_id
        self.ctx = ctx
        self.tx = tx
        self.ty = ty
        self.tile_size = tile_size
        self.is_prefetch = is_prefetch
        self.abort_check = abort_check
        self.signals = signals
        self.milestone_cache = milestone_cache

    def run(self) -> None:
        if self.abort_check():
            return
        try:
            tile_u8 = render_single_tile(
                self.ctx,
                self.tx,
                self.ty,
                tile_size=self.tile_size,
                abort_flag=self.abort_check,
                milestone_cache=self.milestone_cache,
            )
            if tile_u8 is not None and not self.abort_check():
                self.signals.tile_done.emit(self.gen_id, self.tx, self.ty, tile_u8, self.is_prefetch)
        except Exception as exc:
            if not self.abort_check():
                logger.error(f"Error rendering tile ({self.tx}, {self.ty}): {exc}", exc_info=True)


class AsyncTileManager(QObject):
    """
    Qt async orchestrator for Zoom 1:1 fixed-grid tiled rendering + background prefetch ring.
    Never blocks the Qt UI thread.
    """

    # (gen_id, mosaic_u8, mosaic_rect, sensor_center, is_complete, is_new_rect)
    mosaic_ready = Signal(int, object, tuple, tuple, bool, bool)

    def __init__(
        self,
        parent: Optional[QObject] = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        cache_max_bytes: int = 256 * 1024 * 1024,
        max_threads: Optional[int] = None,
    ):
        super().__init__(parent)
        self.engine = TileGridEngine(tile_size=tile_size, cache_max_bytes=cache_max_bytes)
        self.thread_pool = QThreadPool(self)
        workers = max_threads or max(2, min(4, (os.cpu_count() or 4) // 2))
        self.thread_pool.setMaxThreadCount(workers)

        # Single parented QObject signal emitter so QThreadPool autoDelete never destroys
        # the sender before Qt delivers queued cross-thread tile_done signals.
        self._worker_signals = _TileWorkerSignals(self)
        self._worker_signals.tile_done.connect(self._on_tile_done, Qt.QueuedConnection)

        self._gen_id: int = 0
        self._active_ctx: Optional[TileContext] = None
        self._active_mosaic: Optional[np.ndarray] = None
        self._active_rect: Optional[Tuple[int, int, int, int]] = None
        self._active_center: Tuple[float, float] = (0.0, 0.0)
        self._active_visible: List[Tuple[int, int]] = []
        self._pending_visible: Set[Tuple[int, int]] = set()
        self._in_flight_prefetch: Set[Tuple[int, int]] = set()

        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.setInterval(150)
        self._prefetch_timer.timeout.connect(self._start_prefetch_ring)

    @property
    def current_gen_id(self) -> int:
        return self._gen_id

    @property
    def cache(self) -> TileCache:
        return self.engine.cache

    def cancel_all(self) -> None:
        self._gen_id += 1
        self._prefetch_timer.stop()
        self._pending_visible.clear()
        self._in_flight_prefetch.clear()

    def clear_cache(self) -> None:
        self.cancel_all()
        self.engine.cache.clear()

    def assemble_active_linear_u16_mosaic(self) -> Optional[Tuple[Tuple[int, int, int, int], Any, np.ndarray]]:
        """Assemble the 16-bit linear mosaic for the currently active 1:1 viewport if all tiles are cached."""
        if (
            self._active_ctx is None
            or self._active_rect is None
            or not self._active_visible
            or self._active_ctx.linear_key is None
        ):
            return None
        mosaic_u16 = self.engine.assemble_linear_u16_mosaic(
            self._active_visible, self._active_rect, self._active_ctx.linear_key
        )
        if mosaic_u16 is None:
            return None
        return self._active_rect, self._active_ctx.linear_key, mosaic_u16

    def request_viewport(
        self,
        ctx: TileContext,
        cx: float,
        cy: float,
        vw: int,
        vh: int,
    ) -> int:
        """
        Request a 1:1 viewport centered at (cx, cy).
        Emits immediately with cached tiles (+ fallback for missing ones), then streams
        missing tiles asynchronously center-first and schedules background prefetch.
        """
        self._gen_id += 1
        gen = self._gen_id
        self._prefetch_timer.stop()
        self._pending_visible.clear()
        self._in_flight_prefetch.clear()

        coords, mosaic_rect = self.engine.visible_tiles(cx, cy, vw, vh, ctx.full_shape)
        mosaic, missing = self.engine.assemble_mosaic(
            coords, mosaic_rect, ctx.state_key, ctx.full_shape, ctx.fallback_preview_u8
        )

        self._active_ctx = ctx
        self._active_mosaic = mosaic
        self._active_rect = mosaic_rect
        self._active_center = (cx, cy)
        self._active_visible = coords
        self._pending_visible = set(missing)

        is_complete = len(missing) == 0
        self.mosaic_ready.emit(gen, mosaic, mosaic_rect, (cx, cy), is_complete, True)

        if is_complete:
            self._prefetch_timer.start()
            return gen

        ts = self.engine.tile_size
        for tx, ty in missing:
            worker = _TileWorker(
                gen_id=gen,
                ctx=ctx,
                tx=tx,
                ty=ty,
                tile_size=ts,
                is_prefetch=False,
                abort_check=lambda g=gen: self._gen_id != g,
                signals=self._worker_signals,
                milestone_cache=self.engine.cache,
            )
            self.thread_pool.start(worker)

        return gen

    def _on_tile_done(self, gen_id: int, tx: int, ty: int, tile_u8: np.ndarray, is_prefetch: bool) -> None:
        if self._active_ctx is None:
            return
        # Store even if generation changed only when state_key still matches active state
        if gen_id != self._gen_id:
            return

        self.engine.cache.put(tx, ty, self._active_ctx.state_key, tile_u8)

        if is_prefetch:
            self._in_flight_prefetch.discard((tx, ty))
            return

        if self._active_mosaic is None or self._active_rect is None:
            return

        mx1, my1, mx2, my2 = self._active_rect
        ts = self.engine.tile_size
        ox = tx * ts - mx1
        oy = ty * ts - my1
        th, tw = tile_u8.shape[:2]
        if 0 <= oy and oy + th <= (my2 - my1) and 0 <= ox and ox + tw <= (mx2 - mx1):
            self._active_mosaic[oy : oy + th, ox : ox + tw] = tile_u8

        self._pending_visible.discard((tx, ty))
        is_complete = len(self._pending_visible) == 0
        self.mosaic_ready.emit(
            gen_id,
            self._active_mosaic,
            self._active_rect,
            self._active_center,
            is_complete,
            False,
        )
        if is_complete:
            self._prefetch_timer.start()

    def _start_prefetch_ring(self) -> None:
        if self._active_ctx is None or not self._active_visible or self._pending_visible:
            return
        gen = self._gen_id
        ctx = self._active_ctx
        cx, cy = self._active_center
        ring = self.engine.prefetch_ring(self._active_visible, cx, cy, ctx.full_shape)
        ts = self.engine.tile_size

        # Cap prefetch burst to 8 adjacent tiles so we stay responsive and RAM-friendly
        scheduled = 0
        for tx, ty in ring:
            if scheduled >= 8:
                break
            if self.engine.cache.contains(tx, ty, ctx.state_key) or (tx, ty) in self._in_flight_prefetch:
                continue
            self._in_flight_prefetch.add((tx, ty))
            worker = _TileWorker(
                gen_id=gen,
                ctx=ctx,
                tx=tx,
                ty=ty,
                tile_size=ts,
                is_prefetch=True,
                abort_check=lambda g=gen: self._gen_id != g,
                signals=self._worker_signals,
                milestone_cache=self.engine.cache,
            )
            self.thread_pool.start(worker)
            scheduled += 1
