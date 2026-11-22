# gui/developer/render_engine.py
from __future__ import annotations
import os
import time
import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from PySide6.QtCore import Qt, QTimer, QPoint, QEvent
from PySide6.QtGui import QImage, QPixmap, QCursor, QColor
from PySide6.QtWidgets import QApplication

from core.log_hub import log_hub
from core.stretch import (
    calculate_mtf_params,
    calculate_sky_neutral_wb,
    calculate_sky_mtf_params,
    calculate_smart_ground_lift,
    manual_stretch,
    compute_ghs_lut,
    apply_ghs_stretch,
    estimate_ghs_params,
    apply_white_balance,
    extract_background_polynomial,
    adjust_vibrance,
    adjust_contrast,
    apply_clarity,
    apply_dehaze,
    apply_curve_lut,
    apply_hue_vs_sat_curve,
    apply_luma_vs_sat_curve,
    realce_multiescala_ondiculas
)
from core.light_pollution import (
    compute_light_pollution_dome,
    apply_parametric_light_pollution,
    estimate_light_pollution_center,
    LP_PRESET_TINTS
)
from core.orton import (
    apply_astronomical_orton,
    ORTON_PRESET_TINTS
)
from core.star_diffusion import apply_star_diffusion
from core.deconvolution import (
    generate_moffat_psf,
    generate_gaussian_psf,
    estimate_psf_from_stars,
    richardson_lucy_deconvolve
)
from core.masking import refine_mask_guided
from core.mask_layers import (
    MaskLayer, apply_all_mask_layers,
    MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS,
    MASK_TYPE_IMAGE_GROUND, MASK_TYPE_BRUSH, MASK_TYPE_LUMA, MASK_TYPE_COLOR,
    MASK_TYPE_GROUND, MASK_TYPE_SKY, MASK_TYPE_IMAGE_RASTER,
    apply_pre_curve_adjustments, get_canvas_below_layer, compute_effective_layer_mask,
    apply_local_adjustments_to_image, clear_subcomposite_cache
)
from gui.developer.developer_state import DeveloperState
from gui.developer.pipeline import ResolutionPyramid, apply_developer_pipeline
from gui.developer.async_worker import AsyncPipelineManager
from core.buffer_cache import PipelineBufferCache
from core.display_transfer import (
    compute_combined_stf_lut,
    apply_display_lut16
)

logger = logging.getLogger(__name__)


class RenderMixin:
    """Mixin que orquesta la generación de proxies de previsualización, cachés multinivel,
    pipeline asíncrono y renderizado interactivo a 60 FPS."""

    def _invalidate_pipeline_caches(self):
        """Invalida todas las cachés intermedias del pipeline de procesado."""
        clear_subcomposite_cache()
        if hasattr(self, 'buffer_cache'):
            self.buffer_cache.clear()
        self._cache_starless_key = None
        self._cache_starless_linear = None
        self._cache_starless_pre_key = None
        self._cache_ms_pre_lp_key = None
        self._cache_ms_pre_lp_img = None
        self._cache_ms_linear_key = None
        self._cache_ms_linear_img = None
        self._cache_ms_linear_u16 = None
        self._cache_ms_stars_u16 = None
        self._scrub_display_buf = None
        self._scrub_before_buf = None
        self._cache_ms_stretch_key = None
        self._cache_ms_stretch_img = None
        self._cache_ms_graded_key = None
        self._cache_ms_graded_img = None
        self._cache_ms_spatial_key = None
        self._cache_ms_spatial_img = None
        self._cache_ms_stars_key = None
        self._cache_ms_stars_img = None
        self._cache_pre_layers_key = None
        self._cache_pre_layers_stretched = None
        self._cache_pre_layers_stars = None
        self._cached_base_stretched = None
        self._cached_stars_stretched = None
        self._cached_sky_mask = None
        self._cached_rendered_image = None
        self._last_rendered_preview = None
        self._fit_view_pixmap = None
        self._fit_view_orig_rgb = None
        self._fit_view_mask_overlay = None

    def _generate_preview_proxy(self):
        self._invalidate_pipeline_caches()
        if self.image_32bit is None:
            self.pyramid_image = None
            self.pyramid_starless = None
            self.pyramid_stars = None
            self.preview_proxy = None
            self.draft_proxy = None
            self.proxy_starless = None
            self.draft_starless = None
            self.proxy_stars = None
            self.draft_stars = None
            self.proxy_mask = None
            self.draft_mask = None
            return

        h, w = self.image_32bit.shape[:2]

        # Pirámides de resolución multi-escala (Level 1: <=1600px Preview, Level 2: <=800px Draft 60 FPS)
        self.pyramid_image = ResolutionPyramid(self.image_32bit, max_preview_dim=1600, max_levels=4)
        self.preview_proxy = self.pyramid_image.preview
        self.draft_proxy = self.pyramid_image.draft

        ph, pw = self.preview_proxy.shape[:2]
        dh, dw = self.draft_proxy.shape[:2]

        if self.image_starless is not None:
            self.pyramid_starless = ResolutionPyramid(self.image_starless, max_preview_dim=1600, max_levels=4)
            p_sl = self.pyramid_starless.preview
            d_sl = self.pyramid_starless.draft
            if p_sl.shape[:2] != (ph, pw):
                p_sl = cv2.resize(p_sl, (pw, ph), interpolation=cv2.INTER_AREA)
            if d_sl.shape[:2] != (dh, dw):
                d_sl = cv2.resize(d_sl, (dw, dh), interpolation=cv2.INTER_AREA)
            self.proxy_starless = np.ascontiguousarray(p_sl, dtype=np.float32)
            self.draft_starless = np.ascontiguousarray(d_sl, dtype=np.float32)

            self.pyramid_stars = ResolutionPyramid(self.image_stars, max_preview_dim=1600, max_levels=4)
            p_st = self.pyramid_stars.preview
            d_st = self.pyramid_stars.draft
            if p_st.shape[:2] != (ph, pw):
                p_st = cv2.resize(p_st, (pw, ph), interpolation=cv2.INTER_AREA)
            if d_st.shape[:2] != (dh, dw):
                d_st = cv2.resize(d_st, (dw, dh), interpolation=cv2.INTER_AREA)
            self.proxy_stars = np.ascontiguousarray(p_st, dtype=np.float32)
            self.draft_stars = np.ascontiguousarray(d_st, dtype=np.float32)
        else:
            self.pyramid_starless = None
            self.proxy_starless = None
            self.draft_starless = None
            self.pyramid_stars = None
            self.proxy_stars = None
            self.draft_stars = None

        clean_mask = None
        if self.current_mask is not None:
            clean_mask = np.squeeze(self.current_mask)
            if clean_mask.ndim > 2:
                clean_mask = clean_mask[..., 0]
            clean_mask = clean_mask.astype(np.float32)
            if clean_mask.shape != (h, w):
                clean_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Máscara nivel Preview (Level 1)
            ph, pw = self.preview_proxy.shape[:2]
            m_small = cv2.resize(clean_mask, (pw, ph), interpolation=cv2.INTER_LINEAR)
            ksize = int(max(7, (min(ph, pw) // 150) | 1))
            if ksize % 2 == 0: 
                ksize += 1
            sm = cv2.GaussianBlur(m_small, (ksize, ksize), sigmaX=ksize / 3.0)
            self.proxy_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)

            # Máscara nivel Draft (Level 2)
            dh, dw = self.draft_proxy.shape[:2]
            m_draft = cv2.resize(clean_mask, (dw, dh), interpolation=cv2.INTER_LINEAR)
            ksize_d = int(max(5, (min(dh, dw) // 150) | 1))
            if ksize_d % 2 == 0:
                ksize_d += 1
            sm_d = cv2.GaussianBlur(m_draft, (ksize_d, ksize_d), sigmaX=ksize_d / 3.0)
            self.draft_mask = np.ascontiguousarray(np.repeat(sm_d[..., np.newaxis], 3, axis=2), dtype=np.float32)
        else:
            self.proxy_mask = None
            self.draft_mask = None

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.invalidate_layer_thumbnail(None)

    def _process_starless_base(
        self, 
        starless: Optional[np.ndarray], 
        mask_3d: Optional[np.ndarray] = None, 
        is_full_res: bool = True,
        crop_rect: Optional[tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """Aplica las mejoras lineales a la capa StarNet (Denoise, Claridad, Dehaze, Ondículas)."""
        if starless is None:
            return None

        if crop_rect is None and getattr(self, 'is_zoomed_100', False):
            crop_rect = getattr(self, '_zoom_crop_rect', None)

        # Bypass ultra-rápido si todos los filtros lineales están neutros
        is_neutral = (
            getattr(self, 'denoise_strength', 0.0) <= 1e-4 and
            abs(getattr(self, 'clarity_starless_val', 0.0)) <= 1e-4 and
            abs(getattr(self, 'dehaze_starless_val', 0.0)) <= 1e-4 and
            abs(getattr(self, 'multiescala_estructura', 0.0)) <= 1e-4 and
            getattr(self, 'multiescala_fondo', 0.0) <= 1e-4 and
            getattr(self, 'orton_strength', 0.0) <= 1e-4
        )
        if is_neutral:
            return starless

        is_draft = (not is_full_res and getattr(self, '_is_scrubbing_heavy_slider', False))
        orig_h, orig_w = starless.shape[:2]

        cache_key_full = (
            id(starless),
            id(mask_3d),
            id(getattr(self, '_cache_starless_ai', None)),
            is_full_res,
            is_draft,
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'denoise_color_strength', 0.5)), 4),
            round(float(getattr(self, 'denoise_masking', 0.5)), 4),
            round(float(getattr(self, 'clarity_starless_val', 0.0)), 4),
            round(float(getattr(self, 'dehaze_starless_val', 0.0)), 4),
            round(float(getattr(self, 'multiescala_estructura', 0.0)), 4),
            round(float(getattr(self, 'multiescala_fondo', 0.0)), 4),
            round(float(getattr(self, 'orton_strength', 0.0)), 4),
            round(float(getattr(self, 'orton_radius', 40.0)), 2),
            round(float(getattr(self, 'orton_threshold', 0.05)), 4),
            getattr(self, 'orton_tint_type', 'warm_gold'),
            getattr(self, 'orton_custom_tint', (1.0, 0.85, 0.55)),
            round(float(getattr(self, 'orton_tint_strength', 0.5)), 4),
            getattr(self, 'orton_mode', 'screen')
        )
        if not is_full_res and self._cache_starless_key == cache_key_full and self._cache_starless_linear is not None:
            return self._cache_starless_linear

        # Verificación en caché de RAM Nivel 1b (Denoise, Claridad, Ondículas pre-Orton)
        cache_key_pre = (
            id(starless),
            id(mask_3d),
            id(getattr(self, '_cache_starless_ai', None)),
            is_full_res,
            is_draft,
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'denoise_color_strength', 0.5)), 4),
            round(float(getattr(self, 'denoise_masking', 0.5)), 4),
            round(float(getattr(self, 'clarity_starless_val', 0.0)), 4),
            round(float(getattr(self, 'dehaze_starless_val', 0.0)), 4),
            round(float(getattr(self, 'multiescala_estructura', 0.0)), 4),
            round(float(getattr(self, 'multiescala_fondo', 0.0)), 4),
        )
        if not is_full_res and self._cache_starless_pre_key == cache_key_pre and self._cache_starless_pre_img is not None:
            proc_starless = self._cache_starless_pre_img.copy()
            proc_mask_3d = mask_3d
        else:
            if is_draft and max(orig_w, orig_h) > 800:
                scale_d = 800.0 / float(max(orig_w, orig_h))
                draft_w = max(32, int(orig_w * scale_d))
                draft_h = max(32, int(orig_h * scale_d))
                proc_starless = cv2.resize(starless, (draft_w, draft_h), interpolation=cv2.INTER_AREA)
                proc_mask_3d = cv2.resize(mask_3d, (draft_w, draft_h), interpolation=cv2.INTER_AREA) if mask_3d is not None else None
            else:
                proc_starless = starless.copy()
                proc_mask_3d = mask_3d

            # 1. Reducción de Ruido Multietapa (Luminancia + Croma + Máscara o NAFNet IA)
            if self.denoise_strength > 1e-4:
                cached_ai = getattr(self, '_cache_starless_ai', None)
                use_ai = getattr(self, 'starless_ai_denoise_enabled', False) and (cached_ai is not None)
                if use_ai:
                    h_f, w_f = self.image_32bit.shape[:2] if self.image_32bit is not None else cached_ai.shape[:2]
                    h_c, w_c = cached_ai.shape[:2]
                    ratio_c = w_c / float(h_c) if h_c > 0 else 1.0
                    ratio_f = w_f / float(h_f) if h_f > 0 else 1.0
                    if abs(ratio_c - ratio_f) > 0.05 and crop_rect is None:
                        self._cache_starless_ai = None
                        use_ai = False
                if use_ai:
                    if crop_rect is not None:
                        x1, y1, x2, y2 = crop_rect
                        sx = w_c / float(w_f)
                        sy = h_c / float(h_f)
                        sx1 = max(0, min(w_c, int(round(x1 * sx))))
                        sy1 = max(0, min(h_c, int(round(y1 * sy))))
                        sx2 = max(0, min(w_c, int(round(x2 * sx))))
                        sy2 = max(0, min(h_c, int(round(y2 * sy))))
                        ai_sub = cached_ai[sy1:sy2, sx1:sx2]
                        if ai_sub.shape[:2] != proc_starless.shape[:2]:
                            ai_aligned = cv2.resize(ai_sub, (proc_starless.shape[1], proc_starless.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            ai_aligned = ai_sub
                    elif cached_ai.shape[:2] != proc_starless.shape[:2]:
                        ai_aligned = cv2.resize(cached_ai, (proc_starless.shape[1], proc_starless.shape[0]), interpolation=cv2.INTER_AREA)
                    else:
                        ai_aligned = cached_ai
                    k = float(np.clip(self.denoise_strength, 0.0, 1.0))
                    denoised = cv2.addWeighted(ai_aligned, k, proc_starless, 1.0 - k, 0.0)
                else:
                    dn_color_ratio = getattr(self, 'denoise_color_strength', 0.5) / 0.5
                    dn_color = min(1.0, self.denoise_strength * dn_color_ratio)
                    from core.denoise import apply_astro_denoise
                    denoised = apply_astro_denoise(
                        proc_starless,
                        strength=self.denoise_strength,
                        color_strength=dn_color,
                        masking=getattr(self, 'denoise_masking', 0.5),
                        sky_mask=proc_mask_3d[..., 0] if proc_mask_3d is not None else None,
                        is_draft=is_draft,
                        is_full_res=(is_full_res and not is_draft)
                    )
                if proc_mask_3d is not None:
                    proc_starless = (denoised * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = denoised

            # 2. Claridad Clásica (Gauss), Claridad Textura (Seti) y Borrar Neblina
            if abs(self.clarity_starless_val) > 1e-4:
                clarified = apply_clarity(proc_starless, strength=self.clarity_starless_val)
                if proc_mask_3d is not None:
                    proc_starless = (clarified * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = clarified

            if abs(self.texture_clarity_starless_val) > 1e-4:
                from core.sharpening import apply_texture_clarity
                t_clarified = apply_texture_clarity(proc_starless, strength=self.texture_clarity_starless_val)
                if proc_mask_3d is not None:
                    proc_starless = (t_clarified * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = t_clarified

            if abs(self.dehaze_starless_val) > 1e-4:
                dehazed = apply_dehaze(proc_starless, strength=self.dehaze_starless_val)
                if proc_mask_3d is not None:
                    proc_starless = (dehazed * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = dehazed

            # 3. Ondículas sobre la capa de fondo
            if abs(self.multiescala_estructura) > 1e-4 or self.multiescala_fondo > 1e-4:
                proc_starless = realce_multiescala_ondiculas(
                    proc_starless,
                    fuerza_estructura=self.multiescala_estructura,
                    reduccion_fondo=self.multiescala_fondo,
                    sky_mask=proc_mask_3d[..., 0] if proc_mask_3d is not None else None
                )

            if not is_full_res:
                self._cache_starless_pre_key = cache_key_pre
                self._cache_starless_pre_img = proc_starless.copy()

        # 4. Efecto Orton y Resplandor de Polvo Galáctico sobre Fondo Nebular
        if self.orton_strength > 1e-4:
            proc_starless = apply_astronomical_orton(
                proc_starless,
                strength=self.orton_strength,
                radius=self.orton_radius * (0.5 if is_draft else 1.0),
                threshold=self.orton_threshold,
                exponent=1.0,
                tint_type=self.orton_tint_type,
                custom_tint=getattr(self, 'orton_custom_tint', (1.0, 0.85, 0.55)),
                tint_strength=self.orton_tint_strength,
                blend_mode=self.orton_mode,
                sky_mask=proc_mask_3d[..., 0] if proc_mask_3d is not None else None,
                is_starless=True
            )

        if is_draft and proc_starless.shape[:2] != (orig_h, orig_w):
            processed_starless = cv2.resize(proc_starless, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            processed_starless = proc_starless
            if not is_full_res:
                self._cache_starless_key = cache_key_full
                self._cache_starless_linear = processed_starless

        return processed_starless

    def _compose_active_base(self, for_export: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if for_export:
            starless, stars, base = self.image_starless, self.image_stars, self.image_32bit
            mask_3d = None
            if self.current_mask is not None and base is not None:
                h_f, w_f = base.shape[:2]
                m_clean = self.current_mask[..., 0] if self.current_mask.ndim == 3 else self.current_mask
                if m_clean.dtype != np.float32:
                    m_clean = (m_clean / 255.0).astype(np.float32) if m_clean.dtype == np.uint8 else m_clean.astype(np.float32)
                elif m_clean.max() > 1.05:
                    m_clean = m_clean / 255.0
                m_f = cv2.resize(m_clean, (w_f, h_f), interpolation=cv2.INTER_LINEAR) if m_clean.shape[:2] != (h_f, w_f) else m_clean
                mask_3d = np.ascontiguousarray(np.repeat(m_f[..., np.newaxis], 3, axis=2), dtype=np.float32)
            crop_active = False
            self._zoom_crop_rect = None
        elif self.is_zoomed_100 and self.image_32bit is not None:
            h_f, w_f = self.image_32bit.shape[:2]

            # Propagación de margen dinámico (Padding) guiado por filtros espaciales activos (Orton / Difusión)
            active_radius = 0.0
            if float(getattr(self, 'orton_strength', 0.0)) > 1e-4:
                active_radius = max(active_radius, float(getattr(self, 'orton_radius', 40.0)))
            if float(getattr(self, 'star_diff_strength', 0.0)) > 1e-4:
                active_radius = max(active_radius, float(getattr(self, 'star_diff_radius', 20.0)))

            # Margen perimetral: para filtros convolucionales (3-sigma). Acotado entre 64 px y 512 px
            pad_margin = int(min(512, max(64, round(3.0 * active_radius)))) if active_radius > 0 else 64
            cw = min(w_f, max(100, self.canvas.width() + pad_margin * 2))
            ch = min(h_f, max(100, self.canvas.height() + pad_margin * 2))
            
            rx, ry = self.zoom_rel_coords
            cx = int(rx * w_f)
            cy = int(ry * h_f)

            x1 = max(0, min(w_f - cw, cx - cw // 2))
            y1 = max(0, min(h_f - ch, cy - ch // 2))
            x2 = x1 + cw
            y2 = y1 + ch
            self._zoom_crop_rect = (x1, y1, x2, y2)

            base = self.image_32bit[y1:y2, x1:x2].copy()
            if self.image_starless is not None:
                if self.image_starless.shape[:2] == (h_f, w_f):
                    starless = self.image_starless[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_starless.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_starless[sy1:sy2, sx1:sx2]
                    starless = cv2.resize(sub, (cw, ch), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (ch, cw) else sub.copy()
            else:
                starless = None

            if self.image_stars is not None:
                if self.image_stars.shape[:2] == (h_f, w_f):
                    stars = self.image_stars[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_stars.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_stars[sy1:sy2, sx1:sx2]
                    stars = cv2.resize(sub, (cw, ch), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (ch, cw) else sub.copy()
            else:
                stars = None

            if self.current_mask is not None:
                m_clean = self.current_mask[..., 0] if self.current_mask.ndim == 3 else self.current_mask
                if m_clean.dtype != np.float32:
                    m_clean = (m_clean / 255.0).astype(np.float32) if m_clean.dtype == np.uint8 else m_clean.astype(np.float32)
                elif m_clean.max() > 1.05:
                    m_clean = m_clean / 255.0
                if m_clean.shape[:2] != (h_f, w_f):
                    m_f = cv2.resize(m_clean, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
                else:
                    m_f = m_clean
                m_sub = np.ascontiguousarray(m_f[y1:y2, x1:x2], dtype=np.float32)
                mask_3d = np.ascontiguousarray(np.repeat(m_sub[..., np.newaxis], 3, axis=2), dtype=np.float32)
            else:
                mask_3d = None
            crop_active = True
        else:
            is_scrubbing = getattr(self, '_is_scrubbing_heavy_slider', False)
            if self.image_starless is not None:
                if is_scrubbing and getattr(self, 'draft_proxy', None) is not None:
                    starless = self.draft_starless
                    stars = self.draft_stars
                else:
                    starless = self.proxy_starless
                    stars = self.proxy_stars
            else:
                starless = None
                stars = None

            if is_scrubbing and getattr(self, 'draft_proxy', None) is not None:
                base = self.draft_proxy
                mask_3d = self.draft_mask
            else:
                base = self.preview_proxy
                mask_3d = self.proxy_mask
            crop_active = False
            self._zoom_crop_rect = None

        if base is None:
            return None, None

        bh, bw = base.shape[:2]
        if starless is not None and starless.shape[:2] != (bh, bw):
            starless = cv2.resize(starless, (bw, bh), interpolation=cv2.INTER_AREA)
        if stars is not None and stars.shape[:2] != (bh, bw):
            stars = cv2.resize(stars, (bw, bh), interpolation=cv2.INTER_AREA)
        if mask_3d is not None:
            if mask_3d.shape[:2] != (bh, bw):
                mask_3d = cv2.resize(mask_3d, (bw, bh), interpolation=cv2.INTER_LINEAR)
            if base.ndim == 3:
                if mask_3d.ndim == 2:
                    mask_3d = np.repeat(mask_3d[..., np.newaxis], 3, axis=2)
                elif mask_3d.ndim == 3 and mask_3d.shape[2] == 1:
                    mask_3d = np.repeat(mask_3d, 3, axis=2)

        if self.image_starless is None or starless is None or stars is None:
            return base, mask_3d

        processed_starless = self._process_starless_base(
            starless,
            mask_3d=mask_3d,
            is_full_res=(for_export or crop_active),
            crop_rect=self._zoom_crop_rect if self.is_zoomed_100 else None
        )

        # 4. Mezcla de capas
        # Si el gestor de capas contiene capas de imagen activas (StarNet / Suelo),
        # delegamos la composición completa al motor multicapa de apply_all_mask_layers
        # para respetar el orden visual de capas del usuario y evitar duplicar estrellas.
        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.has_image_layers():
            if self.view_layer_mode == 2:
                return stars, mask_3d
            return processed_starless, mask_3d

        if self.view_layer_mode == 0:
            composed = np.clip(processed_starless + (stars * self.star_intensity), 0.0, 1.0)
            return composed, mask_3d
        elif self.view_layer_mode == 1:
            return processed_starless, mask_3d
        else:
            return stars, mask_3d

    def _apply_pipeline_on_image(
        self,
        target_img: np.ndarray,
        precomputed_mask: Optional[np.ndarray] = None,
        apply_mask_layers: bool = True,
        is_preview: bool = True,
        asynchronous: bool = False
    ) -> Optional[np.ndarray]:
        if target_img is None:
            return None

        is_full_res = (self.image_32bit is not None and target_img.shape[:2] == self.image_32bit.shape[:2])
        is_zoomed = (self.is_zoomed_100 and getattr(self, '_zoom_crop_rect', None) is not None)
        full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None
        crop_rect = getattr(self, '_zoom_crop_rect', None) if is_zoomed else None

        curve_state = tuple((m, tuple(pts)) for m, pts in sorted(self.curve_widget.get_state().items())) if hasattr(self, 'curve_widget') else ()

        # Hito 0a: Parámetros Pre-Polución (Dual-Band, Corrección Estelar, PCC, Balance de blancos, Saturación, Vibrance)
        col_st = self.panel_color_wb.get_state() if hasattr(self, 'panel_color_wb') else {}
        key_pre_lp = (
            id(target_img),
            id(precomputed_mask),
            is_full_res,
            is_zoomed,
            crop_rect,
            col_st.get("dualband_mode", "none"),
            int(col_st.get("dualband_oiii_boost", 100)),
            int(col_st.get("unclip_stars_val", 0)),
            int(col_st.get("unpurple_val", 0)),
            int(col_st.get("scnr_green_val", 0)),
            round(float(col_st.get("pcc_kr", 1.0)), 4),
            round(float(col_st.get("pcc_kb", 1.0)), 4),
            round(float(self.temp_val), 4),
            round(float(self.tint_val), 4),
            round(float(self.sat_sky_val), 4),
            round(float(self.sat_gnd_val), 4),
            round(float(self.vibrance_val), 4),
        )

        # Hito 0b: Parámetros Lineales (Polución Lumínica, Suelo, Denoise Lineal)
        key_linear = (
            key_pre_lp,
            round(float(self.lp_reduction), 4),
            round(float(self.lp_center_x), 4),
            round(float(self.lp_center_y), 4),
            round(float(self.lp_width), 4),
            round(float(self.lp_height), 4),
            getattr(self, 'lp_tint_type', 'sodium'),
            getattr(self, 'lp_custom_tint', (1.0, 0.68, 0.22)),
            round(float(getattr(self, 'lp_tint_strength', 0.5)), 4),
            getattr(self, 'lp_mode', 'attenuate'),
            getattr(self, 'lp_legacy_1d', False),
            round(float(self.gnd_ev_val), 4),
            round(float(self.gnd_shadows_val), 4),
            round(float(self.gnd_bp_val), 4),
            round(float(getattr(self, 'gnd_denoise_val', 0.0)), 4),
            bool(getattr(self, 'gnd_ai_denoise_enabled', False)),
            id(getattr(self, '_cache_ground_ai', None)),
            int(getattr(self, 'gnd_edge_shift_val', 0)),
            round(float(getattr(self, 'gnd_antihalo_val', 0.0)), 4),
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'denoise_color_strength', 0.5)), 4),
            round(float(getattr(self, 'denoise_masking', 0.5)), 4),
        )

        # Hito 1: Parámetros de Estirado (MTF / GHS)
        st = self.panel_stretch.get_state() if hasattr(self, 'panel_stretch') else {}
        key_stretch = (
            key_linear,
            st.get("stretch_mode", "mtf"),
            st.get("ghs_color_mode", "luma"),
            round(float(st.get("ghs_sp_val", 0.02)), 5),
            round(float(st.get("ghs_d_val", 5.0)), 4),
            round(float(st.get("ghs_b_val", 2.0)), 4),
            round(float(st.get("ghs_hp_val", 0.0)), 4),
            round(float(st.get("ghs_lp_val", 0.0)), 4),
            round(float(st.get("ghs_bp_val", 0.0)), 5),
            round(float(st.get("bp_val", 0.0)), 5),
            round(float(st.get("mtf_val", 0.15)), 5),
        )
        # Hito 2: Parámetros de Curvas y Contraste
        key_graded = (
            key_stretch,
            curve_state,
            round(float(self.contrast_starless_val), 4),
            round(float(self.contrast_val), 4),
        )

        # Hito 3: Parámetros Espaciales (Deconvolución, Orton)
        key_spatial = (
            key_graded,
            getattr(self, "deconv_iterations", 0),
            round(float(getattr(self, "deconv_fwhm", 3.0)), 2),
            round(float(getattr(self, "deconv_deringing", 0.08)), 4),
            round(float(getattr(self, "deconv_damping", 0.25)), 4),
            round(float(getattr(self, "deconv_aspect", 1.0)), 2),
            round(float(getattr(self, "deconv_angle", 0.0)), 2),
            getattr(self, "deconv_model", "gaussian"),
            getattr(self, "deconv_scope", "full"),
            getattr(self, "deconv_luma_only", True),
            round(float(self.orton_strength), 4),
            round(float(self.orton_radius), 2),
            round(float(self.orton_threshold), 4),
            getattr(self, 'orton_tint_type', 'warm_gold'),
            getattr(self, 'orton_custom_tint', (1.0, 0.85, 0.55)),
            round(float(getattr(self, 'orton_tint_strength', 0.5)), 4),
            getattr(self, 'orton_mode', 'screen'),
        )

        # Resolve starless and stars image slices
        if is_zoomed:
            x1, y1, x2, y2 = self._zoom_crop_rect
            if self.image_stars is not None and full_shape is not None:
                h_f, w_f = full_shape
                if self.image_stars.shape[:2] == (h_f, w_f):
                    st_img = self.image_stars[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_stars.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_stars[sy1:sy2, sx1:sx2]
                    st_img = cv2.resize(sub, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (y2 - y1, x2 - x1) else sub.copy()
            else:
                st_img = None

            if self.image_starless is not None and full_shape is not None:
                h_f, w_f = full_shape
                if self.image_starless.shape[:2] == (h_f, w_f):
                    sl_img = self.image_starless[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_starless.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_starless[sy1:sy2, sx1:sx2]
                    sl_img = cv2.resize(sub, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (y2 - y1, x2 - x1) else sub.copy()
            else:
                sl_img = None
        elif is_full_res:
            st_img = self.image_stars if self.image_starless is not None else None
            sl_img = self.image_starless
        else:
            if self.image_starless is not None:
                if getattr(self, '_is_scrubbing_heavy_slider', False):
                    st_img = self.draft_stars if self.draft_stars is not None else self.proxy_stars
                    sl_img = self.draft_starless if self.draft_starless is not None else self.proxy_starless
                else:
                    st_img = self.proxy_stars if self.proxy_stars is not None else self.image_stars
                    sl_img = self.proxy_starless if self.proxy_starless is not None else self.image_starless
            else:
                st_img = None
                sl_img = None

        # Mask layers
        mask_layers = self.mask_layers_widget.layers if (apply_mask_layers and hasattr(self, 'mask_layers_widget')) else None

        # Hito Estrellas: Estirado y Filtro Difusor de Estrellas
        key_stars = (
            id(st_img),
            key_stretch,
            round(float(getattr(self, 'star_intensity', 1.0)), 4),
            round(float(getattr(self, 'star_diff_strength', 0.0)), 4),
            round(float(getattr(self, 'star_diff_radius', 20.0)), 2),
            round(float(getattr(self, 'star_diff_threshold', 0.45)), 4),
            round(float(getattr(self, 'star_diff_color_boost', 0.50)), 4),
            getattr(self, 'star_diff_mode', 'screen'),
            id(precomputed_mask),
            bool(mask_layers and len(mask_layers) > 0),
        )

        # Determinar hitos en caché reutilizables mediante caché LRU unificada por hash y dimensiones
        t_shape = target_img.shape[:2]
        if is_preview and hasattr(self, 'buffer_cache'):
            pre_spatial = self.buffer_cache.get("spatial", key_spatial, t_shape)
            pre_graded = self.buffer_cache.get("graded", key_graded, t_shape) if pre_spatial is None else None
            pre_stretched = self.buffer_cache.get("stretch", key_stretch, t_shape) if (pre_spatial is None and pre_graded is None) else None
            pre_linear = self.buffer_cache.get("linear", key_linear, t_shape) if (pre_spatial is None and pre_graded is None and pre_stretched is None) else None
            pre_pre_lp = self.buffer_cache.get("pre_lp", key_pre_lp, t_shape) if (pre_spatial is None and pre_graded is None and pre_stretched is None and pre_linear is None) else None
            pre_stars = self.buffer_cache.get("stars", key_stars, t_shape)
        else:
            pre_spatial = pre_graded = pre_stretched = pre_linear = pre_pre_lp = pre_stars = None

        # Build DeveloperState snapshot (omitiendo mask_layers para no serializar innecesariamente)
        state = DeveloperState.from_dict(self._capture_state(include_mask_layers=False))

        # Collect tone curves
        curve_luts = {}
        if hasattr(self, 'curve_widget'):
            if not self.curve_widget.is_identity("rgb"):
                curve_luts["rgb"] = self.curve_widget.get_lut("rgb")
            if not self.curve_widget.is_identity("hue_sat"):
                curve_luts["hue_sat"] = self.curve_widget.get_lut("hue_sat")
            if not self.curve_widget.is_identity("luma_sat"):
                curve_luts["luma_sat"] = self.curve_widget.get_lut("luma_sat")

        # Get PSF kernel
        psf_kernel = self._get_current_deconv_psf() if hasattr(self, '_get_current_deconv_psf') else None

        # Mask layers (instantánea superficial rápida para aislar el hilo de render)
        raw_layers = self.mask_layers_widget.layers if (apply_mask_layers and hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.layers) else None
        mask_layers = [l.shallow_snapshot() for l in raw_layers] if raw_layers else None

        is_scrubbing = getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_scrubbing_layer_slider', False)
        is_dragging_antihalo = getattr(self, '_is_dragging_antihalo', False)

        cached_gnd_ai = self._cache_ground_ai if getattr(self, 'gnd_ai_denoise_enabled', False) else None

        if asynchronous and getattr(self, 'async_pipeline', None) is not None:
            self._last_async_meta = (
                key_pre_lp, key_linear, key_stretch, key_graded, key_spatial, key_stars,
                is_zoomed, is_scrubbing, st_img, precomputed_mask
            )
            self._last_async_base = target_img
            self.async_pipeline.submit(
                target_img=target_img,
                state=state,
                cached_ground_ai=cached_gnd_ai,
                precomputed_mask=precomputed_mask,
                image_stars=st_img,
                image_starless=sl_img,
                mask_layers=mask_layers,
                curve_luts=curve_luts,
                psf_kernel=psf_kernel,
                is_preview=is_preview,
                is_scrubbing=is_scrubbing,
                crop_rect=crop_rect,
                full_shape=full_shape,
                precomputed_pre_lp_base=pre_pre_lp,
                precomputed_linear_base=pre_linear,
                precomputed_stretched_base=pre_stretched,
                precomputed_graded_base=pre_graded,
                precomputed_spatial_base=pre_spatial,
                precomputed_stretched_stars=pre_stars,
                is_dragging_deconv=getattr(self, '_is_dragging_deconv', False),
                is_dragging_antihalo=is_dragging_antihalo,
            )
            return None

        final_img, out_pre_lp, out_lin, out_str, out_grd, out_spt, out_stars = apply_developer_pipeline(
            target_img=target_img,
            state=state,
            cached_ground_ai=cached_gnd_ai,
            precomputed_mask=precomputed_mask,
            image_stars=st_img,
            image_starless=sl_img,
            mask_layers=mask_layers,
            curve_luts=curve_luts,
            psf_kernel=psf_kernel,
            is_preview=is_preview,
            is_scrubbing=is_scrubbing,
            crop_rect=crop_rect,
            full_shape=full_shape,
            precomputed_pre_lp_base=pre_pre_lp,
            precomputed_linear_base=pre_linear,
            precomputed_stretched_base=pre_stretched,
            precomputed_graded_base=pre_graded,
            precomputed_spatial_base=pre_spatial,
            precomputed_stretched_stars=pre_stars,
            return_milestones=True,
            is_dragging_deconv=getattr(self, '_is_dragging_deconv', False),
            is_dragging_antihalo=is_dragging_antihalo,
        )

        if is_preview:
            if not is_scrubbing:
                if hasattr(self, 'buffer_cache'):
                    self.buffer_cache.put("pre_lp", key_pre_lp, t_shape, out_pre_lp)
                    self.buffer_cache.put("linear", key_linear, t_shape, out_lin)
                    self.buffer_cache.put("stretch", key_stretch, t_shape, out_str)
                    self.buffer_cache.put("graded", key_graded, t_shape, out_grd)
                    self.buffer_cache.put("spatial", key_spatial, t_shape, out_spt)
                    self.buffer_cache.put("stars", key_stars, t_shape, out_stars)

                self._cache_ms_pre_lp_key = key_pre_lp
                self._cache_ms_pre_lp_img = out_pre_lp
                self._cache_ms_linear_key = key_linear
                self._cache_ms_linear_img = out_lin
                if out_lin is not None and not is_zoomed:
                    self._cache_ms_linear_u16 = np.clip(out_lin * 65535.0, 0, 65535).astype(np.uint16)
                if st_img is not None and not is_zoomed:
                    self._cache_ms_stars_u16 = np.clip(st_img * 65535.0, 0, 65535).astype(np.uint16)
                self._cache_ms_stretch_key = key_stretch
                self._cache_ms_stretch_img = out_str
                self._cache_ms_graded_key = key_graded
                self._cache_ms_graded_img = out_grd
                self._cache_ms_spatial_key = key_spatial
                self._cache_ms_spatial_img = out_spt
                self._cache_ms_stars_key = key_stars
                self._cache_ms_stars_img = out_stars
                self._cache_pre_layers_key = (key_spatial, key_stars)
                self._cache_pre_layers_stretched = out_spt
                self._cache_pre_layers_stars = out_stars
                if not is_zoomed:
                    self._cached_base_stretched = out_spt
                    self._cached_stars_stretched = out_stars
                    self._cached_sky_mask = precomputed_mask
                    self._cached_rendered_image = final_img

        return final_img

    def _on_async_preview_ready(self, req_id: int, result: Any):
        if result is None:
            return
        final_img, out_pre_lp, out_lin, out_str, out_grd, out_spt, out_stars = result
        if final_img is None:
            return

        meta = getattr(self, '_last_async_meta', None)
        if meta is not None:
            key_pre_lp, key_linear, key_stretch, key_graded, key_spatial, key_stars, is_zoomed, is_scrubbing, st_img, precomputed_mask = meta
            if is_scrubbing and not getattr(self, '_is_slider_pressed', False) and not getattr(self, '_is_scrubbing_heavy_slider', False) and not getattr(self, '_is_scrubbing_layer_slider', False):
                # Descartar solicitud de borrador tardía si el usuario ya soltó el deslizador
                return
            if not is_scrubbing:
                t_shape = final_img.shape[:2] if hasattr(final_img, 'shape') else ()
                if hasattr(self, 'buffer_cache') and t_shape:
                    self.buffer_cache.put("pre_lp", key_pre_lp, t_shape, out_pre_lp)
                    self.buffer_cache.put("linear", key_linear, t_shape, out_lin)
                    self.buffer_cache.put("stretch", key_stretch, t_shape, out_str)
                    self.buffer_cache.put("graded", key_graded, t_shape, out_grd)
                    self.buffer_cache.put("spatial", key_spatial, t_shape, out_spt)
                    self.buffer_cache.put("stars", key_stars, t_shape, out_stars)

                self._cache_ms_pre_lp_key = key_pre_lp
                self._cache_ms_pre_lp_img = out_pre_lp
                self._cache_ms_linear_key = key_linear
                self._cache_ms_linear_img = out_lin
                if out_lin is not None and not is_zoomed:
                    self._cache_ms_linear_u16 = np.clip(out_lin * 65535.0, 0, 65535).astype(np.uint16)
                if st_img is not None and not is_zoomed:
                    self._cache_ms_stars_u16 = np.clip(st_img * 65535.0, 0, 65535).astype(np.uint16)
                self._cache_ms_stretch_key = key_stretch
                self._cache_ms_stretch_img = out_str
                self._cache_ms_graded_key = key_graded
                self._cache_ms_graded_img = out_grd
                self._cache_ms_spatial_key = key_spatial
                self._cache_ms_spatial_img = out_spt
                self._cache_ms_stars_key = key_stars
                self._cache_ms_stars_img = out_stars
                self._cache_pre_layers_key = (key_spatial, key_stars)
                self._cache_pre_layers_stretched = out_spt
                self._cache_pre_layers_stars = out_stars
                if not is_zoomed:
                    self._cached_base_stretched = out_spt
                    self._cached_stars_stretched = out_stars
                    self._cached_sky_mask = precomputed_mask
                    self._cached_rendered_image = final_img

        self._last_rendered_preview = final_img
        base_to_render = getattr(self, '_last_async_base', None)
        if base_to_render is not None:
            full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None
            self.canvas.set_external_crop(None, full_shape)
            self.canvas.load_image(base_to_render, display_stretched=final_img)

        # Actualización fluida y controlada del histograma en vivo
        if not self.is_zoomed_100 and hasattr(self, '_hist_throttle_timer'):
            if not self._hist_throttle_timer.isActive():
                self._hist_throttle_timer.start(50)
        self._update_canvas_info_label()

    def _on_async_pipeline_failed(self, req_id: int, error_msg: str):
        logger.error(f"[Async Pipeline #{req_id} Error] {error_msg}")

    def update_stretch_preview(self, immediate: bool = False):
        if getattr(self, '_is_loading_project', False):
            return
        if self.is_zoomed_100 and not immediate:
            self._slider_changed_during_zoom = True
        if immediate:
            if hasattr(self, '_preview_throttle_timer'):
                self._preview_throttle_timer.stop()
            if hasattr(self, '_refine_timer'):
                self._refine_timer.stop()
            self._do_update_stretch_preview(synchronous=True)
            self._refresh_histograms_now()
        else:
            if getattr(self, '_is_slider_pressed', False) and not getattr(self, '_is_scrubbing_lut_slider', False) and not getattr(self, '_is_scrubbing_layer_slider', False):
                self._is_scrubbing_heavy_slider = True

            if hasattr(self, '_refine_timer') and (getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_scrubbing_layer_slider', False)):
                if not getattr(self, '_is_scrubbing_lut_slider', False):
                    self._refine_timer.start(350)

            if hasattr(self, '_preview_throttle_timer'):
                if getattr(self, '_is_dragging_gradient', False):
                    if not self._preview_throttle_timer.isActive():
                        self._preview_throttle_timer.start(40)
                else:
                    self._preview_throttle_timer.start(16)
            else:
                self._do_update_stretch_preview(synchronous=False)
                self._refresh_histograms_now()

    def _on_refine_preview_timeout(self):
        """Refina progresivamente la previsualización al nivel de alta calidad cuando cesa el arrastre."""
        if getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_scrubbing_layer_slider', False):
            self._is_scrubbing_heavy_slider = False
            self._is_scrubbing_layer_slider = False
            self.update_stretch_preview(immediate=True)

    def _refresh_histograms_now(self):
        if getattr(self, '_hist_throttle_timer', None) is not None:
            self._hist_throttle_timer.stop()
        stretched = getattr(self, '_last_rendered_preview', None)
        if stretched is None or self.is_zoomed_100:
            return
        # El histograma en vivo optimizado (< 0.2 ms) se actualiza de forma fluida
        if hasattr(self, 'live_histogram') and self.live_histogram.isVisible():
            self.live_histogram.update_from_image(stretched)
        # Omitir widgets más pesados durante arrastre activo para garantizar 60 FPS
        if getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_slider_pressed', False) or getattr(self, '_is_dragging_gradient', False):
            return
        if hasattr(self, 'curve_widget') and self.curve_widget.isVisible():
            self.curve_widget.set_histogram_from_image(stretched)
        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.isVisible():
            if getattr(self.mask_layers_widget, 'sec_layer_curve', None) and self.mask_layers_widget.sec_layer_curve.is_expanded:
                self.mask_layers_widget.update_active_layer_histogram()

    def _is_lut_preview_applicable(self) -> bool:
        """Determina si la previsualización directa por LUT 1D es segura y matemáticamente idéntica."""
        if not getattr(self, '_is_scrubbing_lut_slider', False):
            return False
        src_u16 = getattr(self, '_cache_ms_linear_u16', None)
        if src_u16 is None:
            return False
        if getattr(self, 'preview_proxy', None) is not None:
            if src_u16.shape[:2] != self.preview_proxy.shape[:2]:
                return False
        if self.view_layer_mode == 2 or self.is_zoomed_100:
            return False
        if hasattr(self, 'mask_layers_widget') and len(getattr(self.mask_layers_widget, 'layers', [])) > 0:
            return False
        if float(getattr(self, 'orton_strength', 0.0)) > 1e-4:
            return False
        if float(getattr(self, 'star_diff_strength', 0.0)) > 1e-4:
            return False
        if int(getattr(self, 'deconv_iterations', 0)) > 0:
            return False
        return True

    def _do_update_stretch_preview(self, synchronous: bool = False):
        if getattr(self, '_is_rendering_preview', False):
            return
        self._is_rendering_preview = True
        try:
            # === RUTA ULTRARRÁPIDA DE VISUALIZACIÓN POR LUT (Display Transfer / STF >500 FPS) ===
            # Activada estrictamente durante el arrastre continuo de deslizadores de estirado/contraste
            if self._is_lut_preview_applicable():
                curve_rgb = self.curve_widget.get_lut("rgb") if (hasattr(self, 'curve_widget') and not self.curve_widget.is_identity("rgb")) else None
                
                # Obtener valores consolidados desde panel_stretch
                st = self.panel_stretch.get_state() if hasattr(self, 'panel_stretch') else {}
                contrast_val = float(st.get("contrast_val", 0)) / 100.0

                lut = compute_combined_stf_lut(
                    stretch_mode=st.get("stretch_mode", "mtf"),
                    bp=float(st.get("bp_val", 0.0)),
                    mtf=float(st.get("mtf_val", 0.15)),
                    ghs_sp=float(st.get("ghs_sp_val", 0.02)),
                    ghs_d=float(st.get("ghs_d_val", 5.0)),
                    ghs_b=float(st.get("ghs_b_val", 2.0)),
                    ghs_hp=float(st.get("ghs_hp_val", 0.0)),
                    ghs_lp=float(st.get("ghs_lp_val", 0.0)),
                    ghs_bp=float(st.get("ghs_bp_val", 0.0)),
                    curve_lut_rgb=curve_rgb,
                    contrast=contrast_val
                )

                src_u16 = self._cache_ms_linear_u16
                h, w = src_u16.shape[:2]
                if getattr(self, '_scrub_display_buf', None) is None or self._scrub_display_buf.shape[:2] != (h, w):
                    self._scrub_display_buf = np.empty((h, w, 3), dtype=np.uint8)

                apply_display_lut16(src_u16, lut, self._scrub_display_buf)

                before_u8 = None
                if self.canvas.split_view_enabled:
                    lut_before = compute_combined_stf_lut(
                        stretch_mode=getattr(self, "initial_stretch_mode", "mtf"),
                        bp=getattr(self, "initial_bp", 0.0),
                        mtf=getattr(self, "initial_mtf", 0.1),
                        ghs_sp=getattr(self, "initial_ghs_sp", 0.02),
                        ghs_d=getattr(self, "initial_ghs_d", 5.0),
                        contrast=0.0
                    )
                    if getattr(self, '_scrub_before_buf', None) is None or self._scrub_before_buf.shape[:2] != (h, w):
                        self._scrub_before_buf = np.empty((h, w, 3), dtype=np.uint8)
                    apply_display_lut16(src_u16, lut_before, self._scrub_before_buf)
                    before_u8 = self._scrub_before_buf

                self.canvas.update_display_u8(self._scrub_display_buf, before_u8=before_u8)
                self._update_canvas_info_label()
                return

            base_to_render, active_mask = self._compose_active_base(for_export=False)
            if base_to_render is None:
                return

            if self.view_layer_mode == 2:
                if getattr(self, "stretch_mode", "mtf") == "ghs":
                    stretched = apply_ghs_stretch(
                        base_to_render,
                        sp=self.spin_ghs_sp.value(),
                        d=self.spin_ghs_d.value(),
                        b=self.spin_ghs_b.value(),
                        hp=self.spin_ghs_hp.value(),
                        lp=self.spin_ghs_lp.value(),
                        bp=self.spin_ghs_bp.value(),
                        color_mode=getattr(self, "ghs_color_mode", "luma")
                    )
                else:
                    stretched = manual_stretch(base_to_render, black_point=self.spin_bp.value(), midtone=self.spin_mtf.value())
            else:
                if synchronous and getattr(self, 'async_pipeline', None) is not None:
                    self.async_pipeline.cancel_pending()
                use_async = (not synchronous and not self.is_zoomed_100 and getattr(self, 'async_pipeline', None) is not None)
                stretched = self._apply_pipeline_on_image(
                    base_to_render,
                    precomputed_mask=active_mask,
                    asynchronous=use_async
                )
                if use_async:
                    return

            if stretched is not None:
                self._last_rendered_preview = stretched
                if self.is_zoomed_100 and getattr(self, '_zoom_crop_rect', None) is not None and self.image_32bit is not None:
                    h_f, w_f = self.image_32bit.shape[:2]
                    rx, ry = self.zoom_rel_coords
                    cx = rx * w_f
                    cy = ry * h_f
                    self.canvas.set_external_crop(
                        self._zoom_crop_rect, 
                        (h_f, w_f), 
                        sensor_center=(cx, cy)
                    )
                else:
                    full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None
                    self.canvas.set_external_crop(None, full_shape)
                self.canvas.load_image(base_to_render, display_stretched=stretched)

                # Sincronización del comparador Antes / Después en modo Zoom 1:1
                if self.is_zoomed_100 and self.canvas.split_view_enabled:
                    if getattr(self, "initial_stretch_mode", "mtf") == "ghs":
                        stretched_before = apply_ghs_stretch(
                            base_to_render,
                            sp=getattr(self, "initial_ghs_sp", 0.02),
                            d=getattr(self, "initial_ghs_d", 5.0),
                            b=getattr(self, "initial_ghs_b", 2.0),
                            hp=getattr(self, "initial_ghs_hp", 0.0),
                            lp=getattr(self, "initial_ghs_lp", 0.0),
                            bp=getattr(self, "initial_ghs_bp", 0.0),
                            color_mode=getattr(self, "initial_ghs_color_mode", "luma")
                        )
                    else:
                        stretched_before = manual_stretch(base_to_render, black_point=self.initial_bp, midtone=self.initial_mtf)
                    self.canvas.set_before_image(stretched_before)

                # Actualización fluida y controlada del histograma en vivo
                if not self.is_zoomed_100 and hasattr(self, '_hist_throttle_timer'):
                    if not self._hist_throttle_timer.isActive():
                        self._hist_throttle_timer.start(50)

                self._update_canvas_info_label()

                # Reiniciar temporizador de consolidación de historial (captura scroll de ratón y flechas)
                if (
                    not self._is_undoing_or_redoing
                    and not getattr(self, '_is_applying_preset', False)
                    and not getattr(self, '_is_previewing_preset', False)
                    and hasattr(self, '_undo_debounce_timer')
                ):
                    self._undo_debounce_timer.start(500)
        finally:
            self._is_rendering_preview = False

    def on_canvas_pan_started(self):
        """Inicia el desplazamiento panorámico en Zoom 1:1 deteniendo refinados pendientes."""
        if not self.is_zoomed_100:
            return
        if hasattr(self, '_pan_refine_timer'):
            self._pan_refine_timer.stop()

    def on_canvas_pan_moved(self, dx: int, dy: int):
        """Desplaza el lienzo a 60-144 FPS mediante traslación de pixmap pura sin ningún bloqueo de CPU."""
        if not self.is_zoomed_100 or self.image_32bit is None:
            return
        h_f, w_f = self.image_32bit.shape[:2]

        rx, ry = self.zoom_rel_coords
        cx = rx * w_f
        cy = ry * h_f

        target_cx = cx - dx
        target_cy = cy - dy

        # Delimitación estricta a los límites visibles de la toma
        half_vw = min(w_f / 2.0, self.canvas.width() / 2.0)
        half_vh = min(h_f / 2.0, self.canvas.height() / 2.0)
        clamped_cx = max(half_vw, min(w_f - half_vw, target_cx)) if w_f > 2 * half_vw else (w_f / 2.0)
        clamped_cy = max(half_vh, min(h_f - half_vh, target_cy)) if h_f > 2 * half_vh else (h_f / 2.0)

        # Delta visual real permitido respetando los límites del sensor
        actual_dx = int(round(cx - clamped_cx))
        actual_dy = int(round(cy - clamped_cy))

        if actual_dx != 0 or actual_dy != 0:
            self.canvas.add_pan_visual_offset(actual_dx, actual_dy)
            self.zoom_rel_coords = (clamped_cx / float(w_f), clamped_cy / float(h_f))

        # Durante el arrastre activo continuo NO se ejecuta ningún temporizador de refinado
        if hasattr(self, '_pan_refine_timer') and self._pan_refine_timer.isActive():
            self._pan_refine_timer.stop()

    def on_canvas_pan_finished(self):
        """Finaliza el arrastre. Si el desplazamiento es amplio (>=120 px), programa refinado diferido (500 ms)."""
        if hasattr(self, '_pan_refine_timer'):
            self._pan_refine_timer.stop()

        vox = abs(self.canvas._pan_visual_offset.x()) if hasattr(self.canvas, '_pan_visual_offset') else 0
        voy = abs(self.canvas._pan_visual_offset.y()) if hasattr(self.canvas, '_pan_visual_offset') else 0

        # Si el desplazamiento es leve (<120 px), no recalcular: la vista es 90% nítida y evitamos congelar la UI
        if vox >= 120 or voy >= 120:
            if hasattr(self, '_pan_refine_timer'):
                self._pan_refine_timer.start(500)

    def _refine_pan_crop(self):
        """Genera el nuevo búfer extendido 1:1 centrado en la nueva posición y refresca histogramas."""
        if not self.is_zoomed_100:
            return
        if hasattr(self, '_pan_refine_timer'):
            self._pan_refine_timer.stop()
        self.update_stretch_preview(immediate=True)
        self._refresh_histograms_now()

    def on_canvas_zoom_toggled(self, is_zoomed: bool, rel_x: float, rel_y: float):
        if not is_zoomed:
            if hasattr(self, '_pan_refine_timer'):
                self._pan_refine_timer.stop()
            self.is_zoomed_100 = False
            self.canvas.is_zoomed = False
            self.canvas.zoom_center_img = None
            if hasattr(self, 'btn_toggle_zoom'):
                self.btn_toggle_zoom.blockSignals(True)
                self.btn_toggle_zoom.setChecked(False)
                self.btn_toggle_zoom.blockSignals(False)

            # Restauración 100% instantánea (0 ms) del mapa general si no se tocaron sliders durante el zoom
            if not getattr(self, '_slider_changed_during_zoom', False) and getattr(self, '_fit_view_pixmap', None) is not None:
                mask_overlay = getattr(self, '_fit_view_mask_overlay', None)
                self.canvas.restore_base_pixmap(
                    self._fit_view_pixmap, 
                    orig_rgb=getattr(self, '_fit_view_orig_rgb', None),
                    mask_overlay=mask_overlay
                )
                self._fit_view_pixmap = None
                self._fit_view_orig_rgb = None
                self._fit_view_mask_overlay = None
                self._update_canvas_info_label()
                return

            self.canvas.set_external_crop(None, None)
            self.canvas.setCursor(Qt.ArrowCursor)
            if hasattr(self.canvas, '_zoom_background_pixmap'):
                self.canvas._zoom_background_pixmap = None
            if hasattr(self.canvas, 'reset_pan_visual_offset'):
                self.canvas.reset_pan_visual_offset()
            self._fit_view_pixmap = None
            self._fit_view_orig_rgb = None
            self._fit_view_mask_overlay = None
            self._generate_before_reference()
            self.update_stretch_preview(immediate=True)
            self._update_canvas_info_label()
            return

        # Entrada en Zoom 1:1
        if self.canvas.base_pixmap and not self.canvas.base_pixmap.isNull():
            self._fit_view_pixmap = self.canvas.base_pixmap.copy()
            self._fit_view_orig_rgb = self.canvas.orig_rgb
            self._fit_view_mask_overlay = self.canvas.refined_overlay.copy() if (self.canvas.refined_overlay and not self.canvas.refined_overlay.isNull()) else None
            if getattr(self.canvas, '_zoom_background_pixmap', None) is None:
                self.canvas._zoom_background_pixmap = self.canvas.base_pixmap.copy()
        self._slider_changed_during_zoom = False
        self.is_zoomed_100 = True
        self.canvas.is_zoomed = True
        w_i = float(self.canvas.base_pixmap.width()) if self.canvas.base_pixmap else 1.0
        h_i = float(self.canvas.base_pixmap.height()) if self.canvas.base_pixmap else 1.0
        self.canvas.zoom_center_img = QPoint(int(rel_x * w_i), int(rel_y * h_i))
        self.zoom_rel_coords = (rel_x, rel_y)
        if hasattr(self, 'btn_toggle_zoom'):
            self.btn_toggle_zoom.blockSignals(True)
            self.btn_toggle_zoom.setChecked(True)
            self.btn_toggle_zoom.blockSignals(False)
        self.update_stretch_preview(immediate=True)
        self._update_canvas_info_label()
