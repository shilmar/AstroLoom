"""Headless rendering pipeline and multi-resolution pyramid engine for AstroLoom Developer."""
from __future__ import annotations

from typing import Optional, Dict, Tuple, List, Any, Union
import numpy as np
import cv2

from core.stretch import (
    apply_white_balance,
    adjust_vibrance,
    apply_ghs_stretch,
    manual_stretch,
    apply_curve_lut,
    apply_hue_vs_sat_curve,
    apply_luma_vs_sat_curve,
    adjust_contrast,
)
from core.light_pollution import LP_PRESET_TINTS, apply_parametric_light_pollution
from core.deconvolution import (
    generate_gaussian_psf,
    generate_moffat_psf,
    richardson_lucy_deconvolve,
)
from core.orton import apply_astronomical_orton
from core.star_diffusion import apply_star_diffusion
from core.mask_layers import apply_all_mask_layers
from gui.developer.developer_state import DeveloperState


class ResolutionPyramid:
    """Multi-resolution image pyramid for high-performance interactive preview rendering.

    Provides multi-scale mipmap representations:
    - Level 0: 100% scale (Native full resolution)
    - Level 1: Preview scale (max_dim <= 1600 px) for crisp interactive rendering on 1080p/1440p viewports.
    - Level 2: Draft scale (max_dim <= 800 px) for ultra-fast 60fps real-time slider scrubbing.
    - Level 3: Thumbnail scale (max_dim <= 400 px) for thumbnails & histograms.
    """

    def __init__(self, full_res_image: np.ndarray, max_preview_dim: int = 1600, max_levels: int = 4):
        if full_res_image is None:
            raise ValueError("full_res_image cannot be None")
        self.levels: List[np.ndarray] = [full_res_image]

        h, w = full_res_image.shape[:2]
        max_dim = max(h, w)

        # Level 1: Preview (clamped to max_preview_dim)
        if max_dim > max_preview_dim:
            scale_preview = max_preview_dim / float(max_dim)
            pw = (int(w * scale_preview) // 4) * 4
            ph = (int(h * scale_preview) // 4) * 4
            lvl1 = cv2.resize(full_res_image, (pw, ph), interpolation=cv2.INTER_AREA)
            self.levels.append(np.ascontiguousarray(lvl1, dtype=np.float32))
        else:
            self.levels.append(full_res_image)

        # Level 2: Draft (calibrado a max 1000 px para arrastre a 60 FPS en 4K)
        lvl1 = self.levels[1]
        h1, w1 = lvl1.shape[:2]
        max_d1 = max(h1, w1)
        max_draft_dim = 1000
        if max_d1 > max_draft_dim:
            scale_draft = max_draft_dim / float(max_d1)
            dw = (int(w1 * scale_draft) // 4) * 4
            dh = (int(h1 * scale_draft) // 4) * 4
            lvl2 = cv2.resize(lvl1, (dw, dh), interpolation=cv2.INTER_AREA)
            self.levels.append(np.ascontiguousarray(lvl2, dtype=np.float32))
        else:
            self.levels.append(lvl1)

        # Level 3+: Halved downsampling from Level 2
        current = self.levels[2]
        for _ in range(3, max_levels):
            ch, cw = current.shape[:2]
            if cw <= 200 or ch <= 150:
                break
            nw = (cw // 2 // 2) * 2
            nh = (ch // 2 // 2) * 2
            if nw == 0 or nh == 0:
                break
            current = cv2.resize(current, (nw, nh), interpolation=cv2.INTER_AREA)
            self.levels.append(np.ascontiguousarray(current, dtype=np.float32))

    def get_level(self, level: int) -> np.ndarray:
        """Retrieve image at specific mipmap level."""
        idx = max(0, min(level, len(self.levels) - 1))
        return self.levels[idx]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def full_res(self) -> np.ndarray:
        return self.levels[0]

    @property
    def preview(self) -> np.ndarray:
        """Level 1 (max_dim <= 1600 px) for crisp interactive rendering on 1080p/1440p viewports."""
        return self.levels[1] if len(self.levels) > 1 else self.levels[0]

    @property
    def draft(self) -> np.ndarray:
        """Level 2 (max_dim <= 800 px) for ultra-fast 60fps real-time dragging of heavy filters."""
        return self.levels[2] if len(self.levels) > 2 else self.levels[-1]

    @property
    def thumb(self) -> np.ndarray:
        """Lowest mipmap level for fast thumbnail and histogram calculations."""
        return self.levels[-1]


def get_psf_for_state(state: DeveloperState) -> np.ndarray:
    """Generate the optical PSF kernel corresponding to state parameters."""
    fwhm = state.deconv_fwhm / 10.0
    aspect = state.deconv_aspect / 100.0
    angle = float(state.deconv_angle)
    models = ["gaussian", "moffat"]
    model = models[state.deconv_model_idx] if 0 <= state.deconv_model_idx < len(models) else "gaussian"

    if model == "gaussian":
        return generate_gaussian_psf(fwhm=fwhm, aspect_ratio=aspect, angle_deg=angle)
    else:
        return generate_moffat_psf(fwhm=fwhm, beta=4.0, aspect_ratio=aspect, angle_deg=angle)


def apply_developer_pipeline(
    target_img: np.ndarray,
    state: DeveloperState,
    precomputed_mask: Optional[np.ndarray] = None,
    image_stars: Optional[np.ndarray] = None,
    image_starless: Optional[np.ndarray] = None,
    mask_layers: Optional[list] = None,
    curve_luts: Optional[Dict[str, np.ndarray]] = None,
    psf_kernel: Optional[np.ndarray] = None,
    is_preview: bool = True,
    is_scrubbing: bool = False,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_shape: Optional[Tuple[int, int]] = None,
    return_intermediate: bool = False,
    precomputed_stretched_base: Optional[np.ndarray] = None,
    precomputed_stretched_stars: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Pure Python headless non-destructive photo development pipeline.

    Processes target_img through all 14 astronomical develop stages based on state.
    Executes headlessly with zero Qt dependencies.
    """
    if target_img is None:
        return None

    if precomputed_stretched_base is not None:
        stretched = precomputed_stretched_base
        st_stretched = precomputed_stretched_stars
    else:
        # 1. Balance de blancos
        temp_val = state.temp_val / 1000.0
        tint_val = state.tint_val / 1000.0
        if abs(temp_val) > 1e-5 or abs(tint_val) > 1e-5:
            img = apply_white_balance(target_img, temp_val, tint_val)
        else:
            img = target_img.copy()

        # 2. Saturación diferencial cielo / suelo
        sat_sky_val = state.sat_sky_val / 100.0
        sat_gnd_val = state.sat_gnd_val / 100.0
        if sat_sky_val != 1.0 or sat_gnd_val != 1.0:
            hsv = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2HSV)
            if precomputed_mask is None or sat_sky_val == sat_gnd_val:
                hsv[..., 1] = np.clip(hsv[..., 1] * sat_sky_val, 0.0, 1.0)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            else:
                hsv_sky = hsv.copy()
                hsv_sky[..., 1] = np.clip(hsv_sky[..., 1] * sat_sky_val, 0.0, 1.0)
                rgb_sky = cv2.cvtColor(hsv_sky, cv2.COLOR_HSV2RGB)

                hsv_gnd = hsv.copy()
                hsv_gnd[..., 1] = np.clip(hsv_gnd[..., 1] * sat_gnd_val, 0.0, 1.0)
                rgb_gnd = cv2.cvtColor(hsv_gnd, cv2.COLOR_HSV2RGB)

                img = (rgb_sky * precomputed_mask) + (rgb_gnd * (1.0 - precomputed_mask))

        # 3. Intensidad / Vibrance
        vibrance_val = state.vibrance_val / 100.0
        if abs(vibrance_val) > 1e-4:
            img = adjust_vibrance(img, vibrance_val)

        # 4. Reducción Paramétrica 2D de Cúpula de Contaminación Lumínica
        lp_strength = state.lp_val / 100.0
        if lp_strength > 1e-4:
            mask_2d = precomputed_mask[..., 0] if precomputed_mask is not None else None
            lp_tints = ["sodium", "mercury", "led", "skyglow", "custom"]
            tint_key = lp_tints[state.lp_tint_idx] if 0 <= state.lp_tint_idx < len(lp_tints) else "sodium"
            tint_rgb = LP_PRESET_TINTS.get(tint_key, state.lp_custom_tint)
            lp_modes = ["attenuate", "subtractive"]
            lp_mode = lp_modes[state.lp_mode_idx] if 0 <= state.lp_mode_idx < len(lp_modes) else "attenuate"

            img = apply_parametric_light_pollution(
                img,
                strength=lp_strength,
                center_x=state.lp_cx,
                center_y=state.lp_cy,
                width=state.lp_w,
                height=state.lp_h,
                tint_rgb=tint_rgb,
                tint_strength=state.lp_tint_strength / 100.0,
                sky_mask=mask_2d,
                mode=lp_mode,
                legacy_1d=state.lp_legacy_1d,
            )

        # 5. Ajuste Tonal Diferencial para el Suelo
        if precomputed_mask is not None:
            gnd_ev = state.gnd_ev_val / 100.0
            gnd_shadows = state.gnd_sh_val / 100.0
            gnd_bp = -(state.gnd_bp_val / 1000.0)
            has_gnd_tone = abs(gnd_ev) > 1e-4 or gnd_shadows > 1e-4 or abs(gnd_bp) > 1e-4
            if has_gnd_tone:
                gnd_part = img.copy()
                if abs(gnd_ev) > 1e-4:
                    gnd_part = gnd_part * (2.0 ** gnd_ev)
                if gnd_shadows > 1e-4:
                    lift_curve = (1.0 - np.clip(gnd_part, 0.0, 1.0)) ** 2
                    gnd_part = gnd_part * (1.0 + gnd_shadows * lift_curve)
                if abs(gnd_bp) > 1e-4:
                    if gnd_bp < 0.0:
                        gnd_part = np.clip(gnd_part - gnd_bp, 0.0, 1.0)
                    else:
                        gnd_part = np.clip(
                            (gnd_part - gnd_bp) / max(1e-4, 1.0 - gnd_bp), 0.0, 1.0
                        )
                img = (img * precomputed_mask) + (gnd_part * (1.0 - precomputed_mask))

        # 6. Estirado Tonal (MTF o GHS)
        if state.stretch_mode == "ghs":
            stretched = apply_ghs_stretch(
                img,
                sp=state.ghs_sp_val,
                d=state.ghs_d_val,
                b=state.ghs_b_val,
                hp=state.ghs_hp_val,
                lp=state.ghs_lp_val,
                bp=state.ghs_bp_val,
                color_mode=state.ghs_color_mode,
            )
        else:
            stretched = manual_stretch(img, black_point=state.bp_val, midtone=state.mtf_val)

        # 7. Curvas Tonales (LUT RGB, Hue vs Sat, Luma vs Sat)
        if curve_luts:
            if "rgb" in curve_luts and curve_luts["rgb"] is not None:
                curved = apply_curve_lut(stretched, curve_luts["rgb"])
                if precomputed_mask is not None:
                    stretched = (curved * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
                else:
                    stretched = curved
            if "hue_sat" in curve_luts and curve_luts["hue_sat"] is not None:
                hsv_curved = apply_hue_vs_sat_curve(stretched, curve_luts["hue_sat"])
                if precomputed_mask is not None:
                    stretched = (hsv_curved * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
                else:
                    stretched = hsv_curved
            if "luma_sat" in curve_luts and curve_luts["luma_sat"] is not None:
                luma_curved = apply_luma_vs_sat_curve(stretched, curve_luts["luma_sat"])
                if precomputed_mask is not None:
                    stretched = (luma_curved * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
                else:
                    stretched = luma_curved

        # 8. Contraste específico del Fondo (Starless)
        contrast_starless = state.contrast_starless_val / 100.0
        if abs(contrast_starless) > 1e-4:
            if precomputed_mask is not None and np.any(precomputed_mask > 0.5):
                sky_median = float(np.median(stretched[precomputed_mask > 0.5]))
            else:
                sky_median = float(np.median(stretched))
            sn_contrasted = adjust_contrast(stretched, contrast_starless, pivot=sky_median)
            if precomputed_mask is not None:
                stretched = (sn_contrasted * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
            else:
                stretched = sn_contrasted

        # 9. Contraste Global
        contrast_global = state.contrast_val / 100.0
        if abs(contrast_global) > 1e-4:
            stretched = adjust_contrast(stretched, contrast_global, pivot=None)

        # 10. Deconvolución Óptica Regularizada (Richardson-Lucy / PSF)
        if state.deconv_iterations > 0:
            is_draft_deconv = is_scrubbing
            orig_h_dec, orig_w_dec = stretched.shape[:2]
            if is_draft_deconv and max(orig_w_dec, orig_h_dec) > 800:
                scale_dec = 800.0 / float(max(orig_w_dec, orig_h_dec))
                dw_dec = max(32, int(orig_w_dec * scale_dec))
                dh_dec = max(32, int(orig_h_dec * scale_dec))
                in_deconv = cv2.resize(stretched, (dw_dec, dh_dec), interpolation=cv2.INTER_AREA)
                in_mask_dec = (
                    cv2.resize(precomputed_mask, (dw_dec, dh_dec), interpolation=cv2.INTER_AREA)
                    if precomputed_mask is not None
                    else None
                )
            else:
                in_deconv = stretched
                in_mask_dec = precomputed_mask

            if psf_kernel is None:
                psf_kernel = get_psf_for_state(state)

            active_psf = psf_kernel
            if is_draft_deconv:
                k_w = max(3, (active_psf.shape[1] // 2) | 1)
                k_h = max(3, (active_psf.shape[0] // 2) | 1)
                active_psf = cv2.resize(active_psf, (k_w, k_h), interpolation=cv2.INTER_AREA)
                active_psf = active_psf / max(1e-7, float(np.sum(active_psf)))

            deconv_scopes = ["full", "stars_only", "background_only"]
            deconv_scope = (
                deconv_scopes[state.deconv_scope_idx]
                if 0 <= state.deconv_scope_idx < len(deconv_scopes)
                else "full"
            )

            star_mask = None
            if deconv_scope in ["stars_only", "background_only"]:
                if image_stars is not None:
                    st_ref = image_stars
                    if is_draft_deconv and st_ref.shape[:2] != (dh_dec, dw_dec):
                        st_ref_scaled = cv2.resize(st_ref, (dw_dec, dh_dec), interpolation=cv2.INTER_AREA)
                    else:
                        st_ref_scaled = st_ref
                    star_mask = (np.mean(st_ref_scaled, axis=2) > 0.05).astype(np.float32)
                if star_mask is None:
                    gray_s = (
                        cv2.cvtColor(in_deconv, cv2.COLOR_RGB2GRAY)
                        if (in_deconv.ndim == 3 and in_deconv.shape[2] == 3)
                        else in_deconv
                    )
                    k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    tophat = cv2.morphologyEx(gray_s, cv2.MORPH_TOPHAT, k_th)
                    star_mask = (tophat > 0.02).astype(np.float32)

            deconvolved = richardson_lucy_deconvolve(
                in_deconv,
                psf_kernel=active_psf,
                iterations=state.deconv_iterations,
                deringing=state.deconv_deringing / 100.0,
                damping=state.deconv_damping / 100.0,
                luminance_only=state.deconv_luma_only,
                star_mask=star_mask,
                target_scope=deconv_scope,
            )
            if in_mask_dec is not None:
                res_deconv = (deconvolved * in_mask_dec) + (in_deconv * (1.0 - in_mask_dec))
            else:
                res_deconv = deconvolved

            if is_draft_deconv and in_deconv.shape[:2] != (orig_h_dec, orig_w_dec):
                stretched = cv2.resize(res_deconv, (orig_w_dec, orig_h_dec), interpolation=cv2.INTER_LINEAR)
            else:
                stretched = res_deconv

        # 11. Efecto Orton y Resplandor de Polvo Galáctico
        has_active_starless = (
            image_starless is not None and mask_layers is not None and len(mask_layers) > 0
        )
        orton_strength = state.orton_strength / 100.0
        if orton_strength > 1e-4 and not has_active_starless:
            is_draft_ort = is_scrubbing
            orig_h_ort, orig_w_ort = stretched.shape[:2]
            if is_draft_ort and max(orig_w_ort, orig_h_ort) > 800:
                scale_ort = 800.0 / float(max(orig_w_ort, orig_h_ort))
                dw_ort = max(32, int(orig_w_ort * scale_ort))
                dh_ort = max(32, int(orig_h_ort * scale_ort))
                in_ort = cv2.resize(stretched, (dw_ort, dh_ort), interpolation=cv2.INTER_AREA)
                in_mask_ort = (
                    cv2.resize(precomputed_mask, (dw_ort, dh_ort), interpolation=cv2.INTER_AREA)
                    if precomputed_mask is not None
                    else None
                )
            else:
                in_ort = stretched
                in_mask_ort = precomputed_mask

            mask_2d = in_mask_ort[..., 0] if in_mask_ort is not None else None
            orton_tints = ["warm_gold", "neutral", "cool_blue", "halpha"]
            orton_tint = (
                orton_tints[state.orton_tint_idx]
                if 0 <= state.orton_tint_idx < len(orton_tints)
                else "warm_gold"
            )
            orton_modes = ["soft_light", "screen", "additive"]
            orton_mode = (
                orton_modes[state.orton_mode_idx]
                if 0 <= state.orton_mode_idx < len(orton_modes)
                else "soft_light"
            )

            res_ort = apply_astronomical_orton(
                in_ort,
                strength=orton_strength,
                radius=float(state.orton_radius) * (0.5 if is_draft_ort else 1.0),
                threshold=state.orton_threshold / 100.0,
                exponent=1.0,
                tint_type=orton_tint,
                custom_tint=(1.0, 0.85, 0.55),
                tint_strength=0.5,
                blend_mode=orton_mode,
                sky_mask=mask_2d,
                is_starless=False,
            )
            if is_draft_ort and in_ort.shape[:2] != (orig_h_ort, orig_w_ort):
                stretched = cv2.resize(res_ort, (orig_w_ort, orig_h_ort), interpolation=cv2.INTER_LINEAR)
            else:
                stretched = res_ort

        # 12. Filtro Difusor y Realce de Constelaciones
        star_diff_strength = state.star_diff_strength / 100.0
        sd_modes = ["screen", "soft_light", "additive"]
        sd_mode = (
            sd_modes[state.star_diff_mode_idx]
            if 0 <= state.star_diff_mode_idx < len(sd_modes)
            else "screen"
        )

        if star_diff_strength > 1e-4 and not has_active_starless:
            is_draft_sd = is_scrubbing
            orig_h_sd, orig_w_sd = stretched.shape[:2]
            if is_draft_sd and max(orig_w_sd, orig_h_sd) > 800:
                scale_sd = 800.0 / float(max(orig_w_sd, orig_h_sd))
                dw_sd = max(32, int(orig_w_sd * scale_sd))
                dh_sd = max(32, int(orig_h_sd * scale_sd))
                in_sd = cv2.resize(stretched, (dw_sd, dh_sd), interpolation=cv2.INTER_AREA)
                in_mask_sd = (
                    cv2.resize(precomputed_mask, (dw_sd, dh_sd), interpolation=cv2.INTER_AREA)
                    if precomputed_mask is not None
                    else None
                )
            else:
                in_sd = stretched
                in_mask_sd = precomputed_mask

            mask_2d = in_mask_sd[..., 0] if in_mask_sd is not None else None
            res_sd = apply_star_diffusion(
                in_sd,
                strength=star_diff_strength,
                radius=float(state.star_diff_radius) * (0.5 if is_draft_sd else 1.0),
                threshold=state.star_diff_threshold / 100.0,
                color_boost=state.star_diff_color / 100.0,
                halo_mode=sd_mode,
                sky_mask=mask_2d,
                stars_layer=None,
            )
            if is_draft_sd and in_sd.shape[:2] != (orig_h_sd, orig_w_sd):
                stretched = cv2.resize(res_sd, (orig_w_sd, orig_h_sd), interpolation=cv2.INTER_LINEAR)
            else:
                stretched = res_sd

        # 13. Preparación de la capa de estrellas para composición
        star_intensity = state.stars_val / 100.0
        if image_stars is not None:
            if mask_layers and len(mask_layers) > 0:
                if state.stretch_mode == "ghs":
                    st_stretched = np.clip(
                        apply_ghs_stretch(
                            image_stars,
                            sp=state.ghs_sp_val,
                            d=state.ghs_d_val,
                            b=state.ghs_b_val,
                            hp=state.ghs_hp_val,
                            lp=state.ghs_lp_val,
                            bp=0.0,
                            color_mode="rgb",
                        )
                        * star_intensity,
                        0.0,
                        1.0,
                    )
                else:
                    st_stretched = np.clip(
                        manual_stretch(image_stars, black_point=0.0, midtone=state.mtf_val)
                        * star_intensity,
                        0.0,
                        1.0,
                    )
                if star_diff_strength > 1e-4:
                    mask_2d = precomputed_mask[..., 0] if precomputed_mask is not None else None
                    st_stretched = apply_star_diffusion(
                        st_stretched,
                        strength=star_diff_strength,
                        radius=float(state.star_diff_radius),
                        threshold=state.star_diff_threshold / 100.0,
                        color_boost=state.star_diff_color / 100.0,
                        halo_mode=sd_mode,
                        sky_mask=mask_2d,
                        stars_layer=st_stretched,
                    )
            else:
                st_stretched = np.clip(image_stars * star_intensity, 0.0, 1.0)
                if star_diff_strength > 1e-4:
                    mask_2d = precomputed_mask[..., 0] if precomputed_mask is not None else None
                    st_stretched = apply_star_diffusion(
                        st_stretched,
                        strength=star_diff_strength,
                        radius=float(state.star_diff_radius),
                        threshold=state.star_diff_threshold / 100.0,
                        color_boost=state.star_diff_color / 100.0,
                        halo_mode=sd_mode,
                        sky_mask=mask_2d,
                        stars_layer=st_stretched,
                    )
        else:
            st_stretched = None

    pre_layers_base = stretched
    pre_layers_stars = st_stretched

    # 14. Aplicación de Capas y Máscaras Locales
    if mask_layers and len(mask_layers) > 0:
        stretched = apply_all_mask_layers(
            stretched,
            mask_layers,
            master_sky_mask=precomputed_mask,
            starless_image=image_starless,
            stars_image=st_stretched,
            crop_rect=crop_rect,
            full_image_shape=full_shape,
        )

    final_result = np.ascontiguousarray(stretched, dtype=np.float32)
    if return_intermediate:
        return final_result, pre_layers_base, pre_layers_stars
    return final_result
