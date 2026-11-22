# gui/developer/layer_sync_bridge.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
import numpy as np


class LayerSyncMixin:
    """Mixin para sincronización bidireccional entre MaskLayersWidget, paneles y lienzo."""

    def on_stars_slider_changed(self, val: int):
        """Ajusta la intensidad relativa de la capa de estrellas separada."""
        self.star_intensity = val / 100.0
        self.lbl_stars.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(presence=val)
        self.update_stretch_preview()

    def _on_mask_widget_stars_intensity(self, val: int):
        self.star_intensity = val / 100.0
        if hasattr(self, 'slider_stars') and self.slider_stars.value() != val:
            self.slider_stars.blockSignals(True)
            self.slider_stars.setValue(val)
            self.slider_stars.blockSignals(False)
        if hasattr(self, 'lbl_stars'):
            self.lbl_stars.setText(f"{val}%")
        self.update_stretch_preview()


    # =========================================================================
    # GESTIÓN Y EVENTOS DE DECONVOLUCIÓN ÓPTICA REGULARIZADA (RICHARDSON-LUCY)
    # =========================================================================
    def _on_mask_widget_stars_diff_strength(self, val: int):
        self.star_diff_strength = val / 100.0
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.star_diff_strength = val / 100.0
            if hasattr(self, 'slider_star_diff') and self.slider_star_diff.value() != val:
                self.slider_star_diff.blockSignals(True)
                self.slider_star_diff.setValue(val)
                self.slider_star_diff.blockSignals(False)
        if hasattr(self, 'lbl_star_diff'):
            self.lbl_star_diff.setText(f"{val}%")
        self.update_stretch_preview()

    def _on_mask_widget_stars_diff_radius(self, val: int):
        self.star_diff_radius = float(val)
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.star_diff_radius = float(val)
            if hasattr(self, 'slider_star_diff_radius') and self.slider_star_diff_radius.value() != val:
                self.slider_star_diff_radius.blockSignals(True)
                self.slider_star_diff_radius.setValue(val)
                self.slider_star_diff_radius.blockSignals(False)
        if hasattr(self, 'lbl_star_diff_radius'):
            self.lbl_star_diff_radius.setText(f"{val} px")
        self.update_stretch_preview()

    def _on_mask_widget_stars_diff_threshold(self, val: int):
        self.star_diff_threshold = val / 100.0
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.star_diff_threshold = val / 100.0
            if hasattr(self, 'slider_star_diff_threshold') and self.slider_star_diff_threshold.value() != val:
                self.slider_star_diff_threshold.blockSignals(True)
                self.slider_star_diff_threshold.setValue(val)
                self.slider_star_diff_threshold.blockSignals(False)
        if hasattr(self, 'lbl_star_diff_threshold'):
            self.lbl_star_diff_threshold.setText(f"{val / 100.0:.2f}")
        self.update_stretch_preview()

    def _on_mask_widget_stars_diff_color(self, val: int):
        self.star_diff_color_boost = val / 100.0
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.star_diff_color_boost = val / 100.0
            if hasattr(self, 'slider_star_diff_color') and self.slider_star_diff_color.value() != val:
                self.slider_star_diff_color.blockSignals(True)
                self.slider_star_diff_color.setValue(val)
                self.slider_star_diff_color.blockSignals(False)
        if hasattr(self, 'lbl_star_diff_color'):
            self.lbl_star_diff_color.setText(f"{val}%")
        self.update_stretch_preview()

    def _on_mask_widget_stars_diff_mode(self, mode_key: str):
        self.star_diff_mode = mode_key
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.star_diff_mode = mode_key
        if hasattr(self, 'combo_star_diff_mode'):
            modes = ["screen", "soft_light", "additive"]
            idx = modes.index(mode_key) if mode_key in modes else 0
            if self.combo_star_diff_mode.currentIndex() != idx:
                self.combo_star_diff_mode.blockSignals(True)
                self.combo_star_diff_mode.setCurrentIndex(idx)
                self.combo_star_diff_mode.blockSignals(False)
        self.update_stretch_preview()

    def reset_stars_intensity(self):
        widgets = [
            getattr(self, 'slider_stars', None),
            getattr(self, 'slider_star_diff', None),
            getattr(self, 'slider_star_diff_radius', None),
            getattr(self, 'slider_star_diff_threshold', None),
            getattr(self, 'slider_star_diff_color', None),
            getattr(self, 'combo_star_diff_mode', None),
        ]
        for w in widgets:
            if w is not None:
                w.blockSignals(True)

        if hasattr(self, 'slider_stars'):
            self.slider_stars.setValue(100)
        self.star_intensity = 1.0
        if hasattr(self, 'lbl_stars'):
            self.lbl_stars.setText("100%")

        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.reset_star_diff()
        else:
            if hasattr(self, 'slider_star_diff'):
                self.slider_star_diff.setValue(0)
            if hasattr(self, 'slider_star_diff_radius'):
                self.slider_star_diff_radius.setValue(20)
            if hasattr(self, 'slider_star_diff_threshold'):
                self.slider_star_diff_threshold.setValue(45)
            if hasattr(self, 'slider_star_diff_color'):
                self.slider_star_diff_color.setValue(50)
            if hasattr(self, 'combo_star_diff_mode'):
                self.combo_star_diff_mode.setCurrentIndex(0)

        self.star_diff_strength = 0.0
        self.star_diff_radius = 20.0
        self.star_diff_threshold = 0.45
        self.star_diff_color_boost = 0.50
        self.star_diff_mode = "screen"

        if hasattr(self, 'lbl_star_diff'):
            self.lbl_star_diff.setText("0%")
        if hasattr(self, 'lbl_star_diff_radius'):
            self.lbl_star_diff_radius.setText("20 px")
        if hasattr(self, 'lbl_star_diff_threshold'):
            self.lbl_star_diff_threshold.setText("0.45")
        if hasattr(self, 'lbl_star_diff_color'):
            self.lbl_star_diff_color.setText("50%")

        for w in widgets:
            if w is not None:
                w.blockSignals(False)

        # Invalidar caché de estrellas para forzar recálculo
        self._cache_ms_stars_key = None
        self._cache_ms_stars_img = None
        self._cache_pre_layers_key = None

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(
                presence=100,
                diff_strength=0,
                diff_radius=20,
                diff_threshold=45,
                diff_color=50,
                diff_mode_idx=0
            )
        self.update_stretch_preview(immediate=True)
        self._commit_action()

    def on_layer_mode_changed(self, idx: int):
        self.view_layer_mode = idx
        self.slider_stars.setEnabled(idx == 0)
        if hasattr(self, 'mask_layers_widget'):
            if idx == 0:
                self.mask_layers_widget.set_starnet_stars_enabled(True)
            elif idx == 1:
                self.mask_layers_widget.set_starnet_stars_enabled(False)
        self.update_stretch_preview()

    def _on_mask_layer_starnet_opacity(self, opacity: float):
        """Manejador de cambio de opacidad de la capa de estrellas (desacoplado de la intensidad estelar)."""
        pass

    def _on_mask_layer_starnet_visibility(self, visible: bool):
        """Sincroniza la visibilidad de la capa de estrellas hacia la Sección 5."""
        if hasattr(self, 'combo_layer'):
            self.combo_layer.blockSignals(True)
            new_idx = 0 if visible else 1
            self.combo_layer.setCurrentIndex(new_idx)
            self.view_layer_mode = new_idx
            self.slider_stars.setEnabled(visible)
            self.combo_layer.blockSignals(False)
        self.update_stretch_preview()

    def _on_auto_detect_dust(self, sensitivity: int):
        """Ejecuta la detección automática de manchas de polvo del sensor en el revelador."""
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return

        from core.dust import detect_dust_spots
        spots = detect_dust_spots(
            target,
            sensitivity=float(sensitivity),
            sky_mask=self.current_mask,
        )
        self.canvas.set_dust_spots(spots)
        self.panel_dust.update_spot_count(len(spots))
        if hasattr(self, 'developer_state') and self.developer_state is not None:
            self.developer_state.dust_spots = spots
            self.developer_state.dust_sensitivity = sensitivity
            self.developer_state.dust_mode = self.panel_dust.mode
        self.log_message(f"Detección de polvo: {len(spots)} manchas encontradas y curadas.")
        self._invalidate_pipeline_caches()
        self.update_stretch_preview()
        self._commit_action()

    def _on_visualize_dust_toggled(self, active: bool, threshold: int):
        """Activa o desactiva la previsualización de diagnóstico de paso alto (Visualize Spots)."""
        if hasattr(self, 'developer_state') and self.developer_state is not None:
            self.developer_state.dust_visualize = active
            self.developer_state.dust_visualize_threshold = threshold
        if hasattr(self, 'panel_dust') and self.panel_dust is not None:
            if self.panel_dust.chk_visualize.isChecked() != active:
                self.panel_dust.chk_visualize.blockSignals(True)
                self.panel_dust.chk_visualize.setChecked(active)
                self.panel_dust._visualize = active
                self.panel_dust.chk_visualize.blockSignals(False)
            if self.panel_dust.slider_thresh.value() != threshold:
                self.panel_dust.slider_thresh.blockSignals(True)
                self.panel_dust.slider_thresh.setValue(threshold)
                self.panel_dust._vis_threshold = threshold
                self.panel_dust.slider_thresh.blockSignals(False)
        self.update_stretch_preview(immediate=True)

    def _on_dust_spot_mode_toggled(self, enabled: bool):
        """Activa o desactiva la colocación manual de manchas en el lienzo."""
        self.canvas.set_dust_spot_mode(enabled)

    def _on_clear_dust_spots(self):
        """Elimina todas las manchas registradas y re-renderiza el revelador."""
        self.canvas.clear_dust_spots()
        if hasattr(self, 'developer_state') and self.developer_state is not None:
            self.developer_state.dust_spots = []
        self.panel_dust.update_spot_count(0)
        self.log_message("Manchas de polvo eliminadas.")
        self._invalidate_pipeline_caches()
        self.update_stretch_preview()
        self._commit_action()

    def _on_dust_parameter_changed(self):
        """Sincroniza los parámetros del panel de polvo con DeveloperState."""
        if hasattr(self, 'panel_dust'):
            if hasattr(self, 'developer_state') and self.developer_state is not None:
                self.developer_state.dust_sensitivity = self.panel_dust.sensitivity
                self.developer_state.dust_mode = self.panel_dust.mode
                self.developer_state.dust_visualize = self.panel_dust.visualize
                self.developer_state.dust_visualize_threshold = self.panel_dust.visualize_threshold
        self._invalidate_pipeline_caches()
        self.update_stretch_preview()

    def _on_canvas_dust_spots_changed(self):
        """Responde a la adición, arrastre o borrado de manchas de polvo sobre el lienzo."""
        spots = self.canvas.get_dust_spots()
        if hasattr(self, 'developer_state') and self.developer_state is not None:
            self.developer_state.dust_spots = spots
        if hasattr(self, 'panel_dust'):
            self.panel_dust.update_spot_count(len(spots))
        self._invalidate_pipeline_caches()
        self.update_stretch_preview()
        self._commit_action()

    def _on_mask_widget_sl_clarity(self, val: int):
        if self.slider_clarity.value() != val:
            self.slider_clarity.setValue(val)

    def _on_mask_widget_sl_texture_clarity(self, val: int):
        if self.slider_texture_clarity.value() != val:
            self.slider_texture_clarity.setValue(val)

    def _on_mask_widget_sl_dehaze(self, val: int):
        if self.slider_dehaze.value() != val:
            self.slider_dehaze.setValue(val)

    def _on_mask_widget_sl_ond_est(self, val: int):
        if self.slider_ond_est.value() != val:
            self.slider_ond_est.setValue(val)

    def _on_mask_widget_sl_ond_bg(self, val: int):
        if self.slider_ond_bg.value() != val:
            self.slider_ond_bg.setValue(val)

    def _on_mask_widget_sl_contrast(self, val: int):
        if self.slider_contrast_starless.value() != val:
            self.slider_contrast_starless.setValue(val)

    def _on_mask_widget_sl_denoise(self, val: int):
        if self.slider_denoise.value() != val:
            self.slider_denoise.setValue(val)

    def _on_mask_widget_sl_denoise_color(self, val: int):
        if hasattr(self, 'slider_denoise_color') and self.slider_denoise_color.value() != val:
            self.slider_denoise_color.setValue(val)

    def _on_mask_widget_sl_denoise_mask(self, val: int):
        if hasattr(self, 'slider_denoise_mask') and self.slider_denoise_mask.value() != val:
            self.slider_denoise_mask.setValue(val)

    def _on_toggle_orton_expand(self, checked: bool):
        if hasattr(self, 'w_orton_advanced'):
            self.w_orton_advanced.setVisible(checked)

   
    def _on_mask_widget_sl_orton_strength(self, val: int):
        if hasattr(self, 'slider_orton') and self.slider_orton.value() != val:
            self.slider_orton.setValue(val)

    def _on_mask_widget_sl_orton_radius(self, val: int):
        if hasattr(self, 'slider_orton_radius') and self.slider_orton_radius.value() != val:
            self.slider_orton_radius.setValue(val)

    def _on_mask_widget_sl_orton_threshold(self, val: int):
        if hasattr(self, 'slider_orton_threshold') and self.slider_orton_threshold.value() != val:
            self.slider_orton_threshold.setValue(val)

    def _on_mask_widget_sl_orton_tint(self, tint_key: str):
        if hasattr(self, 'combo_orton_tint'):
            idx = self.combo_orton_tint.findData(tint_key)
            if idx >= 0 and self.combo_orton_tint.currentIndex() != idx:
                self.combo_orton_tint.setCurrentIndex(idx)

    def _on_mask_widget_sl_orton_mode(self, mode_key: str):
        if hasattr(self, 'combo_orton_mode'):
            idx = self.combo_orton_mode.findData(mode_key)
            if idx >= 0 and self.combo_orton_mode.currentIndex() != idx:
                self.combo_orton_mode.setCurrentIndex(idx)

    
    def _on_mask_widget_gnd_ev(self, val: int):
        if self.slider_gnd_ev.value() != val:
            self.slider_gnd_ev.setValue(val)

    def _on_mask_widget_gnd_sh(self, val: int):
        if self.slider_gnd_sh.value() != val:
            self.slider_gnd_sh.setValue(val)

    def _on_mask_widget_gnd_bp(self, val: int):
        if self.slider_gnd_bp.value() != val:
            self.slider_gnd_bp.setValue(val)

    def reset_gnd_tone(self):
        """Restablece los ajustes tonales del suelo delegando en el panel."""
        self.gnd_ev_val = 0.0
        self.gnd_shadows_val = 0.0
        self.gnd_bp_val = 0.0
        self.gnd_antihalo_val = 0.0
        self.gnd_edge_shift_val = 0

        if hasattr(self, 'panel_ground_sky') and self.panel_ground_sky is not None:
            self.panel_ground_sky.set_state({
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0
            })

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(0, 0, 0, 0, 0, 0)
        if hasattr(self, 'spin_mask_shift'):
            self.spin_mask_shift.setValue(0)

        self.update_stretch_preview()
        self._commit_action()
    
