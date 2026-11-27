# gui/developer/history_presets.py
from __future__ import annotations
import os
import gc
from typing import Optional, Dict, Any, List
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QMessageBox, QDialog

from core.log_hub import log_hub
from core.developer_presets import (
    DeveloperPreset, get_builtin_presets, load_user_presets,
    SCOPE_ALL, SCOPE_TONE, SCOPE_COLOR, SCOPE_DETAILS
)
from gui.dialog_presets import SavePresetDialog, PresetManagerDialog
from gui.icons import get_icon
from core.mask_layers import MASK_TYPE_IMAGE_RASTER


class HistoryPresetsMixin:
    """Mixin que gestiona el historial Undo/Redo, serialización de estado y presets de revelado."""
    MAX_UNDO_STEPS = 50

    def reset_wb(self):
        """Restablece el balance de blancos delegando en el panel de color."""
        if hasattr(self, 'panel_color_wb') and self.panel_color_wb is not None:
            self.panel_color_wb.reset_wb()

    def reset_saturation(self):
        """Restablece la saturación de cielo y suelo delegando en el panel de color."""
        if hasattr(self, 'panel_color_wb') and self.panel_color_wb is not None:
            self.panel_color_wb.reset_saturation()

    def reset_deconv(self):
        """Restablece los controles de deconvolución delegando en el panel."""
        if hasattr(self, 'panel_deconv') and self.panel_deconv is not None:
            self.panel_deconv.reset_deconv()

        if hasattr(self, 'panel_sharpening') and self.panel_sharpening is not None:
            self.panel_sharpening.reset_sharpening()

    def reset_sharpening(self):
        """Restablece los controles de máscara de enfoque delegando en el panel."""
        if hasattr(self, 'panel_sharpening') and self.panel_sharpening is not None:
            self.panel_sharpening.reset_sharpening()

    def reset_gradients(self):
        """Restablece los controles de contaminación lumínica delegando en el panel."""
        if hasattr(self, 'panel_lp') and self.panel_lp is not None:
            self.panel_lp.reset_gradients()

    def reset_ghs(self):
        """Restablece los parámetros de estirado hiperbólico GHS delegando en el panel."""
        if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
            self.panel_stretch.reset_ghs()

    def reset_starless_adjustments(self):
        """Restablece las herramientas de ajuste de fondo starless y efectos."""
        if hasattr(self, 'panel_starnet') and self.panel_starnet is not None:
            self.panel_starnet.set_state({
                "clarity_val": 0, "texture_clarity_val": 0, "dehaze_val": 0, "ond_est_val": 0, "ond_bg_val": 0,
                "contrast_starless_val": 0, "denoise_val": 0, "denoise_color_val": 50, "denoise_mask_val": 50
            })
        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget is not None:
            self.mask_layers_widget.set_starless_tone_values(
                0, 0, 0, 0, 0, 0,
                orton_strength=0, orton_radius=40, orton_threshold=5,
                orton_tint_idx=0, orton_mode_idx=0,
                denoise_color=50, denoise_mask=50,
                texture_clarity=0
            )
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.reset_orton()
            self.panel_effects.reset_star_diff()
        self.update_stretch_preview()
        self._commit_action()

    def reset_all_parameters(self):
        """Devuelve todos los controles y variables del revelador a su estado neutro original delegando en los paneles."""
        # 1. Delegar reseteo en cada panel modular
        if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
            self.panel_stretch.reset_sliders()
            
        if hasattr(self, 'panel_lp') and self.panel_lp is not None:
            self.panel_lp.reset_gradients()
            
        if hasattr(self, 'panel_deconv') and self.panel_deconv is not None:
            self.panel_deconv.reset_deconv()

        if hasattr(self, 'panel_sharpening') and self.panel_sharpening is not None:
            self.panel_sharpening.reset_sharpening()
            
        if hasattr(self, 'panel_color_wb') and self.panel_color_wb is not None:
            self.panel_color_wb.reset_all()
            
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
            self.panel_effects.reset_orton()
            self.panel_effects.reset_star_diff()

        if hasattr(self, 'panel_starnet') and self.panel_starnet is not None:
            self.panel_starnet.set_state({
                "layer_idx": 0,
                "stars_val": 100,
                "contrast_starless_val": 0,
                "clarity_val": 0,
                "texture_clarity_val": 0,
                "dehaze_val": 0,
                "ond_est_val": 0,
                "ond_bg_val": 0,
                "denoise_val": 0,
                "denoise_color_val": 50,
                "denoise_mask_val": 50
            })
            if hasattr(self.panel_starnet, 'set_ai_denoise_active'):
                self.panel_starnet.set_ai_denoise_active(False)
            
        if hasattr(self, 'panel_ground_sky') and self.panel_ground_sky is not None:
            self.panel_ground_sky.set_state({
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "gnd_dn_val": 0
            })

        # Limpiar variables internas de Suelo y StarNet AI
        self.gnd_ev_val = 0.0
        self.gnd_shadows_val = 0.0
        self.gnd_bp_val = 0.0
        self.gnd_antihalo_val = 0.0
        self.gnd_edge_shift_val = 0
        self.gnd_denoise_val = 0.0
        self.gnd_ai_denoise_enabled = False
        self._cache_ground_ai = None
        if getattr(self, 'ground_ai_denoise_worker', None) and self.ground_ai_denoise_worker.isRunning():
            self.ground_ai_denoise_worker.cancel()
        self.ground_ai_denoise_worker = None

        self.starless_ai_denoise_enabled = False
        self._cache_starless_ai = None
        if getattr(self, 'ai_denoise_worker', None) and self.ai_denoise_worker.isRunning():
            self.ai_denoise_worker.cancel()
        self.ai_denoise_worker = None

        # 2. Resetear curvas y capas
        if hasattr(self, 'curve_widget') and self.curve_widget is not None:
            self.curve_widget.reset_all_curves()

        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget is not None:
            self.mask_layers_widget.set_state([])
            if hasattr(self.mask_layers_widget, 'set_ground_ai_denoise_active'):
                self.mask_layers_widget.set_ground_ai_denoise_active(False)
            if hasattr(self.mask_layers_widget, 'set_ai_denoise_active'):
                self.mask_layers_widget.set_ai_denoise_active(False)
            if hasattr(self.mask_layers_widget, 'set_ground_tone_values'):
                self.mask_layers_widget.set_ground_tone_values(0, 0, 0, 0)
            if hasattr(self.mask_layers_widget, 'slider_gnd_special_dn'):
                self.mask_layers_widget.slider_gnd_special_dn.setValue(0)
            if hasattr(self.mask_layers_widget, 'slider_sl_denoise'):
                self.mask_layers_widget.slider_sl_denoise.setValue(0)

        # 3. Restablecer zoom e invalidar cachés
        self.is_zoomed_100 = False
        self.zoom_rel_coords = (0.5, 0.5)
        self._zoom_crop_rect = None
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.is_zoomed = False
            self.canvas.set_external_crop(None, None)
            self.canvas.reset_crop_rect()
            if self.canvas.crop_mode:
                self.canvas.set_crop_mode(False)
            self.canvas.set_layer_brush_enabled(False)
            self.canvas.set_active_layer_mask(None)

        if hasattr(self, 'sec_presets') and self.sec_presets is not None:
            self.sec_presets.set_status_badge("")

        self._invalidate_pipeline_caches()

        if hasattr(self, '_update_mask_status_ui'):
            self._update_mask_status_ui()
            
    def clear_session(self):
        """Descarga por completo la sesión del revelador y fuerza recolección de memoria."""
        self.reset_all_parameters()

        self.image_32bit = None
        self.image_starless = None
        self.raw_linear_starless = None
        self.starless_is_stretched = False
        self.image_stars = None

        self.preview_proxy = None
        self.draft_proxy = None
        self.proxy_starless = None
        self.draft_starless = None
        self.proxy_stars = None
        self.draft_stars = None
        self.proxy_mask = None
        self.draft_mask = None

        self.current_mask = None
        self.active_filepath = None
        self.current_project_path: Optional[str] = None
        self._is_dirty: bool = False
        self._cache_ground_ai = None
        self._cache_starless_ai = None
        self.gnd_ai_denoise_enabled = False
        self.starless_ai_denoise_enabled = False
        self.gnd_denoise_val = 0.0
        self._zoom_crop_rect = None
        self.is_zoomed_100 = False

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_state([])

        self.has_custom_snapshot = False
        self.snapshot_before_pixmap = None
        self.initial_bp = 0.0
        self.initial_mtf = 0.1

        if hasattr(self, 'undo_stack'):
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._last_committed_state = None
            self._update_undo_redo_buttons()

        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.set_native_image_size(None)
            self.canvas.reset_crop_rect()
            self.canvas.set_external_crop(None, None)
            if self.canvas.crop_mode:
                self.canvas.set_crop_mode(False)
            self.canvas.set_layer_brush_enabled(False)
            self.canvas.set_active_layer_mask(None)
            self.canvas.clear_image()
            self.canvas.set_refined_mask(None)
            self.canvas.set_mask_visible(False)
            self.canvas.orig_rgb = None
            self.canvas.base_pixmap = None
            self.canvas.before_pixmap = None
            self.canvas.split_view_enabled = False
            self.canvas.split_orientation = "vertical"
            self.canvas.display_stretched = None
            if hasattr(self.canvas, 'scribble_pixmap') and self.canvas.scribble_pixmap is not None:
                self.canvas.scribble_pixmap.fill(Qt.transparent)
            if hasattr(self.canvas, 'refined_overlay'):
                self.canvas.refined_overlay = None
            self.canvas.update()

        if hasattr(self, 'btn_crop'):
            self.btn_crop.blockSignals(True)
            self.btn_crop.setChecked(False)
            self.btn_crop.blockSignals(False)
        if hasattr(self, 'crop_toolbar'):
            self.crop_toolbar.setVisible(False)

        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.blockSignals(True)
            self.btn_split_view.setChecked(False)
            self.btn_split_view.blockSignals(False)
        self._update_split_orientation_ui()
        if hasattr(self, 'btn_toggle_clipping'):
            self.btn_toggle_clipping.blockSignals(True)
            self.btn_toggle_clipping.setChecked(False)
            self.btn_toggle_clipping.blockSignals(False)
        if hasattr(self, 'btn_toggle_zoom'):
            self.btn_toggle_zoom.blockSignals(True)
            self.btn_toggle_zoom.setChecked(False)
            self.btn_toggle_zoom.blockSignals(False)

        self._update_canvas_info_label()
        log_hub.log("Revelador", "--- Sesión restablecida ---")
        gc.collect()
        
    def _set_starnet_tools_enabled(self, enabled: bool):
        """Habilita o deshabilita individualmente todos los controles que dependen de StarNet."""
        if hasattr(self, "sec_starnet"):
            if enabled:
                self.sec_starnet.set_status_badge("● Separadas", "#8B5CF6")
            else:
                self.sec_starnet.set_status_badge("")

        from core.i18n import tr
        if hasattr(self, "btn_starnet") and self.btn_starnet is not None:
            if enabled:
                self.btn_starnet.setText(tr("developer.starnet.btn_reseparate", "Volver a separar estrellas (Sobrescribir)"))
                self.btn_starnet.setToolTip("StarNet ya ha sido ejecutado. Pulsa para volver a desacoplar las estrellas con nuevos parámetros.")
            else:
                self.btn_starnet.setText(tr("developer.starnet.btn_separate", "Separar Estrellas con StarNet"))
                self.btn_starnet.setToolTip("Ejecuta StarNet AI para desacoplar estrellas del fondo cósmico.")

        if hasattr(self, "btn_go_to_masks_starnet"):
            self.btn_go_to_masks_starnet.setVisible(enabled)

        if hasattr(self, "lbl_starnet_info"):
            if enabled:
                self.lbl_starnet_info.setText(
                    "Estrellas separadas correctamente.\n"
                    "Las capas 'Fondo Nebular' y 'Estrellas' están activas en el Gestor de Capas (Sección 9). "
                    "Selecciona cada capa allí para regular su presencia estelar, claridad, neblina, contraste y reducción de ruido."
                )
            else:
                self.lbl_starnet_info.setText(
                    "Ejecuta StarNet para desacoplar las estrellas del fondo cósmico.\n"
                    "Se generarán automáticamente las capas 'Fondo Nebular' y 'Estrellas' en el Gestor de Capas (Sección 9), donde podrás ajustar de forma contextual su presencia, claridad, contraste y reducción de ruido."
                )

        if hasattr(self, "container_starnet_tools"):
            self.container_starnet_tools.setEnabled(enabled)

        # Activación forzada de cada control hijo
        controls = [
            getattr(self, "combo_layer", None),
            getattr(self, "slider_stars", None),
            getattr(self, "slider_contrast_starless", None),
            getattr(self, "slider_clarity", None),
            getattr(self, "slider_dehaze", None),
            getattr(self, "slider_ond_est", None),
            getattr(self, "slider_ond_bg", None),
            getattr(self, "slider_denoise", None),
            getattr(self, "slider_denoise_color", None),
            getattr(self, "slider_denoise_mask", None),
        ]
        for ctrl in controls:
            if ctrl is not None:
                ctrl.setEnabled(enabled)

        # Bloqueo de controles de balance de blancos, saturación y estirado al desacoplar capas
        if hasattr(self, "panel_color_wb") and self.panel_color_wb is not None:
            if hasattr(self.panel_color_wb, "sec_wb"):
                self.panel_color_wb.sec_wb.content_area.setEnabled(not enabled)
                self.panel_color_wb.sec_wb.btn_reset.setEnabled(not enabled)
                if enabled:
                    self.panel_color_wb.sec_wb.set_status_badge("🔒 Integrado", "#90a4ae")
                else:
                    self.panel_color_wb.sec_wb.set_status_badge("")
            if hasattr(self.panel_color_wb, "sec_color"):
                self.panel_color_wb.sec_color.content_area.setEnabled(not enabled)
                self.panel_color_wb.sec_color.btn_reset.setEnabled(not enabled)
                if enabled:
                    self.panel_color_wb.sec_color.set_status_badge("🔒 En capas", "#90a4ae")
                else:
                    self.panel_color_wb.sec_color.set_status_badge("")

        if hasattr(self, "panel_stretch") and self.panel_stretch is not None:
            if hasattr(self.panel_stretch, "sec_stretch"):
                self.panel_stretch.sec_stretch.content_area.setEnabled(not enabled)
                self.panel_stretch.sec_stretch.btn_reset.setEnabled(not enabled)
                if enabled:
                    self.panel_stretch.sec_stretch.set_status_badge("🔒 Base fijada", "#90a4ae")
                else:
                    self.panel_stretch.sec_stretch.set_status_badge("")

        # Contaminación lumínica: permanece siempre 100% interactiva
        if hasattr(self, "panel_light_pollution") and self.panel_light_pollution is not None:
            if hasattr(self.panel_light_pollution, "sec_lp"):
                self.panel_light_pollution.sec_lp.content_area.setEnabled(True)
                self.panel_light_pollution.sec_lp.btn_reset.setEnabled(True)

    # -------------------------------------------------------------
    # GESTOR DE HISTORIAL DE DESHACER / REHACER (UNDO / REDO)
    # -------------------------------------------------------------
    def _connect_undo_triggers(self):
        """Conecta los eventos de finalización de interacción para registrar acciones en el historial."""
        
        # 1. Los paneles modulares ya emiten su propia señal action_committed al soltar sliders o editar spinboxes
        # Se conectan aquí junto con interaction_started e interaction_finished para modo borrador 60 FPS:
        panel_press_handlers = (
            (getattr(self, 'panel_stretch', None), self._on_lut_slider_pressed),
            (getattr(self, 'panel_lp', None), self._on_heavy_slider_pressed),
            (getattr(self, 'panel_deconv', None), self._on_deconv_slider_pressed),
            (getattr(self, 'panel_sharpening', None), self._on_heavy_slider_pressed),
            (getattr(self, 'panel_color_wb', None), self._on_heavy_slider_pressed),
            (getattr(self, 'panel_effects', None), self._on_heavy_slider_pressed),
        )
        for panel, on_pressed in panel_press_handlers:
            if panel is not None:
                if hasattr(panel, 'action_committed'):
                    try:
                        panel.action_committed.connect(self._commit_action)
                    except Exception:
                        pass
                if hasattr(panel, 'interaction_started'):
                    try:
                        panel.interaction_started.connect(on_pressed)
                    except Exception:
                        pass
                if hasattr(panel, 'interaction_finished'):
                    try:
                        panel.interaction_finished.connect(self._on_slider_interaction_finished)
                    except Exception:
                        pass

        # 2. Conectar eventos del Gestor de Capas si existen
        if hasattr(self, 'mask_layers_widget'):
            if hasattr(self.mask_layers_widget, 'action_committed'):
                try:
                    self.mask_layers_widget.action_committed.connect(self._commit_action)
                except Exception:
                    pass
            if hasattr(self.mask_layers_widget, 'interaction_started'):
                self.mask_layers_widget.interaction_started.connect(self._on_layer_slider_pressed)
            if hasattr(self.mask_layers_widget, 'interaction_finished'):
                self.mask_layers_widget.interaction_finished.connect(self._on_slider_interaction_finished)

        # 3. Conectar la curva tonal
        if hasattr(self, 'curve_widget'):
            self.curve_widget.curveCommitted.connect(self._on_control_committed)

    def _on_lut_slider_pressed(self):
        """Activa el modo de visualización directa por LUT 1D (>1000 FPS) para estirado y contraste."""
        if hasattr(self, 'async_pipeline'):
            self.async_pipeline.cancel_pending()
        if getattr(self, 'is_zoomed_100', False) and hasattr(self, 'tile_manager'):
            self.tile_manager.cancel_all()
        self._is_slider_pressed = True
        self._is_scrubbing_heavy_slider = True
        self._is_scrubbing_lut_slider = True
        self._is_scrubbing_layer_slider = False
        self._is_dragging_deconv = False

    def _on_heavy_slider_pressed(self):
        """Activa el modo de proxy borrador ultra-fluido (Draft 60 FPS) durante el arrastre continuo."""
        if hasattr(self, 'async_pipeline'):
            self.async_pipeline.cancel_pending()
        self._is_slider_pressed = True
        self._is_scrubbing_heavy_slider = True
        self._is_scrubbing_lut_slider = False
        self._is_scrubbing_layer_slider = False
        self._is_dragging_deconv = False

    def _on_layer_slider_pressed(self):
        """Activa el modo de interacción para capas locales, bajando a Draft si hay muchas capas/filtros encima."""
        if hasattr(self, 'async_pipeline'):
            self.async_pipeline.cancel_pending()
        use_draft = False
        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget is not None:
            sel = self.mask_layers_widget.get_selected_layer()
            layers = getattr(self.mask_layers_widget, 'layers', [])
            if sel is not None and layers:
                idx = next((i for i, l in enumerate(layers) if l.id == sel.id), 0)
                above = [l for l in layers[:idx] if l.enabled and l.opacity > 1e-4]
                affected = above + [sel]
                has_spatial = any(
                    abs(getattr(l, 'clarity', 0.0)) > 1e-3
                    or abs(getattr(l, 'texture_clarity', 0.0)) > 1e-3
                    or getattr(l, 'sharp_amount', 0.0) > 1e-3
                    or (getattr(l, 'refine_edges', False) and getattr(l, 'refine_strength', 0.0) > 1e-4)
                    for l in affected
                )
                use_draft = (len(above) >= 3) or has_spatial
        self._is_slider_pressed = True
        self._is_scrubbing_heavy_slider = use_draft
        self._is_scrubbing_lut_slider = False
        self._is_scrubbing_layer_slider = True
        self._is_dragging_deconv = False
        self._is_dragging_antihalo = False

    def _on_deconv_slider_pressed(self):
        """Activa el modo de arrastre de deconvolución manteniendo fidelidad del filtro de PSF."""
        if hasattr(self, 'async_pipeline'):
            self.async_pipeline.cancel_pending()
        self._is_slider_pressed = True
        self._is_scrubbing_heavy_slider = True
        self._is_scrubbing_lut_slider = False
        self._is_scrubbing_layer_slider = False
        self._is_dragging_deconv = True
        self._is_dragging_antihalo = False

    def _on_slider_interaction_finished(self):
        """Finaliza inmediatamente el renderizado y los histogramas al soltar el ratón y consolida en historial."""
        if hasattr(self, '_refine_timer'):
            self._refine_timer.stop()
        self._is_slider_pressed = False
        self._is_scrubbing_heavy_slider = False
        self._is_scrubbing_lut_slider = False
        self._is_scrubbing_layer_slider = False
        self._is_dragging_deconv = False
        self._is_dragging_antihalo = False
        if getattr(self, 'is_zoomed_100', False) and hasattr(self, 'tile_manager'):
            self._request_zoom_100_tiles()
        else:
            self.update_stretch_preview(immediate=True)
        self._commit_action()

    def _on_control_committed(self):
        """Finaliza inmediatamente el renderizado y consolida en historial tras editar spinboxes o combos."""
        if hasattr(self, '_refine_timer'):
            self._refine_timer.stop()
        self._is_slider_pressed = False
        self._is_scrubbing_heavy_slider = False
        self._is_scrubbing_lut_slider = False
        self._is_scrubbing_layer_slider = False
        self._is_dragging_deconv = False
        self._is_dragging_antihalo = False
        self.update_stretch_preview(immediate=True)
        self._commit_action()

    def _capture_state(self, include_mask_layers: bool = True) -> dict:
        """Captura una instantánea paramétrica ligera delegando en los paneles modulares."""
        state = {}
        if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
            state.update(self.panel_stretch.get_state())
        if hasattr(self, 'panel_lp') and self.panel_lp is not None:
            state.update(self.panel_lp.get_state())
        if hasattr(self, 'panel_deconv') and self.panel_deconv is not None:
            state.update(self.panel_deconv.get_state())
        if hasattr(self, 'panel_sharpening') and self.panel_sharpening is not None:
            state.update(self.panel_sharpening.get_state())
        if hasattr(self, 'panel_color_wb') and self.panel_color_wb is not None:
            state.update(self.panel_color_wb.get_state())
        if hasattr(self, 'panel_starnet') and self.panel_starnet is not None:
            state.update(self.panel_starnet.get_state())
        if hasattr(self, 'panel_ground_sky') and self.panel_ground_sky is not None:
            state.update(self.panel_ground_sky.get_state())
        if hasattr(self, 'panel_dust') and self.panel_dust is not None:
            state["dust_spots"] = self.canvas.get_dust_spots() if hasattr(self, 'canvas') else []
            state["dust_sensitivity"] = self.panel_dust.sensitivity
            state["dust_mode"] = self.panel_dust.mode
            state["dust_visualize"] = self.panel_dust.visualize
            state["dust_visualize_threshold"] = self.panel_dust.visualize_threshold
        state["curves"] = self.curve_widget.get_state() if hasattr(self, 'curve_widget') else None
        state["mask_layers"] = self.mask_layers_widget.get_state() if (include_mask_layers and hasattr(self, 'mask_layers_widget')) else None
        state["starless_is_stretched"] = getattr(self, "starless_is_stretched", False)
        return state

    def _state_different(self, s1: Optional[dict], s2: Optional[dict]) -> bool:
        """Comprueba si dos instantáneas paramétricas presentan diferencias efectivas."""
        if s1 is None or s2 is None:
            return True
        for k, v1 in s1.items():
            if k in ("dust_visualize", "dust_visualize_threshold"):
                continue
            v2 = s2.get(k)
            if k in ("curves", "mask_layers", "dust_spots"):
                if v1 != v2:
                    return True
            elif isinstance(v1, float):
                if abs(v1 - (v2 or 0.0)) > 1e-5:
                    return True
            elif v1 != v2:
                return True
        return False

    def _commit_action(self):
        """Registra el cambio actual en la pila de deshacer si difiere del último estado consolidado."""
        if (
            self._is_undoing_or_redoing
            or getattr(self, '_is_applying_preset', False)
            or getattr(self, '_is_previewing_preset', False)
            or self._last_committed_state is None
        ):
            return
        current = self._capture_state()
        if self._state_different(self._last_committed_state, current):
            self.undo_stack.append(self._last_committed_state)
            if len(self.undo_stack) > self.MAX_UNDO_STEPS:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self._last_committed_state = current
            self._mark_dirty()
            self._update_undo_redo_buttons()

    def undo(self):
        """Deshace el último ajuste paramétrico o recorte realizado (Ctrl+Z)."""
        if not self.undo_stack or self._is_undoing_or_redoing:
            return
        self._is_undoing_or_redoing = True
        try:
            target = self.undo_stack.pop()
            if target.get("type") in ("crop", "bg_extract"):
                op_type = target.get("type")
                redo_item = {
                    "type": op_type,
                    "image_32bit": self.image_32bit.copy(),
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
                    "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                    "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                    "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
                    "_cache_starless_ai": self._cache_starless_ai.copy() if getattr(self, "_cache_starless_ai", None) is not None else None,
                    "_cache_ground_ai": self._cache_ground_ai.copy() if getattr(self, "_cache_ground_ai", None) is not None else None,
                    "mask_layers": [l.shallow_snapshot() for l in self.mask_layers_widget.layers] if hasattr(self, 'mask_layers_widget') else [],
                    "dust_spots": [dict(s) for s in self.canvas.get_dust_spots()] if hasattr(self, 'canvas') else [],
                }
                self.redo_stack.append(redo_item)
                self.image_32bit = target["image_32bit"]
                self.image_starless = target["image_starless"]
                self.raw_linear_starless = target.get("raw_linear_starless")
                self.starless_is_stretched = target.get("starless_is_stretched", False)
                self.image_stars = target["image_stars"]
                self.current_mask = target["current_mask"]
                self._cache_starless_ai = target.get("_cache_starless_ai")
                self._cache_ground_ai = target.get("_cache_ground_ai")
                if "mask_layers" in target and hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.blockSignals(True)
                    self.mask_layers_widget.set_state(target["mask_layers"] or [])
                    self.mask_layers_widget.blockSignals(False)
                    self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
                    self.mask_layers_widget.update_active_layer_histogram()
                    if hasattr(self.mask_layers_widget, '_update_list_and_props_state'):
                        self.mask_layers_widget._update_list_and_props_state()

                if "dust_spots" in target and hasattr(self, 'canvas'):
                    self.canvas.set_dust_spots(target["dust_spots"] or [])
                    if hasattr(self, 'developer_state') and self.developer_state is not None:
                        self.developer_state.dust_spots = target["dust_spots"] or []
                    if hasattr(self, 'panel_dust') and self.panel_dust is not None:
                        self.panel_dust.update_spot_count(len(target["dust_spots"] or []))

                self.snapshot_before_pixmap = None
                self.has_custom_snapshot = False
                h, w = self.image_32bit.shape[:2]
                self.canvas.set_native_image_size((w, h))
                if self.current_mask is not None:
                    self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
                self._generate_preview_proxy()
                self._generate_before_reference()
                self.update_stretch_preview(immediate=True)
                self._update_canvas_info_label()
                self._update_mask_status_ui()
                op_label = "extracción de gradiente" if op_type == "bg_extract" else "recorte"
                self.log_message(f"↺ Deshacer {op_label}: Restaurada imagen ({w}x{h} px)")
            elif target.get("type") == "starless":
                redo_item = {
                    "type": "starless",
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                    "_cache_starless_ai": self._cache_starless_ai.copy() if getattr(self, "_cache_starless_ai", None) is not None else None,
                    "starless_ai_denoise_enabled": getattr(self, "starless_ai_denoise_enabled", False),
                    "denoise_strength": getattr(self, "denoise_strength", 0.0),
                    "denoise_slider_val": self.slider_denoise.value() if hasattr(self, 'slider_denoise') else 0,
                }
                self.redo_stack.append(redo_item)
                self.image_starless = target["image_starless"]
                self.starless_is_stretched = target.get("starless_is_stretched", False)
                self._cache_starless_ai = target.get("_cache_starless_ai")
                self.starless_ai_denoise_enabled = target.get("starless_ai_denoise_enabled", False)
                self.denoise_strength = target.get("denoise_strength", 0.0)
                dn_slider_val = target.get("denoise_slider_val", int(round(self.denoise_strength * 100)))
                if hasattr(self, 'slider_denoise'):
                    self.slider_denoise.setValue(dn_slider_val)
                has_ai = (self._cache_starless_ai is not None) and self.starless_ai_denoise_enabled
                if hasattr(self, 'panel_starnet'):
                    self.panel_starnet.set_ai_denoise_active(has_ai)
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.set_ai_denoise_active(has_ai)
                self._invalidate_pipeline_caches()
                self._generate_preview_proxy()
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
                    self.mask_layers_widget.update_active_layer_histogram()
                self.update_stretch_preview(immediate=True)
                self.log_message("\u21ba Deshacer: Estirado Adaptativo de Starless revertido")
            elif target.get("type") == "starnet_separation":
                if hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.stop()
                redo_item = {
                    "type": "starnet_separation",
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
                    "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                    "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                    "mask_layers": [l.shallow_snapshot() for l in self.mask_layers_widget.layers] if hasattr(self, 'mask_layers_widget') else [],
                }
                self.redo_stack.append(redo_item)
                self.image_starless = target["image_starless"]
                self.raw_linear_starless = target.get("raw_linear_starless")
                self.starless_is_stretched = target.get("starless_is_stretched", False)
                self.image_stars = target.get("image_stars")
                if self.image_starless is None:
                    self.proxy_starless = None
                    self.proxy_stars = None
                    self.draft_starless = None
                    self.draft_stars = None
                    self.pyramid_starless = None
                    self.pyramid_stars = None
                has_starnet = self.image_starless is not None
                self._set_starnet_tools_enabled(has_starnet)
                self.combo_layer.setEnabled(has_starnet)
                self.slider_stars.setEnabled(has_starnet)
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.blockSignals(True)
                    prev_layers = target.get("mask_layers", [])
                    if prev_layers:
                        self.mask_layers_widget.set_state(prev_layers)
                    else:
                        self.mask_layers_widget.clear_all_layers()
                        self.mask_layers_widget.ensure_base_layer()
                    self.mask_layers_widget.blockSignals(False)
                    self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
                    self.mask_layers_widget.update_active_layer_histogram()
                self._invalidate_pipeline_caches()
                self._generate_preview_proxy()
                self.update_stretch_preview(immediate=True)
                if hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.stop()
                self._last_committed_state = self._capture_state()
                self.log_message(f"↺ Deshacer: Separación StarNet revertida ({len(self.undo_stack)} pasos restantes)")
            elif target.get("type") == "mask":
                redo_item = {
                    "type": "mask",
                    "name": target.get("name", "Máscara"),
                    "old_mask": target.get("old_mask").copy() if target.get("old_mask") is not None else None,
                    "new_mask": target.get("new_mask").copy() if target.get("new_mask") is not None else None,
                }
                self.redo_stack.append(redo_item)
                self.current_mask = target.get("old_mask")
                self.canvas.set_refined_mask(self.current_mask)
                self._generate_preview_proxy()
                self.update_stretch_preview(immediate=True)
                self._update_mask_status_ui()
                self.log_message("↺ Deshacer: Cambio de máscara revertido")
            elif target.get("type") == "preset":
                redo_item = {
                    "type": "preset",
                    "preset_name": target.get("preset_name", "Preset"),
                    "params": self._capture_state(),
                    "mask": self.current_mask.copy() if self.current_mask is not None else None,
                    "mask_source_name": getattr(self, '_mask_source_name', ""),
                    "badge_text": self.sec_presets.lbl_status.text() if hasattr(self, 'sec_presets') and hasattr(self.sec_presets, 'lbl_status') else "",
                }
                self.redo_stack.append(redo_item)

                old_mask = target.get("mask")
                self.current_mask = old_mask.copy() if old_mask is not None else None
                self._mask_source_name = target.get("mask_source_name", "")
                self.canvas.set_refined_mask(self.current_mask)
                self.canvas.set_mask_visible(self.current_mask is not None)
                self._update_mask_status_ui()

                if hasattr(self, 'sec_presets'):
                    badge = target.get("badge_text", "")
                    self.sec_presets.set_status_badge(badge, "#80d8ff" if badge else "")

                self._restore_state(target.get("params", {}))
                self._last_committed_state = self._capture_state()
                if hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.stop()
                self.log_message(f"↺ Deshacer: Preset '{target.get('preset_name', '')}' revertido ({len(self.undo_stack)} pasos restantes)")
            else:
                current = self._capture_state()
                self.redo_stack.append(current)
                self._restore_state(target)
                self._last_committed_state = self._capture_state()
                self.log_message(f"↺ Deshacer aplicado ({len(self.undo_stack)} pasos restantes)")
        finally:
            self._is_undoing_or_redoing = False
            self._mark_dirty()
            self._update_undo_redo_buttons()

    def redo(self):
        """Rehace el ajuste paramétrico o recorte previamente deshecho (Ctrl+Y / Ctrl+Shift+Z)."""
        if not self.redo_stack or self._is_undoing_or_redoing:
            return
        self._is_undoing_or_redoing = True
        try:
            target = self.redo_stack.pop()
            if target.get("type") in ("crop", "bg_extract"):
                op_type = target.get("type")
                undo_item = {
                    "type": op_type,
                    "image_32bit": self.image_32bit.copy(),
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
                    "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                    "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                    "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
                    "_cache_starless_ai": self._cache_starless_ai.copy() if getattr(self, "_cache_starless_ai", None) is not None else None,
                    "_cache_ground_ai": self._cache_ground_ai.copy() if getattr(self, "_cache_ground_ai", None) is not None else None,
                    "mask_layers": [l.shallow_snapshot() for l in self.mask_layers_widget.layers] if hasattr(self, 'mask_layers_widget') else [],
                    "dust_spots": [dict(s) for s in self.canvas.get_dust_spots()] if hasattr(self, 'canvas') else [],
                }
                self.undo_stack.append(undo_item)
                self.image_32bit = target["image_32bit"]
                self.image_starless = target["image_starless"]
                self.raw_linear_starless = target.get("raw_linear_starless")
                self.starless_is_stretched = target.get("starless_is_stretched", False)
                self.image_stars = target["image_stars"]
                self.current_mask = target["current_mask"]
                self._cache_starless_ai = target.get("_cache_starless_ai")
                self._cache_ground_ai = target.get("_cache_ground_ai")
                if "mask_layers" in target and hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.blockSignals(True)
                    self.mask_layers_widget.set_state(target["mask_layers"] or [])
                    self.mask_layers_widget.blockSignals(False)
                    self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
                    self.mask_layers_widget.update_active_layer_histogram()
                    if hasattr(self.mask_layers_widget, '_update_list_and_props_state'):
                        self.mask_layers_widget._update_list_and_props_state()

                if "dust_spots" in target and hasattr(self, 'canvas'):
                    self.canvas.set_dust_spots(target["dust_spots"] or [])
                    if hasattr(self, 'developer_state') and self.developer_state is not None:
                        self.developer_state.dust_spots = target["dust_spots"] or []
                    if hasattr(self, 'panel_dust') and self.panel_dust is not None:
                        self.panel_dust.update_spot_count(len(target["dust_spots"] or []))

                self.snapshot_before_pixmap = None
                self.has_custom_snapshot = False
                h, w = self.image_32bit.shape[:2]
                self.canvas.set_native_image_size((w, h))
                if self.current_mask is not None:
                    self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
                self._generate_preview_proxy()
                self._generate_before_reference()
                self.update_stretch_preview(immediate=True)
                self._update_canvas_info_label()
                self._update_mask_status_ui()
                op_label = "extracción de gradiente" if op_type == "bg_extract" else "recorte"
                self.log_message(f"↻ Rehacer {op_label}: Restaurada ({w}x{h} px)")
            elif target.get("type") == "starless":
                undo_item = {
                    "type": "starless",
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                    "_cache_starless_ai": self._cache_starless_ai.copy() if getattr(self, "_cache_starless_ai", None) is not None else None,
                    "starless_ai_denoise_enabled": getattr(self, "starless_ai_denoise_enabled", False),
                    "denoise_strength": getattr(self, "denoise_strength", 0.0),
                    "denoise_slider_val": self.slider_denoise.value() if hasattr(self, 'slider_denoise') else 0,
                }
                self.undo_stack.append(undo_item)
                self.image_starless = target["image_starless"]
                self.starless_is_stretched = target.get("starless_is_stretched", True)
                self._cache_starless_ai = target.get("_cache_starless_ai")
                self.starless_ai_denoise_enabled = target.get("starless_ai_denoise_enabled", False)
                self.denoise_strength = target.get("denoise_strength", 0.0)
                dn_slider_val = target.get("denoise_slider_val", int(round(self.denoise_strength * 100)))
                if hasattr(self, 'slider_denoise'):
                    self.slider_denoise.setValue(dn_slider_val)
                has_ai = (self._cache_starless_ai is not None) and self.starless_ai_denoise_enabled
                if hasattr(self, 'panel_starnet'):
                    self.panel_starnet.set_ai_denoise_active(has_ai)
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.set_ai_denoise_active(has_ai)
                self._invalidate_pipeline_caches()
                self._generate_preview_proxy()
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
                    self.mask_layers_widget.update_active_layer_histogram()
                self.update_stretch_preview(immediate=True)
                self.log_message("\u21bb Rehacer: Estirado Adaptativo de Starless restaurado")
            elif target.get("type") == "starnet_separation":
                if hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.stop()
                undo_item = {
                    "type": "starnet_separation",
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
                    "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                    "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                    "mask_layers": [l.shallow_snapshot() for l in self.mask_layers_widget.layers] if hasattr(self, 'mask_layers_widget') else [],
                }
                self.undo_stack.append(undo_item)
                self.image_starless = target["image_starless"]
                self.raw_linear_starless = target.get("raw_linear_starless")
                self.starless_is_stretched = target.get("starless_is_stretched", False)
                self.image_stars = target.get("image_stars")
                if self.image_starless is None:
                    self.proxy_starless = None
                    self.proxy_stars = None
                    self.draft_starless = None
                    self.draft_stars = None
                    self.pyramid_starless = None
                    self.pyramid_stars = None
                has_starnet = self.image_starless is not None
                self._set_starnet_tools_enabled(has_starnet)
                self.combo_layer.setEnabled(has_starnet)
                self.slider_stars.setEnabled(has_starnet)
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.blockSignals(True)
                    next_layers = target.get("mask_layers", [])
                    if next_layers:
                        self.mask_layers_widget.set_state(next_layers)
                    else:
                        self.mask_layers_widget.clear_all_layers()
                        self.mask_layers_widget.ensure_base_layer()
                    self.mask_layers_widget.blockSignals(False)
                    self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
                    self.mask_layers_widget.update_active_layer_histogram()
                self._invalidate_pipeline_caches()
                self._generate_preview_proxy()
                self.update_stretch_preview(immediate=True)
                if hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.stop()
                self._last_committed_state = self._capture_state()
                self.log_message(f"↻ Rehacer: Separación StarNet restaurada ({len(self.redo_stack)} pasos por rehacer)")
            elif target.get("type") == "mask":
                undo_item = {
                    "type": "mask",
                    "name": target.get("name", "Máscara"),
                    "old_mask": self.current_mask.copy() if self.current_mask is not None else None,
                    "new_mask": target["new_mask"].copy() if target.get("new_mask") is not None else None,
                }
                self.undo_stack.append(undo_item)
                self.current_mask = target.get("new_mask")
                self.canvas.set_refined_mask(self.current_mask)
                self._generate_preview_proxy()
                self.update_stretch_preview(immediate=True)
                self._update_mask_status_ui()
                self.log_message("↻ Rehacer: Cambio de máscara restablecido")
            elif target.get("type") == "preset":
                undo_item = {
                    "type": "preset",
                    "preset_name": target.get("preset_name", "Preset"),
                    "params": self._capture_state(),
                    "mask": self.current_mask.copy() if self.current_mask is not None else None,
                    "mask_source_name": getattr(self, '_mask_source_name', ""),
                    "badge_text": self.sec_presets.lbl_status.text() if hasattr(self, 'sec_presets') and hasattr(self.sec_presets, 'lbl_status') else "",
                }
                self.undo_stack.append(undo_item)

                new_mask = target.get("mask")
                self.current_mask = new_mask.copy() if new_mask is not None else None
                self._mask_source_name = target.get("mask_source_name", "")
                self.canvas.set_refined_mask(self.current_mask)
                self.canvas.set_mask_visible(self.current_mask is not None)
                self._update_mask_status_ui()

                if hasattr(self, 'sec_presets'):
                    badge = target.get("badge_text", "")
                    self.sec_presets.set_status_badge(badge, "#80d8ff" if badge else "")

                self._restore_state(target.get("params", {}))
                self._last_committed_state = self._capture_state()
                if hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.stop()
                self.log_message(f"↻ Rehacer: Preset '{target.get('preset_name', '')}' restaurado ({len(self.redo_stack)} pasos por rehacer)")
            else:
                current = self._capture_state()
                self.undo_stack.append(current)
                self._restore_state(target)
                self._last_committed_state = self._capture_state()
                self.log_message(f"↻ Rehacer aplicado ({len(self.redo_stack)} pasos por rehacer)")
        finally:
            self._is_undoing_or_redoing = False
            self._mark_dirty()
            self._update_undo_redo_buttons()

    def _restore_state(self, state: dict, render_preview: bool = True) -> None:
        """Restaura todos los controles al estado paramétrico especificado delegando en los paneles modulares."""
        if not state:
            return

        if hasattr(self, 'panel_stretch'):
            self.panel_stretch.set_state(state)
            self.stretch_mode = self.panel_stretch.stretch_mode
            self.ghs_color_mode = self.panel_stretch.ghs_color_mode
            self.bp_val = self.panel_stretch.bp_val
            self.mtf_val = self.panel_stretch.mtf_val
            self.contrast_val = self.panel_stretch.contrast_val
            self.ghs_sp_val = self.panel_stretch.ghs_sp_val
            self.ghs_d_val = self.panel_stretch.ghs_d_val
            self.ghs_b_val = self.panel_stretch.ghs_b_val
            self.ghs_hp_val = self.panel_stretch.ghs_hp_val
            self.ghs_lp_val = self.panel_stretch.ghs_lp_val
            self.ghs_bp_val = self.panel_stretch.ghs_bp_val

        if hasattr(self, 'panel_lp'):
            self.panel_lp.set_state(state)
            self.lp_reduction = self.panel_lp.lp_reduction
            self.lp_center_x = self.panel_lp.lp_center_x
            self.lp_center_y = self.panel_lp.lp_center_y
            self.lp_width = self.panel_lp.lp_width
            self.lp_height = self.panel_lp.lp_height
            self.lp_tint_type = self.panel_lp.lp_tint_type
            self.lp_custom_tint = self.panel_lp.lp_custom_tint
            self.lp_tint_strength = self.panel_lp.lp_tint_strength
            self.lp_mode = self.panel_lp.lp_mode
            self.lp_legacy_1d = self.panel_lp.lp_legacy_1d

        if hasattr(self, 'panel_deconv'):
            self.panel_deconv.set_state(state)
            self.deconv_iterations = self.panel_deconv.deconv_iterations
            self.deconv_fwhm = self.panel_deconv.deconv_fwhm
            self.deconv_deringing = self.panel_deconv.deconv_deringing
            self.deconv_damping = self.panel_deconv.deconv_damping
            self.deconv_aspect_ratio = self.panel_deconv.deconv_aspect_ratio
            self.deconv_angle_deg = self.panel_deconv.deconv_angle_deg
            self.deconv_model = self.panel_deconv.deconv_model
            self.deconv_scope = self.panel_deconv.deconv_scope
            self.deconv_luma_only = self.panel_deconv.deconv_luma_only

        if hasattr(self, 'panel_sharpening') and self.panel_sharpening is not None:
            self.panel_sharpening.set_state(state)
            self.sharp_amount = getattr(self.panel_sharpening, 'row_amount', None) and self.panel_sharpening.row_amount.slider.value() or 0
            self.sharp_radius = getattr(self.panel_sharpening, 'row_radius', None) and self.panel_sharpening.row_radius.slider.value() or 10
            self.sharp_detail = getattr(self.panel_sharpening, 'row_detail', None) and self.panel_sharpening.row_detail.slider.value() or 25
            self.sharp_masking = getattr(self.panel_sharpening, 'row_masking', None) and self.panel_sharpening.row_masking.slider.value() or 0
            self.sharp_scope_idx = getattr(self.panel_sharpening, 'combo_scope', None) and self.panel_sharpening.combo_scope.currentIndex() or 0

        if hasattr(self, 'panel_color_wb'):
            self.panel_color_wb.set_state(state)
            self.temp_val = self.panel_color_wb.temp_val
            self.tint_val = self.panel_color_wb.tint_val
            self.sat_sky_val = self.panel_color_wb.sat_sky_val
            self.sat_gnd_val = self.panel_color_wb.sat_gnd_val
            self.vibrance_val = self.panel_color_wb.vibrance_val
            self.scnr_green_val = getattr(self.panel_color_wb, 'scnr_green_val', 0)
            self.unpurple_val = getattr(self.panel_color_wb, 'unpurple_val', 0)
            self.dualband_mode = getattr(self.panel_color_wb, 'dualband_mode', 'none')
            self.pcc_kr = getattr(self.panel_color_wb, 'pcc_kr', 1.0)
            self.pcc_kb = getattr(self.panel_color_wb, 'pcc_kb', 1.0)

        if hasattr(self, 'panel_starnet'):
            self.panel_starnet.set_state(state)
            self.view_layer_mode = self.panel_starnet.combo_layer.currentIndex()
            self.star_intensity = self.panel_starnet.slider_stars.value() / 100.0
            self.contrast_starless_val = self.panel_starnet.slider_contrast_starless.value() / 100.0
            self.clarity_starless_val = self.panel_starnet.slider_clarity.value() / 100.0
            self.texture_clarity_starless_val = self.panel_starnet.slider_texture_clarity.value() / 100.0
            self.dehaze_starless_val = self.panel_starnet.slider_dehaze.value() / 100.0
            self.multiescala_estructura = self.panel_starnet.slider_ond_est.value() / 100.0
            self.multiescala_fondo = self.panel_starnet.slider_ond_bg.value() / 100.0
            self.denoise_strength = self.panel_starnet.slider_denoise.value() / 100.0
            self.denoise_color_strength = getattr(self.panel_starnet, 'slider_denoise_color', None).value() / 100.0 if hasattr(self.panel_starnet, 'slider_denoise_color') else 0.50
            self.denoise_masking = getattr(self.panel_starnet, 'slider_denoise_mask', None).value() / 100.0 if hasattr(self.panel_starnet, 'slider_denoise_mask') else 0.50
            if hasattr(self, 'panel_effects'):
                self.orton_strength = self.panel_effects.orton_strength
                self.orton_radius = self.panel_effects.orton_radius
                self.orton_threshold = self.panel_effects.orton_threshold
                self.orton_tint_type = self.panel_effects.orton_tint_type
                self.orton_mode = self.panel_effects.orton_mode
                self.star_diff_strength = self.panel_effects.star_diff_strength
                self.star_diff_radius = self.panel_effects.star_diff_radius
                self.star_diff_threshold = self.panel_effects.star_diff_threshold
                self.star_diff_color_boost = self.panel_effects.star_diff_color_boost
                self.star_diff_mode = self.panel_effects.star_diff_mode

        if hasattr(self, 'panel_ground_sky'):
            self.panel_ground_sky.set_state(state)
            self.gnd_ev_val = self.panel_ground_sky.slider_gnd_ev.value() / 100.0
            self.gnd_shadows_val = self.panel_ground_sky.slider_gnd_sh.value() / 100.0
            self.gnd_bp_val = self.panel_ground_sky.slider_gnd_bp.value() / 1000.0
            if hasattr(self.panel_ground_sky, 'slider_gnd_dn'):
                self.gnd_denoise_val = self.panel_ground_sky.slider_gnd_dn.value() / 100.0
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.set_ground_denoise_value(self.panel_ground_sky.slider_gnd_dn.value())

        if hasattr(self, 'panel_dust') and self.panel_dust is not None:
            spots = state.get("dust_spots", [])
            self.canvas.set_dust_spots(spots)
            self.panel_dust.update_spot_count(len(spots))
            if hasattr(self, 'developer_state') and self.developer_state is not None:
                self.developer_state.dust_spots = spots
            if "dust_sensitivity" in state:
                sens = int(state["dust_sensitivity"])
                self.panel_dust.slider_sens.setValue(sens)
                if hasattr(self, 'developer_state') and self.developer_state is not None:
                    self.developer_state.dust_sensitivity = sens
            if "dust_mode" in state:
                mode = state["dust_mode"]
                modes = ["heal", "inpaint", "clone"]
                if mode in modes:
                    self.panel_dust.combo_mode.setCurrentIndex(modes.index(mode))
                if hasattr(self, 'developer_state') and self.developer_state is not None:
                    self.developer_state.dust_mode = mode
        if "starless_is_stretched" in state:
            self.starless_is_stretched = bool(state["starless_is_stretched"])

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                self.slider_gnd_ev.value(),
                self.slider_gnd_sh.value(),
                self.slider_gnd_bp.value()
            )
            self.mask_layers_widget.set_stars_special_values(
                presence=self.slider_stars.value(),
                diff_strength=self.slider_star_diff.value() if hasattr(self, 'slider_star_diff') else 0,
                diff_radius=self.slider_star_diff_radius.value() if hasattr(self, 'slider_star_diff_radius') else 20,
                diff_threshold=self.slider_star_diff_threshold.value() if hasattr(self, 'slider_star_diff_threshold') else 45,
                diff_color=self.slider_star_diff_color.value() if hasattr(self, 'slider_star_diff_color') else 50,
                diff_mode_idx=self.combo_star_diff_mode.currentIndex() if hasattr(self, 'combo_star_diff_mode') else 0
            )
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(),
                self.slider_dehaze.value(),
                self.slider_ond_est.value(),
                self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(),
                self.slider_denoise.value(),
                self.slider_orton.value() if hasattr(self, 'slider_orton') else 0,
                self.slider_orton_radius.value() if hasattr(self, 'slider_orton_radius') else 40,
                self.slider_orton_threshold.value() if hasattr(self, 'slider_orton_threshold') else 5,
                self.combo_orton_tint.currentIndex() if hasattr(self, 'combo_orton_tint') else 0,
                self.combo_orton_mode.currentIndex() if hasattr(self, 'combo_orton_mode') else 0,
                denoise_color=getattr(self.panel_starnet, 'slider_denoise_color', None).value() if hasattr(self.panel_starnet, 'slider_denoise_color') else 50,
                denoise_mask=getattr(self.panel_starnet, 'slider_denoise_mask', None).value() if hasattr(self.panel_starnet, 'slider_denoise_mask') else 50,
                texture_clarity=self.slider_texture_clarity.value() if hasattr(self, 'slider_texture_clarity') else 0
            )

        if hasattr(self, 'curve_widget'):
            self.curve_widget.blockSignals(True)
            if "curves" in state and state["curves"] is not None:
                self.curve_widget.set_state(state["curves"])
            else:
                self.curve_widget.reset_all_curves()
            self.curve_widget.blockSignals(False)

        if "mask_layers" in state and hasattr(self, 'mask_layers_widget'):
            # Guardamos la capa que el usuario tiene seleccionada actualmente
            current_layer = self.mask_layers_widget.get_selected_layer()
            current_id = current_layer.id if current_layer else None

            self.mask_layers_widget.blockSignals(True)
            self.mask_layers_widget.set_state(state["mask_layers"] or [])
            self.mask_layers_widget.blockSignals(False)

            # Restauramos la selección a la capa donde estaba el usuario, no a estrellas
            if current_id:
                self.mask_layers_widget.select_layer_by_id(current_id)

            sel_layer = self.mask_layers_widget.get_selected_layer()
            if sel_layer and getattr(sel_layer, 'is_gradient', False) and sel_layer.enabled:
                self.canvas.set_active_gradient_layer(sel_layer)
            elif sel_layer and (sel_layer.mask_type == "brush" or sel_layer.mask_type == MASK_TYPE_IMAGE_RASTER) and sel_layer.enabled:
                self.canvas.set_active_gradient_layer(None)
                show_overlay = self.mask_layers_widget.is_mask_preview_active()
                self.canvas.set_active_layer_mask(sel_layer.brush_mask, show_overlay=show_overlay)
            else:
                self.canvas.set_active_gradient_layer(None)
                if not self.mask_layers_widget.is_mask_preview_active():
                    self.canvas.set_refined_mask(None)
                    self.canvas.set_mask_visible(False)

        if render_preview:
            self.update_stretch_preview(immediate=True)

    def _update_undo_redo_buttons(self):
        """Actualiza la disponibilidad visual y tooltips de los botones Deshacer y Rehacer."""
        has_undo = len(self.undo_stack) > 0
        has_redo = len(self.redo_stack) > 0
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setEnabled(has_undo)
            if has_undo:
                last_type = self.undo_stack[-1].get("type", "param")
                desc = "recorte" if last_type == "crop" else ("separación StarNet" if last_type == "starnet_separation" else ("máscara" if last_type == "mask" else ("preset" if last_type == "preset" else "ajuste")))
                self.btn_undo.setToolTip(f"Deshacer último {desc} (Ctrl+Z) — {len(self.undo_stack)} pasos en historial")
            else:
                self.btn_undo.setToolTip("Deshacer (Ctrl+Z) — Sin acciones para deshacer")

        if hasattr(self, 'btn_redo'):
            self.btn_redo.setEnabled(has_redo)
            if has_redo:
                next_type = self.redo_stack[-1].get("type", "param")
                desc = "recorte" if next_type == "crop" else ("separación StarNet" if next_type == "starnet_separation" else ("máscara" if next_type == "mask" else ("preset" if next_type == "preset" else "ajuste")))
                self.btn_redo.setToolTip(f"Rehacer {desc} (Ctrl+Y / Ctrl+Shift+Z) — {len(self.redo_stack)} pasos disponibles")
            else:
                self.btn_redo.setToolTip("Rehacer (Ctrl+Y) — Sin acciones para rehacer")

    # -------------------------------------------------------------
    # GESTOR DE PRESETS DE REVELADO (AJUSTES PREESTABLECIDOS)
    # -------------------------------------------------------------
    def _populate_presets_combo(self, select_preset_id: Optional[str] = None):
        """Carga y categoriza los presets en el desplegable con separadores elegantes."""
        if not getattr(self, "combo_presets", None):
            return
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()

        builtins = get_builtin_presets()
        user_presets = load_user_presets()

        # Agrupar por categoría
        categories_dict: Dict[str, List[DeveloperPreset]] = {}
        for p in builtins:
            categories_dict.setdefault(p.category, []).append(p)
        for p in user_presets:
            categories_dict.setdefault(p.category, []).append(p)

        cat_order = ["Básicos", "Vía Láctea", "Paisaje Nocturno", "Cielo Profundo", "Corrección", "Estrellas", "Usuario"]
        sorted_cats = [c for c in cat_order if c in categories_dict] + [c for c in categories_dict if c not in cat_order]

        model = self.combo_presets.model()
        target_index = -1

        for cat in sorted_cats:
            # Cabecera de categoría visual no seleccionable
            self.combo_presets.addItem(f"── {cat.upper()} ──", None)
            hdr_idx = self.combo_presets.count() - 1
            if model:
                hdr_item = model.item(hdr_idx)
                if hdr_item:
                    hdr_item.setEnabled(False)
                    hdr_item.setForeground(QColor("#80d8ff"))
                    font = hdr_item.font()
                    font.setBold(True)
                    hdr_item.setFont(font)

            for p in categories_dict[cat]:
                self.combo_presets.addItem(f"  {p.name}", p)
                self.combo_presets.setItemIcon(self.combo_presets.count() - 1, get_icon("sliders" if p.is_builtin else "file", color="#38bdf8" if p.is_builtin else "#fbbf24", size=13))
                if select_preset_id and p.id == select_preset_id:
                    target_index = self.combo_presets.count() - 1

        self.combo_presets.blockSignals(False)

        if target_index >= 0:
            self.combo_presets.setCurrentIndex(target_index)
        else:
            # Seleccionar el primer preset seleccionable real (saltando la primera cabecera)
            for i in range(self.combo_presets.count()):
                if self.combo_presets.itemData(i, Qt.UserRole) is not None:
                    self.combo_presets.setCurrentIndex(i)
                    break

        self._on_preset_combo_changed(self.combo_presets.currentIndex())

    def _on_preset_combo_changed(self, index: int):
        """Actualiza la descripción al cambiar de preset en el desplegable."""
        if not getattr(self, "combo_presets", None):
            return
        preset = self.combo_presets.itemData(index, Qt.UserRole)
        if isinstance(preset, DeveloperPreset):
            if self.lbl_preset_desc:
                self.lbl_preset_desc.setText(preset.description or "")
            if self.btn_apply_preset:
                self.btn_apply_preset.setEnabled(True)
        else:
            if self.lbl_preset_desc:
                self.lbl_preset_desc.setText("")
            if self.btn_apply_preset:
                self.btn_apply_preset.setEnabled(False)

    def apply_selected_preset(self):
        """Aplica el preset actualmente seleccionado en el desplegable."""
        if not getattr(self, "combo_presets", None):
            return
        preset = self.combo_presets.currentData(Qt.UserRole)
        if not isinstance(preset, DeveloperPreset):
            return
        scope = self.combo_preset_scope.currentData() if self.combo_preset_scope else SCOPE_ALL
        intensity = getattr(self.panel_presets, 'spin_intensity', None)
        intensity_val = float(intensity.value()) if intensity is not None else 100.0
        self.apply_preset(preset, scope=scope, intensity_pct=intensity_val)

    def apply_preset(
        self,
        preset: DeveloperPreset,
        scope: str = SCOPE_ALL,
        intensity_pct: float = 100.0,
        auto_horizon: bool = True
    ):
        """Aplica los parámetros de un preset a la imagen actual registrando la acción en el historial de Deshacer."""
        if not preset or not preset.params:
            return

        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Carga o apila una imagen en el Revelador primero.")
            return

        # 0. Verificación de StarNet si el preset lo requiere
        effective_scope = scope
        if getattr(preset, "requires_starnet", False) and self.image_starless is None:
            msg = QMessageBox(self)
            msg.setWindowTitle("Separación Estelar Requerida")
            msg.setIcon(QMessageBox.Question)
            msg.setText(
                f"El preset <b>'{preset.name}'</b> incluye ajustes específicos para estrellas y fondo desacoplado.<br><br>"
                "Para obtener el 100% de su efecto estético, se requiere desacoplar las estrellas con <b>StarNet++ AI</b> (~15-30s).<br><br>"
                "¿Cómo deseas proceder?"
            )
            btn_run_starnet = msg.addButton("Ejecutar StarNet y Aplicar", QMessageBox.AcceptRole)
            btn_apply_partial = msg.addButton("Aplicar sin StarNet (Solo Color y Tono)", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_run_starnet)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_cancel:
                return
            elif clicked == btn_run_starnet:
                self._preset_pending_after_starnet = (preset, scope, intensity_pct)
                self.run_starnet()
                return

        # Capturamos el estado base y máscara ANTES de cualquier modificación del preset.
        base_state = getattr(self, "_state_before_preset_dialog", None)
        if base_state is None:
            base_state = self._last_committed_state.copy() if self._last_committed_state is not None else self._capture_state()
        else:
            base_state = base_state.copy()

        old_mask = getattr(self, "_mask_before_preset_dialog", None)
        if old_mask is None:
            old_mask = self.current_mask.copy() if self.current_mask is not None else None
        else:
            old_mask = old_mask.copy() if old_mask is not None else None

        old_mask_source = getattr(self, "_mask_source_before_preset_dialog", getattr(self, "_mask_source_name", ""))
        old_badge = getattr(self, "_badge_before_preset_dialog", None)
        if old_badge is None:
            old_badge = self.sec_presets.lbl_status.text() if hasattr(self, 'sec_presets') and hasattr(self.sec_presets, 'lbl_status') else ""

        # Limpiamos las referencias del diálogo
        self._state_before_preset_dialog = None
        self._mask_before_preset_dialog = None
        self._mask_source_before_preset_dialog = None
        self._badge_before_preset_dialog = None

        if hasattr(self, '_undo_debounce_timer'):
            self._undo_debounce_timer.stop()

        self._is_applying_preset = True
        self._is_previewing_preset = False
        try:
            # 1. Detección automática de horizonte si el preset lo requiere y no hay máscara activa
            if auto_horizon and getattr(preset, "requires_sky_mask", False) and self.current_mask is None:
                try:
                    self.log_message(f"[PRESET] El preset '{preset.name}' requiere separación cielo/suelo. Detectando horizonte con IA...")
                    from core.ai_horizon import auto_detect_sky_mask
                    mask, method = auto_detect_sky_mask(self.image_32bit)
                    if mask is not None:
                        self._apply_loaded_mask(mask, source_desc=f"Auto-Horizonte ({method})")
                        self.canvas.set_mask_visible(False)
                        self._set_mask_overlay_ui_state(False)
                        self.log_message(f"✓ [PRESET] Horizonte auto-detectado con éxito ({method}).")
                except Exception as e:
                    self.log_message(f"[AVISO PRESET] No se pudo auto-detectar el horizonte: {e}. Puedes asignarlo manualmente.")

            # 2. Si la imagen está en estado lineal crudo (negra), lanzar Auto-MTF adaptativo
            curr_mtf = getattr(getattr(self, 'state', None), 'mtf_val', 0.5)
            curr_bp = getattr(getattr(self, 'state', None), 'bp_val', 0.0)
            if curr_mtf >= 0.499 and curr_bp == 0.0:
                self.log_message("[PRESET] Imagen lineal detectada: aplicando Auto-MTF adaptativo de arranque...")
                self.apply_auto_mtf()

            # 3. Modulación con el slider de intensidad y filtrado de ámbito
            from core.developer_presets import interpolate_preset_params
            target_state = interpolate_preset_params(
                base_state,
                preset.params,
                intensity_pct=intensity_pct,
                scope=effective_scope
            )

            # 4. Restaurar parámetros en todos los controles
            self._restore_state(target_state)
            self._last_committed_state = self._capture_state()

            # 5. Registrar exactamente UNA acción atómica en el historial de Deshacer
            undo_item = {
                "type": "preset",
                "preset_name": preset.name,
                "params": base_state,
                "mask": old_mask,
                "mask_source_name": old_mask_source,
                "badge_text": old_badge,
            }
            self.undo_stack.append(undo_item)
            if len(self.undo_stack) > self.MAX_UNDO_STEPS:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self._mark_dirty()
            self._update_undo_redo_buttons()

            # 6. Mensaje explicativo en el log y badge de estado
            scope_names = {
                SCOPE_ALL: "Todo (Sin estirado)",
                SCOPE_TONE: "Solo Contraste y Suelo",
                SCOPE_COLOR: "Solo Color",
                SCOPE_DETAILS: "Solo Estrellas/Detalles"
            }
            scope_str = scope_names.get(effective_scope, effective_scope)
            intensity_str = f" al {int(intensity_pct)}%" if abs(intensity_pct - 100.0) > 1.0 else ""
            self.log_message(f"[PRESET] Preset aplicado: '{preset.name}' [{scope_str}]{intensity_str}")

            if hasattr(self, 'sec_presets'):
                if preset and preset.name and "Neutro" not in preset.name:
                    badge_title = preset.name.split("-")[0].strip() if "-" in preset.name else preset.name
                    for em in ("⭐", "🌌", "✨", "🏞️", "🔭", "🏙️", "🌫️", "👤"):
                        badge_title = badge_title.replace(em, "")
                    badge_title = badge_title.strip()
                    self.sec_presets.set_status_badge(f"● {badge_title}{intensity_str}", "#80d8ff")
                else:
                    self.sec_presets.set_status_badge("")
        finally:
            self._is_applying_preset = False
            if hasattr(self, '_undo_debounce_timer'):
                self._undo_debounce_timer.stop()

    def open_save_preset_dialog(self):
        """Abre el diálogo para guardar los ajustes actuales como un preset."""
        current_params = self._capture_state(include_mask_layers=False)
        builtins = get_builtin_presets()
        users = load_user_presets()
        categories = list(set(p.category for p in builtins + users))

        dlg = SavePresetDialog(current_params, categories, self)
        if dlg.exec() == QDialog.Accepted and dlg.created_preset:
            if hasattr(self, '_populate_presets_combo'):
                self._populate_presets_combo(select_preset_id=dlg.created_preset.id)
            self.log_message(f"[PRESET] Nuevo preset guardado con éxito: '{dlg.created_preset.name}' ({dlg.created_preset.category})")

    def open_preset_manager_dialog(self):
        """Abre la galería visual y gestor de presets con soporte de previsualización en vivo e intensidad."""
        if hasattr(self, '_undo_debounce_timer'):
            self._undo_debounce_timer.stop()

        current_params = self._capture_state()
        self._state_before_preset_dialog = current_params.copy()
        self._mask_before_preset_dialog = self.current_mask.copy() if self.current_mask is not None else None
        self._mask_source_before_preset_dialog = getattr(self, '_mask_source_name', "")
        self._badge_before_preset_dialog = self.sec_presets.lbl_status.text() if hasattr(self, 'sec_presets') and hasattr(self.sec_presets, 'lbl_status') else ""

        dlg = PresetManagerDialog(current_params, self)
        dlg.preview_preset_requested.connect(self._on_preset_dialog_preview)
        dlg.preview_cancelled.connect(self._on_preset_dialog_cancel)
        dlg.apply_preset_requested.connect(lambda p, s, intensity: self.apply_preset(p, s, intensity_pct=intensity))

        self._is_previewing_preset = True
        try:
            result = dlg.exec()
            if result != QDialog.Accepted:
                self._on_preset_dialog_cancel()
        finally:
            self._is_previewing_preset = False
            self._state_before_preset_dialog = None
            self._mask_before_preset_dialog = None
            self._mask_source_before_preset_dialog = None
            self._badge_before_preset_dialog = None
            if hasattr(self, '_undo_debounce_timer'):
                self._undo_debounce_timer.stop()

        if hasattr(self, "combo_presets") and self.combo_presets is not None:
            curr_preset = self.combo_presets.currentData(Qt.UserRole)
            curr_id = curr_preset.id if isinstance(curr_preset, DeveloperPreset) else None
            self._populate_presets_combo(select_preset_id=curr_id)

    def _on_preset_dialog_preview(self, preset: DeveloperPreset, scope: str, intensity_pct: float):
        """Previsualiza un preset temporalmente en el lienzo principal sin guardarlo en el historial."""
        if not preset or not preset.params:
            return
        base = getattr(self, "_state_before_preset_dialog", None)
        if base is None:
            base = self._capture_state()
            self._state_before_preset_dialog = base.copy()

        from core.developer_presets import interpolate_preset_params
        preview_state = interpolate_preset_params(
            base,
            preset.params,
            intensity_pct=intensity_pct,
            scope=scope
        )
        self._is_previewing_preset = True
        try:
            self._restore_state(preview_state)
        finally:
            self._is_previewing_preset = True

    def _on_preset_dialog_cancel(self):
        """Restaura el estado previo del revelador al cancelar o cerrar el diálogo."""
        saved_state = getattr(self, "_state_before_preset_dialog", None)
        if saved_state is not None:
            self._is_previewing_preset = True
            try:
                self._restore_state(saved_state)
                if hasattr(self, '_mask_before_preset_dialog'):
                    old_mask = self._mask_before_preset_dialog
                    self.current_mask = old_mask.copy() if old_mask is not None else None
                    self._mask_source_name = getattr(self, '_mask_source_before_preset_dialog', "")
                    self.canvas.set_refined_mask(self.current_mask)
                    self.canvas.set_mask_visible(self.current_mask is not None)
                    self._update_mask_status_ui()
                if hasattr(self, 'sec_presets') and getattr(self, '_badge_before_preset_dialog', None) is not None:
                    badge_text = self._badge_before_preset_dialog
                    self.sec_presets.set_status_badge(badge_text, "#80d8ff" if badge_text else "")
                self._last_committed_state = self._capture_state()
            finally:
                if hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.stop()

    # =========================================================================
    # GESTIÓN DE PROYECTOS Y SESIONES (.devloom)
    # =========================================================================
    def _mark_dirty(self):
        """Marca el proyecto como modificado con cambios pendientes de guardar."""
        if not self._is_dirty:
            self._is_dirty = True
            self._update_project_ui_status()

    def _update_project_ui_status(self):
        """Actualiza la etiqueta del proyecto y el indicador de cambios pendientes (*)."""
        if not hasattr(self, 'lbl_project_title'):
            return
        name = os.path.basename(self.current_project_path) if self.current_project_path else "Sin guardar"
        dirty_star = " *" if self._is_dirty else ""
        self.lbl_project_title.setText(f"Proyecto: {name}{dirty_star}")
        if self.current_project_path:
            self.lbl_project_title.setToolTip(f"{self.current_project_path}{' (cambios no guardados)' if self._is_dirty else ''}")
        else:
            self.lbl_project_title.setToolTip("Proyecto no guardado")

    def _check_unsaved_changes(self) -> bool:
        """
        Verifica si hay cambios no guardados en el proyecto.
        Retorna True si es seguro proceder (guardó o descartó), o False si el usuario canceló.
        """
        if not self._is_dirty or self.image_32bit is None:
            return True
        proj_name = os.path.basename(self.current_project_path) if self.current_project_path else "Sin guardar"
        reply = QMessageBox.question(
            self,
            "Cambios no guardados",
            f"El proyecto '{proj_name}' tiene cambios sin guardar.\n\n¿Deseas guardarlos antes de continuar?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )
        if reply == QMessageBox.Save:
            return self.save_project()
        elif reply == QMessageBox.Discard:
            return True
        else:
            return False

