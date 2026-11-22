# gui/developer/io_project_manager.py
from __future__ import annotations
import os
import gc
import json
import zipfile
import io
import traceback
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import tifffile
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from core.log_hub import log_hub
from core.stacking import load_image_as_float32
from core.config_manager import load_config, get_config_val
from core.icc_manager import save_tiff_with_icc, normalize_colorspace_name, get_icc_profile_bytes
from core.stretch import apply_ghs_stretch, manual_stretch
from core.star_diffusion import apply_star_diffusion


class ProjectIOMixin:
    """Mixin para gestión de I/O de imágenes, exportación multicapa y proyectos .devloom."""

    def load_image_direct(self, filepath: str, mask: np.ndarray = None):
        # 1. Cancelar workers previos si seguían ejecutándose
        if getattr(self, 'ground_ai_denoise_worker', None) and self.ground_ai_denoise_worker.isRunning():
            self.ground_ai_denoise_worker.cancel()
        self.ground_ai_denoise_worker = None

        if getattr(self, 'ai_denoise_worker', None) and self.ai_denoise_worker.isRunning():
            self.ai_denoise_worker.cancel()
        self.ai_denoise_worker = None

        # 2. Limpieza completa de datos de imagen y cachés de IA previos
        self._invalidate_pipeline_caches()
        self.active_filepath = filepath
        if mask is not None:
            m_norm = mask[..., 0].astype(np.float32) if mask.ndim == 3 else mask.astype(np.float32)
            if m_norm.max() > 1.05:
                m_norm /= 255.0
            self.current_mask = np.ascontiguousarray(np.clip(m_norm, 0.0, 1.0), dtype=np.float32)
        else:
            self.current_mask = None
        self.image_starless = None
        self.raw_linear_starless = None
        self.starless_is_stretched = False
        self.image_stars = None
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None
        self.preview_proxy = None
        self.draft_proxy = None
        self.draft_starless = None
        self.draft_stars = None
        self.draft_mask = None

        self._cache_ground_ai = None
        self._cache_starless_ai = None
        self.gnd_ai_denoise_enabled = False
        self.starless_ai_denoise_enabled = False
        self.gnd_denoise_val = 0.0

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
            self.canvas.clear_all_masking()

        self.reset_all_parameters()
        self._set_starnet_tools_enabled(False)
        self.combo_layer.setEnabled(False)
        self.slider_stars.setEnabled(False)
        
        self.log_message(f"Cargando imagen: {os.path.basename(filepath)}...")
        self.slider_contrast_starless.setEnabled(False)
        self.slider_contrast_starless.setValue(0)
        self.contrast_starless_val = 0.0
        self.lbl_contrast_starless.setText("0.00")        
        if hasattr(self, 'curve_widget'):
            self.curve_widget.reset_all_curves()
        self.slider_ond_est.setEnabled(False)
        self.slider_ond_bg.setEnabled(False)
        
        try:
            from core.config_manager import get_config_val
            from core.icc_manager import (
                extract_icc_profile,
                identify_colorspace_from_icc,
                normalize_colorspace_name
            )
            configured_cs = normalize_colorspace_name(get_config_val("icc_profile", "srgb"))
            embedded_icc = extract_icc_profile(filepath) if filepath.lower().endswith(('.tif', '.tiff')) else None
            if embedded_icc:
                self.current_color_space = identify_colorspace_from_icc(embedded_icc)
            else:
                self.current_color_space = configured_cs

            raw_img = load_image_as_float32(filepath, color_space=self.current_color_space)
            
            # Saneamiento preventivo de NaNs e infinitos
            if np.isnan(raw_img).any() or np.isinf(raw_img).any():
                self.log_message("[AVISO] La imagen contenía píxeles indefinidos (NaN/Inf). Han sido saneados automáticamente.")
                raw_img = np.nan_to_num(raw_img, nan=0.0, posinf=1.0, neginf=0.0)

            self.image_32bit = np.clip(raw_img, 0.0, 1.0).astype(np.float32)
            self.canvas.set_native_image_size((self.image_32bit.shape[1], self.image_32bit.shape[0]))

            is_loading = getattr(self, '_is_loading_project', False)
            if not is_loading:
                self._generate_preview_proxy()
                self.apply_auto_mtf()
                self.initial_bp = self.panel_stretch.bp_val if hasattr(self, 'panel_stretch') else 0.0
                self.initial_mtf = self.panel_stretch.mtf_val if hasattr(self, 'panel_stretch') else 0.15
                self.has_custom_snapshot = False
                self.snapshot_before_pixmap = None
                self._generate_before_reference()
            self._update_canvas_info_label()

            if self.current_mask is not None:
                self.canvas.set_refined_mask(self.current_mask)
            else:
                self.canvas.set_refined_mask(None)
            self._update_mask_status_ui()

            # Asegurar que se crea la capa Fondo (Imagen Base) en el Gestor de Capas
            if not is_loading and hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.ensure_base_layer()

            # Inicializar historial de Deshacer / Rehacer limpio para la nueva imagen
            if hasattr(self, '_undo_debounce_timer'):
                self._undo_debounce_timer.stop()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._last_committed_state = self._capture_state()
            self._update_undo_redo_buttons()

            self.log_message(f"Imagen en memoria ({self.image_32bit.shape[1]}x{self.image_32bit.shape[0]} px). Vista acelerada activa.")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo cargar: {e}")
            QMessageBox.critical(self, "Error", f"Fallo al abrir archivo:\n{e}")

    def load_coronal_stack(self, filepath: str, raster_layers: list) -> None:
        """
        Carga la imagen base HDR de eclipse y apila los filtros coronales calculados
        como capas ráster de imagen independientes con modos de fusión y opacidades.
        """
        self.load_image_direct(filepath, mask=None)

        if hasattr(self, 'mask_layers_widget') and raster_layers:
            # Las capas raster_layers vienen ordenadas de base a cima:
            # [NRGF, SWAP, Larson, Steerable, NAFE, WOW, MGN]
            # Al insertarlas secuencialmente en position=0, la última (MGN) quedará en el TOP absoluto
            for layer_dict in raster_layers:
                name = layer_dict.get("name", "Filtro Coronal")
                img_data = layer_dict.get("image_data")
                blend_mode = layer_dict.get("blend_mode", "normal")
                opacity = float(layer_dict.get("opacity", 1.0))
                if img_data is not None:
                    self.mask_layers_widget.add_raster_layer(
                        name=name,
                        image_data=img_data,
                        blend_mode=blend_mode,
                        opacity=opacity,
                        position=0
                    )

            self.update_stretch_preview(immediate=True)
            self._last_committed_state = self._capture_state()
            self._update_undo_redo_buttons()
            self.log_message(f"Multicapa coronal transferida con éxito ({len(raster_layers)} capas ráster activas).")

    def _on_import_raster_layer(self) -> None:
        """Abre un diálogo de archivo para cargar una imagen externa como nueva capa ráster (H-Alfa, Suelo, etc.)."""
        if self.image_32bit is None:
            QMessageBox.information(
                self, "Sin Imagen Base",
                "Debes abrir primero una imagen base en el Revelador antes de añadir capas ráster de composición."
            )
            return

        filtros = (
            "Imágenes Astronómicas y RAW (*.tif *.tiff *.fits *.fit *.nef *.cr2 *.cr3 *.arw *.dng *.png *.jpg *.jpeg);;"
            "Archivos TIFF / FITS (*.tif *.tiff *.fits *.fit);;"
            "Archivos RAW (*.nef *.cr2 *.cr3 *.arw *.dng);;"
            "Todos los archivos (*.*)"
        )
        cfg = load_config()
        default_dir = cfg.get("stacked_dir", os.getcwd())
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Cargar Capa de Imagen Ráster (H-Alfa, Suelo Hora Azul, Cometa...)", default_dir, filtros
        )
        if not filepath:
            return

        try:
            self.log_message(f"Cargando capa ráster: {os.path.basename(filepath)}...")
            raw_img = load_image_as_float32(filepath, color_space=self.current_color_space)
            if raw_img is None:
                QMessageBox.warning(self, "Error de Carga", f"No se pudo cargar la imagen:\n{filepath}")
                return

            if np.isnan(raw_img).any() or np.isinf(raw_img).any():
                raw_img = np.nan_to_num(raw_img, nan=0.0, posinf=1.0, neginf=0.0)

            if raw_img.ndim == 2:
                raw_img = np.repeat(raw_img[..., np.newaxis], 3, axis=-1)
            elif raw_img.ndim == 3 and raw_img.shape[2] == 1:
                raw_img = np.repeat(raw_img[..., np.newaxis], 3, axis=-1)

            # Redimensionado adaptativo si difiere del tamaño de la imagen base
            target_h, target_w = self.image_32bit.shape[:2]
            if raw_img.shape[:2] != (target_h, target_w):
                h_r, w_r = raw_img.shape[:2]
                ar_raw = float(w_r) / float(h_r)
                ar_target = float(target_w) / float(target_h)
                if abs(ar_raw - ar_target) < 0.02:
                    self.log_message(
                        f"Ajustando dimensiones de la capa ráster ({w_r}x{h_r} -> {target_w}x{target_h})..."
                    )
                    raw_img = cv2.resize(raw_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    self.log_message(
                        f"Encajando capa ráster preservando aspect ratio ({w_r}x{h_r} en lienzo {target_w}x{target_h})..."
                    )
                    scale = min(float(target_w) / float(w_r), float(target_h) / float(h_r))
                    new_w = max(1, int(round(w_r * scale)))
                    new_h = max(1, int(round(h_r * scale)))
                    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    fitted = cv2.resize(raw_img, (new_w, new_h), interpolation=interp)

                    canvas_layer = np.zeros((target_h, target_w, 3), dtype=np.float32)
                    off_x = (target_w - new_w) // 2
                    off_y = (target_h - new_h) // 2
                    canvas_layer[off_y:off_y + new_h, off_x:off_x + new_w] = fitted
                    raw_img = canvas_layer

            layer_name = os.path.splitext(os.path.basename(filepath))[0]
            layer = self.mask_layers_widget.add_raster_layer(
                name=layer_name,
                image_data=np.clip(raw_img, 0.0, 1.0).astype(np.float32),
                blend_mode="normal",
                opacity=1.0,
                position=0
            )

            self.update_stretch_preview(immediate=True)
            self._commit_action()
            self.log_message(f"Capa ráster '{layer.name}' añadida con éxito.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error al Cargar Capa Ráster", f"Error inesperado al importar la imagen:\n{str(e)}")

    def _on_master_mask_updated_from_raster(self, new_mask: np.ndarray) -> None:
        """Actualiza la máscara maestra de horizonte de la escena a partir de la detección realizada en una capa ráster."""
        if new_mask is None or self.image_32bit is None:
            return
        h, w = self.image_32bit.shape[:2]
        if new_mask.shape[:2] != (h, w):
            new_mask = cv2.resize(new_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        self.current_mask = np.ascontiguousarray(np.clip(new_mask, 0.0, 1.0), dtype=np.float32)
        self._mask_source_name = "Hora Azul (Capa Ráster)"
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
        self._update_mask_status_ui()
        self.log_message("[MÁSCARA] Horizonte maestro de la escena sincronizado desde Capa Ráster (Hora Azul).")

    def open_image_file(self):
        if not self._check_unsaved_changes():
            return
        filtros = (
            "Imágenes Astronómicas y RAW (*.tif *.tiff *.fits *.fit *.nef *.cr2 *.cr3 *.arw *.dng);;"
            "Archivos RAW (*.nef *.cr2 *.cr3 *.arw *.dng);;"
            "Archivos TIFF (*.tif *.tiff);;"
            "Archivos FITS (*.fits *.fit)"
        )
        cfg = load_config()
        default_dir = cfg.get("stacked_dir", os.getcwd())

        p, _ = QFileDialog.getOpenFileName(
            self, "Abrir Imagen de Astronomía o RAW", default_dir, filtros
        )
        if p:
            self.load_image_direct(p)
            self.current_project_path = None
            self._is_dirty = False
            self._update_project_ui_status()

    def _on_canvas_files_dropped(self, paths: list):
        """Carga automáticamente la primera imagen arrastrada sobre el lienzo."""
        if paths:
            first_path = paths[0]
            if first_path.lower().endswith((".devloom", ".loom")):
                if not self._check_unsaved_changes():
                    return
                self.load_project(first_path)
            else:
                if not self._check_unsaved_changes():
                    return
                self.load_image_direct(first_path)
                self.current_project_path = None
                self._is_dirty = False
                self._update_project_ui_status()

    def export_image(self, force_format: Optional[str] = None):
        if self.image_32bit is None:
            return

        filtros = (
            "TIFF 16-bit (*.tif *.tiff);;"
            "TIFF 32-bit Float (*.tif *.tiff);;"
            "Photoshop por Capas (*.psd);;"
            "JPEG (*.jpg *.jpeg)"
        )
        if force_format == "psd":
            filtro_por_defecto = "Photoshop por Capas (*.psd)"
            default_filename = "revelado_capas.psd"
        else:
            filtro_por_defecto = "TIFF 16-bit (*.tif *.tiff)"
            default_filename = "revelado_final.tif"
        
        cfg = load_config()
        default_dir = cfg.get("export_dir", os.getcwd())
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, default_filename)

        p, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Exportar a Photoshop por Capas" if force_format == "psd" else "Exportar Revelado", 
            default_path, 
            filtros, 
            selectedFilter=filtro_por_defecto
        )
        if not p:
            return

        p = os.path.abspath(os.path.normpath(str(p).strip()))

        # Asegurar extensión adecuada según lo introducido por el usuario o el filtro seleccionado
        ext = os.path.splitext(p)[1].lower()
        if not ext:
            if "Photoshop" in selected_filter or "*.psd" in selected_filter:
                ext = ".psd"
            elif "JPEG" in selected_filter or "*.jpg" in selected_filter:
                ext = ".jpg"
            else:
                ext = ".tif"
            p += ext

        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError as err:
                QMessageBox.critical(
                    self, "Error de Exportación",
                    f"No se puede sobreescribir el archivo destino porque está abierto en otro programa:\n{p}\n\nDetalle: {err}"
                )
                return

        h_full, w_full = self.image_32bit.shape[:2]
        self.set_canvas_progress(5, "Iniciando exportación...")
        self.log_message(f"Exportando imagen completa ({w_full}x{h_full} px)...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()

        try:
            self.set_canvas_progress(10, "Componiendo base...")
            full_base, _ = self._compose_active_base(for_export=True)

            full_mask = None
            clean_m = None
            if self.current_mask is not None:
                clean_m = np.squeeze(self.current_mask)
                if clean_m.ndim > 2:
                    clean_m = clean_m[..., 0]
                clean_m = clean_m.astype(np.float32)

                if clean_m.shape != (h_full, w_full):
                    clean_m = cv2.resize(clean_m, (w_full, h_full), interpolation=cv2.INTER_LINEAR)

                ksize = int(max(15, (min(h_full, w_full) // 150) | 1))
                if ksize % 2 == 0:
                    ksize += 1
                sm = cv2.GaussianBlur(clean_m, (ksize, ksize), sigmaX=ksize / 3.0)
                full_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)

            self.set_canvas_progress(18, "Renderizando compuesto base...")
            processed = self._apply_pipeline_on_image(full_base, precomputed_mask=full_mask, is_preview=False)
            self.set_canvas_progress(35, "Compuesto listo")

            is_psd = (ext == '.psd') or ("Photoshop" in selected_filter and ext not in ['.tif', '.tiff', '.jpg', '.jpeg'])
            if is_psd:
                from core.psd_export import export_developer_psd
                from core.config_manager import get_config_val
                from core.icc_manager import get_icc_profile_bytes, normalize_colorspace_name

                self.set_canvas_progress(40, "Preparando capas 16-bit...")
                self.log_message("[PSD] Preparando capas reveladas y estiradas en 16-bit...")

                def _render_base():
                    return self._apply_pipeline_on_image(
                        self.image_32bit, 
                        precomputed_mask=None, 
                        apply_mask_layers=False,
                        is_preview=False
                    )

                def _render_ground():
                    if clean_m is not None and full_mask is not None:
                        zero_mask = np.zeros_like(full_mask)
                        return self._apply_pipeline_on_image(
                            self.image_32bit, 
                            precomputed_mask=zero_mask, 
                            apply_mask_layers=False,
                            is_preview=False
                        )
                    return None

                def _render_starless():
                    if self.image_starless is not None:
                        starless_full = self._process_starless_base(
                            self.image_starless,
                            mask_3d=full_mask,
                            is_full_res=True
                        )
                        return self._apply_pipeline_on_image(
                            starless_full,
                            precomputed_mask=full_mask if full_mask is not None else None,
                            apply_mask_layers=False,
                            is_preview=False
                        )
                    return None

                def _render_stars():
                    if self.image_starless is not None and self.image_stars is not None:
                        st_img = self.image_stars.copy()
                        st_params = self.panel_stretch.get_state() if hasattr(self, 'panel_stretch') and self.panel_stretch is not None else {}
                        stretch_mode = st_params.get("stretch_mode", "mtf")
                        
                        if stretch_mode == "ghs":
                            st_stretched = np.clip(
                                apply_ghs_stretch(
                                    st_img,
                                    sp=st_params.get("ghs_sp_val", 0.02),
                                    d=st_params.get("ghs_d_val", 5.0),
                                    b=st_params.get("ghs_b_val", 2.0),
                                    hp=st_params.get("ghs_hp_val", 0.0),
                                    lp=st_params.get("ghs_lp_val", 0.0),
                                    bp=0.0,
                                    color_mode="rgb"
                                ) * self.star_intensity,
                                0.0, 1.0
                            )
                        else:
                            st_stretched = np.clip(
                                manual_stretch(st_img, black_point=0.0, midtone=st_params.get("mtf_val", 0.15)) * self.star_intensity,
                                0.0, 1.0
                            )

                        if self.star_diff_strength > 1e-4:
                            st_stretched = apply_star_diffusion(
                                st_stretched,
                                strength=self.star_diff_strength,
                                radius=self.star_diff_radius,
                                threshold=self.star_diff_threshold,
                                color_boost=self.star_diff_color_boost,
                                halo_mode=self.star_diff_mode,
                                sky_mask=clean_m if clean_m is not None else None,
                                stars_layer=st_stretched
                            )

                        if hasattr(self, 'mask_layers_widget'):
                            stars_layer = next((l for l in self.mask_layers_widget.layers if getattr(l, 'mask_type', '') == 'image_stars'), None)
                            if stars_layer is not None:
                                has_adj = (abs(stars_layer.temp) > 1e-4 or abs(stars_layer.tint) > 1e-4 or 
                                           abs(stars_layer.exposure) > 1e-3 or abs(stars_layer.contrast) > 1e-3 or 
                                           abs(stars_layer.saturation - 1.0) > 1e-3)
                                if has_adj:
                                    from core.mask_layers import apply_local_adjustments_to_image
                                    st_stretched = apply_local_adjustments_to_image(st_stretched, np.ones((h_full, w_full), dtype=np.float32), stars_layer)

                        if clean_m is not None:
                            st_stretched = st_stretched * clean_m[..., np.newaxis]

                        return np.ascontiguousarray(st_stretched, dtype=np.float32)
                    return None

                from concurrent.futures import ThreadPoolExecutor, as_completed
                tasks = {"base": _render_base}
                if clean_m is not None and full_mask is not None:
                    tasks["ground"] = _render_ground
                if self.image_starless is not None:
                    tasks["starless"] = _render_starless
                    if self.image_stars is not None:
                        tasks["stars"] = _render_stars

                max_w = min(4, len(tasks))
                task_names = {
                    "base": "Capa base",
                    "ground": "Capa suelo",
                    "starless": "Fondo starless",
                    "stars": "Capa estrellas"
                }
                results = {}
                with ThreadPoolExecutor(max_workers=max_w) as executor:
                    fut_map = {executor.submit(fn): k for k, fn in tasks.items()}
                    done = 0
                    for fut in as_completed(fut_map):
                        k = fut_map[fut]
                        results[k] = fut.result()
                        done += 1
                        pct = 40 + int(28 * (done / len(tasks)))
                        self.set_canvas_progress(pct, f"Listo: {task_names.get(k, k)}")
                        QCoreApplication.processEvents()

                stretched_base = results.get("base")
                ground_dev = results.get("ground")
                stretched_starless = results.get("starless")
                stretched_stars = results.get("stars")

                # Obtener bytes del perfil de color ICC seleccionado en configuración o detectado
                target_cs = getattr(self, 'current_color_space', normalize_colorspace_name(get_config_val("icc_profile", "srgb")))
                icc_bytes = get_icc_profile_bytes(target_cs)

                adj_layers = self.mask_layers_widget.layers if hasattr(self, 'mask_layers_widget') else None
                ok, desc, n_layers = export_developer_psd(
                    p,
                    composite_image=processed,
                    base_image=stretched_base,
                    starless_image=stretched_starless,
                    stars_image=stretched_stars,
                    sky_mask=clean_m if self.current_mask is not None else None,
                    ground_image=ground_dev,
                    adjustment_layers=adj_layers,
                    depth=16,
                    icc_profile_bytes=icc_bytes,
                    log_fn=self.log_message,
                    progress_fn=self.set_canvas_progress
                )
                if not ok:
                    raise RuntimeError(desc)
                self.log_message(f"Imagen PSD guardada en: {os.path.basename(p)} ({desc})")
                QMessageBox.information(
                    self, "Exportación a Photoshop (PSD)",
                    f"Archivo PSD multicapa guardado con éxito:\n{p}\n\n"
                    f"Capas generadas: {n_layers} (16-bit con Máscaras de Capa nativas)\n"
                    f"Resolución: {w_full}x{h_full} px\n\n"
                    f"Estructura del proyecto en Photoshop:\n"
                    f"• 00 - Revelado Final (Referencia oculta para comparar)\n"
                    f"• 04 - Suelo / Paisaje (con Máscara de Capa nativa de suelo)\n"
                    f"• 03 - Estrellas (con Máscara de Capa nativa de cielo, modo Trama)\n"
                    f"• 02 - Fondo Starless (con Máscara de Capa nativa de cielo)\n"
                    f"• 01 - Imagen Base Revelada\n"
                    f"• Perfil de color ICC incrustado ({target_cs.upper()})"
                )
                return
            elif ext in ['.jpg', '.jpeg']:
                self.set_canvas_progress(75, "Guardando JPEG...")
                bgr8 = cv2.cvtColor((np.clip(processed, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(p, bgr8, [cv2.IMWRITE_JPEG_QUALITY, 96])
                desc = "JPEG 8-bit"
                self.set_canvas_progress(100, "JPEG completado")
            elif "32-bit" in selected_filter:
                self.set_canvas_progress(75, "Guardando TIFF 32-bit...")
                out32 = np.ascontiguousarray(np.clip(processed, 0.0, 1.0), dtype=np.float32)
                src_cs = getattr(self, 'current_color_space', 'srgb')
                save_tiff_with_icc(p, out32, compression='zlib', photometric='rgb', source_color_space=src_cs)
                desc = "TIFF 32-bit float (ICC)"
                self.set_canvas_progress(100, "TIFF guardado")
            else:
                self.set_canvas_progress(75, "Guardando TIFF 16-bit...")
                u16 = np.empty(processed.shape, dtype=np.uint16)
                np.clip(processed * 65535.0 + 0.5, 0, 65535, out=u16, casting='unsafe')
                src_cs = getattr(self, 'current_color_space', 'srgb')
                save_tiff_with_icc(p, u16, compression='zlib', photometric='rgb', source_color_space=src_cs)
                desc = "TIFF 16-bit (ICC)"
                self.set_canvas_progress(100, "TIFF guardado")

            from core.config_manager import get_config_val
            from core.icc_manager import normalize_colorspace_name
            target_cs = getattr(self, 'current_color_space', normalize_colorspace_name(get_config_val("icc_profile", "srgb")))
            icc_labels = {"srgb": "sRGB", "adobe": "Adobe RGB (1998)", "prophoto": "ProPhoto RGB"}
            icc_desc_str = icc_labels.get(target_cs, "sRGB")

            self.log_message(f"Imagen guardada en: {os.path.basename(p)} ({desc} [{icc_desc_str}])")
            QMessageBox.information(
                self, "Exportación",
                f"Guardada con éxito en resolución completa:\n{p}\n\nFormato: {desc} ({w_full}x{h_full} px)\nPerfil ICC: {icc_desc_str}"
            )
        except Exception as e:
            self.log_message(f"[ERROR EXPORT] Fallo al escribir el archivo: {e}")
            QMessageBox.critical(self, "Error de Exportación", f"Fallo al escribir el archivo:\n{e}")
        finally:
            self.hide_canvas_progress()
            self.unsetCursor()
            
    def save_project(self) -> bool:
        """Guarda el proyecto en el archivo actual (.devloom) o solicita ruta si es nuevo (Ctrl+S)."""
        if self.image_32bit is None:
            self.log_message("[AVISO] No hay ninguna imagen cargada para guardar como proyecto.")
            return False

        if self.current_project_path:
            return self._do_save_project(self.current_project_path)
        else:
            return self.save_project_as_dialog()

    def save_project_as_dialog(self) -> bool:
        """Abre el diálogo para guardar el proyecto con un nuevo nombre (Ctrl+Shift+S)."""
        if self.image_32bit is None:
            self.log_message("[AVISO] No hay ninguna imagen cargada para guardar como proyecto.")
            return False

        cfg = load_config()
        default_dir = cfg.get("developer_projects_dir") or cfg.get("stacked_dir", os.getcwd())
        default_name = "mi_revelado.devloom"
        if self.active_filepath:
            base_stem = os.path.splitext(os.path.basename(self.active_filepath))[0]
            default_name = f"{base_stem}.devloom"

        init_path = os.path.join(default_dir, default_name)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Proyecto de Revelado",
            init_path,
            "Proyecto AstroLoom (*.devloom *.loom);;Todos los archivos (*.*)"
        )
        if not filepath:
            return False

        if not filepath.lower().endswith((".devloom", ".loom")):
            filepath += ".devloom"

        return self._do_save_project(filepath)

    def _do_save_project(self, filepath: str) -> bool:
        """Ejecuta la serialización y compresión del proyecto .devloom con barra de progreso en el pie del lienzo."""
        try:
            self.setCursor(Qt.WaitCursor)
            self.log_message(f"--- Guardando proyecto: {os.path.basename(filepath)} ---")
            self.set_canvas_progress(5, "Iniciando guardado de proyecto...")
            QCoreApplication.processEvents()

            def on_progress(pct: int, msg: str):
                self.set_canvas_progress(pct, msg)
                self.log_message(f"[PROYECTO] {msg}")
                QCoreApplication.processEvents()

            # Recolectar datos (omitiendo capas en memoria ya que se serializan aparte)
            state_dict = self._capture_state(include_mask_layers=False)
            layers = self.mask_layers_widget.layers if hasattr(self, 'mask_layers_widget') else []
            master_mask = self.current_mask

            starnet_data = None
            if getattr(self, 'image_starless', None) is not None or getattr(self, 'image_stars', None) is not None:
                starnet_data = {
                    "image_starless": getattr(self, 'image_starless', None),
                    "image_stars": getattr(self, 'image_stars', None),
                    "raw_linear_starless": getattr(self, 'raw_linear_starless', None),
                }

            cropped_base = None
            if self.active_filepath and os.path.exists(self.active_filepath) and self.image_32bit is not None and starnet_data is None:
                try:
                    import tifffile
                    with tifffile.TiffFile(self.active_filepath) as tf:
                        orig_shape = tf.pages[0].shape[:2]
                    if self.image_32bit.shape[:2] != orig_shape:
                        cropped_base = self.image_32bit
                except Exception:
                    pass

            from core.project_io import save_devloom_project
            ok = save_devloom_project(
                file_path=filepath,
                developer_state=state_dict,
                mask_layers=layers,
                master_sky_mask=master_mask,
                source_image_path=self.active_filepath,
                starnet_data=starnet_data,
                cropped_base_image=cropped_base,
                progress_callback=on_progress,
            )

            if ok:
                self.current_project_path = os.path.abspath(filepath)
                self._is_dirty = False
                self._update_project_ui_status()
                self.log_message(f"✓ Proyecto guardado correctamente: {os.path.basename(filepath)}")
                self.set_canvas_progress(100, "Proyecto guardado")
                return True
            else:
                self.log_message("[ERROR] No se pudo guardar el archivo de proyecto.")
                return False
        except Exception as e:
            self.log_message(f"[ERROR] Excepción al guardar proyecto: {e}")
            QMessageBox.critical(self, "Error al guardar", f"Fallo al guardar el proyecto:\n{e}")
            return False
        finally:
            self.hide_canvas_progress()
            self.unsetCursor()

    def open_project_dialog(self):
        """Abre un diálogo para seleccionar y cargar un proyecto .devloom (Ctrl+O)."""
        if not self._check_unsaved_changes():
            return

        cfg = load_config()
        default_dir = cfg.get("developer_projects_dir") or cfg.get("stacked_dir", os.getcwd())

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir Proyecto de Revelado",
            default_dir,
            "Proyecto AstroLoom (*.devloom *.loom);;Todos los archivos (*.*)"
        )
        if filepath:
            self.load_project(filepath)

    def load_project(self, filepath: str) -> bool:
        """Carga una sesión completa de desarrollo desde un archivo .devloom con barra de progreso."""
        from core.project_io import load_devloom_project, resolve_source_image_path
        self._is_loading_project = True
        progress = None
        try:
            self.setCursor(Qt.WaitCursor)
            self.log_message(f"--- Abriendo proyecto: {os.path.basename(filepath)} ---")

            progress = QProgressDialog("Abriendo proyecto...", None, 0, 100, self)
            progress.setWindowTitle("Cargar Proyecto AstroLoom")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setCancelButton(None)
            progress.setValue(5)
            progress.show()
            QCoreApplication.processEvents()

            def on_progress(pct: int, msg: str):
                if progress:
                    progress.setValue(pct)
                    progress.setLabelText(msg)
                self.log_message(f"[PROYECTO] {msg}")
                QCoreApplication.processEvents()

            proj_data = load_devloom_project(filepath, progress_callback=on_progress)
            source_info = proj_data.get("source_image", {})
            img_filename = source_info.get("filename") or "desconocida"

            # 1. Resolver ruta de la imagen base
            on_progress(30, f"Localizando imagen base: {img_filename}...")
            resolved_img_path = resolve_source_image_path(source_info, filepath)
            if not resolved_img_path:
                if progress:
                    progress.hide()
                self.unsetCursor()
                reply = QMessageBox.question(
                    self,
                    "Imagen base no encontrada",
                    f"No se pudo encontrar la imagen base vinculada al proyecto:\n\n"
                    f"  ▶ Archivo: \"{img_filename}\"\n\n"
                    f"¿Deseas buscarla manualmente en tu equipo?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    filtros = (
                        "Imágenes Astronómicas y RAW (*.tif *.tiff *.fits *.fit *.nef *.cr2 *.cr3 *.arw *.dng);;"
                        "Archivos TIFF (*.tif *.tiff);;Archivos FITS (*.fits *.fit);;Todos los archivos (*.*)"
                    )
                    found_path, _ = QFileDialog.getOpenFileName(
                        self,
                        f"Localizar imagen base: {img_filename}",
                        os.path.dirname(filepath),
                        filtros
                    )
                    if found_path:
                        resolved_img_path = found_path
                    else:
                        self.log_message("[AVISO] Carga de proyecto cancelada: no se seleccionó ninguna imagen.")
                        return False
                else:
                    self.log_message(f"[AVISO] Carga de proyecto cancelada: imagen '{img_filename}' no disponible.")
                    return False
                if progress:
                    progress.show()
                self.setCursor(Qt.WaitCursor)

            # 2. Cargar imagen base para inicializar canvas, ICC y proxies
            on_progress(45, f"Cargando imagen en memoria ({img_filename})...")
            master_mask = proj_data.get("master_sky_mask")
            self.load_image_direct(resolved_img_path, mask=master_mask)

            # Restaurar base recortada previa si existía en el contenedor
            cropped_base = proj_data.get("cropped_base_image")
            if cropped_base is not None:
                self.image_32bit = np.ascontiguousarray(cropped_base, dtype=np.float32)
                self.canvas.set_native_image_size((self.image_32bit.shape[1], self.image_32bit.shape[0]))

            # 3. Restaurar capas StarNet si estaban empaquetadas
            starnet_dict = proj_data.get("starnet", {})
            sl_img = starnet_dict.get("image_starless")
            st_img = starnet_dict.get("image_stars")
            raw_sl_img = starnet_dict.get("raw_linear_starless")

            if sl_img is not None and st_img is not None:
                on_progress(65, "Restaurando capas desacopladas de StarNet...")
                self.image_starless = np.ascontiguousarray(sl_img, dtype=np.float32)
                self.image_stars = np.ascontiguousarray(st_img, dtype=np.float32)
                self.raw_linear_starless = np.ascontiguousarray(raw_sl_img, dtype=np.float32) if raw_sl_img is not None else self.image_starless.copy()
                self.starless_is_stretched = proj_data.get("starless_is_stretched", False)

                # Si las capas StarNet difieren dimensionalmente de la imagen base cargada
                # (por ejemplo, si se recortó la imagen en sesión antes de guardar el proyecto),
                # reconstruimos la base lineal 32-bit exacta a partir de las capas desacopladas.
                if self.image_32bit is None or self.image_32bit.shape[:2] != self.image_starless.shape[:2]:
                    base_reconstructed = self.raw_linear_starless + self.image_stars
                    self.image_32bit = np.clip(base_reconstructed, 0.0, 1.0).astype(np.float32)
                    self.canvas.set_native_image_size((self.image_32bit.shape[1], self.image_32bit.shape[0]))

                self._set_starnet_tools_enabled(True)
                self.combo_layer.setEnabled(True)
                self.slider_stars.setEnabled(True)
                self._cache_starless_ai = None

            # Sincronizar máscara maestra a las dimensiones de la imagen base activa
            if master_mask is not None and self.image_32bit is not None:
                if master_mask.shape[:2] != self.image_32bit.shape[:2]:
                    self.current_mask = cv2.resize(master_mask, (self.image_32bit.shape[1], self.image_32bit.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    self.current_mask = master_mask.copy()
                self.canvas.set_refined_mask(self.current_mask)
                self._update_mask_status_ui()

            self._generate_preview_proxy()

            # 4. Restaurar capas de ajuste y máscaras locales
            on_progress(80, "Restaurando pila de capas y máscaras...")
            raw_layers = proj_data.get("mask_layers", [])
            dev_state = proj_data.get("developer_state", {})
            dev_state["mask_layers"] = raw_layers

            # 5. Restaurar estado de controles y paneles
            on_progress(90, "Restaurando parámetros de revelado y curvas...")
            if dev_state:
                self._restore_state(dev_state, render_preview=False)

            # 6. Resetear historial de Deshacer/Rehacer al estado recién cargado
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._last_committed_state = self._capture_state()
            self._update_undo_redo_buttons()

            # 7. Actualizar estado de proyecto
            self.current_project_path = os.path.abspath(filepath)
            self._is_dirty = False
            self._update_project_ui_status()

            # 8. Forzar renderizado y refresco inmediato
            on_progress(98, "Renderizando vista previa...")
            self._is_loading_project = False
            self._invalidate_pipeline_caches()
            self.update_stretch_preview(immediate=True)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)

            on_progress(100, f"Proyecto cargado: {os.path.basename(filepath)}")
            self.log_message(f"✓ Proyecto cargado con éxito: {os.path.basename(filepath)}")
            return True

        except Exception as e:
            self.log_message(f"[ERROR] Error al cargar proyecto: {e}")
            QMessageBox.critical(self, "Error al cargar proyecto", f"No se pudo cargar el archivo:\n{e}")
            return False
        finally:
            self._is_loading_project = False
            if progress:
                progress.close()
            self.unsetCursor()
