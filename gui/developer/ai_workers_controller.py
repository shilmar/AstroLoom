# gui/developer/ai_workers_controller.py
from __future__ import annotations
import os
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QCoreApplication, QTimer
from PySide6.QtWidgets import QMessageBox, QDialog, QProgressDialog, QApplication

from core.log_hub import log_hub
from core.i18n import tr
from core.deconvolution import estimate_psf_from_stars, generate_moffat_psf, generate_gaussian_psf
from core.pcc import detect_stars_photometry, fit_theil_sen_pcc
from core.background_extraction import (
    generate_sky_sample_points,
    extract_background_rbf,
    auto_extract_background_sky
)
from core.light_pollution import estimate_light_pollution_center
from core.stretch import (
    calculate_sky_neutral_wb,
    calculate_sky_mtf_params,
    calculate_smart_ground_lift,
    apply_white_balance,
    auto_statistical_stretch,
    extract_background_polynomial
)
from core.ghs import estimate_ghs_params
from core.ai_horizon import auto_detect_sky_mask
from gui.worker import StarNetWorker, AIDenoiseWorker, ModelDownloadWorker


class AIWorkersMixin:
    """Mixin para orquestación de hilos de inferencia IA, descarga de modelos y calibración pesada."""

    def run_starnet(self):
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Carga o apila una imagen primero.")
            return

        if self.image_starless is not None:
            ret = QMessageBox.question(
                self,
                "Volver a ejecutar StarNet",
                "Ya existen capas desacopladas de estrellas en la sesión activa.\n\n"
                "¿Deseas volver a ejecutar StarNet y sobrescribir las capas actuales con los nuevos parámetros?\n\n"
                "(Podrás deshacer este cambio en cualquier momento con Ctrl+Z).",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if ret != QMessageBox.Yes:
                return

        from core.starnet_onnx import is_starnet_onnx_available
        if not is_starnet_onnx_available():
            msg = QMessageBox(self)
            msg.setWindowTitle("Modelo de IA StarNet v2 no encontrado")
            msg.setText(
                "El modelo de IA para separación de estrellas (starnet2.onnx, ~131 MB) "
                "no se encuentra instalado en 'models/'.\n\n"
                "¿Deseas descargarlo ahora automáticamente?"
            )
            btn_download = msg.addButton("Descargar IA (~131 MB)", QMessageBox.AcceptRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_download)
            msg.exec()

            if msg.clickedButton() == btn_download:
                self._start_starnet_model_download()
            return

        self._set_ai_processing_state(True)
        self.log_message("=== INICIANDO SEPARACIÓN STARNET++ AI ===")
        if hasattr(self, 'sec_masks'):
            self.sec_masks.expand()

        stride = self.spin_stride.value()
        upsample = bool(self.chk_upsample and self.chk_upsample.isChecked())

        progress = QProgressDialog("Inicializando StarNet++ AI...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Separación de Estrellas — StarNet++ AI")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        self.sn_worker = StarNetWorker(
            self.image_32bit, 
            sky_mask=self.current_mask, 
            stride=stride,
            upsample=upsample
        )

        def on_sn_prog(curr, total, msg):
            if total > 0:
                pct = int((curr / total) * 100)
                progress.setValue(pct)
            progress.setLabelText(msg)

        def on_finished(starless, stars):
            progress.close()
            self.on_starnet_success(starless, stars)

        def on_error(err):
            progress.close()
            self.on_starnet_error(err)

        def on_cancel():
            self.sn_worker.cancel()
            progress.close()
            self._set_ai_processing_state(False)
            self.log_message("[STARNET AI] Separación cancelada por el usuario.")

        self.sn_worker.status_changed.connect(self.log_message)
        self.sn_worker.progress_changed.connect(on_sn_prog)
        self.sn_worker.finished_success.connect(on_finished)
        self.sn_worker.error_occurred.connect(on_error)
        self.sn_worker.cancelled.connect(on_cancel)
        progress.canceled.connect(on_cancel)

        self.sn_worker.start()

    def on_starnet_success(self, starless_img: np.ndarray, stars_img: np.ndarray):
        self._set_ai_processing_state(False)
        if hasattr(self, '_undo_debounce_timer'):
            self._undo_debounce_timer.stop()

        # Snapshot de estado previo para Deshacer (Undo) de la separación StarNet
        undo_item = {
            "type": "starnet_separation",
            "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
            "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
            "starless_is_stretched": getattr(self, "starless_is_stretched", False),
            "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
            "mask_layers": [l.to_dict() for l in self.mask_layers_widget.layers] if hasattr(self, 'mask_layers_widget') else [],
        }
        self.undo_stack.append(undo_item)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

        # CORREGIDO: Usar self.current_mask en lugar de un inexistente self.sky_mask
        sky_mask = self.current_mask
        if sky_mask is not None and stars_img is not None:
            h, w = stars_img.shape[:2]
            mask = sky_mask
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            mask_3d = mask[..., np.newaxis] if mask.ndim == 2 else mask

            # Lo que StarNet creyó que eran estrellas en el suelo vuelve al fondo
            starless_img = starless_img + (stars_img * (1.0 - mask_3d))
            # Restringir estrellas estrictamente al cielo
            stars_img = stars_img * mask_3d

        self.image_starless = np.ascontiguousarray(starless_img, dtype=np.float32)
        self.raw_linear_starless = self.image_starless.copy()
        self.starless_is_stretched = False
        self.image_stars = np.ascontiguousarray(stars_img, dtype=np.float32)
        self._cache_starless_ai = None
        self.starless_ai_denoise_enabled = False
        if hasattr(self, 'slider_denoise') and self.slider_denoise.value() > 0:
            self.slider_denoise.setValue(0)
        else:
            self.denoise_strength = 0.0
        if hasattr(self, 'panel_starnet'):
            self.panel_starnet.set_ai_denoise_active(False)
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ai_denoise_active(False)

        # Integrar capas de imagen (Fondo, Estrellas y opcionalmente Suelo) en el Gestor de Capas
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.blockSignals(True)
            self.mask_layers_widget.ensure_starnet_layers(has_ground=(self.current_mask is not None))
            self.mask_layers_widget.blockSignals(False)

        self._set_starnet_tools_enabled(True)
        self._generate_preview_proxy()
        self.update_stretch_preview(immediate=True)
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
            self.mask_layers_widget.update_active_layer_histogram()

        # Sincronizar el estado consolidado para evitar pasos espurios de debounce
        if hasattr(self, '_undo_debounce_timer'):
            self._undo_debounce_timer.stop()
        self._last_committed_state = self._capture_state()

        self.log_message("[STARNET] Estrellas separadas y purgadas con máscara de cielo. Capas activadas.")

        # Desplegar automáticamente la sección de capas para acceso inmediato
        if hasattr(self, 'sec_masks'):
            self.sec_masks.expand()
            if hasattr(self, 'left_scroll_area'):
                self.left_scroll_area.ensureWidgetVisible(self.sec_masks)

        # Si había un preset pendiente de aplicar tras la separación de estrellas, aplicarlo ahora
        if getattr(self, '_preset_pending_after_starnet', None) is not None:
            pending = self._preset_pending_after_starnet
            self._preset_pending_after_starnet = None
            preset, scope, intensity_pct = pending
            self.apply_preset(preset, scope=scope, intensity_pct=intensity_pct)

    def on_starnet_error(self, err_msg: str):
        self._set_ai_processing_state(False)
        self.log_message(f"[ERROR STARNET] {err_msg}")
        import gc; gc.collect()
        err_lower = str(err_msg).lower()
        if any(k in err_lower for k in ("out of memory", "outofmemory", "memoryerror", "bad_alloc", "0x8007000e", "unable to allocate", "memoria insuficiente")):
            QMessageBox.critical(
                self,
                "Memoria Insuficiente (StarNet v2)",
                "La tarjeta gráfica (GPU) o la memoria RAM se han quedado sin espacio para procesar la imagen completa con StarNet.\n\n"
                "Sugerencias para resolverlo:\n"
                "• Aumenta el Stride (desplazamiento) en el panel StarNet (ej. 256 o 384 px) para reducir la cantidad de bloques en memoria.\n"
                "• Si la imagen es muy grande, realiza un recorte previo del área que deseas desacoplar.\n"
                "• En Ajustes > Rendimiento, selecciona almacenamiento intermedio en Disco.\n"
                "• Cierra navegadores web u otros programas que consuman memoria de vídeo (VRAM)."
            )
        else:
            QMessageBox.warning(self, "Error en StarNet++", err_msg)

    def _start_starnet_model_download(self):
        """Descarga asistida del modelo ONNX de StarNet v2 con barra modal de progreso."""
        progress = QProgressDialog("Conectando con StarNet...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Descarga de Modelo IA StarNet v2")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        from core.starnet_onnx import download_starnet_model
        from gui.worker import ModelDownloadWorker
        self.starnet_download_worker = ModelDownloadWorker(download_fn=download_starnet_model, parent=self)

        def on_prog(pct, d_mb, t_mb):
            progress.setValue(pct)
            progress.setLabelText(f"Instalando modelo StarNet v2 (starnet2.onnx)...\n{d_mb:.1f} MB de {t_mb:.1f} MB ({pct}%)")

        def on_finished(path):
            progress.close()
            self.log_message(f"✓ [STARNET AI] Modelo StarNet v2 ONNX instalado en: {os.path.basename(path)}")
            self.run_starnet()

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR DESCARGA STARNET] {err}")
            QMessageBox.warning(self, "Error en Descarga", f"No se pudo descargar el modelo StarNet v2:\n{err}")

        def on_cancel():
            self.starnet_download_worker.cancel()
            self.log_message("[STARNET AI] Descarga cancelada por el usuario.")

        self.starnet_download_worker.progress_changed.connect(on_prog)
        self.starnet_download_worker.finished_success.connect(on_finished)
        self.starnet_download_worker.error_occurred.connect(on_error)
        self.starnet_download_worker.cancelled.connect(on_cancel)
        progress.canceled.connect(on_cancel)

        self.starnet_download_worker.start()


    # =========================================================================
    # REDUCCIÓN DE RUIDO EN CAPA DE SUELO (CLÁSICO + IA NAFNET)
    # =========================================================================
    def on_gnd_denoise_changed(self, val: int):
        """Actualiza el valor de fuerza de reducción de ruido en suelo."""
        self.gnd_denoise_val = val / 100.0
        if hasattr(self, 'panel_ground_sky') and self.panel_ground_sky is not None:
            if hasattr(self.panel_ground_sky, 'slider_gnd_dn') and self.panel_ground_sky.slider_gnd_dn.value() != val:
                self.panel_ground_sky.slider_gnd_dn.setValue(val)
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_denoise_value(val)
        self.update_stretch_preview()

    def on_run_ground_ai_denoise_requested(self):
        """Inicia o conmuta la reducción de ruido por IA (NAFNet) en la capa de Suelo."""
        if self.image_32bit is None:
            self.log_message("[IA SUELO] No hay imagen cargada en el revelador.")
            return

        # Si ya existe buffer en caché, conmutar directamente A/B
        if self._cache_ground_ai is not None:
            self.gnd_ai_denoise_enabled = not self.gnd_ai_denoise_enabled
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_ground_ai_denoise_active(self.gnd_ai_denoise_enabled)
            mode_lbl = "IA (NAFNet)" if self.gnd_ai_denoise_enabled else "Clásico (Bilateral)"
            self.log_message(f"✓ [DENOISE SUELO] Conmutado a modo: {mode_lbl}.")
            self._invalidate_pipeline_caches()
            self._generate_preview_proxy()
            self.update_stretch_preview()
            return

        from core.ai_denoise import is_nafnet_available
        if not is_nafnet_available():
            msg = QMessageBox(self)
            msg.setWindowTitle("Modelo de IA NAFNet no encontrado")
            msg.setText(
                "El modelo de IA para reducción profunda de ruido (nafnet_denoise.onnx, ~117 MB) "
                "no se encuentra en la carpeta 'models/'.\n\n"
                "¿Deseas descargarlo ahora automáticamente desde Hugging Face?"
            )
            btn_download = msg.addButton("Descargar IA (~117 MB)", QMessageBox.AcceptRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_download)
            msg.exec()

            if msg.clickedButton() == btn_download:
                self._start_nafnet_model_download()
            return

        self._run_ground_nafnet_inference()

    def _get_stretched_base_for_ai(self) -> np.ndarray:
        """Retorna la imagen base desarrollada/estirada para alimentar NAFNet en rango visible sRGB."""
        if not getattr(self, 'is_zoomed_100', False):
            if hasattr(self, '_cache_ms_stretch_img') and self._cache_ms_stretch_img is not None:
                return self._cache_ms_stretch_img.copy()
            if hasattr(self, '_cached_base_stretched') and self._cached_base_stretched is not None:
                return self._cached_base_stretched.copy()
        from core.stretch import manual_stretch, apply_ghs_stretch
        st_params = self.panel_stretch.get_state() if hasattr(self, 'panel_stretch') else {}
        st_mode = st_params.get("stretch_mode", "mtf")
        src = np.clip(self.image_32bit.copy(), 0.0, 1.0).astype(np.float32)
        if st_mode == "ghs":
            return apply_ghs_stretch(
                src,
                sp=st_params.get("ghs_sp_val", 0.02),
                d=st_params.get("ghs_d_val", 5.0),
                b=st_params.get("ghs_b_val", 2.0),
                hp=st_params.get("ghs_hp_val", 0.0),
                lp=st_params.get("ghs_lp_val", 0.0),
                bp=st_params.get("ghs_bp_val", 0.0),
                color_mode="rgb"
            )
        else:
            return manual_stretch(
                src,
                black_point=st_params.get("bp_val", 0.0),
                midtone=st_params.get("mtf_val", 0.15)
            )

    def _run_ground_nafnet_inference(self):
        """Ejecuta inferencia NAFNet sobre la base terrestre."""
        if self.image_32bit is None:
            return

        progress = QProgressDialog("Inicializando NAFNet para Capa de Suelo...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Reducción de Ruido IA — Suelo")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        from gui.worker import AIDenoiseWorker
        # Procesar la base estirada en rango visible sRGB para evitar deriva matemática en NAFNet
        base_src = self._get_stretched_base_for_ai()

        self.ground_ai_denoise_worker = AIDenoiseWorker(
            base_src,
            tile_size=512,
            overlap=64,
            clean_chroma=True,
            chroma_strength=0.5,
            parent=self
        )

        def on_tile_prog(curr, total):
            pct = int((curr / total) * 100) if total > 0 else 0
            progress.setValue(pct)
            progress.setLabelText(f"Reduciendo ruido en superficie terrestre con NAFNet...\nBloque {curr} de {total} ({pct}%)")

        def on_finished(denoised_rgb):
            progress.close()
            self._cache_ground_ai = denoised_rgb
            self.gnd_ai_denoise_enabled = True
            self.log_message("✓ [IA SUELO] Reducción de ruido NAFNet en suelo calculada y almacenada en memoria caché.")

            # Activar fuerza recomendada (70%) si estaba en cero
            if getattr(self, 'gnd_denoise_val', 0.0) <= 1e-4:
                self.gnd_denoise_val = 0.70
                if hasattr(self, 'mask_layers_widget'):
                    self.mask_layers_widget.set_ground_denoise_value(70)

            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_ground_ai_denoise_active(True)

            self._invalidate_pipeline_caches()
            self._generate_preview_proxy()
            self.update_stretch_preview()

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR IA SUELO] {err}")
            import gc; gc.collect()
            err_lower = str(err).lower()
            if any(k in err_lower for k in ("out of memory", "outofmemory", "memoryerror", "bad_alloc", "0x8007000e", "unable to allocate", "memoria insuficiente")):
                QMessageBox.critical(
                    self,
                    "Memoria Insuficiente (IA Suelo)",
                    "No hay suficiente memoria RAM o VRAM disponible para procesar el suelo con NAFNet.\n\n"
                    "Sugerencias:\n"
                    "• Asegúrate de cerrar programas pesados en segundo plano.\n"
                    "• Prueba a procesar un recorte o seleccionar una zona más acotada de la imagen."
                )
            else:
                QMessageBox.warning(self, "Error en Inferencia", f"Ocurrió un error en la inferencia NAFNet de suelo:\n{err}")

        def on_cancel():
            self.ground_ai_denoise_worker.cancel()
            self.log_message("[IA SUELO] Inferencia cancelada por el usuario.")

        self.ground_ai_denoise_worker.progress_changed.connect(on_tile_prog)
        self.ground_ai_denoise_worker.finished_success.connect(on_finished)
        self.ground_ai_denoise_worker.error_occurred.connect(on_error)
        self.ground_ai_denoise_worker.cancelled.connect(on_cancel)
        progress.canceled.connect(on_cancel)

        self.ground_ai_denoise_worker.start()

    # =========================================================================
    # REDUCCIÓN DE RUIDO CON INTELIGENCIA ARTIFICIAL (NAFNET ONNX)
    # =========================================================================
    def on_run_ai_denoise_requested(self):
        """Inicia o conmuta la reducción de ruido neuronal profunda NAFNet sobre la capa Starless."""
        if self.image_starless is None:
            self.log_message("[IA DENOISE] No hay capa StarNet generada todavía. Separa las estrellas primero.")
            QMessageBox.information(
                self,
                "Reducción de Ruido con IA",
                "Primero debes desacoplar las estrellas con StarNet para poder aplicar reducción de ruido profunda al fondo."
            )
            return

        # Si ya existe un buffer de IA calculado, conmutar inmediatamente entre modo Clásico e IA
        if self._cache_starless_ai is not None:
            self.starless_ai_denoise_enabled = not self.starless_ai_denoise_enabled
            if hasattr(self, 'panel_starnet'):
                self.panel_starnet.set_ai_denoise_active(self.starless_ai_denoise_enabled)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_ai_denoise_active(self.starless_ai_denoise_enabled)
            mode_lbl = "IA (NAFNet)" if self.starless_ai_denoise_enabled else "Clásico (Bilateral)"
            self.log_message(f"✓ [DENOISE STARLESS] Conmutado a modo: {mode_lbl}.")
            self._invalidate_pipeline_caches()
            self._generate_preview_proxy()
            self.update_stretch_preview()
            return

        if not getattr(self, "starless_is_stretched", False):
            reply = QMessageBox.question(
                self,
                "Estirado Recomendado para IA",
                "La capa de Fondo (Starless) aún se encuentra en modo lineal sin estirar.\n\n"
                "La red neuronal NAFNet está calibrada para imágenes no lineales (estiradas).\n"
                "Si se aplica directamente en lineal puede recortar información tenue en sombras profundas.\n\n"
                "¿Deseas continuar con la inferencia de todos modos?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        from core.ai_denoise import is_nafnet_available
        if not is_nafnet_available():
            msg = QMessageBox(self)
            msg.setWindowTitle("Modelo de IA NAFNet no encontrado")
            msg.setText(
                "El modelo de IA para reducción profunda de ruido (nafnet_denoise.onnx, ~117 MB) "
                "no se encuentra en la carpeta 'models/'.\n\n"
                "¿Deseas descargarlo ahora automáticamente desde el repositorio seguro de Hugging Face?"
            )
            btn_download = msg.addButton("Descargar IA (~117 MB)", QMessageBox.AcceptRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_download)
            msg.exec()

            if msg.clickedButton() == btn_download:
                self._start_nafnet_model_download()
            return

        self._run_nafnet_inference()

    def _start_nafnet_model_download(self):
        """Descarga asistida del modelo ONNX de NAFNet con barra modal de progreso."""
        progress = QProgressDialog("Conectando con Hugging Face...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Descarga de Modelo IA Denoise (NAFNet)")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        from core.ai_denoise import download_nafnet_model
        from gui.worker import ModelDownloadWorker
        self.nafnet_download_worker = ModelDownloadWorker(download_fn=download_nafnet_model, parent=self)

        def on_prog(pct, d_mb, t_mb):
            progress.setValue(pct)
            progress.setLabelText(f"Descargando modelo NAFNet (nafnet_denoise.onnx)...\n{d_mb:.1f} MB de {t_mb:.1f} MB ({pct}%)")

        def on_finished(path):
            progress.close()
            self.log_message(f"✓ [IA DENOISE] Modelo NAFNet descargado e instalado en: {os.path.basename(path)}")
            self._run_nafnet_inference()

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR DESCARGA IA] {err}")
            QMessageBox.warning(self, "Error en Descarga", f"No se pudo descargar el modelo NAFNet:\n{err}")

        def on_cancel():
            self.nafnet_download_worker.cancel()
            self.log_message("[IA DENOISE] Descarga cancelada por el usuario.")

        self.nafnet_download_worker.progress_changed.connect(on_prog)
        self.nafnet_download_worker.finished_success.connect(on_finished)
        self.nafnet_download_worker.error_occurred.connect(on_error)
        self.nafnet_download_worker.cancelled.connect(on_cancel)
        progress.canceled.connect(on_cancel)

        self.nafnet_download_worker.start()

    def _run_nafnet_inference(self):
        """Ejecuta el worker de inferencia NAFNet con tiling 512x512 y ponderación Hann 2D."""
        if self.image_starless is None:
            return

        progress = QProgressDialog("Inicializando red neuronal NAFNet...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Reducción de Ruido IA (NAFNet)")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        from gui.worker import AIDenoiseWorker
        c_strength = float(getattr(self, 'denoise_color_strength', 0.5))

        self.ai_denoise_worker = AIDenoiseWorker(
            self.image_starless.copy(),
            tile_size=512,
            overlap=64,
            clean_chroma=True,
            chroma_strength=c_strength,
            parent=self
        )

        def on_tile_prog(curr, total):
            pct = int((curr / total) * 100) if total > 0 else 0
            progress.setValue(pct)
            progress.setLabelText(f"Procesando reducción de ruido con NAFNet...\nBloque {curr} de {total} ({pct}%)")

        def on_finished(denoised_rgb):
            progress.close()
            self._cache_starless_ai = denoised_rgb
            self.starless_ai_denoise_enabled = True
            self.log_message("✓ [IA DENOISE] Reducción de ruido NAFNet completada y almacenada en memoria caché.")

            # Ajustar fuerza sugerida (70%) si el control estaba en cero
            if hasattr(self, 'slider_denoise') and self.slider_denoise.value() == 0:
                self.slider_denoise.setValue(70)
            if hasattr(self, 'mask_layers_widget'):
                if getattr(self.mask_layers_widget, 'slider_sl_denoise', None) and self.mask_layers_widget.slider_sl_denoise.value() == 0:
                    self.mask_layers_widget.set_starless_denoise_value(70)

            # Notificar al botón para reflejar estado activo
            if hasattr(self, 'panel_starnet'):
                self.panel_starnet.set_ai_denoise_active(True)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_ai_denoise_active(True)

            self._invalidate_pipeline_caches()
            self._generate_preview_proxy()
            self.update_stretch_preview()

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR IA DENOISE] {err}")
            import gc; gc.collect()
            err_lower = str(err).lower()
            if any(k in err_lower for k in ("out of memory", "outofmemory", "memoryerror", "bad_alloc", "0x8007000e", "unable to allocate", "memoria insuficiente")):
                QMessageBox.critical(
                    self,
                    "Memoria Insuficiente (NAFNet Denoise)",
                    "No hay suficiente memoria RAM o VRAM disponible para completar la reducción de ruido con NAFNet.\n\n"
                    "Sugerencias:\n"
                    "• Asegúrate de cerrar programas pesados en segundo plano.\n"
                    "• Si la imagen supera los 40–50 Mpx, considera realizar un recorte del área de interés antes de aplicar IA.\n"
                    "• En Ajustes > Rendimiento, activa almacenamiento temporal en Disco."
                )
            else:
                QMessageBox.warning(self, "Error en Inferencia", f"Ocurrió un error en la inferencia NAFNet:\n{err}")

        def on_cancel():
            self.ai_denoise_worker.cancel()
            self.log_message("[IA DENOISE] Inferencia cancelada por el usuario.")

        self.ai_denoise_worker.progress_changed.connect(on_tile_prog)
        self.ai_denoise_worker.finished_success.connect(on_finished)
        self.ai_denoise_worker.error_occurred.connect(on_error)
        self.ai_denoise_worker.cancelled.connect(on_cancel)
        progress.canceled.connect(on_cancel)

        self.ai_denoise_worker.start()

    def on_auto_stretch_starless_requested(self):
        """Ejecuta el estirado no lineal adaptativo sobre la capa starless (Estirado Adaptativo RGB)."""
        if self.image_starless is None:
            self.log_message("[AUTO-STRETCH] No hay capa StarNet generada todavía. Separa las estrellas primero.")
            QMessageBox.information(self, "Estirado Adaptativo Starless", "Primero debes desacoplar las estrellas con StarNet.")
            return

        has_ai_denoise = (getattr(self, '_cache_starless_ai', None) is not None) or getattr(self, 'starless_ai_denoise_enabled', False)
        if has_ai_denoise:
            reply = QMessageBox.question(
                self,
                "Re-estirado y Reducción de Ruido IA",
                "Se ha detectado una reducción de ruido con IA (NAFNet) activa o calculada sobre el estirado actual.\n\n"
                "Al re-estirar la capa Starless, la base tonal cambia y la inferencia de IA previa "
                "debe descartarse para evitar artefactos graves de luminancia y color.\n\n"
                "• El modo IA se desactivará.\n"
                "• El control de reducción de ruido se restablecerá a 0%.\n"
                "• Podrás volver a calcular la IA ('✨ IA') cuando finalices el nuevo estirado.\n\n"
                "¿Deseas continuar y re-estirar?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Snapshot de estado previo para Deshacer
        undo_item = {
            "type": "starless",
            "image_starless": self.image_starless.copy(),
            "starless_is_stretched": getattr(self, "starless_is_stretched", False),
            "_cache_starless_ai": self._cache_starless_ai.copy() if getattr(self, "_cache_starless_ai", None) is not None else None,
            "starless_ai_denoise_enabled": getattr(self, "starless_ai_denoise_enabled", False),
            "denoise_strength": getattr(self, "denoise_strength", 0.0),
            "denoise_slider_val": self.slider_denoise.value() if hasattr(self, 'slider_denoise') else 0,
        }
        self.undo_stack.append(undo_item)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

        # Usar la base lineal virgen para garantizar idempotencia estadística
        src = getattr(self, "raw_linear_starless", None)
        if src is None:
            src = self.image_starless.copy()
            self.raw_linear_starless = src.copy()

        # Leer parámetros configurados desde el gestor de capas
        params = self.mask_layers_widget.get_autostretch_params() if hasattr(self, 'mask_layers_widget') else {}
        target_bg = params.get("target_bg", 0.15)
        shadow_clip_sigma = params.get("shadow_clip_sigma", 2.0)
        saturation_boost = params.get("saturation_boost", 2.5)
        unlinked = params.get("unlinked", True)
        feather_radius = params.get("feather_radius", 2)

        from core.stretch import auto_stretch_adaptive_starless
        mode_str = "RGB independiente" if unlinked else "vinculado"
        self.log_message(
            f"[AUTO-STRETCH] Calculando estadísticas MTF ({mode_str}, Bg={target_bg*100:.0f}%, {shadow_clip_sigma:.1f}σ) "
            f"en capa Starless..."
        )
        stretched_starless = auto_stretch_adaptive_starless(
            src,
            sky_mask=self.current_mask,
            target_bg=target_bg,
            shadow_clip_sigma=shadow_clip_sigma,
            saturation_boost=saturation_boost,
            unlinked=unlinked,
            feather_radius=feather_radius
        )
        self.image_starless = stretched_starless
        self.starless_is_stretched = True
        self._cache_starless_ai = None
        self.starless_ai_denoise_enabled = False
        if hasattr(self, 'slider_denoise'):
            self.slider_denoise.setValue(0)
        else:
            self.denoise_strength = 0.0
        if hasattr(self, 'panel_starnet'):
            self.panel_starnet.set_ai_denoise_active(False)
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ai_denoise_active(False)

        self._invalidate_pipeline_caches()
        self._generate_preview_proxy()
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
            self.mask_layers_widget.update_active_layer_histogram()
        self.update_stretch_preview(immediate=True)
        if has_ai_denoise:
            self.log_message("[AUTO-STRETCH] ℹ️ Base re-estirada: Reducción de ruido IA desactivada y control restablecido a 0%.")
        self.log_message(
            f"[AUTO-STRETCH] ✨ Estirado Adaptativo RGB aplicado sobre Starless "
            f"(Bg={target_bg*100:.0f}%, {shadow_clip_sigma:.1f}σ, sat {saturation_boost:.1f}x, {mode_str})."
        )
        
    def _get_current_deconv_psf(self) -> np.ndarray:
        """Devuelve el kernel PSF actual, utilizándolo desde la caché si los parámetros no han cambiado."""
        key = (
            self.deconv_model,
            round(self.deconv_fwhm, 2),
            round(self.deconv_beta, 2),
            round(self.deconv_aspect_ratio, 2),
            round(self.deconv_angle_deg, 1)
        )
        if self._cached_psf is not None and self._cached_psf_key == key:
            return self._cached_psf

        if self.deconv_model == "gaussian":
            psf = generate_gaussian_psf(
                fwhm=self.deconv_fwhm,
                aspect_ratio=self.deconv_aspect_ratio,
                angle_deg=self.deconv_angle_deg
            )
        else:
            psf = generate_moffat_psf(
                fwhm=self.deconv_fwhm,
                beta=self.deconv_beta,
                aspect_ratio=self.deconv_aspect_ratio,
                angle_deg=self.deconv_angle_deg
            )

        self._cached_psf = psf
        self._cached_psf_key = key
        return psf

    def on_auto_detect_psf_clicked(self):
        """Analiza estrellas en la imagen para estimar automáticamente FWHM y relación de aspecto."""
        img_target = None
        if self.image_32bit is not None:
            img_target = self.image_32bit
        elif self.preview_proxy is not None:
            img_target = self.preview_proxy

        if img_target is None:
            QMessageBox.information(
                self,
                "Deconvolución",
                "Carga primero una imagen para poder estimar la PSF óptica de las estrellas."
            )
            return

        sky_mask_2d = None
        if self.current_mask is not None:
            sky_mask_2d = self.current_mask

        self.lbl_psf_status.setText("Analizando estrellas aisladas...")
        QApplication.processEvents()

        try:
            psf_kernel, stats = estimate_psf_from_stars(
                img_target,
                sky_mask=sky_mask_2d,
                max_stars=60
            )

            est_fwhm = stats.get("fwhm", 3.0)
            est_aspect = stats.get("aspect_ratio", 1.0)
            stars_sampled = stats.get("stars_sampled", 0)
            is_fallback = stats.get("fallback", False)

            # Actualizar controles
            self.deconv_fwhm = float(est_fwhm)
            self.deconv_aspect_ratio = float(est_aspect)
            self._cached_psf = None
            self._cached_psf_key = ()

            if hasattr(self, 'slider_deconv_fwhm'):
                self.slider_deconv_fwhm.blockSignals(True)
                self.slider_deconv_fwhm.setValue(int(np.clip(est_fwhm * 10.0, 10, 80)))
                self.slider_deconv_fwhm.blockSignals(False)
                self.lbl_deconv_fwhm.setText(f"{est_fwhm:.1f} px")

            if hasattr(self, 'slider_deconv_aspect'):
                self.slider_deconv_aspect.blockSignals(True)
                self.slider_deconv_aspect.setValue(int(np.clip(est_aspect * 100.0, 100, 250)))
                self.slider_deconv_aspect.blockSignals(False)
                self.lbl_deconv_aspect.setText(f"{est_aspect:.2f}x")

            status_txt = f"PSF: Moffat 2D ({stars_sampled} estrellas | FWHM: {est_fwhm:.2f} px | Aspecto: {est_aspect:.2f}x)"
            if is_fallback:
                status_txt += " [Ajuste estándar]"
            self.lbl_psf_status.setText(status_txt)

            self.log_message(
                f"[DECONV] PSF estimada automáticamente: {stars_sampled} estrellas analizadas -> "
                f"FWHM = {est_fwhm:.2f} px, Aspecto = {est_aspect:.2f}x"
            )

            if self.deconv_iterations > 0:
                self.update_stretch_preview()

            self._commit_action()

        except Exception as e:
            self.log_message(f"[ERROR] Error al estimar PSF: {str(e)}")
            self.lbl_psf_status.setText("Error al estimar PSF")


    def _on_deconv_parameter_changed(self):
        """Sincroniza el estado del panel modular de Deconvolución con el revelador."""
        if hasattr(self, 'panel_deconv'):
            self.deconv_iterations = self.panel_deconv.deconv_iterations
            self.deconv_fwhm = self.panel_deconv.deconv_fwhm
            self.deconv_deringing = self.panel_deconv.deconv_deringing
            self.deconv_damping = self.panel_deconv.deconv_damping
            self.deconv_aspect_ratio = self.panel_deconv.deconv_aspect_ratio
            self.deconv_angle_deg = self.panel_deconv.deconv_angle_deg
            self.deconv_model = self.panel_deconv.deconv_model
            self.deconv_scope = self.panel_deconv.deconv_scope
            self.deconv_luma_only = self.panel_deconv.deconv_luma_only
            self._cached_psf = None
            self._cached_psf_key = ()
        if self.deconv_iterations > 0:
            self.update_stretch_preview()

    def _on_color_wb_parameter_changed(self):
        """Sincroniza el estado del panel modular de Color/WB con el revelador y actualiza la vista."""
        if hasattr(self, 'panel_color_wb'):
            self.temp_val = self.panel_color_wb.temp_val
            self.tint_val = self.panel_color_wb.tint_val
            self.sat_sky_val = self.panel_color_wb.sat_sky_val
            self.sat_gnd_val = self.panel_color_wb.sat_gnd_val
            self.vibrance_val = self.panel_color_wb.vibrance_val
            self.pcc_kr = getattr(self.panel_color_wb, 'pcc_kr', 1.0)
            self.pcc_kb = getattr(self.panel_color_wb, 'pcc_kb', 1.0)
            self.dualband_mode = getattr(self.panel_color_wb, 'dualband_mode', 'none')
            self.dualband_oiii_boost = getattr(self.panel_color_wb, 'dualband_oiii_boost', 100)
            self.unclip_stars_val = getattr(self.panel_color_wb, 'unclip_stars_val', 0)
            self.unpurple_val = getattr(self.panel_color_wb, 'unpurple_val', 0)
            self.scnr_green_val = getattr(self.panel_color_wb, 'scnr_green_val', 0)
        self.update_stretch_preview()

    def apply_pcc_calibration(self):
        """Calcula y aplica la Calibración Fotométrica del Color (PCC) contra Gaia/Locus estelar."""
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return

        from core.pcc import detect_stars_photometry, fit_theil_sen_pcc
        stars = detect_stars_photometry(target)
        if len(stars) < 10:
            self.log_message("PCC: Estrellas insuficientes no saturadas para calibración fotométrica (< 10).")
            return

        kr, kg, kb = fit_theil_sen_pcc(stars)
        if hasattr(self, 'panel_color_wb'):
            self.panel_color_wb.set_pcc_gains(kr, kb)
            self.panel_color_wb.action_committed.emit()
        self._on_color_wb_parameter_changed()
        self.update_stretch_preview(immediate=True)
        self.log_message(f"PCC completado con {len(stars)} estrellas. Ganancias: R={kr:.3f}, G=1.000, B={kb:.3f}")

    def start_astro_reveal(self) -> None:
        """
        Ejecuta el arranque inteligente '1-Click Start Astro-Reveal':
        1. Detección automática de máscara de horizonte por IA (U²-Net + Guided Filter).
        2. Neutralización del fondo celestial (Balance de blancos sobre cielo libre de nebulosas).
        3. Auto-estirado asimétrico del cielo (MTF con Target=0.12).
        4. Evaluación fotométrica del suelo y compensación controlada (Smart Ground Lift).
        """
        if self.image_32bit is None:
            self.log_message("[ASTRO-REVEAL] No hay imagen cargada para realizar el arranque.")
            return

        self.setCursor(Qt.WaitCursor)
        self.log_message("[ASTRO-REVEAL] ✨ Iniciando '1-Click Start Astro-Reveal'...")
        QCoreApplication.processEvents()

        try:
            from core.ai_horizon import auto_detect_sky_mask
            from core.ghs import estimate_ghs_params

            # 1. Máscara de Horizonte (Cielo / Suelo)
            if self.current_mask is None:
                self.log_message("[ASTRO-REVEAL] Paso 1/4: Segmentando horizonte con IA...")
                QCoreApplication.processEvents()
                try:
                    mask, method = auto_detect_sky_mask(self.image_32bit, force_algorithmic=False)
                    self._apply_loaded_mask(mask, source_desc=f"1-Click Reveal ({method})")
                except Exception as e_mask:
                    self.log_message(f"[ASTRO-REVEAL] Aviso al detectar horizonte: {e_mask}")
            else:
                self.log_message("[ASTRO-REVEAL] Paso 1/4: Utilizando máscara de horizonte existente.")

            sky_mask = self.current_mask
            target_img = self.preview_proxy if self.preview_proxy is not None else self.image_32bit

            # 2. Neutralización del Fondo Celestial
            self.log_message("[ASTRO-REVEAL] Paso 2/4: Neutralizando fondo celestial...")
            temp_calc, tint_calc = calculate_sky_neutral_wb(target_img, sky_mask)
            temp_int = int(round(temp_calc * 1000.0))
            tint_int = int(round(tint_calc * 1000.0))
            if hasattr(self, 'slider_temp'):
                self.slider_temp.setValue(temp_int)
            if hasattr(self, 'slider_tint'):
                self.slider_tint.setValue(tint_int)
            self.temp_val = temp_calc
            self.tint_val = tint_calc
            self.log_message(f"[ASTRO-REVEAL] ✓ Fondo neutralizado: Temp={temp_calc:+.3f}, Tinte={tint_calc:+.3f}")

            # Muestra con balance aplicado para cálculo exacto de estirado y suelo
            balanced_sample = apply_white_balance(target_img, temp_calc, tint_calc)

            # 3. Auto-Estirado Óptimo del Cielo (Calibración Dual MTF + GHS)
            self.log_message("[ASTRO-REVEAL] Paso 3/4: Calculando auto-estirado óptimo de cielo (MTF Target=0.20 / GHS Target=0.18)...")
            bp_mtf, m_mtf = calculate_sky_mtf_params(balanced_sample, sky_mask, target_background=0.20)
            ghs_params = estimate_ghs_params(balanced_sample, target_background=0.18, sky_mask=sky_mask)

            # Sincronizar ambos modos en el panel de estirado
            self.sync_controls(bp_mtf, m_mtf)
            if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
                self.panel_stretch.set_state({
                    "ghs_sp_val": ghs_params["sp"],
                    "ghs_d_val": ghs_params["d"],
                    "ghs_b_val": ghs_params["b"],
                    "ghs_hp_val": ghs_params["hp"],
                    "ghs_lp_val": ghs_params["lp"],
                    "ghs_bp_val": ghs_params["bp"],
                })

            # Respetar el modo activo elegido por el usuario (o mantener coherencia si ya usaba GHS)
            current_mode = getattr(self, 'stretch_mode', 'mtf')
            if current_mode == 'ghs':
                self.log_message(f"[ASTRO-REVEAL] ✓ Estirado GHS aplicado: SP={ghs_params['sp']:.5f}, D={ghs_params['d']:.2f}, BP={ghs_params['bp']:.5f} (HP={ghs_params['hp']:.2f})")
            else:
                self.log_message(f"[ASTRO-REVEAL] ✓ Estirado MTF aplicado: BP={bp_mtf:.5f}, MTF m={m_mtf:.5f} (Target=0.20)")

            # 4. Evaluación y Smart Lift del Suelo
            self.log_message("[ASTRO-REVEAL] Paso 4/4: Evaluando condiciones fotométricas del suelo...")
            gnd_ev, gnd_sh, gnd_dn, diag_str = calculate_smart_ground_lift(balanced_sample, sky_mask)
            if hasattr(self, 'slider_gnd_ev'):
                self.slider_gnd_ev.setValue(int(round(gnd_ev * 100.0)))
            if hasattr(self, 'slider_gnd_sh'):
                self.slider_gnd_sh.setValue(int(round(gnd_sh)))
            if hasattr(self, 'panel_ground_sky') and hasattr(self.panel_ground_sky, 'slider_gnd_dn') and self.panel_ground_sky.slider_gnd_dn is not None:
                self.panel_ground_sky.slider_gnd_dn.setValue(int(round(gnd_dn)))
            self.log_message(f"[ASTRO-REVEAL] ✓ {diag_str}")

            # 5. Actualización final
            self.update_stretch_preview(immediate=True)
            self._commit_action()
            self.log_message("✓ [ASTRO-REVEAL] 1-Click Start Astro-Reveal completado con éxito.")
        except Exception as e:
            self.log_message(f"[ERROR ASTRO-REVEAL] {e}")
            QMessageBox.warning(self, "Astro-Reveal", f"No se pudo completar el auto-revelado:\n{e}")
        finally:
            self.unsetCursor()

    def apply_auto_mtf(self):
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return

        sky_m = self.current_mask if self.current_mask is not None else getattr(self.canvas, 'last_mask_array', None)
        hp = getattr(self.panel_stretch, 'mtf_hp_val', 0.50)
        from core.stretch import auto_statistical_stretch
        _, bp_suggested, m_suggested = auto_statistical_stretch(
            target,
            sky_mask=sky_m,
            target_background=0.20,
            highlight_protection=hp
        )
        self.sync_controls(bp_suggested, m_suggested)
        self.update_stretch_preview(immediate=True)
        self.log_message(f"Auto-MTF Estadístico aplicado (Punto Negro: {bp_suggested:.5f}, MTF m: {m_suggested:.5f}, Prot. Estrellas: {int(hp*100)}%).")

    def sync_controls(self, bp: float, m: float):
        if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
            self.panel_stretch.sync_controls(bp, m)

    def apply_auto_ghs(self):
        """Calcula y aplica parámetros óptimos de GHS delegando en el panel de estirado."""
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return

        params = estimate_ghs_params(target, target_background=0.18)

        if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
            self.panel_stretch.set_state({
                "ghs_sp_val": params["sp"],
                "ghs_d_val": params["d"],
                "ghs_b_val": params["b"],
                "ghs_hp_val": params["hp"],
                "ghs_lp_val": params["lp"],
                "ghs_bp_val": params["bp"],
            })

        self.log_message(
            f"[GHS] Auto-GHS aplicado: SP={params['sp']:.5f}, D={params['d']:.2f}, b={params['b']:.2f}, "
            f"BP={params['bp']:.5f}, HP={params['hp']:.2f}"
        )
        self.update_stretch_preview()
        self._commit_action()

    def _on_toggle_bg_sample_tool(self, active: bool):
        if active and self.image_32bit is None:
            self.panel_lp.set_sample_tool_active(False)
            self.log_message("[AVISO] Carga una toma antes de activar la herramienta de muestreo de fondo.")
            return
        self.canvas.set_bg_sample_mode(active)
        if active:
            self.log_message("[FONDO] Herramienta de Muestreo (DBE) activa: Clic izquierdo = Añadir/Arrastrar muestra. Clic derecho = Eliminar muestra.")
            if not self.canvas.get_bg_samples():
                self.auto_generate_bg_samples()

    def auto_generate_bg_samples(self):
        if self.image_32bit is None:
            self.log_message("[AVISO] Carga una toma antes de generar puntos de muestreo.")
            return
        h, w = self.image_32bit.shape[:2]
        box_size = getattr(self.panel_lp, 'rbf_box_size', 25)
        sample_img = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        pts = generate_sky_sample_points(h, w, sky_mask=self.current_mask, img_rgb=sample_img, box_size=box_size)
        self.canvas.set_bg_samples(pts)
        self.panel_lp.set_points_count(len(pts))
        self.log_message(f"[FONDO] Generados automáticamente {len(pts)} puntos de muestreo en el cielo.")

    def clear_bg_samples(self):
        self.canvas.clear_bg_samples()
        self.panel_lp.set_points_count(0)
        self.log_message("[FONDO] Puntos de muestreo eliminados.")

    def _on_canvas_bg_samples_changed(self):
        pts = self.canvas.get_bg_samples()
        self.panel_lp.set_points_count(len(pts))

    def on_extract_background_clicked(self):
        if self.image_32bit is None:
            return

        model_text = self.panel_lp.combo_poly_degree.currentText()
        if "Muestreo" in model_text or "RBF" in model_text:
            pts = self.canvas.get_bg_samples()
            if len(pts) < 4:
                self.auto_generate_bg_samples()
                pts = self.canvas.get_bg_samples()
            if len(pts) < 4:
                QMessageBox.warning(self, "Aviso", "No se detectó suficiente cielo para muestrear fondo.")
                return

            mode = getattr(self.panel_lp, 'rbf_mode', 'subtraction')
            box_size = getattr(self.panel_lp, 'rbf_box_size', 25)
            self.log_message(f"[FONDO] Extrayendo gradiente con modelo RBF ({len(pts)} puntos, modo {mode})...")

            # Guardar instantánea para Deshacer
            undo_item = {
                "type": "bg_extract",
                "image_32bit": self.image_32bit.copy(),
                "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
                "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
            }
            self.undo_stack.append(undo_item)
            if len(self.undo_stack) > self.MAX_UNDO_STEPS:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self._update_undo_redo_buttons()

            try:
                sky_m = self.current_mask if self.current_mask is not None else getattr(self.canvas, 'last_mask_array', None)
                corrected, _ = extract_background_rbf(
                    self.image_32bit,
                    sample_points=pts,
                    sky_mask=sky_m,
                    box_size=box_size,
                    mode=mode,
                    kernel="multiquadric",
                    smoothing=0.15,
                    epsilon=1.0
                )
                self.on_bg_extraction_success(corrected)
                self.canvas.set_bg_sample_mode(False)
                self.log_message(f"[FONDO] Gradiente RBF ({len(pts)} puntos) neutralizado con éxito. Usa Ctrl+Z para revertir.")
            except Exception as e:
                self.log_message(f"[ERROR FONDO] Fallo en interpolación RBF: {e}")
                self.on_bg_extraction_error(str(e))
        else:
            degree = 1 if "Lineal" in model_text else 2
            deg_label = "Lineal (Grado 1)" if degree == 1 else "Cuadrático (Grado 2)"
            self.log_message(f"[FONDO] Extrayendo gradiente con modelo Polinómico {deg_label}...")

            undo_item = {
                "type": "bg_extract",
                "image_32bit": self.image_32bit.copy(),
                "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
                "starless_is_stretched": getattr(self, "starless_is_stretched", False),
                "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
            }
            self.undo_stack.append(undo_item)
            if len(self.undo_stack) > self.MAX_UNDO_STEPS:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self._update_undo_redo_buttons()

            try:
                corrected = extract_background_polynomial(
                    self.image_32bit, 
                    sky_mask=self.current_mask, 
                    degree=degree
                )
                self.on_bg_extraction_success(corrected)
                self.log_message(f"[FONDO] Gradiente polinómico ({deg_label}) neutralizado con éxito. Usa Ctrl+Z para revertir.")
            except Exception as e:
                self.log_message(f"[ERROR FONDO] Fallo en cálculo polinómico: {e}")
                self.on_bg_extraction_error(str(e))

    def on_bg_extraction_success(self, corrected_img: np.ndarray):
        self._set_ai_processing_state(False)
        self.image_32bit = np.ascontiguousarray(corrected_img, dtype=np.float32)
        self.canvas.set_native_image_size((self.image_32bit.shape[1], self.image_32bit.shape[0]))
        
        self.image_starless = None
        self.image_stars = None
        self.combo_layer.setEnabled(False)
        self.slider_stars.setEnabled(False)
        
        self._set_starnet_tools_enabled(False)
        self._generate_preview_proxy()
        self.update_stretch_preview()
        
    def on_auto_abe_clicked(self):
        """Ejecuta la extracción automática de gradientes de fondo en 1-clic (ABE / Seti Astro)."""
        if self.image_32bit is None:
            return

        mode = getattr(self.panel_lp, 'rbf_mode', 'subtraction')
        box_size = getattr(self.panel_lp, 'rbf_box_size', 25)
        self.log_message(f"[ABE 1-CLIC] Extrayendo gradiente de fondo de forma automática ({mode})...")

        # Guardar instantánea para Deshacer
        undo_item = {
            "type": "bg_extract",
            "image_32bit": self.image_32bit.copy(),
            "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
            "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
            "starless_is_stretched": getattr(self, "starless_is_stretched", False),
            "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
            "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
        }
        self.undo_stack.append(undo_item)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

        try:
            sky_m = self.current_mask if self.current_mask is not None else getattr(self.canvas, 'last_mask_array', None)
            corrected, _ = auto_extract_background_sky(
                self.image_32bit,
                sky_mask=sky_m,
                box_size=box_size,
                mode=mode,
                smoothing=0.15,
                kernel="multiquadric"
            )
            self.on_bg_extraction_success(corrected)
            self.canvas.set_bg_sample_mode(False)
            self.log_message("[ABE 1-CLIC] ✓ Gradiente neutralizado con éxito. Usa Ctrl+Z para revertir.")
        except Exception as e:
            self.log_message(f"[ERROR ABE] {e}")
            self.on_bg_extraction_error(str(e))

    def on_bg_extraction_error(self, err_msg: str):
        self._set_ai_processing_state(False)
        self.log_message(f"[ERROR FONDO] {err_msg}")
        QMessageBox.warning(self, "Error en Extracción de Gradiente", err_msg)


    def _on_toggle_lp_expand(self, checked: bool):
        if hasattr(self, 'w_lp_params'):
            self.w_lp_params.setVisible(checked)

    def auto_detect_lp_center(self):
        """Detecta automáticamente la ubicación más probable de la cúpula de polución urbana en el horizonte."""
        img = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if img is None:
            return

        mask_2d = None
        if self.current_mask is not None:
            m = np.squeeze(self.current_mask)
            if m.ndim == 2:
                mask_2d = m
            elif m.ndim == 3:
                mask_2d = m[..., 0]

        cx, cy = estimate_light_pollution_center(img, sky_mask=mask_2d)
        self.lp_center_x = cx
        self.lp_center_y = cy

        if hasattr(self, 'spin_lp_cx'):
            self.spin_lp_cx.blockSignals(True)
            self.spin_lp_cx.setValue(cx)
            self.spin_lp_cx.blockSignals(False)
        if hasattr(self, 'slider_lp_cx'):
            self.slider_lp_cx.blockSignals(True)
            self.slider_lp_cx.setValue(int(round(cx * 100.0)))
            self.slider_lp_cx.blockSignals(False)

        if hasattr(self, 'spin_lp_cy'):
            self.spin_lp_cy.blockSignals(True)
            self.spin_lp_cy.setValue(cy)
            self.spin_lp_cy.blockSignals(False)
        if hasattr(self, 'slider_lp_cy'):
            self.slider_lp_cy.blockSignals(True)
            self.slider_lp_cy.setValue(int(round(cy * 100.0)))
            self.slider_lp_cy.blockSignals(False)

        if hasattr(self, 'btn_lp_expand') and not self.btn_lp_expand.isChecked():
            self.btn_lp_expand.setChecked(True)

        self.log_message(f"[Cúpula 2D] Centro detectado automáticamente: X={cx:.2f}, Y={cy:.2f}")
        self.update_stretch_preview()
        self._commit_action()

    def _on_lp_parameter_changed(self):
        """Sincroniza el estado del panel modular de Contaminación Lumínica con el revelador."""
        if hasattr(self, 'panel_lp'):
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
        self.update_stretch_preview()


    def _set_ai_processing_state(self, busy: bool):
        """Bloquea o desbloquea controles mientras se ejecutan tareas pesadas de AI."""
        self.btn_starnet.setEnabled(not busy)
        self.btn_extract_bg.setEnabled(not busy)
        self.btn_open_img.setEnabled(not busy)
        self.btn_export.setEnabled(not busy)
        if busy:
            self.setCursor(Qt.WaitCursor)
        else:
            self.unsetCursor()

