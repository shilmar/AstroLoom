# gui/dialog_web_export.py
"""
gui/dialog_web_export.py - Diálogo modal de exportación optimizada para Web y Redes Sociales.

Diseñado para astrofotografía y paisaje nocturno:
- Soporte síncrono o asíncrono en segundo plano (renderizado de capas multirresolución sin congelar la UI).
- Conversión a espacio sRGB 8-bit con dithering triangular para evitar banding en cielos oscuros.
- Submuestreo cromático 4:4:4 en JPEG (conserva el color puntual en estrellas rojas/azules).
- Soporte para WebP de alta compresión.
- Micro-enfoque de salida opcional (Unsharp Mask) calibrado para estrellas y detalles terrestres.
- Reescalado Lanczos de alta fidelidad.
- Estimación de peso en disco en tiempo real con debounce.
- Incrustación de perfil ICC sRGB y conservación de metadatos EXIF.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFile, ImageFilter
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import get_config_val
from core.icc_manager import convert_colorspace, get_icc_profile_bytes
from gui.icons import get_icon
from gui.theme import (
    BG_CARD,
    BG_PANEL,
    BORDER_BASE,
    STYLE_RUN_IDLE,
    STYLE_SECONDARY_BTN,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# Evitar 'broken data stream / Suspension not allowed here' en Pillow al codificar JPEG progresivo de alto detalle
ImageFile.MAXBLOCK = max(getattr(ImageFile, "MAXBLOCK", 65536), 16 * 1024 * 1024)


def extract_exif_bytes(filepath: str) -> Optional[bytes]:
    """Extrae los bytes binarios de metadatos EXIF originales desde un archivo de imagen."""
    if not filepath or not os.path.isfile(filepath):
        return None
    try:
        with Image.open(filepath) as img:
            return img.info.get("exif")
    except Exception:
        return None


def normalize_to_uint8_rgb(raw_buf: np.ndarray, source_colorspace: str = "srgb") -> np.ndarray:
    """Normaliza cualquier buffer numérico a un array uint8 RGB de 3 canales con dithering."""
    # 1. Asegurar formato 3 canales
    if raw_buf.ndim == 2:
        raw_buf = np.repeat(raw_buf[..., np.newaxis], 3, axis=2)
    elif raw_buf.ndim == 3 and raw_buf.shape[2] == 4:
        raw_buf = raw_buf[..., :3]

    # 2. Conversión cromática a sRGB si procede
    if source_colorspace and source_colorspace.lower() != "srgb":
        try:
            raw_buf = convert_colorspace(raw_buf, source_colorspace, "srgb")
        except Exception:
            pass

    # 3. Cuantización a uint8 con dithering si viene en float
    if np.issubdtype(raw_buf.dtype, np.floating):
        buf_clipped = np.clip(raw_buf, 0.0, 1.0).astype(np.float32)
        noise = (np.random.random(buf_clipped.shape).astype(np.float32) - 0.5) / 255.0
        return np.clip((buf_clipped + noise) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif raw_buf.dtype == np.uint16:
        return (np.clip(raw_buf, 0, 65535) >> 8).astype(np.uint8)
    else:
        return np.ascontiguousarray(raw_buf, dtype=np.uint8)


class WebExportWorker(QThread):
    """Hilo en segundo plano para renderizar la composición a máxima resolución sin bloquear la UI."""
    finished_render = Signal(object)
    failed_render = Signal(str)

    def __init__(self, render_fn: Callable[[], Optional[np.ndarray]], parent: Optional[QObject] = None):
        super().__init__(parent)
        self._render_fn = render_fn
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            res = self._render_fn()
            if not self._is_cancelled:
                self.finished_render.emit(res)
        except Exception as e:
            if not self._is_cancelled:
                self.failed_render.emit(str(e))


class WebExportDialog(QDialog):
    """
    Diálogo modal reutilizable para exportación optimizada a Web y Redes Sociales.
    Soporta generación asíncrona de la imagen en segundo plano con apertura instantánea de la UI.
    """

    def __init__(
        self,
        image_buffer: Optional[Union[np.ndarray, Callable[[], Optional[np.ndarray]]]] = None,
        render_fn: Optional[Callable[[], Optional[np.ndarray]]] = None,
        initial_dimensions: Optional[Tuple[int, int]] = None,
        default_filename: str = "astroloom_web",
        exif_bytes: Optional[bytes] = None,
        source_colorspace: str = "srgb",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Exportar para Web / Redes Sociales")
        self.setMinimumWidth(440)
        self.setModal(True)
        self.setWindowIcon(get_icon("globe", size=16))

        self._source_colorspace = source_colorspace
        self._default_filename = default_filename or "astroloom_web"
        self._exif_bytes = exif_bytes
        self._updating_dims = False
        self._worker: Optional[WebExportWorker] = None
        self._base_8bit: Optional[np.ndarray] = None

        # Si image_buffer es una función invocable, tratarlo como render_fn
        if callable(image_buffer):
            render_fn = image_buffer
            image_buffer = None

        # Determinar si la imagen ya está disponible o requiere renderizado asíncrono
        if image_buffer is not None:
            self._base_8bit = normalize_to_uint8_rgb(image_buffer, self._source_colorspace)
            self._orig_h, self._orig_w = self._base_8bit.shape[:2]
            self._is_rendering = False
        else:
            self._is_rendering = bool(render_fn is not None)
            if initial_dimensions and len(initial_dimensions) == 2:
                self._orig_w, self._orig_h = int(initial_dimensions[0]), int(initial_dimensions[1])
            else:
                self._orig_w, self._orig_h = (2048, 1365)

        self._aspect_ratio = self._orig_w / max(1, self._orig_h)

        # Timer debounce para estimación de peso en tiempo real (180 ms)
        self._estimate_timer = QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.setInterval(180)
        self._estimate_timer.timeout.connect(self._update_size_estimate)

        self._init_ui()
        self._connect_signals()
        self._on_preset_changed(self.combo_preset.currentIndex())

        # Si se especificó función de renderizado asíncrono, lanzarla en segundo plano
        if self._is_rendering and render_fn is not None:
            self.btn_export.setEnabled(False)
            self.lbl_estimate.setText("Componiendo imagen a máxima resolución...")
            self.lbl_estimate.setStyleSheet("font-weight: bold; color: #38BDF8; font-size: 11px;")
            self.progress_bar.setVisible(True)

            self._worker = WebExportWorker(render_fn, self)
            self._worker.finished_render.connect(self._on_render_finished)
            self._worker.failed_render.connect(self._on_render_failed)
            self._worker.start()
        elif self._base_8bit is not None:
            self._schedule_estimate()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        # --- Bloque 1: Dimensiones y Escalado ---
        grp_size = QGroupBox("Dimensiones y Reescalado")
        form_size = QFormLayout(grp_size)
        form_size.setSpacing(8)

        self.combo_preset = QComboBox()
        self.combo_preset.addItem(f"Original (100% — {self._orig_w} × {self._orig_h} px)", "orig")
        self.combo_preset.addItem("4K UHD (3840 px lado largo)", 3840)
        self.combo_preset.addItem("Web / Portfolio (2048 px lado largo)", 2048)
        self.combo_preset.addItem("Full HD (1920 px lado largo)", 1920)
        self.combo_preset.addItem("Instagram / Vertical (1350 px lado largo)", 1350)
        self.combo_preset.addItem("Personalizado...", "custom")
        self.combo_preset.setCurrentIndex(2)  # 2048px por defecto

        dim_layout = QHBoxLayout()
        self.spin_width = QSpinBox()
        self.spin_width.setRange(64, max(self._orig_w, 16384))
        self.spin_width.setSuffix(" px")

        self.spin_height = QSpinBox()
        self.spin_height.setRange(64, max(self._orig_h, 16384))
        self.spin_height.setSuffix(" px")

        dim_layout.addWidget(QLabel("Ancho:"))
        dim_layout.addWidget(self.spin_width)
        dim_layout.addSpacing(10)
        dim_layout.addWidget(QLabel("Alto:"))
        dim_layout.addWidget(self.spin_height)

        self.combo_sharpen = QComboBox()
        self.combo_sharpen.addItem("Ninguno", 0)
        self.combo_sharpen.addItem("Suave (Estrellas puntuales y cielo)", 1)
        self.combo_sharpen.addItem("Medio (Detalles de paisaje y textura)", 2)
        self.combo_sharpen.setCurrentIndex(1)

        form_size.addRow("Preset:", self.combo_preset)
        form_size.addRow("Resolución:", dim_layout)
        form_size.addRow("Enfoque salida:", self.combo_sharpen)
        main_layout.addWidget(grp_size)

        # --- Bloque 2: Formato y Compresión ---
        grp_comp = QGroupBox("Formato y Calidad")
        form_comp = QFormLayout(grp_comp)
        form_comp.setSpacing(8)

        self.combo_format = QComboBox()
        self.combo_format.addItem("JPEG (.jpg) — Compatibilidad universal", "JPEG")
        self.combo_format.addItem("WebP (.webp) — Menor peso / Alta fidelidad", "WEBP")
        self.combo_format.addItem("PNG (.png) — Sin pérdidas / Web", "PNG")

        qual_layout = QHBoxLayout()
        self.slider_quality = QSlider(Qt.Orientation.Horizontal)
        self.slider_quality.setRange(50, 100)
        self.slider_quality.setValue(90)

        self.spin_quality = QSpinBox()
        self.spin_quality.setRange(50, 100)
        self.spin_quality.setValue(90)
        self.spin_quality.setSuffix(" %")

        qual_layout.addWidget(self.slider_quality)
        qual_layout.addWidget(self.spin_quality)

        self.chk_progressive = QCheckBox("JPEG Progresivo + Submuestreo 4:4:4 (mejor color en estrellas)")
        self.chk_progressive.setChecked(True)
        self.chk_progressive.setToolTip(
            "Desactiva la compresión de crominancia estándar 4:2:0 para preservar el color intenso\n"
            "en estrellas individuales pequeñas y nebulosas."
        )

        form_comp.addRow("Formato:", self.combo_format)
        form_comp.addRow("Calidad:", qual_layout)
        form_comp.addRow("", self.chk_progressive)
        main_layout.addWidget(grp_comp)

        # --- Bloque 3: Color y Metadatos ---
        grp_meta = QGroupBox("Color y Metadatos")
        vbox_meta = QVBoxLayout(grp_meta)
        vbox_meta.setSpacing(6)

        self.chk_srgb = QCheckBox("Incrustar perfil ICC sRGB IEC61966-2.1")
        self.chk_srgb.setChecked(True)
        self.chk_srgb.setToolTip("Garantiza que navegadores y móviles interpreten los tonos oscuros sin levantar el punto negro.")

        self.chk_keep_exif = QCheckBox("Conservar metadatos EXIF de cámara y exposición")
        has_exif = bool(self._exif_bytes)
        self.chk_keep_exif.setChecked(has_exif)
        self.chk_keep_exif.setEnabled(has_exif)
        if not has_exif:
            self.chk_keep_exif.setToolTip("No se detectaron metadatos EXIF binarios en la imagen cargada.")

        vbox_meta.addWidget(self.chk_srgb)
        vbox_meta.addWidget(self.chk_keep_exif)
        main_layout.addWidget(grp_meta)

        # Barra de progreso delgada de actividad para renderizado asíncrono
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # --- Pie de Diálogo: Estimación de peso y Botones de acción ---
        footer_layout = QHBoxLayout()
        self.lbl_estimate = QLabel("Calculando peso...")
        self.lbl_estimate.setStyleSheet(f"font-weight: bold; color: #60A5FA; font-size: 12px;")

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setStyleSheet(STYLE_SECONDARY_BTN)

        self.btn_export = QPushButton("Exportar...")
        self.btn_export.setIcon(get_icon("globe", color="#ffffff", size=14))
        self.btn_export.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_export.setDefault(True)

        footer_layout.addWidget(self.lbl_estimate)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_cancel)
        footer_layout.addWidget(self.btn_export)
        main_layout.addLayout(footer_layout)

    def _connect_signals(self) -> None:
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.spin_width.valueChanged.connect(self._on_width_changed)
        self.spin_height.valueChanged.connect(self._on_height_changed)
        self.combo_sharpen.currentIndexChanged.connect(self._schedule_estimate)

        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        self.slider_quality.valueChanged.connect(self.spin_quality.setValue)
        self.spin_quality.valueChanged.connect(self.slider_quality.setValue)
        self.slider_quality.valueChanged.connect(self._schedule_estimate)
        self.chk_progressive.toggled.connect(self._schedule_estimate)
        self.chk_srgb.toggled.connect(self._schedule_estimate)
        self.chk_keep_exif.toggled.connect(self._schedule_estimate)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_export.clicked.connect(self._on_export_clicked)

    def _on_render_finished(self, result_image: Optional[np.ndarray]) -> None:
        """Slot llamado cuando el worker asíncrono completa la composición de la imagen."""
        self.progress_bar.setVisible(False)
        if result_image is None:
            self._on_render_failed("El pipeline no devolvió ninguna imagen.")
            return

        self._base_8bit = normalize_to_uint8_rgb(result_image, self._source_colorspace)
        new_h, new_w = self._base_8bit.shape[:2]

        if (new_w, new_h) != (self._orig_w, self._orig_h):
            self._orig_w, self._orig_h = new_w, new_h
            self._aspect_ratio = self._orig_w / max(1, self._orig_h)
            self.combo_preset.setItemText(0, f"Original (100% — {self._orig_w} × {self._orig_h} px)")
            self.spin_width.setRange(64, max(self._orig_w, 16384))
            self.spin_height.setRange(64, max(self._orig_h, 16384))
            self._on_preset_changed(self.combo_preset.currentIndex())

        self.lbl_estimate.setStyleSheet("font-weight: bold; color: #60A5FA; font-size: 12px;")
        self.btn_export.setEnabled(True)
        self._schedule_estimate()

    def _on_render_failed(self, error_msg: str) -> None:
        """Slot llamado en caso de error durante la composición asíncrona."""
        self.progress_bar.setVisible(False)
        self.lbl_estimate.setText(f"Error al generar imagen: {error_msg}")
        self.lbl_estimate.setStyleSheet("font-weight: bold; color: #EF4444; font-size: 11px;")
        self.btn_export.setEnabled(False)

    def _on_preset_changed(self, index: int) -> None:
        data = self.combo_preset.itemData(index)
        is_custom = (data == "custom")
        self.spin_width.setEnabled(is_custom)
        self.spin_height.setEnabled(is_custom)

        self._updating_dims = True
        if data == "orig":
            self.spin_width.setValue(self._orig_w)
            self.spin_height.setValue(self._orig_h)
        elif isinstance(data, int):
            long_edge = min(data, max(self._orig_w, self._orig_h))
            if self._orig_w >= self._orig_h:
                new_w = long_edge
                new_h = max(1, round(long_edge / self._aspect_ratio))
            else:
                new_h = long_edge
                new_w = max(1, round(long_edge * self._aspect_ratio))
            self.spin_width.setValue(new_w)
            self.spin_height.setValue(new_h)
        self._updating_dims = False
        self._schedule_estimate()

    def _on_width_changed(self, new_w: int) -> None:
        if self._updating_dims:
            return
        self._updating_dims = True
        self.spin_height.setValue(max(1, round(new_w / self._aspect_ratio)))
        self._updating_dims = False
        self._schedule_estimate()

    def _on_height_changed(self, new_h: int) -> None:
        if self._updating_dims:
            return
        self._updating_dims = True
        self.spin_width.setValue(max(1, round(new_h * self._aspect_ratio)))
        self._updating_dims = False
        self._schedule_estimate()

    def _on_format_changed(self) -> None:
        fmt = self.combo_format.currentData()
        is_jpeg = (fmt == "JPEG")
        is_lossy = (fmt in ("JPEG", "WEBP"))
        self.chk_progressive.setEnabled(is_jpeg)
        self.slider_quality.setEnabled(is_lossy)
        self.spin_quality.setEnabled(is_lossy)
        self._schedule_estimate()

    def _schedule_estimate(self) -> None:
        if self._base_8bit is None:
            self.lbl_estimate.setText("Componiendo imagen a máxima resolución...")
            return
        self.lbl_estimate.setText("Calculando peso...")
        self._estimate_timer.start()

    def _prepare_pil_image(self) -> Image.Image:
        if self._base_8bit is None:
            raise RuntimeError("La imagen aún no ha terminado de generarse.")

        target_w = self.spin_width.value()
        target_h = self.spin_height.value()

        pil_img = Image.fromarray(self._base_8bit, mode="RGB")

        if (target_w, target_h) != (self._orig_w, self._orig_h):
            pil_img = pil_img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

        sharpen_level = self.combo_sharpen.currentData()
        if sharpen_level == 1:
            pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=45, threshold=3))
        elif sharpen_level == 2:
            pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1.1, percent=75, threshold=3))

        return pil_img

    def _encode_to_bytes(self) -> bytes:
        pil_img = self._prepare_pil_image()
        fmt = self.combo_format.currentData()
        quality = self.spin_quality.value()

        save_kwargs = {}

        if fmt == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
            save_kwargs["progressive"] = self.chk_progressive.isChecked()
            save_kwargs["subsampling"] = 0 if self.chk_progressive.isChecked() else 2
        elif fmt == "WEBP":
            save_kwargs["quality"] = quality
            save_kwargs["method"] = 4
        elif fmt == "PNG":
            save_kwargs["optimize"] = True

        if self.chk_srgb.isChecked():
            try:
                icc_bytes = get_icc_profile_bytes("srgb")
                if icc_bytes:
                    save_kwargs["icc_profile"] = icc_bytes
            except Exception:
                pass

        if self.chk_keep_exif.isChecked() and self._exif_bytes:
            save_kwargs["exif"] = self._exif_bytes

        buffer = io.BytesIO()
        pil_img.save(buffer, format=fmt, **save_kwargs)
        return buffer.getvalue()

    def _update_size_estimate(self) -> None:
        if self._base_8bit is None:
            return
        try:
            raw_bytes = self._encode_to_bytes()
            size_kb = len(raw_bytes) / 1024.0
            w, h = self.spin_width.value(), self.spin_height.value()
            if size_kb >= 1024:
                self.lbl_estimate.setText(f"~{size_kb / 1024.0:.2f} MB ({w} × {h} px)")
            else:
                self.lbl_estimate.setText(f"~{size_kb:.0f} KB ({w} × {h} px)")
        except Exception:
            self.lbl_estimate.setText("Estimación no disponible")

    def _on_export_clicked(self) -> None:
        if self._base_8bit is None:
            QMessageBox.warning(self, "Espera", "La imagen aún se está componiendo en segundo plano. Por favor, espera.")
            return

        fmt = self.combo_format.currentData()
        if fmt == "JPEG":
            ext = ".jpg"
            file_filter = "JPEG (*.jpg *.jpeg)"
        elif fmt == "WEBP":
            ext = ".webp"
            file_filter = "WebP (*.webp)"
        else:
            ext = ".png"
            file_filter = "PNG (*.png)"

        default_dir = get_config_val("export_dir", os.getcwd())
        os.makedirs(default_dir, exist_ok=True)
        suggested = os.path.join(default_dir, f"{self._default_filename}_web{ext}")

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar imagen optimizada para Web / Redes",
            suggested,
            file_filter,
        )
        if not filepath:
            return

        filepath = os.path.abspath(os.path.normpath(str(filepath).strip()))
        if not filepath.lower().endswith(ext):
            filepath += ext

        try:
            data = self._encode_to_bytes()
            Path(filepath).write_bytes(data)
            self.accept()
            QMessageBox.information(
                self,
                "Exportación Completada",
                f"Imagen web guardada con éxito en:\n{filepath}\n\n"
                f"Resolución: {self.spin_width.value()} × {self.spin_height.value()} px\n"
                f"Peso: {len(data) / 1024.0:.1f} KB"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al Exportar", f"No se pudo guardar la imagen:\n{e}")

    def reject(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            try:
                self._worker.finished_render.disconnect()
                self._worker.failed_render.disconnect()
            except Exception:
                pass
        super().reject()

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            try:
                self._worker.finished_render.disconnect()
                self._worker.failed_render.disconnect()
            except Exception:
                pass
        super().closeEvent(event)
