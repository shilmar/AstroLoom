# gui/tab_panorama.py
"""
gui/tab_panorama.py - Pestaña dedicada a la Composición Panorámica y Mosaicos Nocturnos (Nightscape Stitcher).

Integra:
1. Ingesta de paneles con tabla interactiva Drag & Drop, extracción EXIF de focal y marcas temporales.
2. Soporte nativo multifila (multi-row 2D) con auto-detección por grafo o cuadrícula asistida M x N.
3. Proyecciones cartográficas: Cilíndrica, Esférica y Plana.
4. Ajustes astro (pre-realce CLAHE para cielo/tierra, compensación de ganancia, corrección de horizonte).
5. Fusión multibanda laplaciana y auto-recorte por rectángulo máximo inscrito.
6. Lienzo interactivo acelerado con zoom por rueda centrado en puntero, paneo por arrastre y conmutador de recorte.
7. Exportación en TIFF 16-bit / 32-bit float y enlace directo con el Revelador / Editor.
"""

import os
import time
import tempfile
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
import tifffile
from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QCursor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QMessageBox, QSlider, QComboBox,
    QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QScrollArea, QSplitter, QFrame, QSizePolicy
)

from core.panorama import (
    PanoramaSettings,
    PanoramaResult,
    PanoramaPanelMetadata,
    extract_panel_metadata,
    stitch_nightscape_panorama,
    save_panorama_session,
    load_panorama_session
)
from core.icc_manager import save_tiff_with_icc
from gui.widgets.drop_widgets import DropTableWidget, collect_image_files_from_mime


STYLE_RUN_IDLE = """
    QPushButton {
        font-weight: bold;
        font-size: 13px;
        background-color: #2e7d32;
        color: white;
        border-radius: 4px;
        padding: 8px 16px;
    }
    QPushButton:hover {
        background-color: #388e3c;
    }
    QPushButton:pressed {
        background-color: #1b5e20;
    }
    QPushButton:disabled {
        background-color: #2b332d;
        color: #718774;
    }
"""

STYLE_CANCEL_ACTIVE = """
    QPushButton {
        font-weight: bold;
        font-size: 12px;
        background-color: #c62828;
        color: white;
        border-radius: 4px;
        padding: 6px 12px;
    }
    QPushButton:hover {
        background-color: #d32f2f;
    }
    QPushButton:pressed {
        background-color: #8e0000;
    }
"""

STYLE_TRANSFER_BTN = """
    QPushButton {
        font-weight: bold;
        font-size: 12px;
        background-color: #00838f;
        color: white;
        border-radius: 4px;
        padding: 7px 14px;
    }
    QPushButton:hover {
        background-color: #0097a7;
    }
    QPushButton:pressed {
        background-color: #006064;
    }
    QPushButton:disabled {
        background-color: #1c2e30;
        color: #527578;
    }
"""


class MetadataExtractWorker(QThread):
    """Hilo de trabajo en segundo plano para leer metadatos EXIF sin congelar la UI."""
    panel_ready = Signal(int, object)   # (index, PanoramaPanelMetadata)
    all_finished = Signal()

    def __init__(self, filepaths: List[str]):
        super().__init__()
        self.filepaths = filepaths

    def run(self):
        for idx, p in enumerate(self.filepaths):
            try:
                meta = extract_panel_metadata(p, idx)
                self.panel_ready.emit(idx, meta)
            except Exception:
                pass
        self.all_finished.emit()


class PanoramaStitchWorker(QThread):
    """Hilo de procesamiento en segundo plano para el cosido de panorámica."""
    progress = Signal(int, str)
    finished = Signal(object)
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, panel_paths: List[str], settings: PanoramaSettings):
        super().__init__()
        self.panel_paths = panel_paths
        self.settings = settings
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            result = stitch_nightscape_panorama(
                self.panel_paths,
                self.settings,
                progress_callback=self.progress.emit,
                abort_flag=lambda: self._is_cancelled
            )
            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.finished.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(str(e))


class PanoramaInteractiveCanvas(QWidget):
    """
    Lienzo interactivo acelerado para previsualizar panorámicas con zoom 1:1,
    paneo dinámico por arrastre y alternancia entre vista completa y vista auto-recortada.
    """
    zoom_changed = Signal(float)
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #0f1015; border: 1px solid #262833; border-radius: 4px;")

        self._pixmap_full: Optional[QPixmap] = None
        self._pixmap_cropped: Optional[QPixmap] = None
        self._show_cropped: bool = True
        self._crop_rect: Optional[Tuple[int, int, int, int]] = None

        self._zoom: float = 1.0
        self._pan_offset: QPointF = QPointF(0, 0)
        self._is_panning: bool = False
        self._drag_start: QPointF = QPointF(0, 0)
        self._fit_to_view: bool = True
        self.placeholder_text: str = (
            "Arrastra las tomas de tu serie panorámica aquí\n"
            "o pulsa 'Añadir Archivos' en el panel izquierdo."
        )

    def set_image(
        self,
        full_rgb: Optional[np.ndarray],
        crop_rect: Optional[Tuple[int, int, int, int]] = None
    ):
        if full_rgb is None:
            self._pixmap_full = None
            self._pixmap_cropped = None
            self._crop_rect = None
            self.update()
            return

        # Generar QPixmap para imagen completa
        u8_full = np.ascontiguousarray((np.clip(full_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
        hf, wf = u8_full.shape[:2]
        qimg_full = QImage(u8_full.data, wf, hf, 3 * wf, QImage.Format_RGB888)
        self._pixmap_full = QPixmap.fromImage(qimg_full)
        self._crop_rect = crop_rect

        # Generar QPixmap para imagen recortada si existe
        if crop_rect is not None:
            rx, ry, rw, rh = crop_rect
            rx = max(0, min(wf - 1, int(rx)))
            ry = max(0, min(hf - 1, int(ry)))
            rw = max(1, min(wf - rx, int(rw)))
            rh = max(1, min(hf - ry, int(rh)))
            u8_crop = np.ascontiguousarray(u8_full[ry:ry+rh, rx:rx+rw])
            qimg_crop = QImage(u8_crop.data, rw, rh, 3 * rw, QImage.Format_RGB888)
            self._pixmap_cropped = QPixmap.fromImage(qimg_crop)
        else:
            self._pixmap_cropped = None

        if self._fit_to_view:
            self.fit_view()
        else:
            self.update()

    def set_placeholder(self, text: str):
        self.placeholder_text = text
        self._pixmap_full = None
        self._pixmap_cropped = None
        self.update()

    def set_show_cropped(self, show_cropped: bool):
        self._show_cropped = show_cropped
        if self._fit_to_view:
            self.fit_view()
        else:
            self.update()

    def _active_pixmap(self) -> Optional[QPixmap]:
        if self._show_cropped and self._pixmap_cropped is not None:
            return self._pixmap_cropped
        return self._pixmap_full

    def fit_view(self):
        self._fit_to_view = True
        pm = self._active_pixmap()
        if pm is None or pm.isNull():
            self._zoom = 1.0
            self._pan_offset = QPointF(0, 0)
            self.update()
            self.zoom_changed.emit(self._zoom)
            return

        vw = max(50, self.width() - 30)
        vh = max(50, self.height() - 30)
        pw = pm.width()
        ph = pm.height()

        scale_w = vw / float(pw)
        scale_h = vh / float(ph)
        self._zoom = min(scale_w, scale_h, 1.5)
        rendered_w = pw * self._zoom
        rendered_h = ph * self._zoom
        self._pan_offset = QPointF((self.width() - rendered_w) / 2.0, (self.height() - rendered_h) / 2.0)
        self.update()
        self.zoom_changed.emit(self._zoom)

    def set_zoom_100(self):
        self._fit_to_view = False
        self._zoom = 1.0
        pm = self._active_pixmap()
        if pm:
            cx = (self.width() - pm.width()) / 2.0
            cy = (self.height() - pm.height()) / 2.0
            self._pan_offset = QPointF(cx, cy)
        self.update()
        self.zoom_changed.emit(self._zoom)

    def zoom_by(self, factor: float):
        self._fit_to_view = False
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom_at(factor, center)

    def _zoom_at(self, factor: float, center_pos: QPointF):
        old_zoom = self._zoom
        new_zoom = np.clip(old_zoom * factor, 0.02, 16.0)
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        # Ajuste de desplazamiento para mantener el centro del puntero
        self._pan_offset = center_pos - (center_pos - self._pan_offset) * (new_zoom / old_zoom)
        self._zoom = new_zoom
        self.update()
        self.zoom_changed.emit(self._zoom)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else (1.0 / 1.25)
        self._fit_to_view = False
        self._zoom_at(factor, event.position())

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = True
            self._drag_start = event.position() - self._pan_offset
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event):
        if self._is_panning:
            self._fit_to_view = False
            self._pan_offset = event.position() - self._drag_start
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_to_view:
            self.fit_view()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = collect_image_files_from_mime(event.mimeData())
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        pm = self._active_pixmap()
        if pm is None or pm.isNull():
            painter.setPen(QColor("#78909c"))
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)
            return

        rw = pm.width() * self._zoom
        rh = pm.height() * self._zoom
        target_rect = QRectF(self._pan_offset.x(), self._pan_offset.y(), rw, rh)

        painter.drawPixmap(target_rect.toRect(), pm)

        # Si mostramos la imagen completa y hay un rectángulo de recorte, dibujar la guía
        if not self._show_cropped and self._crop_rect is not None and self._pixmap_full is not None:
            rx, ry, cw, ch = self._crop_rect
            crop_screen_x = self._pan_offset.x() + rx * self._zoom
            crop_screen_y = self._pan_offset.y() + ry * self._zoom
            crop_screen_w = cw * self._zoom
            crop_screen_h = ch * self._zoom

            pen = QPen(QColor("#00e676"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(crop_screen_x, crop_screen_y, crop_screen_w, crop_screen_h))


class PanoramaTab(QWidget):
    """
    Pestaña Principal del Módulo de Panorámicas y Mosaicos Nocturnos.
    """
    export_to_developer = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.panels_metadata: List[PanoramaPanelMetadata] = []
        self.current_result: Optional[PanoramaResult] = None
        self._stitch_worker: Optional[PanoramaStitchWorker] = None
        self._meta_worker: Optional[MetadataExtractWorker] = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        # =========================================================
        # PANEL IZQUIERDO: CONTROLES EN ÁREA DE DESPLAZAMIENTO
        # =========================================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(430)
        scroll_area.setMaximumWidth(520)
        scroll_area.setFrameShape(QFrame.NoFrame)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 8, 4)
        left_layout.setSpacing(10)

        # --- SECCIÓN 1: INGESTA DE PANELES ---
        grp_ingest = QGroupBox("1. Ingesta de Paneles / Teselas")
        ingest_layout = QVBoxLayout(grp_ingest)
        ingest_layout.setSpacing(6)

        btn_bar = QHBoxLayout()
        self.btn_add_files = QPushButton("➕ Añadir Tomas")
        self.btn_add_dir = QPushButton("📁 Carpeta")
        self.btn_remove = QPushButton("❌ Quitar")
        self.btn_clear = QPushButton("🧹 Limpiar")
        btn_bar.addWidget(self.btn_add_files)
        btn_bar.addWidget(self.btn_add_dir)
        btn_bar.addWidget(self.btn_remove)
        btn_bar.addWidget(self.btn_clear)
        ingest_layout.addLayout(btn_bar)

        # Tabla de paneles con Drag & Drop
        self.table_panels = DropTableWidget()
        self.table_panels.setColumnCount(6)
        self.table_panels.setHorizontalHeaderLabels(["Usar", "#", "Archivo", "Hora", "Focal", "Resolución"])
        self.table_panels.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_panels.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_panels.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_panels.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_panels.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_panels.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_panels.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_panels.setAlternatingRowColors(True)
        self.table_panels.setMinimumHeight(180)
        ingest_layout.addWidget(self.table_panels)

        # Botones de ordenación
        order_bar = QHBoxLayout()
        self.btn_up = QPushButton("⬆️ Subir")
        self.btn_down = QPushButton("⬇️ Bajar")
        self.btn_sort_time = QPushButton("🕒 Ordenar por Hora")
        order_bar.addWidget(self.btn_up)
        order_bar.addWidget(self.btn_down)
        order_bar.addWidget(self.btn_sort_time)
        ingest_layout.addLayout(order_bar)

        self.lbl_ingest_summary = QLabel("0 paneles cargados.")
        self.lbl_ingest_summary.setStyleSheet("color: #90a4ae; font-size: 11px;")
        ingest_layout.addWidget(self.lbl_ingest_summary)

        left_layout.addWidget(grp_ingest)

        # --- SECCIÓN 2: DISPOSICIÓN Y GEOMETRÍA ---
        grp_geo = QGroupBox("2. Disposición y Geometría")
        geo_layout = QVBoxLayout(grp_geo)
        geo_layout.setSpacing(6)

        # Modo de solapamiento
        geo_layout.addWidget(QLabel("Modo de Disposición / Cuadrícula:"))
        self.combo_layout_mode = QComboBox()
        self.combo_layout_mode.addItem("Auto 2D (Grafo de Conectividad Multifila - Recomendado)", "auto")
        self.combo_layout_mode.addItem("Matriz Regular M x N (Asistente de Cuadrícula)", "grid")
        geo_layout.addWidget(self.combo_layout_mode)

        # Configuración de matriz M x N (oculta/visible según modo)
        self.box_matrix = QWidget()
        matrix_layout = QHBoxLayout(self.box_matrix)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.addWidget(QLabel("Filas (M):"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 20)
        self.spin_rows.setValue(1)
        matrix_layout.addWidget(self.spin_rows)

        matrix_layout.addWidget(QLabel("Columnas (N):"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 50)
        self.spin_cols.setValue(3)
        matrix_layout.addWidget(self.spin_cols)
        self.box_matrix.setVisible(False)
        geo_layout.addWidget(self.box_matrix)

        # Proyección cartográfica
        geo_layout.addWidget(QLabel("Proyección Cartográfica:"))
        self.combo_projection = QComboBox()
        self.combo_projection.addItem("Cilíndrica (Panoramas Horizontales Amplios - Recomendado)", "cylindrical")
        self.combo_projection.addItem("Esférica (Gran angular y 360°)", "spherical")
        self.combo_projection.addItem("Plana / Rectilínea (Mosaicos pequeños y teleobjetivo)", "plane")
        geo_layout.addWidget(self.combo_projection)

        # Focal
        focal_bar = QHBoxLayout()
        self.chk_focal_auto = QCheckBox("Focal Auto (EXIF):")
        self.chk_focal_auto.setChecked(True)
        self.spin_focal = QSpinBox()
        self.spin_focal.setRange(8, 1000)
        self.spin_focal.setValue(24)
        self.spin_focal.setSuffix(" mm")
        self.spin_focal.setEnabled(False)
        focal_bar.addWidget(self.chk_focal_auto)
        focal_bar.addWidget(self.spin_focal)
        geo_layout.addLayout(focal_bar)

        self.chk_wave_correction = QCheckBox("Corrección de Onda / Nivelación de Horizonte")
        self.chk_wave_correction.setChecked(True)
        geo_layout.addWidget(self.chk_wave_correction)

        left_layout.addWidget(grp_geo)

        # --- SECCIÓN 3: ALGORITMOS ASTRO Y FUSIÓN ---
        grp_fusion = QGroupBox("3. Algoritmos Astro y Fusión")
        fusion_layout = QVBoxLayout(grp_fusion)
        fusion_layout.setSpacing(6)

        self.chk_astro_mode = QCheckBox("Pre-realce Astro (CLAHE + sombras profundas)")
        self.chk_astro_mode.setChecked(True)
        self.chk_astro_mode.setToolTip(
            "Mejora enormemente la detección de estrellas débiles y relieve terrestre en penumbra."
        )
        fusion_layout.addWidget(self.chk_astro_mode)

        # Inliers mínimos
        inliers_bar = QHBoxLayout()
        inliers_bar.addWidget(QLabel("Mínimo de Coincidencias:"))
        self.spin_min_inliers = QSpinBox()
        self.spin_min_inliers.setRange(6, 100)
        self.spin_min_inliers.setValue(15)
        inliers_bar.addWidget(self.spin_min_inliers)
        fusion_layout.addLayout(inliers_bar)

        self.chk_exposure_comp = QCheckBox("Compensación de Viñeteo y Ganancia entre Tomas")
        self.chk_exposure_comp.setChecked(True)
        fusion_layout.addWidget(self.chk_exposure_comp)

        # Fusión
        fusion_layout.addWidget(QLabel("Tipo de Fusión:"))
        self.combo_blend = QComboBox()
        self.combo_blend.addItem("Multibanda Laplaciana (Preserva estrellas nítidas)", "multiband")
        self.combo_blend.addItem("Degradado Suave por Distancia (Feathering)", "feather")
        fusion_layout.addWidget(self.combo_blend)

        bands_bar = QHBoxLayout()
        bands_bar.addWidget(QLabel("Niveles de Pirámide:"))
        self.spin_bands = QSpinBox()
        self.spin_bands.setRange(2, 7)
        self.spin_bands.setValue(5)
        bands_bar.addWidget(self.spin_bands)
        fusion_layout.addLayout(bands_bar)

        self.chk_auto_crop = QCheckBox("✂️ Auto-Recorte de Bordes (Rectángulo Máximo)")
        self.chk_auto_crop.setChecked(True)
        self.chk_auto_crop.setToolTip(
            "Calcula el rectángulo de máxima superficie completamente relleno dentro del cosido."
        )
        fusion_layout.addWidget(self.chk_auto_crop)

        left_layout.addWidget(grp_fusion)

        # --- SECCIÓN 4: PROCESAMIENTO Y ENLACE ---
        grp_proc = QGroupBox("4. Procesamiento y Exportación")
        proc_layout = QVBoxLayout(grp_proc)
        proc_layout.setSpacing(8)

        self.btn_stitch = QPushButton("🧩 Coser Panorámica")
        self.btn_stitch.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_cancel = QPushButton("🛑 Cancelar")
        self.btn_cancel.setStyleSheet(STYLE_CANCEL_ACTIVE)
        self.btn_cancel.setVisible(False)

        proc_layout.addWidget(self.btn_stitch)
        proc_layout.addWidget(self.btn_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        proc_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Listo para componer.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #b0bec5; font-size: 11px;")
        proc_layout.addWidget(self.lbl_status)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #37474f;")
        proc_layout.addWidget(line)

        self.btn_save_tiff = QPushButton("💾 Guardar Panorámica (TIFF 16/32-bit)")
        self.btn_save_tiff.setEnabled(False)
        proc_layout.addWidget(self.btn_save_tiff)

        self.btn_to_developer = QPushButton("➡️ Enviar Panorámica al Revelador / Editor")
        self.btn_to_developer.setStyleSheet(STYLE_TRANSFER_BTN)
        self.btn_to_developer.setEnabled(False)
        proc_layout.addWidget(self.btn_to_developer)

        sess_bar = QHBoxLayout()
        self.btn_save_sess = QPushButton("💾 Guardar Sesión")
        self.btn_load_sess = QPushButton("📂 Cargar Sesión")
        sess_bar.addWidget(self.btn_save_sess)
        sess_bar.addWidget(self.btn_load_sess)
        proc_layout.addLayout(sess_bar)

        left_layout.addWidget(grp_proc)
        left_layout.addStretch()

        scroll_area.setWidget(left_container)
        splitter.addWidget(scroll_area)

        # =========================================================
        # PANEL DERECHO: BARRA SUPERIOR + LIENZO INTERACTIVO
        # =========================================================
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        # Barra de herramientas superior
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_zoom_out = QPushButton("🔍 -")
        self.btn_zoom_out.setFixedWidth(50)
        self.btn_zoom_100 = QPushButton("100%")
        self.btn_zoom_100.setFixedWidth(55)
        self.btn_zoom_in = QPushButton("🔍 +")
        self.btn_zoom_in.setFixedWidth(50)
        self.btn_zoom_fit = QPushButton("⛶ Ajustar")
        self.btn_zoom_fit.setFixedWidth(75)

        self.btn_toggle_crop = QPushButton("✂️ Ver Recorte")
        self.btn_toggle_crop.setCheckable(True)
        self.btn_toggle_crop.setChecked(True)
        self.btn_toggle_crop.setFixedWidth(110)

        self.lbl_stats = QLabel("Sin panorámica generada")
        self.lbl_stats.setStyleSheet("color: #78909c; font-size: 11px; margin-left: 10px;")

        toolbar.addWidget(self.btn_zoom_out)
        toolbar.addWidget(self.btn_zoom_100)
        toolbar.addWidget(self.btn_zoom_in)
        toolbar.addWidget(self.btn_zoom_fit)
        toolbar.addWidget(self.btn_toggle_crop)
        toolbar.addWidget(self.lbl_stats)
        toolbar.addStretch()

        right_layout.addLayout(toolbar)

        # Lienzo interactivo
        self.canvas = PanoramaInteractiveCanvas()
        right_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(right_container)
        splitter.setSizes([450, 910])

        main_layout.addWidget(splitter)

        # Conexiones
        self._connect_signals()

    def _connect_signals(self):
        # Ingesta
        self.btn_add_files.clicked.connect(self._on_add_files)
        self.btn_add_dir.clicked.connect(self._on_add_dir)
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_clear.clicked.connect(self._on_clear_panels)
        self.btn_up.clicked.connect(lambda: self._move_row(-1))
        self.btn_down.clicked.connect(lambda: self._move_row(1))
        self.btn_sort_time.clicked.connect(self._sort_panels_by_time)

        self.table_panels.files_dropped.connect(self.load_filepaths)
        self.canvas.files_dropped.connect(self.load_filepaths)

        # Configuración UI
        self.combo_layout_mode.currentIndexChanged.connect(self._on_layout_mode_changed)
        self.chk_focal_auto.toggled.connect(lambda c: self.spin_focal.setEnabled(not c))

        # Procesamiento
        self.btn_stitch.clicked.connect(self._start_stitching)
        self.btn_cancel.clicked.connect(self._cancel_stitching)

        # Exportación
        self.btn_save_tiff.clicked.connect(self._on_save_tiff)
        self.btn_to_developer.clicked.connect(self._on_export_to_developer)
        self.btn_save_sess.clicked.connect(self._on_save_session)
        self.btn_load_sess.clicked.connect(self._on_load_session)

        # Toolbar lienzo
        self.btn_zoom_out.clicked.connect(lambda: self.canvas.zoom_by(1.0 / 1.25))
        self.btn_zoom_in.clicked.connect(lambda: self.canvas.zoom_by(1.25))
        self.btn_zoom_100.clicked.connect(self.canvas.set_zoom_100)
        self.btn_zoom_fit.clicked.connect(self.canvas.fit_view)
        self.btn_toggle_crop.toggled.connect(self.canvas.set_show_cropped)

    def _on_layout_mode_changed(self):
        is_grid = self.combo_layout_mode.currentData() == "grid"
        self.box_matrix.setVisible(is_grid)

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar Paneles de la Panorámica",
            "",
            "Archivos de Imagen (*.nef *.cr2 *.cr3 *.arw *.dng *.raw *.tif *.tiff *.jpg *.jpeg *.png);;Todos (*.*)"
        )
        if paths:
            self.load_filepaths(paths)

    def _on_add_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta con Tomas Panorámicas")
        if directory:
            files = []
            for root, _, filenames in os.walk(directory):
                for f in sorted(filenames):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ('.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw', '.tif', '.tiff', '.jpg', '.jpeg', '.png'):
                        files.append(os.path.join(root, f))
            if files:
                self.load_filepaths(files)

    def load_filepaths(self, paths: List[str]):
        """Carga y agrega una lista de rutas de paneles a la tabla."""
        existing_paths = {m.filepath for m in self.panels_metadata}
        new_paths = [p for p in paths if p not in existing_paths]
        if not new_paths:
            return

        start_idx = len(self.panels_metadata)
        for i, p in enumerate(new_paths):
            meta = PanoramaPanelMetadata(
                filepath=p,
                filename=os.path.basename(p),
                index=start_idx + i
            )
            self.panels_metadata.append(meta)

        self._refresh_table()

        # Iniciar extracción de metadatos en segundo plano
        self._meta_worker = MetadataExtractWorker(new_paths)
        self._meta_worker.panel_ready.connect(self._on_panel_metadata_ready)
        self._meta_worker.all_finished.connect(self._on_all_metadata_ready)
        self._meta_worker.start()

    def _on_panel_metadata_ready(self, local_idx: int, meta: PanoramaPanelMetadata):
        # Encontrar y actualizar metadatos en la lista
        for idx, m in enumerate(self.panels_metadata):
            if m.filepath == meta.filepath:
                self.panels_metadata[idx] = meta
                self._update_table_row(idx, meta)
                break

    def _on_all_metadata_ready(self):
        # Actualizar focal estimada si está en modo Auto
        focals = [m.focal_length_mm for m in self.panels_metadata if m.focal_length_mm is not None and m.focal_length_mm > 0]
        if focals:
            med_focal = float(np.median(focals))
            self.spin_focal.setValue(int(round(med_focal)))
            self.chk_focal_auto.setText(f"Focal Auto ({med_focal:.0f} mm):")
        self._update_ingest_summary()

    def _update_table_row(self, row: int, meta: PanoramaPanelMetadata):
        if row < 0 or row >= self.table_panels.rowCount():
            return
        self.table_panels.item(row, 3).setText(meta.time_str)
        focal_str = f"{meta.focal_length_mm:.0f} mm" if meta.focal_length_mm else "--"
        self.table_panels.item(row, 4).setText(focal_str)
        res_str = f"{meta.width}x{meta.height}" if meta.width > 0 else "--"
        self.table_panels.item(row, 5).setText(res_str)

    def _refresh_table(self):
        self.table_panels.blockSignals(True)
        self.table_panels.setRowCount(len(self.panels_metadata))

        for row, meta in enumerate(self.panels_metadata):
            # 0. Usar (Checkbox)
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if meta.is_included else Qt.Unchecked)
            self.table_panels.setItem(row, 0, chk_item)

            # 1. Índice
            idx_item = QTableWidgetItem(str(row + 1))
            idx_item.setTextAlignment(Qt.AlignCenter)
            self.table_panels.setItem(row, 1, idx_item)

            # 2. Archivo
            file_item = QTableWidgetItem(meta.filename)
            file_item.setToolTip(meta.filepath)
            self.table_panels.setItem(row, 2, file_item)

            # 3. Hora
            time_item = QTableWidgetItem(meta.time_str)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.table_panels.setItem(row, 3, time_item)

            # 4. Focal
            focal_str = f"{meta.focal_length_mm:.0f} mm" if meta.focal_length_mm else "--"
            focal_item = QTableWidgetItem(focal_str)
            focal_item.setTextAlignment(Qt.AlignCenter)
            self.table_panels.setItem(row, 4, focal_item)

            # 5. Resolución
            res_str = f"{meta.width}x{meta.height}" if meta.width > 0 else "--"
            res_item = QTableWidgetItem(res_str)
            res_item.setTextAlignment(Qt.AlignCenter)
            self.table_panels.setItem(row, 5, res_item)

        self.table_panels.blockSignals(False)
        self._update_ingest_summary()

    def _update_ingest_summary(self):
        total = len(self.panels_metadata)
        included = sum(1 for row in range(self.table_panels.rowCount())
                       if self.table_panels.item(row, 0) and self.table_panels.item(row, 0).checkState() == Qt.Checked)
        self.lbl_ingest_summary.setText(f"{total} paneles ({included} seleccionados).")

    def _on_remove_selected(self):
        selected_rows = sorted(set(idx.row() for idx in self.table_panels.selectedIndexes()), reverse=True)
        if not selected_rows:
            return
        for r in selected_rows:
            if 0 <= r < len(self.panels_metadata):
                del self.panels_metadata[r]
        self._refresh_table()

    def _on_clear_panels(self):
        self.panels_metadata.clear()
        self._refresh_table()
        self.canvas.set_image(None)
        self.lbl_stats.setText("Sin panorámica generada")
        self.btn_save_tiff.setEnabled(False)
        self.btn_to_developer.setEnabled(False)

    def _move_row(self, delta: int):
        selected_rows = sorted(set(idx.row() for idx in self.table_panels.selectedIndexes()))
        if not selected_rows:
            return
        r = selected_rows[0]
        new_r = r + delta
        if 0 <= new_r < len(self.panels_metadata):
            self.panels_metadata[r], self.panels_metadata[new_r] = self.panels_metadata[new_r], self.panels_metadata[r]
            self._refresh_table()
            self.table_panels.selectRow(new_r)

    def _sort_panels_by_time(self):
        self.panels_metadata.sort(key=lambda m: (m.timestamp, m.filename))
        self._refresh_table()

    def _get_selected_panel_paths(self) -> List[str]:
        paths = []
        for r in range(self.table_panels.rowCount()):
            chk = self.table_panels.item(r, 0)
            if chk and chk.checkState() == Qt.Checked:
                paths.append(self.panels_metadata[r].filepath)
        return paths

    def _start_stitching(self):
        paths = self._get_selected_panel_paths()
        if len(paths) < 2:
            QMessageBox.warning(
                self,
                "Tomas Insuficientes",
                "Por favor selecciona al menos 2 paneles con la casilla activada para componer la panorámica."
            )
            return

        settings = PanoramaSettings(
            projection=self.combo_projection.currentData(),
            astro_mode=self.chk_astro_mode.isChecked(),
            min_inliers=self.spin_min_inliers.value(),
            exposure_comp=self.chk_exposure_comp.isChecked(),
            blend_mode=self.combo_blend.currentData(),
            blend_bands=self.spin_bands.value(),
            auto_crop=self.chk_auto_crop.isChecked(),
            wave_correction=self.chk_wave_correction.isChecked(),
            layout_mode=self.combo_layout_mode.currentData(),
            grid_rows=self.spin_rows.value(),
            grid_cols=self.spin_cols.value(),
            focal_length_override=None if self.chk_focal_auto.isChecked() else float(self.spin_focal.value())
        )

        self._set_processing_ui(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Iniciando cosido panorámico nocturno...")

        self._stitch_worker = PanoramaStitchWorker(paths, settings)
        self._stitch_worker.progress.connect(self._on_stitch_progress)
        self._stitch_worker.finished.connect(self._on_stitch_finished)
        self._stitch_worker.cancelled.connect(self._on_stitch_cancelled)
        self._stitch_worker.error.connect(self._on_stitch_error)
        self._stitch_worker.start()

    def _cancel_stitching(self):
        if self._stitch_worker:
            self.lbl_status.setText("Cancelando procesamiento...")
            self._stitch_worker.cancel()

    def _on_stitch_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_stitch_finished(self, result: PanoramaResult):
        self._set_processing_ui(False)
        self.current_result = result

        w, h = result.resolution
        mp = (w * h) / 1_000_000.0
        self.lbl_stats.setText(
            f"Resolución: {w} x {h} px ({mp:.1f} MP) | "
            f"Paneles: {result.panels_stitched}/{result.total_panels} | "
            f"Proyección: {result.projection.capitalize()}"
        )
        self.lbl_status.setText(f"✓ {result.message}")

        # Enviar al lienzo
        self.canvas.set_image(result.image, crop_rect=result.crop_rect)
        self.canvas.set_show_cropped(self.btn_toggle_crop.isChecked())

        self.btn_save_tiff.setEnabled(True)
        self.btn_to_developer.setEnabled(True)

    def _on_stitch_cancelled(self):
        self._set_processing_ui(False)
        self.lbl_status.setText("Proceso cancelado por el usuario.")

    def _on_stitch_error(self, err_msg: str):
        self._set_processing_ui(False)
        self.lbl_status.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "Error en Cosido Panorámico", f"Ocurrió un error al componer la panorámica:\n\n{err_msg}")

    def _set_processing_ui(self, is_processing: bool):
        self.btn_stitch.setVisible(not is_processing)
        self.btn_cancel.setVisible(is_processing)
        self.progress_bar.setVisible(is_processing)
        self.btn_add_files.setEnabled(not is_processing)
        self.btn_add_dir.setEnabled(not is_processing)
        self.btn_remove.setEnabled(not is_processing)
        self.btn_clear.setEnabled(not is_processing)

    def _on_save_tiff(self):
        if self.current_result is None:
            return

        img_to_save = (
            self.current_result.cropped_image
            if (self.btn_toggle_crop.isChecked() and self.current_result.cropped_image is not None)
            else self.current_result.image
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Panorámica",
            "panoramica_nocturna.tif",
            "TIFF 16-bit / 32-bit (*.tif *.tiff)"
        )
        if not path:
            return

        try:
            save_tiff_with_icc(path, img_to_save, compression='zlib', photometric='rgb')
            QMessageBox.information(self, "Panorámica Guardada", f"Panorámica guardada exitosamente en:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar", f"No se pudo guardar la imagen:\n{e}")

    def _on_export_to_developer(self):
        if self.current_result is None:
            return

        img_to_export = (
            self.current_result.cropped_image
            if (self.btn_toggle_crop.isChecked() and self.current_result.cropped_image is not None)
            else self.current_result.image
        )

        try:
            temp_path = os.path.join(tempfile.gettempdir(), f"apilator_panorama_{int(time.time())}.tif")
            save_tiff_with_icc(temp_path, img_to_export, compression='zlib', photometric='rgb')
            self.export_to_developer.emit(temp_path)
        except Exception as e:
            QMessageBox.critical(self, "Error de Transferencia", f"No se pudo transferir la panorámica al Revelador:\n{e}")

    def _on_save_session(self):
        if not self.panels_metadata:
            QMessageBox.warning(self, "Sesión Vacía", "No hay paneles cargados para guardar.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Sesión de Panorámica",
            "sesion_panoramica.panosess",
            "Sesión Panorámica (*.panosess)"
        )
        if not path:
            return

        session_data = {
            "panels": [
                {
                    "filepath": m.filepath,
                    "filename": m.filename,
                    "is_included": m.is_included,
                    "focal": m.focal_length_mm,
                    "time": m.time_str
                } for m in self.panels_metadata
            ],
            "projection": self.combo_projection.currentData(),
            "astro_mode": self.chk_astro_mode.isChecked(),
            "min_inliers": self.spin_min_inliers.value(),
            "exposure_comp": self.chk_exposure_comp.isChecked(),
            "blend_mode": self.combo_blend.currentData(),
            "blend_bands": self.spin_bands.value(),
            "auto_crop": self.chk_auto_crop.isChecked(),
            "layout_mode": self.combo_layout_mode.currentData(),
            "grid_rows": self.spin_rows.value(),
            "grid_cols": self.spin_cols.value()
        }

        if save_panorama_session(path, session_data):
            QMessageBox.information(self, "Sesión Guardada", f"Sesión guardada en:\n{path}")
        else:
            QMessageBox.critical(self, "Error", "No se pudo guardar el archivo de sesión.")

    def _on_load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Cargar Sesión de Panorámica",
            "",
            "Sesión Panorámica (*.panosess)"
        )
        if not path:
            return

        data = load_panorama_session(path)
        if not data:
            QMessageBox.critical(self, "Error", "No se pudo cargar o interpretar el archivo de sesión.")
            return

        # Restaurar paneles
        paths = [p["filepath"] for p in data.get("panels", []) if os.path.exists(p.get("filepath", ""))]
        if paths:
            self.panels_metadata.clear()
            self.load_filepaths(paths)

        # Restaurar configuración
        proj = data.get("projection", "cylindrical")
        idx = self.combo_projection.findData(proj)
        if idx >= 0:
            self.combo_projection.setCurrentIndex(idx)

        self.chk_astro_mode.setChecked(data.get("astro_mode", True))
        self.spin_min_inliers.setValue(data.get("min_inliers", 15))
        self.chk_exposure_comp.setChecked(data.get("exposure_comp", True))

        blend = data.get("blend_mode", "multiband")
        idx_b = self.combo_blend.findData(blend)
        if idx_b >= 0:
            self.combo_blend.setCurrentIndex(idx_b)

        self.spin_bands.setValue(data.get("blend_bands", 5))
        self.chk_auto_crop.setChecked(data.get("auto_crop", True))

        layout = data.get("layout_mode", "auto")
        idx_l = self.combo_layout_mode.findData(layout)
        if idx_l >= 0:
            self.combo_layout_mode.setCurrentIndex(idx_l)

        self.spin_rows.setValue(data.get("grid_rows", 1))
        self.spin_cols.setValue(data.get("grid_cols", 1))
        QMessageBox.information(self, "Sesión Cargada", f"Sesión cargada exitosamente.")
