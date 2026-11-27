# gui/tab_eclipse_collage.py
"""
gui/tab_eclipse_collage.py - Pestaña dedicada para el Diseñador interactivo de secuencias y collages de eclipses.
Permite:
  - Ingesta y cribado masivo de tomas con detección de nitidez/trepidación y % de ocultación en lote.
  - Recomendación automática de las mejores tomas simétricas (C1 a C4).
  - Maquetación en Cinta Horizontal, Escala Romboide (<>), Arco Parabólico o Cuadrícula.
  - Visor interactivo acelerado con Zoom por rueda centrado en ratón y Paneo.
  - Exportación directa a PSD multicapa 16-bit para Photoshop, guardado TIFF/JPG y envío directo al Revelador.
"""
import os
import tempfile
from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
    QSlider,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QSplitter,
    QScrollArea,
    QAbstractItemView,
    QApplication,
    QSizePolicy,
    QSpinBox,
    QColorDialog,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QDoubleSpinBox
)

from gui.widgets.viewer_toolbar import ViewerToolbar, CanvasFooterBar


from core.eclipse_collage import (
    EclipseFrameMetadata,
    batch_analyze_eclipse_series,
    recommend_key_frames,
    compute_collage_layout,
    render_flat_collage,
    export_layered_psd,
    save_eclipse_collage_session,
    load_eclipse_collage_session,
    LAYOUT_HORIZONTAL,
    LAYOUT_DIAMOND,
    LAYOUT_ARC,
    LAYOUT_GRID,
    HAS_PYTOSHOP
)
from core.config_manager import get_config_val
from core.log_hub import log_hub
from gui.theme import STYLE_RUN_IDLE, STYLE_TRANSFER_BTN, STYLE_SECONDARY_BTN, STYLE_SPECIAL_ACTION_BTN
from core.icc_manager import save_tiff_with_icc
from gui.widgets.drop_widgets import DropTableWidget
from gui.icons import get_icon, get_svg_pixmap, ICON_COLOR_DEFAULT, ICON_COLOR_ACTIVE, ICON_COLOR_ACCENT
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.widgets.export_button import ExportSplitButton


class CollageInteractiveCanvas(QWidget):
    """
    Lienzo interactivo acelerado para previsualizar secuencias y collages de eclipses.
    Permite:
      - Zoom dinámico con rueda de ratón centrado en la posición del puntero (0.05x a 8.0x).
      - Paneo interactivo mediante arrastre con clic izquierdo o botón central.
      - Botones directos de zoom in/out, 1:1 (tamaño real) y Ajustar al lienzo (Fit).
      - Renderizado suave y centrado con fondo oscuro estelar.
    """
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #0a0a0b; border: 1px solid #2e2e35; border-radius: 4px;")

        self._pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan_offset: QPointF = QPointF(0, 0)
        self._is_panning: bool = False
        self._drag_start: QPointF = QPointF(0, 0)
        self._fit_to_view: bool = True
        self.placeholder_text: str = "Carga tomas del eclipse y pulsa 'Analizar' para diseñar el collage."

    def set_image(self, flat_rgb: Optional[np.ndarray]):
        if flat_rgb is None:
            self._pixmap = None
            self.update()
            return

        u8 = (np.clip(flat_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w, c = u8.shape
        bytes_per_line = 3 * w
        qimg = QImage(u8.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        if self._fit_to_view:
            self.fit_view()
        else:
            self.update()

    def set_placeholder(self, text: str):
        self.placeholder_text = text
        self._pixmap = None
        self.update()

    def fit_view(self):
        self._fit_to_view = True
        if self._pixmap is None or self._pixmap.isNull():
            self._zoom = 1.0
            self._pan_offset = QPointF(0, 0)
            self.update()
            self.zoom_changed.emit(self._zoom)
            return

        vw = max(50, self.width() - 30)
        vh = max(50, self.height() - 30)
        pw = self._pixmap.width()
        ph = self._pixmap.height()

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
        if self._pixmap:
            cx = (self.width() - self._pixmap.width()) / 2.0
            cy = (self.height() - self._pixmap.height()) / 2.0
            self._pan_offset = QPointF(cx, cy)
        self.update()
        self.zoom_changed.emit(self._zoom)

    def zoom_by(self, factor: float):
        self._fit_to_view = False
        center_pos = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._apply_zoom_at(factor, center_pos)

    def _apply_zoom_at(self, factor: float, center_pos: QPointF):
        old_zoom = self._zoom
        new_zoom = max(0.05, min(10.0, self._zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        self._pan_offset = center_pos - (center_pos - self._pan_offset) * (new_zoom / old_zoom)
        self._zoom = new_zoom
        self.update()
        self.zoom_changed.emit(self._zoom)

    def wheelEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull():
            return
        self._fit_to_view = False
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self._apply_zoom_at(factor, event.position())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = True
            self._drag_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            self._fit_to_view = False
            delta = event.position() - self._drag_start
            self._pan_offset += delta
            self._drag_start = event.position()
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.fit_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_to_view:
            self.fit_view()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), QColor(14, 15, 20))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(120, 144, 156))
            font = painter.font()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, self.placeholder_text)
            return

        pw = self._pixmap.width() * self._zoom
        ph = self._pixmap.height() * self._zoom
        target_rect = QRectF(self._pan_offset.x(), self._pan_offset.y(), pw, ph)
        painter.drawPixmap(target_rect.toRect(), self._pixmap)


class EclipseAnalysisWorker(QThread):
    """Hilo secundario para el análisis de calidad y porcentaje de eclipse en lote."""
    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(
        self, 
        filepaths: List[str], 
        mode: str = "solar", 
        max_workers: Optional[int] = None, 
        fast_mode: bool = True
    ):
        super().__init__()
        self.filepaths = filepaths
        self.mode = mode
        self.max_workers = max_workers
        self.fast_mode = fast_mode
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            def cb(curr, total, msg):
                if not self._is_cancelled:
                    self.progress.emit(curr, total, msg)

            results = batch_analyze_eclipse_series(
                self.filepaths,
                mode=self.mode,
                progress_callback=cb,
                max_workers=self.max_workers,
                fast_mode=self.fast_mode,
                is_cancelled_fn=lambda: self._is_cancelled
            )
            if not self._is_cancelled:
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class EclipseCollageTab(QWidget):
    """
    Pestaña integrada de diseño interactivo de secuencias y collages de eclipses.
    """
    export_to_developer = Signal(str)

    def __init__(self, parent=None, initial_filepaths: Optional[List[str]] = None, mode: str = "solar"):
        super().__init__(parent)
        self.eclipse_mode = mode  # "solar" o "lunar"
        self.frames_meta: List[EclipseFrameMetadata] = []
        self.preview_image: Optional[np.ndarray] = None
        self.current_placements = []
        self.current_canvas_w = 1920
        self.current_canvas_h = 1080
        self._tile_cache: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        self.canvas_bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.canvas_bg_hex: str = "#000000"

        self.worker: Optional[EclipseAnalysisWorker] = None

        self._setup_ui()
        self.log_message("Módulo de Collage de Eclipses inicializado.")

        if initial_filepaths:
            self.load_filepaths(initial_filepaths)

    def log_message(self, message: str) -> None:
        """Registra un mensaje de actividad en LogHub y en la interfaz."""
        log_hub.log("Eclipses", f"[Collage] {message}")
        if hasattr(self, 'lbl_info'):
            self.lbl_info.setText(message)

    def _get_configured_workers(self) -> int:
        """Obtiene el número de hilos concurrentes configurado globalmente en AstroLoom."""
        cpu_total = os.cpu_count() or 16
        configured = int(get_config_val("cpu_workers", max(2, cpu_total - 2)))
        return max(1, min(configured, cpu_total))

    def set_mode(self, mode: str):
        """Actualiza la modalidad solar o lunar."""
        self.eclipse_mode = mode
        if hasattr(self, "combo_mode"):
            self.combo_mode.blockSignals(True)
            self.combo_mode.setCurrentIndex(0 if mode == "solar" else 1)
            self.combo_mode.blockSignals(False)

    def load_filepaths(self, filepaths: List[str]):
        """Carga una lista de archivos en la tabla."""
        self._load_filepaths(filepaths)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -----------------------------------------------------------------
        # Panel Izquierdo: Redimensionable con Splitter Vertical (420 - 750 px de ancho)
        # -----------------------------------------------------------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left_panel.setMaximumWidth(750)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(0)

        left_v_splitter = QSplitter(Qt.Vertical)
        left_v_splitter.setChildrenCollapsible(False)
        self.left_v_splitter = left_v_splitter

        # --- SECCIÓN 1: Ingesta y Cribado de Calidad (Zona Superior Redimensionable) ---
        self.sec_files = CollapsibleSection("1. Ingesta y Cribado de Calidad", is_expanded=True, icon_name="camera")
        self.sec_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self.sec_files.layout():
            self.sec_files.layout().setStretch(0, 0)
            self.sec_files.layout().setStretch(1, 1)
        self.sec_files.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        box_files = self.sec_files.content_layout
        box_files.setSpacing(6)

        # Modalidad Sol / Luna
        row_mode_sel = QHBoxLayout()
        row_mode_sel.addWidget(QLabel("Modalidad:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem(get_icon("sun"), "Eclipse Solar", "solar")
        self.combo_mode.addItem(get_icon("moon"), "Eclipse Lunar", "lunar")
        self.combo_mode.setCurrentIndex(0 if self.eclipse_mode == "solar" else 1)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_combo_changed)
        row_mode_sel.addWidget(self.combo_mode, stretch=1)
        box_files.addLayout(row_mode_sel)

        # Botones de sesión .ecollage
        btn_row_sess = QHBoxLayout()
        self.btn_load_session = QPushButton("Abrir Sesión...")
        self.btn_load_session.setIcon(get_icon("folder"))
        self.btn_load_session.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_load_session.setToolTip("Carga una sesión guardada (.ecollage) restaurando todo el análisis y selección instantáneamente")
        self.btn_load_session.clicked.connect(self._on_load_session_clicked)

        self.btn_save_session = QPushButton("Guardar Sesión...")
        self.btn_save_session.setIcon(get_icon("save"))
        self.btn_save_session.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_save_session.setToolTip("Guarda la sesión actual con todos los cálculos de análisis, calidad y selección en un archivo .ecollage")
        self.btn_save_session.clicked.connect(self._on_save_session_clicked)

        btn_row_sess.addWidget(self.btn_load_session)
        btn_row_sess.addWidget(self.btn_save_session)
        box_files.addLayout(btn_row_sess)

        # Cargar fotos / limpiar
        btn_row_f = QHBoxLayout()
        self.btn_load_files = QPushButton("Cargar Fotos...")
        self.btn_load_files.setIcon(get_icon("folder"))
        self.btn_load_files.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_load_files.clicked.connect(self._on_load_files_clicked)
        self.btn_clear_files = QPushButton("Limpiar")
        self.btn_clear_files.setIcon(get_icon("trash"))
        self.btn_clear_files.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_clear_files.clicked.connect(self._on_clear_clicked)
        self.btn_blink = QPushButton("Visor Blink")
        self.btn_blink.setIcon(get_icon("eye"))
        self.btn_blink.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_blink.setToolTip("Abre el visor interactivo Blink para revisar tomas del eclipse y cribar con teclado (X/P)")
        self.btn_blink.clicked.connect(self.open_blink_viewer)
        btn_row_f.addWidget(self.btn_load_files)
        btn_row_f.addWidget(self.btn_clear_files)
        btn_row_f.addWidget(self.btn_blink)
        box_files.addLayout(btn_row_f)

        # Tabla interactiva con Drag & Drop - Sección redimensionable y expandible
        self.table_files = DropTableWidget()
        self.table_files.setColumnCount(6)
        self.table_files.setHorizontalHeaderLabels(["Usar", "Archivo", "Hora", "% Ocult.", "Fase Detectada", "Nitidez / Estado"])
        self.table_files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_files.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_files.setMinimumHeight(190)
        self.table_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_files.itemChanged.connect(self._on_table_item_changed)
        self.table_files.files_dropped.connect(self._on_files_dropped)
        box_files.addWidget(self.table_files, 1)

        # Opciones de Rendimiento (hilos de CPU globales vía Ajustes)
        self.combo_analysis_speed = QComboBox()
        self.combo_analysis_speed.addItem(get_icon("zap"), "Modo Rápido (Half-size RAW 4x)", True)
        self.combo_analysis_speed.addItem(get_icon("target"), "Modo Precisión 1:1 (Full RAW)", False)
        self.combo_analysis_speed.setToolTip(
            "Modo Rápido: Procesa RAWs con binning 2x2 acelerando 4x a 8x el análisis sin perder precisión geométrica.\n"
            "Modo Precisión: Procesa RAWs a resolución 1:1 nativa completa.\n"
            "Nota: Los hilos concurrentes de CPU se gestionan desde la configuración global de la aplicación."
        )
        box_files.addWidget(self.combo_analysis_speed)

        self.btn_analyze = QPushButton("Analizar Calidad y Ocultación en Lote")
        self.btn_analyze.setIcon(get_icon("search"))
        self.btn_analyze.setFixedHeight(34)
        self.btn_analyze.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        box_files.addWidget(self.btn_analyze)

        # Contenedor superior para el splitter vertical
        ingest_container = QWidget()
        ingest_container_layout = QVBoxLayout(ingest_container)
        ingest_container_layout.setContentsMargins(4, 4, 10, 4)
        ingest_container_layout.setSpacing(6)
        ingest_container_layout.addWidget(self.sec_files, 1)

        # --- SECCIONES 2, 3 y 4 en ScrollArea Inferior ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        panel_content = QWidget()
        left_layout = QVBoxLayout(panel_content)
        left_layout.setContentsMargins(4, 4, 10, 4)
        left_layout.setSpacing(10)

        # --- SECCIÓN 2: Recomendador Inteligente ---
        self.sec_rec = CollapsibleSection("2. Recomendador Inteligente de Fases", is_expanded=True, icon_name="sparkles")
        box_rec = self.sec_rec.content_layout
        box_rec.setSpacing(6)

        row_rec_ctrl = QHBoxLayout()
        row_rec_ctrl.addWidget(QLabel("Nº Fases:"))
        self.combo_target_count = QComboBox()
        self.combo_target_count.addItems(["5 fases", "7 fases", "9 fases", "11 fases", "13 fases", "15 fases", "Todas"])
        self.combo_target_count.setCurrentIndex(2)  # 9 fases por defecto
        row_rec_ctrl.addWidget(self.combo_target_count)

        self.btn_auto_select = QPushButton("Auto-Seleccionar Mejores Tomas")
        self.btn_auto_select.setIcon(get_icon("zap"))
        self.btn_auto_select.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_auto_select.setToolTip("Selecciona simétricamente las fotos más nítidas descartando trepidaciones")
        self.btn_auto_select.clicked.connect(self._auto_select_recommended)
        row_rec_ctrl.addWidget(self.btn_auto_select)
        box_rec.addLayout(row_rec_ctrl)

        left_layout.addWidget(self.sec_rec)

        # --- SECCIÓN 3: Maquetación y Espaciado ---
        self.sec_layout = CollapsibleSection("3. Maquetación y Composición Espacial", is_expanded=True, icon_name="sliders")
        box_layout = self.sec_layout.content_layout
        box_layout.setSpacing(6)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Modalidad:"))
        self.combo_layout_mode = QComboBox()
        self.combo_layout_mode.addItems([
            "Cinta Horizontal (Timeline)",
            "Escala Romboide (<>)",
            "Arco Celeste (Parábola)",
            "Cuadrícula (Grid)"
        ])
        row_mode.addWidget(self.combo_layout_mode, stretch=1)
        box_layout.addLayout(row_mode)

        # Precision Sliders homogéneos
        self.slider_gap = PrecisionSliderRow("Espaciado", 10, 150, 40, unit=" px", decimals=0, label_width=110)
        self.lbl_gap_val = self.slider_gap.spin
        self.slider_gap.valueChanged.connect(self._on_gap_changed)
        box_layout.addWidget(self.slider_gap)

        self.slider_tot_scale = PrecisionSliderRow("Escala Centro", 100, 200, 130, scale=100.0, unit="x", decimals=2, label_width=110)
        self.lbl_tot_scale_val = self.slider_tot_scale.spin
        self.slider_tot_scale.valueChanged.connect(self._on_tot_scale_changed)
        box_layout.addWidget(self.slider_tot_scale)

        self.slider_pad = PrecisionSliderRow("Margen Corona", 120, 250, 160, scale=100.0, unit="x", decimals=2, label_width=110)
        self.lbl_pad_val = self.slider_pad.spin
        self.slider_pad.valueChanged.connect(self._on_pad_changed)
        box_layout.addWidget(self.slider_pad)

        # Selector de color de fondo del lienzo
        row_bg = QHBoxLayout()
        row_bg.addWidget(QLabel("Color Fondo:"))
        self.combo_bg_color = QComboBox()
        self.combo_bg_color.addItem("Negro Espacio", "#000000")
        self.combo_bg_color.addItem("Rojo Realce", "#8F2828")
        self.combo_bg_color.addItem("Gris Carbón", "#1E1E1E")
        self.combo_bg_color.addItem("Blanco Editorial", "#FFFFFF")
        self.combo_bg_color.addItem("Personalizado...", "custom")
        self.combo_bg_color.currentIndexChanged.connect(self._on_bg_color_changed)
        row_bg.addWidget(self.combo_bg_color, stretch=1)

        self.btn_bg_custom = QPushButton()
        self.btn_bg_custom.setFixedSize(26, 24)
        self.btn_bg_custom.setStyleSheet(f"background-color: {self.canvas_bg_hex}; border: 1px solid #2e2e35; border-radius: 3px;")
        self.btn_bg_custom.setToolTip("Elegir color de fondo personalizado")
        self.btn_bg_custom.clicked.connect(self._choose_custom_bg_color)
        row_bg.addWidget(self.btn_bg_custom)
        box_layout.addLayout(row_bg)

        self.chk_captions = QCheckBox("Mostrar marcas horarias EXIF bajo cada fase")
        self.chk_captions.setChecked(True)
        box_layout.addWidget(self.chk_captions)

        self.btn_apply_layout = QPushButton("Actualizar Composición / Collage")
        self.btn_apply_layout.setIcon(get_icon("refresh", color="#ffffff"))
        self.btn_apply_layout.setFixedHeight(36)
        self.btn_apply_layout.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_apply_layout.setToolTip("Aplica los cambios de modalidad, espaciado, escala y margen generando la vista previa")
        self.btn_apply_layout.clicked.connect(self._update_preview)
        box_layout.addWidget(self.btn_apply_layout)

        left_layout.addWidget(self.sec_layout)

        # --- SECCIÓN 4: Exportación y Enlace ---
        self.sec_exp = CollapsibleSection("4. Exportación y Acabado", is_expanded=True, icon_name="save")
        box_exp = self.sec_exp.content_layout
        box_exp.setSpacing(6)

        btn_row_exp = QHBoxLayout()
        btn_row_exp.setSpacing(6)

        self.btn_save_image = ExportSplitButton(
            text="Guardar...",
            on_master_export=self._on_save_collage_clicked,
            on_web_export=self._on_export_web_clicked,
            style="secondary",
            parent=self
        )
        self.btn_save_image.setup_standard_menu(
            on_format_export=lambda fmt: self._on_save_collage_clicked(force_format=fmt),
            psd_label="Guardar Photoshop en Capas (PSD)..."
        )
        self.btn_save_image.setFixedHeight(36)
        self.btn_save_image.setToolTip(
            "Exporta el collage según la extensión seleccionada en el diálogo:\n"
            "• Clic principal: Formatos maestros (TIFF 16/32-bit, PSD multicapa, PNG, JPEG directo).\n"
            "• Desplegable (▼) o Ctrl+Alt+W: Acceso directo a formatos y Exportar para Web / Redes Sociales (sRGB 8-bit, Lanczos, WebP/JPEG/PNG)."
        )

        # Aliases para compatibilidad hacia atrás
        self.btn_export_flat = self.btn_save_image
        self.btn_export_psd = self.btn_save_image

        # Enlace directo al Revelador (en línea a la derecha de Guardar)
        self.btn_export_developer = QPushButton("Enviar al Revelador")
        self.btn_export_developer.setIcon(get_icon("sliders", size=13, color="#2f81f7"))
        self.btn_export_developer.setFixedHeight(36)
        self.btn_export_developer.setStyleSheet(STYLE_TRANSFER_BTN)
        self.btn_export_developer.setToolTip("Exporta el collage como imagen plana y la abre inmediatamente en el Revelador para ajuste tonal y color")
        self.btn_export_developer.clicked.connect(self._export_to_developer_clicked)

        btn_row_exp.addWidget(self.btn_save_image, 1)
        btn_row_exp.addWidget(self.btn_export_developer, 1)
        box_exp.addLayout(btn_row_exp)

        left_layout.addWidget(self.sec_exp)
        left_layout.addStretch()

        scroll_area.setWidget(panel_content)

        # Ensamblar divisor vertical
        left_v_splitter.addWidget(ingest_container)
        left_v_splitter.addWidget(scroll_area)
        left_v_splitter.setStretchFactor(0, 3)
        left_v_splitter.setStretchFactor(1, 4)
        left_v_splitter.setSizes([400, 460])

        left_panel_layout.addWidget(left_v_splitter)
        self.splitter = splitter
        self.left_panel = left_panel
        splitter.addWidget(left_panel)

        # -----------------------------------------------------------------
        # Panel Derecho: Toolbar Unificada + Vista Previa Interactiva del Collage
        # -----------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.toolbar = ViewerToolbar(
            self,
            show_left_toggle=True,
            show_right_toggle=False,
            show_view_mode=False,
            show_composite_btn=False,
            show_clipping=False,
            show_zoom=True,
            show_magnifier=False,
        )
        self.toolbar.left_panel_toggled.connect(self.toggle_left_panel)
        self.btn_zoom_100 = self.toolbar.btn_zoom_100
        self.btn_zoom_fit = self.toolbar.btn_zoom_fit

        # Botón Actualizar composición
        self.btn_refresh = QPushButton("Actualizar")
        self.btn_refresh.setIcon(get_icon("refresh"))
        self.btn_refresh.setToolTip("Re-renderizar la composición previa del collage")
        self.btn_refresh.clicked.connect(self._update_preview)
        self.toolbar.insert_custom_widget_before_stretch(self.btn_refresh)

        self.toolbar.add_separator_before_stretch()

        # Botones de navegación interactiva y zoom gradual (+ / - / %)
        self.btn_zoom_out = QPushButton()
        self.btn_zoom_out.setIcon(get_icon("zoom_out", color="#94a3b8", hover_color="#ffffff", size=13))
        self.btn_zoom_out.setToolTip("Alejar zoom (-)")
        self.btn_zoom_out.setFixedSize(28, 26)
        self.btn_zoom_out.clicked.connect(lambda: self.canvas_preview.zoom_by(1.0 / 1.25))
        self.toolbar.insert_custom_widget_before_stretch(self.btn_zoom_out)

        self.lbl_zoom_val = QLabel("100%")
        self.lbl_zoom_val.setStyleSheet("color: #2f81f7; font-weight: bold; min-width: 44px;")
        self.lbl_zoom_val.setAlignment(Qt.AlignCenter)
        self.toolbar.insert_custom_widget_before_stretch(self.lbl_zoom_val)

        self.btn_zoom_in = QPushButton()
        self.btn_zoom_in.setIcon(get_icon("zoom_in", color="#94a3b8", hover_color="#ffffff", size=13))
        self.btn_zoom_in.setToolTip("Acercar zoom (+)")
        self.btn_zoom_in.setFixedSize(28, 26)
        self.btn_zoom_in.clicked.connect(lambda: self.canvas_preview.zoom_by(1.25))
        self.toolbar.insert_custom_widget_before_stretch(self.btn_zoom_in)

        # Badge de Modo a la derecha (estilo idéntico a Eclipse HDR)
        self.lbl_mode_badge = QLabel("● Modo: Solar (Collage)")
        self.lbl_mode_badge.setStyleSheet(
            "background-color: #222227; border: 1px solid #2e2e35; border-radius: 4px; padding: 2px 8px; color: #2f81f7; font-weight: bold; font-size: 11px;"
        )
        self.toolbar.add_widget(self.lbl_mode_badge)

        # Etiqueta informativa de fases
        self.lbl_info = QLabel("0 fases seleccionadas")
        self.lbl_info.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        self.toolbar.add_widget(self.lbl_info)

        right_layout.addWidget(self.toolbar)

        # Crear lienzo interactivo
        self.canvas_preview = CollageInteractiveCanvas()
        self.canvas_preview.zoom_changed.connect(self._on_zoom_changed)
        self.btn_zoom_100.clicked.connect(self.canvas_preview.set_zoom_100)
        self.btn_zoom_fit.clicked.connect(self.canvas_preview.fit_view)

        right_layout.addWidget(self.canvas_preview, stretch=1)
        self.canvas_footer = CanvasFooterBar(self)
        self.progress_bar = self.canvas_footer.progress_bar
        right_layout.addWidget(self.canvas_footer)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        # Atajos de teclado locales
        self.shortcut_f7 = QShortcut(QKeySequence(Qt.Key_F7), self)
        self.shortcut_f7.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_f7.activated.connect(lambda: self.toggle_left_panel())

        self.shortcut_bracket = QShortcut(QKeySequence(Qt.Key_BracketLeft), self)
        self.shortcut_bracket.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_bracket.activated.connect(lambda: self.toggle_left_panel())

    def _on_mode_combo_changed(self, idx: int):
        self.eclipse_mode = "solar" if idx == 0 else "lunar"
        if hasattr(self, 'lbl_mode_badge'):
            if self.eclipse_mode == "solar":
                self.lbl_mode_badge.setText("● Modo: Solar (Collage)")
                self.lbl_mode_badge.setStyleSheet(
                    "background-color: #222227; border: 1px solid #2e2e35; border-radius: 4px; padding: 2px 8px; color: #2f81f7; font-weight: bold; font-size: 11px;"
                )
            else:
                self.lbl_mode_badge.setText("● Modo: Lunar (Collage)")
                self.lbl_mode_badge.setStyleSheet(
                    "background-color: #222227; border: 1px solid #2e2e35; border-radius: 4px; padding: 2px 8px; color: #2f81f7; font-weight: bold; font-size: 11px;"
                )
        self.log_message(f"Modalidad cambiada a: Eclipse {'Solar' if self.eclipse_mode == 'solar' else 'Lunar'}")

    def toggle_left_panel(self, visible: Optional[bool] = None) -> None:
        """Muestra u oculta el panel lateral izquierdo de controles de collage de eclipses (Atajo: F7 o [)."""
        if not hasattr(self, 'left_panel'):
            return
        if visible is None:
            visible = not self.left_panel.isVisible()
        if not visible and self.left_panel.isVisible() and hasattr(self, 'splitter'):
            sizes = self.splitter.sizes()
            if len(sizes) >= 2 and sizes[0] > 50:
                self._last_left_panel_width = sizes[0]
        self.left_panel.setVisible(visible)
        if hasattr(self, 'toolbar'):
            self.toolbar.set_left_panel_checked(visible)
        if visible and hasattr(self, 'splitter'):
            sizes = self.splitter.sizes()
            if len(sizes) >= 2 and sizes[0] < 50:
                sizes[0] = getattr(self, '_last_left_panel_width', 450)
                self.splitter.setSizes(sizes)

    def keyPressEvent(self, event):
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key_F7, Qt.Key_BracketLeft):
            self.toggle_left_panel()
            event.accept()
            return
        super().keyPressEvent(event)

    # -----------------------------------------------------------------
    # Gestión de Sesiones de Collage (.ecollage)
    # -----------------------------------------------------------------
    def _on_save_session_clicked(self):
        if not self.frames_meta:
            QMessageBox.warning(self, "Aviso", "No hay fotos cargadas ni analizadas para guardar en la sesión.")
            return

        sessions_dir = get_config_val("sessions_dir", "sesiones")
        os.makedirs(sessions_dir, exist_ok=True)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Sesión de Collage de Eclipse",
            os.path.join(sessions_dir, "mi_eclipse.ecollage"),
            "Sesión de Collage (*.ecollage);;JSON (*.json)"
        )
        if not filepath:
            return

        settings = {
            "layout_mode_idx": self.combo_layout_mode.currentIndex(),
            "gap_spacing": self.slider_gap.value(),
            "totality_scale": self.slider_tot_scale.value(),
            "crop_margin_factor": self.slider_pad.value(),
            "bg_hex": self.canvas_bg_hex,
            "draw_captions": self.chk_captions.isChecked(),
            "target_count_idx": self.combo_target_count.currentIndex(),
            "fast_mode_idx": self.combo_analysis_speed.currentIndex(),
            "analysis_workers": self._get_configured_workers()
        }

        ok = save_eclipse_collage_session(filepath, self.frames_meta, settings, mode=self.eclipse_mode)
        if ok:
            self.log_message(f"✓ Sesión de collage guardada: {os.path.basename(filepath)}")
            QMessageBox.information(
                self,
                "Sesión Guardada",
                f"Sesión guardada exitosamente con {len(self.frames_meta)} fotogramas analizados:\n{filepath}"
            )
        else:
            self.log_message(f"[ERROR] No se pudo guardar la sesión en: {filepath}")
            QMessageBox.critical(self, "Error", "No se pudo guardar el archivo de sesión.")

    def _on_load_session_clicked(self):
        sessions_dir = get_config_val("sessions_dir", "sesiones")
        os.makedirs(sessions_dir, exist_ok=True)
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir Sesión de Collage de Eclipse",
            sessions_dir,
            "Sesión de Collage (*.ecollage *.json);;Todos (*.*)"
        )
        if not filepath:
            return

        try:
            frames, settings, mode = load_eclipse_collage_session(filepath)
            if not frames:
                QMessageBox.warning(self, "Aviso", "El archivo de sesión no contiene fotogramas válidos.")
                return

            self.set_mode(mode)
            self.frames_meta = frames
            self._tile_cache.clear()

            # Restaurar parámetros de maquetación si existen
            if "layout_mode_idx" in settings:
                self.combo_layout_mode.setCurrentIndex(int(settings["layout_mode_idx"]))
            if "gap_spacing" in settings:
                val = int(settings["gap_spacing"])
                self.slider_gap.setValue(val)
                self.lbl_gap_val.setText(f"{val} px")
            if "totality_scale" in settings:
                val = int(settings["totality_scale"])
                self.slider_tot_scale.setValue(val)
                val_f = val / 100.0
                diff = val - 100
                self.lbl_tot_scale_val.setText(f"{val_f:.2f}x ({'+' if diff >= 0 else ''}{diff}%)")
            if "crop_margin_factor" in settings:
                val = int(settings["crop_margin_factor"])
                self.slider_pad.setValue(val)
                self.lbl_pad_val.setText(f"{val / 100.0:.2f}x")
            if "bg_hex" in settings:
                hex_col = str(settings["bg_hex"])
                self._set_bg_hex(hex_col)
                found = False
                for i in range(self.combo_bg_color.count() - 1):
                    if str(self.combo_bg_color.itemData(i)).lower() == hex_col.lower():
                        self.combo_bg_color.blockSignals(True)
                        self.combo_bg_color.setCurrentIndex(i)
                        self.combo_bg_color.blockSignals(False)
                        found = True
                        break
                if not found:
                    self.combo_bg_color.blockSignals(True)
                    self.combo_bg_color.setCurrentIndex(4)
                    self.combo_bg_color.blockSignals(False)
            if "draw_captions" in settings:
                self.chk_captions.setChecked(bool(settings["draw_captions"]))

            if "target_count_idx" in settings:
                self.combo_target_count.setCurrentIndex(int(settings["target_count_idx"]))
            if "fast_mode_idx" in settings:
                self.combo_analysis_speed.setCurrentIndex(int(settings["fast_mode_idx"]))
            if "analysis_workers" in settings:
                pass  # Los hilos de CPU ahora se leen de la configuración global de la aplicación

            self._populate_table()
            self._update_preview()

            num_rec = len([f for f in frames if f.is_recommended])
            msg = f"Sesión restaurada: {len(frames)} fotogramas cargados ({num_rec} seleccionados)."
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(f"Sesión cargada ({len(frames)} fotos, {num_rec} sel.)")
            self.lbl_info.setText(msg)
            if hasattr(self, 'canvas_footer'):
                self.canvas_footer.set_progress(100, f"Sesión cargada: {len(frames)} fotos")
            self.log_message(f"✓ {msg}")
        except Exception as e:
            self.log_message(f"[ERROR] Al cargar sesión: {e}")
            QMessageBox.critical(self, "Error al cargar sesión", f"No se pudo cargar la sesión:\n{e}")

    # -----------------------------------------------------------------
    # Gestión de Archivos y Tabla
    # -----------------------------------------------------------------
    def _on_load_files_clicked(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar fotos del eclipse",
            "",
            "Imágenes (*.tif *.tiff *.jpg *.jpeg *.png *.cr2 *.cr3 *.nef *.arw *.dng *.fits *.fit)"
        )
        if files:
            self._load_filepaths(files)

    def _on_files_dropped(self, files: List[str]):
        if files:
            self._load_filepaths(files)

    def _load_filepaths(self, filepaths: List[str]):
        if not filepaths:
            return
        total = len(filepaths)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Cargando ficheros... (%v/{total})")
        self.lbl_info.setText(f"Cargando {total} ficheros...")
        QApplication.processEvents()

        new_metas = []
        for idx, fp in enumerate(filepaths):
            fn = os.path.basename(fp)
            new_metas.append(EclipseFrameMetadata(
                filepath=fp,
                filename=fn,
                timestamp=0.0,
                time_str="--:--:--",
                exp_time=0.001,
                iso=100,
                center_x=0.0,
                center_y=0.0,
                radius=100.0,
                sharpness_score=0.0,
                is_trepidated=False,
                obscuration_pct=0.0,
                is_ingress=True,
                phase_label="Pendiente",
                is_recommended=True
            ))
            if (idx + 1) % 25 == 0 or idx == total - 1:
                self.progress_bar.setValue(idx + 1)
                if hasattr(self, 'canvas_footer'):
                    self.canvas_footer.set_progress(int((idx + 1) / total * 100), f"Cargando {idx + 1}/{total}")
                QApplication.processEvents()

        self.frames_meta.extend(new_metas)
        self._populate_table()
        msg = f"Carga finalizada: {len(new_metas)} tomas añadidas (Total: {len(self.frames_meta)})."
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(f"Cargadas {len(self.frames_meta)} fotos (pulsa 'Analizar')")
        self.lbl_info.setText(f"{msg} Pulsa 'Analizar'.")
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.set_progress(100, f"Cargadas {len(self.frames_meta)} fotos")
        self.log_message(f"✓ {msg}")

    def _on_clear_clicked(self):
        self._tile_cache.clear()
        self.frames_meta.clear()
        self.table_files.setRowCount(0)
        self.canvas_preview.set_image(None)
        self.lbl_info.setText("Vista Previa del Collage (0 fases seleccionadas)")
        self.progress_bar.setVisible(False)
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.hide_progress()
        self.log_message("Lista de fotos y vista previa vaciadas.")

    def _populate_table(self):
        self.table_files.blockSignals(True)
        self.table_files.setRowCount(len(self.frames_meta))

        for row, f in enumerate(self.frames_meta):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if f.is_recommended else Qt.Unchecked)
            self.table_files.setItem(row, 0, chk_item)

            name_item = QTableWidgetItem(f.filename)
            name_item.setToolTip(f.filepath)
            self.table_files.setItem(row, 1, name_item)

            time_item = QTableWidgetItem(f.time_str)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.table_files.setItem(row, 2, time_item)

            obsc_str = f"{f.obscuration_pct:.1f}%" if f.sharpness_score > 0 else "--"
            obsc_item = QTableWidgetItem(obsc_str)
            obsc_item.setTextAlignment(Qt.AlignCenter)
            self.table_files.setItem(row, 3, obsc_item)

            # Columna 4: Fase Detectada con estilizado cromático
            phase_text = f.phase_label or "Pendiente"
            phase_item = QTableWidgetItem(phase_text)
            phase_item.setTextAlignment(Qt.AlignCenter)
            if "TOTALIDAD" in phase_text:
                phase_item.setForeground(QColor("#2f81f7"))
                font = phase_item.font()
                font.setBold(True)
                phase_item.setFont(font)
            elif "Diamante" in phase_text or "Baily" in phase_text:
                phase_item.setForeground(QColor("#e4e4e7"))
                font = phase_item.font()
                font.setBold(True)
                phase_item.setFont(font)
            elif "Entrada" in phase_text or "Salida" in phase_text:
                phase_item.setForeground(QColor("#a1a1aa"))
            else:
                phase_item.setForeground(QColor("#71717a"))
            self.table_files.setItem(row, 4, phase_item)

            # Columna 5: Nitidez / Estado
            if f.sharpness_score > 0:
                if f.is_trepidated:
                    status_text = f"● {f.sharpness_score:.0f} (Movida)"
                    color = QColor("#ef4444")
                elif f.sharpness_score >= 80:
                    status_text = f"● {f.sharpness_score:.0f} (Nítida)"
                    color = QColor("#2f81f7")
                else:
                    status_text = f"● {f.sharpness_score:.0f}"
                    color = QColor("#a1a1aa")
            else:
                status_text = "Pendiente"
                color = QColor("#71717a")

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(color)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.table_files.setItem(row, 5, status_item)

        self.table_files.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            row = item.row()
            if 0 <= row < len(self.frames_meta):
                self.frames_meta[row].is_recommended = (item.checkState() == Qt.Checked)
                self._update_preview()

    def _compute_eclipse_contacts(self) -> Dict[str, float]:
        """Calcula o estima las marcas temporales de contacto (C1, C2, MAX, C3, C4) a partir de frames_meta."""
        valid = [f for f in self.frames_meta if f.timestamp > 0]
        if not valid:
            return {}
        valid.sort(key=lambda f: f.timestamp)
        t_c1 = valid[0].timestamp
        t_c4 = valid[-1].timestamp

        tot_frames = [f for f in valid if ("TOTALIDAD" in (f.phase_label or "").upper()) or (f.obscuration_pct >= 98.0)]
        c2_frames = [f for f in valid if (("DIAMANTE" in (f.phase_label or "").upper() or "BAILY" in (f.phase_label or "").upper()) and f.is_ingress) or ("(C2)" in (f.phase_label or "").upper())]
        c3_frames = [f for f in valid if (("DIAMANTE" in (f.phase_label or "").upper() or "BAILY" in (f.phase_label or "").upper()) and not f.is_ingress) or ("(C3)" in (f.phase_label or "").upper())]

        contacts: Dict[str, float] = {
            "C1": t_c1,
            "C4": t_c4,
        }

        if tot_frames or c2_frames or c3_frames:
            t_c2 = c2_frames[0].timestamp if c2_frames else tot_frames[0].timestamp
            t_c3 = c3_frames[-1].timestamp if c3_frames else tot_frames[-1].timestamp
            t_max = tot_frames[len(tot_frames) // 2].timestamp if tot_frames else (t_c2 + t_c3) / 2.0
            contacts["C2"] = t_c2
            contacts["MAX"] = t_max
            contacts["C3"] = t_c3
        else:
            best_frame = max(valid, key=lambda f: f.obscuration_pct if f.obscuration_pct > 0 else 0)
            if best_frame.obscuration_pct > 0:
                t_max = best_frame.timestamp
            else:
                t_max = (t_c1 + t_c4) / 2.0
            contacts["MAX"] = t_max

        return contacts

    def open_blink_viewer(self):
        """Abre el visor interactivo Blink para revisar tomas del eclipse."""
        if not self.frames_meta:
            QMessageBox.information(
                self,
                "Visor Blink",
                "No hay fotogramas cargados para inspeccionar.\n\n"
                "Carga primero tomas del eclipse antes de abrir el visor Blink."
            )
            return

        from gui.dialog_blink import BlinkDialog

        file_list = [f.filepath for f in self.frames_meta]
        excluded = {f.filepath for f in self.frames_meta if not f.is_recommended}

        meta_cache: Dict[str, Dict[str, Any]] = {}
        for f in self.frames_meta:
            exp = f.exp_time
            if exp is not None and exp > 0:
                if exp >= 1.0:
                    exp_str = f"{exp:.1f}s" if exp % 1 != 0 else f"{int(exp)}s"
                else:
                    exp_str = f"1/{int(round(1.0 / exp))}s"
            else:
                exp_str = "—"
            iso_str = f"{int(f.iso)}" if (f.iso is not None and f.iso > 0) else "—"
            meta_cache[f.filepath] = {
                "exp": f.exp_time,
                "exp_str": exp_str,
                "iso": f.iso,
                "iso_str": iso_str,
                "aperture": None,
                "aperture_str": "—",
                "timestamp": f.timestamp,
                "time_str": f.time_str,
                "datetime_str": f.time_str,
                "phase_label": f.phase_label,
                "obscuration_pct": f.obscuration_pct,
                "is_ingress": f.is_ingress,
                "sharpness_score": f.sharpness_score,
                "is_trepidated": f.is_trepidated,
            }

        contacts = self._compute_eclipse_contacts()

        selected_rows = [idx.row() for idx in self.table_files.selectedIndexes()]
        curr_idx = selected_rows[0] if selected_rows else 0

        dlg = BlinkDialog(
            file_list=file_list,
            excluded_files=excluded,
            current_index=curr_idx,
            metadata_cache=meta_cache,
            mode="eclipse",
            contacts=contacts,
            parent=self
        )
        dlg.exclusion_changed.connect(self._on_blink_exclusion_changed)
        dlg.selection_applied.connect(self._on_blink_selection_applied)
        dlg.exec()

    def _on_blink_exclusion_changed(self, filepath: str, is_included: bool):
        for row, f in enumerate(self.frames_meta):
            if f.filepath == filepath:
                f.is_recommended = is_included
                item = self.table_files.item(row, 0)
                if item:
                    self.table_files.blockSignals(True)
                    item.setCheckState(Qt.Checked if is_included else Qt.Unchecked)
                    self.table_files.blockSignals(False)
                break
        self._update_preview()

    def _on_blink_selection_applied(self, excluded_set: set):
        self.table_files.blockSignals(True)
        for row, f in enumerate(self.frames_meta):
            f.is_recommended = (f.filepath not in excluded_set)
            item = self.table_files.item(row, 0)
            if item:
                item.setCheckState(Qt.Checked if f.is_recommended else Qt.Unchecked)
        self.table_files.blockSignals(False)
        self._update_preview()

    # -----------------------------------------------------------------
    # Análisis y Clasificación Automática
    # -----------------------------------------------------------------
    def _on_analyze_clicked(self):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.btn_analyze.setText("Cancelando...")
            self.btn_analyze.setEnabled(False)
            return
        self._start_analysis()

    def _start_analysis(self):
        if not self.frames_meta:
            QMessageBox.information(self, "Aviso", "Añade fotos primero.")
            return

        fps = [f.filepath for f in self.frames_meta]
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.set_progress(0, "Iniciando análisis...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_analyze.setText("Cancelar Análisis")
        self.btn_analyze.setIcon(get_icon("stop"))
        self.btn_analyze.setStyleSheet("font-weight: bold; background: #352020; color: #f87171; border: 1px solid #7f1d1d; border-radius: 4px; padding: 6px;")
        self.btn_analyze.setEnabled(True)

        fast_mode = bool(self.combo_analysis_speed.currentData())
        workers = self._get_configured_workers()
        self.log_message(f"Iniciando análisis de calidad y ocultación ({len(fps)} tomas, {workers} hilos de CPU, modo {'rápido' if fast_mode else 'precisión'})...")

        self.worker = EclipseAnalysisWorker(
            fps, 
            mode=self.eclipse_mode, 
            max_workers=workers, 
            fast_mode=fast_mode
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, curr, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)
        self.progress_bar.setFormat(f"{curr}/{total} - {msg}")
        if hasattr(self, 'canvas_footer'):
            pct = int((curr / max(1, total)) * 100)
            self.canvas_footer.set_progress(pct, msg)

    def _on_worker_finished(self, results: List[EclipseFrameMetadata]):
        self.progress_bar.setVisible(False)
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.hide_progress()
        self.btn_analyze.setText("Analizar Calidad y Ocultación en Lote")
        self.btn_analyze.setIcon(get_icon("search"))
        self.btn_analyze.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_analyze.setEnabled(True)
        if results:
            self._tile_cache.clear()
            self.frames_meta = results
            self._populate_table()
            self._auto_select_recommended()
            self.log_message(f"✓ Análisis de lote completado: {len(results)} tomas procesadas con éxito.")

    def _on_worker_error(self, err: str):
        self.progress_bar.setVisible(False)
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.hide_progress()
        self.btn_analyze.setText("Analizar Calidad y Ocultación en Lote")
        self.btn_analyze.setIcon(get_icon("search"))
        self.btn_analyze.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_analyze.setEnabled(True)
        self.log_message(f"[ERROR] En análisis de lote: {err}")
        QMessageBox.critical(self, "Error de Análisis", f"Ocurrió un error analizando el lote:\n{err}")

    def _auto_select_recommended(self):
        count_str = self.combo_target_count.currentText()
        if count_str == "Todas":
            num = len(self.frames_meta)
        else:
            num = int(count_str.split()[0])

        recommend_key_frames(self.frames_meta, num_targets=num)
        self._populate_table()
        self._update_preview()
        self.log_message(f"Recomendador automático: seleccionadas {num} fases clave del eclipse.")

    # -----------------------------------------------------------------
    # Callbacks de Controles y Zoom
    # -----------------------------------------------------------------
    def _on_gap_changed(self, val: int):
        self.lbl_gap_val.setText(f"{val} px")

    def _on_tot_scale_changed(self, val: int):
        val_f = val / 100.0
        diff = val - 100
        self.lbl_tot_scale_val.setText(f"{val_f:.2f}x ({'+' if diff >= 0 else ''}{diff}%)")

    def _on_pad_changed(self, val: int):
        val_f = val / 100.0
        self.lbl_pad_val.setText(f"{val_f:.2f}x")

    def _on_bg_color_changed(self, idx: int):
        val = self.combo_bg_color.currentData()
        if val == "custom":
            self._choose_custom_bg_color()
        else:
            self._set_bg_hex(val)
            self._update_preview()

    def _choose_custom_bg_color(self):
        col = QColorDialog.getColor(QColor(self.canvas_bg_hex), self, "Seleccionar Color de Fondo del Lienzo")
        if col.isValid():
            hex_col = col.name()
            self._set_bg_hex(hex_col)
            self.combo_bg_color.blockSignals(True)
            self.combo_bg_color.setCurrentIndex(4)
            self.combo_bg_color.blockSignals(False)
            self._update_preview()

    def _set_bg_hex(self, hex_str: str):
        self.canvas_bg_hex = hex_str
        c = QColor(hex_str)
        self.canvas_bg_color = (c.redF(), c.greenF(), c.blueF())
        self.btn_bg_custom.setStyleSheet(f"background-color: {hex_str}; border: 1px solid #2e2e35; border-radius: 3px;")


    def _on_zoom_changed(self, zoom: float):
        pct = int(round(zoom * 100))
        self.lbl_zoom_val.setText(f"{pct}%")
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.set_zoom_text(f"Zoom: {pct}%")

    def _get_current_layout_mode(self) -> str:
        idx = self.combo_layout_mode.currentIndex()
        if idx == 0:
            return LAYOUT_HORIZONTAL
        elif idx == 1:
            return LAYOUT_DIAMOND
        elif idx == 2:
            return LAYOUT_ARC
        return LAYOUT_GRID

    # -----------------------------------------------------------------
    # Generación y Vista Previa del Collage
    # -----------------------------------------------------------------
    def _get_selected_frames(self) -> List[EclipseFrameMetadata]:
        return [f for f in self.frames_meta if f.is_recommended]

    def _update_preview(self):
        selected = self._get_selected_frames()
        self.lbl_info.setText(f"Vista Previa del Collage ({len(selected)} fases seleccionadas)")
        if not selected:
            self.canvas_preview.set_placeholder("Selecciona al menos una fase para previsualizar.")
            return

        layout_mode = self._get_current_layout_mode()
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        crop_factor = self.slider_pad.value() / 100.0
        draw_caps = self.chk_captions.isChecked()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            workers = self._get_configured_workers()
            cw, ch, placements = compute_collage_layout(
                selected,
                layout_type=layout_mode,
                canvas_width=1800,
                target_disk_radius=90,
                gap_spacing=gap,
                totality_scale=tot_scale,
                crop_margin_factor=crop_factor,
                fast_preview=True,
                max_workers=workers,
                cached_tiles=self._tile_cache
            )
            self.current_placements = placements
            self.current_canvas_w = cw
            self.current_canvas_h = ch

            flat = render_flat_collage(cw, ch, placements, bg_color=self.canvas_bg_color, draw_captions=draw_caps)
            self.preview_image = flat
            self.canvas_preview.set_image(flat)
            if hasattr(self, 'canvas_footer') and flat is not None:
                self.canvas_footer.set_image_info(cw, ch, "8-bit", "sRGB")
        except Exception as e:
            self.canvas_preview.set_placeholder(f"Error generando vista previa: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    # -----------------------------------------------------------------
    # Exportaciones y Guardado Unificado
    # -----------------------------------------------------------------
    def _on_save_collage_clicked(self, force_format: Optional[str] = None):
        """Guarda el collage en el formato seleccionado por extensión o filtro (TIFF 16/32, PSD, PNG, JPG)."""
        selected = self._get_selected_frames()
        if not selected:
            QMessageBox.warning(self, "Aviso", "No hay fases seleccionadas para exportar.")
            return

        export_dir = get_config_val("export_dir", "exportaciones")
        os.makedirs(export_dir, exist_ok=True)

        filtros = (
            "TIFF 16-bit (*.tif *.tiff);;"
            "TIFF 32-bit Float (*.tif *.tiff);;"
            "Photoshop Multicapa (*.psd);;"
            "PNG (*.png);;"
            "JPEG Alta Calidad (*.jpg *.jpeg)"
        )

        if force_format == "psd":
            filtro_por_defecto = "Photoshop Multicapa (*.psd)"
            default_name = "eclipse_collage_capas.psd"
        elif force_format == "tiff32":
            filtro_por_defecto = "TIFF 32-bit Float (*.tif *.tiff)"
            default_name = "eclipse_collage_32bit.tif"
        elif force_format == "png":
            filtro_por_defecto = "PNG (*.png)"
            default_name = "eclipse_collage.png"
        elif force_format in ("jpg", "jpeg"):
            filtro_por_defecto = "JPEG Alta Calidad (*.jpg *.jpeg)"
            default_name = "eclipse_collage.jpg"
        else:
            filtro_por_defecto = "TIFF 16-bit (*.tif *.tiff)"
            default_name = "eclipse_collage_16bit.tif"

        default_path = os.path.join(export_dir, default_name)

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Guardar Collage de Eclipse",
            default_path,
            filtros,
            selectedFilter=filtro_por_defecto
        )
        if not filepath:
            return

        filepath = os.path.abspath(os.path.normpath(str(filepath).strip()))
        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            if "Photoshop" in selected_filter or "*.psd" in selected_filter:
                ext = ".psd"
            elif "PNG" in selected_filter or "*.png" in selected_filter:
                ext = ".png"
            elif "JPEG" in selected_filter or "*.jpg" in selected_filter:
                ext = ".jpg"
            else:
                ext = ".tif"
            filepath += ext

        is_32bit = ("32-bit" in selected_filter)
        if ext == ".psd" or ("Photoshop" in selected_filter and ext not in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]):
            self._do_export_psd(filepath, selected)
        else:
            self._do_export_flat(filepath, selected, ext, is_32bit=is_32bit)

    def _on_export_web_clicked(self):
        """Abre inmediatamente el diálogo modal de exportación web mientras el collage se genera en segundo plano."""
        selected = self._get_selected_frames()
        if not selected:
            QMessageBox.warning(self, "Aviso", "No hay fases seleccionadas para exportar.")
            return

        layout_mode = self._get_current_layout_mode()
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        crop_factor = self.slider_pad.value() / 100.0
        draw_caps = self.chk_captions.isChecked()
        workers = self._get_configured_workers()
        bg_col = self.canvas_bg_color

        def _render_full_collage() -> Optional[np.ndarray]:
            cw, ch, placements = compute_collage_layout(
                selected,
                layout_type=layout_mode,
                canvas_width=3840,
                target_disk_radius=150,
                gap_spacing=gap,
                totality_scale=tot_scale,
                crop_margin_factor=crop_factor,
                fast_preview=False,
                max_workers=workers
            )
            return render_flat_collage(cw, ch, placements, bg_color=bg_col, draw_captions=draw_caps)

        from gui.dialog_web_export import WebExportDialog, extract_exif_bytes
        first_path = selected[0].filepath if selected and hasattr(selected[0], "filepath") else None
        exif_bytes = extract_exif_bytes(first_path) if first_path else None
        base_name = "eclipse_collage"

        dlg = WebExportDialog(
            image_buffer=None,
            render_fn=_render_full_collage,
            initial_dimensions=(3840, 2160),
            default_filename=base_name,
            exif_bytes=exif_bytes,
            source_colorspace="srgb",
            parent=self,
        )
        dlg.exec()

    def _do_export_psd(self, filepath: str, selected: List[EclipseFrameMetadata]):
        if not HAS_PYTOSHOP:
            QMessageBox.warning(
                self,
                "Aviso",
                "El módulo 'pytoshop' no está disponible en este entorno.\n"
                "Para exportar archivos PSD multicapa, asegúrate de tener instalada la dependencia pytoshop."
            )
            return

        layout_mode = self._get_current_layout_mode()
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        crop_factor = self.slider_pad.value() / 100.0
        draw_caps = self.chk_captions.isChecked()
        workers = self._get_configured_workers()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Generando archivo PSD por capas...")
        self.log_message(f"Iniciando exportación de PSD multicapa a: {os.path.basename(filepath)}...")
        QApplication.processEvents()

        try:
            cw, ch, placements = compute_collage_layout(
                selected,
                layout_type=layout_mode,
                canvas_width=3840,
                target_disk_radius=150,
                gap_spacing=gap,
                totality_scale=tot_scale,
                crop_margin_factor=crop_factor,
                fast_preview=False,
                max_workers=workers
            )

            ok = export_layered_psd(filepath, cw, ch, placements, bg_color=self.canvas_bg_color, depth=16, include_captions=draw_caps)
            if ok:
                msg = f"Collage exportado en PSD por capas de 16-bit:\n{filepath}\n\nLienzo: {cw} × {ch} px ({len(placements)} fases + capa marcas horarias)."
                self.log_message(f"✓ PSD exportado: {os.path.basename(filepath)} ({len(placements)} capas)")
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat(f"✓ PSD exportado ({len(placements)} capas)")
                QMessageBox.information(self, "PSD Generado", msg)
            else:
                self.log_message(f"[ERROR] No se pudo generar el archivo PSD en: {filepath}")
                QMessageBox.critical(self, "Error", "No se pudo generar el archivo PSD.")
        except Exception as e:
            self.log_message(f"[ERROR] Al exportar PSD: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo generar el archivo PSD:\n{e}")
        finally:
            self.progress_bar.setVisible(False)

    def _do_export_flat(self, filepath: str, selected: List[EclipseFrameMetadata], ext: str, is_32bit: bool = False):
        layout_mode = self._get_current_layout_mode()
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        crop_factor = self.slider_pad.value() / 100.0
        draw_caps = self.chk_captions.isChecked()
        workers = self._get_configured_workers()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat(f"Renderizando imagen de collage ({ext.upper()})...")
        self.log_message(f"Renderizando imagen de collage ({ext.upper()})...")
        QApplication.processEvents()

        try:
            cw, ch, placements = compute_collage_layout(
                selected,
                layout_type=layout_mode,
                canvas_width=3840,
                target_disk_radius=150,
                gap_spacing=gap,
                totality_scale=tot_scale,
                crop_margin_factor=crop_factor,
                fast_preview=False,
                max_workers=workers
            )

            flat = render_flat_collage(cw, ch, placements, bg_color=self.canvas_bg_color, draw_captions=draw_caps)

            if ext in (".tif", ".tiff"):
                if is_32bit:
                    out_img = np.ascontiguousarray(np.clip(flat, 0.0, 1.0), dtype=np.float32)
                else:
                    out_img = (np.clip(flat, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
                save_tiff_with_icc(filepath, out_img, photometric='rgb')
            elif ext == ".png":
                u16 = (np.clip(flat, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
                bgr16 = cv2.cvtColor(u16, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            else:
                u8 = (np.clip(flat, 0.0, 1.0) * 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])

            self.log_message(f"✓ Imagen de collage guardada: {os.path.basename(filepath)} ({cw}x{ch} px)")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(f"✓ Guardado: {os.path.basename(filepath)}")
            QMessageBox.information(
                self,
                "Collage Guardado",
                f"Imagen de collage guardada correctamente:\n{filepath}\n\nDimensiones: {cw} × {ch} px"
            )
        except Exception as e:
            self.log_message(f"[ERROR] Al guardar imagen: {e}")
            QMessageBox.critical(self, "Error de Exportación", f"No se pudo guardar la imagen:\n{e}")
        finally:
            self.progress_bar.setVisible(False)

    def _export_to_developer_clicked(self):
        """Exporta el collage como imagen plana y la abre inmediatamente en el Revelador."""
        selected = self._get_selected_frames()
        if not selected:
            QMessageBox.warning(self, "Aviso", "No hay fases seleccionadas para enviar al Revelador.")
            return

        layout_mode = self._get_current_layout_mode()
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        crop_factor = self.slider_pad.value() / 100.0
        draw_caps = self.chk_captions.isChecked()
        workers = self._get_configured_workers()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.log_message("Transfiriendo imagen del collage al módulo Revelador...")
        try:
            cw, ch, placements = compute_collage_layout(
                selected,
                layout_type=layout_mode,
                canvas_width=3840,
                target_disk_radius=150,
                gap_spacing=gap,
                totality_scale=tot_scale,
                crop_margin_factor=crop_factor,
                fast_preview=False,
                max_workers=workers
            )

            flat = render_flat_collage(cw, ch, placements, bg_color=self.canvas_bg_color, draw_captions=draw_caps)

            temp_dir = get_config_val("export_dir", tempfile.gettempdir())
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "eclipse_collage_temp_developer.tif")
            save_tiff_with_icc(temp_path, flat, bit_depth=16, profile_name="sRGB")

            self.log_message("✓ Collage transferido al Revelador con éxito.")
            self.export_to_developer.emit(temp_path)
        except Exception as e:
            self.log_message(f"[ERROR] Error al transferir al Revelador: {e}")
            QMessageBox.critical(self, "Error al transferir al Revelador", f"No se pudo generar la imagen:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().closeEvent(event)
