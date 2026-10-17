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
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QCursor
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
    QColorDialog
)


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
from gui.theme import STYLE_RUN_IDLE, STYLE_TRANSFER_BTN
from core.icc_manager import save_tiff_with_icc
from gui.widgets.drop_widgets import DropTableWidget
from gui.icons import get_icon, get_svg_pixmap, ICON_COLOR_DEFAULT, ICON_COLOR_ACTIVE, ICON_COLOR_ACCENT
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow


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
        self.setStyleSheet("background: #0f1015; border: 1px solid #262833; border-radius: 4px;")

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

        if initial_filepaths:
            self.load_filepaths(initial_filepaths)

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
        # Panel Izquierdo: Scrollable (420 - 520 px de ancho unificado)
        # -----------------------------------------------------------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left_panel.setMaximumWidth(520)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(6)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        panel_content = QWidget()
        left_layout = QVBoxLayout(panel_content)
        left_layout.setContentsMargins(4, 4, 10, 4)
        left_layout.setSpacing(10)

        # --- SECCIÓN 1: Ingesta y Cribado de Calidad ---
        self.sec_files = CollapsibleSection("1. Ingesta y Cribado de Calidad", is_expanded=True, icon_name="camera")
        box_files = self.sec_files.content_layout
        box_files.setSpacing(6)

        # Modalidad Sol / Luna
        row_mode_sel = QHBoxLayout()
        row_mode_sel.addWidget(QLabel("Modalidad:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem(get_icon("sun", color="#ff4081"), "Eclipse Solar", "solar")
        self.combo_mode.addItem(get_icon("moon", color="#00e5ff"), "Eclipse Lunar", "lunar")
        self.combo_mode.setCurrentIndex(0 if self.eclipse_mode == "solar" else 1)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_combo_changed)
        row_mode_sel.addWidget(self.combo_mode, stretch=1)
        box_files.addLayout(row_mode_sel)

        # Botones de sesión .ecollage
        btn_row_sess = QHBoxLayout()
        self.btn_load_session = QPushButton("Abrir Sesión...")
        self.btn_load_session.setIcon(get_icon("folder", color="#d1c4e9"))
        self.btn_load_session.setStyleSheet("font-weight: bold; background: #2a1b4e; color: #d1c4e9; border: 1px solid #7e57c2;")
        self.btn_load_session.setToolTip("Carga una sesión guardada (.ecollage) restaurando todo el análisis y selección instantáneamente")
        self.btn_load_session.clicked.connect(self._on_load_session_clicked)

        self.btn_save_session = QPushButton("Guardar Sesión...")
        self.btn_save_session.setIcon(get_icon("save", color="#d1c4e9"))
        self.btn_save_session.setStyleSheet("font-weight: bold; background: #2a1b4e; color: #d1c4e9; border: 1px solid #7e57c2;")
        self.btn_save_session.setToolTip("Guarda la sesión actual con todos los cálculos de análisis, calidad y selección en un archivo .ecollage")
        self.btn_save_session.clicked.connect(self._on_save_session_clicked)

        btn_row_sess.addWidget(self.btn_load_session)
        btn_row_sess.addWidget(self.btn_save_session)
        box_files.addLayout(btn_row_sess)

        # Cargar fotos / limpiar
        btn_row_f = QHBoxLayout()
        self.btn_load_files = QPushButton("Cargar Fotos...")
        self.btn_load_files.setIcon(get_icon("folder", color="#80d8ff"))
        self.btn_load_files.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_load_files.clicked.connect(self._on_load_files_clicked)
        self.btn_clear_files = QPushButton("Limpiar")
        self.btn_clear_files.setIcon(get_icon("trash", color="#ff8a80"))
        self.btn_clear_files.setStyleSheet("font-weight: bold; color: #ff8a80;")
        self.btn_clear_files.clicked.connect(self._on_clear_clicked)
        btn_row_f.addWidget(self.btn_load_files)
        btn_row_f.addWidget(self.btn_clear_files)
        box_files.addLayout(btn_row_f)

        # Tabla interactiva con Drag & Drop
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
        self.table_files.setMinimumHeight(170)
        self.table_files.itemChanged.connect(self._on_table_item_changed)
        self.table_files.files_dropped.connect(self._on_files_dropped)
        box_files.addWidget(self.table_files)

        # Opciones de Rendimiento y Multihilo
        row_perf = QHBoxLayout()
        row_perf.setSpacing(6)
        
        self.combo_analysis_speed = QComboBox()
        self.combo_analysis_speed.addItem(get_icon("zap", color="#ffd54f"), "Modo Rápido (Half-size RAW 4x)", True)
        self.combo_analysis_speed.addItem(get_icon("target", color="#80d8ff"), "Modo Precisión 1:1 (Full RAW)", False)
        self.combo_analysis_speed.setToolTip(
            "Modo Rápido: Procesa RAWs con binning 2x2 acelerando 4x a 8x el análisis sin perder precisión geométrica.\n"
            "Modo Precisión: Procesa RAWs a resolución 1:1 nativa completa."
        )
        row_perf.addWidget(self.combo_analysis_speed, stretch=1)

        lbl_threads = QLabel("Hilos:")
        lbl_threads.setStyleSheet("color: #b0bec5;")
        row_perf.addWidget(lbl_threads)

        self.spin_analysis_workers = QSpinBox()
        cpu_total = os.cpu_count() or 16
        self.spin_analysis_workers.setRange(1, cpu_total)
        configured_workers = int(get_config_val("cpu_workers", 4))
        def_workers = max(configured_workers, max(2, cpu_total - 2))
        self.spin_analysis_workers.setValue(min(def_workers, cpu_total))
        self.spin_analysis_workers.setFixedWidth(68)
        self.spin_analysis_workers.setToolTip(
            f"Número de hilos de CPU concurrentes para el análisis en lote (Detectados: {cpu_total} núcleos lógicos)"
        )
        row_perf.addWidget(self.spin_analysis_workers)
        box_files.addLayout(row_perf)

        self.btn_analyze = QPushButton("Analizar Calidad y Ocultación en Lote")
        self.btn_analyze.setIcon(get_icon("search", color="#80d8ff"))
        self.btn_analyze.setFixedHeight(34)
        self.btn_analyze.setStyleSheet("font-weight: bold; background: #263238; color: #80d8ff; border: 1px solid #80d8ff;")
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        box_files.addWidget(self.btn_analyze)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        box_files.addWidget(self.progress_bar)

        left_layout.addWidget(self.sec_files)

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
        self.btn_auto_select.setIcon(get_icon("zap", color="#ffd54f"))
        self.btn_auto_select.setStyleSheet("font-weight: bold; color: #ffd54f;")
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
        self.btn_bg_custom.setStyleSheet(f"background-color: {self.canvas_bg_hex}; border: 1px solid #78909c; border-radius: 3px;")
        self.btn_bg_custom.setToolTip("Elegir color de fondo personalizado")
        self.btn_bg_custom.clicked.connect(self._choose_custom_bg_color)
        row_bg.addWidget(self.btn_bg_custom)
        box_layout.addLayout(row_bg)

        self.chk_captions = QCheckBox("Mostrar marcas horarias EXIF bajo cada fase")
        self.chk_captions.setChecked(True)
        box_layout.addWidget(self.chk_captions)

        self.btn_apply_layout = QPushButton("Actualizar Composición / Collage")
        self.btn_apply_layout.setIcon(get_icon("refresh", color="#80d8ff"))
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
        self.btn_export_psd = QPushButton("Exportar PSD (Photoshop 16-bit)")
        self.btn_export_psd.setIcon(get_icon("layers", color="#ffffff"))
        self.btn_export_psd.setFixedHeight(36)
        self.btn_export_psd.setStyleSheet("font-weight: bold; background: #0d47a1; color: #ffffff;")
        self.btn_export_psd.setToolTip("Exporta un archivo de Photoshop nativo con cada fase en su propia capa nombrada")
        self.btn_export_psd.clicked.connect(self._export_psd_clicked)
        btn_row_exp.addWidget(self.btn_export_psd)

        self.btn_export_flat = QPushButton("Guardar Imagen (TIFF / JPG)")
        self.btn_export_flat.setIcon(get_icon("image", color="#80cbc4"))
        self.btn_export_flat.setFixedHeight(36)
        self.btn_export_flat.setStyleSheet("font-weight: bold; color: #80cbc4;")
        self.btn_export_flat.clicked.connect(self._export_flat_clicked)
        btn_row_exp.addWidget(self.btn_export_flat)
        box_exp.addLayout(btn_row_exp)

        # Enlace directo al Revelador
        self.btn_export_developer = QPushButton("Enviar Collage al Revelador / Editor")
        self.btn_export_developer.setIcon(get_icon("arrow_right", color="#ffffff"))
        self.btn_export_developer.setFixedHeight(36)
        self.btn_export_developer.setStyleSheet(STYLE_TRANSFER_BTN)
        self.btn_export_developer.setToolTip("Exporta el collage como imagen plana y la abre inmediatamente en el Revelador para ajuste tonal y color")
        self.btn_export_developer.clicked.connect(self._export_to_developer_clicked)
        box_exp.addWidget(self.btn_export_developer)

        left_layout.addWidget(self.sec_exp)
        left_layout.addStretch()

        scroll_area.setWidget(panel_content)
        left_panel_layout.addWidget(scroll_area)
        splitter.addWidget(left_panel)

        # -----------------------------------------------------------------
        # Panel Derecho: Vista Previa Interactiva del Collage
        # -----------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        header_row = QHBoxLayout()
        self.lbl_info = QLabel("Vista Previa del Collage (0 fases seleccionadas)")
        self.lbl_info.setStyleSheet("font-weight: bold; color: #cfd8dc; font-size: 12px;")
        header_row.addWidget(self.lbl_info)
        header_row.addStretch()

        # Crear lienzo interactivo
        self.canvas_preview = CollageInteractiveCanvas()
        self.canvas_preview.zoom_changed.connect(self._on_zoom_changed)

        # Botones de navegación interactiva y zoom
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setIcon(get_icon("search", color="#94a3b8"))
        self.btn_zoom_out.setFixedWidth(34)
        self.btn_zoom_out.setToolTip("Alejar zoom")
        self.btn_zoom_out.clicked.connect(lambda: self.canvas_preview.zoom_by(1.0 / 1.25))

        self.lbl_zoom_val = QLabel("100%")
        self.lbl_zoom_val.setStyleSheet("color: #80d8ff; font-weight: bold; min-width: 48px;")
        self.lbl_zoom_val.setAlignment(Qt.AlignCenter)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setIcon(get_icon("zoom_in", color="#94a3b8"))
        self.btn_zoom_in.setFixedWidth(34)
        self.btn_zoom_in.setToolTip("Acercar zoom")
        self.btn_zoom_in.clicked.connect(lambda: self.canvas_preview.zoom_by(1.25))

        self.btn_zoom_100 = QPushButton("1:1")
        self.btn_zoom_100.setFixedWidth(36)
        self.btn_zoom_100.setToolTip("Ver a tamaño real 100%")
        self.btn_zoom_100.clicked.connect(self.canvas_preview.set_zoom_100)

        self.btn_zoom_fit = QPushButton("Ajustar")
        self.btn_zoom_fit.setIcon(get_icon("maximize", color="#80d8ff"))
        self.btn_zoom_fit.setToolTip("Ajustar al tamaño de la ventana (o doble clic en el lienzo)")
        self.btn_zoom_fit.clicked.connect(self.canvas_preview.fit_view)

        header_row.addWidget(self.btn_zoom_out)
        header_row.addWidget(self.lbl_zoom_val)
        header_row.addWidget(self.btn_zoom_in)
        header_row.addWidget(self.btn_zoom_100)
        header_row.addWidget(self.btn_zoom_fit)

        self.btn_refresh = QPushButton("Actualizar")
        self.btn_refresh.setIcon(get_icon("refresh", color="#80d8ff"))
        self.btn_refresh.clicked.connect(self._update_preview)
        header_row.addWidget(self.btn_refresh)
        right_layout.addLayout(header_row)

        right_layout.addWidget(self.canvas_preview, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _on_mode_combo_changed(self, idx: int):
        self.eclipse_mode = "solar" if idx == 0 else "lunar"

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
            "analysis_workers": self.spin_analysis_workers.value()
        }

        ok = save_eclipse_collage_session(filepath, self.frames_meta, settings, mode=self.eclipse_mode)
        if ok:
            QMessageBox.information(
                self,
                "Sesión Guardada",
                f"Sesión guardada exitosamente con {len(self.frames_meta)} fotogramas analizados:\n{filepath}"
            )
        else:
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
                self.spin_analysis_workers.setValue(int(settings["analysis_workers"]))

            self._populate_table()
            self._update_preview()

            num_rec = len([f for f in frames if f.is_recommended])
            QMessageBox.information(
                self,
                "Sesión Cargada",
                f"Sesión restaurada correctamente:\n{len(frames)} fotogramas cargados ({num_rec} seleccionados).\n"
                f"No es necesario reanalizar los archivos."
            )
        except Exception as e:
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
                QApplication.processEvents()

        self.frames_meta.extend(new_metas)
        self._populate_table()
        self.progress_bar.setVisible(False)
        self.lbl_info.setText(f"Carga finalizada: {len(self.frames_meta)} tomas añadidas (pulsa 'Analizar').")

    def _on_clear_clicked(self):
        self._tile_cache.clear()
        self.frames_meta.clear()
        self.table_files.setRowCount(0)
        self.canvas_preview.set_image(None)
        self.lbl_info.setText("Vista Previa del Collage (0 fases seleccionadas)")

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
            if "Diamante" in phase_text:
                phase_item.setForeground(QColor("#80d8ff"))
                font = phase_item.font()
                font.setBold(True)
                phase_item.setFont(font)
            elif "Baily" in phase_text:
                phase_item.setForeground(QColor("#ffd54f"))
                font = phase_item.font()
                font.setBold(True)
                phase_item.setFont(font)
            elif "TOTALIDAD" in phase_text:
                phase_item.setForeground(QColor("#b388ff"))
                font = phase_item.font()
                font.setBold(True)
                phase_item.setFont(font)
            elif "Entrada" in phase_text:
                phase_item.setForeground(QColor("#90caf9"))
            elif "Salida" in phase_text:
                phase_item.setForeground(QColor("#a5d6a7"))
            else:
                phase_item.setForeground(QColor("#90a4ae"))
            self.table_files.setItem(row, 4, phase_item)

            # Columna 5: Nitidez / Estado
            if f.sharpness_score > 0:
                if f.is_trepidated:
                    status_text = f"● {f.sharpness_score:.0f} (Movida)"
                    color = QColor("#ff5252")
                elif f.sharpness_score >= 80:
                    status_text = f"● {f.sharpness_score:.0f} (Nítida)"
                    color = QColor("#00e676")
                else:
                    status_text = f"● {f.sharpness_score:.0f}"
                    color = QColor("#ffd54f")
            else:
                status_text = "Pendiente"
                color = QColor("#90a4ae")

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
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_analyze.setText("Cancelar Análisis")
        self.btn_analyze.setIcon(get_icon("stop", color="#ffcdd2"))
        self.btn_analyze.setStyleSheet("font-weight: bold; background: #b71c1c; color: #ffcdd2; border: 1px solid #e57373;")
        self.btn_analyze.setEnabled(True)

        fast_mode = bool(self.combo_analysis_speed.currentData())
        workers = self.spin_analysis_workers.value()

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

    def _on_worker_finished(self, results: List[EclipseFrameMetadata]):
        self.progress_bar.setVisible(False)
        self.btn_analyze.setText("Analizar Calidad y Ocultación en Lote")
        self.btn_analyze.setIcon(get_icon("search", color="#80d8ff"))
        self.btn_analyze.setStyleSheet("font-weight: bold; background: #263238; color: #80d8ff; border: 1px solid #80d8ff;")
        self.btn_analyze.setEnabled(True)
        if results:
            self._tile_cache.clear()
            self.frames_meta = results
            self._populate_table()
            self._auto_select_recommended()

    def _on_worker_error(self, err: str):
        self.progress_bar.setVisible(False)
        self.btn_analyze.setText("Analizar Calidad y Ocultación en Lote")
        self.btn_analyze.setIcon(get_icon("search", color="#80d8ff"))
        self.btn_analyze.setStyleSheet("font-weight: bold; background: #263238; color: #80d8ff; border: 1px solid #80d8ff;")
        self.btn_analyze.setEnabled(True)
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
        self.btn_bg_custom.setStyleSheet(f"background-color: {hex_str}; border: 1px solid #78909c; border-radius: 3px;")


    def _on_zoom_changed(self, zoom: float):
        pct = int(round(zoom * 100))
        self.lbl_zoom_val.setText(f"{pct}%")

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
            workers = self.spin_analysis_workers.value() if hasattr(self, 'spin_analysis_workers') else None
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
        except Exception as e:
            self.canvas_preview.set_placeholder(f"Error generando vista previa: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    # -----------------------------------------------------------------
    # Exportaciones
    # -----------------------------------------------------------------
    def _export_psd_clicked(self):
        selected = self._get_selected_frames()
        if not selected:
            QMessageBox.warning(self, "Aviso", "No hay fases seleccionadas para exportar.")
            return

        if not HAS_PYTOSHOP:
            QMessageBox.warning(self, "Aviso", "El módulo 'pytoshop' no está disponible en este entorno.")
            return

        export_dir = get_config_val("export_dir", "exportaciones")
        os.makedirs(export_dir, exist_ok=True)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Collage de Eclipse como PSD por Capas",
            os.path.join(export_dir, "eclipse_collage_capas.psd"),
            "Photoshop PSD (*.psd)"
        )
        if not filepath:
            return

        layout_mode = self._get_current_layout_mode()
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        crop_factor = self.slider_pad.value() / 100.0
        draw_caps = self.chk_captions.isChecked()
        workers = self.spin_analysis_workers.value() if hasattr(self, 'spin_analysis_workers') else None

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
            QMessageBox.information(
                self, 
                "Exportación Exitosa", 
                f"Collage exportado en PSD por capas de 16-bit:\n{filepath}\n\n"
                f"Lienzo: {cw} × {ch} px ({len(placements)} fases + capa marcas horarias)."
            )
        else:
            QMessageBox.critical(self, "Error", "No se pudo generar el archivo PSD.")

    def _export_flat_clicked(self):
        selected = self._get_selected_frames()
        if not selected:
            QMessageBox.warning(self, "Aviso", "No hay fases seleccionadas para exportar.")
            return

        export_dir = get_config_val("export_dir", "exportaciones")
        os.makedirs(export_dir, exist_ok=True)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Imagen del Collage",
            os.path.join(export_dir, "eclipse_collage_final.tif"),
            "TIFF 16-bit (*.tif *.tiff);;JPEG Alta Calidad (*.jpg *.jpeg)"
        )
        if not filepath:
            return

        layout_mode = self._get_current_layout_mode()
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        crop_factor = self.slider_pad.value() / 100.0
        draw_caps = self.chk_captions.isChecked()
        workers = self.spin_analysis_workers.value() if hasattr(self, 'spin_analysis_workers') else None

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

        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".tif", ".tiff"):
            save_tiff_with_icc(filepath, flat, bit_depth=16, profile_name="sRGB")
        else:
            u8 = (np.clip(flat, 0.0, 1.0) * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])

        QMessageBox.information(
            self,
            "Guardado Exitoso",
            f"Imagen de collage guardada correctamente:\n{filepath}\n\nDimensiones: {cw} × {ch} px"
        )

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
        workers = self.spin_analysis_workers.value() if hasattr(self, 'spin_analysis_workers') else None

        QApplication.setOverrideCursor(Qt.WaitCursor)
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

            self.export_to_developer.emit(temp_path)
        except Exception as e:
            QMessageBox.critical(self, "Error al transferir al Revelador", f"No se pudo generar la imagen:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
        super().closeEvent(event)
