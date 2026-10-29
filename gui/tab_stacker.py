# gui/tab_stacker.py
import os
import cv2
import numpy as np
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTabWidget, QListWidget, QListWidgetItem, QProgressBar,
    QMessageBox, QGroupBox, QRadioButton, QSlider, QDoubleSpinBox,
    QComboBox, QSplitter, QTextEdit, QSizePolicy, QCheckBox, QSpinBox,
    QScrollArea, QProgressDialog, QFrame
)
from PySide6.QtCore import Qt, Signal, QCoreApplication
from PySide6.QtGui import QTextCursor, QCursor

from core.project_manager import ProjectManager
from core.masking import refine_mask_guided, apply_direct_brush_to_mask
from core.stacking import load_image_as_float32
from gui.canvas import MaskCanvas
from gui.widgets.drop_widgets import DropListWidget
from gui.worker import StackingWorker, SubstackingWorker, QualityCullingWorker, AIHorizonWorker, ModelDownloadWorker
from core.gpu_backend import is_gpu_enabled
from core.config_manager import load_config
from gui.theme import STYLE_RUN_IDLE, STYLE_RUN_ACTIVE, STYLE_RUN_CANCELLING
from gui.icons import get_icon, ICON_COLOR_DEFAULT, ICON_COLOR_ACTIVE, ICON_COLOR_MUTED
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow


class FileListRow(QWidget):
    """Widget de fila con casilla de inclusión, etiqueta de referencia, badge de calidad y botón de borrado."""
    delete_requested = Signal(str, str)         # (tab_name, filepath)
    include_toggled = Signal(str, str, bool)    # (tab_name, filepath, is_included)

    def __init__(self, filepath: str, tab_name: str, is_ref: bool = False, is_included: bool = True, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.tab_name = tab_name
        self._is_included = is_included

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Casilla de inclusión (para Lights)
        self.chk_include = QCheckBox()
        self.chk_include.setChecked(is_included)
        self.chk_include.setToolTip("Incluir / descartar esta toma del apilado")
        self.chk_include.toggled.connect(self._on_include_toggled)
        if tab_name != "Lights":
            self.chk_include.setVisible(False)
        layout.addWidget(self.chk_include)

        # Badge de calidad (FWHM / Redondez / Cribado)
        self.lbl_quality = QLabel("")
        self.lbl_quality.setVisible(False)
        layout.addWidget(self.lbl_quality)

        # Indicador de referencia / activo tipo badge
        self.lbl_ref = QLabel()
        self.set_as_reference(is_ref)
        layout.addWidget(self.lbl_ref)

        # Nombre del archivo
        self.lbl_name = QLabel(os.path.basename(filepath))
        self.lbl_name.setToolTip(filepath)
        self.lbl_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._update_name_style()
        layout.addWidget(self.lbl_name)

        # Botón de eliminar individual (Cruz roja minimalista)
        self.btn_del = QPushButton("×")
        self.btn_del.setFixedSize(18, 18)
        self.btn_del.setToolTip("Eliminar esta toma de la lista")
        self.btn_del.setStyleSheet(
            "QPushButton { color: #f87171; background: transparent; border: 1px solid #4a2028; border-radius: 3px; font-weight: bold; font-size: 13px; line-height: 14px; padding: 0px; } "
            "QPushButton:hover { background: #b91c1c; color: #ffffff; border-color: #ef4444; }"
        )
        self.btn_del.clicked.connect(lambda: self.delete_requested.emit(self.tab_name, self.filepath))
        layout.addWidget(self.btn_del)

    def set_as_reference(self, is_ref: bool):
        if is_ref:
            self.lbl_ref.setText("REF")
            self.lbl_ref.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 9px; background-color: #0c2938; border: 1px solid #00bcd4; border-radius: 3px; padding: 1px 4px;")
            self.lbl_ref.setVisible(True)
        else:
            self.lbl_ref.setText("")
            self.lbl_ref.setVisible(False)

    def set_quality_result(self, res):
        """Asigna y muestra el badge de calidad óptica del frame culling."""
        if res is None:
            self.lbl_quality.setVisible(False)
            return
        self.lbl_quality.setVisible(True)

        tip = (
            f"Puntuación: {res.score:.1f}%\n"
            f"Nitidez (FWHM): {res.fwhm:.1f} px\n"
            f"Redondez centro (eje óptico): {res.center_ecc:.2f}\n"
            f"Coma esquinas: {res.corner_ecc:.2f}\n"
            f"Estrellas detectadas: {res.star_count}"
        )
        if res.defect_reason:
            tip += f"\nMotivo: {res.defect_reason}"
        self.lbl_quality.setToolTip(tip)

        score_int = int(round(res.score))
        if res.status == "EXCELLENT":
            self.lbl_quality.setText(f"● {score_int}%")
            self.lbl_quality.setStyleSheet("color: #4ade80; background-color: #132b1e; border: 1px solid #22c55e; border-radius: 3px; padding: 1px 5px; font-size: 10px; font-weight: bold;")
        elif res.status == "ACCEPTABLE":
            self.lbl_quality.setText(f"● {score_int}%")
            self.lbl_quality.setStyleSheet("color: #facc15; background-color: #2e2612; border: 1px solid #eab308; border-radius: 3px; padding: 1px 5px; font-size: 10px; font-weight: bold;")
        else:
            defect_type = "Movida" if "Movida" in res.status_label else ("Nubes" if "Nubes" in res.status_label else ("Desenfoque" if "Desenfoque" in res.status_label else ""))
            suffix = f" {defect_type}" if defect_type else ""
            self.lbl_quality.setText(f"● {score_int}%{suffix}")
            self.lbl_quality.setStyleSheet("color: #f87171; background-color: #2e141a; border: 1px solid #ef4444; border-radius: 3px; padding: 1px 5px; font-size: 10px; font-weight: bold;")

    def set_included(self, is_included: bool):
        self.chk_include.blockSignals(True)
        self.chk_include.setChecked(is_included)
        self.chk_include.blockSignals(False)
        self._is_included = is_included
        self._update_name_style()

    def _on_include_toggled(self, checked: bool):
        self._is_included = checked
        self._update_name_style()
        self.include_toggled.emit(self.tab_name, self.filepath, checked)

    def _update_name_style(self):
        if self._is_included:
            self.lbl_name.setStyleSheet("color: #eceff4; font-size: 11px;")
        else:
            self.lbl_name.setStyleSheet("color: #546e7a; font-size: 11px; text-decoration: line-through;")


class StackerTab(QWidget):
    stacking_finished = Signal(str, object)
    session_title_changed = Signal(str)
    new_session_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_mgr = ProjectManager()
        self.lights_list = []
        self.darks_list = []
        self.flats_list = []
        self.bias_list = []
        self.ground_list = []
        self.excluded_lights = set()       # Conjunto de rutas de Lights desmarcadas / excluidas
        self.culling_results = {}          # Diccionario filepath -> FrameQualityResult
        self.culling_worker = None         # Hilo secundario para análisis en lote
        self.computed_mask = None
        self.last_stacked_output_path = None
        self.current_ref_path = None
        self.worker = None

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 1. Lienzo Interactivo con soporte de arrastrar y soltar
        self.canvas = MaskCanvas(self, enable_masking=True)
        self.canvas.brush_size_changed.connect(self.on_canvas_brush_changed)
        self.canvas.files_dropped.connect(lambda paths: self.add_files_to_tab("Lights", paths))

        # 2. Panel lateral (Scrollable con Log y Ejecución Fijos Abajo)
        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left_panel.setMaximumWidth(520)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(4, 4, 4, 4)
        left_panel_layout.setSpacing(6)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        panel_content = QWidget()
        left_layout = QVBoxLayout(panel_content)
        left_layout.setContentsMargins(4, 4, 10, 4)
        left_layout.setSpacing(12)

        # SECCIÓN 1: Gestión de Sesión / Proyecto
        self.grp_project = CollapsibleSection("1. Sesión y Proyecto (.mwstack)", icon_name="save")
        proj_layout = self.grp_project.content_layout
        proj_layout.setContentsMargins(10, 8, 10, 10)
        proj_layout.setSpacing(8)
        
        self.lbl_session = QLabel("Sesión: Sin guardar")
        self.lbl_session.setStyleSheet("color: #80cbc4; font-size: 11px; font-weight: bold;")
        proj_layout.addWidget(self.lbl_session)

        btn_proj_row = QHBoxLayout()
        btn_proj_row.setSpacing(6)
        
        self.btn_new_sess = QPushButton("Nueva")
        self.btn_new_sess.setIcon(get_icon("plus", size=13, color="#ffab91"))
        self.btn_new_sess.setStyleSheet("font-weight: bold; color: #ffab91;")
        self.btn_new_sess.setToolTip("Reiniciar proyecto completo, vaciar tomas, máscaras y revelador")
        self.btn_new_sess.clicked.connect(self.on_new_session_clicked)
        btn_proj_row.addWidget(self.btn_new_sess)

        self.btn_open_sess = QPushButton("Cargar")
        self.btn_open_sess.setIcon(get_icon("folder", size=13, color="#80d8ff"))
        self.btn_open_sess.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_open_sess.clicked.connect(self.open_project)

        self.btn_save_sess = QPushButton("Guardar")
        self.btn_save_sess.setIcon(get_icon("save", size=13, color="#a5d6a7"))
        self.btn_save_sess.setStyleSheet("font-weight: bold; color: #a5d6a7;")
        self.btn_save_sess.clicked.connect(self.save_project)

        self.btn_save_as_sess = QPushButton("Guardar Como...")
        self.btn_save_as_sess.setIcon(get_icon("copy", size=13, color="#cfd8dc"))
        self.btn_save_as_sess.setStyleSheet("font-weight: bold; color: #cfd8dc;")
        self.btn_save_as_sess.clicked.connect(self.save_project_as)

        btn_proj_row.addWidget(self.btn_open_sess)
        btn_proj_row.addWidget(self.btn_save_sess)
        btn_proj_row.addWidget(self.btn_save_as_sess)
        proj_layout.addLayout(btn_proj_row)
        left_layout.addWidget(self.grp_project)

        # SECCIÓN 2: Pestañas de Archivos (Lights / Darks / Flats / Suelo)
        self.grp_files = CollapsibleSection("2. Calibración y Tomas", icon_name="camera")
        files_layout = self.grp_files.content_layout
        files_layout.setContentsMargins(10, 8, 10, 10)
        files_layout.setSpacing(8)

        self.tabs_files = QTabWidget()
        self.tabs_files.setMinimumHeight(160)
        self.tabs_files.setDocumentMode(True)
        self.tabs_files.setUsesScrollButtons(False)
        self.tabs_files.tabBar().setExpanding(True)
        self.tabs_files.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1e2230;
                background-color: #0b0d13;
                border-radius: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #10121a;
                color: #94a3b8;
                padding: 6px 4px;
                border: 1px solid #1e2230;
                border-bottom: 2px solid transparent;
                font-weight: 600;
                font-size: 11px;
                margin-right: 1px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #161b26;
                color: #38bdf8;
                border-color: #2b354d;
                border-bottom: 2px solid #38bdf8;
            }
            QTabBar::tab:hover:!selected {
                background-color: #141722;
                color: #e2e8f0;
                border-bottom: 2px solid #334155;
            }
        """)
        
        self.list_lights = DropListWidget()
        self.list_lights.itemDoubleClicked.connect(self.on_light_double_clicked)
        self.list_lights.files_dropped.connect(lambda paths: self.add_files_to_tab("Lights", paths))
        self.tabs_files.addTab(self.list_lights, get_icon("star", size=14, color="#ffd54f"), "Lights")

        self.list_darks = DropListWidget()
        self.list_darks.files_dropped.connect(lambda paths: self.add_files_to_tab("Darks", paths))
        self.tabs_files.addTab(self.list_darks, get_icon("moon", size=14, color="#94a3b8"), "Darks")

        self.list_flats = DropListWidget()
        self.list_flats.files_dropped.connect(lambda paths: self.add_files_to_tab("Flats", paths))
        self.tabs_files.addTab(self.list_flats, get_icon("sun", size=14, color="#80d8ff"), "Flats")

        self.list_bias = DropListWidget()
        self.list_bias.files_dropped.connect(lambda paths: self.add_files_to_tab("Bias", paths))
        self.tabs_files.addTab(self.list_bias, get_icon("zap", size=14, color="#a78bfa"), "Bias")

        self.list_ground = DropListWidget()
        self.list_ground.itemDoubleClicked.connect(self.on_ground_double_clicked)
        self.list_ground.files_dropped.connect(lambda paths: self.add_files_to_tab("Suelo", paths))
        self.tabs_files.addTab(self.list_ground, get_icon("mountain", size=14, color="#4ade80"), "Suelo")
        
        files_layout.addWidget(self.tabs_files)

        # Barra de Control de Calidad y Cribado (Frame Culling) para Lights
        self.culling_bar = QFrame()
        self.culling_bar.setObjectName("cullingBar")
        self.culling_bar.setStyleSheet("""
            QFrame#cullingBar {
                background-color: #10141f;
                border: 1px solid #1e2638;
                border-radius: 6px;
            }
        """)
        culling_vbox = QVBoxLayout(self.culling_bar)
        culling_vbox.setContentsMargins(6, 6, 6, 6)
        culling_vbox.setSpacing(6)

        # Fila 1: Botones de Acción de Cribado
        row_cull_btns = QHBoxLayout()
        row_cull_btns.setContentsMargins(0, 0, 0, 0)
        row_cull_btns.setSpacing(6)

        self.btn_cull_analyze = QPushButton("Analizar Calidad")
        self.btn_cull_analyze.setIcon(get_icon("search", size=13, color="#38bdf8"))
        self.btn_cull_analyze.setCursor(Qt.PointingHandCursor)
        self.btn_cull_analyze.setStyleSheet("""
            QPushButton {
                background-color: #0c2338;
                color: #38bdf8;
                font-weight: 600;
                border: 1px solid #0284c7;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #0284c7; color: #ffffff; }
            QPushButton:disabled { background-color: #111827; color: #475569; border-color: #1f2937; }
        """)
        self.btn_cull_analyze.setToolTip("Analiza FWHM, redondez de estrellas (viento/deriva) y presencia de nubes en todo el lote")
        self.btn_cull_analyze.clicked.connect(self.start_culling_analysis)
        row_cull_btns.addWidget(self.btn_cull_analyze)

        self.btn_cull_discard = QPushButton("Auto-Descartar")
        self.btn_cull_discard.setIcon(get_icon("zap", size=13, color="#f472b6"))
        self.btn_cull_discard.setCursor(Qt.PointingHandCursor)
        self.btn_cull_discard.setStyleSheet("""
            QPushButton {
                background-color: #2a1520;
                color: #f472b6;
                font-weight: 600;
                border: 1px solid #be185d;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #be185d; color: #ffffff; }
            QPushButton:disabled { background-color: #111827; color: #475569; border-color: #1f2937; }
        """)
        self.btn_cull_discard.setToolTip("Desmarca automáticamente las tomas clasificadas como defectuosas o con estrellas movidas")
        self.btn_cull_discard.clicked.connect(self.auto_discard_defective_frames)
        row_cull_btns.addWidget(self.btn_cull_discard)

        self.btn_cull_best_ref = QPushButton("Fijar Mejor Ref.")
        self.btn_cull_best_ref.setIcon(get_icon("target", size=13, color="#a5d6a7"))
        self.btn_cull_best_ref.setCursor(Qt.PointingHandCursor)
        self.btn_cull_best_ref.setStyleSheet("""
            QPushButton {
                background-color: #15271d;
                color: #a5d6a7;
                font-weight: 600;
                border: 1px solid #2e7d32;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #2e7d32; color: #ffffff; }
            QPushButton:disabled { background-color: #111827; color: #475569; border-color: #1f2937; }
        """)
        self.btn_cull_best_ref.setToolTip("Selecciona y fija automáticamente como base de alineación la toma con mejor FWHM y mayor calidad estelar")
        self.btn_cull_best_ref.clicked.connect(self.auto_select_best_reference)
        row_cull_btns.addWidget(self.btn_cull_best_ref)

        culling_vbox.addLayout(row_cull_btns)

        # Fila 2: Criterio óptico y balance de tomas aptas
        row_cull_opts = QHBoxLayout()
        row_cull_opts.setContentsMargins(0, 0, 0, 0)
        row_cull_opts.setSpacing(6)

        lbl_cull_profile = QLabel("Perfil Óptico:")
        lbl_cull_profile.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        lbl_cull_profile.setToolTip(
            "Criterio óptico para evaluar la redondez y calidad estelar:\n"
            "• Gran Angular: Ignora la coma/aberración natural de las esquinas y mide trepidación en el centro.\n"
            "• Equilibrado: Recomendado para focales medias (35mm-85mm).\n"
            "• Exigente: Para telescopios y monturas ecuatoriales (estricto en todo el campo)."
        )
        row_cull_opts.addWidget(lbl_cull_profile)

        self.combo_cull_profile = QComboBox()
        self.combo_cull_profile.addItem("Gran Angular (Tolerante)", "wide_angle")
        self.combo_cull_profile.addItem("Equilibrado (Estándar)", "balanced")
        self.combo_cull_profile.addItem("Exigente (Tracker/Telescopio)", "strict")
        self.combo_cull_profile.setToolTip(lbl_cull_profile.toolTip())
        self.combo_cull_profile.setStyleSheet("""
            QComboBox {
                background-color: #161b26;
                color: #e2e8f0;
                border: 1px solid #28324a;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QComboBox:hover {
                border-color: #38bdf8;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
            QComboBox QAbstractItemView {
                background-color: #161b26;
                color: #e2e8f0;
                selection-background-color: #0284c7;
                selection-color: #ffffff;
                border: 1px solid #28324a;
            }
        """)
        row_cull_opts.addWidget(self.combo_cull_profile)
        row_cull_opts.addStretch(1)

        self.lbl_cull_summary = QLabel("")
        self.lbl_cull_summary.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: bold;")
        self.lbl_cull_summary.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_cull_opts.addWidget(self.lbl_cull_summary)

        culling_vbox.addLayout(row_cull_opts)

        files_layout.addWidget(self.culling_bar)
        self.tabs_files.currentChanged.connect(self._on_files_tab_changed)

        btn_box = QHBoxLayout()
        btn_box.setSpacing(6)
        self.btn_add_files = QPushButton("Añadir Tomas...")
        self.btn_add_files.setIcon(get_icon("plus", size=13, color="#80d8ff"))
        self.btn_add_files.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_add_files.clicked.connect(self.add_current_tab_files)
        self.btn_clear_files = QPushButton("Limpiar Pestaña")
        self.btn_clear_files.setIcon(get_icon("trash", size=13, color="#ff8a80"))
        self.btn_clear_files.setStyleSheet("font-weight: bold; color: #ff8a80;")
        self.btn_clear_files.clicked.connect(self.clear_current_tab_files)
        btn_box.addWidget(self.btn_add_files)
        btn_box.addWidget(self.btn_clear_files)
        files_layout.addLayout(btn_box)

        left_layout.addWidget(self.grp_files)

        # SECCIÓN 3: Parámetros de Integración / Salida
        self.grp_settings = CollapsibleSection("3. Parámetros de Integración / Salida", icon_name="sliders")
        set_layout = self.grp_settings.content_layout
        set_layout.setContentsMargins(10, 8, 10, 10)
        set_layout.setSpacing(8)

        # Selector de Tipo de Salida: Imagen Fija vs Vídeo Time-Lapse
        row_out_type = QHBoxLayout()
        row_out_type.setSpacing(10)
        self.rb_out_single = QRadioButton("Imagen Fija")
        self.rb_out_single.setChecked(True)
        self.rb_out_single.setStyleSheet("color: #80d8ff; font-weight: bold;")
        self.rb_out_video = QRadioButton("Vídeo Time-Lapse (Sub-stacking)")
        self.rb_out_video.setStyleSheet("color: #b388ff; font-weight: bold;")
        self.rb_out_single.toggled.connect(self._on_stack_type_toggled)
        row_out_type.addWidget(self.rb_out_single)
        row_out_type.addWidget(self.rb_out_video)
        set_layout.addLayout(row_out_type)

        # Modo de Captura (Imagen Fija)
        self.row_mode_widget = QWidget()
        row_mode = QHBoxLayout(self.row_mode_widget)
        row_mode.setContentsMargins(0, 0, 0, 0)
        lbl_mode = QLabel("Modo de Captura:")
        lbl_mode.setMinimumWidth(140)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Trípode Fijo (Suelo Estático)", "Star Tracker (Seguimiento)"])
        self.combo_mode.currentIndexChanged.connect(self._on_capture_mode_changed)
        row_mode.addWidget(lbl_mode)
        row_mode.addWidget(self.combo_mode)
        set_layout.addWidget(self.row_mode_widget)

        # Tratamiento del Suelo (Imagen Fija)
        self.row_ground_widget = QWidget()
        row_ground = QHBoxLayout(self.row_ground_widget)
        row_ground.setContentsMargins(0, 0, 0, 0)
        lbl_ground = QLabel("Tratamiento del Suelo:")
        lbl_ground.setMinimumWidth(140)
        self.combo_ground = QComboBox()
        self.combo_ground.addItems([
            "Apilar Suelo Completo (Dual)",
            "Suelo de Referencia (Sin apilar)",
            "Usar Toma de Pestaña Suelo"
        ])
        row_ground.addWidget(lbl_ground)
        row_ground.addWidget(self.combo_ground)
        set_layout.addWidget(self.row_ground_widget)

        # Contenedor de Configuración para Vídeo Time-Lapse (Sub-stacking)
        self.box_substack_settings = QWidget()
        sub_layout = QVBoxLayout(self.box_substack_settings)
        sub_layout.setContentsMargins(0, 4, 0, 4)
        sub_layout.setSpacing(6)

        # 1. Ventana Deslizante (W) con PrecisionSliderRow
        self.slider_sub_w = PrecisionSliderRow(
            label="Ventana Deslizante (W):",
            min_val=3,
            max_val=15,
            default_val=5,
            step=2,
            unit=" tomas"
        )
        self.lbl_sub_w_val = QLabel("5 tomas (+2.2x SNR, +1.2 EV)")
        self.lbl_sub_w_val.setStyleSheet("color: #ffd54f; font-weight: bold; font-size: 11px; margin-left: 4px;")
        self.slider_sub_w.valueChanged.connect(self._on_sub_w_slider_changed)
        sub_layout.addWidget(self.slider_sub_w)
        sub_layout.addWidget(self.lbl_sub_w_val)

        # 2. Paso Temporal (Stride) & Ponderación
        row_sp = QHBoxLayout()
        row_sp.setSpacing(6)
        lbl_stride = QLabel("Paso:")
        lbl_stride.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.combo_sub_stride = QComboBox()
        self.combo_sub_stride.addItem("1 toma (Vídeo fluido 24/30 fps)", 1)
        self.combo_sub_stride.addItem("W tomas (Bloques acelerados)", 0)
        self.combo_sub_stride.setToolTip("1 toma genera un cuadro por cada foto disparada; W tomas divide la secuencia en bloques no solapados.")
        row_sp.addWidget(lbl_stride)
        row_sp.addWidget(self.combo_sub_stride, stretch=1)
        sub_layout.addLayout(row_sp)

        row_wmode = QHBoxLayout()
        lbl_wmode = QLabel("Ponderación:")
        lbl_wmode.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.combo_sub_weight = QComboBox()
        self.combo_sub_weight.addItem("Gaussiana (Máxima Nitidez Temporal)", "gaussian")
        self.combo_sub_weight.addItem("Uniforme (Media Aritmética)", "uniform")
        self.combo_sub_weight.addItem("Triangular (Decaimiento Lineal)", "triangular")
        self.combo_sub_weight.setToolTip("Gaussiana pondera con mayor fuerza el fotograma central para preservar estrellas fugaces y nitidez fina.")
        row_wmode.addWidget(lbl_wmode)
        row_wmode.addWidget(self.combo_sub_weight, stretch=1)
        sub_layout.addLayout(row_wmode)

        # 3. Tratamiento de Suelo en Vídeo
        row_s_gnd = QHBoxLayout()
        lbl_s_gnd = QLabel("Suelo en Vídeo:")
        lbl_s_gnd.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.combo_sub_ground = QComboBox()
        self.combo_sub_ground.addItem("Suelo Limpio Apilado (Trípode fijo)", "temporal_average")
        self.combo_sub_ground.addItem("Suelo Toma Central (Slider / Movimiento)", "static_anchor")
        self.combo_sub_ground.setToolTip("En trípode fijo, el promedio temporal del suelo reduce radicalmente el ruido en las sombras del paisaje sin mover el horizonte.")
        row_s_gnd.addWidget(lbl_s_gnd)
        row_s_gnd.addWidget(self.combo_sub_ground, stretch=1)
        sub_layout.addLayout(row_s_gnd)

        # 4. Formato de Vídeo (Resolución & FPS)
        row_vid = QHBoxLayout()
        row_vid.setSpacing(6)
        lbl_res = QLabel("Resolución:")
        lbl_res.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.combo_sub_res = QComboBox()
        self.combo_sub_res.addItem("1080p Full HD", "1080p")
        self.combo_sub_res.addItem("4K UHD", "4k")
        self.combo_sub_res.addItem("Original", "original")

        lbl_fps = QLabel("FPS:")
        lbl_fps.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.combo_sub_fps = QComboBox()
        self.combo_sub_fps.addItems(["24", "25", "30", "60"])
        self.combo_sub_fps.setCurrentText("24")

        row_vid.addWidget(lbl_res)
        row_vid.addWidget(self.combo_sub_res)
        row_vid.addWidget(lbl_fps)
        row_vid.addWidget(self.combo_sub_fps)
        sub_layout.addLayout(row_vid)

        # 5. Casilla de exportación de cuadros
        self.chk_sub_save_frames = QCheckBox("Guardar también secuencia de imágenes (JPEG)")
        self.chk_sub_save_frames.setStyleSheet("color: #b0bec5; font-size: 11px;")
        self.chk_sub_save_frames.setChecked(False)
        sub_layout.addWidget(self.chk_sub_save_frames)

        self.box_substack_settings.setVisible(False)
        set_layout.addWidget(self.box_substack_settings)

        # Sub-bloque Anti-Ghosting de Árboles en Horizonte (Min-Horizon) - Compartido
        self.box_anti_ghosting = QWidget()
        ag_layout = QVBoxLayout(self.box_anti_ghosting)
        ag_layout.setContentsMargins(0, 4, 0, 4)
        ag_layout.setSpacing(4)

        self.chk_anti_ghosting = QCheckBox("Anti-Ghosting de Árboles (Min-Horizon)")
        self.chk_anti_ghosting.setChecked(True)
        self.chk_anti_ghosting.setToolTip(
            "Erradica ramas fantasma y sangrado estelar en vegetación y horizontes complejos\n"
            "usando proyección Min-Horizon y transición continua por transformada de distancias."
        )
        ag_layout.addWidget(self.chk_anti_ghosting)

        row_ag_params = QHBoxLayout()
        row_ag_params.setContentsMargins(18, 0, 0, 0)
        row_ag_params.setSpacing(8)

        lbl_ag_rad = QLabel("Radio Frontera:")
        lbl_ag_rad.setStyleSheet("font-size: 11px; color: #90a4ae;")
        self.spin_ag_radius = QSpinBox()
        self.spin_ag_radius.setRange(5, 60)
        self.spin_ag_radius.setValue(25)
        self.spin_ag_radius.setSuffix(" px")
        self.spin_ag_radius.setToolTip("Ancho de la banda perimétrica de seguridad alrededor del contorno del suelo")

        lbl_ag_sens = QLabel("Follaje:")
        lbl_ag_sens.setStyleSheet("font-size: 11px; color: #90a4ae;")
        self.combo_ag_sens = QComboBox()
        self.combo_ag_sens.addItem("Equilibrada (Recomendada)", 0.5)
        self.combo_ag_sens.addItem("Fino / Con Viento (Alta opacidad)", 0.8)
        self.combo_ag_sens.addItem("Bordes Gruesos / Roca", 0.3)
        self.combo_ag_sens.setToolTip("Sensibilidad de opacidad para ramas finas y hojas")

        row_ag_params.addWidget(lbl_ag_rad)
        row_ag_params.addWidget(self.spin_ag_radius)
        row_ag_params.addWidget(lbl_ag_sens)
        row_ag_params.addWidget(self.combo_ag_sens)
        row_ag_params.addStretch()
        ag_layout.addLayout(row_ag_params)

        self.chk_anti_ghosting.toggled.connect(
            lambda checked: self.spin_ag_radius.setEnabled(checked) and self.combo_ag_sens.setEnabled(checked)
        )
        set_layout.addWidget(self.box_anti_ghosting)

        # Factor Kappa (Imagen Fija)
        self.row_kappa_widget = QWidget()
        row_kappa = QHBoxLayout(self.row_kappa_widget)
        row_kappa.setContentsMargins(0, 0, 0, 0)
        row_kappa.addWidget(QLabel("Factor Kappa (MAD Rejection):"))
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(0.5, 5.0)
        self.spin_kappa.setValue(2.2)
        self.spin_kappa.setSingleStep(0.1)
        self.spin_kappa.setFixedWidth(75)
        row_kappa.addWidget(self.spin_kappa)
        set_layout.addWidget(self.row_kappa_widget)

        left_layout.addWidget(self.grp_settings)

        # SECCIÓN 4: Antipolución en Apilado
        self.grp_stack_lp = CollapsibleSection("4. Reducción de Polución Lumínica", icon_name="sparkles")
        layout_stack_lp = self.grp_stack_lp.content_layout
        layout_stack_lp.setContentsMargins(10, 8, 10, 10)
        layout_stack_lp.setSpacing(8)

        row_lp_algo = QHBoxLayout()
        row_lp_algo.addWidget(QLabel("Algoritmo:"))
        self.combo_lp_algo = QComboBox()
        self.combo_lp_algo.addItems([
            "Estándar (Sin filtro)",
            "Sustracción de Domo (Sequator)",
            "Rechazo Asimétrico (Min-Sigma)",
            "Normalización Local"
        ])
        row_lp_algo.addWidget(self.combo_lp_algo)
        layout_stack_lp.addLayout(row_lp_algo)

        # Precision Slider para fuerza antipolución
        self.slider_stack_lp = PrecisionSliderRow(
            label="Fuerza Antipolución:",
            min_val=0,
            max_val=100,
            default_val=50,
            step=1,
            unit="%"
        )
        self.slider_stack_lp.valueChanged.connect(self._on_stack_lp_changed)
        layout_stack_lp.addWidget(self.slider_stack_lp)
        self.lbl_lp_str = QLabel("50%")
        self.lbl_lp_str.setVisible(False)

        left_layout.addWidget(self.grp_stack_lp)

        # SECCIÓN 5: Máscara Cielo / Suelo
        self.grp_mask = CollapsibleSection("5. Segmentación Cielo / Suelo (Máscara)", icon_name="layers")
        mask_layout = self.grp_mask.content_layout
        mask_layout.setContentsMargins(10, 8, 10, 10)
        mask_layout.setSpacing(8)

        row_mask_toggles = QHBoxLayout()
        self.chk_show_mask = QCheckBox("Mostrar Máscara")
        self.chk_show_mask.setChecked(True)
        self.chk_show_mask.toggled.connect(self.canvas.set_mask_visible)
        row_mask_toggles.addWidget(self.chk_show_mask)

        self.chk_show_strokes = QCheckBox("Mostrar Trazos")
        self.chk_show_strokes.setChecked(True)
        self.chk_show_strokes.toggled.connect(self.canvas.set_scribbles_visible)
        row_mask_toggles.addWidget(self.chk_show_strokes)
        mask_layout.addLayout(row_mask_toggles)

        self.rb_sky = QRadioButton("Pintar Cielo (Verde)")
        self.rb_sky.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        self.rb_ground = QRadioButton("Pintar Suelo (Rojo)")
        self.rb_ground.setStyleSheet("color: #ef9a9a; font-weight: bold;")
        self.rb_sky.setChecked(True)
        self.rb_sky.toggled.connect(self.update_brush_mode)
        mask_layout.addWidget(self.rb_sky)
        mask_layout.addWidget(self.rb_ground)

        # Precision Slider para tamaño de pincel
        self.slider_brush = PrecisionSliderRow(
            label="Radio de Cursor:",
            min_val=5,
            max_val=120,
            default_val=20,
            step=1,
            unit=" px"
        )
        self.slider_brush.valueChanged.connect(self.update_brush_slider)
        mask_layout.addWidget(self.slider_brush)
        self.lbl_brush = QLabel("40 px")
        self.lbl_brush.setVisible(False)

        # Controles de Refinamiento
        row_params = QHBoxLayout()
        row_params.addWidget(QLabel("Suavizado Borde:"))
        self.spin_feather = QSpinBox()
        self.spin_feather.setRange(1, 31)
        self.spin_feather.setSingleStep(2)
        self.spin_feather.setValue(7)
        self.spin_feather.setFixedWidth(55)
        row_params.addWidget(self.spin_feather)

        row_params.addWidget(QLabel("Pasos:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 8)
        self.spin_iter.setValue(3)
        self.spin_iter.setFixedWidth(50)
        row_params.addWidget(self.spin_iter)
        mask_layout.addLayout(row_params)

        # Botón principal 1-Click IA
        self.btn_auto_ai_mask = QPushButton("Auto-Detectar Suelo (IA 1-Click)")
        self.btn_auto_ai_mask.setIcon(get_icon("sparkles", size=15, color="#ffffff"))
        self.btn_auto_ai_mask.setFixedHeight(34)
        self.btn_auto_ai_mask.setCursor(Qt.PointingHandCursor)
        self.btn_auto_ai_mask.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4a148c, stop:1 #7b1fa2); "
            "color: #ffffff; font-weight: bold; font-size: 11px; border: 1px solid #ba68c8; border-radius: 4px; padding: 4px 8px; } "
            "QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a1b9a, stop:1 #8e24aa); border-color: #e1bee7; } "
            "QPushButton:disabled { background: #2a2035; color: #777777; border-color: #443355; }"
        )
        self.btn_auto_ai_mask.setToolTip("Segmentación semántica profunda instantánea: detecta y separa el suelo del cielo sin necesidad de dibujar trazos manuales.")
        self.btn_auto_ai_mask.clicked.connect(self.start_auto_ai_mask)
        mask_layout.addWidget(self.btn_auto_ai_mask)

        # Acciones de trazos y refinado manual
        btn_mask_actions = QHBoxLayout()
        btn_mask_actions.setSpacing(6)
        self.btn_refine = QPushButton("Refinar con GrabCut")
        self.btn_refine.setIcon(get_icon("brush", size=13, color="#b388ff"))
        self.btn_refine.setStyleSheet("font-weight: bold; color: #b388ff; padding: 5px;")
        self.btn_refine.setToolTip("Refina la máscara usando GrabCut y los trazos verde (cielo) y rojo (suelo) sobre la foto base.")
        self.btn_refine.clicked.connect(self.refine_mask)

        self.btn_direct_brush = QPushButton("Pintar en Máscara")
        self.btn_direct_brush.setIcon(get_icon("sparkles", size=13, color="#80d8ff"))
        self.btn_direct_brush.setStyleSheet("font-weight: bold; color: #80d8ff; padding: 5px;")
        self.btn_direct_brush.setToolTip("Aplica los trazos dibujados directamente sobre la máscara actual (aditivo/sustractivo) al instante sin esperar.")
        self.btn_direct_brush.clicked.connect(self.apply_direct_brush)

        self.btn_clear_m = QPushButton("Limpiar")
        self.btn_clear_m.setIcon(get_icon("trash", size=13, color="#ff8a80"))
        self.btn_clear_m.setStyleSheet("font-weight: bold; color: #ff8a80; padding: 5px;")
        self.btn_clear_m.setToolTip("Elimina la máscara actual y los trazos de pincel para empezar de cero.")
        self.btn_clear_m.clicked.connect(self.clear_mask)

        btn_mask_actions.addWidget(self.btn_refine)
        btn_mask_actions.addWidget(self.btn_direct_brush)
        btn_mask_actions.addWidget(self.btn_clear_m)
        mask_layout.addLayout(btn_mask_actions)

        # Ajuste fino perimétrico de silueta (Expandir / Contraer)
        row_expand = QHBoxLayout()
        row_expand.setSpacing(6)
        self.btn_expand_gnd = QPushButton("Expandir Suelo (+2px)")
        self.btn_expand_gnd.setStyleSheet("color: #ffd54f; font-size: 11px; font-weight: bold; padding: 4px 6px;")
        self.btn_expand_gnd.setToolTip("Dilata la silueta del suelo 2 píxeles hacia el cielo para asegurar que las ramas finas queden cubiertas.")
        self.btn_expand_gnd.clicked.connect(lambda: self.expand_ground(2))

        self.btn_shrink_gnd = QPushButton("Contraer (-2px)")
        self.btn_shrink_gnd.setStyleSheet("color: #b0bec5; font-size: 11px; padding: 4px 6px;")
        self.btn_shrink_gnd.setToolTip("Contrae la silueta del suelo 2 píxeles hacia adentro.")
        self.btn_shrink_gnd.clicked.connect(lambda: self.expand_ground(-2))

        row_expand.addWidget(self.btn_expand_gnd)
        row_expand.addWidget(self.btn_shrink_gnd)
        mask_layout.addLayout(row_expand)

        btn_mask_io = QHBoxLayout()
        btn_mask_io.setSpacing(6)
        self.btn_load_m = QPushButton("Cargar PNG")
        self.btn_load_m.setIcon(get_icon("folder", size=13, color="#80cbc4"))
        self.btn_load_m.setStyleSheet("font-weight: bold; color: #80cbc4;")
        self.btn_load_m.clicked.connect(self.load_mask)
        self.btn_save_m = QPushButton("Exportar PNG")
        self.btn_save_m.setIcon(get_icon("save", size=13, color="#90caf9"))
        self.btn_save_m.setStyleSheet("font-weight: bold; color: #90caf9;")
        self.btn_save_m.clicked.connect(self.save_mask)
        btn_mask_io.addWidget(self.btn_load_m)
        btn_mask_io.addWidget(self.btn_save_m)
        mask_layout.addLayout(btn_mask_io)

        left_layout.addWidget(self.grp_mask)

        # Cierre del scroll_area
        scroll_area.setWidget(panel_content)
        left_panel_layout.addWidget(scroll_area, stretch=1)

        # SECCIÓN INFERIOR FIJA: Progreso, Registro de Actividad y Ejecución
        lbl_log = QLabel("Registro de Actividad:")
        lbl_log.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold;")
        left_panel_layout.addWidget(lbl_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(100)
        left_panel_layout.addWidget(self.txt_log)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        left_panel_layout.addWidget(self.progress_bar)

        self.btn_run = QPushButton("INICIAR APILADO DUAL")
        self.btn_run.setIcon(get_icon("play", size=16, color="#ffffff"))
        self.btn_run.setFixedHeight(44)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_run.clicked.connect(self._on_btn_run_clicked)
        left_panel_layout.addWidget(self.btn_run)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        splitter.setSizes([520, 1400])
        layout.addWidget(splitter)

    def _apply_styles(self) -> None:
        """Los estilos base provienen de gui.theme (Cosmic Dark Pro)."""
        pass

    def log_message(self, text: str):
        self.txt_log.append(text)
        self.txt_log.moveCursor(QTextCursor.End)

    def _on_capture_mode_changed(self, idx: int):
        """Si es star tracker, inhabilita el modo de apilar suelo de ráfaga y anti-ghosting."""
        is_fixed = (idx == 0)
        self.combo_ground.setEnabled(is_fixed)
        if hasattr(self, 'box_anti_ghosting'):
            self.box_anti_ghosting.setEnabled(is_fixed)
        if not is_fixed:
            self.log_message("[MODO] Star Tracker activado: apilado restringido al cielo en movimiento.")

    def _on_sub_w_slider_changed(self, val: int):
        if val % 2 == 0:
            val += 1
            self.slider_sub_w.setValue(val)
        snr = np.sqrt(val)
        ev = np.log2(snr)
        self.lbl_sub_w_val.setText(f"{val} tomas (+{snr:.1f}x SNR, +{ev:.1f} EV)")

    def _on_stack_type_toggled(self, checked: bool):
        is_video = self.rb_out_video.isChecked()
        self.box_substack_settings.setVisible(is_video)
        if hasattr(self, 'row_kappa_widget'):
            self.row_kappa_widget.setVisible(not is_video)
        if hasattr(self, 'row_ground_widget'):
            self.row_ground_widget.setVisible(not is_video)
        if hasattr(self, 'row_mode_widget'):
            self.row_mode_widget.setVisible(not is_video)

        if is_video:
            self.btn_run.setText("GENERAR VÍDEO TIME-LAPSE (SUB-STACKING)")
            self.btn_run.setIcon(get_icon("video", size=16, color="#ffffff"))
            self.log_message("[MODO] Activado Apilado de Ventana Deslizante (Vídeo Time-Lapse).")
        else:
            self.btn_run.setText("INICIAR APILADO DUAL")
            self.btn_run.setIcon(get_icon("play", size=16, color="#ffffff"))
            self.log_message("[MODO] Activado Apilado Estático (1 Imagen Compuesta).")

    def _on_files_tab_changed(self, idx: int):
        """Muestra la barra de análisis de calidad únicamente en la pestaña de Lights."""
        if hasattr(self, 'culling_bar'):
            self.culling_bar.setVisible(idx == 0)

    # --- Reconstrucción de la lista visual ---
    def _refresh_list_view(self, tab_name: str):
        if tab_name == "Lights":
            widget_list = self.list_lights
            files_data = self.lights_list
        elif tab_name == "Darks":
            widget_list = self.list_darks
            files_data = self.darks_list
        elif tab_name == "Flats":
            widget_list = self.list_flats
            files_data = self.flats_list
        elif tab_name == "Bias":
            widget_list = self.list_bias
            files_data = self.bias_list
        else:
            widget_list = self.list_ground
            files_data = self.ground_list

        widget_list.clear()
        for idx, path in enumerate(files_data):
            item = QListWidgetItem(widget_list)
            is_ref = (tab_name == "Lights") and (path == self.current_ref_path)
            is_included = (path not in self.excluded_lights) if tab_name == "Lights" else True
            row_widget = FileListRow(path, tab_name=tab_name, is_ref=is_ref, is_included=is_included)
            row_widget.delete_requested.connect(self.delete_single_file)
            row_widget.include_toggled.connect(self.on_file_include_toggled)

            if tab_name == "Lights" and path in self.culling_results:
                row_widget.set_quality_result(self.culling_results[path])

            item.setSizeHint(row_widget.sizeHint())
            widget_list.setItemWidget(item, row_widget)

        if tab_name == "Lights":
            self._update_culling_summary_label()

    def delete_single_file(self, tab_name: str, filepath: str):
        if tab_name == "Lights":
            file_list = self.lights_list
        elif tab_name == "Darks":
            file_list = self.darks_list
        elif tab_name == "Flats":
            file_list = self.flats_list
        elif tab_name == "Bias":
            file_list = self.bias_list
        else:
            file_list = self.ground_list

        if filepath in file_list:
            file_list.remove(filepath)
            self.log_message(f"Eliminada toma de {tab_name}: {os.path.basename(filepath)}")

            if tab_name == "Lights":
                if filepath == self.current_ref_path:
                    if self.lights_list:
                        self.load_frame_to_canvas(self.lights_list[0])
                    else:
                        self.current_ref_path = None
                        self.clear_mask()
                        self.canvas.clear_image()
                elif not self.lights_list:
                    self.current_ref_path = None
                    self.clear_mask()
                    self.canvas.clear_image()
                self._refresh_list_view("Lights")
            elif tab_name == "Darks":
                self._refresh_list_view("Darks")
            elif tab_name == "Flats":
                self._refresh_list_view("Flats")
            elif tab_name == "Bias":
                self._refresh_list_view("Bias")
            else:
                self._refresh_list_view("Suelo")

    def add_current_tab_files(self):
        idx = self.tabs_files.currentIndex()
        tab_names = ["Lights", "Darks", "Flats", "Bias", "Suelo"]
        tab_name = tab_names[idx]

        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Seleccionar {tab_name}", "",
            "Astro Images (*.nef *.cr2 *.cr3 *.arw *.dng *.tif *.tiff *.fits *.jpg *.png)"
        )
        if not paths: 
            return

        self.add_files_to_tab(tab_name, paths)

    def add_files_to_tab(self, tab_name: str, paths: list):
        """Añade tomas a la pestaña indicada, deduplicando y actualizando vista previa si procede."""
        if not paths:
            return

        if tab_name == "Lights":
            first_add = len(self.lights_list) == 0
            new_paths = [p for p in paths if p not in self.lights_list]
            self.lights_list.extend(new_paths)
            if first_add and self.lights_list:
                self.load_frame_to_canvas(self.lights_list[0])
            self._refresh_list_view("Lights")
            self.tabs_files.setCurrentIndex(0)
            self.log_message(f"Añadidos {len(new_paths)} Lights (Total: {len(self.lights_list)})")

        elif tab_name == "Darks":
            new_paths = [p for p in paths if p not in self.darks_list]
            self.darks_list.extend(new_paths)
            self._refresh_list_view("Darks")
            self.tabs_files.setCurrentIndex(1)
            self.log_message(f"Añadidos {len(new_paths)} Darks (Total: {len(self.darks_list)})")

        elif tab_name == "Flats":
            new_paths = [p for p in paths if p not in self.flats_list]
            self.flats_list.extend(new_paths)
            self._refresh_list_view("Flats")
            self.tabs_files.setCurrentIndex(2)
            self.log_message(f"Añadidos {len(new_paths)} Flats (Total: {len(self.flats_list)})")

        elif tab_name == "Bias":
            new_paths = [p for p in paths if p not in self.bias_list]
            self.bias_list.extend(new_paths)
            self._refresh_list_view("Bias")
            self.tabs_files.setCurrentIndex(3)
            self.log_message(f"Añadidos {len(new_paths)} Bias (Total: {len(self.bias_list)})")

        else:
            new_paths = [p for p in paths if p not in self.ground_list]
            self.ground_list.extend(new_paths)
            self._refresh_list_view("Suelo")
            self.tabs_files.setCurrentIndex(4)
            self.combo_ground.setCurrentIndex(2)
            self.log_message(f"Añadida(s) {len(new_paths)} toma(s) de Suelo. Modo suelo actualizado a 'Usar Toma de Pestaña Suelo'.")

    def clear_current_tab_files(self):
        idx = self.tabs_files.currentIndex()
        if idx == 0:
            self.lights_list.clear()
            self.excluded_lights.clear()
            self.culling_results.clear()
            self.list_lights.clear()
            if hasattr(self, 'lbl_cull_summary'):
                self.lbl_cull_summary.setText("")
            self.current_ref_path = None
            self.clear_mask()
            self.canvas.clear_image()
            self.log_message("Lista de Lights vaciada y máscara reiniciada.")
        elif idx == 1:
            self.darks_list.clear()
            self.list_darks.clear()
            self.log_message("Lista de Darks vaciada.")
        elif idx == 2:
            self.flats_list.clear()
            self.list_flats.clear()
            self.log_message("Lista de Flats vaciada.")
        elif idx == 3:
            self.bias_list.clear()
            self.list_bias.clear()
            self.log_message("Lista de Bias vaciada.")
        else:
            self.ground_list.clear()
            self.list_ground.clear()
            self.log_message("Lista de Suelo vaciada.")

    def on_file_include_toggled(self, tab_name: str, filepath: str, is_included: bool):
        """Gestiona la inclusión/exclusión de tomas para el apilado."""
        if tab_name == "Lights":
            if is_included:
                self.excluded_lights.discard(filepath)
            else:
                self.excluded_lights.add(filepath)
            self._update_culling_summary_label()

    def _update_culling_summary_label(self):
        """Actualiza la etiqueta de balance de calidad en la barra de cribado."""
        if not hasattr(self, 'lbl_cull_summary'):
            return
        total = len(self.lights_list)
        if total == 0:
            self.lbl_cull_summary.setText("")
            return
        active = total - len(self.excluded_lights)
        pct = (active / total) * 100.0 if total > 0 else 0
        self.lbl_cull_summary.setText(f"Aptas: {active}/{total} ({pct:.0f}%)")

    def start_culling_analysis(self):
        """Lanza el análisis paralelo de calidad estelar sobre las tomas de Lights."""
        if not self.lights_list:
            QMessageBox.information(self, "Aviso", "Añade primero tomas de Lights para analizar.")
            return

        self.btn_cull_analyze.setEnabled(False)
        self.btn_cull_discard.setEnabled(False)
        self.btn_cull_best_ref.setEnabled(False)
        self.lbl_cull_summary.setText("Analizando calidad...")
        self.log_message(f"=== INICIANDO ANÁLISIS DE CALIDAD ESTELAR ({len(self.lights_list)} tomas) ===")

        profile = self.combo_cull_profile.currentData() or "wide_angle"
        self.culling_worker = QualityCullingWorker(
            file_paths=list(self.lights_list),
            sky_mask=self.computed_mask,
            profile_mode=profile,
            parent=self
        )
        self.culling_worker.progress_changed.connect(self._on_culling_progress)
        self.culling_worker.all_finished.connect(self._on_culling_finished)
        self.culling_worker.error_occurred.connect(self._on_culling_error)
        self.culling_worker.start()

    def _on_culling_progress(self, current: int, total: int, filename: str):
        pct = int((current / max(1, total)) * 100)
        self.lbl_cull_summary.setText(f"Analizando: {current}/{total} ({pct}%)")

    def _on_culling_finished(self, results: list):
        self.btn_cull_analyze.setEnabled(True)
        self.btn_cull_discard.setEnabled(True)

        self.btn_cull_best_ref.setEnabled(True)
        defective_count = 0
        for r in results:
            self.culling_results[r.filepath] = r
            if not r.is_recommended:
                defective_count += 1

        self._refresh_list_view("Lights")
        total = len(results)
        aptas = total - defective_count
        self.log_message(f"✓ [CRIBADO] Análisis completado: {aptas}/{total} tomas excelentes/aceptables, {defective_count} con anomalías.")
        if defective_count > 0:
            self.log_message(f"-> Pulsa 'Auto-Descartar' para excluir automáticamente las {defective_count} tomas defectuosas.")

        self.auto_select_best_reference(silent=True)

    def _on_culling_error(self, err_msg: str):
        self.btn_cull_analyze.setEnabled(True)
        self.btn_cull_discard.setEnabled(True)
        self.btn_cull_best_ref.setEnabled(True)
        self.lbl_cull_summary.setText("Error en análisis")
        self.log_message(f"[ERROR CRIBADO] {err_msg}")

    def auto_discard_defective_frames(self):
        """Desmarca automáticamente las tomas clasificadas como defectuosas."""
        if not self.culling_results:
            self.start_culling_analysis()
            return

        discarded_now = 0
        for path, res in self.culling_results.items():
            if not res.is_recommended:
                self.excluded_lights.add(path)
                discarded_now += 1

        self._refresh_list_view("Lights")
        self.log_message(f"[CRIBADO] Auto-descartadas {discarded_now} tomas defectuosas (viento/nubes/foco).")

    def auto_select_best_reference(self, *args, silent: bool = False):
        """Selecciona y fija automáticamente la toma con mejor calidad estelar y menor FWHM como referencia."""
        if not self.culling_results:
            self.start_culling_analysis()
            return

        is_silent = silent or (len(args) > 0 and isinstance(args[0], bool) and args[0] is True)
        from core.frame_culling import pick_best_reference_frame
        best = pick_best_reference_frame(list(self.culling_results.values()), excluded_paths=self.excluded_lights)
        if best is None and self.culling_results:
            best = pick_best_reference_frame(list(self.culling_results.values()), excluded_paths=None)

        if not best or not best.filepath:
            if not is_silent:
                self.log_message("[REFERENCIA] No se pudo determinar una toma óptima.")
            return

        best_path = best.filepath
        if best_path in self.lights_list:
            self.lights_list.remove(best_path)
            self.lights_list.insert(0, best_path)
            self.load_frame_to_canvas(best_path)
            self._refresh_list_view("Lights")
            fwhm_str = f"{best.fwhm:.2f} px" if (best.fwhm and best.fwhm > 0) else "N/D"
            score_str = f"{best.score:.1f}" if (best.score is not None and not np.isnan(best.score)) else "N/D"
            self.log_message(
                f"✓ [REFERENCIA ÓPTIMA] Base fijada: {os.path.basename(best_path)} "
                f"(Score: {score_str}, FWHM: {fwhm_str}, Estrellas: {best.star_count})"
            )

    def on_light_double_clicked(self, item):
        row = self.list_lights.row(item)
        if 0 <= row < len(self.lights_list):
            selected_path = self.lights_list[row]
            self.lights_list.remove(selected_path)
            self.lights_list.insert(0, selected_path)
            self.load_frame_to_canvas(selected_path)
            self._refresh_list_view("Lights")
            self.log_message(f"[REFERENCIA] Toma fijada como base: {os.path.basename(selected_path)}")

    def load_frame_to_canvas(self, path: str):
        self.current_ref_path = path
        self.log_message(f"Cargando toma de referencia: {os.path.basename(path)}...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()
        try:
            img = load_image_as_float32(path)
            self.canvas.load_image(img)
            self.log_message(f"Vista previa activa: {os.path.basename(path)}")
        except Exception as e:
            self.log_message(f"[ERROR] Al cargar {os.path.basename(path)}: {e}")
            QMessageBox.critical(self, "Error al cargar imagen", f"No se pudo cargar {path}:\n{e}")
        finally:
            self.unsetCursor()

    # --- Métodos de Máscara ---
    def update_brush_mode(self):
        self.canvas.brush_mode = 2 if self.rb_sky.isChecked() else 1

    def update_brush_slider(self, val):
        self.canvas.set_brush_radius(val)
        self.lbl_brush.setText(f"{val * 2} px")

    def on_canvas_brush_changed(self, radius):
        self.slider_brush.blockSignals(True)
        self.slider_brush.setValue(radius)
        self.slider_brush.blockSignals(False)
        self.lbl_brush.setText(f"{radius * 2} px")

    def start_auto_ai_mask(self):
        """Lanza la detección automática de cielo/suelo mediante IA Ligera o fallback."""
        if self.canvas.orig_rgb is None:
            if self.lights_list:
                ref_path = self.current_ref_path or self.lights_list[0]
                self.load_frame_to_canvas(ref_path)
            else:
                QMessageBox.information(self, "Aviso", "Añade primero tomas de Lights para detectar el suelo.")
                return

        if self.canvas.orig_rgb is None:
            QMessageBox.warning(self, "Aviso", "No se pudo cargar la imagen de referencia para el análisis.")
            return

        from core.ai_horizon import is_ai_model_available
        if not is_ai_model_available():
            msg = QMessageBox(self)
            msg.setWindowTitle("Modelo de IA no encontrado")
            msg.setText(
                "El modelo de IA para detección semántica precisa de suelo (skyseg.onnx, ~167 MB) no se encuentra en la carpeta 'models/'.\n\n"
                "¿Deseas descargarlo ahora automáticamente o prefieres usar el método algorítmico rápido?"
            )
            btn_download = msg.addButton("Descargar IA (~167 MB)", QMessageBox.AcceptRole)
            btn_algo = msg.addButton("Usar Algorítmico", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_download)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_download:
                self._start_model_download_stacker()
                return
            elif clicked == btn_algo:
                self._run_horizon_worker(force_algorithmic=True)
                return
            else:
                return

        self._run_horizon_worker(force_algorithmic=False)

    def _start_model_download_stacker(self):
        """Descarga asistida del modelo ONNX con diálogo de progreso."""
        progress = QProgressDialog("Conectando con el servidor de modelos...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Descarga de Modelo IA (AstroLoom)")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        self.download_worker = ModelDownloadWorker(parent=self)

        def on_prog(pct, d_mb, t_mb):
            progress.setValue(pct)
            progress.setLabelText(f"Descargando modelo de IA (skyseg.onnx)...\n{d_mb:.1f} MB de {t_mb:.1f} MB ({pct}%)")

        def on_finished(path):
            progress.close()
            self.log_message(f"✓ [IA HORIZONTE] Modelo descargado e instalado en: {os.path.basename(path)}")
            self._run_horizon_worker(force_algorithmic=False)

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR DESCARGA] {err}")
            QMessageBox.warning(self, "Error en Descarga", f"No se pudo descargar el modelo:\n{err}\n\nSe usará el método algorítmico.")
            self._run_horizon_worker(force_algorithmic=True)

        def on_cancel():
            self.download_worker.cancel()
            self.log_message("[IA HORIZONTE] Descarga cancelada por el usuario.")

        self.download_worker.progress_changed.connect(on_prog)
        self.download_worker.finished_success.connect(on_finished)
        self.download_worker.error_occurred.connect(on_error)
        progress.canceled.connect(on_cancel)

        self.download_worker.start()

    def _run_horizon_worker(self, force_algorithmic: bool = False):
        self.btn_auto_ai_mask.setEnabled(False)
        self.btn_auto_ai_mask.setText("Detectando suelo...")
        self.btn_auto_ai_mask.setIcon(get_icon("refresh", size=15, color="#ffffff"))
        self.log_message("[IA HORIZONTE] Ejecutando segmentación de cielo y suelo...")
        self.setCursor(Qt.WaitCursor)

        feather = self.spin_feather.value()
        self.ai_worker = AIHorizonWorker(
            image_rgb=self.canvas.orig_rgb,
            feather_radius=feather,
            parent=self
        )
        self.ai_worker.finished_success.connect(self._on_ai_mask_finished)
        self.ai_worker.error_occurred.connect(self._on_ai_mask_error)
        self.ai_worker.start()

    def _on_ai_mask_finished(self, mask: np.ndarray, method_name: str):
        self.btn_auto_ai_mask.setEnabled(True)
        self.btn_auto_ai_mask.setText("Auto-Detectar Suelo (IA 1-Click)")
        self.btn_auto_ai_mask.setIcon(get_icon("sparkles", size=15, color="#ffffff"))
        self.unsetCursor()

        self.computed_mask = mask
        self.canvas.set_refined_mask(self.computed_mask)
        self.log_message(f"✓ [IA HORIZONTE] Suelo y cielo detectados con éxito usando {method_name}.")

    def _on_ai_mask_error(self, err_msg: str):
        self.btn_auto_ai_mask.setEnabled(True)
        self.btn_auto_ai_mask.setText("Auto-Detectar Suelo (IA 1-Click)")
        self.btn_auto_ai_mask.setIcon(get_icon("sparkles", size=15, color="#ffffff"))
        self.unsetCursor()
        self.log_message(f"[ERROR IA HORIZONTE] {err_msg}")
        QMessageBox.warning(self, "Error en Detección IA", f"No se pudo completar la segmentación automática:\n{err_msg}")

    def refine_mask(self):
        if self.canvas.orig_rgb is None:
            QMessageBox.warning(self, "Aviso", "Abre primero una toma base.")
            return
        
        scribbles = self.canvas.get_scribbles_matrix()
        has_any_stroke = scribbles is not None and (np.any(scribbles == 1) or np.any(scribbles == 2))
        if not has_any_stroke:
            QMessageBox.warning(self, "Aviso", "Pinta primero al menos un trazo en el visor (Verde = Cielo, Rojo = Suelo).")
            return

        if self.computed_mask is None and not (np.any(scribbles == 1) and np.any(scribbles == 2)):
            QMessageBox.warning(self, "Aviso", "Sin máscara previa, debes pintar al menos un trazo verde (cielo) y uno rojo (suelo).")
            return

        self.log_message("[MÁSCARA] Iniciando refinamiento con GrabCut asistido...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()

        feather = self.spin_feather.value()
        iters = self.spin_iter.value()
        t_start = time.perf_counter()

        try:
            self.computed_mask = refine_mask_guided(
                self.canvas.orig_rgb, 
                scribbles,
                feather_radius=feather,
                iterations=iters,
                base_mask=self.computed_mask
            )
            self.canvas.set_refined_mask(self.computed_mask)
            elapsed = time.perf_counter() - t_start
            self.log_message(f"✓ [MÁSCARA FINALIZADA] Refinamiento completado con éxito en {elapsed:.2f} s (Borde: {feather}px, Pasos: {iters}).")
        except Exception as e:
            self.log_message(f"[ERROR MÁSCARA] Falló el refinado: {e}")
            QMessageBox.critical(self, "Error", f"Fallo al refinar la máscara:\n{e}")
        finally:
            self.unsetCursor()

    def apply_direct_brush(self):
        """Aplica los trazos dibujados directamente sobre la máscara actual (aditivo/sustractivo)."""
        if self.canvas.orig_rgb is None or self.computed_mask is None:
            QMessageBox.warning(self, "Aviso", "Genera o carga primero una máscara (ej. con IA 1-Click).")
            return
        scribbles = self.canvas.get_scribbles_matrix()
        if scribbles is None or not (np.any(scribbles == 1) or np.any(scribbles == 2)):
            QMessageBox.warning(self, "Aviso", "Pinta primero trazos en el visor (Verde = Cielo, Rojo = Suelo).")
            return
        feather = self.spin_feather.value()
        self.computed_mask = apply_direct_brush_to_mask(self.computed_mask, scribbles, feather_radius=max(1, feather // 2))
        self.canvas.set_refined_mask(self.computed_mask)
        # Limpiar trazos tras aplicar para dejar el visor limpio
        if hasattr(self.canvas, 'scribble_pixmap') and self.canvas.scribble_pixmap:
            self.canvas.scribble_pixmap.fill(Qt.transparent)
            self.canvas.update()
        self.log_message("✓ Trazos de pincel integrados directamente en la máscara.")

    def expand_ground(self, pixels: int = 2):
        """Expande o contrae la silueta del suelo para asegurar cobertura en follaje fino."""
        if self.computed_mask is None:
            QMessageBox.warning(self, "Aviso", "Genera o carga primero una máscara de horizonte.")
            return
        k_size = 2 * abs(pixels) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        if pixels > 0:
            # Expandir suelo = erosionar cielo (1.0 = cielo, 0.0 = suelo)
            self.computed_mask = cv2.erode(self.computed_mask, kernel)
            self.log_message(f"✓ Silueta de suelo expandida +{pixels}px sobre las ramas.")
        else:
            # Contraer suelo = dilatar cielo
            self.computed_mask = cv2.dilate(self.computed_mask, kernel)
            self.log_message(f"✓ Silueta de suelo contraída {abs(pixels)}px.")
        self.canvas.set_refined_mask(self.computed_mask)

    def clear_mask(self):
        """Elimina todos los trazos y la máscara calculada para empezar de cero."""
        self.computed_mask = None
        if hasattr(self.canvas, 'clear_all_masking'):
            self.canvas.clear_all_masking()
        else:
            if self.canvas.scribble_pixmap:
                self.canvas.scribble_pixmap.fill(Qt.transparent)
            self.canvas.set_refined_mask(None)
        self.canvas.update()
        self.log_message("Máscara y trazos eliminados por completo.")

    def save_mask(self):
        if self.computed_mask is None:
            return
        cfg = load_config()
        default_dir = cfg.get("masks_dir", os.getcwd())
        default_path = os.path.join(default_dir, "mascara_cielo.png")
        
        p, _ = QFileDialog.getSaveFileName(self, "Exportar Máscara PNG", default_path, "PNG (*.png)")
        if p:
            cv2.imwrite(p, (self.computed_mask * 65535.0).astype(np.uint16))
            self.log_message(f"Máscara exportada en: {os.path.basename(p)}")

    def load_mask(self):
        if self.canvas.orig_rgb is None:
            return
        cfg = load_config()
        default_dir = cfg.get("masks_dir", os.getcwd())
        
        p, _ = QFileDialog.getOpenFileName(self, "Cargar Máscara PNG", default_dir, "Imágenes (*.png *.tif *.tiff)")
        if not p:
            return
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3:
            raw = raw[..., 0]
        h, w = self.canvas.orig_rgb.shape[:2]
        if raw.shape != (h, w):
            raw = cv2.resize(raw, (w, h))
        max_v = 65535.0 if raw.dtype == np.uint16 else 255.0
        self.computed_mask = np.clip(raw.astype(np.float32) / max_v, 0.0, 1.0)
        self.canvas.set_refined_mask(self.computed_mask)
        self.log_message(f"Máscara cargada desde: {os.path.basename(p)}")

    # --- Sesiones ---
    def save_project(self):
        if self.project_mgr.project_path:
            self._do_save(self.project_mgr.project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        cfg = load_config()
        default_dir = cfg.get("sessions_dir", os.getcwd())
        default_file = os.path.join(default_dir, "mi_sesion.mwstack")

        p, _ = QFileDialog.getSaveFileName(
            self, "Guardar Sesión de Apilado", default_file,
            "Astro Session (*.mwstack *.json)"
        )
        if not p:
            return
        self._do_save(p)

    def open_project(self):
        cfg = load_config()
        default_dir = cfg.get("sessions_dir", os.getcwd())

        p, _ = QFileDialog.getOpenFileName(
            self, "Cargar Sesión de Apilado", default_dir,
            "Astro Session (*.mwstack *.json)"
        )
        if not p:
            return

        try:
            data, loaded_mask = self.project_mgr.load(p)

            self.clear_mask()
            self.canvas.clear_image()

            self.lights_list.clear()
            self.darks_list.clear()
            self.flats_list.clear()
            self.ground_list.clear()
            self.bias_list.clear()

            for fpath in data.get("light_frames", []):
                if os.path.exists(fpath):
                    self.lights_list.append(fpath)

            for fpath in data.get("dark_frames", []):
                if os.path.exists(fpath):
                    self.darks_list.append(fpath)

            for fpath in data.get("flat_frames", []):
                if os.path.exists(fpath):
                    self.flats_list.append(fpath)

            for fpath in data.get("bias_frames", []):
                if os.path.exists(fpath):
                    self.bias_list.append(fpath)

            for fpath in data.get("ground_frames", []):
                if os.path.exists(fpath):
                    self.ground_list.append(fpath)

            mode = data.get("mode", "fixed_tripod")
            self.combo_mode.setCurrentIndex(0 if mode == "fixed_tripod" else 1)
            self.combo_ground.setCurrentIndex(data.get("ground_mode_idx", 0))
            
            params = data.get("parameters", {})
            self.spin_kappa.setValue(params.get("kappa", 2.2))
            self.combo_lp_algo.setCurrentIndex(params.get("lp_method_idx", 0))
            self.slider_stack_lp.setValue(params.get("lp_strength", 50))

            if self.lights_list:
                self.load_frame_to_canvas(self.lights_list[0])

            self._refresh_list_view("Lights")
            self._refresh_list_view("Darks")
            self._refresh_list_view("Flats")
            self._refresh_list_view("Suelo")
            self._refresh_list_view("Bias")

            if loaded_mask is not None:
                self.computed_mask = loaded_mask
                self.canvas.set_refined_mask(self.computed_mask)
                self.chk_show_mask.setChecked(True)
                self.log_message("[PROYECTO] Máscara restaurada.")

            name = os.path.basename(p)
            self.lbl_session.setText(f"Sesión: {name}")
            self.session_title_changed.emit(name)
            self.log_message(
                f"[PROYECTO] Cargada: {len(self.lights_list)} Lights, "
                f"{len(self.darks_list)} Darks, {len(self.flats_list)} Flats, "
                f"{len(self.bias_list)} Bias, {len(self.ground_list)} Suelo."
            )

        except Exception as e:
            self.log_message(f"[ERROR] Al abrir sesión: {e}")
            QMessageBox.critical(self, "Error al abrir sesión", str(e))

    def _do_save(self, filepath: str):
        self.project_mgr.data["light_frames"] = self.lights_list
        self.project_mgr.data["dark_frames"] = self.darks_list
        self.project_mgr.data["flat_frames"] = self.flats_list
        self.project_mgr.data["bias_frames"] = self.bias_list
        self.project_mgr.data["ground_frames"] = self.ground_list
        self.project_mgr.data["mode"] = "fixed_tripod" if self.combo_mode.currentIndex() == 0 else "star_tracker"
        self.project_mgr.data["ground_mode_idx"] = self.combo_ground.currentIndex()
        self.project_mgr.data["parameters"] = {
            "kappa": self.spin_kappa.value(),
            "lp_method_idx": self.combo_lp_algo.currentIndex(),
            "lp_strength": self.slider_stack_lp.value()
        }

        self.project_mgr.save(filepath, mask_array=self.computed_mask)
        name = os.path.basename(filepath)
        self.lbl_session.setText(f"Sesión: {name}")
        self.session_title_changed.emit(name)
        self.log_message(f"[PROYECTO] Sesión guardada en: {name}")
        QMessageBox.information(self, "Sesión Guardada", f"Guardada en:\n{filepath}")

    def reset_session(self):
        self.lights_list.clear()
        self.darks_list.clear()
        self.flats_list.clear()
        self.ground_list.clear()
        self.list_lights.clear()
        self.list_darks.clear()
        self.list_flats.clear()
        self.bias_list.clear()
        self.list_bias.clear()      
        self.list_ground.clear()
        self.excluded_lights.clear()
        self.culling_results.clear()
        self._update_culling_summary_label()
        self.current_ref_path = None
        self.last_stacked_output_path = None

        self.project_mgr = ProjectManager()
        self.lbl_session.setText("Sesión: Sin guardar")
        self.session_title_changed.emit("")

        self.clear_mask()
        self.canvas.clear_image()

        self.combo_mode.setCurrentIndex(0)
        self.combo_ground.setCurrentIndex(0)
        self.spin_kappa.setValue(2.2)
        self.combo_lp_algo.setCurrentIndex(0)
        self.slider_stack_lp.setValue(50)
        self.spin_feather.setValue(7)
        self.spin_iter.setValue(3)
        self.chk_anti_ghosting.setChecked(True)
        self.spin_ag_radius.setValue(12)
        self.combo_ag_sens.setCurrentIndex(0)
        if hasattr(self, 'rb_out_single'):
            self.rb_out_single.setChecked(True)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.txt_log.clear()

    def start_stacking(self):
        active_lights = [p for p in self.lights_list if p not in self.excluded_lights]
        if self.current_ref_path and self.current_ref_path in active_lights:
            if active_lights[0] != self.current_ref_path:
                active_lights.remove(self.current_ref_path)
                active_lights.insert(0, self.current_ref_path)
        if len(active_lights) < 2:
            QMessageBox.warning(
                self, "Aviso", 
                f"Añade al menos 2 tomas de luz (Lights) activas para apilar.\n"
                f"(Activas: {len(active_lights)} / Total: {len(self.lights_list)}, Excluidas: {len(self.excluded_lights)})"
            )
            return

        app_cfg = load_config()
        default_stacked_dir = app_cfg.get("stacked_dir", os.getcwd())
        default_file = os.path.join(default_stacked_dir, "resultado_dual_32bit.tiff")

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Resultado 32-bit", default_file, "TIFF (*.tiff *.tif)"
        )
        if not out_path: 
            return

        self._stack_start_time = time.perf_counter()
        self.last_stacked_output_path = out_path

        self._set_ui_busy(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("CANCELAR APILADO")
        self.btn_run.setIcon(get_icon("stop", size=16, color="#ffffff"))
        self.btn_run.setStyleSheet(STYLE_RUN_ACTIVE)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Iniciando... %p%")

        lp_mode_idx = self.combo_lp_algo.currentIndex()
        lp_methods = ["standard", "sequator_subtraction", "min_rejection", "local_norm"]
        selected_lp_method = lp_methods[lp_mode_idx]
        selected_lp_strength = self.slider_stack_lp.value() / 100.0

        cpu_workers = int(app_cfg.get("cpu_workers", max(1, (os.cpu_count() or 4) - 1)))

        try:
            saved_storage = self.window().tab_settings.combo_storage.currentData()
        except Exception:
            saved_storage = "auto"
            
        ground_modes = ["dual", "reference", "external"]
        selected_ground_mode = ground_modes[self.combo_ground.currentIndex()]

        external_path = None
        if selected_ground_mode == "external":
            if not self.ground_list:
                self._set_ui_busy(False)
                self._reset_run_button()
                QMessageBox.warning(
                    self, "Aviso", 
                    "Has seleccionado 'Usar Toma de Pestaña Suelo' pero la pestaña Suelo está vacía.\n"
                    "Añade una toma en la pestaña Suelo o cambia el modo de suelo."
                )
                return
            external_path = self.ground_list[0]

        cfg = {
            "lights": active_lights,
            "reference_frame": self.current_ref_path or active_lights[0],
            "darks": self.darks_list,
            "flats": self.flats_list,
            "mask": self.computed_mask,
            "bias": self.bias_list,
            "mode": "fixed_tripod" if self.combo_mode.currentIndex() == 0 else "star_tracker",
            "ground_mode": selected_ground_mode,
            "external_ground_path": external_path,
            "kappa": self.spin_kappa.value(),
            "output_path": out_path,
            "lp_method": selected_lp_method,
            "lp_strength": selected_lp_strength,
            "anti_ghosting": self.chk_anti_ghosting.isChecked(),
            "anti_ghosting_radius": self.spin_ag_radius.value(),
            "anti_ghosting_sensitivity": self.combo_ag_sens.currentData() or 0.5,
            "use_gpu": is_gpu_enabled(),
            "storage_strategy": saved_storage,
            "cpu_workers": cpu_workers,
            "icc_profile": app_cfg.get("icc_profile", "srgb")
        }

        self.worker = StackingWorker(cfg)
        self.worker.progress_changed.connect(self.on_progress_changed)
        self.worker.status_changed.connect(self.log_message)
        self.worker.finished_success.connect(self.on_stack_success)
        self.worker.error_occurred.connect(self.on_stack_error)
        self.worker.cancelled.connect(self.on_stack_cancelled)
        self.worker.start()

    def on_stack_success(self, path):
        self._reset_run_button()

        if hasattr(self, '_stack_start_time') and self._stack_start_time is not None:
            elapsed = time.perf_counter() - self._stack_start_time
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s" if mins > 0 else f"{secs:.2f}s"
            self.log_message(f"[COMPLETADO] Tiempo total de apilado: {time_str}")
            self._stack_start_time = None

        self.log_message(f"[COMPLETADO] Imagen guardada en: {path}")
        QMessageBox.information(
            self, "Éxito", 
            f"Apilado completado con éxito.\nGuardado en:\n{path}\n\nTransfiriendo al Revelador..."
        )
        self.stacking_finished.emit(path, self.computed_mask)

    def on_stack_error(self, err_msg):
        self._reset_run_button()
        self._stack_start_time = None
        self.progress_bar.setFormat("Error")
        self.log_message(f"[ERROR CRÍTICO] {err_msg}")
        QMessageBox.critical(self, "Error durante el apilado", f"Ocurrió un error:\n{err_msg}")

    def on_stack_cancelled(self):
        """Maneja la finalización tras una interrupción manual."""
        self._reset_run_button()
        self._stack_start_time = None
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Apilado Cancelado")
        self.log_message("[INFO] Proceso detenido con éxito. Recursos y memoria liberados.")  

    def _on_stack_lp_changed(self, val: int):
        self.lbl_lp_str.setText(f"{val}%")
        
    def on_progress_changed(self, value: int):
        self.progress_bar.setValue(value)
        
        if hasattr(self, "_stack_start_time") and value > 3:
            elapsed = time.perf_counter() - self._stack_start_time
            total_est = (elapsed / value) * 100.0
            remaining = max(0.0, total_est - elapsed)
            
            rem_m, rem_s = divmod(int(remaining), 60)
            if rem_m > 0:
                eta_str = f"ETA: ~{rem_m}m {rem_s:02d}s"
            else:
                eta_str = f"ETA: ~{rem_s}s"
                
            self.progress_bar.setFormat(f"%p%  ({eta_str})")
        else:
            self.progress_bar.setFormat("%p%")

    def on_new_session_clicked(self):
        """Pide confirmación y resetea la sesión en todo el software."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Aviso", "No puedes reiniciar la sesión mientras el apilado está en ejecución.")
            return

        resp = QMessageBox.question(
            self, "Nueva Sesión",
            "¿Deseas reiniciar toda la sesión actual?\n\n"
            "Se vaciarán las listas de Lights/Darks, la máscara dibujada "
            "y se restablecerá el revelador a su estado inicial.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        self.reset_session()
        self.new_session_requested.emit()
        self.log_message("[SESIÓN] Nueva sesión iniciada. Entorno restablecido.")
        
    def _set_ui_busy(self, busy: bool):
        """Bloquea o desbloquea controles secundarios durante el procesamiento."""
        self.grp_project.setEnabled(not busy)
        self.tabs_files.setEnabled(not busy)
        if hasattr(self, "grp_files"):
            self.grp_files.setEnabled(not busy)
        self.btn_add_files.setEnabled(not busy)
        self.btn_clear_files.setEnabled(not busy)
        self.grp_settings.setEnabled(not busy)        
        self.grp_stack_lp.setEnabled(not busy)
        self.grp_mask.setEnabled(not busy)

    def _on_btn_run_clicked(self):
        """Alterna entre iniciar el apilado/time-lapse o cancelarlo según el estado del worker."""
        if self.worker is not None and self.worker.isRunning():
            self.cancel_stacking()
        else:
            if hasattr(self, 'rb_out_video') and self.rb_out_video.isChecked():
                self.start_substacking()
            else:
                self.start_stacking()

    def start_substacking(self):
        active_lights = [p for p in self.lights_list if p not in self.excluded_lights]
        if self.current_ref_path and self.current_ref_path in active_lights:
            if active_lights[0] != self.current_ref_path:
                active_lights.remove(self.current_ref_path)
                active_lights.insert(0, self.current_ref_path)
        if len(active_lights) < 2:
            QMessageBox.warning(
                self, "Aviso",
                f"Añade al menos 2 tomas de luz (Lights) activas para generar el vídeo time-lapse.\n"
                f"(Activas: {len(active_lights)} / Total: {len(self.lights_list)})"
            )
            return

        w_val = self.slider_sub_w.value()
        if w_val % 2 == 0:
            w_val += 1

        app_cfg = load_config()
        first_dir = os.path.dirname(active_lights[0]) if active_lights else os.getcwd()
        default_video_dir = app_cfg.get("timelapse_dir", first_dir)
        default_video_file = os.path.join(default_video_dir, "timelapse_substack.mp4")

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Vídeo Time-Lapse Sub-Stacking", default_video_file, "Vídeo MP4 (*.mp4)"
        )
        if not out_path:
            return

        self._stack_start_time = time.perf_counter()

        self._set_ui_busy(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("CANCELAR TIME-LAPSE")
        self.btn_run.setIcon(get_icon("stop", size=16, color="#ffffff"))
        self.btn_run.setStyleSheet(STYLE_RUN_ACTIVE)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Iniciando Sub-Stacking... %p%")

        stride_choice = self.combo_sub_stride.currentData()
        stride = 1 if stride_choice == 1 else w_val

        save_frames = self.chk_sub_save_frames.isChecked()
        frames_dir = os.path.splitext(out_path)[0] + "_frames" if save_frames else None

        cfg = {
            "files": active_lights,
            "output_video_path": out_path,
            "window_size": w_val,
            "stride": stride,
            "weighting": self.combo_sub_weight.currentData() or "gaussian",
            "ground_mode": self.combo_sub_ground.currentData() or "temporal_average",
            "mask": self.computed_mask,
            "use_anti_ghosting": self.chk_anti_ghosting.isChecked(),
            "ag_radius": self.spin_ag_radius.value(),
            "ag_sensitivity": self.combo_ag_sens.currentData() or 0.5,
            "darks": self.darks_list,
            "flats": self.flats_list,
            "bias": self.bias_list,
            "video_fps": float(self.combo_sub_fps.currentText()),
            "video_resolution": self.combo_sub_res.currentData() or "1080p",
            "save_frames": save_frames,
            "frames_dir": frames_dir,
            "cpu_workers": int(app_cfg.get("cpu_workers", max(1, (os.cpu_count() or 4) - 1)))
        }

        self.worker = SubstackingWorker(cfg)
        self.worker.progress_changed.connect(self.on_progress_changed)
        self.worker.status_changed.connect(self.log_message)
        self.worker.frame_rendered.connect(self._on_sub_frame_rendered)
        self.worker.finished_success.connect(self.on_substack_success)
        self.worker.error_occurred.connect(self.on_stack_error)
        self.worker.cancelled.connect(self.on_stack_cancelled)
        self.worker.start()

    def _on_sub_frame_rendered(self, idx: int, frame_rgb: np.ndarray):
        """Actualiza la vista previa del canvas en tiempo real mientras se genera el vídeo."""
        try:
            self.canvas.set_image(frame_rgb)
        except Exception:
            pass

    def on_substack_success(self, video_path: str, stats: dict):
        self._reset_run_button()

        if hasattr(self, '_stack_start_time') and self._stack_start_time is not None:
            elapsed = time.perf_counter() - self._stack_start_time
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s" if mins > 0 else f"{secs:.2f}s"
            self.log_message(f"[COMPLETADO] Tiempo total de generación de vídeo: {time_str}")
            self._stack_start_time = None

        total_frames = stats.get("total_frames", 0)
        res = stats.get("resolution", "")
        fps = stats.get("fps", 24)

        self.log_message(f"[COMPLETADO] Vídeo Time-Lapse exportado: {video_path}")
        self.log_message(f"             Total fotogramas: {total_frames} | Res: {res} | FPS: {fps}")

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Vídeo Time-Lapse Generado")
        msg_box.setText(
            f"¡Vídeo Time-Lapse completado con éxito!\n\n"
            f"• Archivo: {os.path.basename(video_path)}\n"
            f"• Fotogramas: {total_frames}\n"
            f"• Resolución: {res} @ {fps} fps\n\n"
            f"Guardado en:\n{video_path}"
        )
        btn_open_folder = msg_box.addButton("Abrir Carpeta", QMessageBox.ActionRole)
        msg_box.addButton(QMessageBox.Ok)
        msg_box.exec()

        if msg_box.clickedButton() == btn_open_folder:
            import subprocess
            if os.name == 'nt':
                subprocess.Popen(['explorer', '/select,', os.path.normpath(video_path)])
            else:
                folder = os.path.dirname(os.path.abspath(video_path))
                subprocess.Popen(['xdg-open', folder])

    def cancel_stacking(self):
        """Solicita la detención inmediata del proceso."""
        if self.worker is not None and self.worker.isRunning():
            self.btn_run.setEnabled(False)
            self.btn_run.setText("CANCELANDO PROCESOS...")
            self.btn_run.setIcon(get_icon("stop", size=16, color="#ffffff"))
            self.btn_run.setStyleSheet(STYLE_RUN_CANCELLING)
            self.progress_bar.setFormat("Deteniendo... %p%")
            self.log_message("[AVISO] Solicitando parada de tareas y liberación de memoria...")
            self.worker.cancel()

    def _reset_run_button(self):
        """Devuelve el botón a su apariencia y comportamiento normal según el modo activo."""
        self.btn_run.setEnabled(True)
        if hasattr(self, 'rb_out_video') and self.rb_out_video.isChecked():
            self.btn_run.setText("GENERAR VÍDEO TIME-LAPSE (SUB-STACKING)")
            self.btn_run.setIcon(get_icon("video", size=16, color="#ffffff"))
        else:
            self.btn_run.setText("INICIAR APILADO DUAL")
            self.btn_run.setIcon(get_icon("play", size=16, color="#ffffff"))
        self.btn_run.setStyleSheet(STYLE_RUN_IDLE)
        self._set_ui_busy(False)
        
    def on_ground_double_clicked(self, item):
        row = self.list_ground.row(item)
        if 0 <= row < len(self.ground_list):
            path = self.ground_list[row]
            self.load_frame_to_canvas(path)
            self.log_message(f"[SUELO] Vista previa de suelo: {os.path.basename(path)}")