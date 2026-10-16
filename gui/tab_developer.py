# gui/tab_developer.py
import os
import gc
import time
import cv2
import numpy as np
import tifffile
from typing import Optional, Dict, Any, List
from PySide6.QtWidgets import (
    QApplication,
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QPushButton, 
    QLabel,
    QFileDialog, 
    QMessageBox, 
    QGroupBox, 
    QSlider, 
    QSplitter,
    QDoubleSpinBox, 
    QLineEdit,
    QPlainTextEdit,
    QTextEdit, 
    QComboBox, 
    QSpinBox,
    QScrollArea, 
    QFrame,
    QButtonGroup,
    QSizePolicy,
    QMenu,
    QProgressDialog
)
from PySide6.QtCore import Qt, QCoreApplication, QTimer
from PySide6.QtGui import QTextCursor, QShortcut, QKeySequence, QImage, QPixmap, QColor, QFont, QCursor
from core.stacking import load_image_as_float32
from core.stretch import (
    calculate_mtf_params, 
    manual_stretch, 
    compute_ghs_lut,
    apply_ghs_stretch,
    estimate_ghs_params,
    apply_white_balance, 
    adjust_saturation_dual, 
    extract_background_polynomial,
    adjust_vibrance, 
    adjust_contrast, 
    apply_light_pollution_gradient,
    apply_denoise,
    apply_clarity,
    apply_dehaze,
    apply_curve_lut,
    apply_hue_vs_sat_curve,
    apply_luma_vs_sat_curve,
    realce_multiescala_ondiculas
)
from core.masking import refine_mask_guided
from core.mask_layers import (
    MaskLayer, apply_all_mask_layers,
    MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS,
    MASK_TYPE_IMAGE_GROUND, MASK_TYPE_BRUSH, MASK_TYPE_LUMA, MASK_TYPE_COLOR,
    MASK_TYPE_GROUND, MASK_TYPE_SKY
)
from gui.widgets.curve_widget import CurveWidget
from gui.widgets.mask_layers_widget import MaskLayersWidget
from gui.widgets.collapsible_section import CollapsibleSection
from gui.canvas import MaskCanvas
from gui.worker import GraXpertWorker, StarNetWorker
from core.icc_manager import save_tiff_with_icc
from core.config_manager import load_config
from core.i18n import tr, get_i18n
from core.developer_presets import (
    DeveloperPreset, get_builtin_presets, load_user_presets,
    save_user_preset, delete_user_preset, filter_preset_params,
    SCOPE_ALL, SCOPE_TONE, SCOPE_COLOR, SCOPE_DETAILS
)
from gui.dialog_presets import SavePresetDialog, PresetManagerDialog
from gui.theme import (
    STYLE_RUN_IDLE, STYLE_TRANSFER_BTN, BG_PANEL, BORDER_BASE, TEXT_PRIMARY,
    TEXT_SECONDARY, TEXT_MUTED, TEXT_CYAN, ACCENT_CYAN
)


class DeveloperTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_32bit = None       # Imagen nativa base (float32)
        self.image_starless = None    # Capa sin estrellas (float32)
        self.image_stars = None       # Capa de solo estrellas (float32)

        # Buffers proxy para previsualización interactiva a 60 fps
        self.preview_proxy = None     
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None        

        self.current_mask = None      
        self.active_filepath = None
        self.gx_worker = None
        self.sn_worker = None
        
        self.is_zoomed_100 = False
        self.zoom_rel_coords = (0.5, 0.5)
        
        self.multiescala_estructura = 0.0   # -1.0 a +1.0
        self.multiescala_fondo = 0.0        # 0.0 a 1.0

        self.lp_reduction = 0.0  # Rango 0.0 a 1.0
        self.clarity_starless_val = 0.0
        self.dehaze_starless_val = 0.0
        
        # Parámetros del revelador
        self.temp_val = 0.0          # -1.000 a +1.000 (precisión milimétrica)
        self.tint_val = 0.0          # -1.000 a +1.000 (precisión milimétrica)
        self.sat_sky_val = 1.0       # 0.00x a 2.50x
        self.sat_gnd_val = 1.0       # 0.00x a 2.50x
        self.vibrance_val = 0.0      # -1.00 a +1.00
        self.contrast_val = 0.0      # -1.00 a +1.00
        
        # Parámetros de Tono del Suelo 
        self.gnd_ev_val = 0.0        # -2.00 EV a +3.00 EV
        self.gnd_shadows_val = 0.0   # 0.0 a 1.0 (0% a 100%)
        self.gnd_bp_val = 0.0        # -0.050 a +0.100
        
        self.star_intensity = 1.0    # 1.0 = 100%, 0.0 = Starless puro
        self.view_layer_mode = 0     # 0: Compuesta, 1: Solo Fondo, 2: Solo Estrellas
        self.contrast_starless_val = 0.0
        self.denoise_strength = 0.0  # <-- CORREGIDO: Inicialización obligatoria

        # Parámetros del comparador Antes / Después
        self.initial_bp: float = 0.0
        self.initial_mtf: float = 0.1
        self.has_custom_snapshot: bool = False
        self.snapshot_before_pixmap: Optional[QPixmap] = None

        # Parámetros de Estirado (MTF vs GHS)
        self.stretch_mode: str = "mtf"       # "mtf" o "ghs"
        self.ghs_color_mode: str = "luma"    # "luma" o "rgb"
        self.ghs_sp: float = 0.02000         # Punto de simetría (SP)
        self.ghs_d: float = 5.00             # Factor de estirado (D)
        self.ghs_b: float = 2.00             # Parámetro de forma (b)
        self.ghs_hp: float = 0.00            # Protección de altas luces / estrellas (HP)
        self.ghs_lp: float = 0.00            # Protección de sombras (LP)
        self.ghs_bp: float = 0.00000         # Punto negro para GHS (BP)
        self._eyedropper_target: str = "wb"  # "wb" o "ghs_sp"

        # Historial de Deshacer / Rehacer (Undo / Redo)
        self.MAX_UNDO_STEPS = 50
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self._last_committed_state: Optional[dict] = None
        self._is_undoing_or_redoing: bool = False
        self._undo_debounce_timer = QTimer(self)
        self._undo_debounce_timer.setSingleShot(True)
        self._undo_debounce_timer.setInterval(500)
        self._undo_debounce_timer.timeout.connect(self._commit_action)

        self._setup_ui()
        self._apply_styles()
        self._connect_undo_triggers()
        self._populate_presets_combo()

        # Conectar actualización reactiva de idioma
        get_i18n().language_changed.connect(lambda _: self.retranslate_ui())
        self.retranslate_ui()
        
    @staticmethod
    def _make_resettable_label(lbl: QLabel, reset_action, default_desc: str = "") -> QLabel:
        """Asigna comportamiento de doble clic para restablecer el control asociado."""
        lbl.setCursor(Qt.PointingHandCursor)
        tt = f"Doble clic para restablecer a {default_desc}" if default_desc else "Doble clic para restablecer"
        lbl.setToolTip(tt)
        lbl.mouseDoubleClickEvent = lambda e: reset_action()
        return lbl

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # 1. Visor interactivo con soporte Drag & Drop
        self.canvas = MaskCanvas(self, enable_masking=False)
        self.canvas.external_zoom_handler = True
        self.canvas.zoom_toggled.connect(self.on_canvas_zoom_toggled)
        self.canvas.files_dropped.connect(self._on_canvas_files_dropped)

        # 2. Panel interno de controles con Acordeón Plegable Inteligente
        left_container = QWidget()
        left_container_layout = QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(4, 4, 4, 4)
        left_container_layout.setSpacing(6)
        left_container.setMinimumWidth(440)
        left_container.setMaximumWidth(540)

        # Carga externa (Siempre fijada arriba)
        self.btn_open_img = QPushButton("📁 Abrir Imagen (TIFF / FITS / RAW)...")
        self.btn_open_img.setStyleSheet("font-weight: bold; color: #80d8ff; padding: 7px; background: #1c2030; border: 1px solid #2f3549; border-radius: 4px;")
        self.btn_open_img.setCursor(Qt.PointingHandCursor)
        self.btn_open_img.clicked.connect(self.open_image_file)
        left_container_layout.addWidget(self.btn_open_img, 0)

        # Scroll área para los acordeones
        self.left_scroll_area = QScrollArea()
        self.left_scroll_area.setWidgetResizable(True)
        self.left_scroll_area.setFrameShape(QFrame.NoFrame)
        self.left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        # --- SECCIÓN 1: Estilos y Presets de Revelado ---
        self.sec_presets = CollapsibleSection("🎨 Presets y Estilos de Revelado", is_expanded=False)
        w_presets = QWidget()
        presets_layout = QVBoxLayout(w_presets)
        presets_layout.setContentsMargins(6, 4, 6, 6)
        presets_layout.setSpacing(6)

        # Fila 1: Selector de Preset (ancho completo)
        self.combo_presets = QComboBox()
        self.combo_presets.setToolTip("Selecciona un estilo astronómico preestablecido oficial o personalizado")
        self.combo_presets.currentIndexChanged.connect(self._on_preset_combo_changed)
        presets_layout.addWidget(self.combo_presets)

        # Fila 2: Descripción corta del preset seleccionado
        self.lbl_preset_desc = QLabel("")
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color: #90a4ae; font-size: 11px; font-style: italic;")
        presets_layout.addWidget(self.lbl_preset_desc)

        # Fila 3: Modo de Aplicación + Botón Aplicar en la misma línea (ahorro vertical)
        row_mode_apply = QHBoxLayout()
        row_mode_apply.setSpacing(6)

        self.combo_preset_scope = QComboBox()
        self.combo_preset_scope.addItem("Todo (Sin estirado)", SCOPE_ALL)
        self.combo_preset_scope.addItem("Solo Color y WB", SCOPE_COLOR)
        self.combo_preset_scope.addItem("Solo Estrellas y Filtros", SCOPE_DETAILS)
        self.combo_preset_scope.addItem("Solo Contraste y Suelo", SCOPE_TONE)
        self.combo_preset_scope.setToolTip("Elige qué controles aplicar del preset (el estirado MTF y el punto negro de la imagen se respetan siempre)")
        row_mode_apply.addWidget(self.combo_preset_scope, stretch=1)

        self.btn_apply_preset = QPushButton("⚡ Aplicar")
        self.btn_apply_preset.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_apply_preset.setFixedWidth(84)
        self.btn_apply_preset.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 11px;
                background-color: #00838f;
                color: #ffffff;
                border: 1px solid #00acc1;
                border-radius: 5px;
                padding: 4px 6px;
            }
            QPushButton:hover {
                background-color: #0097a7;
                border-color: #26c6da;
            }
            QPushButton:pressed {
                background-color: #006064;
            }
            QPushButton:disabled {
                background-color: #152528;
                color: #43656b;
                border-color: #1f373b;
            }
        """)
        self.btn_apply_preset.setToolTip("Aplica el preset seleccionado a la imagen actual según el modo elegido")
        self.btn_apply_preset.clicked.connect(self.apply_selected_preset)
        row_mode_apply.addWidget(self.btn_apply_preset)
        presets_layout.addLayout(row_mode_apply)

        # Fila 4: Botones Guardar Actual y Administrar
        row_preset_btns = QHBoxLayout()
        self.btn_save_preset = QPushButton("💾 Guardar...")
        self.btn_save_preset.setToolTip("Guarda los ajustes y deslizadores actuales como un nuevo preset")
        self.btn_save_preset.clicked.connect(self.open_save_preset_dialog)
        row_preset_btns.addWidget(self.btn_save_preset)

        self.btn_manage_presets = QPushButton("⚙️ Gestor Presets...")
        self.btn_manage_presets.setToolTip("Abre el administrador para importar, exportar, inspeccionar o eliminar presets")
        self.btn_manage_presets.clicked.connect(self.open_preset_manager_dialog)
        row_preset_btns.addWidget(self.btn_manage_presets)
        presets_layout.addLayout(row_preset_btns)

        self.sec_presets.set_content_widget(w_presets)
        left_layout.addWidget(self.sec_presets)

        # --- SECCIÓN 2: Extracción de Fondo y Polución Lumínica (GraXpert AI) ---
        self.sec_gradients = CollapsibleSection(
            "⚡ Calibración y Gradientes (GraXpert AI)",
            is_expanded=False,
            on_reset=lambda: self.slider_lp.setValue(0)
        )
        w_bg = QWidget()
        layout_bg = QVBoxLayout(w_bg)
        layout_bg.setContentsMargins(6, 4, 6, 6)
        layout_bg.setSpacing(8)

        row_engine = QHBoxLayout()
        row_engine.addWidget(QLabel("Motor:"))
        self.combo_bg_engine = QComboBox()
        self.combo_bg_engine.addItems(["Polinómico (Paisaje)", "GraXpert AI"])
        row_engine.addWidget(self.combo_bg_engine)

        self.lbl_smoothing = QLabel("Suavizado:")
        row_engine.addWidget(self.lbl_smoothing)
        self.spin_smoothing = QDoubleSpinBox()
        self.spin_smoothing.setRange(0.1, 1.0)
        self.spin_smoothing.setSingleStep(0.05)
        self.spin_smoothing.setValue(0.50)
        self.spin_smoothing.setFixedWidth(65)
        row_engine.addWidget(self.spin_smoothing)
        layout_bg.addLayout(row_engine)

        self.btn_extract_bg = QPushButton("⚡ Eliminar Gradientes")
        self.btn_extract_bg.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_extract_bg.clicked.connect(self.on_extract_background_clicked)
        layout_bg.addWidget(self.btn_extract_bg)

        # Cúpula de luz (Contaminación Lumínica) integrada
        row_lp = QHBoxLayout()
        row_lp.addWidget(self._make_resettable_label(QLabel("Atenuar Cúpula de Luz:"), lambda: self.slider_lp.setValue(0), "0%"))
        self.lbl_lp = QLabel("0%")
        self.lbl_lp.setStyleSheet("color: #ffd54f; font-weight: bold;")
        self.lbl_lp.setMinimumWidth(38)
        row_lp.addWidget(self.lbl_lp)
        row_lp.addSpacing(6)
        self.slider_lp = QSlider(Qt.Horizontal)
        self.slider_lp.setRange(0, 100)
        self.slider_lp.setValue(0)
        self.slider_lp.valueChanged.connect(self.on_lp_changed)
        row_lp.addWidget(self.slider_lp, stretch=1)
        layout_bg.addLayout(row_lp)

        self.sec_gradients.set_content_widget(w_bg)
        left_layout.addWidget(self.sec_gradients)

        # --- SECCIÓN 3: Estirado Tonal y Contraste (MTF / GHS) ---
        self.sec_stretch = CollapsibleSection(
            "📈 Estirado Tonal y Contraste",
            is_expanded=True,
            on_reset=self.reset_sliders
        )
        w_stretch = QWidget()
        stretch_layout = QVBoxLayout(w_stretch)
        stretch_layout.setContentsMargins(6, 4, 6, 6)
        stretch_layout.setSpacing(8)

        # 3.1 Selector de Modo de Estirado: MTF vs GHS
        row_stretch_mode = QHBoxLayout()
        row_stretch_mode.setSpacing(4)

        self.btn_mode_mtf = QPushButton("📊 MTF Clásico")
        self.btn_mode_mtf.setCheckable(True)
        self.btn_mode_mtf.setChecked(True)
        self.btn_mode_mtf.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #80d8ff; border: 1px solid #80d8ff; }"
        )
        self.btn_mode_mtf.setToolTip("Estirado estándar por Función de Transferencia de Medios Tonos (MTF).")
        self.btn_mode_mtf.clicked.connect(lambda: self._set_stretch_mode("mtf"))

        self.btn_mode_ghs = QPushButton("📈 GHS Hiperbólico")
        self.btn_mode_ghs.setCheckable(True)
        self.btn_mode_ghs.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #ffb74d; border: 1px solid #ffb74d; }"
        )
        self.btn_mode_ghs.setToolTip("Estirado Hiperbólico Generalizado (Payne & Cranfield).\nControl quirúrgico de contraste en el fondo sin quemar estrellas ni núcleos.")
        self.btn_mode_ghs.clicked.connect(lambda: self._set_stretch_mode("ghs"))

        self.grp_stretch_mode = QButtonGroup(self)
        self.grp_stretch_mode.setExclusive(True)
        self.grp_stretch_mode.addButton(self.btn_mode_mtf)
        self.grp_stretch_mode.addButton(self.btn_mode_ghs)

        row_stretch_mode.addWidget(self.btn_mode_mtf)
        row_stretch_mode.addWidget(self.btn_mode_ghs)
        stretch_layout.addLayout(row_stretch_mode)

        # 3.2 Contenedor MTF
        self.w_mtf_container = QWidget()
        mtf_layout = QVBoxLayout(self.w_mtf_container)
        mtf_layout.setContentsMargins(0, 0, 0, 0)
        mtf_layout.setSpacing(6)

        btn_row_mtf = QHBoxLayout()
        btn_auto_mtf = QPushButton("⚡ Auto-Estirado MTF")
        btn_auto_mtf.setStyleSheet("font-weight: bold; color: #80d8ff;")
        btn_auto_mtf.clicked.connect(self.apply_auto_mtf)
        btn_reset_tone = QPushButton("Restablecer MTF")
        btn_reset_tone.clicked.connect(self.reset_sliders)
        btn_row_mtf.addWidget(btn_auto_mtf)
        btn_row_mtf.addWidget(btn_reset_tone)
        mtf_layout.addLayout(btn_row_mtf)

        # Emparejamiento: Punto Negro (izq) y Medios Tonos (der)
        row_spins = QHBoxLayout()
        row_spins.setSpacing(14)

        col_bp_lbl = QHBoxLayout()
        col_bp_lbl.addWidget(self._make_resettable_label(QLabel("P. Negro:"), lambda: self.spin_bp.setValue(getattr(self, 'initial_bp', 0.0)), "0.00000"))
        self.spin_bp = QDoubleSpinBox()
        self.spin_bp.setDecimals(5)
        self.spin_bp.setRange(0.0, 0.20000)
        self.spin_bp.setSingleStep(0.00010)
        self.spin_bp.setValue(0.0)
        self.spin_bp.valueChanged.connect(self.on_spin_bp_changed)
        col_bp_lbl.addWidget(self.spin_bp)
        row_spins.addLayout(col_bp_lbl)

        col_mtf_lbl = QHBoxLayout()
        col_mtf_lbl.addWidget(self._make_resettable_label(QLabel("Medios (m):"), lambda: self.spin_mtf.setValue(getattr(self, 'initial_mtf', 0.10000)), "0.10000"))
        self.spin_mtf = QDoubleSpinBox()
        self.spin_mtf.setDecimals(5)
        self.spin_mtf.setRange(0.00010, 0.50000)
        self.spin_mtf.setSingleStep(0.00050)
        self.spin_mtf.setValue(0.10000)
        self.spin_mtf.valueChanged.connect(self.on_spin_mtf_changed)
        col_mtf_lbl.addWidget(self.spin_mtf)
        row_spins.addLayout(col_mtf_lbl)
        mtf_layout.addLayout(row_spins)

        row_sliders = QHBoxLayout()
        row_sliders.setSpacing(14)
        self.slider_bp = QSlider(Qt.Horizontal)
        self.slider_bp.setRange(0, 2000)
        self.slider_bp.setValue(0)
        self.slider_bp.valueChanged.connect(self.on_slider_bp_changed)
        row_sliders.addWidget(self.slider_bp)

        self.slider_mtf = QSlider(Qt.Horizontal)
        self.slider_mtf.setRange(1, 2000)
        self.slider_mtf.setValue(400)
        self.slider_mtf.valueChanged.connect(self.on_slider_mtf_changed)
        row_sliders.addWidget(self.slider_mtf)
        mtf_layout.addLayout(row_sliders)

        stretch_layout.addWidget(self.w_mtf_container)

        # 3.3 Contenedor GHS (Hiperbólico Generalizado)
        self.w_ghs_container = QWidget()
        self.w_ghs_container.setVisible(False)
        ghs_layout = QVBoxLayout(self.w_ghs_container)
        ghs_layout.setContentsMargins(0, 0, 0, 0)
        ghs_layout.setSpacing(6)

        # Botonera superior GHS: Auto-GHS, Cuentagotas SP, Restablecer GHS
        btn_row_ghs = QHBoxLayout()
        self.btn_auto_ghs = QPushButton("⚡ Auto-GHS")
        self.btn_auto_ghs.setStyleSheet("font-weight: bold; color: #ffb74d;")
        self.btn_auto_ghs.clicked.connect(self.apply_auto_ghs)
        self.btn_auto_ghs.setToolTip("Calcula y aplica automáticamente los parámetros óptimos de GHS según el histograma")

        self.btn_ghs_pipette = QPushButton("🎯 Cuentagotas SP")
        self.btn_ghs_pipette.setCheckable(True)
        self.btn_ghs_pipette.setCursor(Qt.PointingHandCursor)
        self.btn_ghs_pipette.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_ghs_pipette.setToolTip("Haz clic aquí y luego pincha en el fondo de cielo o estructura del lienzo para centrar el Punto de Simetría (SP)")
        self.btn_ghs_pipette.toggled.connect(self._on_toggle_ghs_pipette)

        self.btn_reset_ghs = QPushButton("Restablecer")
        self.btn_reset_ghs.clicked.connect(self.reset_ghs)

        btn_row_ghs.addWidget(self.btn_auto_ghs)
        btn_row_ghs.addWidget(self.btn_ghs_pipette)
        btn_row_ghs.addWidget(self.btn_reset_ghs)
        ghs_layout.addLayout(btn_row_ghs)

        # Selector de Modo de Color GHS: Luminancia vs RGB
        row_ghs_color = QHBoxLayout()
        row_ghs_color.setSpacing(4)
        self.btn_ghs_luma = QPushButton("💡 Luminancia (Color)")
        self.btn_ghs_luma.setCheckable(True)
        self.btn_ghs_luma.setChecked(True)
        self.btn_ghs_luma.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px; font-weight: bold; border-radius: 3px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #80d8ff; border: 1px solid #80d8ff; }"
        )
        self.btn_ghs_luma.setToolTip("Preserva la saturación natural y matices estirando solo la luminancia Rec.709")
        self.btn_ghs_luma.clicked.connect(lambda: self._set_ghs_color_mode("luma"))

        self.btn_ghs_rgb = QPushButton("🎨 RGB Enlazado")
        self.btn_ghs_rgb.setCheckable(True)
        self.btn_ghs_rgb.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px; font-weight: bold; border-radius: 3px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #00e676; border: 1px solid #00e676; }"
        )
        self.btn_ghs_rgb.setToolTip("Aplica la curva GHS uniformemente sobre los canales R, G y B")
        self.btn_ghs_rgb.clicked.connect(lambda: self._set_ghs_color_mode("rgb"))

        self.grp_ghs_color = QButtonGroup(self)
        self.grp_ghs_color.setExclusive(True)
        self.grp_ghs_color.addButton(self.btn_ghs_luma)
        self.grp_ghs_color.addButton(self.btn_ghs_rgb)
        row_ghs_color.addWidget(self.btn_ghs_luma)
        row_ghs_color.addWidget(self.btn_ghs_rgb)
        ghs_layout.addLayout(row_ghs_color)

        # Fila SP: Punto de Simetría
        row_sp = QHBoxLayout()
        row_sp.addWidget(self._make_resettable_label(QLabel("P. Simetría (SP):"), lambda: self.spin_ghs_sp.setValue(0.02000), "0.02000"))
        self.spin_ghs_sp = QDoubleSpinBox()
        self.spin_ghs_sp.setDecimals(5)
        self.spin_ghs_sp.setRange(0.00000, 1.00000)
        self.spin_ghs_sp.setSingleStep(0.00050)
        self.spin_ghs_sp.setValue(0.02000)
        self.spin_ghs_sp.valueChanged.connect(self.on_spin_ghs_sp_changed)
        row_sp.addWidget(self.spin_ghs_sp)
        ghs_layout.addLayout(row_sp)

        self.slider_ghs_sp = QSlider(Qt.Horizontal)
        self.slider_ghs_sp.setRange(0, 10000)
        self.slider_ghs_sp.setValue(200)
        self.slider_ghs_sp.valueChanged.connect(self.on_slider_ghs_sp_changed)
        ghs_layout.addWidget(self.slider_ghs_sp)

        # Fila D y b: Factor de Estirado e Intensidad Local
        row_db_spins = QHBoxLayout()
        row_db_spins.setSpacing(10)

        col_d = QHBoxLayout()
        col_d.addWidget(self._make_resettable_label(QLabel("Factor D:"), lambda: self.spin_ghs_d.setValue(5.00), "5.00"))
        self.spin_ghs_d = QDoubleSpinBox()
        self.spin_ghs_d.setDecimals(2)
        self.spin_ghs_d.setRange(0.00, 20.00)
        self.spin_ghs_d.setSingleStep(0.10)
        self.spin_ghs_d.setValue(5.00)
        self.spin_ghs_d.valueChanged.connect(self.on_spin_ghs_d_changed)
        col_d.addWidget(self.spin_ghs_d)
        row_db_spins.addLayout(col_d)

        col_b = QHBoxLayout()
        col_b.addWidget(self._make_resettable_label(QLabel("Forma (b):"), lambda: self.spin_ghs_b.setValue(2.00), "2.00"))
        self.spin_ghs_b = QDoubleSpinBox()
        self.spin_ghs_b.setDecimals(2)
        self.spin_ghs_b.setRange(0.00, 15.00)
        self.spin_ghs_b.setSingleStep(0.10)
        self.spin_ghs_b.setValue(2.00)
        self.spin_ghs_b.valueChanged.connect(self.on_spin_ghs_b_changed)
        col_b.addWidget(self.spin_ghs_b)
        row_db_spins.addLayout(col_b)
        ghs_layout.addLayout(row_db_spins)

        row_db_sliders = QHBoxLayout()
        row_db_sliders.setSpacing(10)
        self.slider_ghs_d = QSlider(Qt.Horizontal)
        self.slider_ghs_d.setRange(0, 2000)
        self.slider_ghs_d.setValue(500)
        self.slider_ghs_d.valueChanged.connect(self.on_slider_ghs_d_changed)
        row_db_sliders.addWidget(self.slider_ghs_d)

        self.slider_ghs_b = QSlider(Qt.Horizontal)
        self.slider_ghs_b.setRange(0, 1500)
        self.slider_ghs_b.setValue(200)
        self.slider_ghs_b.valueChanged.connect(self.on_slider_ghs_b_changed)
        row_db_sliders.addWidget(self.slider_ghs_b)
        ghs_layout.addLayout(row_db_sliders)

        # Fila HP (Protección de Estrellas) y LP (Protección de Sombras)
        row_prot_spins = QHBoxLayout()
        row_prot_spins.setSpacing(10)

        col_hp = QHBoxLayout()
        col_hp.addWidget(self._make_resettable_label(QLabel("Prot. Estrellas (HP):"), lambda: self.spin_ghs_hp.setValue(0.00), "0.00"))
        self.spin_ghs_hp = QDoubleSpinBox()
        self.spin_ghs_hp.setDecimals(2)
        self.spin_ghs_hp.setRange(0.00, 1.00)
        self.spin_ghs_hp.setSingleStep(0.02)
        self.spin_ghs_hp.setValue(0.00)
        self.spin_ghs_hp.valueChanged.connect(self.on_spin_ghs_hp_changed)
        col_hp.addWidget(self.spin_ghs_hp)
        row_prot_spins.addLayout(col_hp)

        col_lp = QHBoxLayout()
        col_lp.addWidget(self._make_resettable_label(QLabel("Prot. Sombras (LP):"), lambda: self.spin_ghs_lp.setValue(0.00), "0.00"))
        self.spin_ghs_lp = QDoubleSpinBox()
        self.spin_ghs_lp.setDecimals(2)
        self.spin_ghs_lp.setRange(0.00, 1.00)
        self.spin_ghs_lp.setSingleStep(0.02)
        self.spin_ghs_lp.setValue(0.00)
        self.spin_ghs_lp.valueChanged.connect(self.on_spin_ghs_lp_changed)
        col_lp.addWidget(self.spin_ghs_lp)
        row_prot_spins.addLayout(col_lp)
        ghs_layout.addLayout(row_prot_spins)

        row_prot_sliders = QHBoxLayout()
        row_prot_sliders.setSpacing(10)
        self.slider_ghs_hp = QSlider(Qt.Horizontal)
        self.slider_ghs_hp.setRange(0, 1000)
        self.slider_ghs_hp.setValue(0)
        self.slider_ghs_hp.valueChanged.connect(self.on_slider_ghs_hp_changed)
        row_prot_sliders.addWidget(self.slider_ghs_hp)

        self.slider_ghs_lp = QSlider(Qt.Horizontal)
        self.slider_ghs_lp.setRange(0, 1000)
        self.slider_ghs_lp.setValue(0)
        self.slider_ghs_lp.valueChanged.connect(self.on_slider_ghs_lp_changed)
        row_prot_sliders.addWidget(self.slider_ghs_lp)
        ghs_layout.addLayout(row_prot_sliders)

        # Fila Punto Negro GHS (BP)
        row_ghs_bp = QHBoxLayout()
        row_ghs_bp.addWidget(self._make_resettable_label(QLabel("P. Negro GHS (BP):"), lambda: self.spin_ghs_bp.setValue(0.00000), "0.00000"))
        self.spin_ghs_bp = QDoubleSpinBox()
        self.spin_ghs_bp.setDecimals(5)
        self.spin_ghs_bp.setRange(0.00000, 0.20000)
        self.spin_ghs_bp.setSingleStep(0.00010)
        self.spin_ghs_bp.setValue(0.00000)
        self.spin_ghs_bp.valueChanged.connect(self.on_spin_ghs_bp_changed)
        row_ghs_bp.addWidget(self.spin_ghs_bp)
        ghs_layout.addLayout(row_ghs_bp)

        self.slider_ghs_bp = QSlider(Qt.Horizontal)
        self.slider_ghs_bp.setRange(0, 2000)
        self.slider_ghs_bp.setValue(0)
        self.slider_ghs_bp.valueChanged.connect(self.on_slider_ghs_bp_changed)
        ghs_layout.addWidget(self.slider_ghs_bp)

        stretch_layout.addWidget(self.w_ghs_container)

        # 3.4 Contraste General (común para MTF y GHS)
        row_cnt = QHBoxLayout()
        row_cnt.addWidget(self._make_resettable_label(QLabel("Contraste:"), lambda: self.slider_contrast.setValue(0), "0.00"))
        self.lbl_contrast = QLabel("0.00")
        self.lbl_contrast.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_cnt.addWidget(self.lbl_contrast)
        row_cnt.addSpacing(10)
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(-100, 100)
        self.slider_contrast.setValue(0)
        self.slider_contrast.valueChanged.connect(self.on_contrast_changed)
        row_cnt.addWidget(self.slider_contrast, stretch=1)
        stretch_layout.addLayout(row_cnt)

        self.sec_stretch.set_content_widget(w_stretch)
        left_layout.addWidget(self.sec_stretch)

        # --- SECCIÓN 4: Curva Tonal Interactiva (RGB / Saturación) ---
        self.sec_curves = CollapsibleSection(
            "📈 Curva Tonal Interactiva (RGB / Saturación)",
            is_expanded=False,
            on_reset=lambda: self.curve_widget.reset_all_curves()
        )
        w_curves = QWidget()
        curves_layout = QVBoxLayout(w_curves)
        curves_layout.setContentsMargins(6, 4, 6, 6)
        curves_layout.setSpacing(8)

        # Selector de Modo de Curva (RGB / Hue vs Sat / Luma vs Sat)
        row_curve_mode = QHBoxLayout()
        row_curve_mode.setSpacing(4)

        self.btn_curve_rgb = QPushButton("📈 Tonos (RGB)")
        self.btn_curve_rgb.setCheckable(True)
        self.btn_curve_rgb.setChecked(True)
        self.btn_curve_rgb.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #80d8ff; border: 1px solid #80d8ff; }"
        )
        self.btn_curve_rgb.setToolTip("Curva tonal RGB continua con histograma logarítmico y avisos de recorte.")
        self.btn_curve_rgb.clicked.connect(lambda: self._set_curve_mode("rgb"))

        self.btn_curve_hue = QPushButton("🌈 Tono vs Sat")
        self.btn_curve_hue.setCheckable(True)
        self.btn_curve_hue.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #ffb74d; border: 1px solid #ffb74d; }"
        )
        self.btn_curve_hue.setToolTip("Curva selectiva de saturación en función del tono (Hue vs Sat).\nPermite saturar selectivamente H-alfa / nebulosas o neutralizar tonalidades no deseadas.")
        self.btn_curve_hue.clicked.connect(lambda: self._set_curve_mode("hue_sat"))

        self.btn_curve_luma = QPushButton("💡 Luma vs Sat")
        self.btn_curve_luma.setCheckable(True)
        self.btn_curve_luma.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #00e676; border: 1px solid #00e676; }"
        )
        self.btn_curve_luma.setToolTip("Curva selectiva de saturación en función de la luminancia (Luma vs Sat).\nPermite desaturar sombras oscuras para eliminar ruido de crominancia del cielo.")
        self.btn_curve_luma.clicked.connect(lambda: self._set_curve_mode("luma_sat"))

        self.grp_curve_mode = QButtonGroup(self)
        self.grp_curve_mode.setExclusive(True)
        self.grp_curve_mode.addButton(self.btn_curve_rgb)
        self.grp_curve_mode.addButton(self.btn_curve_hue)
        self.grp_curve_mode.addButton(self.btn_curve_luma)

        row_curve_mode.addWidget(self.btn_curve_rgb)
        row_curve_mode.addWidget(self.btn_curve_hue)
        row_curve_mode.addWidget(self.btn_curve_luma)
        curves_layout.addLayout(row_curve_mode)

        # Curva de Tono con Histograma
        self.curve_widget = CurveWidget()
        self.curve_widget.curveChanged.connect(self.update_stretch_preview)
        self.curve_widget.clippingChanged.connect(self.on_clipping_changed_sync)
        curves_layout.addWidget(self.curve_widget)

        row_crv_btn = QHBoxLayout()
        row_crv_btn.setSpacing(6)
        self.btn_reset_curve = QPushButton("Resetear Curva Activa")
        self.btn_reset_curve.setStyleSheet("font-size: 11px; padding: 3px;")
        self.btn_reset_curve.setToolTip("Restablece únicamente la curva actualmente visible a su valor neutro.")
        self.btn_reset_curve.clicked.connect(lambda: self.curve_widget.reset_curve())

        self.btn_reset_all_curves = QPushButton("Resetear Todas")
        self.btn_reset_all_curves.setStyleSheet("font-size: 11px; padding: 3px; color: #ffab91;")
        self.btn_reset_all_curves.setToolTip("Restablece simultáneamente las tres curvas (Tonos, Tono vs Sat, Luma vs Sat).")
        self.btn_reset_all_curves.clicked.connect(self.curve_widget.reset_all_curves)

        row_crv_btn.addWidget(self.btn_reset_curve)
        row_crv_btn.addWidget(self.btn_reset_all_curves)
        curves_layout.addLayout(row_crv_btn)

        self.sec_curves.set_content_widget(w_curves)
        left_layout.addWidget(self.sec_curves)

        # --- SECCIÓN 5: StarNet++ AI — Desacoplo y Procesado de Capas ---
        self.sec_starnet = CollapsibleSection("⭐ StarNet++ AI — Desacoplo de Estrellas", is_expanded=False)
        w_starnet = QWidget()
        layout_starnet = QVBoxLayout(w_starnet)
        layout_starnet.setContentsMargins(6, 4, 6, 6)
        layout_starnet.setSpacing(8)

        # 1. Disparador de Ejecución
        row_sn = QHBoxLayout()
        row_sn.addWidget(QLabel("Paso (Stride):"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(64, 512)
        self.spin_stride.setSingleStep(64)
        self.spin_stride.setValue(256)
        self.spin_stride.setFixedWidth(65)
        row_sn.addWidget(self.spin_stride)

        self.btn_starnet = QPushButton("✨ Separar Estrellas con StarNet")
        self.btn_starnet.setStyleSheet("font-weight: bold; color: #b388ff;")
        self.btn_starnet.clicked.connect(self.run_starnet)
        row_sn.addWidget(self.btn_starnet)
        layout_starnet.addLayout(row_sn)

        # 2. Tarjeta informativa y acceso directo al Gestor de Capas
        self.card_starnet_info = QFrame()
        self.card_starnet_info.setStyleSheet("""
            QFrame {
                background-color: #1a1e2d;
                border: 1px solid #33384c;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout_card_sn = QVBoxLayout(self.card_starnet_info)
        layout_card_sn.setContentsMargins(8, 8, 8, 8)
        layout_card_sn.setSpacing(8)

        self.lbl_starnet_info = QLabel(
            "💡 Ejecuta StarNet para desacoplar las estrellas del fondo cósmico.\n"
            "Se generarán automáticamente las capas '🌌 Fondo Nebular' y '⭐ Estrellas' en el Gestor de Capas (Sección 9), donde podrás ajustar de forma contextual su presencia, claridad, contraste y reducción de ruido."
        )
        self.lbl_starnet_info.setWordWrap(True)
        self.lbl_starnet_info.setStyleSheet("color: #b0bec5; font-size: 11px; line-height: 1.4;")
        layout_card_sn.addWidget(self.lbl_starnet_info)

        self.btn_go_to_masks_starnet = QPushButton("🌌 Abrir Capas en Gestor de Capas (Sección 9) ➔")
        self.btn_go_to_masks_starnet.setCursor(Qt.PointingHandCursor)
        self.btn_go_to_masks_starnet.setStyleSheet("""
            QPushButton {
                background-color: #2b1f3d;
                color: #ce93d8;
                border: 1px solid #7b1fa2;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b2854;
                border-color: #ba68c8;
                color: #ffffff;
            }
        """)
        self.btn_go_to_masks_starnet.clicked.connect(lambda: self._jump_to_mask_layers(MASK_TYPE_IMAGE_STARLESS))
        self.btn_go_to_masks_starnet.setVisible(False)
        layout_card_sn.addWidget(self.btn_go_to_masks_starnet)

        layout_starnet.addWidget(self.card_starnet_info)

        # 3. Contenedor de herramientas dependientes en memoria (oculto de la UI para mantener compatibilidad con presets)
        self.container_starnet_tools = QWidget()
        self.container_starnet_tools.setVisible(False)
        tools_layout = QVBoxLayout(self.container_starnet_tools)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(8)

        # Capas y Estrellas
        row_view = QHBoxLayout()
        row_view.addWidget(QLabel("Capa visible:"))
        self.combo_layer = QComboBox()
        self.combo_layer.addItems(["Compuesta (Normal)", "Solo Fondo (Starless)", "Solo Estrellas"])
        self.combo_layer.currentIndexChanged.connect(self.on_layer_mode_changed)
        row_view.addWidget(self.combo_layer)

        row_view.addSpacing(10)
        row_view.addWidget(self._make_resettable_label(QLabel("Estrellas:"), lambda: self.slider_stars.setValue(100), "100%"))
        self.lbl_stars = QLabel("100%")
        self.lbl_stars.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_view.addWidget(self.lbl_stars)
        tools_layout.addLayout(row_view)

        self.slider_stars = QSlider(Qt.Horizontal)
        self.slider_stars.setRange(0, 200)
        self.slider_stars.setValue(100)
        self.slider_stars.valueChanged.connect(self.on_stars_slider_changed)
        tools_layout.addWidget(self.slider_stars)

        # Emparejamiento 3.2 - Par 1: Claridad Fondo (izq) y Borrar Neblina (der)
        row_pair1 = QHBoxLayout()
        row_pair1.setSpacing(14)

        col_clarity = QVBoxLayout()
        col_clarity.setSpacing(4)
        row_cl_lbl = QHBoxLayout()
        row_cl_lbl.addWidget(self._make_resettable_label(QLabel("Claridad Fondo:"), lambda: self.slider_clarity.setValue(0), "0.00"))
        self.lbl_clarity = QLabel("0.00")
        self.lbl_clarity.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_cl_lbl.addWidget(self.lbl_clarity)
        col_clarity.addLayout(row_cl_lbl)
        self.slider_clarity = QSlider(Qt.Horizontal)
        self.slider_clarity.setRange(-200, 200)
        self.slider_clarity.setValue(0)
        self.slider_clarity.valueChanged.connect(self.on_clarity_changed)
        col_clarity.addWidget(self.slider_clarity)
        row_pair1.addLayout(col_clarity)

        col_dehaze = QVBoxLayout()
        col_dehaze.setSpacing(4)
        row_dh_lbl = QHBoxLayout()
        row_dh_lbl.addWidget(self._make_resettable_label(QLabel("Borrar Neblina:"), lambda: self.slider_dehaze.setValue(0), "0.00"))
        self.lbl_dehaze = QLabel("0.00")
        self.lbl_dehaze.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_dh_lbl.addWidget(self.lbl_dehaze)
        col_dehaze.addLayout(row_dh_lbl)
        self.slider_dehaze = QSlider(Qt.Horizontal)
        self.slider_dehaze.setRange(-200, 200)
        self.slider_dehaze.setValue(0)
        self.slider_dehaze.valueChanged.connect(self.on_dehaze_changed)
        col_dehaze.addWidget(self.slider_dehaze)
        row_pair1.addLayout(col_dehaze)
        tools_layout.addLayout(row_pair1)

        # Emparejamiento 3.2 - Par 2: Estructura Ondículas (izq) y Atenuar Fondo (der)
        row_pair2 = QHBoxLayout()
        row_pair2.setSpacing(14)

        col_ond = QVBoxLayout()
        col_ond.setSpacing(4)
        row_ond_lbl = QHBoxLayout()
        row_ond_lbl.addWidget(self._make_resettable_label(QLabel("Estructura (Ondículas):"), lambda: self.slider_ond_est.setValue(0), "0.00"))
        self.lbl_ond_est = QLabel("0.00")
        self.lbl_ond_est.setStyleSheet("color: #b388ff; font-weight: bold;")
        row_ond_lbl.addWidget(self.lbl_ond_est)
        col_ond.addLayout(row_ond_lbl)
        self.slider_ond_est = QSlider(Qt.Horizontal)
        self.slider_ond_est.setRange(-100, 100)
        self.slider_ond_est.setValue(0)
        self.slider_ond_est.valueChanged.connect(self.on_ond_estructura_changed)
        col_ond.addWidget(self.slider_ond_est)
        row_pair2.addLayout(col_ond)

        col_ond_bg = QVBoxLayout()
        col_ond_bg.setSpacing(4)
        row_ondbg_lbl = QHBoxLayout()
        row_ondbg_lbl.addWidget(self._make_resettable_label(QLabel("Atenuar Fondo:"), lambda: self.slider_ond_bg.setValue(0), "0%"))
        self.lbl_ond_bg = QLabel("0%")
        self.lbl_ond_bg.setStyleSheet("color: #b388ff; font-weight: bold;")
        row_ondbg_lbl.addWidget(self.lbl_ond_bg)
        col_ond_bg.addLayout(row_ondbg_lbl)
        self.slider_ond_bg = QSlider(Qt.Horizontal)
        self.slider_ond_bg.setRange(0, 100)
        self.slider_ond_bg.setValue(0)
        self.slider_ond_bg.valueChanged.connect(self.on_ond_fondo_changed)
        col_ond_bg.addWidget(self.slider_ond_bg)
        row_pair2.addLayout(col_ond_bg)
        tools_layout.addLayout(row_pair2)

        # Emparejamiento 3.2 - Par 3: Contraste Starless (izq) y Reducción de Ruido (der)
        row_pair3 = QHBoxLayout()
        row_pair3.setSpacing(14)

        col_cs = QVBoxLayout()
        col_cs.setSpacing(4)
        row_cs_lbl = QHBoxLayout()
        row_cs_lbl.addWidget(self._make_resettable_label(QLabel("Contraste Fondo:"), lambda: self.slider_contrast_starless.setValue(0), "0.00"))
        self.lbl_contrast_starless = QLabel("0.00")
        self.lbl_contrast_starless.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_cs_lbl.addWidget(self.lbl_contrast_starless)
        col_cs.addLayout(row_cs_lbl)
        self.slider_contrast_starless = QSlider(Qt.Horizontal)
        self.slider_contrast_starless.setRange(-100, 100)
        self.slider_contrast_starless.setValue(0)
        self.slider_contrast_starless.valueChanged.connect(self.on_contrast_starless_changed)
        col_cs.addWidget(self.slider_contrast_starless)
        row_pair3.addLayout(col_cs)

        col_dn = QVBoxLayout()
        col_dn.setSpacing(4)
        row_dn_lbl = QHBoxLayout()
        row_dn_lbl.addWidget(self._make_resettable_label(QLabel("Reducción Ruido:"), lambda: self.slider_denoise.setValue(0), "0%"))
        self.lbl_denoise = QLabel("0%")
        self.lbl_denoise.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_dn_lbl.addWidget(self.lbl_denoise)
        col_dn.addLayout(row_dn_lbl)
        self.slider_denoise = QSlider(Qt.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(0)
        self.slider_denoise.valueChanged.connect(self.on_denoise_changed)
        col_dn.addWidget(self.slider_denoise)
        row_pair3.addLayout(col_dn)
        tools_layout.addLayout(row_pair3)

        self.sec_starnet.set_content_widget(w_starnet)
        left_layout.addWidget(self.sec_starnet)
        self._set_starnet_tools_enabled(False)

        # --- SECCIÓN 6: Balance de Blancos (Precisión Fina) ---
        self.sec_wb = CollapsibleSection(
            "🌡️ Balance de Blancos y Calibración",
            is_expanded=True,
            on_reset=self.reset_wb
        )
        w_wb = QWidget()
        wb_layout = QVBoxLayout(w_wb)
        wb_layout.setContentsMargins(6, 4, 6, 6)
        wb_layout.setSpacing(8)

        # Emparejamiento 3.2: Temp (izq) y Tinte (der) en 2 columnas
        row_wb = QHBoxLayout()
        row_wb.setSpacing(14)

        col_temp = QVBoxLayout()
        col_temp.setSpacing(4)
        row_temp_lbl = QHBoxLayout()
        lbl_temp_title = QLabel("Temp:")
        lbl_temp_title.setToolTip("Doble clic para restablecer Temperatura a 0.000")
        lbl_temp_title.setCursor(Qt.PointingHandCursor)
        row_temp_lbl.addWidget(lbl_temp_title)

        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setDecimals(3)
        self.spin_temp.setRange(-1.000, 1.000)
        self.spin_temp.setSingleStep(0.010)
        self.spin_temp.setValue(0.000)
        self.spin_temp.setFixedWidth(82)
        self.spin_temp.setStyleSheet("color: #ffd54f; font-weight: bold;")
        self.spin_temp.setToolTip("Ajuste numérico manual de Temperatura (-1.000 a +1.000). Doble clic en 'Temp:' para reiniciar")
        self.spin_temp.valueChanged.connect(self.on_spin_temp_changed)
        row_temp_lbl.addWidget(self.spin_temp)
        col_temp.addLayout(row_temp_lbl)

        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(-1000, 1000)
        self.slider_temp.setValue(0)
        self.slider_temp.setToolTip("Temperatura de color (-1.000 azul/frío a +1.000 rojo/cálido). Doble clic en 'Temp:' para reiniciar")
        self.slider_temp.valueChanged.connect(self.on_temp_changed)
        lbl_temp_title.mouseDoubleClickEvent = lambda e: self.spin_temp.setValue(0.0)
        col_temp.addWidget(self.slider_temp)
        row_wb.addLayout(col_temp)

        col_tint = QVBoxLayout()
        col_tint.setSpacing(4)
        row_tint_lbl = QHBoxLayout()
        lbl_tint_title = QLabel("Tinte:")
        lbl_tint_title.setToolTip("Doble clic para restablecer Tinte a 0.000")
        lbl_tint_title.setCursor(Qt.PointingHandCursor)
        row_tint_lbl.addWidget(lbl_tint_title)

        self.spin_tint = QDoubleSpinBox()
        self.spin_tint.setDecimals(3)
        self.spin_tint.setRange(-1.000, 1.000)
        self.spin_tint.setSingleStep(0.010)
        self.spin_tint.setValue(0.000)
        self.spin_tint.setFixedWidth(82)
        self.spin_tint.setStyleSheet("color: #b388ff; font-weight: bold;")
        self.spin_tint.setToolTip("Ajuste numérico manual de Tinte (-1.000 a +1.000). Doble clic en 'Tinte:' para reiniciar")
        self.spin_tint.valueChanged.connect(self.on_spin_tint_changed)
        row_tint_lbl.addWidget(self.spin_tint)
        col_tint.addLayout(row_tint_lbl)

        self.slider_tint = QSlider(Qt.Horizontal)
        self.slider_tint.setRange(-1000, 1000)
        self.slider_tint.setValue(0)
        self.slider_tint.setToolTip("Tinte de color (-1.000 verde a +1.000 magenta). Doble clic en 'Tinte:' para reiniciar")
        self.slider_tint.valueChanged.connect(self.on_tint_changed)
        lbl_tint_title.mouseDoubleClickEvent = lambda e: self.spin_tint.setValue(0.0)
        col_tint.addWidget(self.slider_tint)
        row_wb.addLayout(col_tint)

        wb_layout.addLayout(row_wb)

        row_wb_btn = QHBoxLayout()
        self.btn_panel_pipette = QPushButton("🧪 Pipeta WB (W)")
        self.btn_panel_pipette.setCheckable(True)
        self.btn_panel_pipette.setCursor(Qt.PointingHandCursor)
        self.btn_panel_pipette.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_panel_pipette.setToolTip("Activa la pipeta para calibrar el balance de blancos haciendo clic sobre un fondo neutro en la imagen (Atajo: W)")
        self.btn_panel_pipette.toggled.connect(self.canvas.set_eyedropper_active)
        row_wb_btn.addWidget(self.btn_panel_pipette)
        self.btn_pipette = self.btn_panel_pipette

        btn_reset_wb = QPushButton("Restablecer")
        btn_reset_wb.clicked.connect(self.reset_wb)
        row_wb_btn.addWidget(btn_reset_wb)
        wb_layout.addLayout(row_wb_btn)

        self.sec_wb.set_content_widget(w_wb)
        left_layout.addWidget(self.sec_wb)

        # --- SECCIÓN 7: Color: Saturación e Intensidad ---
        self.sec_color = CollapsibleSection(
            "🎨 Color y Saturación",
            is_expanded=True,
            on_reset=self.reset_saturation
        )
        w_sat = QWidget()
        sat_layout = QVBoxLayout(w_sat)
        sat_layout.setContentsMargins(6, 4, 6, 6)
        sat_layout.setSpacing(8)

        # Emparejamiento 3.2: Saturación Cielo (izq) y Suelo (der) en 2 columnas
        row_sat_dual = QHBoxLayout()
        row_sat_dual.setSpacing(14)

        col_sky = QVBoxLayout()
        col_sky.setSpacing(4)
        row_sky_lbl = QHBoxLayout()
        row_sky_lbl.addWidget(self._make_resettable_label(QLabel("Sat. Cielo:"), lambda: self.slider_sat_sky.setValue(100), "1.00x"))
        self.lbl_sat_sky = QLabel("1.00x")
        self.lbl_sat_sky.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_sky_lbl.addWidget(self.lbl_sat_sky)
        col_sky.addLayout(row_sky_lbl)

        self.slider_sat_sky = QSlider(Qt.Horizontal)
        self.slider_sat_sky.setRange(0, 250)
        self.slider_sat_sky.setValue(100)
        self.slider_sat_sky.valueChanged.connect(self.on_sat_sky_changed)
        col_sky.addWidget(self.slider_sat_sky)
        row_sat_dual.addLayout(col_sky)

        col_gnd = QVBoxLayout()
        col_gnd.setSpacing(4)
        row_gnd_lbl = QHBoxLayout()
        row_gnd_lbl.addWidget(self._make_resettable_label(QLabel("Sat. Suelo:"), lambda: self.slider_sat_gnd.setValue(100), "1.00x"))
        self.lbl_sat_gnd = QLabel("1.00x")
        self.lbl_sat_gnd.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_gnd_lbl.addWidget(self.lbl_sat_gnd)
        col_gnd.addLayout(row_gnd_lbl)

        self.slider_sat_gnd = QSlider(Qt.Horizontal)
        self.slider_sat_gnd.setRange(0, 250)
        self.slider_sat_gnd.setValue(100)
        self.slider_sat_gnd.valueChanged.connect(self.on_sat_gnd_changed)
        col_gnd.addWidget(self.slider_sat_gnd)
        row_sat_dual.addLayout(col_gnd)

        sat_layout.addLayout(row_sat_dual)

        # Vibrance y Reset en una fila compacta
        row_vib = QHBoxLayout()
        row_vib.addWidget(self._make_resettable_label(QLabel("Vibranza:"), lambda: self.slider_vibrance.setValue(0), "0.00"))
        self.lbl_vibrance = QLabel("0.00")
        self.lbl_vibrance.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_vib.addWidget(self.lbl_vibrance)
        row_vib.addSpacing(6)
        self.slider_vibrance = QSlider(Qt.Horizontal)
        self.slider_vibrance.setRange(-100, 100)
        self.slider_vibrance.setValue(0)
        self.slider_vibrance.valueChanged.connect(self.on_vibrance_changed)
        row_vib.addWidget(self.slider_vibrance, stretch=1)

        btn_reset_sat = QPushButton("Restablecer")
        btn_reset_sat.setToolTip("Restablecer Saturación y Vibranza")
        btn_reset_sat.clicked.connect(self.reset_saturation)
        row_vib.addWidget(btn_reset_sat)
        sat_layout.addLayout(row_vib)

        self.sec_color.set_content_widget(w_sat)
        left_layout.addWidget(self.sec_color)

        # --- SECCIÓN 8: Ajuste Tonal Suelo (Paisaje) ---
        self.sec_gnd_tone = CollapsibleSection(
            "🏞️ Ajuste Tonal Suelo (Paisaje)",
            is_expanded=False,
            on_reset=self.reset_gnd_tone
        )
        w_gnd_tone = QWidget()
        gnd_tone_layout = QVBoxLayout(w_gnd_tone)
        gnd_tone_layout.setContentsMargins(6, 4, 6, 6)
        gnd_tone_layout.setSpacing(8)

        # Acceso y estado de la Máscara de Suelo
        row_mask_status = QHBoxLayout()
        self.lbl_gnd_mask_status = QLabel("⚠️ Sin máscara de suelo activa")
        self.lbl_gnd_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
        row_mask_status.addWidget(self.lbl_gnd_mask_status, stretch=1)

        self.btn_edit_ground_mask = QPushButton("🖌️ Crear / Editar Máscara...")
        self.btn_edit_ground_mask.setCursor(Qt.PointingHandCursor)
        self.btn_edit_ground_mask.setToolTip("Abre la herramienta de trazado interactivo de máscara en el visor (Atajo: M)")
        self.btn_edit_ground_mask.clicked.connect(self._open_mask_tool_from_section)
        row_mask_status.addWidget(self.btn_edit_ground_mask)
        gnd_tone_layout.addLayout(row_mask_status)

        # Tarjeta informativa y acceso directo a la Capa de Suelo
        self.card_gnd_info = QFrame()
        self.card_gnd_info.setStyleSheet("""
            QFrame {
                background-color: #152219;
                border: 1px solid #2e7d32;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout_card_gnd = QVBoxLayout(self.card_gnd_info)
        layout_card_gnd.setContentsMargins(8, 8, 8, 8)
        layout_card_gnd.setSpacing(8)

        self.lbl_gnd_info = QLabel(
            "💡 Los ajustes tonales de la superficie terrestre (Exposición, Sombras, Punto Negro y Balance de Blancos dedicado) "
            "se gestionan ahora de forma centralizada e intuitiva en la Capa de Suelo dentro del Gestor de Capas (Sección 9)."
        )
        self.lbl_gnd_info.setWordWrap(True)
        self.lbl_gnd_info.setStyleSheet("color: #c8e6c9; font-size: 11px; line-height: 1.4;")
        layout_card_gnd.addWidget(self.lbl_gnd_info)

        self.btn_go_to_ground_layer = QPushButton("🏞️ Abrir Capa de Suelo en Sección 9 ➔")
        self.btn_go_to_ground_layer.setCursor(Qt.PointingHandCursor)
        self.btn_go_to_ground_layer.setStyleSheet("""
            QPushButton {
                background-color: #1b3822;
                color: #a5d6a7;
                border: 1px solid #43a047;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #254d2f;
                border-color: #81c784;
                color: #ffffff;
            }
        """)
        self.btn_go_to_ground_layer.clicked.connect(lambda: self._jump_to_mask_layers(MASK_TYPE_IMAGE_GROUND))
        layout_card_gnd.addWidget(self.btn_go_to_ground_layer)

        gnd_tone_layout.addWidget(self.card_gnd_info)

        # Controles en memoria (ocultos de la interfaz para mantener compatibilidad con presets)
        self._hidden_gnd_container = QWidget()
        self._hidden_gnd_container.setVisible(False)
        hidden_gnd_layout = QVBoxLayout(self._hidden_gnd_container)

        self.lbl_gnd_ev = QLabel("0.00 EV")
        self.slider_gnd_ev = QSlider(Qt.Horizontal)
        self.slider_gnd_ev.setRange(-200, 300)
        self.slider_gnd_ev.setValue(0)
        self.slider_gnd_ev.valueChanged.connect(self.on_gnd_ev_changed)
        hidden_gnd_layout.addWidget(self.slider_gnd_ev)

        self.lbl_gnd_sh = QLabel("0%")
        self.slider_gnd_sh = QSlider(Qt.Horizontal)
        self.slider_gnd_sh.setRange(0, 100)
        self.slider_gnd_sh.setValue(0)
        self.slider_gnd_sh.valueChanged.connect(self.on_gnd_shadows_changed)
        hidden_gnd_layout.addWidget(self.slider_gnd_sh)

        self.lbl_gnd_bp = QLabel("0.000")
        self.slider_gnd_bp = QSlider(Qt.Horizontal)
        self.slider_gnd_bp.setRange(-50, 100)
        self.slider_gnd_bp.setValue(0)
        self.slider_gnd_bp.valueChanged.connect(self.on_gnd_bp_changed)
        hidden_gnd_layout.addWidget(self.slider_gnd_bp)

        btn_reset_gnd = QPushButton("Restablecer Tono Suelo")
        btn_reset_gnd.clicked.connect(self.reset_gnd_tone)
        hidden_gnd_layout.addWidget(btn_reset_gnd)

        self.sec_gnd_tone.set_content_widget(w_gnd_tone)
        left_layout.addWidget(self.sec_gnd_tone)

        # --- SECCIÓN 9: Máscaras y Capas de Ajuste Local ---
        self.sec_masks = CollapsibleSection("🎭 Máscaras y Capas de Ajuste Local", is_expanded=False)
        self.mask_layers_widget = MaskLayersWidget()
        self.mask_layers_widget.set_getters(
            img_getter=lambda: self.preview_proxy if self.preview_proxy is not None else self.image_32bit,
            mask_getter=lambda: self.current_mask,
            stars_getter=lambda: self.proxy_stars if self.proxy_stars is not None else self.image_stars,
            starless_getter=lambda: self.proxy_starless if self.proxy_starless is not None else self.image_starless
        )
        self.mask_layers_widget.layers_changed.connect(self.update_stretch_preview)
        self.mask_layers_widget.preview_mask_requested.connect(self._on_layer_mask_preview_requested)
        self.mask_layers_widget.edit_ground_mask_requested.connect(self._open_mask_tool_from_section)
        self.mask_layers_widget.ai_ground_mask_requested.connect(self._detect_mask_ai_from_toolbar)
        self.mask_layers_widget.brush_tool_toggled.connect(self._on_layer_brush_toggled)
        self.mask_layers_widget.brush_params_changed.connect(self._on_layer_brush_params_changed)
        self.mask_layers_widget.starnet_opacity_changed.connect(self._on_mask_layer_starnet_opacity)
        self.mask_layers_widget.starnet_visibility_changed.connect(self._on_mask_layer_starnet_visibility)
        self.mask_layers_widget.ground_ev_changed.connect(self._on_mask_widget_gnd_ev)
        self.mask_layers_widget.ground_shadows_changed.connect(self._on_mask_widget_gnd_sh)
        self.mask_layers_widget.ground_bp_changed.connect(self._on_mask_widget_gnd_bp)
        self.mask_layers_widget.ground_reset_requested.connect(self.reset_gnd_tone)
        self.mask_layers_widget.ground_slider_released.connect(self._commit_action)
        self.mask_layers_widget.set_ground_tone_values(
            self.slider_gnd_ev.value(),
            self.slider_gnd_sh.value(),
            self.slider_gnd_bp.value()
        )

        # Sincronización contextual de Estrellas y Fondo Nebular (Starless)
        self.mask_layers_widget.stars_intensity_changed.connect(self._on_mask_widget_stars_intensity)
        self.mask_layers_widget.stars_reset_requested.connect(self.reset_stars_intensity)
        self.mask_layers_widget.starless_clarity_changed.connect(self._on_mask_widget_sl_clarity)
        self.mask_layers_widget.starless_dehaze_changed.connect(self._on_mask_widget_sl_dehaze)
        self.mask_layers_widget.starless_ond_est_changed.connect(self._on_mask_widget_sl_ond_est)
        self.mask_layers_widget.starless_ond_bg_changed.connect(self._on_mask_widget_sl_ond_bg)
        self.mask_layers_widget.starless_contrast_changed.connect(self._on_mask_widget_sl_contrast)
        self.mask_layers_widget.starless_denoise_changed.connect(self._on_mask_widget_sl_denoise)
        self.mask_layers_widget.starless_reset_requested.connect(self.reset_starless_adjustments)

        self.mask_layers_widget.set_stars_special_value(self.slider_stars.value())
        self.mask_layers_widget.set_starless_tone_values(
            self.slider_clarity.value(),
            self.slider_dehaze.value(),
            self.slider_ond_est.value(),
            self.slider_ond_bg.value(),
            self.slider_contrast_starless.value(),
            self.slider_denoise.value()
        )

        def _update_masks_badge():
            count = len(self.mask_layers_widget.layers)
            if count > 0:
                self.sec_masks.set_status_badge(f"● {count} {'capas' if count > 1 else 'capa'}", "#80d8ff")
            else:
                self.sec_masks.set_status_badge("")
        self.mask_layers_widget.layers_changed.connect(_update_masks_badge)

        self.sec_masks.set_content_widget(self.mask_layers_widget)
        left_layout.addWidget(self.sec_masks)

        left_layout.addStretch(1)

        # Asignar panel de acordeones al área de scroll
        self.left_scroll_area.setWidget(left_panel)
        left_container_layout.addWidget(self.left_scroll_area, 1)

        # Botón Exportar siempre fijado al pie del panel izquierdo
        self.btn_export = QPushButton("💾 EXPORTAR IMAGEN REVELADA...")
        self.btn_export.setFixedHeight(40)
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_export.clicked.connect(self.export_image)
        left_container_layout.addWidget(self.btn_export, 0)

        self.btn_export_psd = QPushButton("📑 EXPORTAR A PHOTOSHOP (PSD)...")
        self.btn_export_psd.setFixedHeight(34)
        self.btn_export_psd.setCursor(Qt.PointingHandCursor)
        self.btn_export_psd.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a237e, stop:1 #283593);
                color: #82b1ff;
                font-weight: bold;
                border: 1px solid #3d5afe;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #283593, stop:1 #3949ab);
                color: #ffffff;
                border-color: #536dfe;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        self.btn_export_psd.setToolTip(
            "Exporta el proyecto como archivo nativo de Adobe Photoshop (.psd) de 16 bits.\n"
            "Incluye por separado las capas de Imagen Base, Fondo Starless, Estrellas (Add), Suelo y Máscara de Horizonte."
        )
        self.btn_export_psd.clicked.connect(self.export_psd_dialog)
        left_container_layout.addWidget(self.btn_export_psd, 0)

        # Registro de Actividad compacto
        self.lbl_log_title = QLabel(tr("developer.activity_log", "Registro de Actividad:"))
        self.lbl_log_title.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold;")
        left_container_layout.addWidget(self.lbl_log_title, 0)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(85)
        left_container_layout.addWidget(self.txt_log, 0)

        splitter.addWidget(left_container)

        # Contenedor derecho: Barra de herramientas superior + Canvas
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.canvas_toolbar = self._create_canvas_toolbar()
        right_layout.addWidget(self.canvas_toolbar)
        self.crop_toolbar = self._create_crop_toolbar()
        self.crop_toolbar.setVisible(False)
        right_layout.addWidget(self.crop_toolbar)
        self.mask_toolbar = self._create_mask_toolbar()
        self.mask_toolbar.setVisible(False)
        right_layout.addWidget(self.mask_toolbar)
        right_layout.addWidget(self.canvas, 1)

        # Sincronización bidireccional de Pipeta, Lupa, Recorte y Pincel con la barra de herramientas
        self.canvas.eyedropper_toggled.connect(self._on_canvas_eyedropper_toggled)
        self.canvas.magnifier_toggled.connect(self.btn_magnifier.setChecked)
        self.canvas.crop_toggled.connect(self.btn_crop.setChecked)
        self.canvas.crop_rect_changed.connect(self._on_crop_rect_changed)
        self.canvas.crop_applied.connect(self._apply_crop)
        self.canvas.color_sampled.connect(self._on_eyedropper_color_sampled)
        self.canvas.brush_size_changed.connect(self._on_brush_size_changed)
        self.canvas.layer_brush_stroke_finished.connect(self._on_layer_brush_stroke_finished)
        self.canvas.layer_brush_radius_changed.connect(self.mask_layers_widget.set_brush_radius_from_canvas)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        splitter.setSizes([520, 1400])
        
        # Atajo global 'M' para alternar herramienta interactiva de Máscara de Suelo
        self.shortcut_mask = QShortcut(QKeySequence(Qt.Key_M), self)
        self.shortcut_mask.setContext(Qt.WindowShortcut)
        self.shortcut_mask.activated.connect(self._on_toggle_mask_shortcut)

        # Atajo global 'C' para alternar herramienta interactiva de Recorte
        self.shortcut_crop = QShortcut(QKeySequence(Qt.Key_C), self)
        self.shortcut_crop.setContext(Qt.WindowShortcut)
        self.shortcut_crop.activated.connect(self._on_toggle_crop_shortcut)

        # Atajo global 'J' para alternar indicadores de recorte (Lightroom / Camera Raw style)
        self.shortcut_clipping = QShortcut(QKeySequence(Qt.Key_J), self)
        self.shortcut_clipping.setContext(Qt.WindowShortcut)
        self.shortcut_clipping.activated.connect(self._on_toggle_clipping_shortcut)

        # Atajo global 'Y' para alternar la cortina divisoria Antes/Después
        self.shortcut_split = QShortcut(QKeySequence(Qt.Key_Y), self)
        self.shortcut_split.setContext(Qt.WindowShortcut)
        self.shortcut_split.activated.connect(self._on_toggle_split_shortcut)

        # Atajo global 'Shift+Y' para alternar la orientación de la cortina divisoria
        self.shortcut_split_orientation = QShortcut(QKeySequence("Shift+Y"), self)
        self.shortcut_split_orientation.setContext(Qt.WindowShortcut)
        self.shortcut_split_orientation.activated.connect(self._on_toggle_split_orientation_shortcut)

        # Atajo global 'W' para la pipeta de balance de blancos
        self.shortcut_pipette = QShortcut(QKeySequence(Qt.Key_W), self)
        self.shortcut_pipette.setContext(Qt.WindowShortcut)
        self.shortcut_pipette.activated.connect(self._on_toggle_pipette_shortcut)

        # Atajo global 'Z' para la lupa de inspección flotante
        self.shortcut_magnifier = QShortcut(QKeySequence(Qt.Key_Z), self)
        self.shortcut_magnifier.setContext(Qt.WindowShortcut)
        self.shortcut_magnifier.activated.connect(self._on_toggle_magnifier_shortcut)

        # Atajo global 'Ctrl+Z' para Deshacer
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.setContext(Qt.WindowShortcut)
        self.shortcut_undo.activated.connect(self.undo)

        # Atajos globales 'Ctrl+Y' y 'Ctrl+Shift+Z' para Rehacer
        self.shortcut_redo1 = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo1.setContext(Qt.WindowShortcut)
        self.shortcut_redo1.activated.connect(self.redo)

        self.shortcut_redo2 = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.shortcut_redo2.setContext(Qt.WindowShortcut)
        self.shortcut_redo2.activated.connect(self.redo)

        layout.addWidget(splitter)

    def _create_canvas_toolbar(self) -> QWidget:
        """Crea la barra de herramientas superior del visor interactivo."""
        toolbar = QWidget()
        toolbar.setObjectName("canvas_toolbar")
        toolbar.setFixedHeight(38)

        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 3, 8, 3)
        tb_layout.setSpacing(8)

        # 0a. Botón Deshacer (Undo)
        self.btn_undo = QPushButton("↺ Deshacer")
        self.btn_undo.setCursor(Qt.PointingHandCursor)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("Deshace el último ajuste de revelado (Atajo: Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        tb_layout.addWidget(self.btn_undo)

        # 0b. Botón Rehacer (Redo)
        self.btn_redo = QPushButton("↻ Rehacer")
        self.btn_redo.setCursor(Qt.PointingHandCursor)
        self.btn_redo.setEnabled(False)
        self.btn_redo.setToolTip("Rehace el ajuste deshecho (Atajo: Ctrl+Y o Ctrl+Shift+Z)")
        self.btn_redo.clicked.connect(self.redo)
        tb_layout.addWidget(self.btn_redo)

        # Separador vertical de historial
        sep_undo = QFrame()
        sep_undo.setFrameShape(QFrame.VLine)
        sep_undo.setFrameShadow(QFrame.Sunken)
        sep_undo.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep_undo)

        # 1. Botón Comparador Antes / Después
        self.btn_split_view = QPushButton("◫ Antes / Después (Y)")
        self.btn_split_view.setCheckable(True)
        self.btn_split_view.setCursor(Qt.PointingHandCursor)
        self.btn_split_view.setToolTip(
            "Activa la cortina divisoria interactiva para comparar el estado original ('Antes')\n"
            "con el revelado actual ('Después'). Puedes arrastrar la línea con el ratón o hacer doble clic para centrar.\n"
            "Atajo de teclado: Y"
        )
        self.btn_split_view.toggled.connect(self.on_split_view_toggled)
        tb_layout.addWidget(self.btn_split_view)

        # 1b. Botón Orientación Cortina (Vertical / Horizontal)
        self.btn_split_orientation = QPushButton("⇋ Vertical")
        self.btn_split_orientation.setCursor(Qt.PointingHandCursor)
        self.btn_split_orientation.setEnabled(False)
        self.btn_split_orientation.setToolTip(
            "Alterna la orientación de la cortina divisoria:\n"
            "• ⇋ Vertical: Izquierda ('Antes') / Derecha ('Después')\n"
            "• ⥯ Horizontal: Arriba ('Antes') / Abajo ('Después')\n"
            "Atajo de teclado: Shift+Y"
        )
        self.btn_split_orientation.clicked.connect(self.on_toggle_split_orientation)
        tb_layout.addWidget(self.btn_split_orientation)

        # 2. Botón Fijar como Antes (Snapshot)
        self.btn_set_before = QPushButton("📸 Fijar como Antes")
        self.btn_set_before.setCursor(Qt.PointingHandCursor)
        self.btn_set_before.setToolTip(
            "Congela el procesado actual como la nueva referencia de 'Antes'\n"
            "para evaluar cambios incrementales sucesivos."
        )
        self.btn_set_before.clicked.connect(self.on_set_before_clicked)
        tb_layout.addWidget(self.btn_set_before)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        sep1.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep1)

        # 3. Botón Indicador de Recorte (Clipping)
        self.btn_toggle_clipping = QPushButton("⚠️ Recorte Sombras/Luces (J)")
        self.btn_toggle_clipping.setCheckable(True)
        self.btn_toggle_clipping.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_clipping.setToolTip(
            "Muestra en el lienzo los píxeles con recorte: Azul para negros empastados (RGB=0)\n"
            "y Rojo para luces quemadas (RGB=255).\n"
            "Atajo de teclado: J"
        )
        self.btn_toggle_clipping.toggled.connect(self.on_toolbar_clipping_toggled)
        tb_layout.addWidget(self.btn_toggle_clipping)

        # Separador vertical
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep2)

        # 4. Botón Zoom 1:1
        self.btn_toggle_zoom = QPushButton("🔍 Zoom 100% (Clic Der.)")
        self.btn_toggle_zoom.setCheckable(True)
        self.btn_toggle_zoom.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_zoom.setToolTip(
            "Alterna entre vista adaptada completa y ampliación 1:1 al píxel nativo del sensor.\n"
            "También puedes hacer clic derecho directamente sobre el lienzo."
        )
        self.btn_toggle_zoom.toggled.connect(self.on_toolbar_zoom_toggled)
        tb_layout.addWidget(self.btn_toggle_zoom)

        # Separador vertical
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        sep3.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep3)

        # 5. Lupa de Inspección Flotante
        self.btn_magnifier = QPushButton("🔎 Lupa (Z)")
        self.btn_magnifier.setCheckable(True)
        self.btn_magnifier.setCursor(Qt.PointingHandCursor)
        self.btn_magnifier.setToolTip(
            "Activa la lupa de inspección flotante sobre el cursor para examinar enfoque, estrellas y ruido a nivel de píxel.\n"
            "Puedes regular el aumento usando la rueda del ratón.\n"
            "Atajo de teclado: Z"
        )
        self.btn_magnifier.toggled.connect(self.canvas.set_magnifier_active)
        tb_layout.addWidget(self.btn_magnifier)

        # Separador vertical
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.VLine)
        sep4.setFrameShadow(QFrame.Sunken)
        sep4.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep4)

        # 7. Botón Recorte Interactivo (Crop)
        self.btn_crop = QPushButton("✂️ Recortar (C)")
        self.btn_crop.setCheckable(True)
        self.btn_crop.setCursor(Qt.PointingHandCursor)
        self.btn_crop.setToolTip(
            "Activa la herramienta interactiva de recorte, composición y nivelación de horizonte.\n"
            "Permite ajustar proporciones (1:1, 16:9, 3:2, etc.), rotar y recomponer con guías.\n"
            "Atajo de teclado: C"
        )
        self.btn_crop.toggled.connect(self._on_toggle_crop)
        tb_layout.addWidget(self.btn_crop)

        # Separador vertical
        sep5 = QFrame()
        sep5.setFrameShape(QFrame.VLine)
        sep5.setFrameShadow(QFrame.Sunken)
        sep5.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep5)

        # 8. Botón Máscara Suelo Interactivo
        self.btn_mask_tool = QPushButton("🖌️ Máscara Suelo (M)")
        self.btn_mask_tool.setCheckable(True)
        self.btn_mask_tool.setCursor(Qt.PointingHandCursor)
        self.btn_mask_tool.setToolTip(
            "Activa la herramienta interactiva de trazado y segmentación inteligente de cielo y suelo.\n"
            "Dibuja trazos verdes sobre el cielo y rojos sobre el suelo para generar la máscara guiada.\n"
            "Atajo de teclado: M"
        )
        self.btn_mask_tool.toggled.connect(self._on_toggle_mask_tool)
        tb_layout.addWidget(self.btn_mask_tool)

        # 8b. Botón Detección Automática con IA Ligera (1-Click)
        self.btn_top_ai_mask = QPushButton("🧠 Máscara IA")
        self.btn_top_ai_mask.setCursor(Qt.PointingHandCursor)
        self.btn_top_ai_mask.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a1b9a, stop:1 #8e24aa);
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #ba68c8;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b1fa2, stop:1 #9c27b0);
                border-color: #ce93d8;
            }
            QPushButton:pressed {
                background-color: #4a148c;
            }
        """)
        self.btn_top_ai_mask.setToolTip("Detección y segmentación automática de cielo y suelo mediante IA ligera U²-Net (1-Click)")
        self.btn_top_ai_mask.clicked.connect(self._detect_mask_ai_from_toolbar)
        tb_layout.addWidget(self.btn_top_ai_mask)

        # Espaciador elástico
        tb_layout.addStretch()

        # 5. Etiqueta informativa de resolución y estado
        self.lbl_canvas_info = QLabel("Sin imagen cargada")
        self.lbl_canvas_info.setStyleSheet("color: #88c0d0; font-size: 11px; font-weight: bold;")
        tb_layout.addWidget(self.lbl_canvas_info)

        return toolbar

    def retranslate_ui(self) -> None:
        """Actualiza todos los textos visibles del Revelador según el idioma activo."""
        if hasattr(self, 'btn_open_img'):
            self.btn_open_img.setText(tr("developer.btn_open", "📁 Abrir Imagen (TIFF / FITS / RAW)..."))
        if hasattr(self, 'btn_export'):
            self.btn_export.setText(tr("developer.btn_export", "💾 EXPORTAR IMAGEN REVELADA..."))
        if hasattr(self, 'lbl_log_title'):
            self.lbl_log_title.setText(tr("developer.activity_log", "Registro de Actividad:"))

        # Secciones del acordeón
        if hasattr(self, 'sec_presets'):
            self.sec_presets.set_title(tr("developer.sections.presets", "🎨 Presets y Estilos de Revelado"))
        if hasattr(self, 'sec_gradients'):
            self.sec_gradients.set_title(tr("developer.sections.gradients", "⚡ Calibración y Gradientes (GraXpert AI)"))
        if hasattr(self, 'sec_stretch'):
            self.sec_stretch.set_title(tr("developer.sections.stretch", "📈 Estirado Tonal y Contraste"))
        if hasattr(self, 'sec_curves'):
            self.sec_curves.set_title(tr("developer.sections.curves", "📈 Curva Tonal Interactiva (RGB / Saturación)"))
        if hasattr(self, 'sec_starnet'):
            self.sec_starnet.set_title(tr("developer.sections.starnet", "⭐ StarNet++ AI — Desacoplo de Estrellas"))
        if hasattr(self, 'sec_wb'):
            self.sec_wb.set_title(tr("developer.sections.wb", "🌡️ Balance de Blancos y Calibración"))
        if hasattr(self, 'sec_color'):
            self.sec_color.set_title(tr("developer.sections.color", "🎨 Color y Saturación"))
        if hasattr(self, 'sec_gnd_tone'):
            self.sec_gnd_tone.set_title(tr("developer.sections.ground_tone", "🏞️ Ajuste Tonal Suelo (Paisaje)"))
        if hasattr(self, 'sec_masks'):
            self.sec_masks.set_title(tr("developer.sections.masks", "🎭 Máscaras y Capas de Ajuste Local"))

        # Botones de herramientas
        if hasattr(self, 'btn_starnet'):
            self.btn_starnet.setText(tr("developer.starnet.btn_separate", "✨ Separar Estrellas con StarNet"))
        if hasattr(self, 'btn_panel_pipette'):
            self.btn_panel_pipette.setText(tr("developer.wb.pipette", "Pipeta WB (W)"))

        # Barra superior del visor
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setText("↺ " + tr("developer.toolbar.undo", "Deshacer"))
        if hasattr(self, 'btn_redo'):
            self.btn_redo.setText("↻ " + tr("developer.toolbar.redo", "Rehacer"))
        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.setText("◫ " + tr("developer.toolbar.before_after", "Antes / Después (Y)"))
        if hasattr(self, 'btn_set_before'):
            self.btn_set_before.setText("📸 " + tr("developer.toolbar.set_as_before", "Fijar como Antes"))
        if hasattr(self, 'btn_toggle_clipping'):
            self.btn_toggle_clipping.setText("⚠️ " + tr("developer.toolbar.clipping", "Recorte Sombras/Luces (J)"))
        if hasattr(self, 'btn_toggle_zoom'):
            self.btn_toggle_zoom.setText("🔍 " + tr("developer.toolbar.zoom_100", "Zoom 100% (Clic Der.)"))
        if hasattr(self, 'btn_magnifier'):
            self.btn_magnifier.setText("🔎 " + tr("developer.toolbar.loupe", "Lupa (Z)"))
        if hasattr(self, 'btn_crop'):
            self.btn_crop.setText("✂️ " + tr("developer.toolbar.crop", "Recortar (C)"))
        if hasattr(self, 'btn_mask_tool'):
            self.btn_mask_tool.setText("🖌️ " + tr("developer.toolbar.ground_mask", "Máscara Suelo (M)"))

    def _create_crop_toolbar(self) -> QWidget:
        """Crea la barra de herramientas interactiva de recorte y nivelación de horizonte."""
        toolbar = QWidget()
        toolbar.setObjectName("crop_toolbar")
        toolbar.setFixedHeight(40)

        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 3, 10, 3)
        tb_layout.setSpacing(8)

        # 1. Título / Icono
        lbl_title = QLabel("✂️ Recorte:")
        lbl_title.setStyleSheet("font-weight: bold; color: #80d8ff; font-size: 11px;")
        tb_layout.addWidget(lbl_title)

        # 2. Selector de Proporción de Aspecto
        lbl_prop = QLabel("Proporción:")
        lbl_prop.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_prop)
        self.combo_crop_ratio = QComboBox()
        self.combo_crop_ratio.setToolTip("Define la relación de aspecto del marco de recorte")
        self.combo_crop_ratio.addItems([
            "Libre",
            "Original",
            "1:1 (Cuadrado)",
            "16:9 (Panorámico)",
            "3:2 (Fotografía)",
            "4:3 (Estándar)",
            "5:4 (Gran Formato)",
            "2:1 (Panorámico 2:1)",
            "9:16 (Vertical)",
            "2:3 (Vertical 35mm)",
            "4:5 (Vertical IG)",
        ])
        self.combo_crop_ratio.currentIndexChanged.connect(self._on_crop_ratio_changed)
        tb_layout.addWidget(self.combo_crop_ratio)

        # 3. Botón Invertir orientación de la proporción
        self.btn_crop_flip_ratio = QPushButton("⇄ Invertir (X)")
        self.btn_crop_flip_ratio.setToolTip("Invierte la orientación de la proporción (Horizontal ⇄ Vertical)\nAtajo: X")
        self.btn_crop_flip_ratio.setCursor(Qt.PointingHandCursor)
        self.btn_crop_flip_ratio.clicked.connect(self._on_crop_flip_ratio)
        tb_layout.addWidget(self.btn_crop_flip_ratio)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        sep1.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep1)

        # 4. Nivelación de Horizonte / Ángulo de rotación
        lbl_ang = self._make_resettable_label(QLabel("📐 Ángulo:"), lambda: self.slider_crop_angle.setValue(0), "0.0°")
        lbl_ang.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_ang)
        self.slider_crop_angle = QSlider(Qt.Horizontal)
        self.slider_crop_angle.setRange(-450, 450)  # -45.0° a +45.0°
        self.slider_crop_angle.setValue(0)
        self.slider_crop_angle.setFixedWidth(120)
        self.slider_crop_angle.setToolTip("Nivela el horizonte rotando la imagen en tiempo real (-45° a +45°)")
        self.slider_crop_angle.valueChanged.connect(self._on_crop_angle_changed)
        tb_layout.addWidget(self.slider_crop_angle)

        self.lbl_crop_angle_val = QLabel("0.0°")
        self.lbl_crop_angle_val.setStyleSheet("color: #88c0d0; font-weight: bold; min-width: 38px; font-size: 11px;")
        tb_layout.addWidget(self.lbl_crop_angle_val)

        self.btn_crop_reset_angle = QPushButton("0°")
        self.btn_crop_reset_angle.setFixedWidth(28)
        self.btn_crop_reset_angle.setToolTip("Restablece el ángulo de nivelación a 0°")
        self.btn_crop_reset_angle.setCursor(Qt.PointingHandCursor)
        self.btn_crop_reset_angle.clicked.connect(lambda: self.slider_crop_angle.setValue(0))
        tb_layout.addWidget(self.btn_crop_reset_angle)

        # Separador vertical
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep2)

        # 5. Guías de composición
        lbl_gui = QLabel("Guías:")
        lbl_gui.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_gui)
        self.combo_crop_guides = QComboBox()
        self.combo_crop_guides.addItems([
            "Regla de Tercios",
            "Cuadrícula 6×6",
            "Sin Guías"
        ])
        self.combo_crop_guides.currentIndexChanged.connect(self._on_crop_guides_changed)
        tb_layout.addWidget(self.combo_crop_guides)

        # 6. Etiqueta informativa de dimensiones
        self.lbl_crop_dimensions = QLabel("0 × 0 px")
        self.lbl_crop_dimensions.setStyleSheet("color: #ffd54f; font-weight: bold; font-size: 11px; margin-left: 4px;")
        tb_layout.addWidget(self.lbl_crop_dimensions)

        tb_layout.addStretch()

        # 7. Botones de acción: Restablecer, Cancelar, Aplicar
        self.btn_crop_reset = QPushButton("↺ Restablecer")
        self.btn_crop_reset.setToolTip("Restablece el marco de recorte al tamaño completo de la imagen")
        self.btn_crop_reset.setCursor(Qt.PointingHandCursor)
        self.btn_crop_reset.clicked.connect(self._reset_crop)
        tb_layout.addWidget(self.btn_crop_reset)

        self.btn_crop_cancel = QPushButton("❌ Cancelar (Esc)")
        self.btn_crop_cancel.setToolTip("Cancela el recorte y descarta los cambios (Esc)")
        self.btn_crop_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_crop_cancel.clicked.connect(self._cancel_crop)
        tb_layout.addWidget(self.btn_crop_cancel)

        self.btn_crop_apply = QPushButton("✅ Aplicar Recorte (Intro)")
        self.btn_crop_apply.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #4caf50;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #388e3c;
                border-color: #81c784;
            }
        """)
        self.btn_crop_apply.setToolTip("Aplica el recorte y la nivelación permanentemente a la imagen (Enter / Intro o Doble clic)")
        self.btn_crop_apply.setCursor(Qt.PointingHandCursor)
        self.btn_crop_apply.clicked.connect(self._apply_crop)
        tb_layout.addWidget(self.btn_crop_apply)

        return toolbar

    def _on_toggle_crop_shortcut(self) -> None:
        """Alterna el modo de recorte con la tecla 'C' ignorando entradas de texto activas."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_crop'):
            self.btn_crop.toggle()

    def _on_toggle_crop(self, enabled: bool) -> None:
        """Activa o desactiva la barra y modo de recorte interactivo en el lienzo."""
        if enabled and self.image_32bit is None:
            self.btn_crop.setChecked(False)
            self.log_message("[AVISO] Debe cargar una imagen antes de utilizar el recorte.")
            return

        if enabled:
            if self.is_zoomed_100:
                self.btn_toggle_zoom.setChecked(False)
            if self.canvas.eyedropper_active:
                self.btn_pipette.setChecked(False)
            if self.canvas.magnifier_active:
                self.btn_magnifier.setChecked(False)

        self.canvas.set_crop_mode(enabled)
        self.crop_toolbar.setVisible(enabled)

        if enabled:
            h, w = self.image_32bit.shape[:2]
            self.canvas.set_native_image_size((w, h))
            self.canvas._emit_crop_size()
            self.log_message("✂️ Modo Recorte activado. Ajusta el marco, nivela el horizonte o presiona Intro para aplicar.")
        else:
            self.canvas.reset_crop_rect()
            if hasattr(self, 'slider_crop_angle'):
                self.slider_crop_angle.setValue(0)

    def _on_crop_ratio_changed(self, idx: int) -> None:
        """Ajusta la relación de aspecto forzada en el lienzo según la selección del combo."""
        ratio_map = {
            0: None,
            1: (self.image_32bit.shape[1] / float(self.image_32bit.shape[0])) if self.image_32bit is not None else None,
            2: 1.0,
            3: 16.0 / 9.0,
            4: 3.0 / 2.0,
            5: 4.0 / 3.0,
            6: 5.0 / 4.0,
            7: 2.0,
            8: 9.0 / 16.0,
            9: 2.0 / 3.0,
            10: 4.0 / 5.0,
        }
        ratio = ratio_map.get(idx, None)
        self.canvas.set_crop_aspect_ratio(ratio)

    def _on_crop_flip_ratio(self) -> None:
        """Invierte la orientación de la proporción actual (Horizontal ⇄ Vertical)."""
        idx = self.combo_crop_ratio.currentIndex()
        flip_pairs = {3: 8, 8: 3, 4: 9, 9: 4, 5: 10, 10: 5}
        if idx in flip_pairs:
            self.combo_crop_ratio.setCurrentIndex(flip_pairs[idx])
            return

        cur_ratio = self.canvas.crop_aspect_ratio
        if cur_ratio is not None and cur_ratio > 0:
            self.canvas.set_crop_aspect_ratio(1.0 / cur_ratio)
        else:
            r = self.canvas.crop_rect_norm
            cx, cy = r.center().x(), r.center().y()
            ih = float(self.image_32bit.shape[0]) if self.image_32bit is not None else 1.0
            iw = float(self.image_32bit.shape[1]) if self.image_32bit is not None else 1.0
            new_w = min(1.0, r.height() * (ih / iw))
            new_h = min(1.0, r.width() * (iw / ih))
            new_x = max(0.0, min(1.0 - new_w, cx - new_w / 2.0))
            new_y = max(0.0, min(1.0 - new_h, cy - new_h / 2.0))
            from PySide6.QtCore import QRectF
            self.canvas.crop_rect_norm = QRectF(new_x, new_y, new_w, new_h)
            self.canvas._emit_crop_size()
            self.canvas.update()

    def _on_crop_angle_changed(self, val: int) -> None:
        """Actualiza el ángulo de nivelación en tiempo real."""
        angle = val / 10.0
        self.lbl_crop_angle_val.setText(f"{angle:+.1f}°" if abs(angle) > 1e-2 else "0.0°")
        self.canvas.set_crop_angle(angle)

    def _on_crop_guides_changed(self, idx: int) -> None:
        """Alterna las guías de composición."""
        guides = ["thirds", "grid", "none"][idx] if idx < 3 else "thirds"
        self.canvas.set_crop_guides(guides)

    def _on_crop_rect_changed(self, px_w: int, px_h: int) -> None:
        """Muestra la resolución resultante en la barra de herramientas."""
        mp = (px_w * px_h) / 1e6
        self.lbl_crop_dimensions.setText(f"{px_w} × {px_h} px ({mp:.1f} MP)")

    def _reset_crop(self) -> None:
        """Restablece el marco de recorte al 100% y el ángulo a 0°."""
        if hasattr(self, 'slider_crop_angle'):
            self.slider_crop_angle.setValue(0)
        if hasattr(self, 'combo_crop_ratio'):
            self.combo_crop_ratio.setCurrentIndex(0)
        self.canvas.reset_crop_rect()

    def _cancel_crop(self) -> None:
        """Cancela el modo de recorte sin aplicar ningún cambio."""
        self.btn_crop.setChecked(False)
        self.log_message("Recorte cancelado.")

    def _apply_crop(self) -> None:
        """Aplica permanentemente el recorte y la nivelación a la imagen en memoria con Undo/Redo."""
        if self.image_32bit is None:
            self.btn_crop.setChecked(False)
            return

        orig_h, orig_w = self.image_32bit.shape[:2]
        angle = self.canvas.crop_angle
        norm = self.canvas.crop_rect_norm

        no_angle = abs(angle) < 1e-3
        full_frame = (norm.x() <= 0.002 and norm.y() <= 0.002 and 
                      norm.width() >= 0.996 and norm.height() >= 0.996)
        if no_angle and full_frame:
            self.log_message("El marco abarca el 100% de la toma sin rotación. Sin cambios.")
            self.btn_crop.setChecked(False)
            return

        # Registrar estado previo en la pila de deshacer (Undo)
        undo_item = {
            "type": "crop",
            "image_32bit": self.image_32bit.copy(),
            "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
            "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
            "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
        }
        self.undo_stack.append(undo_item)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

        # Transformación rotación y corte
        def rotate_and_crop(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
            h, w = arr.shape[:2]
            if not no_angle:
                center = (w / 2.0, h / 2.0)
                rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
                interp = cv2.INTER_LINEAR if is_mask else cv2.INTER_LANCZOS4
                arr = cv2.warpAffine(arr, rot_mat, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT)

            px_x = int(round(norm.x() * w))
            px_y = int(round(norm.y() * h))
            px_w = int(round(norm.width() * w))
            px_h = int(round(norm.height() * h))
            px_x = max(0, min(w - 1, px_x))
            px_y = max(0, min(h - 1, px_y))
            px_w = max(10, min(w - px_x, px_w))
            px_h = max(10, min(h - px_y, px_h))
            return np.ascontiguousarray(arr[px_y:px_y + px_h, px_x:px_x + px_w], dtype=np.float32)

        self.image_32bit = rotate_and_crop(self.image_32bit)
        if self.image_starless is not None:
            self.image_starless = rotate_and_crop(self.image_starless)
        if self.image_stars is not None:
            self.image_stars = rotate_and_crop(self.image_stars)
        if self.current_mask is not None:
            self.current_mask = rotate_and_crop(self.current_mask, is_mask=True)

        self.btn_crop.setChecked(False)
        if hasattr(self, 'slider_crop_angle'):
            self.slider_crop_angle.setValue(0)
        self.canvas.reset_crop_rect()

        new_h, new_w = self.image_32bit.shape[:2]
        self.canvas.set_native_image_size((new_w, new_h))
        self._generate_preview_proxy()
        self.snapshot_before_pixmap = None
        self.has_custom_snapshot = False
        self._generate_before_reference()
        self.update_stretch_preview()
        self._update_canvas_info_label()
        self._last_committed_state = self._capture_state()
        self._update_undo_redo_buttons()

        ang_info = f" con rotación {angle:+.1f}°" if not no_angle else ""
        self.log_message(f"✅ Recorte aplicado{ang_info}: {orig_w}x{orig_h} → {new_w}x{new_h} px. (Deshacer con Ctrl+Z)")

    def _create_mask_toolbar(self) -> QWidget:
        """Crea la barra de herramientas interactiva para dibujar semillas y refinar la máscara de suelo."""
        toolbar = QWidget()
        toolbar.setObjectName("mask_toolbar")
        toolbar.setFixedHeight(40)

        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 3, 10, 3)
        tb_layout.setSpacing(8)

        # 1. Título / Icono
        lbl_title = QLabel("🖌️ Máscara Suelo:")
        lbl_title.setStyleSheet("font-weight: bold; color: #80d8ff; font-size: 11px;")
        tb_layout.addWidget(lbl_title)

        # 2. Selector de Clase de Pincel (Cielo vs Suelo)
        self.btn_brush_sky = QPushButton("🟢 Cielo (Verde)")
        self.btn_brush_sky.setCheckable(True)
        self.btn_brush_sky.setChecked(True)
        self.btn_brush_sky.setCursor(Qt.PointingHandCursor)
        self.btn_brush_sky.setStyleSheet("""
            QPushButton {
                background-color: #1a2e22;
                color: #a5d6a7;
                border: 1px solid #2e7d32;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2e7d32;
                color: #ffffff;
                border-color: #66bb6a;
            }
        """)
        self.btn_brush_sky.setToolTip("Pinta semillas de CIELO (trazo verde)")
        tb_layout.addWidget(self.btn_brush_sky)

        self.btn_brush_gnd = QPushButton("🔴 Suelo (Rojo)")
        self.btn_brush_gnd.setCheckable(True)
        self.btn_brush_gnd.setCursor(Qt.PointingHandCursor)
        self.btn_brush_gnd.setStyleSheet("""
            QPushButton {
                background-color: #2e1a1a;
                color: #ef9a9a;
                border: 1px solid #c62828;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #c62828;
                color: #ffffff;
                border-color: #ef5350;
            }
        """)
        self.btn_brush_gnd.setToolTip("Pinta semillas de SUELO (trazo rojo)")
        tb_layout.addWidget(self.btn_brush_gnd)

        # Grupo exclusivo para los botones de pincel
        self.brush_group = QButtonGroup(self)
        self.brush_group.addButton(self.btn_brush_sky, 2)
        self.brush_group.addButton(self.btn_brush_gnd, 1)
        self.brush_group.idClicked.connect(self._on_brush_mode_changed)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        sep1.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep1)

        # 3. Radio del pincel
        lbl_r = self._make_resettable_label(QLabel("Pincel:"), lambda: self.slider_mask_brush.setValue(20), "20 px")
        lbl_r.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_r)

        self.slider_mask_brush = QSlider(Qt.Horizontal)
        self.slider_mask_brush.setRange(5, 120)
        self.slider_mask_brush.setValue(20)
        self.slider_mask_brush.setFixedWidth(90)
        self.slider_mask_brush.setToolTip("Tamaño del pincel en píxeles de pantalla (o usa la rueda del ratón)")
        self.slider_mask_brush.valueChanged.connect(self._on_slider_brush_changed)
        tb_layout.addWidget(self.slider_mask_brush)

        self.lbl_mask_brush_val = QLabel("20 px")
        self.lbl_mask_brush_val.setStyleSheet("color: #88c0d0; font-weight: bold; min-width: 38px; font-size: 11px;")
        tb_layout.addWidget(self.lbl_mask_brush_val)

        # Separador vertical
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep2)

        # 4. Parámetros de refinamiento (Feather e Iterations)
        lbl_f = QLabel("Borde:")
        lbl_f.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_f)

        self.spin_mask_feather = QSpinBox()
        self.spin_mask_feather.setRange(1, 45)
        self.spin_mask_feather.setValue(7)
        self.spin_mask_feather.setSuffix(" px")
        self.spin_mask_feather.setFixedWidth(66)
        self.spin_mask_feather.setToolTip("Radio de suavizado / transición gaussiana en el perímetro del horizonte")
        tb_layout.addWidget(self.spin_mask_feather)

        lbl_it = QLabel("Pasos:")
        lbl_it.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_it)

        self.spin_mask_iters = QSpinBox()
        self.spin_mask_iters.setRange(1, 10)
        self.spin_mask_iters.setValue(3)
        self.spin_mask_iters.setFixedWidth(46)
        self.spin_mask_iters.setToolTip("Número de iteraciones del algoritmo de segmentación GrabCut")
        tb_layout.addWidget(self.spin_mask_iters)

        # Separador vertical
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        sep3.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        # 5a. Detección Automática con IA Ligera
        self.btn_ai_mask = QPushButton("🧠 Máscara IA")
        self.btn_ai_mask.setCursor(Qt.PointingHandCursor)
        self.btn_ai_mask.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a1b9a, stop:1 #8e24aa);
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #ba68c8;
                border-radius: 4px;
                padding: 3px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b1fa2, stop:1 #9c27b0);
                border-color: #ce93d8;
            }
            QPushButton:pressed {
                background-color: #4a148c;
            }
        """)
        self.btn_ai_mask.setToolTip("Detecta automáticamente el horizonte y suelo mediante IA ligera U²-Net con 1 solo clic")
        self.btn_ai_mask.clicked.connect(self._detect_mask_ai_from_toolbar)
        tb_layout.addWidget(self.btn_ai_mask)

        # 5b. Generar Máscara desde trazos
        self.btn_generate_mask = QPushButton("✨ Generar (Intro)")
        self.btn_generate_mask.setCursor(Qt.PointingHandCursor)
        self.btn_generate_mask.setStyleSheet("""
            QPushButton {
                background-color: #00796b;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #26a69a;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00897b;
                border-color: #4db6ac;
            }
            QPushButton:pressed {
                background-color: #004d40;
            }
        """)
        self.btn_generate_mask.setToolTip("Calcula la máscara optimizada con GrabCut asistido por luminancia y preservación de siluetas finas (Intro)")
        self.btn_generate_mask.clicked.connect(self._compute_mask_from_scribbles)
        tb_layout.addWidget(self.btn_generate_mask)

        # 5c. Cargar Máscara (del Apilado o desde Archivo)
        self.btn_load_mask = QPushButton("📂 Cargar...")
        self.btn_load_mask.setCursor(Qt.PointingHandCursor)
        self.btn_load_mask.setStyleSheet("""
            QPushButton {
                background-color: #1a2736;
                color: #80d8ff;
                font-weight: bold;
                border: 1px solid #0288d1;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0288d1;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #01579b;
            }
        """)
        self.btn_load_mask.setToolTip("Carga una máscara generada previamente durante el apilado o desde un archivo (.png, .tif, .fits)")
        self.btn_load_mask.clicked.connect(self._load_mask_from_toolbar)
        tb_layout.addWidget(self.btn_load_mask)

        # 6. Ver / Ocultar Máscara Overlay
        self.btn_mask_overlay_toggle = QPushButton("👁️ Ver")
        self.btn_mask_overlay_toggle.setCheckable(True)
        self.btn_mask_overlay_toggle.setChecked(True)
        self.btn_mask_overlay_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_mask_overlay_toggle.setToolTip("Muestra u oculta la superposición semitransparente (Verde=Cielo, Rojo=Suelo)")
        self.btn_mask_overlay_toggle.toggled.connect(self.canvas.set_mask_visible)
        tb_layout.addWidget(self.btn_mask_overlay_toggle)

        # 7. Limpiar Trazos
        self.btn_clear_scribbles = QPushButton("🗑️ Trazos")
        self.btn_clear_scribbles.setCursor(Qt.PointingHandCursor)
        self.btn_clear_scribbles.setToolTip("Borra los trazos dibujados sin eliminar la máscara actual calculada")
        self.btn_clear_scribbles.clicked.connect(self._clear_scribbles)
        tb_layout.addWidget(self.btn_clear_scribbles)

        # 8. Quitar / Eliminar Máscara
        self.btn_delete_mask = QPushButton("❌ Quitar")
        self.btn_delete_mask.setCursor(Qt.PointingHandCursor)
        self.btn_delete_mask.setToolTip("Elimina la máscara activa por completo de la imagen")
        self.btn_delete_mask.clicked.connect(self._delete_mask)
        tb_layout.addWidget(self.btn_delete_mask)

        tb_layout.addStretch()

        # 9. Botón Finalizar / Cerrar
        self.btn_mask_done = QPushButton("✅ Listo (Esc)")
        self.btn_mask_done.setCursor(Qt.PointingHandCursor)
        self.btn_mask_done.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #4caf50;
                border-radius: 4px;
                padding: 3px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #388e3c;
                border-color: #81c784;
            }
        """)
        self.btn_mask_done.setToolTip("Cierra el modo de trazado de máscara y vuelve al visor estándar (Esc)")
        self.btn_mask_done.clicked.connect(lambda: self.btn_mask_tool.setChecked(False))
        tb_layout.addWidget(self.btn_mask_done)

        return toolbar

    def _on_toggle_mask_shortcut(self) -> None:
        """Alterna el modo de trazado de máscara con la tecla 'M' ignorando entradas de texto activas."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_mask_tool'):
            self.btn_mask_tool.toggle()

    def _on_toggle_mask_tool(self, enabled: bool) -> None:
        """Activa o desactiva la barra y modo de trazado interactivo de máscara en el lienzo."""
        if enabled and self.image_32bit is None:
            self.btn_mask_tool.setChecked(False)
            self.log_message("[AVISO] Debe cargar una imagen antes de utilizar el trazador de máscaras.")
            return

        self.mask_toolbar.setVisible(enabled)

        if enabled:
            # Desactivar herramientas excluyentes si estuvieran activas
            if hasattr(self, 'btn_crop') and self.btn_crop.isChecked():
                self.btn_crop.setChecked(False)
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_magnifier') and self.btn_magnifier.isChecked():
                self.btn_magnifier.setChecked(False)
            if self.canvas.split_view_enabled:
                self.btn_split_view.setChecked(False)

            self.canvas.set_masking_enabled(True)
            self.canvas.set_brush_mode(2 if self.btn_brush_sky.isChecked() else 1)
            self.canvas.set_brush_radius(self.slider_mask_brush.value())
            self.log_message("[MÁSCARA] Modo de trazado activo: dibuja trazos verdes en el CIELO y rojos en el SUELO.")
        else:
            self.canvas.set_masking_enabled(False)
            self.log_message("[MÁSCARA] Modo de trazado cerrado.")

    def _on_brush_mode_changed(self, btn_id: int) -> None:
        """Actualiza la clase de semilla seleccionada (2 = Cielo, 1 = Suelo)."""
        self.canvas.set_brush_mode(btn_id)

    def _on_slider_brush_changed(self, val: int) -> None:
        """Actualiza el radio del pincel al mover el deslizador."""
        self.lbl_mask_brush_val.setText(f"{val} px")
        self.canvas.set_brush_radius(val)

    def _on_brush_size_changed(self, val: int) -> None:
        """Actualiza el deslizador cuando el usuario cambia el tamaño con la rueda del ratón."""
        if hasattr(self, 'slider_mask_brush'):
            self.slider_mask_brush.blockSignals(True)
            self.slider_mask_brush.setValue(val)
            self.slider_mask_brush.blockSignals(False)
            self.lbl_mask_brush_val.setText(f"{val} px")

    def _open_mask_tool_from_section(self) -> None:
        """Abre y activa la herramienta de máscara desde el botón de la Sección 6."""
        if hasattr(self, 'btn_mask_tool'):
            if not self.btn_mask_tool.isChecked():
                self.btn_mask_tool.setChecked(True)
            else:
                self.canvas.set_masking_enabled(True)

    def _jump_to_mask_layers(self, target_layer_type: Optional[str] = None) -> None:
        """Expande la Sección 9 y selecciona la capa especificada para edición contextual."""
        if hasattr(self, 'sec_masks'):
            self.sec_masks.expand()
        if hasattr(self, 'mask_layers_widget') and target_layer_type:
            found = self.mask_layers_widget.select_layer_by_type(target_layer_type)
            if not found and target_layer_type == MASK_TYPE_IMAGE_GROUND:
                self.mask_layers_widget.select_layer_by_type(MASK_TYPE_GROUND)


    def _compute_mask_from_scribbles(self) -> None:
        """Calcula la segmentación continua cielo/suelo mediante GrabCut y fotometría guiada."""
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Abre primero una toma en el revelador.")
            return

        scribbles = self.canvas.get_scribbles_matrix()
        if scribbles is None or not (np.any(scribbles == 1) and np.any(scribbles == 2)):
            QMessageBox.warning(
                self, 
                "Trazos insuficientes", 
                "Debes pintar al menos un trazo verde (🟢 Cielo) y un trazo rojo (🔴 Suelo) sobre la imagen."
            )
            return

        self.log_message("[MÁSCARA] Iniciando segmentación inteligente de cielo y suelo...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()

        old_m = self.current_mask.copy() if self.current_mask is not None else None
        t0 = time.perf_counter()

        try:
            h, w = self.image_32bit.shape[:2]
            sh, sw = scribbles.shape[:2]
            if (sh, sw) != (h, w):
                scribbles_full = cv2.resize(scribbles, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                scribbles_full = scribbles

            feather = self.spin_mask_feather.value()
            iters = self.spin_mask_iters.value()

            new_mask = refine_mask_guided(
                self.image_32bit,
                scribbles_full,
                feather_radius=feather,
                iterations=iters
            )

            self.current_mask = new_mask
            self.canvas.set_refined_mask(self.current_mask)
            self._generate_preview_proxy()
            self.update_stretch_preview()
            self._update_mask_status_ui()
            self._record_mask_action(old_m, self.current_mask, "Generar Máscara")

            elapsed = time.perf_counter() - t0
            self.log_message(f"✓ [MÁSCARA FINALIZADA] Máscara cielo/suelo calculada en {elapsed:.2f} s ({w}x{h} px, Borde: {feather}px, Pasos: {iters}).")
        except Exception as e:
            self.log_message(f"[ERROR MÁSCARA] Fallo al generar la máscara: {e}")
            QMessageBox.critical(self, "Error de Segmentación", f"No se pudo generar la máscara:\n{e}")
        finally:
            self.unsetCursor()

    def _clear_scribbles(self) -> None:
        """Borra únicamente los trazos dibujados sin eliminar la máscara actual."""
        self.canvas.clear_scribbles()
        self.log_message("[MÁSCARA] Trazos de semillas eliminados del visor.")

    def _delete_mask(self) -> None:
        """Elimina completamente la máscara de cielo/suelo y actualiza el revelador."""
        if self.current_mask is None:
            return
        old_m = self.current_mask.copy()
        self.current_mask = None
        self.canvas.clear_scribbles()
        self.canvas.set_refined_mask(None)
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self._update_mask_status_ui()
        self._record_mask_action(old_m, None, "Eliminar Máscara")
        self.log_message("[MÁSCARA] Máscara de suelo eliminada.")

    def _load_mask_from_toolbar(self) -> None:
        """Permite cargar una máscara de cielo/suelo previamente generada en el apilado o desde archivo."""
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Abre primero una toma en el revelador.")
            return

        # Comprobar si existe una máscara activa en el Apilador (Stacker) o Startrails
        stacker_mask = None
        win = self.window()
        if win is not None:
            if hasattr(win, "tab_stacker") and getattr(win.tab_stacker, "computed_mask", None) is not None:
                stacker_mask = win.tab_stacker.computed_mask
            elif hasattr(win, "tab_startrails") and getattr(win.tab_startrails, "current_mask", None) is not None:
                stacker_mask = win.tab_startrails.current_mask

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1e2d;
                color: #e0e0e0;
                border: 1px solid #33384c;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 10px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #0288d1;
                color: #ffffff;
            }
        """)
        act_ai = menu.addAction("🧠 Auto-Detectar Suelo con IA (1-Click)")
        act_stacker = None
        if stacker_mask is not None:
            act_stacker = menu.addAction("📥 Usar Máscara generada en el Apilado")
        act_file = menu.addAction("📁 Cargar Archivo de Máscara (.png, .tif, .fits)...")

        pos = self.btn_load_mask.mapToGlobal(self.btn_load_mask.rect().bottomLeft())
        selected = menu.exec(pos)
        if selected == act_ai:
            self._detect_mask_ai_from_toolbar()
        elif act_stacker and selected == act_stacker:
            self._apply_loaded_mask(stacker_mask, source_desc="Apilador (Stacker)")
        elif selected == act_file:
            self._load_mask_from_file()

    def _detect_mask_ai_from_toolbar(self) -> None:
        """Calcula la máscara de suelo/cielo con IA ligera directamente desde el visor del revelador."""
        if self.image_32bit is None:
            return

        from core.ai_horizon import is_ai_model_available
        if not is_ai_model_available():
            msg = QMessageBox(self)
            msg.setWindowTitle("Modelo de IA no encontrado")
            msg.setText(
                "El modelo de IA para detección semántica precisa de suelo (skyseg.onnx, ~167 MB) no se encuentra en la carpeta 'models/'.\n\n"
                "¿Deseas descargarlo ahora automáticamente o prefieres usar el método algorítmico rápido?"
            )
            btn_download = msg.addButton("📥 Descargar IA (~167 MB)", QMessageBox.AcceptRole)
            btn_algo = msg.addButton("⚡ Usar Algorítmico", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_download)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_download:
                self._start_model_download_developer()
                return
            elif clicked == btn_algo:
                self._run_horizon_developer(force_algorithmic=True)
                return
            else:
                return

        self._run_horizon_developer(force_algorithmic=False)

    def _start_model_download_developer(self):
        """Descarga asistida del modelo ONNX con diálogo de progreso para el revelador."""
        progress = QProgressDialog("Conectando con el servidor de modelos...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Descarga de Modelo IA (AstroLoom)")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        from gui.worker import ModelDownloadWorker
        self.download_worker = ModelDownloadWorker(parent=self)

        def on_prog(pct, d_mb, t_mb):
            progress.setValue(pct)
            progress.setLabelText(f"Descargando modelo de IA (skyseg.onnx)...\n{d_mb:.1f} MB de {t_mb:.1f} MB ({pct}%)")

        def on_finished(path):
            progress.close()
            self.log_message(f"✓ [IA HORIZONTE] Modelo descargado e instalado en: {os.path.basename(path)}")
            self._run_horizon_developer(force_algorithmic=False)

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR DESCARGA] {err}")
            QMessageBox.warning(self, "Error en Descarga", f"No se pudo descargar el modelo:\n{err}\n\nSe usará el método algorítmico.")
            self._run_horizon_developer(force_algorithmic=True)

        def on_cancel():
            self.download_worker.cancel()
            self.log_message("[IA HORIZONTE] Descarga cancelada por el usuario.")

        self.download_worker.progress_changed.connect(on_prog)
        self.download_worker.finished_success.connect(on_finished)
        self.download_worker.error_occurred.connect(on_error)
        progress.canceled.connect(on_cancel)

        self.download_worker.start()

    def _run_horizon_developer(self, force_algorithmic: bool = False):
        self.setCursor(Qt.WaitCursor)
        self.log_message("[IA HORIZONTE] Ejecutando detección de cielo/suelo...")
        QCoreApplication.processEvents()
        try:
            from core.ai_horizon import auto_detect_sky_mask
            mask, method = auto_detect_sky_mask(self.image_32bit, force_algorithmic=force_algorithmic)
            self._apply_loaded_mask(mask, source_desc=f"Auto-Detección {method}")
        except Exception as e:
            self.log_message(f"[ERROR IA HORIZONTE] Falló la detección: {e}")
            QMessageBox.critical(self, "Error Detección IA", f"No se pudo completar la segmentación automática:\n{e}")
        finally:
            self.unsetCursor()

    def _load_mask_from_file(self) -> None:
        """Abre un selector de archivo para cargar una máscara desde disco."""
        cfg = load_config()
        default_dir = cfg.get("masks_dir", os.getcwd())
        filtros = "Imágenes de Máscara (*.png *.tif *.tiff *.fits *.fit);;Todos los archivos (*.*)"
        filepath, _ = QFileDialog.getOpenFileName(self, "Cargar Máscara de Suelo / Cielo", default_dir, filtros)
        if not filepath:
            return

        try:
            raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raw = load_image_as_float32(filepath)
                if raw is not None and raw.ndim == 3:
                    raw = raw[..., 0]

            if raw is None:
                QMessageBox.warning(self, "Error", f"No se pudo decodificar el archivo de máscara:\n{filepath}")
                return

            if raw.ndim == 3:
                raw = raw[..., 0]

            max_v = 65535.0 if raw.dtype == np.uint16 else (1.0 if raw.dtype in (np.float32, np.float64) and np.max(raw) <= 1.0 else 255.0)
            mask_float = np.clip(raw.astype(np.float32) / max_v, 0.0, 1.0)
            self._apply_loaded_mask(mask_float, source_desc=os.path.basename(filepath))
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar máscara", f"Fallo al procesar la máscara:\n{e}")

    def _apply_loaded_mask(self, mask: np.ndarray, source_desc: str = "") -> None:
        """Adapta y aplica una máscara cargada a la imagen actual en el Revelador."""
        if self.image_32bit is None or mask is None:
            return

        h, w = self.image_32bit.shape[:2]
        if mask.shape[:2] != (h, w):
            self.log_message(f"[MÁSCARA] Redimensionando máscara ({mask.shape[1]}x{mask.shape[0]}) a resolución de lienzo ({w}x{h})...")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        old_m = self.current_mask.copy() if self.current_mask is not None else None
        self.current_mask = np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)
        self.canvas.set_refined_mask(self.current_mask)
        self.canvas.set_mask_visible(True)
        if hasattr(self, 'btn_mask_overlay_toggle'):
            self.btn_mask_overlay_toggle.blockSignals(True)
            self.btn_mask_overlay_toggle.setChecked(True)
            self.btn_mask_overlay_toggle.blockSignals(False)

        self._generate_preview_proxy()
        self.update_stretch_preview()
        self._update_mask_status_ui()
        self._record_mask_action(old_m, self.current_mask, f"Cargar Máscara ({source_desc})")
        self.log_message(f"✓ [MÁSCARA] Máscara cargada y aplicada con éxito desde: {source_desc}.")


    def _update_mask_status_ui(self) -> None:
        """Actualiza el indicador visual de presencia de máscara en la Sección 8 y botones."""
        if hasattr(self, 'sec_gnd_tone'):
            if self.current_mask is not None:
                self.sec_gnd_tone.set_status_badge("● Suelo Activo", "#a5d6a7")
            else:
                self.sec_gnd_tone.set_status_badge("")

        if hasattr(self, 'mask_layers_widget') and self.image_starless is not None:
            self.mask_layers_widget.ensure_starnet_layers(has_ground=(self.current_mask is not None))

        if not hasattr(self, 'lbl_gnd_mask_status'):
            return
        if self.current_mask is not None:
            self.lbl_gnd_mask_status.setText("✓ Máscara de suelo activa")
            self.lbl_gnd_mask_status.setStyleSheet("color: #a5d6a7; font-size: 11px; font-weight: bold;")
            if hasattr(self, 'btn_edit_ground_mask'):
                self.btn_edit_ground_mask.setText("🖌️ Editar Máscara...")
            if hasattr(self, 'btn_delete_mask'):
                self.btn_delete_mask.setEnabled(True)
        else:
            self.lbl_gnd_mask_status.setText("⚠️ Sin máscara de suelo activa")
            self.lbl_gnd_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
            if hasattr(self, 'btn_edit_ground_mask'):
                self.btn_edit_ground_mask.setText("🖌️ Crear Máscara Suelo...")
            if hasattr(self, 'btn_delete_mask'):
                self.btn_delete_mask.setEnabled(False)

    def _record_mask_action(self, old_mask: Optional[np.ndarray], new_mask: Optional[np.ndarray], action_name: str = "Máscara") -> None:
        """Registra una acción de máscara en el historial de Deshacer / Rehacer."""
        undo_item = {
            "type": "mask",
            "name": action_name,
            "old_mask": old_mask.copy() if old_mask is not None else None,
            "new_mask": new_mask.copy() if new_mask is not None else None,
        }
        self.undo_stack.append(undo_item)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

    def _on_layer_mask_preview_requested(self, mask_array: Optional[np.ndarray]) -> None:
        """Muestra u oculta la máscara de la capa seleccionada sobre el visor interactivo."""
        if mask_array is not None and np.max(mask_array) > 1e-4:
            self.canvas.set_refined_mask(mask_array, mode="ruby")
            self.canvas.set_mask_visible(True)
        else:
            self.canvas.set_mask_visible(False)
            if self.canvas.enable_masking and self.current_mask is not None:
                self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
            else:
                self.canvas.set_refined_mask(None)

    def _on_layer_brush_toggled(self, active: bool, layer: Optional[MaskLayer]) -> None:
        """Activa o desactiva la interacción del pincel en el lienzo según la capa seleccionada."""
        if active and layer is not None and getattr(layer, 'enabled', True):
            self.canvas.set_layer_brush_enabled(True)
            self.canvas.set_layer_brush_params(
                layer.brush_radius, layer.brush_hardness, layer.brush_flow, layer.brush_mode
            )
            show_overlay = bool(
                hasattr(self, 'mask_layers_widget') and 
                hasattr(self.mask_layers_widget, 'btn_preview_mask') and 
                self.mask_layers_widget.btn_preview_mask.isChecked()
            )
            self.canvas.set_active_layer_mask(layer.brush_mask, show_overlay=show_overlay)
            if not show_overlay:
                self.canvas.set_refined_mask(None)
                self.canvas.set_mask_visible(False)
        else:
            self.canvas.set_layer_brush_enabled(False)
            self.canvas.set_active_layer_mask(None)

            # Limpiar overlay del lienzo si no hay previsualización válida y activa de la capa
            sel_layer = self.mask_layers_widget.get_selected_layer() if hasattr(self, 'mask_layers_widget') else None
            is_preview_valid = (
                sel_layer is not None and 
                sel_layer.enabled and 
                not sel_layer.is_system_layer and 
                hasattr(self.mask_layers_widget, 'btn_preview_mask') and 
                self.mask_layers_widget.btn_preview_mask.isChecked()
            )
            if not is_preview_valid:
                if self.canvas.enable_masking and self.current_mask is not None:
                    self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
                else:
                    self.canvas.set_refined_mask(None)
                    self.canvas.set_mask_visible(False)

    def _on_layer_brush_params_changed(self, radius: int, hardness: float, flow: float, mode: str) -> None:
        """Actualiza los parámetros interactivos de trazo en el lienzo."""
        self.canvas.set_layer_brush_params(radius, hardness, flow, mode)

    def _on_layer_brush_stroke_finished(self, stroke_mask: np.ndarray) -> None:
        """Manejador ejecutado al soltar el clic izquierdo tras pintar con el pincel de capa."""
        layer = self.mask_layers_widget.get_selected_layer()
        if layer and layer.mask_type == "brush":
            layer.brush_mask = stroke_mask.copy()
            # Si el botón de previsualización NO está activado expresamente, retirar el overlay temporal
            # del trazo para que el usuario pueda apreciar de inmediato los cambios tonales en la imagen
            if hasattr(self, 'mask_layers_widget') and not self.mask_layers_widget.btn_preview_mask.isChecked():
                self.canvas.set_refined_mask(None)
                self.canvas.set_mask_visible(False)
            self._commit_action()
            self.update_stretch_preview()

    def _on_toggle_clipping_shortcut(self) -> None:
        """Alterna los indicadores de recorte con la tecla 'J' ignorando entradas de texto activas."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'curve_widget'):
            self.curve_widget.toggle_both_clipping()

    def _on_toggle_split_shortcut(self) -> None:
        """Alterna el modo comparador Antes / Después con la tecla 'Y'."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.toggle()

    def _on_toggle_split_orientation_shortcut(self) -> None:
        """Alterna la orientación de la cortina con Shift+Y."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_split_orientation') and self.canvas.split_view_enabled:
            self.on_toggle_split_orientation()

    def _on_toggle_pipette_shortcut(self) -> None:
        """Alterna la pipeta de balance de blancos con la tecla 'W'."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_pipette'):
            self.btn_pipette.toggle()

    def _on_toggle_magnifier_shortcut(self) -> None:
        """Alterna la lupa de inspección flotante con la tecla 'Z'."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_magnifier'):
            self.btn_magnifier.toggle()

    def _on_canvas_eyedropper_toggled(self, active: bool) -> None:
        """Sincroniza el estado del cuentagotas con los botones de la interfaz."""
        if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'wb':
            self.btn_pipette.blockSignals(True)
            self.btn_pipette.setChecked(active)
            self.btn_pipette.blockSignals(False)

        if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'ghs_sp':
            self.btn_ghs_pipette.blockSignals(True)
            self.btn_ghs_pipette.setChecked(active)
            self.btn_ghs_pipette.blockSignals(False)

        if not active:
            self._eyedropper_target = 'wb'

    def _on_toggle_ghs_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el Punto de Simetría (SP) de GHS en el lienzo."""
        if active:
            self._eyedropper_target = 'ghs_sp'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            self.canvas.set_eyedropper_active(True)
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'ghs_sp':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_eyedropper_color_sampled(self, r: float, g: float, b: float) -> None:
        """Procesa la muestra del cuentagotas para Balance de Blancos o Punto de Simetría GHS."""
        if getattr(self, '_eyedropper_target', 'wb') == 'ghs_sp':
            # Muestra para Punto de Simetría (SP) en GHS
            sample_lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            sp_val = float(np.clip(sample_lum, 0.00001, 0.99999))
            if hasattr(self, 'spin_ghs_sp'):
                self.spin_ghs_sp.blockSignals(True)
                self.spin_ghs_sp.setValue(sp_val)
                self.spin_ghs_sp.blockSignals(False)
            if hasattr(self, 'slider_ghs_sp'):
                slider_val = int(np.clip(sp_val * 10000.0, 0, 10000))
                self.slider_ghs_sp.blockSignals(True)
                self.slider_ghs_sp.setValue(slider_val)
                self.slider_ghs_sp.blockSignals(False)
            if hasattr(self, 'btn_ghs_pipette'):
                self.btn_ghs_pipette.setChecked(False)
            self._eyedropper_target = 'wb'
            self.log_message(
                f"[GHS] Cuentagotas SP: muestra RGB=({r:.3f}, {g:.3f}, {b:.3f}) ➜ SP fijado en {sp_val:.5f}"
            )
            self.update_stretch_preview()
            self._commit_action()
            return

        r = max(1e-5, float(r))
        g = max(1e-5, float(g))
        b = max(1e-5, float(b))

        w_r = 1.0 / r
        w_g = 1.0 / g
        w_b = 1.0 / b
        w_mean = (w_r + w_g + w_b) / 3.0

        gain_r = w_r / w_mean
        gain_g = w_g / w_mean
        gain_b = w_b / w_mean

        # Solución de apply_white_balance:
        # gain_r - gain_b = temp
        # tint = 2.0 * (1.0 - gain_g)
        calc_temp = gain_r - gain_b
        calc_tint = 2.0 * (1.0 - gain_g)

        # Rango del slider (-1000 a +1000 con paso 0.001)
        clamped_temp = float(np.clip(calc_temp, -1.000, 1.000))
        clamped_tint = float(np.clip(calc_tint, -1.000, 1.000))

        temp_int = int(round(clamped_temp * 1000.0))
        tint_int = int(round(clamped_tint * 1000.0))

        self.slider_temp.setValue(temp_int)
        self.slider_tint.setValue(tint_int)

        if hasattr(self, 'btn_pipette'):
            self.btn_pipette.setChecked(False)

        self.log_message(
            f"[WB] Pipeta aplicada: Muestra RGB=({r:.3f}, {g:.3f}, {b:.3f}) ➜ "
            f"Temp={self.temp_val:+.3f}, Tinte={self.tint_val:+.3f}"
        )

    def on_toggle_split_orientation(self) -> None:
        """Alterna entre cortina vertical y horizontal."""
        new_ori = self.canvas.toggle_split_orientation()
        self._update_split_orientation_ui()
        ori_nombre = "VERTICAL (Izquierda / Derecha)" if new_ori == "vertical" else "HORIZONTAL (Arriba / Abajo)"
        self.log_message(f"[CANVAS] Orientación del comparador cambiada a: {ori_nombre}")
        self._update_canvas_info_label()

    def _update_split_orientation_ui(self) -> None:
        """Actualiza el texto y estado del botón de orientación de la cortina."""
        if not hasattr(self, 'btn_split_orientation'):
            return
        is_vert = (self.canvas.split_orientation == "vertical")
        self.btn_split_orientation.setText("⇋ Vertical" if is_vert else "⥯ Horizontal")
        self.btn_split_orientation.setEnabled(self.canvas.split_view_enabled)

    def on_clipping_changed_sync(self, clip_s: bool, clip_h: bool) -> None:
        """Sincroniza el lienzo y el botón de la barra superior cuando cambia el recorte."""
        self.canvas.set_clipping_state(clip_s, clip_h)
        if hasattr(self, 'btn_toggle_clipping'):
            self.btn_toggle_clipping.blockSignals(True)
            self.btn_toggle_clipping.setChecked(clip_s or clip_h)
            self.btn_toggle_clipping.blockSignals(False)

    def on_toolbar_clipping_toggled(self, checked: bool) -> None:
        """Manejador al conmutar el botón de recorte en la barra superior."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_clipping_state(checked, checked)
            estado = "ACTIVADOS (Azul=Negros, Rojo=Blancos)" if checked else "DESACTIVADOS"
            self.log_message(f"[CANVAS] Indicadores de recorte: {estado}")

    def on_split_view_toggled(self, checked: bool) -> None:
        """Manejador al conmutar el modo comparador Antes / Después."""
        if checked and (self.canvas.before_pixmap is None or self.canvas.before_pixmap.isNull()):
            self._generate_before_reference()
        self.canvas.set_split_view_enabled(checked)
        self._update_split_orientation_ui()
        if checked and self.is_zoomed_100:
            self.update_stretch_preview()
        estado = "ACTIVADO (Arrastra la línea divisoria para comparar)" if checked else "DESACTIVADO"
        self.log_message(f"[CANVAS] Modo Comparador Antes / Después: {estado}")
        self._update_canvas_info_label()

    def on_set_before_clicked(self) -> None:
        """Congela el revelado actual como nueva imagen de referencia 'Antes'."""
        if self.canvas.base_pixmap and not self.canvas.base_pixmap.isNull():
            self.snapshot_before_pixmap = self.canvas.base_pixmap.copy()
            self.canvas.set_before_pixmap(self.snapshot_before_pixmap)
            self.initial_bp = self.spin_bp.value()
            self.initial_mtf = self.spin_mtf.value()
            self.initial_stretch_mode = getattr(self, "stretch_mode", "mtf")
            self.initial_ghs_sp = self.spin_ghs_sp.value() if hasattr(self, 'spin_ghs_sp') else 0.02
            self.initial_ghs_d = self.spin_ghs_d.value() if hasattr(self, 'spin_ghs_d') else 5.0
            self.initial_ghs_b = self.spin_ghs_b.value() if hasattr(self, 'spin_ghs_b') else 2.0
            self.initial_ghs_hp = self.spin_ghs_hp.value() if hasattr(self, 'spin_ghs_hp') else 0.0
            self.initial_ghs_lp = self.spin_ghs_lp.value() if hasattr(self, 'spin_ghs_lp') else 0.0
            self.initial_ghs_bp = self.spin_ghs_bp.value() if hasattr(self, 'spin_ghs_bp') else 0.0
            self.initial_ghs_color_mode = getattr(self, "ghs_color_mode", "luma")
            self.has_custom_snapshot = True
            self.log_message("[CANVAS] Estado actual fijado como nueva referencia de 'Antes'.")
            if not self.btn_split_view.isChecked():
                self.btn_split_view.setChecked(True)
        else:
            self.log_message("[AVISO] No hay ninguna imagen cargada en el lienzo para fijar.")

    def on_toolbar_zoom_toggled(self, checked: bool) -> None:
        """Manejador al conmutar el botón de zoom en la barra superior."""
        if self.is_zoomed_100 != checked:
            self.is_zoomed_100 = checked
            self.canvas.is_zoomed = checked
            if checked:
                self.zoom_rel_coords = (0.5, 0.5)
                self.canvas.zoom_center_img = None
            else:
                self._generate_before_reference()
            self.update_stretch_preview()
            self._update_canvas_info_label()

    def _generate_before_reference(self) -> None:
        """Genera el QPixmap base de referencia 'Antes' a partir de la toma cruda original."""
        if self.image_32bit is None:
            self.canvas.set_before_pixmap(None)
            return

        if self.has_custom_snapshot and self.snapshot_before_pixmap is not None and not self.is_zoomed_100:
            self.canvas.set_before_pixmap(self.snapshot_before_pixmap)
            return

        bp = self.initial_bp
        mtf = self.initial_mtf

        base_to_render, _ = self._compose_active_base(for_export=False)
        if base_to_render is not None:
            if getattr(self, "initial_stretch_mode", "mtf") == "ghs":
                stretched_init = apply_ghs_stretch(
                    base_to_render,
                    sp=getattr(self, "initial_ghs_sp", 0.02),
                    d=getattr(self, "initial_ghs_d", 5.0),
                    b=getattr(self, "initial_ghs_b", 2.0),
                    hp=getattr(self, "initial_ghs_hp", 0.0),
                    lp=getattr(self, "initial_ghs_lp", 0.0),
                    bp=getattr(self, "initial_ghs_bp", 0.0),
                    color_mode=getattr(self, "initial_ghs_color_mode", "luma")
                )
            else:
                stretched_init = manual_stretch(base_to_render, black_point=bp, midtone=mtf)
            disp = (np.clip(stretched_init, 0.0, 1.0) * 255.0).astype(np.uint8)
            h, w = disp.shape[:2]
            disp_c = np.ascontiguousarray(disp, dtype=np.uint8)
            qimg = QImage(disp_c.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.canvas.set_before_pixmap(QPixmap.fromImage(qimg))

    def _update_canvas_info_label(self) -> None:
        """Actualiza la etiqueta informativa de resolución y estado en la barra del lienzo."""
        if not hasattr(self, 'lbl_canvas_info'):
            return
        if self.image_32bit is None:
            self.lbl_canvas_info.setText("Sin imagen cargada")
            return
        h, w = self.image_32bit.shape[:2]
        mp = (w * h) / 1e6
        zoom_txt = "Zoom 100% (1:1)" if self.is_zoomed_100 else "Ajustado"
        split_txt = ""
        if self.canvas.split_view_enabled:
            ori_txt = "Vert." if self.canvas.split_orientation == "vertical" else "Horiz."
            split_txt = f" | ◫ Comparador ({ori_txt})"
        self.lbl_canvas_info.setText(f"{w}x{h} px ({mp:.1f} MP) | {zoom_txt}{split_txt}")

    def keyPressEvent(self, event) -> None:
        """Captura directa de teclas rápidas 'M', 'J', 'Y', 'Shift+Y', 'C' y controles interactivos."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            super().keyPressEvent(event)
            return

        if hasattr(self, 'canvas') and self.canvas.enable_masking:
            if event.key() == Qt.Key_Escape:
                if hasattr(self, 'btn_mask_tool'):
                    self.btn_mask_tool.setChecked(False)
                event.accept()
                return
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._compute_mask_from_scribbles()
                event.accept()
                return

        if self.canvas.crop_mode:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._apply_crop()
                event.accept()
                return
            elif event.key() == Qt.Key_Escape:
                self._cancel_crop()
                event.accept()
                return
            elif event.key() == Qt.Key_X:
                self._on_crop_flip_ratio()
                event.accept()
                return

        if event.key() == Qt.Key_M:
            self._on_toggle_mask_shortcut()
            event.accept()
            return
        elif event.key() == Qt.Key_C:
            self._on_toggle_crop_shortcut()
            event.accept()
            return
        elif event.key() == Qt.Key_J:
            self._on_toggle_clipping_shortcut()
            event.accept()
            return
        elif event.key() == Qt.Key_Y:
            if event.modifiers() & Qt.ShiftModifier:
                self._on_toggle_split_orientation_shortcut()
            else:
                self._on_toggle_split_shortcut()
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_styles(self) -> None:
        """Aplica estilos específicos para las barras de herramientas del visor de revelado."""
        self.setStyleSheet(f"""
            #canvas_toolbar {{
                background-color: {BG_PANEL};
                border-bottom: 1px solid {BORDER_BASE};
            }}
            #canvas_toolbar QPushButton {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                border-radius: 4px;
                padding: 4px 10px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
            }}
            #canvas_toolbar QPushButton:hover {{
                background-color: #282e42;
                border-color: {TEXT_CYAN};
            }}
            #canvas_toolbar QPushButton:checked {{
                background-color: #0d47a1;
                color: #ffffff;
                border-color: {ACCENT_CYAN};
            }}
            #canvas_toolbar QPushButton:disabled {{
                background-color: #141620;
                color: {TEXT_MUTED};
                border-color: {BORDER_BASE};
            }}
            #crop_toolbar {{
                background-color: #10121a;
                border-bottom: 1px solid {BORDER_BASE};
            }}
            #crop_toolbar QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
            }}
            #crop_toolbar QComboBox {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                border-radius: 4px;
                padding: 2px 8px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
            }}
            #crop_toolbar QPushButton {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                border-radius: 4px;
                padding: 3px 8px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
            }}
            #crop_toolbar QPushButton:hover {{
                background-color: #282e42;
                border-color: {TEXT_CYAN};
            }}
            #mask_toolbar {{
                background-color: #10121a;
                border-bottom: 1px solid {BORDER_BASE};
            }}
            #mask_toolbar QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
            }}
            #mask_toolbar QSpinBox {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                border-radius: 4px;
                padding: 2px 4px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
            }}
            #mask_toolbar QPushButton {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                border-radius: 4px;
                padding: 3px 8px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
            }}
            #mask_toolbar QPushButton:hover {{
                background-color: #282e42;
                border-color: {TEXT_CYAN};
            }}
        """)

    def log_message(self, text: str):
        self.txt_log.append(text)
        self.txt_log.moveCursor(QTextCursor.End)

    def _generate_preview_proxy(self):
        if self.image_32bit is None:
            self.preview_proxy = None
            self.proxy_starless = None
            self.proxy_stars = None
            self.proxy_mask = None
            return

        h, w = self.image_32bit.shape[:2]
        max_dim = 1600.0
        scale = min(1.0, max_dim / max(h, w))

        clean_mask = None
        if self.current_mask is not None:
            clean_mask = np.squeeze(self.current_mask)
            if clean_mask.ndim > 2:
                clean_mask = clean_mask[..., 0]
            clean_mask = clean_mask.astype(np.float32)
            if clean_mask.shape != (h, w):
                clean_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        if scale < 1.0:
            pw = (int(w * scale) // 4) * 4
            ph = (int(h * scale) // 4) * 4
            self.preview_proxy = np.ascontiguousarray(
                cv2.resize(self.image_32bit, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
            )
            if self.image_starless is not None:
                self.proxy_starless = np.ascontiguousarray(
                    cv2.resize(self.image_starless, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
                )
                self.proxy_stars = np.ascontiguousarray(
                    cv2.resize(self.image_stars, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
                )
            else:
                self.proxy_starless = None
                self.proxy_stars = None

            if clean_mask is not None:
                m_small = cv2.resize(clean_mask, (pw, ph), interpolation=cv2.INTER_LINEAR)
                ksize = int(max(7, (min(ph, pw) // 150) | 1))
                if ksize % 2 == 0: 
                    ksize += 1
                sm = cv2.GaussianBlur(m_small, (ksize, ksize), sigmaX=ksize / 3.0)
                self.proxy_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)
            else:
                self.proxy_mask = None
        else:
            self.preview_proxy = np.ascontiguousarray(self.image_32bit, dtype=np.float32)
            self.proxy_starless = self.image_starless.copy() if self.image_starless is not None else None
            self.proxy_stars = self.image_stars.copy() if self.image_stars is not None else None
            if clean_mask is not None:
                ksize = int(max(15, (min(h, w) // 150) | 1))
                if ksize % 2 == 0: 
                    ksize += 1
                sm = cv2.GaussianBlur(clean_mask, (ksize, ksize), sigmaX=ksize / 3.0)
                self.proxy_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)
            else:
                self.proxy_mask = None

    def load_image_direct(self, filepath: str, mask: np.ndarray = None):
        self.active_filepath = filepath
        self.current_mask = mask
        self.image_starless = None
        self.image_stars = None
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None
        self.preview_proxy = None
        
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
            raw_img = load_image_as_float32(filepath, color_space="srgb")
            
            # Saneamiento preventivo de NaNs e infinitos
            if np.isnan(raw_img).any() or np.isinf(raw_img).any():
                self.log_message("[AVISO] La imagen contenía píxeles indefinidos (NaN/Inf). Han sido saneados automáticamente.")
                raw_img = np.nan_to_num(raw_img, nan=0.0, posinf=1.0, neginf=0.0)

            self.image_32bit = np.clip(raw_img, 0.0, 1.0).astype(np.float32)
            self.canvas.set_native_image_size((self.image_32bit.shape[1], self.image_32bit.shape[0]))
            self._generate_preview_proxy()
            self.apply_auto_mtf()
            self.initial_bp = self.spin_bp.value()
            self.initial_mtf = self.spin_mtf.value()
            self.has_custom_snapshot = False
            self.snapshot_before_pixmap = None
            self._generate_before_reference()
            self._update_canvas_info_label()

            # Inicializar historial de Deshacer / Rehacer limpio para la nueva imagen
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._last_committed_state = self._capture_state()
            self._update_undo_redo_buttons()

            if self.current_mask is not None:
                self.canvas.set_refined_mask(self.current_mask)
            else:
                self.canvas.set_refined_mask(None)
            self._update_mask_status_ui()

            # Asegurar que se crea la capa Fondo (Imagen Base) en el Gestor de Capas
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.ensure_base_layer()

            self.log_message(f"Imagen en memoria ({self.image_32bit.shape[1]}x{self.image_32bit.shape[0]} px). Vista acelerada activa.")
        except Exception as e:
            self.log_message(f"[ERROR] No se pudo cargar: {e}")
            QMessageBox.critical(self, "Error", f"Fallo al abrir archivo:\n{e}")

    def open_image_file(self):
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

    def _on_canvas_files_dropped(self, paths: list):
        """Carga automáticamente la primera imagen arrastrada sobre el lienzo."""
        if paths:
            self.load_image_direct(paths[0])

    def _process_starless_base(self, starless: Optional[np.ndarray], mask_3d: Optional[np.ndarray] = None, is_full_res: bool = True) -> Optional[np.ndarray]:
        """Aplica las mejoras lineales a la capa StarNet (Denoise, Claridad, Dehaze, Ondículas)."""
        if starless is None:
            return None
        processed_starless = starless.copy()

        # 1. Reducción de Ruido
        if self.denoise_strength > 1e-4:
            denoised = apply_denoise(
                processed_starless,
                strength=self.denoise_strength,
                is_full_res=is_full_res
            )
            if mask_3d is not None:
                processed_starless = (denoised * mask_3d) + (processed_starless * (1.0 - mask_3d))
            else:
                processed_starless = denoised

        # 2. Claridad y Borrar Neblina
        if abs(self.clarity_starless_val) > 1e-4:
            clarified = apply_clarity(processed_starless, strength=self.clarity_starless_val)
            if mask_3d is not None:
                processed_starless = (clarified * mask_3d) + (processed_starless * (1.0 - mask_3d))
            else:
                processed_starless = clarified

        if abs(self.dehaze_starless_val) > 1e-4:
            dehazed = apply_dehaze(processed_starless, strength=self.dehaze_starless_val)
            if mask_3d is not None:
                processed_starless = (dehazed * mask_3d) + (processed_starless * (1.0 - mask_3d))
            else:
                processed_starless = dehazed

        # 3. Ondículas sobre la capa de fondo
        if abs(self.multiescala_estructura) > 1e-4 or self.multiescala_fondo > 1e-4:
            processed_starless = realce_multiescala_ondiculas(
                processed_starless,
                fuerza_estructura=self.multiescala_estructura,
                reduccion_fondo=self.multiescala_fondo,
                sky_mask=mask_3d[..., 0] if mask_3d is not None else None
            )

        return processed_starless

    def _compose_active_base(self, for_export: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if for_export:
            starless, stars, base = self.image_starless, self.image_stars, self.image_32bit
            mask_3d = None
            if self.current_mask is not None and base is not None:
                h_f, w_f = base.shape[:2]
                m_f = cv2.resize(self.current_mask, (w_f, h_f), interpolation=cv2.INTER_LINEAR) if self.current_mask.shape[:2] != (h_f, w_f) else self.current_mask
                mask_3d = np.repeat(m_f[..., np.newaxis], 3, axis=2)
            crop_active = False
            self._zoom_crop_rect = None
        elif self.is_zoomed_100 and self.image_32bit is not None:
            h_f, w_f = self.image_32bit.shape[:2]
            cw = min(w_f, max(100, self.canvas.width()))
            ch = min(h_f, max(100, self.canvas.height()))
            
            rx, ry = self.zoom_rel_coords
            cx = int(rx * w_f)
            cy = int(ry * h_f)

            x1 = max(0, min(w_f - cw, cx - cw // 2))
            y1 = max(0, min(h_f - ch, cy - ch // 2))
            x2 = x1 + cw
            y2 = y1 + ch
            self._zoom_crop_rect = (x1, y1, x2, y2)

            base = self.image_32bit[y1:y2, x1:x2].copy()
            starless = self.image_starless[y1:y2, x1:x2].copy() if self.image_starless is not None else None
            stars = self.image_stars[y1:y2, x1:x2].copy() if self.image_stars is not None else None

            if self.current_mask is not None:
                if self.current_mask.shape[:2] != (h_f, w_f):
                    m_f = cv2.resize(self.current_mask, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
                else:
                    m_f = self.current_mask
                m_sub = m_f[y1:y2, x1:x2]
                mask_3d = np.repeat(m_sub[..., np.newaxis], 3, axis=2)
            else:
                mask_3d = None
            crop_active = True
        else:
            starless, stars, base = self.proxy_starless, self.proxy_stars, self.preview_proxy
            mask_3d = self.proxy_mask
            crop_active = False
            self._zoom_crop_rect = None

        if base is None:
            return None, None

        if starless is None or stars is None:
            return base, mask_3d

        processed_starless = self._process_starless_base(
            starless,
            mask_3d=mask_3d,
            is_full_res=(for_export or crop_active)
        )

        # 4. Mezcla de capas
        # Si el gestor de capas contiene capas de imagen activas (StarNet / Suelo),
        # delegamos la composición completa al motor multicapa de apply_all_mask_layers
        # para respetar el orden visual de capas del usuario y evitar duplicar estrellas.
        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.has_image_layers():
            if self.view_layer_mode == 2:
                return stars, mask_3d
            return processed_starless, mask_3d

        if self.view_layer_mode == 0:
            composed = np.clip(processed_starless + (stars * self.star_intensity), 0.0, 1.0)
            return composed, mask_3d
        elif self.view_layer_mode == 1:
            return processed_starless, mask_3d
        else:
            return stars, mask_3d

    def _apply_pipeline_on_image(
        self,
        target_img: np.ndarray,
        precomputed_mask: Optional[np.ndarray] = None,
        apply_mask_layers: bool = True
    ) -> np.ndarray:
        if target_img is None:
            return None

        # 1. Balance de blancos
        img = apply_white_balance(target_img, self.temp_val, self.tint_val)

        # 2. Saturación diferencial cielo / suelo
        if self.sat_sky_val != 1.0 or self.sat_gnd_val != 1.0:
            hsv = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2HSV)
            if precomputed_mask is None or self.sat_sky_val == self.sat_gnd_val:
                hsv[..., 1] = np.clip(hsv[..., 1] * self.sat_sky_val, 0.0, 1.0)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            else:
                hsv_sky = hsv.copy()
                hsv_sky[..., 1] = np.clip(hsv_sky[..., 1] * self.sat_sky_val, 0.0, 1.0)
                rgb_sky = cv2.cvtColor(hsv_sky, cv2.COLOR_HSV2RGB)

                hsv_gnd = hsv.copy()
                hsv_gnd[..., 1] = np.clip(hsv_gnd[..., 1] * self.sat_gnd_val, 0.0, 1.0)
                rgb_gnd = cv2.cvtColor(hsv_gnd, cv2.COLOR_HSV2RGB)

                img = (rgb_sky * precomputed_mask) + (rgb_gnd * (1.0 - precomputed_mask))

        # 3. Intensidad
        if abs(self.vibrance_val) > 1e-4:
            img = adjust_vibrance(img, self.vibrance_val)

        # 4. Atenuación de Cúpula de Luz
        if self.lp_reduction > 1e-4:
            mask_2d = precomputed_mask[..., 0] if precomputed_mask is not None else None
            img = apply_light_pollution_gradient(img, strength=self.lp_reduction, height_ratio=0.50, sky_mask=mask_2d)

        # 5. Ajuste Tonal Diferencial para el Suelo
        if precomputed_mask is not None:
            has_gnd_tone_change = (
                abs(self.gnd_ev_val) > 1e-4 or 
                self.gnd_shadows_val > 1e-4 or 
                abs(self.gnd_bp_val) > 1e-4
            )
            if has_gnd_tone_change:
                gnd_part = img.copy()

                if abs(self.gnd_ev_val) > 1e-4:
                    gnd_part = gnd_part * (2.0 ** self.gnd_ev_val)

                if self.gnd_shadows_val > 1e-4:
                    lift_curve = (1.0 - np.clip(gnd_part, 0.0, 1.0)) ** 2
                    gnd_part = gnd_part * (1.0 + self.gnd_shadows_val * lift_curve)

                if abs(self.gnd_bp_val) > 1e-4:
                    if self.gnd_bp_val < 0.0:
                        gnd_part = np.clip(gnd_part - self.gnd_bp_val, 0.0, 1.0)
                    else:
                        gnd_part = np.clip((gnd_part - self.gnd_bp_val) / max(1e-4, 1.0 - self.gnd_bp_val), 0.0, 1.0)

                img = (img * precomputed_mask) + (gnd_part * (1.0 - precomputed_mask))

        # 6. Estirado Tonal (MTF o GHS)
        if getattr(self, "stretch_mode", "mtf") == "ghs":
            stretched = apply_ghs_stretch(
                img,
                sp=self.spin_ghs_sp.value(),
                d=self.spin_ghs_d.value(),
                b=self.spin_ghs_b.value(),
                hp=self.spin_ghs_hp.value(),
                lp=self.spin_ghs_lp.value(),
                bp=self.spin_ghs_bp.value(),
                color_mode=getattr(self, "ghs_color_mode", "luma")
            )
        else:
            bp_val = self.spin_bp.value()
            m_val = self.spin_mtf.value()
            stretched = manual_stretch(img, black_point=bp_val, midtone=m_val)

        # 7. Curvas Tonales (LUT RGB)
        if not self.curve_widget.is_identity("rgb"):
            curved = apply_curve_lut(stretched, self.curve_widget.get_lut("rgb"))
            if precomputed_mask is not None:
                stretched = (curved * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
            else:
                stretched = curved

        # 7b. Curva Selectiva Tono vs Saturación (Hue vs Sat)
        if not self.curve_widget.is_identity("hue_sat"):
            hsv_curved = apply_hue_vs_sat_curve(stretched, self.curve_widget.get_lut("hue_sat"))
            if precomputed_mask is not None:
                stretched = (hsv_curved * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
            else:
                stretched = hsv_curved

        # 7c. Curva Selectiva Luminancia vs Saturación (Luma vs Sat)
        if not self.curve_widget.is_identity("luma_sat"):
            luma_curved = apply_luma_vs_sat_curve(stretched, self.curve_widget.get_lut("luma_sat"))
            if precomputed_mask is not None:
                stretched = (luma_curved * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
            else:
                stretched = luma_curved

        # 8. Contraste específico del Fondo (Starless)
        if abs(self.contrast_starless_val) > 1e-4:
            if precomputed_mask is not None and np.any(precomputed_mask > 0.5):
                sky_median = float(np.median(stretched[precomputed_mask > 0.5]))
            else:
                sky_median = float(np.median(stretched))

            sn_contrasted = adjust_contrast(stretched, self.contrast_starless_val, pivot=sky_median)
            if precomputed_mask is not None:
                stretched = (sn_contrasted * precomputed_mask) + (stretched * (1.0 - precomputed_mask))
            else:
                stretched = sn_contrasted

        # 9. Contraste Global
        if abs(self.contrast_val) > 1e-4:
            stretched = adjust_contrast(stretched, self.contrast_val, pivot=None)

        # 10. Aplicación de Capas y Máscaras Locales (Compositor Multicapa estilo Photoshop)
        if apply_mask_layers and hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.layers:
            is_full_res = (self.image_32bit is not None and stretched.shape[:2] == self.image_32bit.shape[:2])
            is_zoomed = (self.is_zoomed_100 and getattr(self, '_zoom_crop_rect', None) is not None)
            full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None

            if is_zoomed:
                x1, y1, x2, y2 = self._zoom_crop_rect
                st_img = self.image_stars[y1:y2, x1:x2].copy() if self.image_stars is not None else None
                sl_img = self.image_starless[y1:y2, x1:x2].copy() if self.image_starless is not None else None
                crop_rect = (x1, y1, x2, y2)
            elif is_full_res:
                st_img = self.image_stars
                sl_img = self.image_starless
                crop_rect = None
            else:
                st_img = self.proxy_stars if self.proxy_stars is not None else self.image_stars
                sl_img = self.proxy_starless if self.proxy_starless is not None else self.image_starless
                crop_rect = None
            
            # Si hay capas de imagen especiales activas, preparamos las estrellas estiradas con el MTF
            # y moduladas por la intensidad estelar relativa (self.star_intensity) para el modo Screen
            if st_img is not None and self.mask_layers_widget.has_image_layers():
                if getattr(self, "stretch_mode", "mtf") == "ghs":
                    st_stretched = np.clip(
                        apply_ghs_stretch(
                            st_img,
                            sp=self.spin_ghs_sp.value(),
                            d=self.spin_ghs_d.value(),
                            b=self.spin_ghs_b.value(),
                            hp=self.spin_ghs_hp.value(),
                            lp=self.spin_ghs_lp.value(),
                            bp=0.0,
                            color_mode="rgb"
                        ) * self.star_intensity,
                        0.0, 1.0
                    )
                else:
                    st_stretched = np.clip(
                        manual_stretch(st_img, black_point=0.0, midtone=self.spin_mtf.value()) * self.star_intensity,
                        0.0, 1.0
                    )
            elif st_img is not None:
                st_stretched = np.clip(st_img * self.star_intensity, 0.0, 1.0)
            else:
                st_stretched = st_img

            stretched = apply_all_mask_layers(
                stretched,
                self.mask_layers_widget.layers,
                master_sky_mask=precomputed_mask,
                starless_image=sl_img,
                stars_image=st_stretched,
                crop_rect=crop_rect,
                full_image_shape=full_shape
            )

        return np.ascontiguousarray(stretched, dtype=np.float32)

    def update_stretch_preview(self):
        base_to_render, active_mask = self._compose_active_base(for_export=False)
        if base_to_render is None:
            return

        if self.view_layer_mode == 2:
            if getattr(self, "stretch_mode", "mtf") == "ghs":
                stretched = apply_ghs_stretch(
                    base_to_render,
                    sp=self.spin_ghs_sp.value(),
                    d=self.spin_ghs_d.value(),
                    b=self.spin_ghs_b.value(),
                    hp=self.spin_ghs_hp.value(),
                    lp=self.spin_ghs_lp.value(),
                    bp=self.spin_ghs_bp.value(),
                    color_mode=getattr(self, "ghs_color_mode", "luma")
                )
            else:
                stretched = manual_stretch(base_to_render, black_point=self.spin_bp.value(), midtone=self.spin_mtf.value())
        else:
            stretched = self._apply_pipeline_on_image(base_to_render, precomputed_mask=active_mask)

        if stretched is not None:
            self.canvas.load_image(base_to_render, display_stretched=stretched)

            # Sincronización del comparador Antes / Después en modo Zoom 1:1
            if self.is_zoomed_100 and self.canvas.split_view_enabled:
                if getattr(self, "initial_stretch_mode", "mtf") == "ghs":
                    stretched_before = apply_ghs_stretch(
                        base_to_render,
                        sp=getattr(self, "initial_ghs_sp", 0.02),
                        d=getattr(self, "initial_ghs_d", 5.0),
                        b=getattr(self, "initial_ghs_b", 2.0),
                        hp=getattr(self, "initial_ghs_hp", 0.0),
                        lp=getattr(self, "initial_ghs_lp", 0.0),
                        bp=getattr(self, "initial_ghs_bp", 0.0),
                        color_mode=getattr(self, "initial_ghs_color_mode", "luma")
                    )
                else:
                    stretched_before = manual_stretch(base_to_render, black_point=self.initial_bp, midtone=self.initial_mtf)
                disp_b = (np.clip(stretched_before, 0.0, 1.0) * 255.0).astype(np.uint8)
                h_b, w_b = disp_b.shape[:2]
                disp_bc = np.ascontiguousarray(disp_b, dtype=np.uint8)
                qimg_b = QImage(disp_bc.data, w_b, h_b, 3 * w_b, QImage.Format_RGB888).copy()
                self.canvas.set_before_pixmap(QPixmap.fromImage(qimg_b))

            # Solo actualizar histograma en vista general para no alterar la escala al hacer zoom 1:1
            if not self.is_zoomed_100:
                self.curve_widget.set_histogram_from_image(stretched)
            self._update_canvas_info_label()

            # Reiniciar temporizador de consolidación de historial (captura scroll de ratón y flechas)
            if not self._is_undoing_or_redoing and hasattr(self, '_undo_debounce_timer'):
                self._undo_debounce_timer.start(500)

    def run_starnet(self):
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Carga o apila una imagen primero.")
            return

        self._set_ai_processing_state(True)
        self.log_message("=== INICIANDO SEPARACIÓN STARNET++ AI ===")
        if hasattr(self, 'sec_masks'):
            self.sec_masks.expand()

        stride = self.spin_stride.value()

        self.sn_worker = StarNetWorker(
            self.image_32bit, 
            sky_mask=self.current_mask, 
            stride=stride
        )
        self.sn_worker.status_changed.connect(self.log_message)
        self.sn_worker.finished_success.connect(self.on_starnet_success)
        self.sn_worker.error_occurred.connect(self.on_starnet_error)
        self.sn_worker.start()

    def on_starnet_success(self, starless_img: np.ndarray, stars_img: np.ndarray):
        self._set_ai_processing_state(False)

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
        self.image_stars = np.ascontiguousarray(stars_img, dtype=np.float32)

        # Integrar capas de imagen (Fondo, Estrellas y opcionalmente Suelo) en el Gestor de Capas
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.ensure_starnet_layers(has_ground=(self.current_mask is not None))

        self._set_starnet_tools_enabled(True)
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.log_message("[STARNET] Estrellas separadas y purgadas con máscara de cielo. Capas activadas.")

        # Desplegar automáticamente la sección de capas para acceso inmediato
        if hasattr(self, 'sec_masks'):
            self.sec_masks.expand()
            if hasattr(self, 'left_scroll_area'):
                self.left_scroll_area.ensureWidgetVisible(self.sec_masks)

    def on_starnet_error(self, err_msg: str):
        self._set_ai_processing_state(False)
        self.log_message(f"[ERROR STARNET] {err_msg}")
        QMessageBox.warning(self, "Error en StarNet++", err_msg)
        
    def on_stars_slider_changed(self, val: int):
        """Ajusta la intensidad relativa de la capa de estrellas separada."""
        self.star_intensity = val / 100.0
        self.lbl_stars.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_value(val)
        self.update_stretch_preview()

    def _on_mask_widget_stars_intensity(self, val: int):
        if hasattr(self, 'slider_stars') and self.slider_stars.value() != val:
            self.slider_stars.setValue(val)

    def reset_stars_intensity(self):
        if hasattr(self, 'slider_stars'):
            self.slider_stars.setValue(100)
        self.star_intensity = 1.0
        if hasattr(self, 'lbl_stars'):
            self.lbl_stars.setText("100%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_value(100)
        self.update_stretch_preview()

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

    def on_temp_changed(self, val: int):
        self.temp_val = val / 1000.0
        if hasattr(self, 'spin_temp'):
            self.spin_temp.blockSignals(True)
            self.spin_temp.setValue(self.temp_val)
            self.spin_temp.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_temp_changed(self, val: float):
        self.temp_val = float(val)
        slider_val = int(round(val * 1000.0))
        if hasattr(self, 'slider_temp'):
            self.slider_temp.blockSignals(True)
            self.slider_temp.setValue(slider_val)
            self.slider_temp.blockSignals(False)
        self.update_stretch_preview()

    def on_tint_changed(self, val: int):
        self.tint_val = val / 1000.0
        if hasattr(self, 'spin_tint'):
            self.spin_tint.blockSignals(True)
            self.spin_tint.setValue(self.tint_val)
            self.spin_tint.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_tint_changed(self, val: float):
        self.tint_val = float(val)
        slider_val = int(round(val * 1000.0))
        if hasattr(self, 'slider_tint'):
            self.slider_tint.blockSignals(True)
            self.slider_tint.setValue(slider_val)
            self.slider_tint.blockSignals(False)
        self.update_stretch_preview()

    def reset_wb(self):
        self.slider_temp.blockSignals(True)
        self.slider_tint.blockSignals(True)
        if hasattr(self, 'spin_temp'):
            self.spin_temp.blockSignals(True)
            self.spin_temp.setValue(0.0)
            self.spin_temp.blockSignals(False)
        if hasattr(self, 'spin_tint'):
            self.spin_tint.blockSignals(True)
            self.spin_tint.setValue(0.0)
            self.spin_tint.blockSignals(False)
        self.slider_temp.setValue(0)
        self.slider_tint.setValue(0)
        self.temp_val = 0.0
        self.tint_val = 0.0
        self.slider_temp.blockSignals(False)
        self.slider_tint.blockSignals(False)
        self.update_stretch_preview()
        self._commit_action()

    def on_sat_sky_changed(self, val: int):
        self.sat_sky_val = val / 100.0
        self.lbl_sat_sky.setText(f"{self.sat_sky_val:.2f}x")
        self.update_stretch_preview()

    def on_sat_gnd_changed(self, val: int):
        self.sat_gnd_val = val / 100.0
        self.lbl_sat_gnd.setText(f"{self.sat_gnd_val:.2f}x")
        self.update_stretch_preview()

    def on_vibrance_changed(self, val: int):
        self.vibrance_val = val / 100.0
        self.lbl_vibrance.setText(f"{self.vibrance_val:+.2f}")
        self.update_stretch_preview()

    def reset_saturation(self):
        self.slider_sat_sky.blockSignals(True)
        self.slider_sat_gnd.blockSignals(True)
        self.slider_vibrance.blockSignals(True)
        self.slider_sat_sky.setValue(100)
        self.slider_sat_gnd.setValue(100)
        self.slider_vibrance.setValue(0)
        self.sat_sky_val = 1.0
        self.sat_gnd_val = 1.0
        self.vibrance_val = 0.0
        self.lbl_sat_sky.setText("1.00x")
        self.lbl_sat_gnd.setText("1.00x")
        self.lbl_vibrance.setText("0.00")
        self.slider_sat_sky.blockSignals(False)
        self.slider_sat_gnd.blockSignals(False)
        self.slider_vibrance.blockSignals(False)
        self.update_stretch_preview()

    def _set_curve_mode(self, mode: str) -> None:
        """Cambia el modo activo de la curva y sincroniza los botones de selección."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_mode(mode)
            self.btn_curve_rgb.setChecked(mode == "rgb")
            self.btn_curve_hue.setChecked(mode == "hue_sat")
            self.btn_curve_luma.setChecked(mode == "luma_sat")

    def on_contrast_changed(self, val: int):
        self.contrast_val = val / 100.0
        self.lbl_contrast.setText(f"{self.contrast_val:+.2f}")
        self.update_stretch_preview()

    def on_slider_bp_changed(self, val: int):
        bp = (val / 2000.0) * 0.20
        self.spin_bp.blockSignals(True)
        self.spin_bp.setValue(bp)
        self.spin_bp.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_bp_changed(self, val: float):
        slider_val = int(np.clip((val / 0.20) * 2000.0, 0, 2000))
        self.slider_bp.blockSignals(True)
        self.slider_bp.setValue(slider_val)
        self.slider_bp.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_mtf_changed(self, val: int):
        m = max(0.0001, (val / 2000.0) * 0.50)
        self.spin_mtf.blockSignals(True)
        self.spin_mtf.setValue(m)
        self.spin_mtf.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_mtf_changed(self, val: float):
        slider_val = int(np.clip((val / 0.50) * 2000.0, 1, 2000))
        self.slider_mtf.blockSignals(True)
        self.slider_mtf.setValue(slider_val)
        self.slider_mtf.blockSignals(False)
        self.update_stretch_preview()

    def apply_auto_mtf(self):
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return

        bp_suggested, m_suggested = calculate_mtf_params(target, target_background=0.20)
        self.sync_controls(bp_suggested, m_suggested)
        self.update_stretch_preview()
        self.log_message(f"Auto-MTF aplicado (Punto Negro: {bp_suggested:.5f}, MTF m: {m_suggested:.5f}).")

    def sync_controls(self, bp: float, m: float):
        self.spin_bp.blockSignals(True)
        self.slider_bp.blockSignals(True)
        self.spin_mtf.blockSignals(True)
        self.slider_mtf.blockSignals(True)

        self.spin_bp.setValue(bp)
        self.slider_bp.setValue(int(np.clip((bp / 0.20) * 2000.0, 0, 2000)))

        self.spin_mtf.setValue(m)
        self.slider_mtf.setValue(int(np.clip((m / 0.50) * 2000.0, 1, 2000)))

        self.spin_bp.blockSignals(False)
        self.slider_bp.blockSignals(False)
        self.spin_mtf.blockSignals(False)
        self.slider_mtf.blockSignals(False)

    def _set_stretch_mode(self, mode: str):
        """Alterna el modo de estirado entre MTF clásico y GHS hiperbólico."""
        if mode not in ("mtf", "ghs"):
            return
        self.stretch_mode = mode
        if hasattr(self, 'btn_mode_mtf'):
            self.btn_mode_mtf.blockSignals(True)
            self.btn_mode_mtf.setChecked(mode == "mtf")
            self.btn_mode_mtf.blockSignals(False)
        if hasattr(self, 'btn_mode_ghs'):
            self.btn_mode_ghs.blockSignals(True)
            self.btn_mode_ghs.setChecked(mode == "ghs")
            self.btn_mode_ghs.blockSignals(False)

        if hasattr(self, 'w_mtf_container'):
            self.w_mtf_container.setVisible(mode == "mtf")
        if hasattr(self, 'w_ghs_container'):
            self.w_ghs_container.setVisible(mode == "ghs")

        self.update_stretch_preview()
        self._commit_action()

    def _set_ghs_color_mode(self, mode: str):
        """Alterna el modo de color de GHS entre Luminancia y RGB Enlazado."""
        if mode not in ("luma", "rgb"):
            return
        self.ghs_color_mode = mode
        if hasattr(self, 'btn_ghs_luma'):
            self.btn_ghs_luma.blockSignals(True)
            self.btn_ghs_luma.setChecked(mode == "luma")
            self.btn_ghs_luma.blockSignals(False)
        if hasattr(self, 'btn_ghs_rgb'):
            self.btn_ghs_rgb.blockSignals(True)
            self.btn_ghs_rgb.setChecked(mode == "rgb")
            self.btn_ghs_rgb.blockSignals(False)

        self.update_stretch_preview()
        self._commit_action()

    def on_slider_ghs_sp_changed(self, val: int):
        sp = val / 10000.0
        self.spin_ghs_sp.blockSignals(True)
        self.spin_ghs_sp.setValue(sp)
        self.spin_ghs_sp.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_ghs_sp_changed(self, val: float):
        slider_val = int(np.clip(val * 10000.0, 0, 10000))
        self.slider_ghs_sp.blockSignals(True)
        self.slider_ghs_sp.setValue(slider_val)
        self.slider_ghs_sp.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_ghs_d_changed(self, val: int):
        d = val / 100.0
        self.spin_ghs_d.blockSignals(True)
        self.spin_ghs_d.setValue(d)
        self.spin_ghs_d.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_ghs_d_changed(self, val: float):
        slider_val = int(np.clip(val * 100.0, 0, 2000))
        self.slider_ghs_d.blockSignals(True)
        self.slider_ghs_d.setValue(slider_val)
        self.slider_ghs_d.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_ghs_b_changed(self, val: int):
        b = val / 100.0
        self.spin_ghs_b.blockSignals(True)
        self.spin_ghs_b.setValue(b)
        self.spin_ghs_b.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_ghs_b_changed(self, val: float):
        slider_val = int(np.clip(val * 100.0, 0, 1500))
        self.slider_ghs_b.blockSignals(True)
        self.slider_ghs_b.setValue(slider_val)
        self.slider_ghs_b.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_ghs_hp_changed(self, val: int):
        hp = val / 1000.0
        self.spin_ghs_hp.blockSignals(True)
        self.spin_ghs_hp.setValue(hp)
        self.spin_ghs_hp.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_ghs_hp_changed(self, val: float):
        slider_val = int(np.clip(val * 1000.0, 0, 1000))
        self.slider_ghs_hp.blockSignals(True)
        self.slider_ghs_hp.setValue(slider_val)
        self.slider_ghs_hp.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_ghs_lp_changed(self, val: int):
        lp = val / 1000.0
        self.spin_ghs_lp.blockSignals(True)
        self.spin_ghs_lp.setValue(lp)
        self.spin_ghs_lp.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_ghs_lp_changed(self, val: float):
        slider_val = int(np.clip(val * 1000.0, 0, 1000))
        self.slider_ghs_lp.blockSignals(True)
        self.slider_ghs_lp.setValue(slider_val)
        self.slider_ghs_lp.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_ghs_bp_changed(self, val: int):
        bp = (val / 2000.0) * 0.20
        self.spin_ghs_bp.blockSignals(True)
        self.spin_ghs_bp.setValue(bp)
        self.spin_ghs_bp.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_ghs_bp_changed(self, val: float):
        slider_val = int(np.clip((val / 0.20) * 2000.0, 0, 2000))
        self.slider_ghs_bp.blockSignals(True)
        self.slider_ghs_bp.setValue(slider_val)
        self.slider_ghs_bp.blockSignals(False)
        self.update_stretch_preview()

    def apply_auto_ghs(self):
        """Calcula y aplica parámetros óptimos de GHS según el histograma de la imagen base."""
        target = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
        if target is None:
            return

        params = estimate_ghs_params(target, target_background=0.18)

        widgets_to_block = [
            self.spin_ghs_sp, self.slider_ghs_sp,
            self.spin_ghs_d, self.slider_ghs_d,
            self.spin_ghs_b, self.slider_ghs_b,
            self.spin_ghs_hp, self.slider_ghs_hp,
            self.spin_ghs_lp, self.slider_ghs_lp,
            self.spin_ghs_bp, self.slider_ghs_bp
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        self.spin_ghs_sp.setValue(params["sp"])
        self.slider_ghs_sp.setValue(int(np.clip(params["sp"] * 10000.0, 0, 10000)))

        self.spin_ghs_d.setValue(params["d"])
        self.slider_ghs_d.setValue(int(np.clip(params["d"] * 100.0, 0, 2000)))

        self.spin_ghs_b.setValue(params["b"])
        self.slider_ghs_b.setValue(int(np.clip(params["b"] * 100.0, 0, 1500)))

        self.spin_ghs_hp.setValue(params["hp"])
        self.slider_ghs_hp.setValue(int(np.clip(params["hp"] * 1000.0, 0, 1000)))

        self.spin_ghs_lp.setValue(params["lp"])
        self.slider_ghs_lp.setValue(int(np.clip(params["lp"] * 1000.0, 0, 1000)))

        self.spin_ghs_bp.setValue(params["bp"])
        self.slider_ghs_bp.setValue(int(np.clip((params["bp"] / 0.20) * 2000.0, 0, 2000)))

        for w in widgets_to_block:
            w.blockSignals(False)

        self.log_message(
            f"[GHS] Auto-GHS aplicado: SP={params['sp']:.5f}, D={params['d']:.2f}, b={params['b']:.2f}, "
            f"BP={params['bp']:.5f}, HP={params['hp']:.2f}"
        )
        self.update_stretch_preview()
        self._commit_action()

    def reset_ghs(self):
        """Restablece los parámetros de GHS a sus valores predeterminados."""
        widgets_to_block = [
            self.spin_ghs_sp, self.slider_ghs_sp,
            self.spin_ghs_d, self.slider_ghs_d,
            self.spin_ghs_b, self.slider_ghs_b,
            self.spin_ghs_hp, self.slider_ghs_hp,
            self.spin_ghs_lp, self.slider_ghs_lp,
            self.spin_ghs_bp, self.slider_ghs_bp
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        self.spin_ghs_sp.setValue(0.02000)
        self.slider_ghs_sp.setValue(200)
        self.spin_ghs_d.setValue(5.00)
        self.slider_ghs_d.setValue(500)
        self.spin_ghs_b.setValue(2.00)
        self.slider_ghs_b.setValue(200)
        self.spin_ghs_hp.setValue(0.00)
        self.slider_ghs_hp.setValue(0)
        self.spin_ghs_lp.setValue(0.00)
        self.slider_ghs_lp.setValue(0)
        self.spin_ghs_bp.setValue(0.00000)
        self.slider_ghs_bp.setValue(0)

        for w in widgets_to_block:
            w.blockSignals(False)

        self.update_stretch_preview()
        self._commit_action()

    def reset_sliders(self):
        self.slider_contrast.blockSignals(True)
        self.slider_contrast.setValue(0)
        self.contrast_val = 0.0
        self.lbl_contrast.setText("0.00")
        self.slider_contrast.blockSignals(False)

        if getattr(self, "stretch_mode", "mtf") == "ghs":
            self.reset_ghs()
        else:
            if self.image_32bit is not None:
                self.apply_auto_mtf()
            else:
                self.sync_controls(0.0, 0.10)
            
    def on_canvas_zoom_toggled(self, is_zoomed: bool, rel_x: float, rel_y: float):
        self.is_zoomed_100 = is_zoomed
        if is_zoomed:
            self.zoom_rel_coords = (rel_x, rel_y)
        else:
            self._generate_before_reference()
        if hasattr(self, 'btn_toggle_zoom'):
            self.btn_toggle_zoom.blockSignals(True)
            self.btn_toggle_zoom.setChecked(is_zoomed)
            self.btn_toggle_zoom.blockSignals(False)
        self.update_stretch_preview()
        self._update_canvas_info_label()

    def on_extract_background_clicked(self):
        if self.image_32bit is None:
            return

        engine = self.combo_bg_engine.currentText()

        if "Polinómico" in engine:
            self.log_message("[FONDO] Extrayendo gradiente con modelo Polinómico Cuadrático...")
            try:
                corrected = extract_background_polynomial(
                    self.image_32bit, 
                    sky_mask=self.current_mask, 
                    degree=2
                )
                self.on_bg_extraction_success(corrected)
                self.log_message("[FONDO] Gradiente polinómico neutralizado con éxito.")
            except Exception as e:
                self.log_message(f"[ERROR FONDO] Fallo en cálculo polinómico: {e}")
        else:
            self._set_ai_processing_state(True)
            self.log_message("=== INICIANDO GRAXPERT ===")
            self.gx_worker = GraXpertWorker(
                self.image_32bit,
                sky_mask=self.current_mask,
                smoothing=self.spin_smoothing.value()
            )
            self.gx_worker.status_changed.connect(self.log_message)
            self.gx_worker.finished_success.connect(self.on_bg_extraction_success)
            self.gx_worker.error_occurred.connect(self.on_bg_extraction_error)
            self.gx_worker.start()

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
        
    def on_bg_extraction_error(self, err_msg: str):
        self._set_ai_processing_state(False)
        self.log_message(f"[ERROR GRAXPERT] {err_msg}")
        QMessageBox.warning(self, "Error en GraXpert", err_msg)

    def on_lp_changed(self, val: int):
        self.lp_reduction = val / 100.0
        self.lbl_lp.setText(f"{val}%")
        self.update_stretch_preview()

    def on_contrast_starless_changed(self, val: int):
        self.contrast_starless_val = val / 100.0
        self.lbl_contrast_starless.setText(f"{self.contrast_starless_val:+.2f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                val, self.slider_denoise.value()
            )
        self.update_stretch_preview()
        
    def on_clarity_changed(self, val: int):
        self.clarity_starless_val = val / 100.0
        self.lbl_clarity.setText(f"{self.clarity_starless_val:+.2f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                val, self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value()
            )
        self.update_stretch_preview()

    def on_dehaze_changed(self, val: int):
        self.dehaze_starless_val = val / 100.0
        self.lbl_dehaze.setText(f"{self.dehaze_starless_val:+.2f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), val,
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value()
            )
        self.update_stretch_preview()

    def on_denoise_changed(self, val: int):
        self.denoise_strength = val / 100.0
        self.lbl_denoise.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), val
            )
        self.update_stretch_preview()
        
    def on_ond_estructura_changed(self, val: int):
        self.multiescala_estructura = val / 100.0
        self.lbl_ond_est.setText(f"{self.multiescala_estructura:+.2f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                val, self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value()
            )
        self.update_stretch_preview()

    def on_ond_fondo_changed(self, val: int):
        self.multiescala_fondo = val / 100.0
        self.lbl_ond_bg.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), val,
                self.slider_contrast_starless.value(), self.slider_denoise.value()
            )
        self.update_stretch_preview()

    def _on_mask_widget_sl_clarity(self, val: int):
        if self.slider_clarity.value() != val:
            self.slider_clarity.setValue(val)

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

    def reset_starless_adjustments(self):
        widgets = [
            self.slider_clarity, self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_contrast_starless, self.slider_denoise
        ]
        for w in widgets:
            w.blockSignals(True)
        self.slider_clarity.setValue(0)
        self.slider_dehaze.setValue(0)
        self.slider_ond_est.setValue(0)
        self.slider_ond_bg.setValue(0)
        self.slider_contrast_starless.setValue(0)
        self.slider_denoise.setValue(0)
        for w in widgets:
            w.blockSignals(False)

        self.clarity_starless_val = 0.0
        self.dehaze_starless_val = 0.0
        self.multiescala_estructura = 0.0
        self.multiescala_fondo = 0.0
        self.contrast_starless_val = 0.0
        self.denoise_strength = 0.0

        self.lbl_clarity.setText("0.00")
        self.lbl_dehaze.setText("0.00")
        self.lbl_ond_est.setText("0.00")
        self.lbl_ond_bg.setText("0%")
        self.lbl_contrast_starless.setText("0.00")
        self.lbl_denoise.setText("0%")

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(0, 0, 0, 0, 0, 0)
        self.update_stretch_preview()
    
    def _on_mask_widget_gnd_ev(self, val: int):
        if self.slider_gnd_ev.value() != val:
            self.slider_gnd_ev.setValue(val)

    def _on_mask_widget_gnd_sh(self, val: int):
        if self.slider_gnd_sh.value() != val:
            self.slider_gnd_sh.setValue(val)

    def _on_mask_widget_gnd_bp(self, val: int):
        if self.slider_gnd_bp.value() != val:
            self.slider_gnd_bp.setValue(val)

    def on_gnd_ev_changed(self, val: int):
        self.gnd_ev_val = val / 100.0
        self.lbl_gnd_ev.setText(f"{self.gnd_ev_val:+.2f} EV")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                val, self.slider_gnd_sh.value(), self.slider_gnd_bp.value()
            )
        self.update_stretch_preview()

    def on_gnd_shadows_changed(self, val: int):
        self.gnd_shadows_val = val / 100.0
        self.lbl_gnd_sh.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                self.slider_gnd_ev.value(), val, self.slider_gnd_bp.value()
            )
        self.update_stretch_preview()

    def on_gnd_bp_changed(self, val: int):
        self.gnd_bp_val = - (val / 1000.0)
        display_val = val / 1000.0
        self.lbl_gnd_bp.setText(f"{display_val:+.3f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                self.slider_gnd_ev.value(), self.slider_gnd_sh.value(), val
            )
        self.update_stretch_preview()

    def reset_gnd_tone(self):
        self.slider_gnd_ev.blockSignals(True)
        self.slider_gnd_sh.blockSignals(True)
        self.slider_gnd_bp.blockSignals(True)

        self.slider_gnd_ev.setValue(0)
        self.slider_gnd_sh.setValue(0)
        self.slider_gnd_bp.setValue(0)

        self.gnd_ev_val = 0.0
        self.gnd_shadows_val = 0.0
        self.gnd_bp_val = 0.0

        self.lbl_gnd_ev.setText("0.00 EV")
        self.lbl_gnd_sh.setText("0%")
        self.lbl_gnd_bp.setText("0.000")

        self.slider_gnd_ev.blockSignals(False)
        self.slider_gnd_sh.blockSignals(False)
        self.slider_gnd_bp.blockSignals(False)

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(0, 0, 0)

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

    def export_psd_dialog(self):
        """Abre el diálogo de exportación configurado directamente en formato Photoshop por capas."""
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Abre o procesa primero una toma en el revelador.")
            return
        self.export_image(force_format="psd")

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
        self.log_message(f"Exportando imagen completa ({w_full}x{h_full} px)...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()

        try:
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

            processed = self._apply_pipeline_on_image(full_base, precomputed_mask=full_mask)

            ext = os.path.splitext(p)[1].lower()
            if ext == '.psd' or "Photoshop" in selected_filter:
                from core.psd_export import export_developer_psd
                from core.config_manager import get_config_val
                from core.icc_manager import get_icc_profile_bytes, normalize_colorspace_name

                self.log_message("[PSD] Preparando capas reveladas y estiradas en 16-bit...")

                # 1. Base estirada completa (imagen base original desarrollada sin capas superpuestas)
                stretched_base = self._apply_pipeline_on_image(
                    self.image_32bit, 
                    precomputed_mask=None, 
                    apply_mask_layers=False
                )

                # 2. Suelo desarrollado (con ajustes diferenciales de suelo/EV/sombras sin capas superpuestas)
                ground_dev = None
                if clean_m is not None and full_mask is not None:
                    zero_mask = np.zeros_like(full_mask)
                    ground_dev = self._apply_pipeline_on_image(
                        self.image_32bit, 
                        precomputed_mask=zero_mask, 
                        apply_mask_layers=False
                    )

                # 3. Capas StarNet desacopladas (Starless puro estirado y Estrellas estiradas)
                stretched_starless = None
                stretched_stars = None
                if self.image_starless is not None:
                    # Fondo starless procesado (con filtros de denoise, claridad, dehaze)
                    starless_full = self._process_starless_base(
                        self.image_starless,
                        mask_3d=full_mask,
                        is_full_res=True
                    )
                    # Estirar el fondo starless con el pipeline de revelado SIN añadir estrellas ni capas
                    stretched_starless = self._apply_pipeline_on_image(
                        starless_full,
                        precomputed_mask=full_mask if full_mask is not None else None,
                        apply_mask_layers=False
                    )

                    # Estirar la capa de estrellas directamente desde self.image_stars con MTF o GHS
                    st_img = self.image_stars.copy()
                    if getattr(self, "stretch_mode", "mtf") == "ghs":
                        st_stretched = np.clip(
                            apply_ghs_stretch(
                                st_img,
                                sp=self.spin_ghs_sp.value(),
                                d=self.spin_ghs_d.value(),
                                b=self.spin_ghs_b.value(),
                                hp=self.spin_ghs_hp.value(),
                                lp=self.spin_ghs_lp.value(),
                                bp=0.0,
                                color_mode="rgb"
                            ) * self.star_intensity,
                            0.0, 1.0
                        )
                    else:
                        st_stretched = np.clip(
                            manual_stretch(st_img, black_point=0.0, midtone=self.spin_mtf.value()) * self.star_intensity,
                            0.0, 1.0
                        )

                    # Aplicar ajustes cromáticos propios de la capa de estrellas si están definidos
                    if hasattr(self, 'mask_layers_widget'):
                        stars_layer = next((l for l in self.mask_layers_widget.layers if getattr(l, 'mask_type', '') == 'image_stars'), None)
                        if stars_layer is not None:
                            has_adj = (abs(stars_layer.temp) > 1e-4 or abs(stars_layer.tint) > 1e-4 or 
                                       abs(stars_layer.exposure) > 1e-3 or abs(stars_layer.contrast) > 1e-3 or 
                                       abs(stars_layer.saturation - 1.0) > 1e-3)
                            if has_adj:
                                from core.mask_layers import apply_local_adjustments_to_image
                                st_stretched = apply_local_adjustments_to_image(st_stretched, np.ones((h_full, w_full), dtype=np.float32), stars_layer)

                    # Restringir estrellas al cielo si hay máscara de horizonte (para anular cualquier residuo en el suelo)
                    if clean_m is not None:
                        st_stretched = st_stretched * clean_m[..., np.newaxis]

                    stretched_stars = np.ascontiguousarray(st_stretched, dtype=np.float32)

                # Obtener bytes del perfil de color ICC seleccionado en configuración
                target_cs = normalize_colorspace_name(get_config_val("icc_profile", "srgb"))
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
                    log_fn=self.log_message
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
                bgr8 = cv2.cvtColor((np.clip(processed, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(p, bgr8, [cv2.IMWRITE_JPEG_QUALITY, 96])
                desc = "JPEG 8-bit"
            elif "32-bit" in selected_filter:
                out32 = np.ascontiguousarray(np.clip(processed, 0.0, 1.0), dtype=np.float32)
                save_tiff_with_icc(p, out32, compression='zlib', photometric='rgb', source_color_space='srgb')
                desc = "TIFF 32-bit float (ICC)"
            else:
                u16 = np.ascontiguousarray((np.clip(processed, 0.0, 1.0) * 65535.0), dtype=np.uint16)
                save_tiff_with_icc(p, u16, compression='zlib', photometric='rgb', source_color_space='srgb')
                desc = "TIFF 16-bit (ICC)"

            from core.config_manager import get_config_val
            from core.icc_manager import normalize_colorspace_name
            target_cs = normalize_colorspace_name(get_config_val("icc_profile", "srgb"))
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
            self.unsetCursor()
            
    def reset_all_parameters(self):
        """Devuelve todos los controles y variables del revelador a su estado neutro original."""
        self.lp_reduction = 0.0
        self.clarity_starless_val = 0.0
        self.dehaze_starless_val = 0.0
        self.multiescala_estructura = 0.0
        self.multiescala_fondo = 0.0
        self.temp_val = 0.0
        self.tint_val = 0.0
        self.sat_sky_val = 1.0
        self.sat_gnd_val = 1.0
        self.vibrance_val = 0.0
        self.contrast_val = 0.0
        self.contrast_starless_val = 0.0
        self.star_intensity = 1.0
        self.view_layer_mode = 0
        self.denoise_strength = 0.0
        self.gnd_ev_val = 0.0
        self.gnd_shadows_val = 0.0
        self.gnd_bp_val = 0.0

        widgets_to_block = [
            self.slider_lp, self.slider_stars, self.slider_contrast_starless,
            self.slider_clarity, self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_denoise, 
            self.slider_temp, self.slider_tint, self.slider_sat_sky,
            self.slider_sat_gnd, self.slider_vibrance, self.slider_contrast,
            self.combo_layer, self.spin_bp, self.slider_bp,
            self.spin_mtf, self.slider_mtf,
            self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp,
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        self.slider_lp.setValue(0)
        self.lbl_lp.setText("0%")

        self.combo_layer.setCurrentIndex(0)

        self.slider_stars.setValue(100)
        self.lbl_stars.setText("100%")

        self.slider_contrast_starless.setValue(0)
        self.lbl_contrast_starless.setText("0.00")

        self.slider_clarity.setValue(0)
        self.lbl_clarity.setText("0.00")

        self.slider_dehaze.setValue(0)
        self.lbl_dehaze.setText("0.00")

        self.slider_ond_est.setValue(0)
        self.lbl_ond_est.setText("0.00")

        self.slider_ond_bg.setValue(0)
        self.lbl_ond_bg.setText("0%")

        self.slider_denoise.setValue(0)
        self.lbl_denoise.setText("0%")
        
        self._set_starnet_tools_enabled(False)
        if hasattr(self, 'sec_presets'):
            self.sec_presets.set_status_badge("")

        if hasattr(self, 'curve_widget'):
            self.curve_widget.reset_all_curves()

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_state([])

        self.slider_temp.setValue(0)
        if hasattr(self, 'spin_temp'):
            self.spin_temp.setValue(0.0)
        self.slider_tint.setValue(0)
        if hasattr(self, 'spin_tint'):
            self.spin_tint.setValue(0.0)

        self.slider_sat_sky.setValue(100)
        self.lbl_sat_sky.setText("1.00x")
        self.slider_sat_gnd.setValue(100)
        self.lbl_sat_gnd.setText("1.00x")

        self.slider_vibrance.setValue(0)
        self.lbl_vibrance.setText("0.00")

        self.slider_contrast.setValue(0)
        self.lbl_contrast.setText("0.00")

        self.slider_gnd_ev.setValue(0)
        self.lbl_gnd_ev.setText("0.00 EV")
        self.slider_gnd_sh.setValue(0)
        self.lbl_gnd_sh.setText("0%")
        self.slider_gnd_bp.setValue(0)
        self.lbl_gnd_bp.setText("0.000")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(0, 0, 0)
            self.mask_layers_widget.set_stars_special_value(100)
            self.mask_layers_widget.set_starless_tone_values(0, 0, 0, 0, 0, 0)

        self.spin_bp.setValue(0.0)
        self.slider_bp.setValue(0)
        self.spin_mtf.setValue(0.10)
        self.slider_mtf.setValue(400)
        
        self.is_zoomed_100 = False
        self.zoom_rel_coords = (0.5, 0.5)
        if hasattr(self, 'canvas'):
            self.canvas.is_zoomed = False

        for w in widgets_to_block:
            w.blockSignals(False)

        if hasattr(self, '_update_mask_status_ui'):
            self._update_mask_status_ui()

        if hasattr(self, '_commit_action'):
            self._commit_action()
            
    def clear_session(self):
        """Descarga por completo la sesión del revelador y fuerza recolección de memoria."""
        self.reset_all_parameters()

        self.image_32bit = None
        self.image_starless = None
        self.image_stars = None

        self.preview_proxy = None
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None

        self.current_mask = None
        self.active_filepath = None

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
            if self.canvas.crop_mode:
                self.canvas.set_crop_mode(False)
            self.canvas.set_layer_brush_enabled(False)
            self.canvas.set_active_layer_mask(None)
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
        self.txt_log.clear()
        gc.collect()
        
    def _set_starnet_tools_enabled(self, enabled: bool):
        """Habilita o deshabilita individualmente todos los controles que dependen de StarNet."""
        if hasattr(self, "sec_starnet"):
            if enabled:
                self.sec_starnet.set_status_badge("● Separadas", "#00e5ff")
            else:
                self.sec_starnet.set_status_badge("")

        if hasattr(self, "btn_go_to_masks_starnet"):
            self.btn_go_to_masks_starnet.setVisible(enabled)

        if hasattr(self, "lbl_starnet_info"):
            if enabled:
                self.lbl_starnet_info.setText(
                    "✅ Estrellas separadas correctamente.\n"
                    "Las capas '🌌 Fondo Nebular' y '⭐ Estrellas' están activas en el Gestor de Capas (Sección 9). "
                    "Selecciona cada capa allí para regular su presencia estelar, claridad, neblina, contraste y reducción de ruido."
                )
            else:
                self.lbl_starnet_info.setText(
                    "💡 Ejecuta StarNet para desacoplar las estrellas del fondo cósmico.\n"
                    "Se generarán automáticamente las capas '🌌 Fondo Nebular' y '⭐ Estrellas' en el Gestor de Capas (Sección 9), donde podrás ajustar de forma contextual su presencia, claridad, contraste y reducción de ruido."
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
        ]
        for ctrl in controls:
            if ctrl is not None:
                ctrl.setEnabled(enabled)

    # -------------------------------------------------------------
    # GESTOR DE HISTORIAL DE DESHACER / REHACER (UNDO / REDO)
    # -------------------------------------------------------------
    def _connect_undo_triggers(self):
        """Conecta los eventos de finalización de interacción para registrar acciones en el historial."""
        all_sliders = [
            self.slider_lp, self.slider_stars, self.slider_contrast_starless,
            self.slider_clarity, self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_denoise, 
            self.slider_temp, self.slider_tint, self.slider_sat_sky,
            self.slider_sat_gnd, self.slider_vibrance, self.slider_contrast,
            self.slider_bp, self.slider_mtf,
            self.slider_ghs_sp, self.slider_ghs_d, self.slider_ghs_b,
            self.slider_ghs_hp, self.slider_ghs_lp, self.slider_ghs_bp,
            self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp,
        ]
        for s in all_sliders:
            s.sliderReleased.connect(self._commit_action)

        self.spin_bp.editingFinished.connect(self._commit_action)
        self.spin_mtf.editingFinished.connect(self._commit_action)
        self.spin_ghs_sp.editingFinished.connect(self._commit_action)
        self.spin_ghs_d.editingFinished.connect(self._commit_action)
        self.spin_ghs_b.editingFinished.connect(self._commit_action)
        self.spin_ghs_hp.editingFinished.connect(self._commit_action)
        self.spin_ghs_lp.editingFinished.connect(self._commit_action)
        self.spin_ghs_bp.editingFinished.connect(self._commit_action)
        self.spin_temp.editingFinished.connect(self._commit_action)
        self.spin_tint.editingFinished.connect(self._commit_action)
        self.combo_layer.activated.connect(lambda _: self._commit_action())

        if hasattr(self, 'curve_widget'):
            self.curve_widget.curveCommitted.connect(self._commit_action)

    def _capture_state(self) -> dict:
        """Captura una instantánea paramétrica ligera (~1 KB) de todos los ajustes del revelador."""
        return {
            "stretch_mode": getattr(self, "stretch_mode", "mtf"),
            "ghs_color_mode": getattr(self, "ghs_color_mode", "luma"),
            "ghs_sp_val": float(self.spin_ghs_sp.value()) if hasattr(self, 'spin_ghs_sp') else 0.02,
            "ghs_d_val": float(self.spin_ghs_d.value()) if hasattr(self, 'spin_ghs_d') else 5.0,
            "ghs_b_val": float(self.spin_ghs_b.value()) if hasattr(self, 'spin_ghs_b') else 2.0,
            "ghs_hp_val": float(self.spin_ghs_hp.value()) if hasattr(self, 'spin_ghs_hp') else 0.0,
            "ghs_lp_val": float(self.spin_ghs_lp.value()) if hasattr(self, 'spin_ghs_lp') else 0.0,
            "ghs_bp_val": float(self.spin_ghs_bp.value()) if hasattr(self, 'spin_ghs_bp') else 0.0,
            "lp_val": self.slider_lp.value(),
            "layer_idx": self.combo_layer.currentIndex(),
            "stars_val": self.slider_stars.value(),
            "contrast_starless_val": self.slider_contrast_starless.value(),
            "clarity_val": self.slider_clarity.value(),
            "dehaze_val": self.slider_dehaze.value(),
            "ond_est_val": self.slider_ond_est.value(),
            "ond_bg_val": self.slider_ond_bg.value(),
            "denoise_val": self.slider_denoise.value(),
            "temp_val": self.slider_temp.value(),
            "tint_val": self.slider_tint.value(),
            "sat_sky_val": self.slider_sat_sky.value(),
            "sat_gnd_val": self.slider_sat_gnd.value(),
            "vibrance_val": self.slider_vibrance.value(),
            "contrast_val": self.slider_contrast.value(),
            "bp_val": float(self.spin_bp.value()),
            "mtf_val": float(self.spin_mtf.value()),
            "gnd_ev_val": self.slider_gnd_ev.value(),
            "gnd_sh_val": self.slider_gnd_sh.value(),
            "gnd_bp_val": self.slider_gnd_bp.value(),
            "curves": self.curve_widget.get_state() if hasattr(self, 'curve_widget') else None,
            "mask_layers": self.mask_layers_widget.get_state() if hasattr(self, 'mask_layers_widget') else None,
        }

    def _state_different(self, s1: Optional[dict], s2: Optional[dict]) -> bool:
        """Comprueba si dos instantáneas paramétricas presentan diferencias efectivas."""
        if s1 is None or s2 is None:
            return True
        for k, v1 in s1.items():
            v2 = s2.get(k)
            if k in ("curves", "mask_layers"):
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
        if self._is_undoing_or_redoing or self._last_committed_state is None:
            return
        current = self._capture_state()
        if self._state_different(self._last_committed_state, current):
            self.undo_stack.append(self._last_committed_state)
            if len(self.undo_stack) > self.MAX_UNDO_STEPS:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self._last_committed_state = current
            self._update_undo_redo_buttons()

    def undo(self):
        """Deshace el último ajuste paramétrico o recorte realizado (Ctrl+Z)."""
        if not self.undo_stack or self._is_undoing_or_redoing:
            return
        self._is_undoing_or_redoing = True
        try:
            target = self.undo_stack.pop()
            if target.get("type") == "crop":
                redo_item = {
                    "type": "crop",
                    "image_32bit": self.image_32bit.copy(),
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                    "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
                }
                self.redo_stack.append(redo_item)
                self.image_32bit = target["image_32bit"]
                self.image_starless = target["image_starless"]
                self.image_stars = target["image_stars"]
                self.current_mask = target["current_mask"]

                self.snapshot_before_pixmap = None
                self.has_custom_snapshot = False
                h, w = self.image_32bit.shape[:2]
                self.canvas.set_native_image_size((w, h))
                self._generate_preview_proxy()
                self._generate_before_reference()
                self.update_stretch_preview()
                self._update_canvas_info_label()
                self._update_mask_status_ui()
                self.log_message(f"↺ Deshacer recorte: Restaurada imagen original ({w}x{h} px)")
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
                self.update_stretch_preview()
                self._update_mask_status_ui()
                self.log_message("↺ Deshacer: Cambio de máscara revertido")
            else:
                current = self._capture_state()
                self.redo_stack.append(current)
                self._restore_state(target)
                self._last_committed_state = target
                self.log_message(f"↺ Deshacer aplicado ({len(self.undo_stack)} pasos restantes)")
        finally:
            self._is_undoing_or_redoing = False
            self._update_undo_redo_buttons()

    def redo(self):
        """Rehace el ajuste paramétrico o recorte previamente deshecho (Ctrl+Y / Ctrl+Shift+Z)."""
        if not self.redo_stack or self._is_undoing_or_redoing:
            return
        self._is_undoing_or_redoing = True
        try:
            target = self.redo_stack.pop()
            if target.get("type") == "crop":
                undo_item = {
                    "type": "crop",
                    "image_32bit": self.image_32bit.copy(),
                    "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
                    "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
                    "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
                }
                self.undo_stack.append(undo_item)
                self.image_32bit = target["image_32bit"]
                self.image_starless = target["image_starless"]
                self.image_stars = target["image_stars"]
                self.current_mask = target["current_mask"]

                self.snapshot_before_pixmap = None
                self.has_custom_snapshot = False
                h, w = self.image_32bit.shape[:2]
                self.canvas.set_native_image_size((w, h))
                self._generate_preview_proxy()
                self._generate_before_reference()
                self.update_stretch_preview()
                self._update_canvas_info_label()
                self._update_mask_status_ui()
                self.log_message(f"↻ Rehacer recorte: Recorte restaurado ({w}x{h} px)")
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
                self.update_stretch_preview()
                self._update_mask_status_ui()
                self.log_message("↻ Rehacer: Cambio de máscara restablecido")
            else:
                current = self._capture_state()
                self.undo_stack.append(current)
                self._restore_state(target)
                self._last_committed_state = target
                self.log_message(f"↻ Rehacer aplicado ({len(self.redo_stack)} pasos por rehacer)")
        finally:
            self._is_undoing_or_redoing = False
            self._update_undo_redo_buttons()

    def _restore_state(self, state: dict) -> None:
        """Restaura todos los controles al estado paramétrico especificado y recalcula la vista."""
        if not state:
            return

        widgets_to_block = [
            self.slider_lp, self.slider_stars, self.slider_contrast_starless,
            self.slider_clarity, self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_denoise, 
            self.slider_temp, self.slider_tint, self.spin_temp, self.spin_tint,
            self.slider_sat_sky, self.slider_sat_gnd, self.slider_vibrance,
            self.slider_contrast, self.combo_layer, self.spin_bp, self.slider_bp,
            self.spin_mtf, self.slider_mtf,
            self.spin_ghs_sp, self.slider_ghs_sp,
            self.spin_ghs_d, self.slider_ghs_d,
            self.spin_ghs_b, self.slider_ghs_b,
            self.spin_ghs_hp, self.slider_ghs_hp,
            self.spin_ghs_lp, self.slider_ghs_lp,
            self.spin_ghs_bp, self.slider_ghs_bp,
            self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp,
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        if "stretch_mode" in state:
            self._set_stretch_mode(state["stretch_mode"])

        if "ghs_color_mode" in state:
            self._set_ghs_color_mode(state["ghs_color_mode"])

        if "ghs_sp_val" in state and hasattr(self, 'spin_ghs_sp'):
            self.spin_ghs_sp.setValue(state["ghs_sp_val"])
            self.slider_ghs_sp.setValue(int(np.clip(state["ghs_sp_val"] * 10000.0, 0, 10000)))

        if "ghs_d_val" in state and hasattr(self, 'spin_ghs_d'):
            self.spin_ghs_d.setValue(state["ghs_d_val"])
            self.slider_ghs_d.setValue(int(np.clip(state["ghs_d_val"] * 100.0, 0, 2000)))

        if "ghs_b_val" in state and hasattr(self, 'spin_ghs_b'):
            self.spin_ghs_b.setValue(state["ghs_b_val"])
            self.slider_ghs_b.setValue(int(np.clip(state["ghs_b_val"] * 100.0, 0, 1500)))

        if "ghs_hp_val" in state and hasattr(self, 'spin_ghs_hp'):
            self.spin_ghs_hp.setValue(state["ghs_hp_val"])
            self.slider_ghs_hp.setValue(int(np.clip(state["ghs_hp_val"] * 1000.0, 0, 1000)))

        if "ghs_lp_val" in state and hasattr(self, 'spin_ghs_lp'):
            self.spin_ghs_lp.setValue(state["ghs_lp_val"])
            self.slider_ghs_lp.setValue(int(np.clip(state["ghs_lp_val"] * 1000.0, 0, 1000)))

        if "ghs_bp_val" in state and hasattr(self, 'spin_ghs_bp'):
            self.spin_ghs_bp.setValue(state["ghs_bp_val"])
            self.slider_ghs_bp.setValue(int(np.clip((state["ghs_bp_val"] / 0.20) * 2000.0, 0, 2000)))

        if "lp_val" in state:
            self.slider_lp.setValue(state["lp_val"])
            self.lp_reduction = state["lp_val"] / 100.0
            self.lbl_lp.setText(f"{state['lp_val']}%")

        if "layer_idx" in state:
            self.combo_layer.setCurrentIndex(state["layer_idx"])
            self.view_layer_mode = state["layer_idx"]

        if "stars_val" in state:
            self.slider_stars.setValue(state["stars_val"])
            self.star_intensity = state["stars_val"] / 100.0
            self.lbl_stars.setText(f"{state['stars_val']}%")

        if "contrast_starless_val" in state:
            self.slider_contrast_starless.setValue(state["contrast_starless_val"])
            self.contrast_starless_val = state["contrast_starless_val"] / 100.0
            self.lbl_contrast_starless.setText(f"{self.contrast_starless_val:+.2f}")

        if "clarity_val" in state:
            self.slider_clarity.setValue(state["clarity_val"])
            self.clarity_starless_val = state["clarity_val"] / 100.0
            self.lbl_clarity.setText(f"{self.clarity_starless_val:+.2f}")

        if "dehaze_val" in state:
            self.slider_dehaze.setValue(state["dehaze_val"])
            self.dehaze_starless_val = state["dehaze_val"] / 100.0
            self.lbl_dehaze.setText(f"{self.dehaze_starless_val:+.2f}")

        if "ond_est_val" in state:
            self.slider_ond_est.setValue(state["ond_est_val"])
            self.multiescala_estructura = state["ond_est_val"] / 100.0
            self.lbl_ond_est.setText(f"{self.multiescala_estructura:+.2f}")

        if "ond_bg_val" in state:
            self.slider_ond_bg.setValue(state["ond_bg_val"])
            self.multiescala_fondo = state["ond_bg_val"] / 100.0
            self.lbl_ond_bg.setText(f"{state['ond_bg_val']}%")

        if "denoise_val" in state:
            self.slider_denoise.setValue(state["denoise_val"])
            self.denoise_strength = state["denoise_val"] / 100.0
            self.lbl_denoise.setText(f"{state['denoise_val']}%")

        if "temp_val" in state:
            self.slider_temp.setValue(state["temp_val"])
            self.temp_val = state["temp_val"] / 1000.0
            if hasattr(self, 'spin_temp'):
                self.spin_temp.setValue(self.temp_val)

        if "tint_val" in state:
            self.slider_tint.setValue(state["tint_val"])
            self.tint_val = state["tint_val"] / 1000.0
            if hasattr(self, 'spin_tint'):
                self.spin_tint.setValue(self.tint_val)

        if "sat_sky_val" in state:
            self.slider_sat_sky.setValue(state["sat_sky_val"])
            self.sat_sky_val = state["sat_sky_val"] / 100.0
            self.lbl_sat_sky.setText(f"{self.sat_sky_val:.2f}x")

        if "sat_gnd_val" in state:
            self.slider_sat_gnd.setValue(state["sat_gnd_val"])
            self.sat_gnd_val = state["sat_gnd_val"] / 100.0
            self.lbl_sat_gnd.setText(f"{self.sat_gnd_val:.2f}x")

        if "vibrance_val" in state:
            self.slider_vibrance.setValue(state["vibrance_val"])
            self.vibrance_val = state["vibrance_val"] / 100.0
            self.lbl_vibrance.setText(f"{self.vibrance_val:+.2f}")

        if "contrast_val" in state:
            self.slider_contrast.setValue(state["contrast_val"])
            self.contrast_val = state["contrast_val"] / 100.0
            self.lbl_contrast.setText(f"{self.contrast_val:+.2f}")

        if "bp_val" in state:
            self.spin_bp.setValue(state["bp_val"])
            self.slider_bp.setValue(int(np.clip((state["bp_val"] / 0.20) * 2000.0, 0, 2000)))

        if "mtf_val" in state:
            self.spin_mtf.setValue(state["mtf_val"])
            self.slider_mtf.setValue(int(np.clip((state["mtf_val"] / 0.50) * 2000.0, 1, 2000)))

        if "gnd_ev_val" in state:
            self.slider_gnd_ev.setValue(state["gnd_ev_val"])
            self.gnd_ev_val = state["gnd_ev_val"] / 100.0
            self.lbl_gnd_ev.setText(f"{self.gnd_ev_val:+.2f} EV")

        if "gnd_sh_val" in state:
            self.slider_gnd_sh.setValue(state["gnd_sh_val"])
            self.gnd_shadows_val = state["gnd_sh_val"] / 100.0
            self.lbl_gnd_sh.setText(f"{state['gnd_sh_val']}%")

        if "gnd_bp_val" in state:
            self.slider_gnd_bp.setValue(state["gnd_bp_val"])
            self.gnd_bp_val = state["gnd_bp_val"] / 1000.0
            self.lbl_gnd_bp.setText(f"{self.gnd_bp_val:+.3f}")

        for w in widgets_to_block:
            w.blockSignals(False)

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                self.slider_gnd_ev.value(),
                self.slider_gnd_sh.value(),
                self.slider_gnd_bp.value()
            )
            self.mask_layers_widget.set_stars_special_value(self.slider_stars.value())
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(),
                self.slider_dehaze.value(),
                self.slider_ond_est.value(),
                self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(),
                self.slider_denoise.value()
            )

        if "curves" in state and state["curves"] is not None and hasattr(self, 'curve_widget'):
            self.curve_widget.blockSignals(True)
            self.curve_widget.set_state(state["curves"])
            self.curve_widget.blockSignals(False)

        if "mask_layers" in state and hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.blockSignals(True)
            self.mask_layers_widget.set_state(state["mask_layers"] or [])
            self.mask_layers_widget.blockSignals(False)
            sel_layer = self.mask_layers_widget.get_selected_layer()
            if sel_layer and sel_layer.mask_type == "brush" and sel_layer.enabled:
                show_overlay = self.mask_layers_widget.btn_preview_mask.isChecked()
                self.canvas.set_active_layer_mask(sel_layer.brush_mask, show_overlay=show_overlay)
            elif not self.mask_layers_widget.btn_preview_mask.isChecked():
                self.canvas.set_refined_mask(None)
                self.canvas.set_mask_visible(False)

        self.update_stretch_preview()

    def _update_undo_redo_buttons(self):
        """Actualiza la disponibilidad visual y tooltips de los botones Deshacer y Rehacer."""
        has_undo = len(self.undo_stack) > 0
        has_redo = len(self.redo_stack) > 0
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setEnabled(has_undo)
            if has_undo:
                last_type = self.undo_stack[-1].get("type", "param")
                desc = "recorte" if last_type == "crop" else "ajuste"
                self.btn_undo.setToolTip(f"Deshacer último {desc} (Ctrl+Z) — {len(self.undo_stack)} pasos en historial")
            else:
                self.btn_undo.setToolTip("Deshacer (Ctrl+Z) — Sin acciones para deshacer")

        if hasattr(self, 'btn_redo'):
            self.btn_redo.setEnabled(has_redo)
            if has_redo:
                next_type = self.redo_stack[-1].get("type", "param")
                desc = "recorte" if next_type == "crop" else "ajuste"
                self.btn_redo.setToolTip(f"Rehacer {desc} (Ctrl+Y / Ctrl+Shift+Z) — {len(self.redo_stack)} pasos disponibles")
            else:
                self.btn_redo.setToolTip("Rehacer (Ctrl+Y) — Sin acciones para rehacer")

    # -------------------------------------------------------------
    # GESTOR DE PRESETS DE REVELADO (AJUSTES PREESTABLECIDOS)
    # -------------------------------------------------------------
    def _populate_presets_combo(self, select_preset_id: Optional[str] = None):
        """Carga y categoriza los presets en el desplegable con separadores elegantes."""
        if not hasattr(self, "combo_presets"):
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
        preset = self.combo_presets.itemData(index, Qt.UserRole)
        if isinstance(preset, DeveloperPreset):
            self.lbl_preset_desc.setText(preset.description or "")
            self.btn_apply_preset.setEnabled(True)
        else:
            self.lbl_preset_desc.setText("")
            self.btn_apply_preset.setEnabled(False)

    def apply_selected_preset(self):
        """Aplica el preset actualmente seleccionado en el desplegable."""
        preset = self.combo_presets.currentData(Qt.UserRole)
        if not isinstance(preset, DeveloperPreset):
            return
        scope = self.combo_preset_scope.currentData() or SCOPE_ALL
        self.apply_preset(preset, scope)

    def apply_preset(self, preset: DeveloperPreset, scope: str = SCOPE_ALL):
        """Aplica los parámetros de un preset a la imagen actual registrando la acción en el historial de Deshacer."""
        if not preset or not preset.params:
            return

        # 1. Registrar el estado previo para poder deshacer con Ctrl+Z
        current_state = self._capture_state()
        base_state = self._last_committed_state if self._last_committed_state is not None else current_state
        self.undo_stack.append(base_state)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

        # 2. Filtrar y fusionar parámetros según el ámbito elegido
        target_state = filter_preset_params(preset.params, current_state, scope)

        # 3. Restaurar parámetros en todos los controles
        self._restore_state(target_state)
        self._last_committed_state = self._capture_state()
        self._update_undo_redo_buttons()

        # 4. Mensaje explicativo en el log y badge de estado
        scope_names = {
            SCOPE_ALL: "Todo (Sin estirado)",
            SCOPE_TONE: "Solo Contraste y Suelo",
            SCOPE_COLOR: "Solo Color",
            SCOPE_DETAILS: "Solo Estrellas/Detalles"
        }
        scope_str = scope_names.get(scope, scope)
        self.log_message(f"🎨 Preset aplicado: '{preset.name}' [{scope_str}]")

        if hasattr(self, 'sec_presets'):
            if preset and preset.name and "Neutro" not in preset.name:
                self.sec_presets.set_status_badge(f"● {preset.name}", "#80d8ff")
            else:
                self.sec_presets.set_status_badge("")

    def open_save_preset_dialog(self):
        """Abre el diálogo para guardar los ajustes actuales como un preset."""
        current_params = self._capture_state()
        builtins = get_builtin_presets()
        users = load_user_presets()
        categories = list(set(p.category for p in builtins + users))

        dlg = SavePresetDialog(current_params, categories, self)
        if dlg.exec() == QDialog.Accepted and dlg.created_preset:
            self._populate_presets_combo(select_preset_id=dlg.created_preset.id)
            self.log_message(f"💾 Nuevo preset guardado con éxito: '{dlg.created_preset.name}' ({dlg.created_preset.category})")

    def open_preset_manager_dialog(self):
        """Abre el diálogo de gestión, importación y exportación de presets."""
        current_params = self._capture_state()
        dlg = PresetManagerDialog(current_params, self)
        dlg.apply_preset_requested.connect(self.apply_preset)
        dlg.exec()
        curr_preset = self.combo_presets.currentData(Qt.UserRole)
        curr_id = curr_preset.id if isinstance(curr_preset, DeveloperPreset) else None
        self._populate_presets_combo(select_preset_id=curr_id)