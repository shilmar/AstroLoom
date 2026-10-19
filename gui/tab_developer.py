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
    QProgressDialog,
    QCheckBox
)
from PySide6.QtCore import Qt, QCoreApplication, QTimer, QEvent
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
from core.light_pollution import (
    compute_light_pollution_dome,
    apply_parametric_light_pollution,
    estimate_light_pollution_center,
    LP_PRESET_TINTS
)
from core.orton import (
    apply_astronomical_orton,
    ORTON_PRESET_TINTS
)
from core.star_diffusion import apply_star_diffusion
from core.deconvolution import (
    generate_moffat_psf,
    generate_gaussian_psf,
    estimate_psf_from_stars,
    richardson_lucy_deconvolve
)
from core.masking import refine_mask_guided
from core.mask_layers import (
    MaskLayer, apply_all_mask_layers,
    MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS,
    MASK_TYPE_IMAGE_GROUND, MASK_TYPE_BRUSH, MASK_TYPE_LUMA, MASK_TYPE_COLOR,
    MASK_TYPE_GROUND, MASK_TYPE_SKY,
    apply_pre_curve_adjustments, get_canvas_below_layer, compute_effective_layer_mask
)
from gui.widgets.curve_widget import CurveWidget
from gui.widgets.live_histogram import LiveRGBHistogramWidget
from gui.widgets.mask_layers_widget import MaskLayersWidget
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow, PrecisionSpinBox
from gui.icons import get_icon, get_svg_pixmap
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
from gui.developer.panels.panel_color_wb import ColorWbPanel
from gui.developer.panels.panel_deconvolution import DeconvolutionPanel
from gui.developer.panels.panel_light_pollution import LightPollutionPanel
from gui.developer.panels.panel_stretching import StretchingPanel
from gui.developer.panels.panel_effects import EffectsPanel
from gui.developer.panels.panel_presets import PresetsPanel
from gui.developer.panels.panel_ground_sky import GroundSkyPanel
from gui.developer.panels.panel_starnet import StarNetPanel
from gui.developer.panels.panel_crop import CropToolbar
from gui.developer.developer_state import DeveloperState
from gui.developer.pipeline import ResolutionPyramid, apply_developer_pipeline
from gui.developer.async_worker import AsyncPipelineManager


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
        self.current_color_space = "srgb"
        self.gx_worker = None
        self.sn_worker = None
        
        self.is_zoomed_100 = False
        self.zoom_rel_coords = (0.5, 0.5)
        
        self.multiescala_estructura = 0.0   # -1.0 a +1.0
        self.multiescala_fondo = 0.0        # 0.0 a 1.0

        # Parámetros de Reducción Paramétrica de Contaminación Lumínica (Cúpula de Luz 2D)
        self.lp_reduction: float = 0.0              # 0.0 a 1.0 (0% a 100%)
        self.lp_center_x: float = 0.50              # Centro horizontal [0.0, 1.0]
        self.lp_center_y: float = 0.90              # Centro vertical (horizonte) [0.0, 1.0]
        self.lp_width: float = 0.60                 # Ancho horizontal de la cúpula
        self.lp_height: float = 0.45                # Alto vertical de la cúpula
        self.lp_tint_type: str = "sodium"           # "sodium", "warm_led", "cold_led", "neutral", "custom"
        self.lp_custom_tint: tuple = (1.0, 0.68, 0.22)
        self.lp_tint_strength: float = 0.50         # Grado de neutralización cromática (0.0 a 1.0)
        self.lp_mode: str = "attenuate"             # "attenuate" o "subtractive"
        self.lp_legacy_1d: bool = False             # Compatibilidad con rampa 1D

        # Parámetros de Efecto Orton Astronómico y Glow de Polvo Galáctico
        self.orton_strength: float = 0.0            # 0.0 a 1.0 (0% a 100%)
        self.orton_radius: float = 40.0             # Radio de difusión en px (10 a 150)
        self.orton_threshold: float = 0.05          # Umbral de luminancia de polvo (0.00 a 0.50)
        self.orton_tint_type: str = "warm_gold"     # "warm_gold", "neutral", "cool_blue", "halpha"
        self.orton_custom_tint: tuple = (1.0, 0.85, 0.55)
        self.orton_tint_strength: float = 0.40      # Grado de coloración del glow (0.0 a 1.0)
        self.orton_mode: str = "soft_light"         # "soft_light", "screen", "additive"

        # Parámetros de Filtro Difusor y Realce de Constelaciones (Star Mist)
        self.star_diff_strength: float = 0.0        # 0.0 a 1.0 (0% a 100%)
        self.star_diff_radius: float = 20.0         # Radio de difusión en px (5 a 60)
        self.star_diff_threshold: float = 0.45      # Umbral relativo de magnitud estelar (0.15 a 0.85)
        self.star_diff_color_boost: float = 0.50    # Saturación cromática espectral (0.0 a 1.0)
        self.star_diff_mode: str = "screen"         # "screen", "soft_light", "additive"

        # Parámetros de Deconvolución Óptica y Enfoque (Richardson-Lucy regularizado)
        self.deconv_iterations: int = 0             # 0 = Desactivado, 1 a 40 iteraciones
        self.deconv_fwhm: float = 3.00              # FWHM de la PSF en píxeles (1.0 a 8.0)
        self.deconv_model: str = "moffat"           # "moffat" o "gaussian"
        self.deconv_beta: float = 4.5               # Pendiente de alas Moffat (1.5 a 8.0)
        self.deconv_aspect_ratio: float = 1.00      # Relación de aspecto (1.00 a 2.50)
        self.deconv_angle_deg: float = 0.0          # Orientación de la elipse (-90.0 a +90.0)
        self.deconv_deringing: float = 0.008        # Protección de anillos oscuros (0.000 a 0.040)
        self.deconv_damping: float = 0.25           # Amortiguación bayesiana de ruido (0.00 a 1.00)
        self.deconv_luma_only: bool = True          # Deconvolución solo en Luminancia
        self.deconv_scope: str = "all"              # "all", "stars_only", "background_only"
        self._cached_psf: Optional[np.ndarray] = None
        self._cached_psf_key: tuple = ()

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
        self._eyedropper_target: str = "wb"  # "wb", "ghs_sp", "lp_center", "lp_color", "mask_color"
        self._mask_source_name: str = ""     # Origen de la máscara de suelo ("IA", "Trazos", "Archivo")

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

        # Temporizadores de renderizado reactivo y coalescencia de eventos (60 FPS)
        self._preview_throttle_timer = QTimer(self)
        self._preview_throttle_timer.setSingleShot(True)
        self._preview_throttle_timer.setInterval(16)
        self._preview_throttle_timer.timeout.connect(self._do_update_stretch_preview)

        self._hist_throttle_timer = QTimer(self)
        self._hist_throttle_timer.setSingleShot(True)
        self._hist_throttle_timer.setInterval(70)
        self._hist_throttle_timer.timeout.connect(self._refresh_histograms_now)

        # Temporizador de refinamiento progresivo (Draft 60 FPS -> Alta Calidad)
        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(120)
        self._refine_timer.timeout.connect(self._on_refine_preview_timeout)

        # Pirámides multi-escala para previsualización instantánea (ResolutionPyramid)
        self.pyramid_image: Optional[ResolutionPyramid] = None
        self.pyramid_starless: Optional[ResolutionPyramid] = None
        self.pyramid_stars: Optional[ResolutionPyramid] = None
        self.draft_proxy: Optional[np.ndarray] = None
        self.draft_starless: Optional[np.ndarray] = None
        self.draft_stars: Optional[np.ndarray] = None
        self.draft_mask: Optional[np.ndarray] = None

        self._is_rendering_preview: bool = False
        self._last_rendered_preview: Optional[np.ndarray] = None

        # Cachés multinivel de aceleración en memoria RAM
        self._cache_starless_key: Optional[tuple] = None
        self._cache_starless_linear: Optional[np.ndarray] = None
        self._cache_starless_pre_key: Optional[tuple] = None
        self._cache_starless_pre_img: Optional[np.ndarray] = None

        # Cachés por hitos del pipeline (Milestone Caching)
        self._cache_ms_stretch_key: Optional[tuple] = None
        self._cache_ms_stretch_img: Optional[np.ndarray] = None
        self._cache_ms_graded_key: Optional[tuple] = None
        self._cache_ms_graded_img: Optional[np.ndarray] = None
        self._cache_ms_spatial_key: Optional[tuple] = None
        self._cache_ms_spatial_img: Optional[np.ndarray] = None
        self._cache_ms_stars_key: Optional[tuple] = None
        self._cache_ms_stars_img: Optional[np.ndarray] = None

        self._cache_pre_layers_key: Optional[tuple] = None
        self._cache_pre_layers_stretched: Optional[np.ndarray] = None
        self._cache_pre_layers_stars: Optional[np.ndarray] = None
        self._is_scrubbing_heavy_slider: bool = False
        self._is_slider_pressed: bool = False
        self._is_dragging_deconv: bool = False
        self.async_pipeline = AsyncPipelineManager(self)

        # Memoria de tamaños para paneles ocultables (Lightroom style)
        self._last_left_panel_width = 380
        self._last_right_panel_width = 360

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
        self.left_container = QWidget()
        left_container_layout = QVBoxLayout(self.left_container)
        left_container_layout.setContentsMargins(4, 4, 4, 4)
        left_container_layout.setSpacing(6)
        self.left_container.setMinimumWidth(280)
        self.left_container.setMaximumWidth(1200)

        # Carga externa (Siempre fijada arriba)
        self.btn_open_img = QPushButton("Abrir Imagen (TIFF / FITS / RAW)...")
        self.btn_open_img.setIcon(get_icon("folder", color="#38bdf8", size=16))
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
        left_panel.setMinimumWidth(0)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        # --- SECCIÓN 1: Estilos y Presets de Revelado (Módulo gui.developer.panels.PresetsPanel) ---
        self.panel_presets = PresetsPanel()
        self.sec_presets = self.panel_presets.sec_presets
        self.combo_presets = self.panel_presets.combo_presets
        self.lbl_preset_desc = self.panel_presets.lbl_preset_desc
        self.combo_preset_scope = self.panel_presets.combo_preset_scope
        self.btn_apply_preset = self.panel_presets.btn_apply_preset
        self.btn_save_preset = self.panel_presets.btn_save_preset
        self.btn_manage_presets = self.panel_presets.btn_manage_presets

        self.panel_presets.preset_changed.connect(self._on_preset_combo_changed)
        self.panel_presets.apply_requested.connect(self.apply_selected_preset)
        self.panel_presets.save_requested.connect(self.open_save_preset_dialog)
        self.panel_presets.manage_requested.connect(self.open_preset_manager_dialog)

        left_layout.addWidget(self.panel_presets)

        # --- SECCIÓN 2: Extracción de Fondo y Polución Lumínica (Módulo gui.developer.panels.LightPollutionPanel) ---
        self.panel_lp = LightPollutionPanel()
        self.sec_gradients = self.panel_lp.sec_bg
        self.sec_bg = self.panel_lp.sec_bg
        self.combo_bg_engine = self.panel_lp.combo_bg_engine
        self.lbl_smoothing = self.panel_lp.lbl_smoothing
        self.spin_smoothing = self.panel_lp.spin_smoothing
        self.btn_extract_bg = self.panel_lp.btn_extract_bg
        self.row_lp = self.panel_lp.row_lp
        self.slider_lp = self.panel_lp.slider_lp
        self.lbl_lp = self.panel_lp.lbl_lp
        self.btn_lp_expand = self.panel_lp.btn_lp_expand
        self.btn_lp_auto = self.panel_lp.btn_lp_auto
        self.btn_lp_center_pipette = self.panel_lp.btn_lp_center_pipette
        self.w_lp_params = self.panel_lp.w_lp_params
        self.row_lp_cx = self.panel_lp.row_lp_cx
        self.slider_lp_cx = self.panel_lp.slider_lp_cx
        self.spin_lp_cx = self.panel_lp.spin_lp_cx
        self.lbl_lp_cx = self.panel_lp.lbl_lp_cx
        self.row_lp_cy = self.panel_lp.row_lp_cy
        self.slider_lp_cy = self.panel_lp.slider_lp_cy
        self.spin_lp_cy = self.panel_lp.spin_lp_cy
        self.lbl_lp_cy = self.panel_lp.lbl_lp_cy
        self.row_lp_w = self.panel_lp.row_lp_w
        self.slider_lp_w = self.panel_lp.slider_lp_w
        self.spin_lp_w = self.panel_lp.spin_lp_w
        self.lbl_lp_w = self.panel_lp.lbl_lp_w
        self.row_lp_h = self.panel_lp.row_lp_h
        self.slider_lp_h = self.panel_lp.slider_lp_h
        self.spin_lp_h = self.panel_lp.spin_lp_h
        self.lbl_lp_h = self.panel_lp.lbl_lp_h
        self.combo_lp_tint = self.panel_lp.combo_lp_tint
        self.btn_lp_color_pipette = self.panel_lp.btn_lp_color_pipette
        self.row_lp_tint_strength = self.panel_lp.row_lp_tint_strength
        self.slider_lp_tint_strength = self.panel_lp.slider_lp_tint_strength
        self.lbl_lp_tint_strength = self.panel_lp.lbl_lp_tint_strength
        self.combo_lp_mode = self.panel_lp.combo_lp_mode
        self.chk_lp_legacy_1d = self.panel_lp.chk_lp_legacy_1d

        # Conexiones reactivas
        self.panel_lp.parameter_changed.connect(self._on_lp_parameter_changed)
        self.panel_lp.extract_bg_requested.connect(self.on_extract_background_clicked)
        self.panel_lp.auto_detect_lp_requested.connect(self.auto_detect_lp_center)
        self.panel_lp.center_pipette_toggled.connect(self._on_toggle_lp_center_pipette)
        self.panel_lp.color_pipette_toggled.connect(self._on_toggle_lp_color_pipette)
        self.panel_lp.action_committed.connect(self._commit_action)

        left_layout.addWidget(self.panel_lp)

        # --- SECCIÓN 3: Estirado Tonal y Contraste (Módulo gui.developer.panels.StretchingPanel) ---
        self.panel_stretch = StretchingPanel()
        self.sec_stretch = self.panel_stretch.sec_stretch
        self.btn_mode_mtf = self.panel_stretch.btn_mode_mtf
        self.btn_mode_ghs = self.panel_stretch.btn_mode_ghs
        self.grp_stretch_mode = self.panel_stretch.grp_stretch_mode
        self.w_mtf_container = self.panel_stretch.w_mtf_container
        self.row_bp = self.panel_stretch.row_bp
        self.slider_bp = self.panel_stretch.slider_bp
        self.spin_bp = self.panel_stretch.spin_bp
        self.row_mtf = self.panel_stretch.row_mtf
        self.slider_mtf = self.panel_stretch.slider_mtf
        self.spin_mtf = self.panel_stretch.spin_mtf
        self.w_ghs_container = self.panel_stretch.w_ghs_container
        self.btn_auto_ghs = self.panel_stretch.btn_auto_ghs
        self.btn_ghs_pipette = self.panel_stretch.btn_ghs_pipette
        self.btn_reset_ghs = self.panel_stretch.btn_reset_ghs
        self.btn_ghs_luma = self.panel_stretch.btn_ghs_luma
        self.btn_ghs_rgb = self.panel_stretch.btn_ghs_rgb
        self.grp_ghs_color = self.panel_stretch.grp_ghs_color
        self.row_ghs_sp = self.panel_stretch.row_ghs_sp
        self.slider_ghs_sp = self.panel_stretch.slider_ghs_sp
        self.spin_ghs_sp = self.panel_stretch.spin_ghs_sp
        self.lbl_ghs_sp = self.panel_stretch.lbl_ghs_sp
        self.row_ghs_d = self.panel_stretch.row_ghs_d
        self.slider_ghs_d = self.panel_stretch.slider_ghs_d
        self.spin_ghs_d = self.panel_stretch.spin_ghs_d
        self.lbl_ghs_d = self.panel_stretch.lbl_ghs_d
        self.row_ghs_b = self.panel_stretch.row_ghs_b
        self.slider_ghs_b = self.panel_stretch.slider_ghs_b
        self.spin_ghs_b = self.panel_stretch.spin_ghs_b
        self.lbl_ghs_b = self.panel_stretch.lbl_ghs_b
        self.row_ghs_hp = self.panel_stretch.row_ghs_hp
        self.slider_ghs_hp = self.panel_stretch.slider_ghs_hp
        self.spin_ghs_hp = self.panel_stretch.spin_ghs_hp
        self.lbl_ghs_hp = self.panel_stretch.lbl_ghs_hp
        self.row_ghs_lp = self.panel_stretch.row_ghs_lp
        self.slider_ghs_lp = self.panel_stretch.slider_ghs_lp
        self.spin_ghs_lp = self.panel_stretch.spin_ghs_lp
        self.lbl_ghs_lp = self.panel_stretch.lbl_ghs_lp
        self.row_ghs_bp = self.panel_stretch.row_ghs_bp
        self.slider_ghs_bp = self.panel_stretch.slider_ghs_bp
        self.spin_ghs_bp = self.panel_stretch.spin_ghs_bp
        self.lbl_ghs_bp = self.panel_stretch.lbl_ghs_bp
        self.row_contrast = self.panel_stretch.row_contrast
        self.slider_contrast = self.panel_stretch.slider_contrast
        self.lbl_contrast = self.panel_stretch.lbl_contrast

        # Conexiones reactivas
        self.panel_stretch.parameter_changed.connect(self._on_stretch_parameter_changed)
        self.panel_stretch.auto_mtf_requested.connect(self.apply_auto_mtf)
        self.panel_stretch.auto_ghs_requested.connect(self.apply_auto_ghs)
        self.panel_stretch.ghs_pipette_toggled.connect(self._on_toggle_ghs_pipette)
        self.panel_stretch.action_committed.connect(self._commit_action)

        left_layout.addWidget(self.panel_stretch)

        # --- SECCIÓN 4: Curva Tonal Interactiva (RGB / Saturación) ---
        self.sec_curves = CollapsibleSection(
            "Curva Tonal Interactiva (RGB / Saturación)",
            is_expanded=False,
            on_reset=lambda: self.curve_widget.reset_all_curves(),
            icon_name="curve"
        )
        w_curves = QWidget()
        curves_layout = QVBoxLayout(w_curves)
        curves_layout.setContentsMargins(6, 4, 6, 6)
        curves_layout.setSpacing(8)

        # Selector de Modo de Curva (RGB / Hue vs Sat / Luma vs Sat)
        row_curve_mode = QHBoxLayout()
        row_curve_mode.setSpacing(4)

        self.btn_curve_rgb = QPushButton("Tonos (RGB)")
        self.btn_curve_rgb.setIcon(get_icon("curve", color="#80d8ff", size=12))
        self.btn_curve_rgb.setCheckable(True)
        self.btn_curve_rgb.setChecked(True)
        self.btn_curve_rgb.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #80d8ff; border: 1px solid #80d8ff; }"
        )
        self.btn_curve_rgb.setToolTip("Curva tonal RGB continua con histograma logarítmico y avisos de recorte.")
        self.btn_curve_rgb.clicked.connect(lambda: self._set_curve_mode("rgb"))

        self.btn_curve_hue = QPushButton("Tono vs Sat")
        self.btn_curve_hue.setIcon(get_icon("palette", color="#ffb74d", size=12))
        self.btn_curve_hue.setCheckable(True)
        self.btn_curve_hue.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #ffb74d; border: 1px solid #ffb74d; }"
        )
        self.btn_curve_hue.setToolTip("Curva selectiva de saturación en función del tono (Hue vs Sat).\nPermite saturar selectivamente H-alfa / nebulosas o neutralizar tonalidades no deseadas.")
        self.btn_curve_hue.clicked.connect(lambda: self._set_curve_mode("hue_sat"))

        self.btn_curve_luma = QPushButton("Luma vs Sat")
        self.btn_curve_luma.setIcon(get_icon("chart", color="#00e676", size=12))
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
        # Curva Tonal migrada directamente al Gestor de Capas (Columna 3) para Estrellas, Starless, Suelo y Base
        self.sec_curves.setVisible(False)

        # --- SECCIÓN 5: StarNet++ AI — Desacoplo de Estrellas (Módulo gui.developer.panels.StarNetPanel) ---
        self.panel_starnet = StarNetPanel(self, resettable_label_factory=self._make_resettable_label)
        self.sec_starnet = self.panel_starnet.sec_starnet
        self.spin_stride = self.panel_starnet.spin_stride
        self.btn_starnet = self.panel_starnet.btn_starnet
        self.card_starnet_info = self.panel_starnet.card_starnet_info
        self.lbl_starnet_info = self.panel_starnet.lbl_starnet_info
        self.btn_go_to_masks_starnet = self.panel_starnet.btn_go_to_masks_starnet
        self.container_starnet_tools = self.panel_starnet.container_starnet_tools
        self.combo_layer = self.panel_starnet.combo_layer
        self.slider_stars = self.panel_starnet.slider_stars
        self.lbl_stars = self.panel_starnet.lbl_stars
        self.slider_clarity = self.panel_starnet.slider_clarity
        self.lbl_clarity = self.panel_starnet.lbl_clarity
        self.slider_dehaze = self.panel_starnet.slider_dehaze
        self.lbl_dehaze = self.panel_starnet.lbl_dehaze
        self.slider_ond_est = self.panel_starnet.slider_ond_est
        self.lbl_ond_est = self.panel_starnet.lbl_ond_est
        self.slider_ond_bg = self.panel_starnet.slider_ond_bg
        self.lbl_ond_bg = self.panel_starnet.lbl_ond_bg
        self.slider_contrast_starless = self.panel_starnet.slider_contrast_starless
        self.lbl_contrast_starless = self.panel_starnet.lbl_contrast_starless
        self.slider_denoise = self.panel_starnet.slider_denoise
        self.lbl_denoise = self.panel_starnet.lbl_denoise

        self.panel_effects = self.panel_starnet.panel_effects
        self.box_orton_tools = self.panel_effects.box_orton_tools
        self.slider_orton = self.panel_effects.slider_orton
        self.lbl_orton = self.panel_effects.lbl_orton
        self.btn_orton_expand = self.panel_effects.btn_orton_expand
        self.w_orton_advanced = self.panel_effects.w_orton_advanced
        self.slider_orton_radius = self.panel_effects.slider_orton_radius
        self.lbl_orton_radius = self.panel_effects.lbl_orton_radius
        self.slider_orton_threshold = self.panel_effects.slider_orton_threshold
        self.lbl_orton_threshold = self.panel_effects.lbl_orton_threshold
        self.combo_orton_tint = self.panel_effects.combo_orton_tint
        self.combo_orton_mode = self.panel_effects.combo_orton_mode

        self.box_star_diff_tools = self.panel_effects.box_star_diff_tools
        self.slider_star_diff = self.panel_effects.slider_star_diff
        self.lbl_star_diff = self.panel_effects.lbl_star_diff
        self.btn_star_diff_expand = self.panel_effects.btn_star_diff_expand
        self.w_star_diff_advanced = self.panel_effects.w_star_diff_advanced
        self.slider_star_diff_radius = self.panel_effects.slider_star_diff_radius
        self.lbl_star_diff_radius = self.panel_effects.lbl_star_diff_radius
        self.slider_star_diff_threshold = self.panel_effects.slider_star_diff_threshold
        self.lbl_star_diff_threshold = self.panel_effects.lbl_star_diff_threshold
        self.slider_star_diff_color = self.panel_effects.slider_star_diff_color
        self.lbl_star_diff_color = self.panel_effects.lbl_star_diff_color
        self.combo_star_diff_mode = self.panel_effects.combo_star_diff_mode

        # Conectar eventos de StarNet y Efectos
        self.panel_starnet.run_starnet_requested.connect(self.run_starnet)
        self.panel_starnet.jump_to_masks_requested.connect(lambda: self._jump_to_mask_layers(MASK_TYPE_IMAGE_STARLESS))
        self.combo_layer.currentIndexChanged.connect(self.on_layer_mode_changed)
        self.slider_stars.valueChanged.connect(self.on_stars_slider_changed)
        self.slider_clarity.valueChanged.connect(self.on_clarity_changed)
        self.slider_dehaze.valueChanged.connect(self.on_dehaze_changed)
        self.slider_ond_est.valueChanged.connect(self.on_ond_estructura_changed)
        self.slider_ond_bg.valueChanged.connect(self.on_ond_fondo_changed)
        self.slider_contrast_starless.valueChanged.connect(self.on_contrast_starless_changed)
        self.slider_denoise.valueChanged.connect(self.on_denoise_changed)

        self.slider_orton.valueChanged.connect(self.on_orton_strength_changed)
        self.slider_orton_radius.valueChanged.connect(self.on_orton_radius_changed)
        self.slider_orton_threshold.valueChanged.connect(self.on_orton_threshold_changed)
        self.combo_orton_tint.currentIndexChanged.connect(self._on_orton_tint_changed)
        self.combo_orton_mode.currentIndexChanged.connect(self._on_orton_mode_changed)

        self.slider_star_diff.valueChanged.connect(self.on_star_diff_strength_changed)
        self.slider_star_diff_radius.valueChanged.connect(self.on_star_diff_radius_changed)
        self.slider_star_diff_threshold.valueChanged.connect(self.on_star_diff_threshold_changed)
        self.slider_star_diff_color.valueChanged.connect(self.on_star_diff_color_changed)
        self.combo_star_diff_mode.currentIndexChanged.connect(self._on_star_diff_mode_changed)

        self.panel_effects.action_committed.connect(self._commit_action)

        left_layout.addWidget(self.panel_starnet)
        self._set_starnet_tools_enabled(False)

        # --- SECCIÓN 5b: Deconvolución Óptica y Enfoque (Módulo gui.developer.panels.DeconvolutionPanel) ---
        self.panel_deconv = DeconvolutionPanel()
        self.sec_deconv = self.panel_deconv.sec_deconv
        self.btn_auto_psf = self.panel_deconv.btn_auto_psf
        self.lbl_psf_status = self.panel_deconv.lbl_psf_status
        self.row_deconv_iter = self.panel_deconv.row_deconv_iter
        self.slider_deconv_iter = self.panel_deconv.slider_deconv_iter
        self.lbl_deconv_iter = self.panel_deconv.lbl_deconv_iter
        self.btn_deconv_expand = self.panel_deconv.btn_deconv_expand
        self.row_deconv_fwhm = self.panel_deconv.row_deconv_fwhm
        self.slider_deconv_fwhm = self.panel_deconv.slider_deconv_fwhm
        self.lbl_deconv_fwhm = self.panel_deconv.lbl_deconv_fwhm
        self.w_deconv_advanced = self.panel_deconv.w_deconv_advanced
        self.row_deconv_deringing = self.panel_deconv.row_deconv_deringing
        self.slider_deconv_deringing = self.panel_deconv.slider_deconv_deringing
        self.lbl_deconv_deringing = self.panel_deconv.lbl_deconv_deringing
        self.row_deconv_damping = self.panel_deconv.row_deconv_damping
        self.slider_deconv_damping = self.panel_deconv.slider_deconv_damping
        self.lbl_deconv_damping = self.panel_deconv.lbl_deconv_damping
        self.row_deconv_aspect = self.panel_deconv.row_deconv_aspect
        self.slider_deconv_aspect = self.panel_deconv.slider_deconv_aspect
        self.lbl_deconv_aspect = self.panel_deconv.lbl_deconv_aspect
        self.row_deconv_angle = self.panel_deconv.row_deconv_angle
        self.slider_deconv_angle = self.panel_deconv.slider_deconv_angle
        self.lbl_deconv_angle = self.panel_deconv.lbl_deconv_angle
        self.combo_deconv_model = self.panel_deconv.combo_deconv_model
        self.combo_deconv_scope = self.panel_deconv.combo_deconv_scope
        self.chk_deconv_luma_only = self.panel_deconv.chk_deconv_luma_only

        # Conexiones reactivas del panel de deconvolución
        self.panel_deconv.parameter_changed.connect(self._on_deconv_parameter_changed)
        self.panel_deconv.auto_psf_requested.connect(self.on_auto_detect_psf_clicked)
        self.panel_deconv.action_committed.connect(self._commit_action)

        left_layout.addWidget(self.panel_deconv)

        # --- SECCIONES 6 Y 7: Balance de Blancos y Color/Saturación (Módulo gui.developer.panels.ColorWbPanel) ---
        self.panel_color_wb = ColorWbPanel()
        self.sec_wb = self.panel_color_wb.sec_wb
        self.row_temp = self.panel_color_wb.row_temp
        self.slider_temp = self.panel_color_wb.slider_temp
        self.spin_temp = self.panel_color_wb.spin_temp
        self.lbl_temp = self.panel_color_wb.lbl_temp
        self.row_tint = self.panel_color_wb.row_tint
        self.slider_tint = self.panel_color_wb.slider_tint
        self.spin_tint = self.panel_color_wb.spin_tint
        self.lbl_tint = self.panel_color_wb.lbl_tint
        self.btn_panel_pipette = self.panel_color_wb.btn_panel_pipette
        self.btn_pipette = self.panel_color_wb.btn_pipette

        self.sec_color = self.panel_color_wb.sec_color
        self.row_sat_sky = self.panel_color_wb.row_sat_sky
        self.slider_sat_sky = self.panel_color_wb.slider_sat_sky
        self.lbl_sat_sky = self.panel_color_wb.lbl_sat_sky
        self.row_sat_gnd = self.panel_color_wb.row_sat_gnd
        self.slider_sat_gnd = self.panel_color_wb.slider_sat_gnd
        self.lbl_sat_gnd = self.panel_color_wb.lbl_sat_gnd
        self.row_vibrance = self.panel_color_wb.row_vibrance
        self.slider_vibrance = self.panel_color_wb.slider_vibrance
        self.lbl_vibrance = self.panel_color_wb.lbl_vibrance

        # Conexiones reactivas del panel
        self.panel_color_wb.parameter_changed.connect(self._on_color_wb_parameter_changed)
        self.panel_color_wb.pipette_toggled.connect(self._on_toggle_wb_pipette)
        self.panel_color_wb.action_committed.connect(self._commit_action)

        left_layout.addWidget(self.panel_color_wb)

        # --- SECCIÓN 8: Ajuste Tonal Suelo (Paisaje) (Módulo gui.developer.panels.GroundSkyPanel) ---
        self.panel_ground_sky = GroundSkyPanel()
        self.sec_gnd_tone = self.panel_ground_sky.sec_gnd_tone
        self.lbl_gnd_mask_status = self.panel_ground_sky.lbl_gnd_mask_status
        self.btn_edit_ground_mask = self.panel_ground_sky.btn_edit_ground_mask
        self.card_gnd_info = self.panel_ground_sky.card_gnd_info
        self.lbl_gnd_info = self.panel_ground_sky.lbl_gnd_info
        self.btn_go_to_ground_layer = self.panel_ground_sky.btn_go_to_ground_layer
        self._hidden_gnd_container = self.panel_ground_sky._hidden_gnd_container
        self.lbl_gnd_ev = self.panel_ground_sky.lbl_gnd_ev
        self.slider_gnd_ev = self.panel_ground_sky.slider_gnd_ev
        self.lbl_gnd_sh = self.panel_ground_sky.lbl_gnd_sh
        self.slider_gnd_sh = self.panel_ground_sky.slider_gnd_sh
        self.lbl_gnd_bp = self.panel_ground_sky.lbl_gnd_bp
        self.slider_gnd_bp = self.panel_ground_sky.slider_gnd_bp

        self.panel_ground_sky.edit_mask_requested.connect(self._open_mask_tool_from_section)
        self.panel_ground_sky.go_to_layer_requested.connect(lambda: self._jump_to_mask_layers(MASK_TYPE_IMAGE_GROUND))
        self.panel_ground_sky.reset_requested.connect(self.reset_gnd_tone)
        self.slider_gnd_ev.valueChanged.connect(self.on_gnd_ev_changed)
        self.slider_gnd_sh.valueChanged.connect(self.on_gnd_shadows_changed)
        self.slider_gnd_bp.valueChanged.connect(self.on_gnd_bp_changed)

        # Ocultar sección de suelo del panel izquierdo (gestionada en la Capa de Suelo de capas)
        self.sec_gnd_tone.setVisible(False)
        left_layout.addWidget(self.panel_ground_sky)

        # --- SECCIÓN 9: Máscaras y Capas de Ajuste Local ---
        self.sec_masks = CollapsibleSection("Máscaras y Capas de Ajuste Local", is_expanded=False, icon_name="layers")
        self.mask_layers_widget = MaskLayersWidget()
        self.mask_layers_widget.set_getters(
            img_getter=lambda: self.preview_proxy if self.preview_proxy is not None else self.image_32bit,
            mask_getter=lambda: self.current_mask,
            stars_getter=lambda: self.proxy_stars if self.proxy_stars is not None else self.image_stars,
            starless_getter=lambda: self.proxy_starless if self.proxy_starless is not None else self.image_starless,
            layer_image_provider=self._get_layer_image_for_histogram
        )
        self.mask_layers_widget.layers_changed.connect(self.update_stretch_preview)
        self.mask_layers_widget.preview_mask_requested.connect(self._on_layer_mask_preview_requested)
        self.mask_layers_widget.edit_ground_mask_requested.connect(self._open_mask_tool_from_section)
        self.mask_layers_widget.ai_ground_mask_requested.connect(self._detect_mask_ai_from_toolbar)
        self.mask_layers_widget.brush_tool_toggled.connect(self._on_layer_brush_toggled)
        self.mask_layers_widget.brush_params_changed.connect(self._on_layer_brush_params_changed)
        self.mask_layers_widget.color_eyedropper_requested.connect(self._on_layer_color_sample_requested)
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
        self.mask_layers_widget.stars_diff_strength_changed.connect(self._on_mask_widget_stars_diff_strength)
        self.mask_layers_widget.stars_diff_radius_changed.connect(self._on_mask_widget_stars_diff_radius)
        self.mask_layers_widget.stars_diff_threshold_changed.connect(self._on_mask_widget_stars_diff_threshold)
        self.mask_layers_widget.stars_diff_color_boost_changed.connect(self._on_mask_widget_stars_diff_color)
        self.mask_layers_widget.stars_diff_mode_changed.connect(self._on_mask_widget_stars_diff_mode)
        self.mask_layers_widget.stars_reset_requested.connect(self.reset_stars_intensity)
        self.mask_layers_widget.starless_clarity_changed.connect(self._on_mask_widget_sl_clarity)
        self.mask_layers_widget.starless_dehaze_changed.connect(self._on_mask_widget_sl_dehaze)
        self.mask_layers_widget.starless_ond_est_changed.connect(self._on_mask_widget_sl_ond_est)
        self.mask_layers_widget.starless_ond_bg_changed.connect(self._on_mask_widget_sl_ond_bg)
        self.mask_layers_widget.starless_contrast_changed.connect(self._on_mask_widget_sl_contrast)
        self.mask_layers_widget.starless_denoise_changed.connect(self._on_mask_widget_sl_denoise)
        self.mask_layers_widget.starless_orton_strength_changed.connect(self._on_mask_widget_sl_orton_strength)
        self.mask_layers_widget.starless_orton_radius_changed.connect(self._on_mask_widget_sl_orton_radius)
        self.mask_layers_widget.starless_orton_threshold_changed.connect(self._on_mask_widget_sl_orton_threshold)
        self.mask_layers_widget.starless_orton_tint_changed.connect(self._on_mask_widget_sl_orton_tint)
        self.mask_layers_widget.starless_orton_mode_changed.connect(self._on_mask_widget_sl_orton_mode)
        self.mask_layers_widget.starless_reset_requested.connect(self.reset_starless_adjustments)

        self.mask_layers_widget.set_stars_special_values(
            presence=self.slider_stars.value(),
            diff_strength=self.slider_star_diff.value(),
            diff_radius=self.slider_star_diff_radius.value(),
            diff_threshold=self.slider_star_diff_threshold.value(),
            diff_color=self.slider_star_diff_color.value(),
            diff_mode_idx=self.combo_star_diff_mode.currentIndex()
        )
        self.mask_layers_widget.set_starless_tone_values(
            self.slider_clarity.value(),
            self.slider_dehaze.value(),
            self.slider_ond_est.value(),
            self.slider_ond_bg.value(),
            self.slider_contrast_starless.value(),
            self.slider_denoise.value(),
            self.slider_orton.value(),
            self.slider_orton_radius.value(),
            self.slider_orton_threshold.value(),
            self.combo_orton_tint.currentIndex(),
            self.combo_orton_mode.currentIndex()
        )

        def _update_masks_badge():
            count = len(self.mask_layers_widget.layers)
            if count > 0:
                self.sec_masks.set_status_badge(f"● {count} {'capas' if count > 1 else 'capa'}", "#80d8ff")
            else:
                self.sec_masks.set_status_badge("")
        self.mask_layers_widget.layers_changed.connect(_update_masks_badge)

        self.sec_masks.set_content_widget(self.mask_layers_widget)
        # Nota: self.sec_masks se añade a la columna derecha (right_layout)

        left_layout.addStretch(1)

        # Asignar panel de acordeones al área de scroll
        self.left_scroll_area.setWidget(left_panel)
        left_container_layout.addWidget(self.left_scroll_area, 1)

        # Botón Exportar siempre fijado al pie del panel izquierdo
        self.btn_export = QPushButton("EXPORTAR IMAGEN REVELADA...")
        self.btn_export.setIcon(get_icon("save", color="#ffffff", size=16))
        self.btn_export.setFixedHeight(40)
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_export.clicked.connect(self.export_image)
        left_container_layout.addWidget(self.btn_export, 0)

        self.btn_export_psd = QPushButton("EXPORTAR A PHOTOSHOP (PSD)...")
        self.btn_export_psd.setIcon(get_icon("layers", color="#82b1ff", size=15))
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

        # 3. Contenedor central (Visor y Lienzo interactivo)
        self.center_container = QWidget()
        self.center_container.setMinimumWidth(260)
        center_layout = QVBoxLayout(self.center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.canvas_toolbar = self._create_canvas_toolbar()
        center_layout.addWidget(self.canvas_toolbar)
        self.crop_toolbar = self._create_crop_toolbar()
        self.crop_toolbar.setVisible(False)
        center_layout.addWidget(self.crop_toolbar)
        self.mask_toolbar = self._create_mask_toolbar()
        self.mask_toolbar.setVisible(False)
        center_layout.addWidget(self.mask_toolbar)
        center_layout.addWidget(self.canvas, 1)
        self.canvas_footer = self._create_canvas_footer_bar()
        center_layout.addWidget(self.canvas_footer)

        # Sincronización bidireccional de Pipeta, Lupa, Recorte y Pincel con la barra de herramientas
        self.canvas.eyedropper_toggled.connect(self._on_canvas_eyedropper_toggled)
        self.canvas.eyedropper_toggled.connect(lambda active: self._update_canvas_info_label())
        self.canvas.magnifier_toggled.connect(self.btn_magnifier.setChecked)
        self.canvas.magnifier_toggled.connect(lambda active: self._update_canvas_info_label())
        self.canvas.crop_toggled.connect(self.btn_crop.setChecked)
        self.canvas.crop_toggled.connect(lambda active: self._update_canvas_info_label())
        self.canvas.crop_rect_changed.connect(self._on_crop_rect_changed)
        self.canvas.crop_applied.connect(self._apply_crop)
        self.canvas.color_sampled.connect(self._on_eyedropper_color_sampled)
        self.canvas.brush_size_changed.connect(self._on_brush_size_changed)
        self.canvas.layer_brush_stroke_finished.connect(self._on_layer_brush_stroke_finished)
        self.canvas.layer_brush_radius_changed.connect(self.mask_layers_widget.set_brush_radius_from_canvas)

        # 4. Contenedor derecho: Histograma RGB en Vivo + Capas y Máscaras
        self.right_container = QWidget()
        right_container_layout = QVBoxLayout(self.right_container)
        right_container_layout.setContentsMargins(4, 4, 4, 4)
        right_container_layout.setSpacing(6)
        self.right_container.setMinimumWidth(280)
        self.right_container.setMaximumWidth(1200)

        # Cabecera de Columna Derecha
        right_header = QWidget()
        right_header.setFixedHeight(28)
        right_header_layout = QHBoxLayout(right_header)
        right_header_layout.setContentsMargins(6, 2, 6, 2)
        right_header_layout.setSpacing(6)

        lbl_right_title = QLabel("CAPAS E HISTOGRAMA")
        lbl_right_title.setStyleSheet("font-weight: 700; font-size: 11px; color: #80d8ff; letter-spacing: 0.6px;")
        right_header_layout.addWidget(lbl_right_title)
        right_header_layout.addStretch()
        right_container_layout.addWidget(right_header, 0)

        # Scroll área para la columna derecha
        self.right_scroll_area = QScrollArea()
        self.right_scroll_area.setWidgetResizable(True)
        self.right_scroll_area.setFrameShape(QFrame.NoFrame)
        self.right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        right_panel = QWidget()
        right_panel.setMinimumWidth(0)
        right_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 4, 0)
        right_layout.setSpacing(6)

        # Widget Histograma RGB en Vivo
        self.live_histogram = LiveRGBHistogramWidget(self)
        self.live_histogram.clippingChanged.connect(self._on_live_histogram_clipping_changed)

        self.sec_histogram = CollapsibleSection("Histograma RGB en Vivo", is_expanded=True, icon_name="activity")
        self.sec_histogram.set_content_widget(self.live_histogram)
        self.sec_histogram.toggled.connect(lambda exp: self._refresh_histograms_now() if exp else None)
        right_layout.addWidget(self.sec_histogram)

        # Capas y Máscaras (sec_masks)
        self.sec_masks.set_expanded(True)
        right_layout.addWidget(self.sec_masks)

        right_layout.addStretch(1)

        self.right_scroll_area.setWidget(right_panel)
        right_container_layout.addWidget(self.right_scroll_area, 1)

        # Splitter principal de 3 columnas
        self.main_splitter = splitter
        self.main_splitter.setChildrenCollapsible(True)
        self.main_splitter.setHandleWidth(6)

        self.main_splitter.addWidget(self.left_container)
        self.main_splitter.addWidget(self.center_container)
        self.main_splitter.addWidget(self.right_container)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 10)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([380, 1100, 360])

        # Asas de divisor: cursor, tooltip y doble clic para alternar expansión al centro
        h1 = self.main_splitter.handle(1)
        if h1:
            h1.setCursor(Qt.SplitHCursor)
            h1.setToolTip("Arrastra para redimensionar o doble clic para alternar panel ancho")
            h1.installEventFilter(self)

        h2 = self.main_splitter.handle(2)
        if h2:
            h2.setCursor(Qt.SplitHCursor)
            h2.setToolTip("Arrastra para redimensionar o doble clic para alternar panel ancho")
            h2.installEventFilter(self)

        # Atajo global 'Tab' para alternar ambos paneles laterales (Modo Inmersivo / Cine Lightroom)
        self.shortcut_tab = QShortcut(QKeySequence(Qt.Key_Tab), self)
        self.shortcut_tab.setContext(Qt.WindowShortcut)
        self.shortcut_tab.activated.connect(self.toggle_both_panels)

        # Atajos 'F7' y 'F8' para alternar paneles individuales
        self.shortcut_f7 = QShortcut(QKeySequence(Qt.Key_F7), self)
        self.shortcut_f7.setContext(Qt.WindowShortcut)
        self.shortcut_f7.activated.connect(lambda: self.toggle_left_panel())

        self.shortcut_f8 = QShortcut(QKeySequence(Qt.Key_F8), self)
        self.shortcut_f8.setContext(Qt.WindowShortcut)
        self.shortcut_f8.activated.connect(lambda: self.toggle_right_panel())
        
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
        tb_layout.setSpacing(6)

        # 0. Botón Ocultar/Mostrar Panel Izquierdo (Lightroom F7 / [)
        self.btn_toggle_left_panel = QPushButton()
        self.btn_toggle_left_panel.setIcon(get_icon("panel_left", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=14))
        self.btn_toggle_left_panel.setCheckable(True)
        self.btn_toggle_left_panel.setChecked(True)
        self.btn_toggle_left_panel.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_left_panel.setToolTip("Ocultar / Mostrar panel lateral izquierdo de ajustes (Atajo: F7 o [)")
        self.btn_toggle_left_panel.setFixedSize(30, 26)
        self.btn_toggle_left_panel.toggled.connect(self.toggle_left_panel)
        tb_layout.addWidget(self.btn_toggle_left_panel)

        sep_left = QFrame()
        sep_left.setFrameShape(QFrame.VLine)
        sep_left.setFrameShadow(QFrame.Sunken)
        sep_left.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep_left)

        # 0a. Botón Deshacer (Undo)
        self.btn_undo = QPushButton("Deshacer")
        self.btn_undo.setIcon(get_icon("undo", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_undo.setCursor(Qt.PointingHandCursor)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("Deshace el último ajuste de revelado (Atajo: Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        tb_layout.addWidget(self.btn_undo)

        # 0b. Botón Rehacer (Redo)
        self.btn_redo = QPushButton("Rehacer")
        self.btn_redo.setIcon(get_icon("redo", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
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
        self.btn_split_view = QPushButton("Antes / Después (Y)")
        self.btn_split_view.setIcon(get_icon("split", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
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
        self.btn_split_orientation = QPushButton("Vertical")
        self.btn_split_orientation.setIcon(get_icon("sliders", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_split_orientation.setCursor(Qt.PointingHandCursor)
        self.btn_split_orientation.setEnabled(False)
        self.btn_split_orientation.setToolTip(
            "Alterna la orientación de la cortina divisoria:\n"
            "• Vertical: Izquierda ('Antes') / Derecha ('Después')\n"
            "• Horizontal: Arriba ('Antes') / Abajo ('Después')\n"
            "Atajo de teclado: Shift+Y"
        )
        self.btn_split_orientation.clicked.connect(self.on_toggle_split_orientation)
        tb_layout.addWidget(self.btn_split_orientation)

        # 2. Botón Fijar como Antes (Snapshot)
        self.btn_set_before = QPushButton("Fijar como Antes")
        self.btn_set_before.setIcon(get_icon("camera", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
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
        self.btn_toggle_clipping = QPushButton("Recorte (J)")
        self.btn_toggle_clipping.setIcon(get_icon("alert", color="#94a3b8", hover_color="#ffffff", active_color="#f87171", size=13))
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
        self.btn_toggle_zoom = QPushButton("Zoom 100%")
        self.btn_toggle_zoom.setIcon(get_icon("zoom_in", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
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
        self.btn_magnifier = QPushButton("Lupa (Z)")
        self.btn_magnifier.setIcon(get_icon("search", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
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
        self.btn_crop = QPushButton("Recortar (C)")
        self.btn_crop.setIcon(get_icon("crop", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
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
        self.btn_mask_tool = QPushButton("Máscara Suelo (M)")
        self.btn_mask_tool.setIcon(get_icon("brush", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
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
        self.btn_top_ai_mask = QPushButton("Máscara IA")
        self.btn_top_ai_mask.setIcon(get_icon("sparkles", color="#ffffff", size=13))
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

        # 8c. Botón Visualización Rápida de Máscara de Suelo (Overlay)
        self.btn_top_mask_overlay = QPushButton("Ver Máscara")
        self.btn_top_mask_overlay.setIcon(get_icon("eye", color="#94a3b8", hover_color="#ffffff", active_color="#4ade80", size=13))
        self.btn_top_mask_overlay.setCheckable(True)
        self.btn_top_mask_overlay.setEnabled(False)
        self.btn_top_mask_overlay.setCursor(Qt.PointingHandCursor)
        self.btn_top_mask_overlay.setStyleSheet("""
            QPushButton {
                background-color: #262933;
                color: #a0a0a0;
                border: 1px solid #3b4252;
                font-weight: normal;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #88c0d0;
                color: #eceff4;
            }
            QPushButton:disabled {
                color: #4c566a;
                background-color: #1a1c23;
                border: 1px solid #2a2e39;
            }
        """)
        self.btn_top_mask_overlay.setToolTip("Muestra u oculta la superposición de la máscara de cielo y suelo en el visor.")
        self.btn_top_mask_overlay.toggled.connect(self._on_top_mask_overlay_toggled)
        tb_layout.addWidget(self.btn_top_mask_overlay)

        # Espaciador elástico
        tb_layout.addStretch()

        # Separador vertical
        sep_right = QFrame()
        sep_right.setFrameShape(QFrame.VLine)
        sep_right.setFrameShadow(QFrame.Sunken)
        sep_right.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep_right)

        # Botón Ocultar/Mostrar Panel Derecho (Lightroom F8 / ])
        self.btn_toggle_right_panel = QPushButton()
        self.btn_toggle_right_panel.setIcon(get_icon("panel_right", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=14))
        self.btn_toggle_right_panel.setCheckable(True)
        self.btn_toggle_right_panel.setChecked(True)
        self.btn_toggle_right_panel.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_right_panel.setToolTip("Ocultar / Mostrar panel lateral derecho de capas e histograma (Atajo: F8 o ])")
        self.btn_toggle_right_panel.setFixedSize(30, 26)
        self.btn_toggle_right_panel.toggled.connect(self.toggle_right_panel)
        tb_layout.addWidget(self.btn_toggle_right_panel)

        return toolbar

    def _create_canvas_footer_bar(self) -> QWidget:
        """Crea la barra inferior de estado, modo de herramienta activa y metadatos del lienzo."""
        footer = QWidget()
        footer.setObjectName("canvas_footer_bar")
        footer.setFixedHeight(26)

        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)

        # 1. Indicador de Herramienta Activa (Pill badge)
        self.lbl_tool_badge = QLabel("● Modo: Vista General")
        self.lbl_tool_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        layout.addWidget(self.lbl_tool_badge)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #232738;")
        layout.addWidget(sep1)

        # 2. Información de Zoom
        self.lbl_canvas_zoom = QLabel("Zoom: Ajustar (Fit)")
        self.lbl_canvas_zoom.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_canvas_zoom)

        # Espaciador elástico
        layout.addStretch(1)

        # 3. Metadatos de la Imagen (Resolución, MP, Profundidad, Espacio de Color)
        self.lbl_canvas_info = QLabel("Sin imagen cargada")
        self.lbl_canvas_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_canvas_info)

        return footer

    def retranslate_ui(self) -> None:
        """Actualiza todos los textos visibles del Revelador según el idioma activo."""
        if hasattr(self, 'btn_open_img'):
            self.btn_open_img.setText(tr("developer.btn_open", "Abrir Imagen (TIFF / FITS / RAW)..."))
        if hasattr(self, 'btn_export'):
            self.btn_export.setText(tr("developer.btn_export", "EXPORTAR IMAGEN REVELADA..."))
        if hasattr(self, 'lbl_log_title'):
            self.lbl_log_title.setText(tr("developer.activity_log", "Registro de Actividad:"))

        # Secciones del acordeón
        if hasattr(self, 'sec_presets'):
            self.sec_presets.set_title(tr("developer.sections.presets", "Presets y Estilos de Revelado"))
        if hasattr(self, 'sec_gradients'):
            self.sec_gradients.set_title(tr("developer.sections.gradients", "Calibración y Gradientes (GraXpert AI)"))
        if hasattr(self, 'sec_stretch'):
            self.sec_stretch.set_title(tr("developer.sections.stretch", "Estirado Tonal y Contraste"))
        if hasattr(self, 'sec_curves'):
            self.sec_curves.set_title(tr("developer.sections.curves", "Curva Tonal Interactiva (RGB / Saturación)"))
        if hasattr(self, 'sec_starnet'):
            self.sec_starnet.set_title(tr("developer.sections.starnet", "StarNet++ AI — Desacoplo de Estrellas"))
        if hasattr(self, 'sec_deconv'):
            self.sec_deconv.set_title(tr("developer.sections.deconv", "Deconvolución Óptica y Enfoque (R-L)"))
        if hasattr(self, 'sec_wb'):
            self.sec_wb.set_title(tr("developer.sections.wb", "Balance de Blancos y Calibración"))
        if hasattr(self, 'sec_color'):
            self.sec_color.set_title(tr("developer.sections.color", "Color y Saturación"))
        if hasattr(self, 'sec_gnd_tone'):
            self.sec_gnd_tone.set_title(tr("developer.sections.ground_tone", "Ajuste Tonal Suelo (Paisaje)"))
        if hasattr(self, 'sec_histogram'):
            self.sec_histogram.set_title(tr("developer.sections.histogram", "Histograma RGB en Vivo"))
        if hasattr(self, 'sec_masks'):
            self.sec_masks.set_title(tr("developer.sections.masks", "Capas y Máscaras"))

        # Botones de herramientas
        if hasattr(self, 'btn_starnet'):
            self.btn_starnet.setText(tr("developer.starnet.btn_separate", "Separar Estrellas con StarNet"))
        if hasattr(self, 'btn_panel_pipette'):
            self.btn_panel_pipette.setText(tr("developer.wb.pipette", "Pipeta WB (W)"))

        # Barra superior del visor (iconos gestionados por setIcon)
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setText(tr("developer.toolbar.undo", "Deshacer"))
        if hasattr(self, 'btn_redo'):
            self.btn_redo.setText(tr("developer.toolbar.redo", "Rehacer"))
        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.setText(tr("developer.toolbar.before_after", "Antes / Después (Y)"))
        if hasattr(self, 'btn_set_before'):
            self.btn_set_before.setText(tr("developer.toolbar.set_as_before", "Fijar como Antes"))
        if hasattr(self, 'btn_toggle_clipping'):
            self.btn_toggle_clipping.setText(tr("developer.toolbar.clipping", "Recorte (J)"))
        if hasattr(self, 'btn_toggle_zoom'):
            self.btn_toggle_zoom.setText(tr("developer.toolbar.zoom_100", "Zoom 100%"))
        if hasattr(self, 'btn_magnifier'):
            self.btn_magnifier.setText(tr("developer.toolbar.loupe", "Lupa (Z)"))
        if hasattr(self, 'btn_crop'):
            self.btn_crop.setText(tr("developer.toolbar.crop", "Recortar (C)"))
        if hasattr(self, 'btn_mask_tool'):
            self.btn_mask_tool.setText(tr("developer.toolbar.ground_mask", "Máscara Suelo (M)"))
        if hasattr(self, 'btn_toggle_left_panel'):
            self.btn_toggle_left_panel.setToolTip(tr("developer.toolbar.toggle_left", "Ocultar / Mostrar panel lateral izquierdo de ajustes (Atajo: F7 o [)"))
        if hasattr(self, 'btn_toggle_right_panel'):
            self.btn_toggle_right_panel.setToolTip(tr("developer.toolbar.toggle_right", "Ocultar / Mostrar panel lateral derecho de capas e histograma (Atajo: F8 o ])"))
        if hasattr(self, '_update_mask_status_ui'):
            self._update_mask_status_ui()

    def _create_crop_toolbar(self) -> QWidget:
        """Crea la barra de herramientas interactiva de recorte y nivelación de horizonte."""
        toolbar = CropToolbar(self, resettable_label_factory=self._make_resettable_label)
        toolbar.setObjectName("crop_toolbar")
        toolbar.setFixedHeight(40)

        self.combo_crop_ratio = toolbar.combo_crop_ratio
        self.btn_crop_flip_ratio = toolbar.btn_crop_flip_ratio
        self.slider_crop_angle = toolbar.slider_crop_angle
        self.lbl_crop_angle_val = toolbar.lbl_crop_angle_val
        self.btn_crop_reset_angle = toolbar.btn_crop_reset_angle
        self.combo_crop_guides = toolbar.combo_crop_guides
        self.lbl_crop_dimensions = toolbar.lbl_crop_dimensions
        self.btn_crop_reset = toolbar.btn_crop_reset
        self.btn_crop_cancel = toolbar.btn_crop_cancel
        self.btn_crop_apply = toolbar.btn_crop_apply

        toolbar.ratio_changed.connect(self._on_crop_ratio_changed)
        toolbar.flip_ratio_clicked.connect(self._on_crop_flip_ratio)
        toolbar.angle_changed.connect(self._on_crop_angle_changed)
        toolbar.guides_changed.connect(self._on_crop_guides_changed)
        toolbar.reset_clicked.connect(self._reset_crop)
        toolbar.cancel_clicked.connect(self._cancel_crop)
        toolbar.apply_clicked.connect(self._apply_crop)

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
            self.log_message("[RECORTE] Modo Recorte activado. Ajusta el marco, nivela el horizonte o presiona Intro para aplicar.")
        else:
            self.canvas.reset_crop_rect()
            if hasattr(self, 'slider_crop_angle'):
                self.slider_crop_angle.setValue(0)
        self._update_canvas_info_label()

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
        self.log_message(f"[RECORTE] Recorte aplicado{ang_info}: {orig_w}x{orig_h} → {new_w}x{new_h} px. (Deshacer con Ctrl+Z)")

    def _create_mask_toolbar(self) -> QWidget:
        """Crea la barra de herramientas interactiva para dibujar semillas y refinar la máscara de suelo."""
        toolbar = QWidget()
        toolbar.setObjectName("mask_toolbar")
        toolbar.setFixedHeight(40)

        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 3, 10, 3)
        tb_layout.setSpacing(8)

        # 1. Título / Icono
        lbl_title = QLabel("Máscara Suelo:")
        lbl_title.setStyleSheet("font-weight: bold; color: #80d8ff; font-size: 11px;")
        tb_layout.addWidget(lbl_title)

        # 2. Selector de Clase de Pincel (Cielo vs Suelo)
        self.btn_brush_sky = QPushButton("● Cielo (Verde)")
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

        self.btn_brush_gnd = QPushButton("● Suelo (Rojo)")
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
        self.btn_ai_mask = QPushButton("Máscara IA")
        self.btn_ai_mask.setIcon(get_icon("sparkles", color="#ffffff", size=13))
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
        self.btn_generate_mask = QPushButton("Generar (Intro)")
        self.btn_generate_mask.setIcon(get_icon("check", color="#ffffff", size=13))
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
        self.btn_load_mask = QPushButton("Cargar...")
        self.btn_load_mask.setIcon(get_icon("folder", color="#80d8ff", size=13))
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
        self.btn_mask_overlay_toggle = QPushButton("Ver")
        self.btn_mask_overlay_toggle.setIcon(get_icon("eye", color="#94a3b8", active_color="#4ade80", size=13))
        self.btn_mask_overlay_toggle.setCheckable(True)
        self.btn_mask_overlay_toggle.setChecked(False)
        self.btn_mask_overlay_toggle.setEnabled(False)
        self.btn_mask_overlay_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_mask_overlay_toggle.setStyleSheet("""
            QPushButton {
                background-color: #262933;
                color: #a0a0a0;
                border: 1px solid #3b4252;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #88c0d0;
                color: #eceff4;
            }
            QPushButton:disabled {
                color: #4c566a;
                background-color: #1a1c23;
                border: 1px solid #2a2e39;
            }
        """)
        self.btn_mask_overlay_toggle.setToolTip("Muestra u oculta la superposición semitransparente (Verde=Cielo, Rojo=Suelo)")
        self.btn_mask_overlay_toggle.toggled.connect(self._on_top_mask_overlay_toggled)
        tb_layout.addWidget(self.btn_mask_overlay_toggle)

        # 7. Limpiar Trazos
        self.btn_clear_scribbles = QPushButton("Trazos")
        self.btn_clear_scribbles.setIcon(get_icon("trash", color="#94a3b8", size=13))
        self.btn_clear_scribbles.setCursor(Qt.PointingHandCursor)
        self.btn_clear_scribbles.setToolTip("Borra los trazos dibujados sin eliminar la máscara actual calculada")
        self.btn_clear_scribbles.clicked.connect(self._clear_scribbles)
        tb_layout.addWidget(self.btn_clear_scribbles)

        # 8. Quitar / Eliminar Máscara
        self.btn_delete_mask = QPushButton("Quitar")
        self.btn_delete_mask.setIcon(get_icon("x", color="#f87171", size=13))
        self.btn_delete_mask.setCursor(Qt.PointingHandCursor)
        self.btn_delete_mask.setToolTip("Elimina la máscara activa por completo de la imagen")
        self.btn_delete_mask.clicked.connect(self._delete_mask)
        tb_layout.addWidget(self.btn_delete_mask)

        tb_layout.addStretch()

        # 9. Botón Finalizar / Cerrar
        self.btn_mask_done = QPushButton("Listo (Esc)")
        self.btn_mask_done.setIcon(get_icon("check", color="#ffffff", size=13))
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
        self._update_canvas_info_label()

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
        """Asegura visibilidad del panel derecho, expande la sección de capas y selecciona la capa especificada."""
        if hasattr(self, 'right_container') and not self.right_container.isVisible():
            self.toggle_right_panel(True)
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
                "Debes pintar al menos un trazo verde (● Cielo) y un trazo rojo (● Suelo) sobre la imagen."
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
            self._mask_source_name = "Trazos"
            self._generate_preview_proxy()
            self.update_stretch_preview()
            self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
            self.canvas.set_scribbles_visible(False)
            self.canvas.set_mask_visible(True)
            self._set_mask_overlay_ui_state(True)
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
        self._mask_source_name = ""
        self.canvas.clear_scribbles()
        self.canvas.set_refined_mask(None)
        self.canvas.set_mask_visible(False)
        self._set_mask_overlay_ui_state(False)
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self._update_mask_status_ui()
        self._record_mask_action(old_m, None, "Eliminar Máscara")
        self.log_message("[MÁSCARA] Máscara de suelo eliminada.")

    def _set_mask_overlay_ui_state(self, visible: bool) -> None:
        """Actualiza el estado visual, textos y colores de los botones de alternancia de máscara."""
        has_mask = (self.current_mask is not None)
        is_active = has_mask and visible

        qss_active = """
            QPushButton {
                background-color: #2e7d32;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #81c784;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #388e3c;
                border-color: #a5d6a7;
            }
        """
        qss_inactive = """
            QPushButton {
                background-color: #262933;
                color: #a0a0a0;
                border: 1px solid #3b4252;
                font-weight: normal;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #88c0d0;
                color: #eceff4;
            }
            QPushButton:disabled {
                color: #4c566a;
                background-color: #1a1c23;
                border: 1px solid #2a2e39;
            }
        """

        if hasattr(self, 'btn_top_mask_overlay'):
            self.btn_top_mask_overlay.setEnabled(has_mask)
            self.btn_top_mask_overlay.blockSignals(True)
            self.btn_top_mask_overlay.setChecked(is_active)
            self.btn_top_mask_overlay.blockSignals(False)
            self.btn_top_mask_overlay.setStyleSheet(qss_active if is_active else qss_inactive)
            if is_active:
                self.btn_top_mask_overlay.setText("Máscara Visible")
                src = f" ({self._mask_source_name})" if self._mask_source_name else ""
                self.btn_top_mask_overlay.setToolTip(f"Ocultar superposición de la máscara{src}. La máscara sigue activa internamente.")
            else:
                self.btn_top_mask_overlay.setText("Ver Máscara")
                if has_mask:
                    src = f" ({self._mask_source_name})" if self._mask_source_name else ""
                    self.btn_top_mask_overlay.setToolTip(f"Mostrar superposición de la máscara{src} sobre la imagen.")
                else:
                    self.btn_top_mask_overlay.setToolTip("No hay máscara activa de cielo/suelo calculada.")

        if hasattr(self, 'btn_mask_overlay_toggle'):
            self.btn_mask_overlay_toggle.setEnabled(has_mask)
            self.btn_mask_overlay_toggle.blockSignals(True)
            self.btn_mask_overlay_toggle.setChecked(is_active)
            self.btn_mask_overlay_toggle.blockSignals(False)
            self.btn_mask_overlay_toggle.setStyleSheet(qss_active if is_active else qss_inactive)
            self.btn_mask_overlay_toggle.setText("Visible" if is_active else "Ver")

    def _on_top_mask_overlay_toggled(self, checked: bool) -> None:
        """Alterna la visibilidad del overlay de la máscara cielo/suelo en el visor interactivo."""
        if self.current_mask is None:
            self._set_mask_overlay_ui_state(False)
            return

        if checked:
            self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
        self.canvas.set_mask_visible(checked)
        self._set_mask_overlay_ui_state(checked)
        self.log_message(f"[MÁSCARA] Visualización de máscara {'activada' if checked else 'oculta'}.")

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
        act_ai = menu.addAction("Auto-Detectar Suelo con IA (1-Click)")
        act_ai.setIcon(get_icon("sparkles", color="#ffffff", size=14))
        act_stacker = None
        if stacker_mask is not None:
            act_stacker = menu.addAction("Usar Máscara generada en el Apilado")
            act_stacker.setIcon(get_icon("layers", color="#80d8ff", size=14))
        act_file = menu.addAction("Cargar Archivo de Máscara (.png, .tif, .fits)...")
        act_file.setIcon(get_icon("folder", color="#80d8ff", size=14))

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
            btn_download = msg.addButton("Descargar IA (~167 MB)", QMessageBox.AcceptRole)
            btn_algo = msg.addButton("Usar Algorítmico", QMessageBox.ActionRole)
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
        self._mask_source_name = source_desc
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
        self.canvas.set_scribbles_visible(False)
        self.canvas.set_mask_visible(True)
        self._set_mask_overlay_ui_state(True)
        self._update_mask_status_ui()
        self._record_mask_action(old_m, self.current_mask, f"Cargar Máscara ({source_desc})")
        self.log_message(f"✓ [MÁSCARA] Máscara cargada y aplicada con éxito desde: {source_desc}.")


    def _update_mask_status_ui(self) -> None:
        """Actualiza el indicador visual de presencia de máscara en la Sección 8 y botones."""
        if hasattr(self, 'sec_gnd_tone'):
            if self.current_mask is not None:
                source_label = f"● Suelo ({self._mask_source_name})" if self._mask_source_name else "● Suelo Activo"
                self.sec_gnd_tone.set_status_badge(source_label, "#a5d6a7")
            else:
                self.sec_gnd_tone.set_status_badge("")

        if hasattr(self, 'mask_layers_widget') and self.image_starless is not None:
            self.mask_layers_widget.ensure_starnet_layers(has_ground=(self.current_mask is not None))

        is_mask_present = (self.current_mask is not None)
        is_visible = (getattr(self.canvas, 'show_mask_overlay', False) and is_mask_present)
        self._set_mask_overlay_ui_state(is_visible)

        if not hasattr(self, 'lbl_gnd_mask_status'):
            return
        if is_mask_present:
            desc = f"Máscara de suelo activa ({self._mask_source_name})" if self._mask_source_name else "Máscara de suelo activa"
            self.lbl_gnd_mask_status.setText(desc)
            self.lbl_gnd_mask_status.setStyleSheet("color: #a5d6a7; font-size: 11px; font-weight: bold;")
            if hasattr(self, 'btn_edit_ground_mask'):
                self.btn_edit_ground_mask.setText("Editar Máscara...")
            if hasattr(self, 'btn_delete_mask'):
                self.btn_delete_mask.setEnabled(True)
        else:
            self.lbl_gnd_mask_status.setText("Sin máscara de suelo activa")
            self.lbl_gnd_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
            if hasattr(self, 'btn_edit_ground_mask'):
                self.btn_edit_ground_mask.setText("Crear Máscara Suelo...")
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
        self._update_canvas_info_label()

    def _on_layer_brush_params_changed(self, radius: int, hardness: float, flow: float, mode: str) -> None:
        """Actualiza los parámetros interactivos de trazo en el lienzo."""
        self.canvas.set_layer_brush_params(radius, hardness, flow, mode)

    def _on_layer_brush_stroke_finished(self, stroke_mask: np.ndarray) -> None:
        """Manejador ejecutado al soltar el clic izquierdo tras pintar con el pincel de capa."""
        layer = self.mask_layers_widget.get_selected_layer()
        if layer and layer.mask_type == "brush":
            layer.brush_mask = stroke_mask.copy()
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.invalidate_layer_thumbnail(layer.id)
            # Si el botón de previsualización NO está activado expresamente, retirar el overlay temporal
            # del trazo para que el usuario pueda apreciar de inmediato los cambios tonales en la imagen
            if hasattr(self, 'mask_layers_widget') and not self.mask_layers_widget.btn_preview_mask.isChecked():
                self.canvas.set_refined_mask(None)
                self.canvas.set_mask_visible(False)
            self._commit_action()
            self.update_stretch_preview()

    def _get_layer_image_for_histogram(self, layer) -> tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Retorna la imagen de trabajo y máscara espacial correspondiente a la capa seleccionada
        para renderizar su histograma de fondo en la Curva Tonal de Capa.
        Aplica los ajustes locales pre-curva de la capa (Exposición, Sombras, Luces, Contraste,
        Saturación, Balance de Blancos) para reflejar dinámicamente los valores que entran a la curva.
        Retorna: (imagen_rgb_estirada, mascara_espacial_2d, es_capa_de_estrellas)
        """
        if self.image_32bit is None:
            return None, None, False

        m_type = getattr(layer, 'mask_type', '')

        # 1. Capa de Estrellas (StarNet AI)
        if m_type == MASK_TYPE_IMAGE_STARS:
            img = getattr(self, '_cached_stars_stretched', None)
            if img is None:
                st_src = self.proxy_stars if self.proxy_stars is not None else self.image_stars
                if st_src is not None:
                    if getattr(self, "stretch_mode", "mtf") == "ghs":
                        img = np.clip(
                            apply_ghs_stretch(
                                st_src,
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
                        img = np.clip(
                            manual_stretch(st_src, black_point=0.0, midtone=self.spin_mtf.value()) * self.star_intensity,
                            0.0, 1.0
                        )
            if img is not None:
                img = apply_pre_curve_adjustments(img, layer)
            return img, None, True

        # 2. Capa de Suelo / Terreno
        if m_type == MASK_TYPE_IMAGE_GROUND:
            img = getattr(self, '_cached_base_stretched', None)
            if img is None:
                img = getattr(self, '_cached_rendered_image', None)
            if img is not None:
                img = apply_pre_curve_adjustments(img, layer)
            sky_m = self.proxy_mask if self.proxy_mask is not None else self.current_mask
            gnd_m = np.clip(1.0 - sky_m, 0.0, 1.0) if sky_m is not None else None
            return img, gnd_m, False

        # 3. Capa de Fondo Nebular (Starless)
        if m_type == MASK_TYPE_IMAGE_STARLESS:
            img = getattr(self, '_cached_base_stretched', None)
            if img is None:
                img = getattr(self, '_cached_rendered_image', None)
            if img is not None:
                img = apply_pre_curve_adjustments(img, layer)
            # Si existe una capa de suelo activa que ya procesa el terreno de forma independiente,
            # aislamos el histograma estrictamente al cielo nocturno usando sky_mask.
            # Si la capa de suelo está desactivada o no existe, la capa starless actúa como base
            # global y su histograma abarca el rango tonal completo del lienzo.
            has_active_ground = False
            if hasattr(self, 'mask_layers_widget'):
                has_active_ground = any(
                    getattr(l, 'mask_type', '') == MASK_TYPE_IMAGE_GROUND and getattr(l, 'enabled', True)
                    for l in self.mask_layers_widget.layers
                )
            sky_m = (self.proxy_mask if self.proxy_mask is not None else self.current_mask) if has_active_ground else None
            return img, sky_m, False

        # 4. Capa de Fondo Base
        if m_type == MASK_TYPE_IMAGE_BASE:
            img = getattr(self, '_cached_base_stretched', None)
            if img is None:
                b_src = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
                if b_src is not None:
                    if getattr(self, "stretch_mode", "mtf") == "ghs":
                        img = apply_ghs_stretch(
                            b_src,
                            sp=self.spin_ghs_sp.value(),
                            d=self.spin_ghs_d.value(),
                            b=self.spin_ghs_b.value(),
                            hp=self.spin_ghs_hp.value(),
                            lp=self.spin_ghs_lp.value(),
                            bp=self.spin_ghs_bp.value(),
                            color_mode=getattr(self, "ghs_color_mode", "luma")
                        )
                    else:
                        img = manual_stretch(b_src, black_point=self.spin_bp.value(), midtone=self.spin_mtf.value())
            if img is None:
                img = getattr(self, '_cached_rendered_image', None)
            if img is not None:
                img = apply_pre_curve_adjustments(img, layer)
            return img, None, False

        # 5. Capas de ajuste local estándar (Pincel, Gradiente, Luminancia, Color, etc.)
        base_canvas = getattr(self, '_cached_base_stretched', None)
        if base_canvas is None:
            base_canvas = getattr(self, '_last_rendered_preview', None)

        if base_canvas is not None and hasattr(self, 'mask_layers_widget'):
            all_layers = getattr(self.mask_layers_widget, 'layers', [])
            sky_m = self.proxy_mask if self.proxy_mask is not None else self.current_mask
            stars_img = getattr(self, '_cached_stars_stretched', None)
            if stars_img is None and getattr(self, 'image_stars', None) is not None:
                stars_img = self.proxy_stars if self.proxy_stars is not None else self.image_stars
            starless_img = self.proxy_starless if self.proxy_starless is not None else self.image_starless

            img = get_canvas_below_layer(
                base_canvas,
                all_layers,
                layer.id,
                master_sky_mask=sky_m,
                starless_image=starless_img,
                stars_image=stars_img
            )
        else:
            img = getattr(self, '_last_rendered_preview', None) or getattr(self, '_cached_base_stretched', None)

        mask = None
        if img is not None:
            sky_m = self.proxy_mask if self.proxy_mask is not None else self.current_mask
            mask = compute_effective_layer_mask(
                layer,
                img,
                master_sky_mask=sky_m
            )
            img = apply_pre_curve_adjustments(img, layer)
        elif m_type == MASK_TYPE_BRUSH:
            mask = getattr(layer, 'brush_mask', None)

        return img, mask, False

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

        if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'lp_center':
            self.btn_lp_center_pipette.blockSignals(True)
            self.btn_lp_center_pipette.setChecked(active)
            self.btn_lp_center_pipette.blockSignals(False)

        if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'lp_color':
            self.btn_lp_color_pipette.blockSignals(True)
            self.btn_lp_color_pipette.setChecked(active)
            self.btn_lp_color_pipette.blockSignals(False)

        if hasattr(self, 'mask_layers_widget') and getattr(self, '_eyedropper_target', 'wb') == 'mask_color':
            self.mask_layers_widget.set_color_picker_active(active)

        if not active:
            self._eyedropper_target = 'wb'

    def _on_toggle_wb_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas para Balance de Blancos asegurando exclusión mutua."""
        if active:
            self._eyedropper_target = 'wb'
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True)
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'wb':
                self.canvas.set_eyedropper_active(False)

    def _on_toggle_ghs_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el Punto de Simetría (SP) de GHS en el lienzo."""
        if active:
            self._eyedropper_target = 'ghs_sp'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True)
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'ghs_sp':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_toggle_lp_center_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el Centro de la Cúpula 2D en el lienzo."""
        if active:
            self._eyedropper_target = 'lp_center'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True)
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'lp_center':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_toggle_lp_color_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el Tinte Cromático de la Cúpula 2D en el lienzo."""
        if active:
            self._eyedropper_target = 'lp_color'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True)
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'lp_color':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_layer_color_sample_requested(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el color de una capa de ajuste en el lienzo."""
        if active:
            self._eyedropper_target = 'mask_color'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            self.canvas.set_eyedropper_active(True)
            self.log_message("[MÁSCARA COLOR] Pipeta activa: haz clic sobre cualquier zona del visor para capturar su color.")
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'mask_color':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_eyedropper_color_sampled(self, r: float, g: float, b: float) -> None:
        """Procesa la muestra del cuentagotas para Balance de Blancos, GHS SP, Cúpula 2D o Máscara de Color."""
        if getattr(self, '_eyedropper_target', 'wb') == 'mask_color':
            disp_rgb = getattr(self.canvas, 'last_sampled_display_rgb', None)
            if disp_rgb is not None:
                dr, dg, db = disp_rgb[0] / 255.0, disp_rgb[1] / 255.0, disp_rgb[2] / 255.0
            else:
                dr, dg, db = r, g, b

            import colorsys
            h, s, v = colorsys.rgb_to_hsv(dr, dg, db)
            hue_deg = int(round(h * 360.0)) % 360
            sat_val = float(np.clip(s, 0.01, 0.90))

            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.apply_sampled_color(hue_deg, sat_val)
                self.mask_layers_widget.set_color_picker_active(False)

            self._eyedropper_target = 'wb'
            self.log_message(
                f"[MÁSCARA COLOR] Color muestreado en visor: Hue={hue_deg}°, Sat={sat_val:.2f}."
            )
            return

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

        if getattr(self, '_eyedropper_target', 'wb') == 'lp_center':
            coords = getattr(self.canvas, 'last_sampled_coords', None)
            if coords is not None:
                cx, cy = coords
            else:
                cx, cy = 0.50, 0.90
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
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
            if hasattr(self, 'btn_lp_center_pipette'):
                self.btn_lp_center_pipette.setChecked(False)
            self._eyedropper_target = 'wb'
            self.log_message(
                f"[Cúpula 2D] Centro seleccionado en el lienzo: X={cx:.2f}, Y={cy:.2f}"
            )
            self.update_stretch_preview()
            self._commit_action()
            return

        if getattr(self, '_eyedropper_target', 'wb') == 'lp_color':
            max_c = max(1e-4, max(r, g, b))
            tint = (float(r / max_c), float(g / max_c), float(b / max_c))
            self.lp_custom_tint = tint
            self.lp_tint_type = "custom"
            if hasattr(self, 'combo_lp_tint'):
                idx = self.combo_lp_tint.findData("custom")
                if idx >= 0:
                    self.combo_lp_tint.blockSignals(True)
                    self.combo_lp_tint.setCurrentIndex(idx)
                    self.combo_lp_tint.blockSignals(False)
            if hasattr(self, 'btn_lp_color_pipette'):
                self.btn_lp_color_pipette.setChecked(False)
            self._eyedropper_target = 'wb'
            self.log_message(
                f"[Cúpula 2D] Tinte muestreado: RGB=({r:.3f}, {g:.3f}, {b:.3f}) ➜ Normalizado=({tint[0]:.2f}, {tint[1]:.2f}, {tint[2]:.2f})"
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

    def _on_live_histogram_clipping_changed(self, clip_s: bool, clip_h: bool) -> None:
        """Sincroniza el lienzo y la curva tonal cuando se conmutan los triángulos de recorte del histograma."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_clipping_state(clip_s, clip_h)
        self.on_clipping_changed_sync(clip_s, clip_h)

    def on_clipping_changed_sync(self, clip_s: bool, clip_h: bool) -> None:
        """Sincroniza el lienzo y el botón de la barra superior cuando cambia el recorte."""
        self.canvas.set_clipping_state(clip_s, clip_h)
        if hasattr(self, 'live_histogram'):
            self.live_histogram.set_clipping_states(clip_s, clip_h)
        if hasattr(self, 'btn_toggle_clipping'):
            self.btn_toggle_clipping.blockSignals(True)
            self.btn_toggle_clipping.setChecked(clip_s or clip_h)
            self.btn_toggle_clipping.blockSignals(False)

    def on_toolbar_clipping_toggled(self, checked: bool) -> None:
        """Manejador al conmutar el botón de recorte en la barra superior."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_clipping_state(checked, checked)
        if hasattr(self, 'live_histogram'):
            self.live_histogram.set_clipping_states(checked, checked)
        self.canvas.set_clipping_state(checked, checked)
        estado = "ACTIVADOS (Azul=Negros, Rojo=Blancos)" if checked else "DESACTIVADOS"
        self.log_message(f"[CANVAS] Indicadores de recorte: {estado}")

    def toggle_left_panel(self, visible: Optional[bool] = None) -> None:
        """Muestra u oculta el panel lateral izquierdo de ajustes (estilo Lightroom)."""
        if not hasattr(self, 'left_container'):
            return
        if visible is None:
            visible = not self.left_container.isVisible()

        # Guardar tamaño previo antes de colapsar
        if not visible and self.left_container.isVisible() and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[0] > 50:
                self._last_left_panel_width = sizes[0]

        self.left_container.setVisible(visible)
        if hasattr(self, 'btn_toggle_left_panel'):
            self.btn_toggle_left_panel.blockSignals(True)
            self.btn_toggle_left_panel.setChecked(visible)
            self.btn_toggle_left_panel.blockSignals(False)

        # Restaurar tamaño previo al desplegar
        if visible and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[0] < 50:
                sizes[0] = getattr(self, '_last_left_panel_width', 380)
                self.main_splitter.setSizes(sizes)

    def toggle_right_panel(self, visible: Optional[bool] = None) -> None:
        """Muestra u oculta el panel lateral derecho de capas e histograma (estilo Lightroom)."""
        if not hasattr(self, 'right_container'):
            return
        if visible is None:
            visible = not self.right_container.isVisible()

        # Guardar tamaño previo antes de colapsar
        if not visible and self.right_container.isVisible() and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[2] > 50:
                self._last_right_panel_width = sizes[2]

        self.right_container.setVisible(visible)
        if hasattr(self, 'btn_toggle_right_panel'):
            self.btn_toggle_right_panel.blockSignals(True)
            self.btn_toggle_right_panel.setChecked(visible)
            self.btn_toggle_right_panel.blockSignals(False)

        # Restaurar tamaño previo al desplegar
        if visible and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[2] < 50:
                sizes[2] = getattr(self, '_last_right_panel_width', 360)
                self.main_splitter.setSizes(sizes)

    def toggle_both_panels(self) -> None:
        """Alterna simultáneamente ambos paneles laterales (Modo Inmersivo / Cine Lightroom)."""
        left_vis = self.left_container.isVisible() if hasattr(self, 'left_container') else False
        right_vis = self.right_container.isVisible() if hasattr(self, 'right_container') else False
        target_vis = not (left_vis or right_vis)
        self.toggle_left_panel(target_vis)
        self.toggle_right_panel(target_vis)

    def _on_toggle_left_wide(self) -> None:
        """Alterna el ancho del panel izquierdo entre normal (~380px) y expandido (~720px o 42% del visor)."""
        if not hasattr(self, 'main_splitter') or not hasattr(self, 'left_container'):
            return
        if not self.left_container.isVisible():
            self.toggle_left_panel(True)
        sizes = self.main_splitter.sizes()
        if len(sizes) >= 3:
            total_w = sum(sizes)
            curr_left = sizes[0]
            target_wide = min(850, max(600, int(total_w * 0.42)))
            if curr_left < 520:
                diff = target_wide - curr_left
                new_center = max(260, sizes[1] - diff)
                self.main_splitter.setSizes([target_wide, new_center, sizes[2]])
            else:
                target_norm = 380
                diff = curr_left - target_norm
                new_center = sizes[1] + diff
                self.main_splitter.setSizes([target_norm, new_center, sizes[2]])

    def _on_toggle_right_wide(self) -> None:
        """Alterna el ancho del panel derecho entre normal (~360px) y expandido (~720px o 42% del visor)."""
        if not hasattr(self, 'main_splitter') or not hasattr(self, 'right_container'):
            return
        if not self.right_container.isVisible():
            self.toggle_right_panel(True)
        sizes = self.main_splitter.sizes()
        if len(sizes) >= 3:
            total_w = sum(sizes)
            curr_right = sizes[2]
            target_wide = min(850, max(600, int(total_w * 0.42)))
            if curr_right < 520:
                diff = target_wide - curr_right
                new_center = max(260, sizes[1] - diff)
                self.main_splitter.setSizes([sizes[0], new_center, target_wide])
            else:
                target_norm = 360
                diff = curr_right - target_norm
                new_center = sizes[1] + diff
                self.main_splitter.setSizes([sizes[0], new_center, target_norm])

    def eventFilter(self, watched, event):
        """Captura eventos del sistema, como doble clic en asas del QSplitter para auto-expandir paneles."""
        if event.type() == QEvent.MouseButtonDblClick:
            if hasattr(self, 'main_splitter'):
                if watched == self.main_splitter.handle(1):
                    self._on_toggle_left_wide()
                    return True
                elif watched == self.main_splitter.handle(2):
                    self._on_toggle_right_wide()
                    return True
        return super().eventFilter(watched, event)

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
        """Actualiza la etiqueta informativa de resolución, herramienta activa y zoom en el pie del lienzo."""
        if hasattr(self, 'lbl_tool_badge'):
            if getattr(self.canvas, 'crop_mode', False):
                self.lbl_tool_badge.setText("● Modo: Recortar y Nivelar (C)")
                self.lbl_tool_badge.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'enable_masking', False):
                self.lbl_tool_badge.setText("● Modo: Máscara Suelo (M)")
                self.lbl_tool_badge.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'layer_brush_active', False):
                self.lbl_tool_badge.setText("● Modo: Pincel de Máscara")
                self.lbl_tool_badge.setStyleSheet("color: #a855f7; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'eyedropper_active', False):
                self.lbl_tool_badge.setText("● Modo: Pipeta de Muestreo (W)")
                self.lbl_tool_badge.setStyleSheet("color: #ffd54f; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'magnifier_active', False):
                self.lbl_tool_badge.setText("● Modo: Lupa de Inspección (Z)")
                self.lbl_tool_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'split_view_enabled', False):
                ori_txt = "Vertical" if self.canvas.split_orientation == "vertical" else "Horizontal"
                self.lbl_tool_badge.setText(f"● Modo: Comparación Split ({ori_txt})")
                self.lbl_tool_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
            else:
                self.lbl_tool_badge.setText("● Modo: Vista General")
                self.lbl_tool_badge.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")

        if hasattr(self, 'lbl_canvas_zoom'):
            zoom_txt = "Zoom 100% (1:1)" if self.is_zoomed_100 else "Ajustar (Fit)"
            self.lbl_canvas_zoom.setText(f"Zoom: {zoom_txt}")

        if not hasattr(self, 'lbl_canvas_info'):
            return
        if self.image_32bit is None:
            self.lbl_canvas_info.setText("Sin imagen cargada")
            return
        h, w = self.image_32bit.shape[:2]
        mp = (w * h) / 1e6
        cs_labels = {
            "srgb": "sRGB",
            "adobe": "Adobe RGB (1998)",
            "prophoto": "ProPhoto RGB"
        }
        cs_name = cs_labels.get(getattr(self, 'current_color_space', 'srgb'), 'sRGB')
        self.lbl_canvas_info.setText(f"{w} × {h} px ({mp:.1f} MP)  |  32-bit Float  |  {cs_name}")

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
        elif event.key() == Qt.Key_Tab:
            self.toggle_both_panels()
            event.accept()
            return
        elif event.key() in (Qt.Key_F7, Qt.Key_BracketLeft):
            self.toggle_left_panel()
            event.accept()
            return
        elif event.key() in (Qt.Key_F8, Qt.Key_BracketRight):
            self.toggle_right_panel()
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

    def _invalidate_pipeline_caches(self):
        """Invalida todas las cachés intermedias del pipeline de procesado."""
        self._cache_starless_key = None
        self._cache_starless_linear = None
        self._cache_starless_pre_key = None
        self._cache_starless_pre_img = None
        self._cache_ms_stretch_key = None
        self._cache_ms_stretch_img = None
        self._cache_ms_graded_key = None
        self._cache_ms_graded_img = None
        self._cache_ms_spatial_key = None
        self._cache_ms_spatial_img = None
        self._cache_ms_stars_key = None
        self._cache_ms_stars_img = None
        self._cache_pre_layers_key = None
        self._cache_pre_layers_stretched = None
        self._cache_pre_layers_stars = None

    def _generate_preview_proxy(self):
        self._invalidate_pipeline_caches()
        if self.image_32bit is None:
            self.pyramid_image = None
            self.pyramid_starless = None
            self.pyramid_stars = None
            self.preview_proxy = None
            self.draft_proxy = None
            self.proxy_starless = None
            self.draft_starless = None
            self.proxy_stars = None
            self.draft_stars = None
            self.proxy_mask = None
            self.draft_mask = None
            return

        h, w = self.image_32bit.shape[:2]

        # Pirámides de resolución multi-escala (Level 1: <=1600px Preview, Level 2: <=800px Draft 60 FPS)
        self.pyramid_image = ResolutionPyramid(self.image_32bit, max_preview_dim=1600, max_levels=4)
        self.preview_proxy = self.pyramid_image.preview
        self.draft_proxy = self.pyramid_image.draft

        if self.image_starless is not None:
            self.pyramid_starless = ResolutionPyramid(self.image_starless, max_preview_dim=1600, max_levels=4)
            self.proxy_starless = self.pyramid_starless.preview
            self.draft_starless = self.pyramid_starless.draft

            self.pyramid_stars = ResolutionPyramid(self.image_stars, max_preview_dim=1600, max_levels=4)
            self.proxy_stars = self.pyramid_stars.preview
            self.draft_stars = self.pyramid_stars.draft
        else:
            self.pyramid_starless = None
            self.draft_starless = None
            self.pyramid_stars = None
            self.draft_stars = None

        clean_mask = None
        if self.current_mask is not None:
            clean_mask = np.squeeze(self.current_mask)
            if clean_mask.ndim > 2:
                clean_mask = clean_mask[..., 0]
            clean_mask = clean_mask.astype(np.float32)
            if clean_mask.shape != (h, w):
                clean_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Máscara nivel Preview (Level 1)
            ph, pw = self.preview_proxy.shape[:2]
            m_small = cv2.resize(clean_mask, (pw, ph), interpolation=cv2.INTER_LINEAR)
            ksize = int(max(7, (min(ph, pw) // 150) | 1))
            if ksize % 2 == 0: 
                ksize += 1
            sm = cv2.GaussianBlur(m_small, (ksize, ksize), sigmaX=ksize / 3.0)
            self.proxy_mask = np.ascontiguousarray(np.repeat(sm[..., np.newaxis], 3, axis=2), dtype=np.float32)

            # Máscara nivel Draft (Level 2)
            dh, dw = self.draft_proxy.shape[:2]
            m_draft = cv2.resize(clean_mask, (dw, dh), interpolation=cv2.INTER_LINEAR)
            ksize_d = int(max(5, (min(dh, dw) // 150) | 1))
            if ksize_d % 2 == 0:
                ksize_d += 1
            sm_d = cv2.GaussianBlur(m_draft, (ksize_d, ksize_d), sigmaX=ksize_d / 3.0)
            self.draft_mask = np.ascontiguousarray(np.repeat(sm_d[..., np.newaxis], 3, axis=2), dtype=np.float32)
        else:
            self.proxy_mask = None
            self.draft_mask = None

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.invalidate_layer_thumbnail(None)

    def load_image_direct(self, filepath: str, mask: np.ndarray = None):
        self._invalidate_pipeline_caches()
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

        # Bypass ultra-rápido si todos los filtros lineales están neutros
        is_neutral = (
            getattr(self, 'denoise_strength', 0.0) <= 1e-4 and
            abs(getattr(self, 'clarity_starless_val', 0.0)) <= 1e-4 and
            abs(getattr(self, 'dehaze_starless_val', 0.0)) <= 1e-4 and
            abs(getattr(self, 'multiescala_estructura', 0.0)) <= 1e-4 and
            getattr(self, 'multiescala_fondo', 0.0) <= 1e-4 and
            getattr(self, 'orton_strength', 0.0) <= 1e-4
        )
        if is_neutral:
            return starless

        is_draft = (not is_full_res and getattr(self, '_is_scrubbing_heavy_slider', False))
        orig_h, orig_w = starless.shape[:2]

        cache_key_full = (
            id(starless),
            id(mask_3d),
            is_full_res,
            is_draft,
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'clarity_starless_val', 0.0)), 4),
            round(float(getattr(self, 'dehaze_starless_val', 0.0)), 4),
            round(float(getattr(self, 'multiescala_estructura', 0.0)), 4),
            round(float(getattr(self, 'multiescala_fondo', 0.0)), 4),
            round(float(getattr(self, 'orton_strength', 0.0)), 4),
            round(float(getattr(self, 'orton_radius', 40.0)), 2),
            round(float(getattr(self, 'orton_threshold', 0.05)), 4),
            getattr(self, 'orton_tint_type', 'warm_gold'),
            getattr(self, 'orton_custom_tint', (1.0, 0.85, 0.55)),
            round(float(getattr(self, 'orton_tint_strength', 0.5)), 4),
            getattr(self, 'orton_mode', 'screen')
        )
        if not is_full_res and self._cache_starless_key == cache_key_full and self._cache_starless_linear is not None:
            return self._cache_starless_linear

        # Verificación en caché de RAM Nivel 1b (Denoise, Claridad, Ondículas pre-Orton)
        cache_key_pre = (
            id(starless),
            id(mask_3d),
            is_full_res,
            is_draft,
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'clarity_starless_val', 0.0)), 4),
            round(float(getattr(self, 'dehaze_starless_val', 0.0)), 4),
            round(float(getattr(self, 'multiescala_estructura', 0.0)), 4),
            round(float(getattr(self, 'multiescala_fondo', 0.0)), 4),
        )
        if not is_full_res and self._cache_starless_pre_key == cache_key_pre and self._cache_starless_pre_img is not None:
            proc_starless = self._cache_starless_pre_img.copy()
            proc_mask_3d = mask_3d
        else:
            if is_draft and max(orig_w, orig_h) > 800:
                scale_d = 800.0 / float(max(orig_w, orig_h))
                draft_w = max(32, int(orig_w * scale_d))
                draft_h = max(32, int(orig_h * scale_d))
                proc_starless = cv2.resize(starless, (draft_w, draft_h), interpolation=cv2.INTER_AREA)
                proc_mask_3d = cv2.resize(mask_3d, (draft_w, draft_h), interpolation=cv2.INTER_AREA) if mask_3d is not None else None
            else:
                proc_starless = starless.copy()
                proc_mask_3d = mask_3d

            # 1. Reducción de Ruido
            if self.denoise_strength > 1e-4:
                denoised = apply_denoise(
                    proc_starless,
                    strength=self.denoise_strength,
                    is_full_res=(is_full_res and not is_draft)
                )
                if proc_mask_3d is not None:
                    proc_starless = (denoised * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = denoised

            # 2. Claridad y Borrar Neblina
            if abs(self.clarity_starless_val) > 1e-4:
                clarified = apply_clarity(proc_starless, strength=self.clarity_starless_val)
                if proc_mask_3d is not None:
                    proc_starless = (clarified * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = clarified

            if abs(self.dehaze_starless_val) > 1e-4:
                dehazed = apply_dehaze(proc_starless, strength=self.dehaze_starless_val)
                if proc_mask_3d is not None:
                    proc_starless = (dehazed * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = dehazed

            # 3. Ondículas sobre la capa de fondo
            if abs(self.multiescala_estructura) > 1e-4 or self.multiescala_fondo > 1e-4:
                proc_starless = realce_multiescala_ondiculas(
                    proc_starless,
                    fuerza_estructura=self.multiescala_estructura,
                    reduccion_fondo=self.multiescala_fondo,
                    sky_mask=proc_mask_3d[..., 0] if proc_mask_3d is not None else None
                )

            if not is_full_res:
                self._cache_starless_pre_key = cache_key_pre
                self._cache_starless_pre_img = proc_starless.copy()

        # 4. Efecto Orton y Resplandor de Polvo Galáctico sobre Fondo Nebular
        if self.orton_strength > 1e-4:
            proc_starless = apply_astronomical_orton(
                proc_starless,
                strength=self.orton_strength,
                radius=self.orton_radius * (0.5 if is_draft else 1.0),
                threshold=self.orton_threshold,
                exponent=1.0,
                tint_type=self.orton_tint_type,
                custom_tint=getattr(self, 'orton_custom_tint', (1.0, 0.85, 0.55)),
                tint_strength=self.orton_tint_strength,
                blend_mode=self.orton_mode,
                sky_mask=proc_mask_3d[..., 0] if proc_mask_3d is not None else None,
                is_starless=True
            )

        if is_draft and proc_starless.shape[:2] != (orig_h, orig_w):
            processed_starless = cv2.resize(proc_starless, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            processed_starless = proc_starless
            if not is_full_res:
                self._cache_starless_key = cache_key_full
                self._cache_starless_linear = processed_starless

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
            is_scrubbing = getattr(self, '_is_scrubbing_heavy_slider', False)
            if is_scrubbing and getattr(self, 'draft_proxy', None) is not None:
                starless = self.draft_starless
                stars = self.draft_stars
                base = self.draft_proxy
                mask_3d = self.draft_mask
            else:
                starless = self.proxy_starless
                stars = self.proxy_stars
                base = self.preview_proxy
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
        apply_mask_layers: bool = True,
        is_preview: bool = True
    ) -> np.ndarray:
        if target_img is None:
            return None

        is_full_res = (self.image_32bit is not None and target_img.shape[:2] == self.image_32bit.shape[:2])
        is_zoomed = (self.is_zoomed_100 and getattr(self, '_zoom_crop_rect', None) is not None)
        full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None
        crop_rect = getattr(self, '_zoom_crop_rect', None) if is_zoomed else None

        curve_state = tuple((m, tuple(pts)) for m, pts in sorted(self.curve_widget.get_state().items())) if hasattr(self, 'curve_widget') else ()

        # Hito 1: Parámetros Lineales y Estirado (MTF / GHS)
        key_stretch = (
            id(target_img),
            id(precomputed_mask),
            is_full_res,
            is_zoomed,
            crop_rect,
            round(float(self.temp_val), 4),
            round(float(self.tint_val), 4),
            round(float(self.sat_sky_val), 4),
            round(float(self.sat_gnd_val), 4),
            round(float(self.vibrance_val), 4),
            round(float(self.lp_reduction), 4),
            round(float(self.lp_center_x), 4),
            round(float(self.lp_center_y), 4),
            round(float(self.lp_width), 4),
            round(float(self.lp_height), 4),
            getattr(self, 'lp_tint_type', 'sodium'),
            getattr(self, 'lp_custom_tint', (1.0, 0.68, 0.22)),
            round(float(getattr(self, 'lp_tint_strength', 0.5)), 4),
            getattr(self, 'lp_mode', 'attenuate'),
            getattr(self, 'lp_legacy_1d', False),
            round(float(self.gnd_ev_val), 4),
            round(float(self.gnd_shadows_val), 4),
            round(float(self.gnd_bp_val), 4),
            getattr(self, "stretch_mode", "mtf"),
            getattr(self, "ghs_color_mode", "luma"),
            round(float(self.spin_ghs_sp.value()), 5) if hasattr(self, 'spin_ghs_sp') else 0.0,
            round(float(self.spin_ghs_d.value()), 4) if hasattr(self, 'spin_ghs_d') else 0.0,
            round(float(self.spin_ghs_b.value()), 4) if hasattr(self, 'spin_ghs_b') else 0.0,
            round(float(self.spin_ghs_hp.value()), 4) if hasattr(self, 'spin_ghs_hp') else 0.0,
            round(float(self.spin_ghs_lp.value()), 4) if hasattr(self, 'spin_ghs_lp') else 0.0,
            round(float(self.spin_ghs_bp.value()), 5) if hasattr(self, 'spin_ghs_bp') else 0.0,
            round(float(self.spin_bp.value()), 5) if hasattr(self, 'spin_bp') else 0.0,
            round(float(self.spin_mtf.value()), 5) if hasattr(self, 'spin_mtf') else 0.0,
        )

        # Hito 2: Parámetros de Curvas y Contraste
        key_graded = (
            key_stretch,
            curve_state,
            round(float(self.contrast_starless_val), 4),
            round(float(self.contrast_val), 4),
        )

        # Hito 3: Parámetros Espaciales (Deconvolución, Orton, Difusión Estelar)
        key_spatial = (
            key_graded,
            getattr(self, "deconv_iterations", 0),
            round(float(getattr(self, "deconv_fwhm", 3.0)), 2),
            round(float(getattr(self, "deconv_deringing", 0.08)), 4),
            round(float(getattr(self, "deconv_damping", 0.25)), 4),
            round(float(getattr(self, "deconv_aspect", 1.0)), 2),
            round(float(getattr(self, "deconv_angle", 0.0)), 2),
            getattr(self, "deconv_model", "gaussian"),
            getattr(self, "deconv_scope", "full"),
            getattr(self, "deconv_luma_only", True),
            round(float(self.orton_strength), 4),
            round(float(self.orton_radius), 2),
            round(float(self.orton_threshold), 4),
            getattr(self, 'orton_tint_type', 'warm_gold'),
            getattr(self, 'orton_custom_tint', (1.0, 0.85, 0.55)),
            round(float(getattr(self, 'orton_tint_strength', 0.5)), 4),
            getattr(self, 'orton_mode', 'screen'),
            round(float(self.star_diff_strength), 4),
            round(float(self.star_diff_radius), 2),
            round(float(self.star_diff_threshold), 4),
            round(float(self.star_diff_color_boost), 4),
            getattr(self, 'star_diff_mode', 'natural'),
        )

        # Determinar hitos en caché reutilizables
        pre_spatial = self._cache_ms_spatial_img if (is_preview and self._cache_ms_spatial_key == key_spatial) else None
        pre_graded = self._cache_ms_graded_img if (is_preview and self._cache_ms_graded_key == key_graded and pre_spatial is None) else None
        pre_stretched = self._cache_ms_stretch_img if (is_preview and self._cache_ms_stretch_key == key_stretch and pre_spatial is None and pre_graded is None) else None
        pre_stars = self._cache_ms_stars_img if (is_preview and self._cache_ms_stretch_key == key_stretch and self._cache_ms_stars_img is not None) else None

        # Resolve starless and stars image slices
        if is_zoomed:
            x1, y1, x2, y2 = self._zoom_crop_rect
            st_img = self.image_stars[y1:y2, x1:x2].copy() if self.image_stars is not None else None
            sl_img = self.image_starless[y1:y2, x1:x2].copy() if self.image_starless is not None else None
        elif is_full_res:
            st_img = self.image_stars
            sl_img = self.image_starless
        else:
            if getattr(self, '_is_scrubbing_heavy_slider', False):
                st_img = self.draft_stars if self.draft_stars is not None else self.proxy_stars
                sl_img = self.draft_starless if self.draft_starless is not None else self.proxy_starless
            else:
                st_img = self.proxy_stars if self.proxy_stars is not None else self.image_stars
                sl_img = self.proxy_starless if self.proxy_starless is not None else self.image_starless

        # Build DeveloperState snapshot
        state = DeveloperState.from_dict(self._capture_state())

        # Collect tone curves
        curve_luts = {}
        if hasattr(self, 'curve_widget'):
            if not self.curve_widget.is_identity("rgb"):
                curve_luts["rgb"] = self.curve_widget.get_lut("rgb")
            if not self.curve_widget.is_identity("hue_sat"):
                curve_luts["hue_sat"] = self.curve_widget.get_lut("hue_sat")
            if not self.curve_widget.is_identity("luma_sat"):
                curve_luts["luma_sat"] = self.curve_widget.get_lut("luma_sat")

        # Get PSF kernel
        psf_kernel = self._get_current_deconv_psf() if hasattr(self, '_get_current_deconv_psf') else None

        # Mask layers
        mask_layers = self.mask_layers_widget.layers if (apply_mask_layers and hasattr(self, 'mask_layers_widget')) else None

        is_scrubbing = getattr(self, '_is_scrubbing_heavy_slider', False)

        final_img, out_str, out_grd, out_spt, out_stars = apply_developer_pipeline(
            target_img=target_img,
            state=state,
            precomputed_mask=precomputed_mask,
            image_stars=st_img,
            image_starless=sl_img,
            mask_layers=mask_layers,
            curve_luts=curve_luts,
            psf_kernel=psf_kernel,
            is_preview=is_preview,
            is_scrubbing=is_scrubbing,
            crop_rect=crop_rect,
            full_shape=full_shape,
            precomputed_stretched_base=pre_stretched,
            precomputed_graded_base=pre_graded,
            precomputed_spatial_base=pre_spatial,
            precomputed_stretched_stars=pre_stars,
            return_milestones=True,
            is_dragging_deconv=getattr(self, '_is_dragging_deconv', False),
        )

        if is_preview:
            self._cache_ms_stretch_key = key_stretch
            self._cache_ms_stretch_img = out_str
            self._cache_ms_graded_key = key_graded
            self._cache_ms_graded_img = out_grd
            self._cache_ms_spatial_key = key_spatial
            self._cache_ms_spatial_img = out_spt
            self._cache_ms_stars_img = out_stars
            if not is_scrubbing:
                self._cache_pre_layers_key = key_spatial
                self._cache_pre_layers_stretched = out_spt
                self._cache_pre_layers_stars = out_stars
                self._cached_base_stretched = out_spt
                self._cached_stars_stretched = out_stars
                self._cached_sky_mask = precomputed_mask
                self._cached_rendered_image = final_img

        return final_img

    def update_stretch_preview(self, immediate: bool = False):
        if immediate:
            if hasattr(self, '_preview_throttle_timer'):
                self._preview_throttle_timer.stop()
            if hasattr(self, '_refine_timer'):
                self._refine_timer.stop()
            self._do_update_stretch_preview()
            self._refresh_histograms_now()
        else:
            if getattr(self, '_is_slider_pressed', False):
                self._is_scrubbing_heavy_slider = True

            if hasattr(self, '_refine_timer') and getattr(self, '_is_scrubbing_heavy_slider', False):
                self._refine_timer.start(120)

            if hasattr(self, '_preview_throttle_timer'):
                self._preview_throttle_timer.start(16)
            else:
                self._do_update_stretch_preview()
                self._refresh_histograms_now()

    def _on_refine_preview_timeout(self):
        """Refina progresivamente la previsualización al nivel de alta calidad cuando cesa el arrastre."""
        if getattr(self, '_is_scrubbing_heavy_slider', False):
            self._is_scrubbing_heavy_slider = False
            self.update_stretch_preview(immediate=True)

    def _refresh_histograms_now(self):
        if getattr(self, '_hist_throttle_timer', None) is not None:
            self._hist_throttle_timer.stop()
        # Omitir cálculo síncrono durante arrastre activo para garantizar 60 FPS
        if getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_slider_pressed', False):
            return
        stretched = getattr(self, '_last_rendered_preview', None)
        if stretched is None or self.is_zoomed_100:
            return
        # Evaluar estrictamente los histogramas cuyos contenedores están visibles
        if hasattr(self, 'live_histogram') and self.live_histogram.isVisible():
            self.live_histogram.update_from_image(stretched)
        if hasattr(self, 'curve_widget') and self.curve_widget.isVisible():
            self.curve_widget.set_histogram_from_image(stretched)
        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.isVisible():
            if getattr(self.mask_layers_widget, 'sec_layer_curve', None) and self.mask_layers_widget.sec_layer_curve.is_expanded:
                self.mask_layers_widget.update_active_layer_histogram()

    def _do_update_stretch_preview(self):
        if getattr(self, '_is_rendering_preview', False):
            return
        self._is_rendering_preview = True
        try:
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
                self._last_rendered_preview = stretched
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
                    disp_b = cv2.convertScaleAbs(np.clip(stretched_before, 0.0, 1.0), alpha=255.0)
                    h_b, w_b = disp_b.shape[:2]
                    disp_bc = np.ascontiguousarray(disp_b, dtype=np.uint8)
                    qimg_b = QImage(disp_bc.data, w_b, h_b, 3 * w_b, QImage.Format_RGB888)
                    self.canvas.set_before_pixmap(QPixmap.fromImage(qimg_b))

                # Actualización diferida del histograma durante reposo
                if not self.is_zoomed_100 and hasattr(self, '_hist_throttle_timer'):
                    if not getattr(self, '_is_scrubbing_heavy_slider', False) and not self._hist_throttle_timer.isActive():
                        self._hist_throttle_timer.start(70)

                self._update_canvas_info_label()

                # Reiniciar temporizador de consolidación de historial (captura scroll de ratón y flechas)
                if not self._is_undoing_or_redoing and hasattr(self, '_undo_debounce_timer'):
                    self._undo_debounce_timer.start(500)
        finally:
            self._is_rendering_preview = False

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
            self.mask_layers_widget.set_stars_special_values(presence=val)
        self.update_stretch_preview()

    def _on_mask_widget_stars_intensity(self, val: int):
        if hasattr(self, 'slider_stars') and self.slider_stars.value() != val:
            self.slider_stars.setValue(val)

    def on_star_diff_strength_changed(self, val: int):
        self.star_diff_strength = val / 100.0
        self.lbl_star_diff.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(diff_strength=val)
        self.update_stretch_preview()

    def on_star_diff_radius_changed(self, val: int):
        self.star_diff_radius = float(val)
        self.lbl_star_diff_radius.setText(f"{val} px")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(diff_radius=val)
        self.update_stretch_preview()

    def on_star_diff_threshold_changed(self, val: int):
        self.star_diff_threshold = val / 100.0
        self.lbl_star_diff_threshold.setText(f"{val / 100.0:.2f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(diff_threshold=val)
        self.update_stretch_preview()

    def on_star_diff_color_changed(self, val: int):
        self.star_diff_color_boost = val / 100.0
        self.lbl_star_diff_color.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(diff_color=val)
        self.update_stretch_preview()

    def _on_star_diff_mode_changed(self, idx: int):
        modes = ["screen", "soft_light", "additive"]
        mode_key = modes[idx] if 0 <= idx < len(modes) else "screen"
        self.star_diff_mode = mode_key
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(diff_mode_idx=idx)
        self.update_stretch_preview()
        self._commit_action()

    def _on_toggle_star_diff_expand(self, checked: bool):
        if hasattr(self, 'w_star_diff_advanced'):
            self.w_star_diff_advanced.setVisible(checked)

    # =========================================================================
    # GESTIÓN Y EVENTOS DE DECONVOLUCIÓN ÓPTICA REGULARIZADA (RICHARDSON-LUCY)
    # =========================================================================
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

    def on_deconv_iterations_changed(self, val: int):
        self.deconv_iterations = int(val)
        txt = f"{val}" if val > 0 else "0 (Off)"
        self.lbl_deconv_iter.setText(txt)
        self.update_stretch_preview()

    def on_deconv_fwhm_changed(self, val: int):
        fwhm = val / 10.0
        self.deconv_fwhm = float(fwhm)
        self.lbl_deconv_fwhm.setText(f"{fwhm:.1f} px")
        self._cached_psf = None
        self._cached_psf_key = ()
        if self.deconv_iterations > 0:
            self.update_stretch_preview()

    def on_deconv_deringing_changed(self, val: int):
        der = val / 1000.0
        self.deconv_deringing = float(der)
        self.lbl_deconv_deringing.setText(f"{der:.3f}")
        if self.deconv_iterations > 0:
            self.update_stretch_preview()

    def on_deconv_damping_changed(self, val: int):
        damp = val / 100.0
        self.deconv_damping = float(damp)
        self.lbl_deconv_damping.setText(f"{val}%")
        if self.deconv_iterations > 0:
            self.update_stretch_preview()

    def on_deconv_aspect_changed(self, val: int):
        aspect = val / 100.0
        self.deconv_aspect_ratio = float(aspect)
        self.lbl_deconv_aspect.setText(f"{aspect:.2f}x")
        self._cached_psf = None
        self._cached_psf_key = ()
        if self.deconv_iterations > 0:
            self.update_stretch_preview()

    def on_deconv_angle_changed(self, val: int):
        angle = float(val)
        self.deconv_angle_deg = angle
        self.lbl_deconv_angle.setText(f"{val}°")
        self._cached_psf = None
        self._cached_psf_key = ()
        if self.deconv_iterations > 0:
            self.update_stretch_preview()

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

    def _on_deconv_model_changed(self, idx: int):
        model = self.combo_deconv_model.itemData(idx) or ("moffat" if idx == 0 else "gaussian")
        self.deconv_model = model
        self._cached_psf = None
        self._cached_psf_key = ()
        if self.deconv_iterations > 0:
            self.update_stretch_preview()
        self._commit_action()

    def _on_deconv_scope_changed(self, idx: int):
        scope = self.combo_deconv_scope.itemData(idx) or "all"
        self.deconv_scope = scope
        if self.deconv_iterations > 0:
            self.update_stretch_preview()
        self._commit_action()

    def _on_deconv_luma_only_toggled(self, checked: bool):
        self.deconv_luma_only = bool(checked)
        if self.deconv_iterations > 0:
            self.update_stretch_preview()
        self._commit_action()

    def _on_toggle_deconv_expand(self, checked: bool):
        if hasattr(self, 'w_deconv_advanced'):
            self.w_deconv_advanced.setVisible(checked)

    def reset_deconv(self):
        """Restablece los parámetros de deconvolución a sus valores neutros por defecto."""
        self.deconv_iterations = 0
        self.deconv_fwhm = 3.0
        self.deconv_model = "moffat"
        self.deconv_beta = 4.5
        self.deconv_aspect_ratio = 1.0
        self.deconv_angle_deg = 0.0
        self.deconv_deringing = 0.008
        self.deconv_damping = 0.25
        self.deconv_luma_only = True
        self.deconv_scope = "all"
        self._cached_psf = None
        self._cached_psf_key = ()

        widgets_to_block = [
            getattr(self, 'slider_deconv_iter', None),
            getattr(self, 'slider_deconv_fwhm', None),
            getattr(self, 'slider_deconv_deringing', None),
            getattr(self, 'slider_deconv_damping', None),
            getattr(self, 'slider_deconv_aspect', None),
            getattr(self, 'slider_deconv_angle', None),
            getattr(self, 'combo_deconv_model', None),
            getattr(self, 'combo_deconv_scope', None),
            getattr(self, 'chk_deconv_luma_only', None),
        ]
        for w in widgets_to_block:
            if w is not None:
                w.blockSignals(True)

        if hasattr(self, 'slider_deconv_iter'):
            self.slider_deconv_iter.setValue(0)
            self.lbl_deconv_iter.setText("0 (Off)")
        if hasattr(self, 'slider_deconv_fwhm'):
            self.slider_deconv_fwhm.setValue(30)
            self.lbl_deconv_fwhm.setText("3.0 px")
        if hasattr(self, 'slider_deconv_deringing'):
            self.slider_deconv_deringing.setValue(8)
            self.lbl_deconv_deringing.setText("0.008")
        if hasattr(self, 'slider_deconv_damping'):
            self.slider_deconv_damping.setValue(25)
            self.lbl_deconv_damping.setText("25%")
        if hasattr(self, 'slider_deconv_aspect'):
            self.slider_deconv_aspect.setValue(100)
            self.lbl_deconv_aspect.setText("1.00x")
        if hasattr(self, 'slider_deconv_angle'):
            self.slider_deconv_angle.setValue(0)
            self.lbl_deconv_angle.setText("0°")
        if hasattr(self, 'combo_deconv_model'):
            self.combo_deconv_model.setCurrentIndex(0)
        if hasattr(self, 'combo_deconv_scope'):
            self.combo_deconv_scope.setCurrentIndex(0)
        if hasattr(self, 'chk_deconv_luma_only'):
            self.chk_deconv_luma_only.setChecked(True)
        if hasattr(self, 'lbl_psf_status'):
            self.lbl_psf_status.setText("PSF: Moffat 2D (FWHM: 3.0 px | Aspecto: 1.00)")

        for w in widgets_to_block:
            if w is not None:
                w.blockSignals(False)

        self.update_stretch_preview()

    def _on_mask_widget_stars_diff_strength(self, val: int):
        if hasattr(self, 'slider_star_diff') and self.slider_star_diff.value() != val:
            self.slider_star_diff.setValue(val)

    def _on_mask_widget_stars_diff_radius(self, val: int):
        if hasattr(self, 'slider_star_diff_radius') and self.slider_star_diff_radius.value() != val:
            self.slider_star_diff_radius.setValue(val)

    def _on_mask_widget_stars_diff_threshold(self, val: int):
        if hasattr(self, 'slider_star_diff_threshold') and self.slider_star_diff_threshold.value() != val:
            self.slider_star_diff_threshold.setValue(val)

    def _on_mask_widget_stars_diff_color(self, val: int):
        if hasattr(self, 'slider_star_diff_color') and self.slider_star_diff_color.value() != val:
            self.slider_star_diff_color.setValue(val)

    def _on_mask_widget_stars_diff_mode(self, mode_key: str):
        if hasattr(self, 'combo_star_diff_mode'):
            modes = ["screen", "soft_light", "additive"]
            idx = modes.index(mode_key) if mode_key in modes else 0
            if self.combo_star_diff_mode.currentIndex() != idx:
                self.combo_star_diff_mode.setCurrentIndex(idx)

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

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_stars_special_values(
                presence=100,
                diff_strength=0,
                diff_radius=20,
                diff_threshold=45,
                diff_color=50,
                diff_mode_idx=0
            )
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

    def _on_color_wb_parameter_changed(self):
        """Sincroniza el estado del panel modular de Color/WB con el revelador y actualiza la vista."""
        if hasattr(self, 'panel_color_wb'):
            self.temp_val = self.panel_color_wb.temp_val
            self.tint_val = self.panel_color_wb.tint_val
            self.sat_sky_val = self.panel_color_wb.sat_sky_val
            self.sat_gnd_val = self.panel_color_wb.sat_gnd_val
            self.vibrance_val = self.panel_color_wb.vibrance_val
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
        self.update_stretch_preview(immediate=True)
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

    def _on_stretch_parameter_changed(self):
        """Sincroniza el estado del panel modular de Estirado con el revelador."""
        if hasattr(self, 'panel_stretch'):
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
        self.update_stretch_preview()

    def _set_stretch_mode(self, mode: str):
        """Alterna el modo de estirado entre MTF clásico y GHS hiperbólico."""
        if mode not in ("mtf", "ghs"):
            return
        self.stretch_mode = mode
        if hasattr(self, 'panel_stretch'):
            self.panel_stretch.set_stretch_mode(mode)
        else:
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
        if hasattr(self, 'panel_stretch'):
            self.panel_stretch.set_ghs_color_mode(mode)
        else:
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

    def on_spin_lp_cx_changed(self, val: float):
        self.lp_center_x = float(val)
        if hasattr(self, 'slider_lp_cx'):
            self.slider_lp_cx.blockSignals(True)
            self.slider_lp_cx.setValue(int(round(val * 100.0)))
            self.slider_lp_cx.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_lp_cx_changed(self, val: int):
        cx = val / 100.0
        self.lp_center_x = cx
        if hasattr(self, 'spin_lp_cx'):
            self.spin_lp_cx.blockSignals(True)
            self.spin_lp_cx.setValue(cx)
            self.spin_lp_cx.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_lp_cy_changed(self, val: float):
        self.lp_center_y = float(val)
        if hasattr(self, 'slider_lp_cy'):
            self.slider_lp_cy.blockSignals(True)
            self.slider_lp_cy.setValue(int(round(val * 100.0)))
            self.slider_lp_cy.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_lp_cy_changed(self, val: int):
        cy = val / 100.0
        self.lp_center_y = cy
        if hasattr(self, 'spin_lp_cy'):
            self.spin_lp_cy.blockSignals(True)
            self.spin_lp_cy.setValue(cy)
            self.spin_lp_cy.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_lp_w_changed(self, val: float):
        self.lp_width = float(val)
        if hasattr(self, 'slider_lp_w'):
            self.slider_lp_w.blockSignals(True)
            self.slider_lp_w.setValue(int(round(val * 100.0)))
            self.slider_lp_w.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_lp_w_changed(self, val: int):
        w = val / 100.0
        self.lp_width = w
        if hasattr(self, 'spin_lp_w'):
            self.spin_lp_w.blockSignals(True)
            self.spin_lp_w.setValue(w)
            self.spin_lp_w.blockSignals(False)
        self.update_stretch_preview()

    def on_spin_lp_h_changed(self, val: float):
        self.lp_height = float(val)
        if hasattr(self, 'slider_lp_h'):
            self.slider_lp_h.blockSignals(True)
            self.slider_lp_h.setValue(int(round(val * 100.0)))
            self.slider_lp_h.blockSignals(False)
        self.update_stretch_preview()

    def on_slider_lp_h_changed(self, val: int):
        h = val / 100.0
        self.lp_height = h
        if hasattr(self, 'spin_lp_h'):
            self.spin_lp_h.blockSignals(True)
            self.spin_lp_h.setValue(h)
            self.spin_lp_h.blockSignals(False)
        self.update_stretch_preview()

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

    def _on_lp_tint_combo_changed(self, idx: int):
        tint_type = self.combo_lp_tint.currentData()
        self.lp_tint_type = tint_type
        self.update_stretch_preview()
        self._commit_action()

    def _on_lp_tint_strength_changed(self, val: int):
        self.lp_tint_strength = val / 100.0
        if hasattr(self, 'lbl_lp_tint_strength'):
            self.lbl_lp_tint_strength.setText(f"{val}%")
        self.update_stretch_preview()

    def _on_lp_mode_combo_changed(self, idx: int):
        mode = self.combo_lp_mode.currentData()
        self.lp_mode = mode
        self.update_stretch_preview()
        self._commit_action()

    def _on_lp_legacy_1d_toggled(self, checked: bool):
        self.lp_legacy_1d = checked
        self.update_stretch_preview()
        self._commit_action()

    def reset_gradients(self):
        """Restablece los parámetros de eliminación de gradientes y cúpula de polución 2D."""
        self.slider_lp.setValue(0)
        self.lp_reduction = 0.0
        self.lbl_lp.setText("0%")
        self.lp_center_x = 0.50
        self.lp_center_y = 0.90
        self.lp_width = 0.60
        self.lp_height = 0.45
        self.lp_tint_type = "sodium"
        self.lp_custom_tint = (1.0, 0.68, 0.22)
        self.lp_tint_strength = 0.50
        self.lp_mode = "attenuate"
        self.lp_legacy_1d = False

        if hasattr(self, 'slider_lp_cx'):
            self.slider_lp_cx.blockSignals(True)
            self.slider_lp_cx.setValue(50)
            self.slider_lp_cx.blockSignals(False)
        if hasattr(self, 'spin_lp_cx'):
            self.spin_lp_cx.blockSignals(True)
            self.spin_lp_cx.setValue(0.50)
            self.spin_lp_cx.blockSignals(False)

        if hasattr(self, 'slider_lp_cy'):
            self.slider_lp_cy.blockSignals(True)
            self.slider_lp_cy.setValue(90)
            self.slider_lp_cy.blockSignals(False)
        if hasattr(self, 'spin_lp_cy'):
            self.spin_lp_cy.blockSignals(True)
            self.spin_lp_cy.setValue(0.90)
            self.spin_lp_cy.blockSignals(False)

        if hasattr(self, 'slider_lp_w'):
            self.slider_lp_w.blockSignals(True)
            self.slider_lp_w.setValue(60)
            self.slider_lp_w.blockSignals(False)
        if hasattr(self, 'spin_lp_w'):
            self.spin_lp_w.blockSignals(True)
            self.spin_lp_w.setValue(0.60)
            self.spin_lp_w.blockSignals(False)

        if hasattr(self, 'slider_lp_h'):
            self.slider_lp_h.blockSignals(True)
            self.slider_lp_h.setValue(45)
            self.slider_lp_h.blockSignals(False)
        if hasattr(self, 'spin_lp_h'):
            self.spin_lp_h.blockSignals(True)
            self.spin_lp_h.setValue(0.45)
            self.spin_lp_h.blockSignals(False)

        if hasattr(self, 'combo_lp_tint'):
            self.combo_lp_tint.blockSignals(True)
            self.combo_lp_tint.setCurrentIndex(0)
            self.combo_lp_tint.blockSignals(False)

        if hasattr(self, 'slider_lp_tint_strength'):
            self.slider_lp_tint_strength.blockSignals(True)
            self.slider_lp_tint_strength.setValue(50)
            self.slider_lp_tint_strength.blockSignals(False)
        if hasattr(self, 'lbl_lp_tint_strength'):
            self.lbl_lp_tint_strength.setText("50%")

        if hasattr(self, 'combo_lp_mode'):
            self.combo_lp_mode.blockSignals(True)
            self.combo_lp_mode.setCurrentIndex(0)
            self.combo_lp_mode.blockSignals(False)

        if hasattr(self, 'chk_lp_legacy_1d'):
            self.chk_lp_legacy_1d.blockSignals(True)
            self.chk_lp_legacy_1d.setChecked(False)
            self.chk_lp_legacy_1d.blockSignals(False)

        self.update_stretch_preview()
        self._commit_action()

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

    def _on_toggle_orton_expand(self, checked: bool):
        if hasattr(self, 'w_orton_advanced'):
            self.w_orton_advanced.setVisible(checked)

    def on_orton_strength_changed(self, val: int):
        self.orton_strength = val / 100.0
        if hasattr(self, 'lbl_orton'):
            self.lbl_orton.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                val, self.slider_orton_radius.value(), self.slider_orton_threshold.value(),
                self.combo_orton_tint.currentIndex(), self.combo_orton_mode.currentIndex()
            )
        self.update_stretch_preview()

    def on_orton_radius_changed(self, val: int):
        self.orton_radius = float(val)
        if hasattr(self, 'lbl_orton_radius'):
            self.lbl_orton_radius.setText(f"{val} px")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                self.slider_orton.value(), val, self.slider_orton_threshold.value(),
                self.combo_orton_tint.currentIndex(), self.combo_orton_mode.currentIndex()
            )
        self.update_stretch_preview()

    def on_orton_threshold_changed(self, val: int):
        self.orton_threshold = val / 100.0
        if hasattr(self, 'lbl_orton_threshold'):
            self.lbl_orton_threshold.setText(f"{self.orton_threshold:.2f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                self.slider_orton.value(), self.slider_orton_radius.value(), val,
                self.combo_orton_tint.currentIndex(), self.combo_orton_mode.currentIndex()
            )
        self.update_stretch_preview()

    def _on_orton_tint_changed(self, idx: int):
        tint_key = self.combo_orton_tint.currentData()
        self.orton_tint_type = tint_key
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                self.slider_orton.value(), self.slider_orton_radius.value(), self.slider_orton_threshold.value(),
                idx, self.combo_orton_mode.currentIndex()
            )
        self.update_stretch_preview()
        self._commit_action()

    def _on_orton_mode_changed(self, idx: int):
        mode_key = self.combo_orton_mode.currentData()
        self.orton_mode = mode_key
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                self.slider_orton.value(), self.slider_orton_radius.value(), self.slider_orton_threshold.value(),
                self.combo_orton_tint.currentIndex(), idx
            )
        self.update_stretch_preview()
        self._commit_action()

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

    def reset_starless_adjustments(self):
        widgets = [
            self.slider_clarity, self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_contrast_starless, self.slider_denoise,
            self.slider_orton, self.slider_orton_radius, self.slider_orton_threshold,
            self.combo_orton_tint, self.combo_orton_mode
        ]
        for w in widgets:
            w.blockSignals(True)
        self.slider_clarity.setValue(0)
        self.slider_dehaze.setValue(0)
        self.slider_ond_est.setValue(0)
        self.slider_ond_bg.setValue(0)
        self.slider_contrast_starless.setValue(0)
        self.slider_denoise.setValue(0)
        self.slider_orton.setValue(0)
        self.slider_orton_radius.setValue(40)
        self.slider_orton_threshold.setValue(5)
        self.combo_orton_tint.setCurrentIndex(0)
        self.combo_orton_mode.setCurrentIndex(0)
        for w in widgets:
            w.blockSignals(False)

        self.clarity_starless_val = 0.0
        self.dehaze_starless_val = 0.0
        self.multiescala_estructura = 0.0
        self.multiescala_fondo = 0.0
        self.contrast_starless_val = 0.0
        self.denoise_strength = 0.0
        self.orton_strength = 0.0
        self.orton_radius = 40.0
        self.orton_threshold = 0.05
        self.orton_tint_type = "warm_gold"
        self.orton_mode = "soft_light"

        self.lbl_clarity.setText("0.00")
        self.lbl_dehaze.setText("0.00")
        self.lbl_ond_est.setText("0.00")
        self.lbl_ond_bg.setText("0%")
        self.lbl_contrast_starless.setText("0.00")
        self.lbl_denoise.setText("0%")
        if hasattr(self, 'lbl_orton'):
            self.lbl_orton.setText("0%")
        if hasattr(self, 'lbl_orton_radius'):
            self.lbl_orton_radius.setText("40 px")
        if hasattr(self, 'lbl_orton_threshold'):
            self.lbl_orton_threshold.setText("0.05")

        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(0, 0, 0, 0, 0, 0, 0, 40, 5, 0, 0)
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

            processed = self._apply_pipeline_on_image(full_base, precomputed_mask=full_mask, is_preview=False)

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
                    apply_mask_layers=False,
                    is_preview=False
                )

                # 2. Suelo desarrollado (con ajustes diferenciales de suelo/EV/sombras sin capas superpuestas)
                ground_dev = None
                if clean_m is not None and full_mask is not None:
                    zero_mask = np.zeros_like(full_mask)
                    ground_dev = self._apply_pipeline_on_image(
                        self.image_32bit, 
                        precomputed_mask=zero_mask, 
                        apply_mask_layers=False,
                        is_preview=False
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
                        apply_mask_layers=False,
                        is_preview=False
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

                    # Aplicar filtro difusor de estrellas si está activo
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
                src_cs = getattr(self, 'current_color_space', 'srgb')
                save_tiff_with_icc(p, out32, compression='zlib', photometric='rgb', source_color_space=src_cs)
                desc = "TIFF 32-bit float (ICC)"
            else:
                u16 = np.ascontiguousarray((np.clip(processed, 0.0, 1.0) * 65535.0), dtype=np.uint16)
                src_cs = getattr(self, 'current_color_space', 'srgb')
                save_tiff_with_icc(p, u16, compression='zlib', photometric='rgb', source_color_space=src_cs)
                desc = "TIFF 16-bit (ICC)"

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
            self.unsetCursor()
            
    def reset_all_parameters(self):
        """Devuelve todos los controles y variables del revelador a su estado neutro original."""
        self.lp_reduction = 0.0
        self.lp_center_x = 0.50
        self.lp_center_y = 0.90
        self.lp_width = 0.60
        self.lp_height = 0.45
        self.lp_tint_type = "sodium"
        self.lp_custom_tint = (1.0, 0.68, 0.22)
        self.lp_tint_strength = 0.50
        self.lp_mode = "attenuate"
        self.lp_legacy_1d = False
        self.orton_strength = 0.0
        self.orton_radius = 40.0
        self.orton_threshold = 0.05
        self.orton_tint_type = "warm_gold"
        self.orton_mode = "soft_light"
        self.star_diff_strength = 0.0
        self.star_diff_radius = 20.0
        self.star_diff_threshold = 0.45
        self.star_diff_color_boost = 0.50
        self.star_diff_mode = "screen"
        self.deconv_iterations = 0
        self.deconv_fwhm = 3.0
        self.deconv_model = "moffat"
        self.deconv_beta = 4.5
        self.deconv_aspect_ratio = 1.0
        self.deconv_angle_deg = 0.0
        self.deconv_deringing = 0.008
        self.deconv_damping = 0.25
        self.deconv_luma_only = True
        self.deconv_scope = "all"
        self._cached_psf = None
        self._cached_psf_key = ()
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
            self.slider_lp, self.slider_lp_cx, self.spin_lp_cx,
            self.slider_lp_cy, self.spin_lp_cy, self.slider_lp_w,
            self.spin_lp_w, self.slider_lp_h, self.spin_lp_h,
            self.slider_lp_tint_strength, self.combo_lp_tint,
            self.combo_lp_mode, self.chk_lp_legacy_1d,
            self.slider_stars, self.slider_contrast_starless,
            self.slider_clarity, self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_denoise, 
            self.slider_orton, self.slider_orton_radius, self.slider_orton_threshold,
            self.combo_orton_tint, self.combo_orton_mode,
            getattr(self, 'slider_star_diff', None),
            getattr(self, 'slider_star_diff_radius', None),
            getattr(self, 'slider_star_diff_threshold', None),
            getattr(self, 'slider_star_diff_color', None),
            getattr(self, 'combo_star_diff_mode', None),
            getattr(self, 'slider_deconv_iter', None),
            getattr(self, 'slider_deconv_fwhm', None),
            getattr(self, 'slider_deconv_deringing', None),
            getattr(self, 'slider_deconv_damping', None),
            getattr(self, 'slider_deconv_aspect', None),
            getattr(self, 'slider_deconv_angle', None),
            getattr(self, 'combo_deconv_model', None),
            getattr(self, 'combo_deconv_scope', None),
            getattr(self, 'chk_deconv_luma_only', None),
            self.slider_temp, self.slider_tint, self.slider_sat_sky,
            self.slider_sat_gnd, self.slider_vibrance, self.slider_contrast,
            self.combo_layer, self.spin_bp, self.slider_bp,
            self.spin_mtf, self.slider_mtf,
            self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp,
        ]
        for w in widgets_to_block:
            if w is not None:
                w.blockSignals(True)

        self.slider_lp.setValue(0)
        self.lbl_lp.setText("0%")
        if hasattr(self, 'slider_lp_cx'):
            self.slider_lp_cx.setValue(50)
            self.spin_lp_cx.setValue(0.50)
            self.slider_lp_cy.setValue(90)
            self.spin_lp_cy.setValue(0.90)
            self.slider_lp_w.setValue(60)
            self.spin_lp_w.setValue(0.60)
            self.slider_lp_h.setValue(45)
            self.spin_lp_h.setValue(0.45)
            self.combo_lp_tint.setCurrentIndex(0)
            self.slider_lp_tint_strength.setValue(50)
            self.lbl_lp_tint_strength.setText("50%")
            self.combo_lp_mode.setCurrentIndex(0)
            self.chk_lp_legacy_1d.setChecked(False)

        if hasattr(self, 'slider_orton'):
            self.slider_orton.setValue(0)
            self.lbl_orton.setText("0%")
        if hasattr(self, 'slider_orton_radius'):
            self.slider_orton_radius.setValue(40)
            self.lbl_orton_radius.setText("40 px")
        if hasattr(self, 'slider_orton_threshold'):
            self.slider_orton_threshold.setValue(5)
            self.lbl_orton_threshold.setText("0.05")
        if hasattr(self, 'combo_orton_tint'):
            self.combo_orton_tint.setCurrentIndex(0)
        if hasattr(self, 'combo_orton_mode'):
            self.combo_orton_mode.setCurrentIndex(0)

        if hasattr(self, 'slider_star_diff'):
            self.slider_star_diff.setValue(0)
            self.lbl_star_diff.setText("0%")
        if hasattr(self, 'slider_star_diff_radius'):
            self.slider_star_diff_radius.setValue(20)
            self.lbl_star_diff_radius.setText("20 px")
        if hasattr(self, 'slider_star_diff_threshold'):
            self.slider_star_diff_threshold.setValue(45)
            self.lbl_star_diff_threshold.setText("0.45")
        if hasattr(self, 'slider_star_diff_color'):
            self.slider_star_diff_color.setValue(50)
            self.lbl_star_diff_color.setText("50%")
        if hasattr(self, 'combo_star_diff_mode'):
            self.combo_star_diff_mode.setCurrentIndex(0)

        if hasattr(self, 'slider_deconv_iter'):
            self.slider_deconv_iter.setValue(0)
            self.lbl_deconv_iter.setText("0 (Off)")
        if hasattr(self, 'slider_deconv_fwhm'):
            self.slider_deconv_fwhm.setValue(30)
            self.lbl_deconv_fwhm.setText("3.0 px")
        if hasattr(self, 'slider_deconv_deringing'):
            self.slider_deconv_deringing.setValue(8)
            self.lbl_deconv_deringing.setText("0.008")
        if hasattr(self, 'slider_deconv_damping'):
            self.slider_deconv_damping.setValue(25)
            self.lbl_deconv_damping.setText("25%")
        if hasattr(self, 'slider_deconv_aspect'):
            self.slider_deconv_aspect.setValue(100)
            self.lbl_deconv_aspect.setText("1.00x")
        if hasattr(self, 'slider_deconv_angle'):
            self.slider_deconv_angle.setValue(0)
            self.lbl_deconv_angle.setText("0°")
        if hasattr(self, 'combo_deconv_model'):
            self.combo_deconv_model.setCurrentIndex(0)
        if hasattr(self, 'combo_deconv_scope'):
            self.combo_deconv_scope.setCurrentIndex(0)
        if hasattr(self, 'chk_deconv_luma_only'):
            self.chk_deconv_luma_only.setChecked(True)
        if hasattr(self, 'lbl_psf_status'):
            self.lbl_psf_status.setText("PSF: Moffat 2D (FWHM: 3.0 px | Aspecto: 1.00)")

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
            self.mask_layers_widget.set_stars_special_values(100, 0, 20, 45, 50, 0)
            self.mask_layers_widget.set_starless_tone_values(0, 0, 0, 0, 0, 0, 0, 40, 5, 0, 0)

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

        self._invalidate_pipeline_caches()

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
        ]
        for ctrl in controls:
            if ctrl is not None:
                ctrl.setEnabled(enabled)

    # -------------------------------------------------------------
    # GESTOR DE HISTORIAL DE DESHACER / REHACER (UNDO / REDO)
    # -------------------------------------------------------------
    def _connect_undo_triggers(self):
        """Conecta los eventos de finalización de interacción para registrar acciones en el historial."""
        heavy_sliders = [
            getattr(self, 'slider_clarity', None),
            getattr(self, 'slider_dehaze', None),
            getattr(self, 'slider_ond_est', None),
            getattr(self, 'slider_ond_bg', None),
            getattr(self, 'slider_denoise', None),
            getattr(self, 'slider_orton', None),
            getattr(self, 'slider_orton_radius', None),
            getattr(self, 'slider_orton_threshold', None),
            getattr(self, 'slider_star_diff', None),
            getattr(self, 'slider_star_diff_radius', None),
            getattr(self, 'slider_star_diff_threshold', None),
            getattr(self, 'slider_star_diff_color', None),
        ]
        for s in heavy_sliders:
            if s is not None:
                s.sliderPressed.connect(self._on_heavy_slider_pressed)
                s.sliderReleased.connect(self._on_slider_interaction_finished)

        # Deslizadores de deconvolución (requieren mantener iteraciones de PSF en su propio arrastre)
        deconv_sliders = [
            getattr(self, 'slider_deconv_iter', None),
            getattr(self, 'slider_deconv_fwhm', None),
            getattr(self, 'slider_deconv_deringing', None),
            getattr(self, 'slider_deconv_damping', None),
            getattr(self, 'slider_deconv_aspect', None),
            getattr(self, 'slider_deconv_angle', None),
        ]
        for s in deconv_sliders:
            if s is not None:
                s.sliderPressed.connect(self._on_deconv_slider_pressed)
                s.sliderReleased.connect(self._on_slider_interaction_finished)

        # Deslizadores de cálculo ligero (contraste, balance, tono, curvas, estrellas)
        # conservan la máxima fidelidad visual (Preview Proxy <= 1600px) en tiempo real
        light_sliders = [
            self.slider_contrast_starless,
            self.slider_stars,
            self.slider_temp, self.slider_tint, self.slider_sat_sky,
            self.slider_sat_gnd, self.slider_vibrance, self.slider_contrast,
            self.slider_bp, self.slider_mtf,
            self.slider_ghs_sp, self.slider_ghs_d, self.slider_ghs_b,
            self.slider_ghs_hp, self.slider_ghs_lp, self.slider_ghs_bp,
            self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp,
            getattr(self, 'slider_lp', None),
            getattr(self, 'slider_lp_cx', None),
            getattr(self, 'slider_lp_cy', None),
            getattr(self, 'slider_lp_w', None),
            getattr(self, 'slider_lp_h', None),
            getattr(self, 'slider_lp_tint_strength', None),
        ]
        for s in light_sliders:
            if s is not None:
                s.sliderPressed.connect(self._on_heavy_slider_pressed)
                s.sliderReleased.connect(self._on_slider_interaction_finished)

        if hasattr(self, 'mask_layers_widget'):
            mask_heavy_sliders = [
                getattr(self.mask_layers_widget, 'slider_sl_clarity', None),
                getattr(self.mask_layers_widget, 'slider_sl_dehaze', None),
                getattr(self.mask_layers_widget, 'slider_sl_ond_est', None),
                getattr(self.mask_layers_widget, 'slider_sl_ond_bg', None),
                getattr(self.mask_layers_widget, 'slider_sl_denoise', None),
                getattr(self.mask_layers_widget, 'slider_sl_orton_strength', None),
                getattr(self.mask_layers_widget, 'slider_sl_orton_radius', None),
                getattr(self.mask_layers_widget, 'slider_sl_orton_threshold', None),
                getattr(self.mask_layers_widget, 'slider_sl_star_diff_strength', None),
                getattr(self.mask_layers_widget, 'slider_sl_star_diff_radius', None),
                getattr(self.mask_layers_widget, 'slider_sl_star_diff_threshold', None),
                getattr(self.mask_layers_widget, 'slider_sl_star_diff_color', None),
            ]
            for s in mask_heavy_sliders:
                if s is not None:
                    s.sliderPressed.connect(self._on_heavy_slider_pressed)
                    s.sliderReleased.connect(self._on_slider_interaction_finished)

            mask_light_sliders = [
                getattr(self.mask_layers_widget, 'slider_sl_contrast', None),
                getattr(self.mask_layers_widget, 'slider_sl_ev', None),
                getattr(self.mask_layers_widget, 'slider_sl_sh', None),
                getattr(self.mask_layers_widget, 'slider_sl_bp', None),
                getattr(self.mask_layers_widget, 'slider_sl_temp', None),
                getattr(self.mask_layers_widget, 'slider_sl_tint', None),
                getattr(self.mask_layers_widget, 'slider_sl_sat', None),
                getattr(self.mask_layers_widget, 'slider_sl_vib', None),
            ]
            for s in mask_light_sliders:
                if s is not None:
                    s.sliderPressed.connect(self._on_heavy_slider_pressed)
                    s.sliderReleased.connect(self._on_slider_interaction_finished)

        self.spin_bp.editingFinished.connect(self._on_control_committed)
        self.spin_mtf.editingFinished.connect(self._on_control_committed)
        self.spin_ghs_sp.editingFinished.connect(self._on_control_committed)
        self.spin_ghs_d.editingFinished.connect(self._on_control_committed)
        self.spin_ghs_b.editingFinished.connect(self._on_control_committed)
        self.spin_ghs_hp.editingFinished.connect(self._on_control_committed)
        self.spin_ghs_lp.editingFinished.connect(self._on_control_committed)
        self.spin_ghs_bp.editingFinished.connect(self._on_control_committed)
        self.spin_lp_cx.editingFinished.connect(self._on_control_committed)
        self.spin_lp_cy.editingFinished.connect(self._on_control_committed)
        self.spin_lp_w.editingFinished.connect(self._on_control_committed)
        self.spin_lp_h.editingFinished.connect(self._on_control_committed)
        self.combo_lp_tint.activated.connect(lambda _: self._on_control_committed())
        self.combo_lp_mode.activated.connect(lambda _: self._on_control_committed())
        self.chk_lp_legacy_1d.toggled.connect(lambda _: self._on_control_committed())
        self.combo_orton_tint.activated.connect(lambda _: self._on_control_committed())
        self.combo_orton_mode.activated.connect(lambda _: self._on_control_committed())
        if hasattr(self, 'combo_star_diff_mode'):
            self.combo_star_diff_mode.activated.connect(lambda _: self._on_control_committed())
        if hasattr(self, 'combo_deconv_model'):
            self.combo_deconv_model.activated.connect(lambda _: self._on_control_committed())
        if hasattr(self, 'combo_deconv_scope'):
            self.combo_deconv_scope.activated.connect(lambda _: self._on_control_committed())
        if hasattr(self, 'chk_deconv_luma_only'):
            self.chk_deconv_luma_only.toggled.connect(lambda _: self._on_control_committed())
        self.spin_temp.editingFinished.connect(self._on_control_committed)
        self.spin_tint.editingFinished.connect(self._on_control_committed)
        self.combo_layer.activated.connect(lambda _: self._on_control_committed())

        if hasattr(self, 'curve_widget'):
            self.curve_widget.curveCommitted.connect(self._on_control_committed)

    def _on_heavy_slider_pressed(self):
        """Activa el modo de proxy borrador ultra-fluido (Draft 60 FPS) durante el arrastre continuo."""
        self._is_slider_pressed = True
        self._is_scrubbing_heavy_slider = True
        self._is_dragging_deconv = False

    def _on_deconv_slider_pressed(self):
        """Activa el modo de arrastre de deconvolución manteniendo fidelidad del filtro de PSF."""
        self._is_slider_pressed = True
        self._is_scrubbing_heavy_slider = True
        self._is_dragging_deconv = True

    def _on_slider_interaction_finished(self):
        """Finaliza inmediatamente el renderizado y los histogramas al soltar el ratón y consolida en historial."""
        if hasattr(self, '_refine_timer'):
            self._refine_timer.stop()
        self._is_slider_pressed = False
        self._is_scrubbing_heavy_slider = False
        self._is_dragging_deconv = False
        self.update_stretch_preview(immediate=True)
        self._commit_action()

    def _on_control_committed(self):
        """Finaliza inmediatamente el renderizado y consolida en historial tras editar spinboxes o combos."""
        if hasattr(self, '_refine_timer'):
            self._refine_timer.stop()
        self._is_slider_pressed = False
        self._is_scrubbing_heavy_slider = False
        self._is_dragging_deconv = False
        self.update_stretch_preview(immediate=True)
        self._commit_action()

    def _capture_state(self) -> dict:
        """Captura una instantánea paramétrica ligera (~1 KB) delegando en los paneles modulares."""
        state = {}
        if hasattr(self, 'panel_stretch'):
            state.update(self.panel_stretch.get_state())
        if hasattr(self, 'panel_lp'):
            state.update(self.panel_lp.get_state())
        if hasattr(self, 'panel_deconv'):
            state.update(self.panel_deconv.get_state())
        if hasattr(self, 'panel_color_wb'):
            state.update(self.panel_color_wb.get_state())
        if hasattr(self, 'panel_starnet'):
            state.update(self.panel_starnet.get_state())
        if hasattr(self, 'panel_ground_sky'):
            state.update(self.panel_ground_sky.get_state())
        state["curves"] = self.curve_widget.get_state() if hasattr(self, 'curve_widget') else None
        state["mask_layers"] = self.mask_layers_widget.get_state() if hasattr(self, 'mask_layers_widget') else None
        return state

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
                self.update_stretch_preview(immediate=True)
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
                self.update_stretch_preview(immediate=True)
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
                self.update_stretch_preview(immediate=True)
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
                self.update_stretch_preview(immediate=True)
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

        if hasattr(self, 'panel_color_wb'):
            self.panel_color_wb.set_state(state)
            self.temp_val = self.panel_color_wb.temp_val
            self.tint_val = self.panel_color_wb.tint_val
            self.sat_sky_val = self.panel_color_wb.sat_sky_val
            self.sat_gnd_val = self.panel_color_wb.sat_gnd_val
            self.vibrance_val = self.panel_color_wb.vibrance_val

        if hasattr(self, 'panel_starnet'):
            self.panel_starnet.set_state(state)
            self.view_layer_mode = self.panel_starnet.combo_layer.currentIndex()
            self.star_intensity = self.panel_starnet.slider_stars.value() / 100.0
            self.contrast_starless_val = self.panel_starnet.slider_contrast_starless.value() / 100.0
            self.clarity_starless_val = self.panel_starnet.slider_clarity.value() / 100.0
            self.dehaze_starless_val = self.panel_starnet.slider_dehaze.value() / 100.0
            self.multiescala_estructura = self.panel_starnet.slider_ond_est.value() / 100.0
            self.multiescala_fondo = self.panel_starnet.slider_ond_bg.value() / 100.0
            self.denoise_strength = self.panel_starnet.slider_denoise.value() / 100.0
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
                self.combo_orton_mode.currentIndex() if hasattr(self, 'combo_orton_mode') else 0
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

        self.update_stretch_preview(immediate=True)

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
        self.log_message(f"[PRESET] Preset aplicado: '{preset.name}' [{scope_str}]")

        if hasattr(self, 'sec_presets'):
            if preset and preset.name and "Neutro" not in preset.name:
                badge_title = preset.name.split("-")[0].strip() if "-" in preset.name else preset.name
                for em in ("⭐", "🌌", "✨", "🏞️", "🔭", "🏙️", "🌫️", "👤"):
                    badge_title = badge_title.replace(em, "")
                badge_title = badge_title.strip()
                self.sec_presets.set_status_badge(f"● {badge_title}", "#80d8ff")
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
            self.log_message(f"[PRESET] Nuevo preset guardado con éxito: '{dlg.created_preset.name}' ({dlg.created_preset.category})")

    def open_preset_manager_dialog(self):
        """Abre el diálogo de gestión, importación y exportación de presets."""
        current_params = self._capture_state()
        dlg = PresetManagerDialog(current_params, self)
        dlg.apply_preset_requested.connect(self.apply_preset)
        dlg.exec()
        curr_preset = self.combo_presets.currentData(Qt.UserRole)
        curr_id = curr_preset.id if isinstance(curr_preset, DeveloperPreset) else None
        self._populate_presets_combo(select_preset_id=curr_id)