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
    QCheckBox,
    QDialog
)
from PySide6.QtCore import Qt, QCoreApplication, QTimer, QEvent, QPoint
from PySide6.QtGui import QTextCursor, QShortcut, QKeySequence, QImage, QPixmap, QColor, QFont, QCursor
from core.log_hub import log_hub
from core.stacking import load_image_as_float32
from core.background_extraction import generate_sky_sample_points, extract_background_rbf, auto_extract_background_sky
from core.stretch import (
    calculate_mtf_params, 
    calculate_sky_neutral_wb,
    calculate_sky_mtf_params,
    calculate_smart_ground_lift,
    manual_stretch, 
    compute_ghs_lut,
    apply_ghs_stretch,
    estimate_ghs_params,
    apply_white_balance, 
    extract_background_polynomial,
    adjust_vibrance, 
    adjust_contrast, 
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
    MASK_TYPE_GROUND, MASK_TYPE_SKY, MASK_TYPE_IMAGE_RASTER,
    apply_pre_curve_adjustments, get_canvas_below_layer, compute_effective_layer_mask,
    apply_local_adjustments_to_image, clear_subcomposite_cache
)
from gui.widgets.curve_widget import CurveWidget
from gui.widgets.live_histogram import LiveRGBHistogramWidget
from gui.widgets.mask_layers_widget import MaskLayersWidget
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow, PrecisionSpinBox
from gui.icons import get_icon, get_svg_pixmap
from gui.canvas import MaskCanvas
from gui.widgets.viewer_toolbar import CanvasFooterBar
from gui.worker import StarNetWorker
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
from gui.developer.panels.panel_dust import DustPanel
from gui.developer.panels.panel_sharpening import SharpeningPanel
from gui.developer.developer_state import DeveloperState
from gui.developer.pipeline import ResolutionPyramid, apply_developer_pipeline
from gui.developer.async_worker import AsyncPipelineManager
from core.buffer_cache import PipelineBufferCache
from core.display_transfer import (
    compute_combined_stf_lut,
    apply_display_lut16
)


from gui.developer.history_presets import HistoryPresetsMixin
from gui.developer.io_project_manager import ProjectIOMixin
from gui.developer.canvas_tools import CanvasToolsMixin
from gui.developer.layer_sync_bridge import LayerSyncMixin


class DeveloperTab(QWidget, HistoryPresetsMixin, ProjectIOMixin, CanvasToolsMixin, LayerSyncMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_32bit = None       # Imagen nativa base (float32)
        self.image_starless = None    # Capa sin estrellas (float32)
        self.raw_linear_starless = None # Copia lineal previa al Auto-Stretch
        self.starless_is_stretched = False # Indicador de estirado no lineal sobre starless
        self.image_stars = None       # Capa de solo estrellas (float32)
        self.developer_state: DeveloperState = DeveloperState()

        # Buffers proxy para previsualización interactiva a 60 fps
        self.preview_proxy = None     
        self.proxy_starless = None
        self.proxy_stars = None
        self.proxy_mask = None        

        self.current_mask = None      
        self.active_filepath = None
        self.current_project_path: Optional[str] = None
        self._is_dirty: bool = False
        self._is_loading_project: bool = False
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
        self.texture_clarity_starless_val = 0.0
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
        self.gnd_bp_val = 0.0
        self.gnd_antihalo_val = 0.0
        self.gnd_edge_shift_val = 0        # -0.050 a +0.100
        self.gnd_antihalo_val = 0.0  # 0.0 a 1.0 (0% a 100%)
        self.gnd_edge_shift_val = 0  # -15 a +15 px
        
        self.star_intensity = 1.0    # 1.0 = 100%, 0.0 = Starless puro
        self.view_layer_mode = 0     # 0: Compuesta, 1: Solo Fondo, 2: Solo Estrellas
        self.contrast_starless_val = 0.0
        self.denoise_strength = 0.0  # <-- CORREGIDO: Inicialización obligatoria
        self.denoise_color_strength = 0.50
        self.denoise_masking = 0.50
        self._cache_starless_ai = None
        self.starless_ai_denoise_enabled: bool = False
        self.gnd_denoise_val: float = 0.0
        self.gnd_ai_denoise_enabled: bool = False
        self._cache_ground_ai: Optional[np.ndarray] = None
        self.ai_denoise_worker = None
        self.ground_ai_denoise_worker = None
        self.nafnet_download_worker = None

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
        self._is_applying_preset: bool = False
        self._is_previewing_preset: bool = False
        self._undo_debounce_timer = QTimer(self)
        self._undo_debounce_timer.setSingleShot(True)
        self._undo_debounce_timer.setInterval(500)
        self._undo_debounce_timer.timeout.connect(self._commit_action)

        # Temporizadores de renderizado reactivo y coalescencia de eventos (60 FPS)
        self._preview_throttle_timer = QTimer(self)
        self._preview_throttle_timer.setSingleShot(True)
        self._preview_throttle_timer.setInterval(16)
        self._preview_throttle_timer.timeout.connect(self._do_update_stretch_preview)

        self._pan_refine_timer = QTimer(self)
        self._pan_refine_timer.setSingleShot(True)
        self._pan_refine_timer.setInterval(180)
        self._pan_refine_timer.timeout.connect(self._refine_pan_crop)

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

        # Cachés por hitos del pipeline (Milestone Caching con LRU y presupuesto de RAM)
        self.buffer_cache = PipelineBufferCache()
        self._cache_ms_pre_lp_key: Optional[tuple] = None
        self._cache_ms_pre_lp_img: Optional[np.ndarray] = None
        self._cache_ms_linear_key: Optional[tuple] = None
        self._cache_ms_linear_img: Optional[np.ndarray] = None
        self._cache_ms_linear_u16: Optional[np.ndarray] = None
        self._cache_ms_stars_u16: Optional[np.ndarray] = None
        self._scrub_display_buf: Optional[np.ndarray] = None
        self._scrub_before_buf: Optional[np.ndarray] = None
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
        self._is_scrubbing_lut_slider: bool = False
        self._is_scrubbing_layer_slider: bool = False
        self._is_slider_pressed: bool = False
        self._is_dragging_deconv: bool = False
        self._is_dragging_gradient: bool = False
        self._is_dragging_antihalo: bool = False
        self.async_pipeline = AsyncPipelineManager(self)
        self.async_pipeline.preview_ready.connect(self._on_async_preview_ready)
        self.async_pipeline.pipeline_failed.connect(self._on_async_pipeline_failed)

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
        self.canvas.pan_started.connect(self.on_canvas_pan_started)
        self.canvas.pan_moved.connect(self.on_canvas_pan_moved)
        self.canvas.pan_finished.connect(self.on_canvas_pan_finished)
        self.canvas.files_dropped.connect(self._on_canvas_files_dropped)

        # 2. Panel interno de controles con Acordeón Plegable Inteligente
        self.left_container = QWidget()
        left_container_layout = QVBoxLayout(self.left_container)
        left_container_layout.setContentsMargins(4, 4, 4, 4)
        left_container_layout.setSpacing(6)
        self.left_container.setMinimumWidth(280)
        self.left_container.setMaximumWidth(1200)

        # 1. Gestión de Proyecto (.devloom) fijada arriba
        self.proj_container = QWidget()
        self.proj_container.setStyleSheet("background: #181c28; border: 1px solid #2a3144; border-radius: 4px;")
        proj_vbox = QVBoxLayout(self.proj_container)
        proj_vbox.setContentsMargins(6, 6, 6, 6)
        proj_vbox.setSpacing(5)

        self.lbl_project_title = QLabel("Proyecto: Sin guardar")
        self.lbl_project_title.setStyleSheet("color: #80cbc4; font-size: 11px; font-weight: bold; border: none; background: transparent;")
        proj_vbox.addWidget(self.lbl_project_title)

        proj_btn_row = QHBoxLayout()
        proj_btn_row.setContentsMargins(0, 0, 0, 0)
        proj_btn_row.setSpacing(4)

        self.btn_open_proj = QPushButton("Abrir")
        self.btn_open_proj.setIcon(get_icon("folder", size=13, color="#80d8ff"))
        self.btn_open_proj.setToolTip("Abrir proyecto de revelado .devloom (Atajo: Ctrl+O)")
        self.btn_open_proj.setCursor(Qt.PointingHandCursor)
        self.btn_open_proj.setStyleSheet("font-weight: bold; color: #80d8ff; background: #22293d; border: 1px solid #333d59; border-radius: 3px; padding: 4px 6px;")
        self.btn_open_proj.clicked.connect(self.open_project_dialog)
        proj_btn_row.addWidget(self.btn_open_proj)

        self.btn_save_proj = QPushButton("Guardar")
        self.btn_save_proj.setIcon(get_icon("save", size=13, color="#a5d6a7"))
        self.btn_save_proj.setToolTip("Guardar sesión de revelado actual (Atajo: Ctrl+S)")
        self.btn_save_proj.setCursor(Qt.PointingHandCursor)
        self.btn_save_proj.setStyleSheet("font-weight: bold; color: #a5d6a7; background: #22293d; border: 1px solid #333d59; border-radius: 3px; padding: 4px 6px;")
        self.btn_save_proj.clicked.connect(self.save_project)
        proj_btn_row.addWidget(self.btn_save_proj)

        self.btn_save_as_proj = QPushButton("Guardar Como...")
        self.btn_save_as_proj.setIcon(get_icon("copy", size=13, color="#cfd8dc"))
        self.btn_save_as_proj.setToolTip("Guardar sesión de revelado en un archivo nuevo (Atajo: Ctrl+Shift+S)")
        self.btn_save_as_proj.setCursor(Qt.PointingHandCursor)
        self.btn_save_as_proj.setStyleSheet("color: #cfd8dc; background: #22293d; border: 1px solid #333d59; border-radius: 3px; padding: 4px 6px;")
        self.btn_save_as_proj.clicked.connect(self.save_project_as_dialog)
        proj_btn_row.addWidget(self.btn_save_as_proj)

        proj_vbox.addLayout(proj_btn_row)
        left_container_layout.addWidget(self.proj_container, 0)

        # Carga externa (Siempre fijada arriba)
        self.btn_open_img = QPushButton("Abrir Imagen (TIFF / FITS / RAW)...")
        self.btn_open_img.setIcon(get_icon("folder", color="#38bdf8", size=16))
        self.btn_open_img.setStyleSheet("font-weight: bold; color: #80d8ff; padding: 7px; background: #1c2030; border: 1px solid #2f3549; border-radius: 4px;")
        self.btn_open_img.setCursor(Qt.PointingHandCursor)
        self.btn_open_img.clicked.connect(self.open_image_file)
        left_container_layout.addWidget(self.btn_open_img, 0)

        # Botón de Arranque Rápido: 1-Click Start Astro-Reveal
        self.btn_start_reveal = QPushButton("✨ 1-Click Start Astro-Reveal")
        self.btn_start_reveal.setIcon(get_icon("sparkles", color="#ffd54f", size=16))
        self.btn_start_reveal.setToolTip("Arranque en 1 clic para paisaje: detecta horizonte por IA, neutraliza el fondo celestial, auto-estira el cielo y evalúa el suelo con Smart Ground Lift.")
        self.btn_start_reveal.setCursor(Qt.PointingHandCursor)
        self.btn_start_reveal.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 12px;
                color: #fff9c4;
                padding: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a237e, stop:1 #283593);
                border: 1px solid #5c6bc0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #283593, stop:1 #3949ab);
                border: 1px solid #9fa8da;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #1a237e;
                border: 1px solid #3949ab;
            }
        """)
        self.btn_start_reveal.clicked.connect(self.start_astro_reveal)
        left_container_layout.addWidget(self.btn_start_reveal, 0)

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
        self.combo_presets = getattr(self.panel_presets, 'combo_presets', None)
        self.lbl_preset_desc = getattr(self.panel_presets, 'lbl_preset_desc', None)
        self.combo_preset_scope = getattr(self.panel_presets, 'combo_preset_scope', None)
        self.btn_apply_preset = getattr(self.panel_presets, 'btn_apply_preset', None)
        self.btn_save_preset = self.panel_presets.btn_save_preset
        self.btn_manage_presets = self.panel_presets.btn_manage_presets

        if hasattr(self.panel_presets, 'preset_changed'):
            self.panel_presets.preset_changed.connect(self._on_preset_combo_changed)
        if hasattr(self.panel_presets, 'apply_requested'):
            self.panel_presets.apply_requested.connect(self.apply_selected_preset)
        self.panel_presets.save_requested.connect(self.open_save_preset_dialog)
        self.panel_presets.manage_requested.connect(self.open_preset_manager_dialog)

        left_layout.addWidget(self.panel_presets)

        # --- PASO 1: Paisaje y Máscara de Suelo (Módulo gui.developer.panels.GroundSkyPanel) ---
        self.panel_ground_sky = GroundSkyPanel()
        self.sec_gnd_tone = self.panel_ground_sky.sec_gnd_tone
        self.lbl_gnd_mask_status = self.panel_ground_sky.lbl_gnd_mask_status
        self.btn_edit_ground_mask = self.panel_ground_sky.btn_edit_ground_mask
        self.card_gnd_info = self.panel_ground_sky.card_gnd_info
        self.lbl_gnd_info = self.panel_ground_sky.lbl_gnd_info
        self._hidden_gnd_container = self.panel_ground_sky._hidden_gnd_container
        self.lbl_gnd_ev = self.panel_ground_sky.lbl_gnd_ev
        self.slider_gnd_ev = self.panel_ground_sky.slider_gnd_ev
        self.lbl_gnd_sh = self.panel_ground_sky.lbl_gnd_sh
        self.slider_gnd_sh = self.panel_ground_sky.slider_gnd_sh
        self.lbl_gnd_bp = self.panel_ground_sky.lbl_gnd_bp
        self.slider_gnd_bp = self.panel_ground_sky.slider_gnd_bp

        self.panel_ground_sky.edit_mask_requested.connect(self._open_mask_tool_from_section)
        self.panel_ground_sky.ai_mask_requested.connect(self._detect_mask_ai_from_toolbar)
        self.panel_ground_sky.overlay_toggled.connect(self._on_top_mask_overlay_toggled)
        self.panel_ground_sky.delete_mask_requested.connect(self._delete_mask)
        self.panel_ground_sky.reset_requested.connect(self.reset_gnd_tone)
        self.slider_gnd_ev.valueChanged.connect(self.on_gnd_ev_changed)
        self.slider_gnd_sh.valueChanged.connect(self.on_gnd_shadows_changed)
        self.slider_gnd_bp.valueChanged.connect(self.on_gnd_bp_changed)

        left_layout.addWidget(self.panel_ground_sky)

        # --- PASO 2: Balance de Blancos y Calibración (Módulo gui.developer.panels.ColorWbPanel) ---
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
        self.panel_color_wb.pcc_requested.connect(self.apply_pcc_calibration)
        self.panel_color_wb.action_committed.connect(self._commit_action)

        left_layout.addWidget(self.panel_color_wb)

        # --- SECCIÓN: Corrección de Manchas y Polvo del Sensor (Módulo gui.developer.panels.DustPanel) ---
        self.panel_dust = DustPanel()
        self.panel_dust.auto_detect_requested.connect(self._on_auto_detect_dust)
        self.panel_dust.visualize_toggled.connect(self._on_visualize_dust_toggled)
        self.panel_dust.spot_mode_toggled.connect(self._on_dust_spot_mode_toggled)
        self.panel_dust.clear_spots_requested.connect(self._on_clear_dust_spots)
        self.panel_dust.parameter_changed.connect(self._on_dust_parameter_changed)
        self.panel_dust.action_committed.connect(self._commit_action)

        left_layout.addWidget(self.panel_dust)

        # --- SECCIÓN 2: Extracción de Fondo y Polución Lumínica (Módulo gui.developer.panels.LightPollutionPanel) ---
        self.panel_lp = LightPollutionPanel()
        self.sec_gradients = self.panel_lp.sec_bg
        self.sec_bg = self.panel_lp.sec_bg
        self.combo_poly_degree = self.panel_lp.combo_poly_degree
        self.combo_bg_engine = self.panel_lp.combo_bg_engine
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
        self.panel_lp.auto_abe_requested.connect(self.on_auto_abe_clicked)
        self.panel_lp.auto_detect_lp_requested.connect(self.auto_detect_lp_center)
        self.panel_lp.center_pipette_toggled.connect(self._on_toggle_lp_center_pipette)
        self.panel_lp.color_pipette_toggled.connect(self._on_toggle_lp_color_pipette)
        self.panel_lp.sample_tool_toggled.connect(self._on_toggle_bg_sample_tool)
        self.panel_lp.auto_grid_requested.connect(self.auto_generate_bg_samples)
        self.panel_lp.clear_points_requested.connect(self.clear_bg_samples)
        self.panel_lp.sample_box_size_changed.connect(self.canvas.set_bg_sample_box_size)
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
        left_layout.addWidget(self.sec_curves)

        # --- SECCIÓN 5: StarNet++ AI — Desacoplo de Estrellas (Módulo gui.developer.panels.StarNetPanel) ---
        self.panel_starnet = StarNetPanel(self, resettable_label_factory=self._make_resettable_label)
        self.sec_starnet = self.panel_starnet.sec_starnet
        self.spin_stride = self.panel_starnet.spin_stride
        self.chk_upsample = getattr(self.panel_starnet, 'chk_upsample', None)
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
        self.slider_texture_clarity = self.panel_starnet.slider_texture_clarity
        self.lbl_texture_clarity = self.panel_starnet.lbl_texture_clarity
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
        self.slider_denoise_color = self.panel_starnet.slider_denoise_color
        self.lbl_denoise_color = self.panel_starnet.lbl_denoise_color
        self.slider_denoise_mask = self.panel_starnet.slider_denoise_mask
        self.lbl_denoise_mask = self.panel_starnet.lbl_denoise_mask
        self.slider_banding = self.panel_starnet.slider_banding
        self.lbl_banding = self.panel_starnet.lbl_banding
        self.combo_banding_dir = self.panel_starnet.combo_banding_dir
        self.btn_denoise_expand = self.panel_starnet.btn_denoise_expand
        self.w_denoise_advanced = self.panel_starnet.w_denoise_advanced

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

        # Conectar eventos de StarNet y Efectos delegados al panel
        self.panel_starnet.run_starnet_requested.connect(self.run_starnet)
        self.panel_starnet.jump_to_masks_requested.connect(lambda: self._jump_to_mask_layers(MASK_TYPE_IMAGE_STARLESS))
        self.panel_starnet.auto_stretch_starless_requested.connect(self.on_auto_stretch_starless_requested)
        self.panel_starnet.run_ai_denoise_requested.connect(self.on_run_ai_denoise_requested)
        self.combo_layer.currentIndexChanged.connect(self.on_layer_mode_changed)
        self.slider_stars.valueChanged.connect(self.on_stars_slider_changed)
        self.slider_clarity.valueChanged.connect(self.on_clarity_changed)
        self.slider_texture_clarity.valueChanged.connect(self.on_texture_clarity_changed)
        self.slider_dehaze.valueChanged.connect(self.on_dehaze_changed)
        self.slider_ond_est.valueChanged.connect(self.on_ond_estructura_changed)
        self.slider_ond_bg.valueChanged.connect(self.on_ond_fondo_changed)
        self.slider_contrast_starless.valueChanged.connect(self.on_contrast_starless_changed)
        self.slider_denoise.valueChanged.connect(self.on_denoise_changed)
        self.slider_denoise_color.valueChanged.connect(self.on_denoise_color_changed)
        self.slider_denoise_mask.valueChanged.connect(self.on_denoise_mask_changed)
        self.slider_banding.valueChanged.connect(self.on_banding_changed)
        self.combo_banding_dir.currentIndexChanged.connect(lambda _: self.update_stretch_preview())

        # Los controles de Orton y Difusión ya se gestionan dentro de panel_effects
        self.panel_effects.parameter_changed.connect(self._on_effects_parameter_changed)
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
        self.panel_sharpening = SharpeningPanel(self)
        self.panel_sharpening.parameter_changed.connect(self.update_stretch_preview)
        self.panel_sharpening.row_ond_est.valueChanged.connect(self.on_ond_estructura_changed)
        self.panel_sharpening.row_ond_bg.valueChanged.connect(self.on_ond_fondo_changed)
        left_layout.addWidget(self.panel_sharpening)

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
        self.mask_layers_widget.ground_denoise_changed.connect(self.on_gnd_denoise_changed)
        self.mask_layers_widget.ground_antihalo_changed.connect(self.on_gnd_antihalo_changed)
        self.mask_layers_widget.ground_edge_shift_changed.connect(self.on_gnd_edge_shift_changed)
        self.mask_layers_widget.ground_ai_denoise_requested.connect(self.on_run_ground_ai_denoise_requested)
        self.mask_layers_widget.raster_layer_import_requested.connect(self._on_import_raster_layer)
        self.mask_layers_widget.master_mask_updated.connect(self._on_master_mask_updated_from_raster)
        self.mask_layers_widget.layer_move_tool_toggled.connect(self._on_layer_move_tool_toggled)
        self.canvas.layer_move_offset_dragged.connect(self._on_canvas_layer_offset_dragged)
        self.canvas.layer_move_scale_wheel.connect(self._on_canvas_layer_scale_wheel)
        self.canvas.layer_move_finished.connect(self._commit_action)
        self.mask_layers_widget.gradient_tool_toggled.connect(self._on_layer_gradient_toggled)
        self.canvas.gradient_drag_started.connect(self._on_canvas_gradient_drag_started)
        self.canvas.gradient_changed.connect(self._on_canvas_gradient_changed)
        self.canvas.gradient_drag_finished.connect(self._on_canvas_gradient_drag_finished)
        if hasattr(self, 'panel_ground_sky') and self.panel_ground_sky is not None:
            self.panel_ground_sky.denoise_changed.connect(self.on_gnd_denoise_changed)
            self.panel_ground_sky.antihalo_changed.connect(self.on_gnd_antihalo_changed)
            self.panel_ground_sky.edge_shift_changed.connect(self.on_gnd_edge_shift_changed)
        self.mask_layers_widget.ground_reset_requested.connect(self.reset_gnd_tone)
        self.mask_layers_widget.ground_slider_released.connect(self._commit_action)
        self.mask_layers_widget.action_committed.connect(self._commit_action)
        self.mask_layers_widget.set_ground_tone_values(
            self.slider_gnd_ev.value(),
            self.slider_gnd_sh.value(),
            self.slider_gnd_bp.value(),
            int(round(getattr(self, 'gnd_denoise_val', 0.0) * 100.0)),
            int(getattr(self, 'gnd_antihalo_val', 0) * 100),
            int(getattr(self, 'gnd_edge_shift_val', 0))
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
        self.mask_layers_widget.starless_texture_clarity_changed.connect(self._on_mask_widget_sl_texture_clarity)
        self.mask_layers_widget.starless_dehaze_changed.connect(self._on_mask_widget_sl_dehaze)
        self.mask_layers_widget.starless_ond_est_changed.connect(self._on_mask_widget_sl_ond_est)
        self.mask_layers_widget.starless_ond_bg_changed.connect(self._on_mask_widget_sl_ond_bg)
        self.mask_layers_widget.starless_contrast_changed.connect(self._on_mask_widget_sl_contrast)
        self.mask_layers_widget.starless_denoise_changed.connect(self._on_mask_widget_sl_denoise)
        self.mask_layers_widget.starless_denoise_color_changed.connect(self._on_mask_widget_sl_denoise_color)
        self.mask_layers_widget.starless_denoise_mask_changed.connect(self._on_mask_widget_sl_denoise_mask)
        self.mask_layers_widget.starless_orton_strength_changed.connect(self._on_mask_widget_sl_orton_strength)
        self.mask_layers_widget.starless_orton_radius_changed.connect(self._on_mask_widget_sl_orton_radius)
        self.mask_layers_widget.starless_orton_threshold_changed.connect(self._on_mask_widget_sl_orton_threshold)
        self.mask_layers_widget.starless_orton_tint_changed.connect(self._on_mask_widget_sl_orton_tint)
        self.mask_layers_widget.starless_orton_mode_changed.connect(self._on_mask_widget_sl_orton_mode)
        self.mask_layers_widget.starless_reset_requested.connect(self.reset_starless_adjustments)
        self.mask_layers_widget.auto_stretch_starless_requested.connect(self.on_auto_stretch_starless_requested)
        self.mask_layers_widget.starless_ai_denoise_requested.connect(self.on_run_ai_denoise_requested)

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
            self.combo_orton_mode.currentIndex(),
            denoise_color=self.slider_denoise_color.value() if hasattr(self, 'slider_denoise_color') else 50,
            denoise_mask=self.slider_denoise_mask.value() if hasattr(self, 'slider_denoise_mask') else 50,
            texture_clarity=self.slider_texture_clarity.value()
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

        # Botón Exportar siempre fijado al pie del panel izquierdo (soporta TIFF, PSD multicapa 16-bit y JPEG)
        self.btn_export = QPushButton("EXPORTAR IMAGEN REVELADA...")
        self.btn_export.setIcon(get_icon("save", color="#ffffff", size=16))
        self.btn_export.setFixedHeight(40)
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_export.setToolTip(
            "Exporta la imagen revelada a disco.\n"
            "Formatos soportados según filtro o extensión:\n"
            "• TIFF (*.tif, *.tiff): 16-bit o 32-bit float con perfil ICC incrustado\n"
            "• Photoshop (*.psd): 16-bit nativo por capas y máscaras editables\n"
            "• JPEG (*.jpg, *.jpeg): 8-bit comprimido de alta calidad"
        )
        self.btn_export.clicked.connect(self.export_image)
        left_container_layout.addWidget(self.btn_export, 0)

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
        self.canvas.bg_samples_changed.connect(self._on_canvas_bg_samples_changed)
        self.canvas.bg_sample_mode_toggled.connect(self.panel_lp.set_sample_tool_active)
        self.canvas.dust_spots_changed.connect(self._on_canvas_dust_spots_changed)
        self.canvas.dust_spot_mode_toggled.connect(self.panel_dust.btn_manual_spot.setChecked)

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

        lbl_right_title = QLabel("HISTOGRAMA Y CAPAS")
        lbl_right_title.setStyleSheet("font-weight: 700; font-size: 11px; color: #80d8ff; letter-spacing: 0.6px;")
        right_header_layout.addWidget(lbl_right_title)
        right_header_layout.addStretch()
        right_container_layout.addWidget(right_header, 0)

        # Widget Histograma RGB en Vivo (Fijo en la parte superior, fuera del scroll)
        self.live_histogram = LiveRGBHistogramWidget(self)
        self.live_histogram.clippingChanged.connect(self._on_live_histogram_clipping_changed)

        self.sec_histogram = CollapsibleSection("Histograma RGB en Vivo", is_expanded=True, icon_name="activity")
        self.sec_histogram.set_content_widget(self.live_histogram)
        self.sec_histogram.toggled.connect(lambda exp: self._refresh_histograms_now() if exp else None)
        right_container_layout.addWidget(self.sec_histogram, 0)

        # Scroll área para la columna derecha (solo para capas y ajustes locales)
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

        # Atajos globales de Gestión de Proyecto (.devloom)
        self.shortcut_save_proj = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save_proj.setContext(Qt.WindowShortcut)
        self.shortcut_save_proj.activated.connect(self.save_project)

        self.shortcut_save_as_proj = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        self.shortcut_save_as_proj.setContext(Qt.WindowShortcut)
        self.shortcut_save_as_proj.activated.connect(self.save_project_as_dialog)

        self.shortcut_open_proj = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open_proj.setContext(Qt.WindowShortcut)
        self.shortcut_open_proj.activated.connect(self.open_project_dialog)

        # Atajo global 'Ctrl+I' para Invertir Máscara de Capa activa
        self.shortcut_invert_mask = QShortcut(QKeySequence("Ctrl+I"), self)
        self.shortcut_invert_mask.setContext(Qt.WindowShortcut)
        self.shortcut_invert_mask.activated.connect(self._on_shortcut_invert_mask)

        layout.addWidget(splitter)

    def retranslate_ui(self) -> None:
        """Actualiza todos los textos visibles del Revelador según el idioma activo."""
        if hasattr(self, 'btn_open_img'):
            self.btn_open_img.setText(tr("developer.btn_open", "Abrir Imagen (TIFF / FITS / RAW)..."))
        if hasattr(self, 'btn_export'):
            self.btn_export.setText(tr("developer.btn_export", "EXPORTAR IMAGEN REVELADA..."))


        # Secciones del acordeón
        if hasattr(self, 'sec_presets'):
            self.sec_presets.set_title(tr("developer.sections.presets", "Presets y Estilos de Revelado"))
        if hasattr(self, 'sec_gnd_tone'):
            self.sec_gnd_tone.set_title(tr("developer.sections.ground_tone", "1. Paisaje y Máscara de Suelo (Horizonte IA)"))
        if hasattr(self, 'sec_wb'):
            self.sec_wb.set_title(tr("developer.sections.wb", "2. Balance de Blancos y Calibración"))
        if hasattr(self, 'sec_color'):
            self.sec_color.set_title(tr("developer.sections.color", "Color y Saturación"))
        if hasattr(self, 'sec_gradients'):
            self.sec_gradients.set_title(tr("developer.sections.gradients", "3. Gradientes y Contaminación Lumínica"))
        if hasattr(self, 'sec_stretch'):
            self.sec_stretch.set_title(tr("developer.sections.stretch", "4. Estirado Tonal y Contraste"))
        if hasattr(self, 'sec_curves'):
            self.sec_curves.set_title(tr("developer.sections.curves", "Curva Tonal Interactiva (RGB / Saturación)"))
        if hasattr(self, 'sec_starnet'):
            self.sec_starnet.set_title(tr("developer.sections.starnet", "5. Desacoplo Estelar y Fondo"))
        if hasattr(self, 'sec_deconv'):
            self.sec_deconv.set_title(tr("developer.sections.deconv", "[Avanzado] Deconvolución Óptica y Enfoque (R-L)"))
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
            self.btn_undo.setText(tr("developer.toolbar.undo", ""))
        if hasattr(self, 'btn_redo'):
            self.btn_redo.setText(tr("developer.toolbar.redo", ""))
        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.setText(tr("developer.toolbar.before_after", "Comparar"))
        if hasattr(self, 'btn_set_before'):
            self.btn_set_before.setText(tr("developer.toolbar.set_as_before", "Fijar ref."))
        if hasattr(self, 'btn_toggle_clipping'):
            self.btn_toggle_clipping.setText(tr("developer.toolbar.clipping", "Aviso recorte"))
        if hasattr(self, 'btn_toggle_zoom'):
            self.btn_toggle_zoom.setText(tr("developer.toolbar.zoom_100", "1:1"))
        if hasattr(self, 'btn_magnifier'):
            self.btn_magnifier.setText(tr("developer.toolbar.loupe", ""))
        if hasattr(self, 'btn_crop'):
            self.btn_crop.setText(tr("developer.toolbar.crop", ""))
        if hasattr(self, 'btn_mask_tool'):
            self.btn_mask_tool.setText(tr("developer.toolbar.ground_mask", "Máscara"))
        if hasattr(self, 'btn_toggle_left_panel'):
            self.btn_toggle_left_panel.setToolTip(tr("developer.toolbar.toggle_left", "Ocultar / Mostrar panel lateral izquierdo de ajustes (Atajo: F7 o [)"))
        if hasattr(self, 'btn_toggle_right_panel'):
            self.btn_toggle_right_panel.setToolTip(tr("developer.toolbar.toggle_right", "Ocultar / Mostrar panel lateral derecho de capas e histograma (Atajo: F8 o ])"))
        if hasattr(self, '_update_mask_status_ui'):
            self._update_mask_status_ui()

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
        log_hub.log("Revelador", text)

    def _invalidate_pipeline_caches(self):
        """Invalida todas las cachés intermedias del pipeline de procesado."""
        clear_subcomposite_cache()
        if hasattr(self, 'buffer_cache'):
            self.buffer_cache.clear()
        self._cache_starless_key = None
        self._cache_starless_linear = None
        self._cache_starless_pre_key = None
        self._cache_ms_pre_lp_key = None
        self._cache_ms_pre_lp_img = None
        self._cache_ms_linear_key = None
        self._cache_ms_linear_img = None
        self._cache_ms_linear_u16 = None
        self._cache_ms_stars_u16 = None
        self._scrub_display_buf = None
        self._scrub_before_buf = None
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
        self._cached_base_stretched = None
        self._cached_stars_stretched = None
        self._cached_sky_mask = None
        self._cached_rendered_image = None
        self._last_rendered_preview = None
        self._fit_view_pixmap = None
        self._fit_view_orig_rgb = None
        self._fit_view_mask_overlay = None

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

        ph, pw = self.preview_proxy.shape[:2]
        dh, dw = self.draft_proxy.shape[:2]

        if self.image_starless is not None:
            self.pyramid_starless = ResolutionPyramid(self.image_starless, max_preview_dim=1600, max_levels=4)
            p_sl = self.pyramid_starless.preview
            d_sl = self.pyramid_starless.draft
            if p_sl.shape[:2] != (ph, pw):
                p_sl = cv2.resize(p_sl, (pw, ph), interpolation=cv2.INTER_AREA)
            if d_sl.shape[:2] != (dh, dw):
                d_sl = cv2.resize(d_sl, (dw, dh), interpolation=cv2.INTER_AREA)
            self.proxy_starless = np.ascontiguousarray(p_sl, dtype=np.float32)
            self.draft_starless = np.ascontiguousarray(d_sl, dtype=np.float32)

            self.pyramid_stars = ResolutionPyramid(self.image_stars, max_preview_dim=1600, max_levels=4)
            p_st = self.pyramid_stars.preview
            d_st = self.pyramid_stars.draft
            if p_st.shape[:2] != (ph, pw):
                p_st = cv2.resize(p_st, (pw, ph), interpolation=cv2.INTER_AREA)
            if d_st.shape[:2] != (dh, dw):
                d_st = cv2.resize(d_st, (dw, dh), interpolation=cv2.INTER_AREA)
            self.proxy_stars = np.ascontiguousarray(p_st, dtype=np.float32)
            self.draft_stars = np.ascontiguousarray(d_st, dtype=np.float32)
        else:
            self.pyramid_starless = None
            self.proxy_starless = None
            self.draft_starless = None
            self.pyramid_stars = None
            self.proxy_stars = None
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

    def _process_starless_base(
        self, 
        starless: Optional[np.ndarray], 
        mask_3d: Optional[np.ndarray] = None, 
        is_full_res: bool = True,
        crop_rect: Optional[tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """Aplica las mejoras lineales a la capa StarNet (Denoise, Claridad, Dehaze, Ondículas)."""
        if starless is None:
            return None

        if crop_rect is None and getattr(self, 'is_zoomed_100', False):
            crop_rect = getattr(self, '_zoom_crop_rect', None)

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
            id(getattr(self, '_cache_starless_ai', None)),
            is_full_res,
            is_draft,
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'denoise_color_strength', 0.5)), 4),
            round(float(getattr(self, 'denoise_masking', 0.5)), 4),
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
            id(getattr(self, '_cache_starless_ai', None)),
            is_full_res,
            is_draft,
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'denoise_color_strength', 0.5)), 4),
            round(float(getattr(self, 'denoise_masking', 0.5)), 4),
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

            # 1. Reducción de Ruido Multietapa (Luminancia + Croma + Máscara o NAFNet IA)
            if self.denoise_strength > 1e-4:
                cached_ai = getattr(self, '_cache_starless_ai', None)
                use_ai = getattr(self, 'starless_ai_denoise_enabled', False) and (cached_ai is not None)
                if use_ai:
                    h_f, w_f = self.image_32bit.shape[:2] if self.image_32bit is not None else cached_ai.shape[:2]
                    h_c, w_c = cached_ai.shape[:2]
                    ratio_c = w_c / float(h_c) if h_c > 0 else 1.0
                    ratio_f = w_f / float(h_f) if h_f > 0 else 1.0
                    if abs(ratio_c - ratio_f) > 0.05 and crop_rect is None:
                        self._cache_starless_ai = None
                        use_ai = False
                if use_ai:
                    if crop_rect is not None:
                        x1, y1, x2, y2 = crop_rect
                        sx = w_c / float(w_f)
                        sy = h_c / float(h_f)
                        sx1 = max(0, min(w_c, int(round(x1 * sx))))
                        sy1 = max(0, min(h_c, int(round(y1 * sy))))
                        sx2 = max(0, min(w_c, int(round(x2 * sx))))
                        sy2 = max(0, min(h_c, int(round(y2 * sy))))
                        ai_sub = cached_ai[sy1:sy2, sx1:sx2]
                        if ai_sub.shape[:2] != proc_starless.shape[:2]:
                            ai_aligned = cv2.resize(ai_sub, (proc_starless.shape[1], proc_starless.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            ai_aligned = ai_sub
                    elif cached_ai.shape[:2] != proc_starless.shape[:2]:
                        ai_aligned = cv2.resize(cached_ai, (proc_starless.shape[1], proc_starless.shape[0]), interpolation=cv2.INTER_AREA)
                    else:
                        ai_aligned = cached_ai
                    k = float(np.clip(self.denoise_strength, 0.0, 1.0))
                    denoised = cv2.addWeighted(ai_aligned, k, proc_starless, 1.0 - k, 0.0)
                else:
                    dn_color_ratio = getattr(self, 'denoise_color_strength', 0.5) / 0.5
                    dn_color = min(1.0, self.denoise_strength * dn_color_ratio)
                    from core.denoise import apply_astro_denoise
                    denoised = apply_astro_denoise(
                        proc_starless,
                        strength=self.denoise_strength,
                        color_strength=dn_color,
                        masking=getattr(self, 'denoise_masking', 0.5),
                        sky_mask=proc_mask_3d[..., 0] if proc_mask_3d is not None else None,
                        is_draft=is_draft,
                        is_full_res=(is_full_res and not is_draft)
                    )
                if proc_mask_3d is not None:
                    proc_starless = (denoised * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = denoised

            # 2. Claridad Clásica (Gauss), Claridad Textura (Seti) y Borrar Neblina
            if abs(self.clarity_starless_val) > 1e-4:
                clarified = apply_clarity(proc_starless, strength=self.clarity_starless_val)
                if proc_mask_3d is not None:
                    proc_starless = (clarified * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = clarified

            if abs(self.texture_clarity_starless_val) > 1e-4:
                from core.sharpening import apply_texture_clarity
                t_clarified = apply_texture_clarity(proc_starless, strength=self.texture_clarity_starless_val)
                if proc_mask_3d is not None:
                    proc_starless = (t_clarified * proc_mask_3d) + (proc_starless * (1.0 - proc_mask_3d))
                else:
                    proc_starless = t_clarified

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
                m_clean = self.current_mask[..., 0] if self.current_mask.ndim == 3 else self.current_mask
                if m_clean.dtype != np.float32:
                    m_clean = (m_clean / 255.0).astype(np.float32) if m_clean.dtype == np.uint8 else m_clean.astype(np.float32)
                elif m_clean.max() > 1.05:
                    m_clean = m_clean / 255.0
                m_f = cv2.resize(m_clean, (w_f, h_f), interpolation=cv2.INTER_LINEAR) if m_clean.shape[:2] != (h_f, w_f) else m_clean
                mask_3d = np.ascontiguousarray(np.repeat(m_f[..., np.newaxis], 3, axis=2), dtype=np.float32)
            crop_active = False
            self._zoom_crop_rect = None
        elif self.is_zoomed_100 and self.image_32bit is not None:
            h_f, w_f = self.image_32bit.shape[:2]

            # Propagación de margen dinámico (Padding) guiado por filtros espaciales activos (Orton / Difusión)
            active_radius = 0.0
            if float(getattr(self, 'orton_strength', 0.0)) > 1e-4:
                active_radius = max(active_radius, float(getattr(self, 'orton_radius', 40.0)))
            if float(getattr(self, 'star_diff_strength', 0.0)) > 1e-4:
                active_radius = max(active_radius, float(getattr(self, 'star_diff_radius', 20.0)))

            # Margen perimetral: para filtros convolucionales (3-sigma). Acotado entre 64 px y 512 px
            pad_margin = int(min(512, max(64, round(3.0 * active_radius)))) if active_radius > 0 else 64
            cw = min(w_f, max(100, self.canvas.width() + pad_margin * 2))
            ch = min(h_f, max(100, self.canvas.height() + pad_margin * 2))
            
            rx, ry = self.zoom_rel_coords
            cx = int(rx * w_f)
            cy = int(ry * h_f)

            x1 = max(0, min(w_f - cw, cx - cw // 2))
            y1 = max(0, min(h_f - ch, cy - ch // 2))
            x2 = x1 + cw
            y2 = y1 + ch
            self._zoom_crop_rect = (x1, y1, x2, y2)

            base = self.image_32bit[y1:y2, x1:x2].copy()
            if self.image_starless is not None:
                if self.image_starless.shape[:2] == (h_f, w_f):
                    starless = self.image_starless[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_starless.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_starless[sy1:sy2, sx1:sx2]
                    starless = cv2.resize(sub, (cw, ch), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (ch, cw) else sub.copy()
            else:
                starless = None

            if self.image_stars is not None:
                if self.image_stars.shape[:2] == (h_f, w_f):
                    stars = self.image_stars[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_stars.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_stars[sy1:sy2, sx1:sx2]
                    stars = cv2.resize(sub, (cw, ch), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (ch, cw) else sub.copy()
            else:
                stars = None

            if self.current_mask is not None:
                m_clean = self.current_mask[..., 0] if self.current_mask.ndim == 3 else self.current_mask
                if m_clean.dtype != np.float32:
                    m_clean = (m_clean / 255.0).astype(np.float32) if m_clean.dtype == np.uint8 else m_clean.astype(np.float32)
                elif m_clean.max() > 1.05:
                    m_clean = m_clean / 255.0
                if m_clean.shape[:2] != (h_f, w_f):
                    m_f = cv2.resize(m_clean, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
                else:
                    m_f = m_clean
                m_sub = np.ascontiguousarray(m_f[y1:y2, x1:x2], dtype=np.float32)
                mask_3d = np.ascontiguousarray(np.repeat(m_sub[..., np.newaxis], 3, axis=2), dtype=np.float32)
            else:
                mask_3d = None
            crop_active = True
        else:
            is_scrubbing = getattr(self, '_is_scrubbing_heavy_slider', False)
            if self.image_starless is not None:
                if is_scrubbing and getattr(self, 'draft_proxy', None) is not None:
                    starless = self.draft_starless
                    stars = self.draft_stars
                else:
                    starless = self.proxy_starless
                    stars = self.proxy_stars
            else:
                starless = None
                stars = None

            if is_scrubbing and getattr(self, 'draft_proxy', None) is not None:
                base = self.draft_proxy
                mask_3d = self.draft_mask
            else:
                base = self.preview_proxy
                mask_3d = self.proxy_mask
            crop_active = False
            self._zoom_crop_rect = None

        if base is None:
            return None, None

        bh, bw = base.shape[:2]
        if starless is not None and starless.shape[:2] != (bh, bw):
            starless = cv2.resize(starless, (bw, bh), interpolation=cv2.INTER_AREA)
        if stars is not None and stars.shape[:2] != (bh, bw):
            stars = cv2.resize(stars, (bw, bh), interpolation=cv2.INTER_AREA)
        if mask_3d is not None:
            if mask_3d.shape[:2] != (bh, bw):
                mask_3d = cv2.resize(mask_3d, (bw, bh), interpolation=cv2.INTER_LINEAR)
            if base.ndim == 3:
                if mask_3d.ndim == 2:
                    mask_3d = np.repeat(mask_3d[..., np.newaxis], 3, axis=2)
                elif mask_3d.ndim == 3 and mask_3d.shape[2] == 1:
                    mask_3d = np.repeat(mask_3d, 3, axis=2)

        if self.image_starless is None or starless is None or stars is None:
            return base, mask_3d

        processed_starless = self._process_starless_base(
            starless,
            mask_3d=mask_3d,
            is_full_res=(for_export or crop_active),
            crop_rect=self._zoom_crop_rect if self.is_zoomed_100 else None
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
        is_preview: bool = True,
        asynchronous: bool = False
    ) -> Optional[np.ndarray]:
        if target_img is None:
            return None

        is_full_res = (self.image_32bit is not None and target_img.shape[:2] == self.image_32bit.shape[:2])
        is_zoomed = (self.is_zoomed_100 and getattr(self, '_zoom_crop_rect', None) is not None)
        full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None
        crop_rect = getattr(self, '_zoom_crop_rect', None) if is_zoomed else None

        curve_state = tuple((m, tuple(pts)) for m, pts in sorted(self.curve_widget.get_state().items())) if hasattr(self, 'curve_widget') else ()

        # Hito 0a: Parámetros Pre-Polución (Dual-Band, Corrección Estelar, PCC, Balance de blancos, Saturación, Vibrance)
        col_st = self.panel_color_wb.get_state() if hasattr(self, 'panel_color_wb') else {}
        key_pre_lp = (
            id(target_img),
            id(precomputed_mask),
            is_full_res,
            is_zoomed,
            crop_rect,
            col_st.get("dualband_mode", "none"),
            int(col_st.get("dualband_oiii_boost", 100)),
            int(col_st.get("unclip_stars_val", 0)),
            int(col_st.get("unpurple_val", 0)),
            int(col_st.get("scnr_green_val", 0)),
            round(float(col_st.get("pcc_kr", 1.0)), 4),
            round(float(col_st.get("pcc_kb", 1.0)), 4),
            round(float(self.temp_val), 4),
            round(float(self.tint_val), 4),
            round(float(self.sat_sky_val), 4),
            round(float(self.sat_gnd_val), 4),
            round(float(self.vibrance_val), 4),
        )

        # Hito 0b: Parámetros Lineales (Polución Lumínica, Suelo, Denoise Lineal)
        key_linear = (
            key_pre_lp,
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
            round(float(getattr(self, 'gnd_denoise_val', 0.0)), 4),
            bool(getattr(self, 'gnd_ai_denoise_enabled', False)),
            id(getattr(self, '_cache_ground_ai', None)),
            int(getattr(self, 'gnd_edge_shift_val', 0)),
            round(float(getattr(self, 'gnd_antihalo_val', 0.0)), 4),
            round(float(getattr(self, 'denoise_strength', 0.0)), 4),
            round(float(getattr(self, 'denoise_color_strength', 0.5)), 4),
            round(float(getattr(self, 'denoise_masking', 0.5)), 4),
        )

        # Hito 1: Parámetros de Estirado (MTF / GHS)
        st = self.panel_stretch.get_state() if hasattr(self, 'panel_stretch') else {}
        key_stretch = (
            key_linear,
            st.get("stretch_mode", "mtf"),
            st.get("ghs_color_mode", "luma"),
            round(float(st.get("ghs_sp_val", 0.02)), 5),
            round(float(st.get("ghs_d_val", 5.0)), 4),
            round(float(st.get("ghs_b_val", 2.0)), 4),
            round(float(st.get("ghs_hp_val", 0.0)), 4),
            round(float(st.get("ghs_lp_val", 0.0)), 4),
            round(float(st.get("ghs_bp_val", 0.0)), 5),
            round(float(st.get("bp_val", 0.0)), 5),
            round(float(st.get("mtf_val", 0.15)), 5),
        )
        # Hito 2: Parámetros de Curvas y Contraste
        key_graded = (
            key_stretch,
            curve_state,
            round(float(self.contrast_starless_val), 4),
            round(float(self.contrast_val), 4),
        )

        # Hito 3: Parámetros Espaciales (Deconvolución, Orton)
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
        )

        # Resolve starless and stars image slices
        if is_zoomed:
            x1, y1, x2, y2 = self._zoom_crop_rect
            if self.image_stars is not None and full_shape is not None:
                h_f, w_f = full_shape
                if self.image_stars.shape[:2] == (h_f, w_f):
                    st_img = self.image_stars[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_stars.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_stars[sy1:sy2, sx1:sx2]
                    st_img = cv2.resize(sub, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (y2 - y1, x2 - x1) else sub.copy()
            else:
                st_img = None

            if self.image_starless is not None and full_shape is not None:
                h_f, w_f = full_shape
                if self.image_starless.shape[:2] == (h_f, w_f):
                    sl_img = self.image_starless[y1:y2, x1:x2].copy()
                else:
                    h_s, w_s = self.image_starless.shape[:2]
                    sx = w_s / float(w_f)
                    sy = h_s / float(h_f)
                    sx1 = max(0, min(w_s, int(round(x1 * sx))))
                    sy1 = max(0, min(h_s, int(round(y1 * sy))))
                    sx2 = max(0, min(w_s, int(round(x2 * sx))))
                    sy2 = max(0, min(h_s, int(round(y2 * sy))))
                    sub = self.image_starless[sy1:sy2, sx1:sx2]
                    sl_img = cv2.resize(sub, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) if sub.shape[:2] != (y2 - y1, x2 - x1) else sub.copy()
            else:
                sl_img = None
        elif is_full_res:
            st_img = self.image_stars if self.image_starless is not None else None
            sl_img = self.image_starless
        else:
            if self.image_starless is not None:
                if getattr(self, '_is_scrubbing_heavy_slider', False):
                    st_img = self.draft_stars if self.draft_stars is not None else self.proxy_stars
                    sl_img = self.draft_starless if self.draft_starless is not None else self.proxy_starless
                else:
                    st_img = self.proxy_stars if self.proxy_stars is not None else self.image_stars
                    sl_img = self.proxy_starless if self.proxy_starless is not None else self.image_starless
            else:
                st_img = None
                sl_img = None

        # Mask layers
        mask_layers = self.mask_layers_widget.layers if (apply_mask_layers and hasattr(self, 'mask_layers_widget')) else None

        # Hito Estrellas: Estirado y Filtro Difusor de Estrellas
        key_stars = (
            id(st_img),
            key_stretch,
            round(float(getattr(self, 'star_intensity', 1.0)), 4),
            round(float(getattr(self, 'star_diff_strength', 0.0)), 4),
            round(float(getattr(self, 'star_diff_radius', 20.0)), 2),
            round(float(getattr(self, 'star_diff_threshold', 0.45)), 4),
            round(float(getattr(self, 'star_diff_color_boost', 0.50)), 4),
            getattr(self, 'star_diff_mode', 'screen'),
            id(precomputed_mask),
            bool(mask_layers and len(mask_layers) > 0),
        )

        # Determinar hitos en caché reutilizables mediante caché LRU unificada por hash y dimensiones
        t_shape = target_img.shape[:2]
        if is_preview and hasattr(self, 'buffer_cache'):
            pre_spatial = self.buffer_cache.get("spatial", key_spatial, t_shape)
            pre_graded = self.buffer_cache.get("graded", key_graded, t_shape) if pre_spatial is None else None
            pre_stretched = self.buffer_cache.get("stretch", key_stretch, t_shape) if (pre_spatial is None and pre_graded is None) else None
            pre_linear = self.buffer_cache.get("linear", key_linear, t_shape) if (pre_spatial is None and pre_graded is None and pre_stretched is None) else None
            pre_pre_lp = self.buffer_cache.get("pre_lp", key_pre_lp, t_shape) if (pre_spatial is None and pre_graded is None and pre_stretched is None and pre_linear is None) else None
            pre_stars = self.buffer_cache.get("stars", key_stars, t_shape)
        else:
            pre_spatial = pre_graded = pre_stretched = pre_linear = pre_pre_lp = pre_stars = None

        # Build DeveloperState snapshot (omitiendo mask_layers para no serializar innecesariamente)
        state = DeveloperState.from_dict(self._capture_state(include_mask_layers=False))

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

        # Mask layers (instantánea superficial rápida para aislar el hilo de render)
        raw_layers = self.mask_layers_widget.layers if (apply_mask_layers and hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.layers) else None
        mask_layers = [l.shallow_snapshot() for l in raw_layers] if raw_layers else None

        is_scrubbing = getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_scrubbing_layer_slider', False)
        is_dragging_antihalo = getattr(self, '_is_dragging_antihalo', False)

        cached_gnd_ai = self._cache_ground_ai if getattr(self, 'gnd_ai_denoise_enabled', False) else None

        if asynchronous and getattr(self, 'async_pipeline', None) is not None:
            self._last_async_meta = (
                key_pre_lp, key_linear, key_stretch, key_graded, key_spatial, key_stars,
                is_zoomed, is_scrubbing, st_img, precomputed_mask
            )
            self._last_async_base = target_img
            self.async_pipeline.submit(
                target_img=target_img,
                state=state,
                cached_ground_ai=cached_gnd_ai,
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
                precomputed_pre_lp_base=pre_pre_lp,
                precomputed_linear_base=pre_linear,
                precomputed_stretched_base=pre_stretched,
                precomputed_graded_base=pre_graded,
                precomputed_spatial_base=pre_spatial,
                precomputed_stretched_stars=pre_stars,
                is_dragging_deconv=getattr(self, '_is_dragging_deconv', False),
                is_dragging_antihalo=is_dragging_antihalo,
            )
            return None

        final_img, out_pre_lp, out_lin, out_str, out_grd, out_spt, out_stars = apply_developer_pipeline(
            target_img=target_img,
            state=state,
            cached_ground_ai=cached_gnd_ai,
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
            precomputed_pre_lp_base=pre_pre_lp,
            precomputed_linear_base=pre_linear,
            precomputed_stretched_base=pre_stretched,
            precomputed_graded_base=pre_graded,
            precomputed_spatial_base=pre_spatial,
            precomputed_stretched_stars=pre_stars,
            return_milestones=True,
            is_dragging_deconv=getattr(self, '_is_dragging_deconv', False),
            is_dragging_antihalo=is_dragging_antihalo,
        )

        if is_preview:
            if not is_scrubbing:
                if hasattr(self, 'buffer_cache'):
                    self.buffer_cache.put("pre_lp", key_pre_lp, t_shape, out_pre_lp)
                    self.buffer_cache.put("linear", key_linear, t_shape, out_lin)
                    self.buffer_cache.put("stretch", key_stretch, t_shape, out_str)
                    self.buffer_cache.put("graded", key_graded, t_shape, out_grd)
                    self.buffer_cache.put("spatial", key_spatial, t_shape, out_spt)
                    self.buffer_cache.put("stars", key_stars, t_shape, out_stars)

                self._cache_ms_pre_lp_key = key_pre_lp
                self._cache_ms_pre_lp_img = out_pre_lp
                self._cache_ms_linear_key = key_linear
                self._cache_ms_linear_img = out_lin
                if out_lin is not None and not is_zoomed:
                    self._cache_ms_linear_u16 = np.clip(out_lin * 65535.0, 0, 65535).astype(np.uint16)
                if st_img is not None and not is_zoomed:
                    self._cache_ms_stars_u16 = np.clip(st_img * 65535.0, 0, 65535).astype(np.uint16)
                self._cache_ms_stretch_key = key_stretch
                self._cache_ms_stretch_img = out_str
                self._cache_ms_graded_key = key_graded
                self._cache_ms_graded_img = out_grd
                self._cache_ms_spatial_key = key_spatial
                self._cache_ms_spatial_img = out_spt
                self._cache_ms_stars_key = key_stars
                self._cache_ms_stars_img = out_stars
                self._cache_pre_layers_key = (key_spatial, key_stars)
                self._cache_pre_layers_stretched = out_spt
                self._cache_pre_layers_stars = out_stars
                if not is_zoomed:
                    self._cached_base_stretched = out_spt
                    self._cached_stars_stretched = out_stars
                    self._cached_sky_mask = precomputed_mask
                    self._cached_rendered_image = final_img

        return final_img

    def _on_async_preview_ready(self, req_id: int, result: Any):
        if result is None:
            return
        final_img, out_pre_lp, out_lin, out_str, out_grd, out_spt, out_stars = result
        if final_img is None:
            return

        meta = getattr(self, '_last_async_meta', None)
        if meta is not None:
            key_pre_lp, key_linear, key_stretch, key_graded, key_spatial, key_stars, is_zoomed, is_scrubbing, st_img, precomputed_mask = meta
            if is_scrubbing and not getattr(self, '_is_slider_pressed', False) and not getattr(self, '_is_scrubbing_heavy_slider', False) and not getattr(self, '_is_scrubbing_layer_slider', False):
                # Descartar solicitud de borrador tardía si el usuario ya soltó el deslizador
                return
            if not is_scrubbing:
                t_shape = final_img.shape[:2] if hasattr(final_img, 'shape') else ()
                if hasattr(self, 'buffer_cache') and t_shape:
                    self.buffer_cache.put("pre_lp", key_pre_lp, t_shape, out_pre_lp)
                    self.buffer_cache.put("linear", key_linear, t_shape, out_lin)
                    self.buffer_cache.put("stretch", key_stretch, t_shape, out_str)
                    self.buffer_cache.put("graded", key_graded, t_shape, out_grd)
                    self.buffer_cache.put("spatial", key_spatial, t_shape, out_spt)
                    self.buffer_cache.put("stars", key_stars, t_shape, out_stars)

                self._cache_ms_pre_lp_key = key_pre_lp
                self._cache_ms_pre_lp_img = out_pre_lp
                self._cache_ms_linear_key = key_linear
                self._cache_ms_linear_img = out_lin
                if out_lin is not None and not is_zoomed:
                    self._cache_ms_linear_u16 = np.clip(out_lin * 65535.0, 0, 65535).astype(np.uint16)
                if st_img is not None and not is_zoomed:
                    self._cache_ms_stars_u16 = np.clip(st_img * 65535.0, 0, 65535).astype(np.uint16)
                self._cache_ms_stretch_key = key_stretch
                self._cache_ms_stretch_img = out_str
                self._cache_ms_graded_key = key_graded
                self._cache_ms_graded_img = out_grd
                self._cache_ms_spatial_key = key_spatial
                self._cache_ms_spatial_img = out_spt
                self._cache_ms_stars_key = key_stars
                self._cache_ms_stars_img = out_stars
                self._cache_pre_layers_key = (key_spatial, key_stars)
                self._cache_pre_layers_stretched = out_spt
                self._cache_pre_layers_stars = out_stars
                if not is_zoomed:
                    self._cached_base_stretched = out_spt
                    self._cached_stars_stretched = out_stars
                    self._cached_sky_mask = precomputed_mask
                    self._cached_rendered_image = final_img

        self._last_rendered_preview = final_img
        base_to_render = getattr(self, '_last_async_base', None)
        if base_to_render is not None:
            full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None
            self.canvas.set_external_crop(None, full_shape)
            self.canvas.load_image(base_to_render, display_stretched=final_img)

        # Actualización fluida y controlada del histograma en vivo
        if not self.is_zoomed_100 and hasattr(self, '_hist_throttle_timer'):
            if not self._hist_throttle_timer.isActive():
                self._hist_throttle_timer.start(50)
        self._update_canvas_info_label()

    def _on_async_pipeline_failed(self, req_id: int, error_msg: str):
        logger.error(f"[Async Pipeline #{req_id} Error] {error_msg}")

    def update_stretch_preview(self, immediate: bool = False):
        if getattr(self, '_is_loading_project', False):
            return
        if self.is_zoomed_100 and not immediate:
            self._slider_changed_during_zoom = True
        if immediate:
            if hasattr(self, '_preview_throttle_timer'):
                self._preview_throttle_timer.stop()
            if hasattr(self, '_refine_timer'):
                self._refine_timer.stop()
            self._do_update_stretch_preview(synchronous=True)
            self._refresh_histograms_now()
        else:
            if getattr(self, '_is_slider_pressed', False) and not getattr(self, '_is_scrubbing_lut_slider', False) and not getattr(self, '_is_scrubbing_layer_slider', False):
                self._is_scrubbing_heavy_slider = True

            if hasattr(self, '_refine_timer') and (getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_scrubbing_layer_slider', False)):
                if not getattr(self, '_is_scrubbing_lut_slider', False):
                    self._refine_timer.start(350)

            if hasattr(self, '_preview_throttle_timer'):
                if getattr(self, '_is_dragging_gradient', False):
                    if not self._preview_throttle_timer.isActive():
                        self._preview_throttle_timer.start(40)
                else:
                    self._preview_throttle_timer.start(16)
            else:
                self._do_update_stretch_preview(synchronous=False)
                self._refresh_histograms_now()

    def _on_refine_preview_timeout(self):
        """Refina progresivamente la previsualización al nivel de alta calidad cuando cesa el arrastre."""
        if getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_scrubbing_layer_slider', False):
            self._is_scrubbing_heavy_slider = False
            self._is_scrubbing_layer_slider = False
            self.update_stretch_preview(immediate=True)

    def _refresh_histograms_now(self):
        if getattr(self, '_hist_throttle_timer', None) is not None:
            self._hist_throttle_timer.stop()
        stretched = getattr(self, '_last_rendered_preview', None)
        if stretched is None or self.is_zoomed_100:
            return
        # El histograma en vivo optimizado (< 0.2 ms) se actualiza de forma fluida
        if hasattr(self, 'live_histogram') and self.live_histogram.isVisible():
            self.live_histogram.update_from_image(stretched)
        # Omitir widgets más pesados durante arrastre activo para garantizar 60 FPS
        if getattr(self, '_is_scrubbing_heavy_slider', False) or getattr(self, '_is_slider_pressed', False) or getattr(self, '_is_dragging_gradient', False):
            return
        if hasattr(self, 'curve_widget') and self.curve_widget.isVisible():
            self.curve_widget.set_histogram_from_image(stretched)
        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.isVisible():
            if getattr(self.mask_layers_widget, 'sec_layer_curve', None) and self.mask_layers_widget.sec_layer_curve.is_expanded:
                self.mask_layers_widget.update_active_layer_histogram()

    def _is_lut_preview_applicable(self) -> bool:
        """Determina si la previsualización directa por LUT 1D es segura y matemáticamente idéntica."""
        if not getattr(self, '_is_scrubbing_lut_slider', False):
            return False
        src_u16 = getattr(self, '_cache_ms_linear_u16', None)
        if src_u16 is None:
            return False
        if getattr(self, 'preview_proxy', None) is not None:
            if src_u16.shape[:2] != self.preview_proxy.shape[:2]:
                return False
        if self.view_layer_mode == 2 or self.is_zoomed_100:
            return False
        if hasattr(self, 'mask_layers_widget') and len(getattr(self.mask_layers_widget, 'layers', [])) > 0:
            return False
        if float(getattr(self, 'orton_strength', 0.0)) > 1e-4:
            return False
        if float(getattr(self, 'star_diff_strength', 0.0)) > 1e-4:
            return False
        if int(getattr(self, 'deconv_iterations', 0)) > 0:
            return False
        return True

    def _do_update_stretch_preview(self, synchronous: bool = False):
        if getattr(self, '_is_rendering_preview', False):
            return
        self._is_rendering_preview = True
        try:
            # === RUTA ULTRARRÁPIDA DE VISUALIZACIÓN POR LUT (Display Transfer / STF >500 FPS) ===
            # Activada estrictamente durante el arrastre continuo de deslizadores de estirado/contraste
            if self._is_lut_preview_applicable():
                curve_rgb = self.curve_widget.get_lut("rgb") if (hasattr(self, 'curve_widget') and not self.curve_widget.is_identity("rgb")) else None
                
                # Obtener valores consolidados desde panel_stretch
                st = self.panel_stretch.get_state() if hasattr(self, 'panel_stretch') else {}
                contrast_val = float(st.get("contrast_val", 0)) / 100.0

                lut = compute_combined_stf_lut(
                    stretch_mode=st.get("stretch_mode", "mtf"),
                    bp=float(st.get("bp_val", 0.0)),
                    mtf=float(st.get("mtf_val", 0.15)),
                    ghs_sp=float(st.get("ghs_sp_val", 0.02)),
                    ghs_d=float(st.get("ghs_d_val", 5.0)),
                    ghs_b=float(st.get("ghs_b_val", 2.0)),
                    ghs_hp=float(st.get("ghs_hp_val", 0.0)),
                    ghs_lp=float(st.get("ghs_lp_val", 0.0)),
                    ghs_bp=float(st.get("ghs_bp_val", 0.0)),
                    curve_lut_rgb=curve_rgb,
                    contrast=contrast_val
                )

                src_u16 = self._cache_ms_linear_u16
                h, w = src_u16.shape[:2]
                if getattr(self, '_scrub_display_buf', None) is None or self._scrub_display_buf.shape[:2] != (h, w):
                    self._scrub_display_buf = np.empty((h, w, 3), dtype=np.uint8)

                apply_display_lut16(src_u16, lut, self._scrub_display_buf)

                before_u8 = None
                if self.canvas.split_view_enabled:
                    lut_before = compute_combined_stf_lut(
                        stretch_mode=getattr(self, "initial_stretch_mode", "mtf"),
                        bp=getattr(self, "initial_bp", 0.0),
                        mtf=getattr(self, "initial_mtf", 0.1),
                        ghs_sp=getattr(self, "initial_ghs_sp", 0.02),
                        ghs_d=getattr(self, "initial_ghs_d", 5.0),
                        contrast=0.0
                    )
                    if getattr(self, '_scrub_before_buf', None) is None or self._scrub_before_buf.shape[:2] != (h, w):
                        self._scrub_before_buf = np.empty((h, w, 3), dtype=np.uint8)
                    apply_display_lut16(src_u16, lut_before, self._scrub_before_buf)
                    before_u8 = self._scrub_before_buf

                self.canvas.update_display_u8(self._scrub_display_buf, before_u8=before_u8)
                self._update_canvas_info_label()
                return

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
                if synchronous and getattr(self, 'async_pipeline', None) is not None:
                    self.async_pipeline.cancel_pending()
                use_async = (not synchronous and not self.is_zoomed_100 and getattr(self, 'async_pipeline', None) is not None)
                stretched = self._apply_pipeline_on_image(
                    base_to_render,
                    precomputed_mask=active_mask,
                    asynchronous=use_async
                )
                if use_async:
                    return

            if stretched is not None:
                self._last_rendered_preview = stretched
                if self.is_zoomed_100 and getattr(self, '_zoom_crop_rect', None) is not None and self.image_32bit is not None:
                    h_f, w_f = self.image_32bit.shape[:2]
                    rx, ry = self.zoom_rel_coords
                    cx = rx * w_f
                    cy = ry * h_f
                    self.canvas.set_external_crop(
                        self._zoom_crop_rect, 
                        (h_f, w_f), 
                        sensor_center=(cx, cy)
                    )
                else:
                    full_shape = self.image_32bit.shape[:2] if self.image_32bit is not None else None
                    self.canvas.set_external_crop(None, full_shape)
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
                    self.canvas.set_before_image(stretched_before)

                # Actualización fluida y controlada del histograma en vivo
                if not self.is_zoomed_100 and hasattr(self, '_hist_throttle_timer'):
                    if not self._hist_throttle_timer.isActive():
                        self._hist_throttle_timer.start(50)

                self._update_canvas_info_label()

                # Reiniciar temporizador de consolidación de historial (captura scroll de ratón y flechas)
                if (
                    not self._is_undoing_or_redoing
                    and not getattr(self, '_is_applying_preset', False)
                    and not getattr(self, '_is_previewing_preset', False)
                    and hasattr(self, '_undo_debounce_timer')
                ):
                    self._undo_debounce_timer.start(500)
        finally:
            self._is_rendering_preview = False

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
        from core.starnet_bridge import find_starnet_executable
        if not is_starnet_onnx_available() and not find_starnet_executable():
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
        if hasattr(self, 'mask_layers_widget') and hasattr(self.mask_layers_widget, 'slider_gnd_special_dn'):
            if self.mask_layers_widget.slider_gnd_special_dn.value() != val:
                self.mask_layers_widget.slider_gnd_special_dn.setValue(val)
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
                if hasattr(self, 'mask_layers_widget') and hasattr(self.mask_layers_widget, 'slider_gnd_special_dn'):
                    self.mask_layers_widget.slider_gnd_special_dn.setValue(70)

            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_ground_ai_denoise_active(True)

            self._invalidate_pipeline_caches()
            self._generate_preview_proxy()
            self.update_stretch_preview()

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR IA SUELO] {err}")
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
            if hasattr(self, 'mask_layers_widget') and hasattr(self.mask_layers_widget, 'slider_sl_denoise'):
                if self.mask_layers_widget.slider_sl_denoise.value() == 0:
                    self.mask_layers_widget.slider_sl_denoise.setValue(70)

            # Notificar al botón para reflejar estado activo
            if hasattr(self, 'panel_starnet'):
                self.panel_starnet.set_ai_denoise_active(True)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_ai_denoise_active(True)

            self._invalidate_pipeline_caches()
            self._generate_preview_proxy()

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR IA DENOISE] {err}")
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

    def _set_curve_mode(self, mode: str) -> None:
        """Cambia el modo activo de la curva y sincroniza los botones de selección."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_mode(mode)
            self.btn_curve_rgb.setChecked(mode == "rgb")
            self.btn_curve_hue.setChecked(mode == "hue_sat")
            self.btn_curve_luma.setChecked(mode == "luma_sat")

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

    def _on_effects_parameter_changed(self):
        """Sincroniza el estado del panel modular de Efectos (Orton y Difusión) con el revelador."""
        if hasattr(self, 'panel_effects') and self.panel_effects is not None:
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

        if hasattr(self, 'mask_layers_widget'):
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
                self.combo_orton_mode.currentIndex(),
                denoise_color=self.slider_denoise_color.value() if hasattr(self, 'slider_denoise_color') else 50,
                denoise_mask=self.slider_denoise_mask.value() if hasattr(self, 'slider_denoise_mask') else 50,
                texture_clarity=self.slider_texture_clarity.value() if hasattr(self, 'slider_texture_clarity') else 0
            )
            self.mask_layers_widget.set_stars_special_values(
                presence=self.slider_stars.value(),
                diff_strength=self.slider_star_diff.value(),
                diff_radius=self.slider_star_diff_radius.value(),
                diff_threshold=self.slider_star_diff_threshold.value(),
                diff_color=self.slider_star_diff_color.value(),
                diff_mode_idx=self.combo_star_diff_mode.currentIndex()
            )

        self.update_stretch_preview()

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

    def reset_sliders(self):
        """Restablece los controles de estirado y contraste delegando en el panel."""
        if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
            self.panel_stretch.reset_sliders()
        else:
            self.sync_controls(0.0, 0.10)
            
    def on_canvas_pan_started(self):
        """Inicia el desplazamiento panorámico en Zoom 1:1 deteniendo refinados pendientes."""
        if not self.is_zoomed_100:
            return
        if hasattr(self, '_pan_refine_timer'):
            self._pan_refine_timer.stop()

    def on_canvas_pan_moved(self, dx: int, dy: int):
        """Desplaza el lienzo a 60-144 FPS mediante traslación de pixmap pura sin ningún bloqueo de CPU."""
        if not self.is_zoomed_100 or self.image_32bit is None:
            return
        h_f, w_f = self.image_32bit.shape[:2]

        rx, ry = self.zoom_rel_coords
        cx = rx * w_f
        cy = ry * h_f

        target_cx = cx - dx
        target_cy = cy - dy

        # Delimitación estricta a los límites visibles de la toma
        half_vw = min(w_f / 2.0, self.canvas.width() / 2.0)
        half_vh = min(h_f / 2.0, self.canvas.height() / 2.0)
        clamped_cx = max(half_vw, min(w_f - half_vw, target_cx)) if w_f > 2 * half_vw else (w_f / 2.0)
        clamped_cy = max(half_vh, min(h_f - half_vh, target_cy)) if h_f > 2 * half_vh else (h_f / 2.0)

        # Delta visual real permitido respetando los límites del sensor
        actual_dx = int(round(cx - clamped_cx))
        actual_dy = int(round(cy - clamped_cy))

        if actual_dx != 0 or actual_dy != 0:
            self.canvas.add_pan_visual_offset(actual_dx, actual_dy)
            self.zoom_rel_coords = (clamped_cx / float(w_f), clamped_cy / float(h_f))

        # Durante el arrastre activo continuo NO se ejecuta ningún temporizador de refinado
        if hasattr(self, '_pan_refine_timer') and self._pan_refine_timer.isActive():
            self._pan_refine_timer.stop()

    def on_canvas_pan_finished(self):
        """Finaliza el arrastre. Si el desplazamiento es amplio (>=120 px), programa refinado diferido (500 ms)."""
        if hasattr(self, '_pan_refine_timer'):
            self._pan_refine_timer.stop()

        vox = abs(self.canvas._pan_visual_offset.x()) if hasattr(self.canvas, '_pan_visual_offset') else 0
        voy = abs(self.canvas._pan_visual_offset.y()) if hasattr(self.canvas, '_pan_visual_offset') else 0

        # Si el desplazamiento es leve (<120 px), no recalcular: la vista es 90% nítida y evitamos congelar la UI
        if vox >= 120 or voy >= 120:
            if hasattr(self, '_pan_refine_timer'):
                self._pan_refine_timer.start(500)

    def _refine_pan_crop(self):
        """Genera el nuevo búfer extendido 1:1 centrado en la nueva posición y refresca histogramas."""
        if not self.is_zoomed_100:
            return
        if hasattr(self, '_pan_refine_timer'):
            self._pan_refine_timer.stop()
        self.update_stretch_preview(immediate=True)
        self._refresh_histograms_now()

    def on_canvas_zoom_toggled(self, is_zoomed: bool, rel_x: float, rel_y: float):
        if not is_zoomed:
            if hasattr(self, '_pan_refine_timer'):
                self._pan_refine_timer.stop()
            self.is_zoomed_100 = False
            self.canvas.is_zoomed = False
            self.canvas.zoom_center_img = None
            if hasattr(self, 'btn_toggle_zoom'):
                self.btn_toggle_zoom.blockSignals(True)
                self.btn_toggle_zoom.setChecked(False)
                self.btn_toggle_zoom.blockSignals(False)

            # Restauración 100% instantánea (0 ms) del mapa general si no se tocaron sliders durante el zoom
            if not getattr(self, '_slider_changed_during_zoom', False) and getattr(self, '_fit_view_pixmap', None) is not None:
                mask_overlay = getattr(self, '_fit_view_mask_overlay', None)
                self.canvas.restore_base_pixmap(
                    self._fit_view_pixmap, 
                    orig_rgb=getattr(self, '_fit_view_orig_rgb', None),
                    mask_overlay=mask_overlay
                )
                self._fit_view_pixmap = None
                self._fit_view_orig_rgb = None
                self._fit_view_mask_overlay = None
                self._update_canvas_info_label()
                return

            self.canvas.set_external_crop(None, None)
            self.canvas.setCursor(Qt.ArrowCursor)
            if hasattr(self.canvas, '_zoom_background_pixmap'):
                self.canvas._zoom_background_pixmap = None
            if hasattr(self.canvas, 'reset_pan_visual_offset'):
                self.canvas.reset_pan_visual_offset()
            self._fit_view_pixmap = None
            self._fit_view_orig_rgb = None
            self._fit_view_mask_overlay = None
            self._generate_before_reference()
            self.update_stretch_preview(immediate=True)
            self._update_canvas_info_label()
            return

        # Entrada en Zoom 1:1
        if self.canvas.base_pixmap and not self.canvas.base_pixmap.isNull():
            self._fit_view_pixmap = self.canvas.base_pixmap.copy()
            self._fit_view_orig_rgb = self.canvas.orig_rgb
            self._fit_view_mask_overlay = self.canvas.refined_overlay.copy() if (self.canvas.refined_overlay and not self.canvas.refined_overlay.isNull()) else None
            if getattr(self.canvas, '_zoom_background_pixmap', None) is None:
                self.canvas._zoom_background_pixmap = self.canvas.base_pixmap.copy()
        self._slider_changed_during_zoom = False
        self.is_zoomed_100 = True
        self.canvas.is_zoomed = True
        w_i = float(self.canvas.base_pixmap.width()) if self.canvas.base_pixmap else 1.0
        h_i = float(self.canvas.base_pixmap.height()) if self.canvas.base_pixmap else 1.0
        self.canvas.zoom_center_img = QPoint(int(rel_x * w_i), int(rel_y * h_i))
        self.zoom_rel_coords = (rel_x, rel_y)
        if hasattr(self, 'btn_toggle_zoom'):
            self.btn_toggle_zoom.blockSignals(True)
            self.btn_toggle_zoom.setChecked(True)
            self.btn_toggle_zoom.blockSignals(False)
        self.update_stretch_preview(immediate=True)
        self._update_canvas_info_label()

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
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                texture_clarity=self.slider_texture_clarity.value()
            )
        self.update_stretch_preview()

    def on_texture_clarity_changed(self, val: int):
        self.texture_clarity_starless_val = val / 100.0
        self.lbl_texture_clarity.setText(f"{self.texture_clarity_starless_val:+.2f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                texture_clarity=val
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
                self.slider_contrast_starless.value(), val,
                self.slider_orton.value(), self.slider_orton_radius.value(), self.slider_orton_threshold.value(),
                self.combo_orton_tint.currentIndex(), self.combo_orton_mode.currentIndex(),
                denoise_color=self.slider_denoise_color.value() if hasattr(self, 'slider_denoise_color') else 50,
                denoise_mask=self.slider_denoise_mask.value() if hasattr(self, 'slider_denoise_mask') else 50,
                texture_clarity=self.slider_texture_clarity.value() if hasattr(self, 'slider_texture_clarity') else 0
            )
        self.update_stretch_preview()

    def on_denoise_color_changed(self, val: int):
        self.denoise_color_strength = val / 100.0
        if hasattr(self, 'lbl_denoise_color'):
            self.lbl_denoise_color.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                self.slider_orton.value(), self.slider_orton_radius.value(), self.slider_orton_threshold.value(),
                self.combo_orton_tint.currentIndex(), self.combo_orton_mode.currentIndex(),
                denoise_color=val,
                denoise_mask=self.slider_denoise_mask.value() if hasattr(self, 'slider_denoise_mask') else 50
            )
        self.update_stretch_preview()

    def on_denoise_mask_changed(self, val: int):
        self.denoise_masking = val / 100.0
        if hasattr(self, 'lbl_denoise_mask'):
            self.lbl_denoise_mask.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value(), self.slider_dehaze.value(),
                self.slider_ond_est.value(), self.slider_ond_bg.value(),
                self.slider_contrast_starless.value(), self.slider_denoise.value(),
                self.slider_orton.value(), self.slider_orton_radius.value(), self.slider_orton_threshold.value(),
                self.combo_orton_tint.currentIndex(), self.combo_orton_mode.currentIndex(),
                denoise_color=self.slider_denoise_color.value() if hasattr(self, 'slider_denoise_color') else 50,
                denoise_mask=val
            )
        self.update_stretch_preview()

    def on_banding_changed(self, val: int):
        if hasattr(self, 'lbl_banding'):
            self.lbl_banding.setText(f"{val}%")
        self.update_stretch_preview()

    def on_ond_estructura_changed(self, val: int):
        self.multiescala_estructura = val / 100.0
        if hasattr(self, 'lbl_ond_est'):
            self.lbl_ond_est.setText(f"{self.multiescala_estructura:+.2f}")
        if hasattr(self, 'slider_ond_est') and self.slider_ond_est.value() != val:
            self.slider_ond_est.blockSignals(True)
            self.slider_ond_est.setValue(val)
            self.slider_ond_est.blockSignals(False)
        if hasattr(self, 'panel_sharpening') and self.panel_sharpening is not None:
            if self.panel_sharpening.row_ond_est.slider.value() != val:
                self.panel_sharpening.row_ond_est.slider.blockSignals(True)
                self.panel_sharpening.row_ond_est.slider.setValue(val)
                self.panel_sharpening.row_ond_est.slider.blockSignals(False)
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value() if hasattr(self, 'slider_clarity') else 0,
                self.slider_dehaze.value() if hasattr(self, 'slider_dehaze') else 0,
                val,
                self.slider_ond_bg.value() if hasattr(self, 'slider_ond_bg') else 0,
                self.slider_contrast_starless.value() if hasattr(self, 'slider_contrast_starless') else 0,
                self.slider_denoise.value() if hasattr(self, 'slider_denoise') else 0
            )
        self.update_stretch_preview()

    def on_ond_fondo_changed(self, val: int):
        self.multiescala_fondo = val / 100.0
        if hasattr(self, 'lbl_ond_bg'):
            self.lbl_ond_bg.setText(f"{val}%")
        if hasattr(self, 'slider_ond_bg') and self.slider_ond_bg.value() != val:
            self.slider_ond_bg.blockSignals(True)
            self.slider_ond_bg.setValue(val)
            self.slider_ond_bg.blockSignals(False)
        if hasattr(self, 'panel_sharpening') and self.panel_sharpening is not None:
            if self.panel_sharpening.row_ond_bg.slider.value() != val:
                self.panel_sharpening.row_ond_bg.slider.blockSignals(True)
                self.panel_sharpening.row_ond_bg.slider.setValue(val)
                self.panel_sharpening.row_ond_bg.slider.blockSignals(False)
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_starless_tone_values(
                self.slider_clarity.value() if hasattr(self, 'slider_clarity') else 0,
                self.slider_dehaze.value() if hasattr(self, 'slider_dehaze') else 0,
                self.slider_ond_est.value() if hasattr(self, 'slider_ond_est') else 0,
                val,
                self.slider_contrast_starless.value() if hasattr(self, 'slider_contrast_starless') else 0,
                self.slider_denoise.value() if hasattr(self, 'slider_denoise') else 0
            )
        self.update_stretch_preview()

    def on_gnd_ev_changed(self, val: int):
        self.gnd_ev_val = val / 100.0
        self.lbl_gnd_ev.setText(f"{self.gnd_ev_val:+.2f} EV")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                val, self.slider_gnd_sh.value(), self.slider_gnd_bp.value(),
                int(round(getattr(self, 'gnd_denoise_val', 0.0) * 100.0)),
                int(getattr(self, 'gnd_antihalo_val', 0) * 100),
                int(getattr(self, 'gnd_edge_shift_val', 0))
            )
        self.update_stretch_preview()

    def on_gnd_shadows_changed(self, val: int):
        self.gnd_shadows_val = val / 100.0
        self.lbl_gnd_sh.setText(f"{val}%")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                self.slider_gnd_ev.value(), val, self.slider_gnd_bp.value(),
                int(round(getattr(self, 'gnd_denoise_val', 0.0) * 100.0)),
                int(getattr(self, 'gnd_antihalo_val', 0) * 100),
                int(getattr(self, 'gnd_edge_shift_val', 0))
            )
        self.update_stretch_preview()

    def on_gnd_bp_changed(self, val: int):
        self.gnd_bp_val = - (val / 1000.0)
        display_val = val / 1000.0
        self.lbl_gnd_bp.setText(f"{display_val:+.3f}")
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_ground_tone_values(
                self.slider_gnd_ev.value(), self.slider_gnd_sh.value(), val,
                int(round(getattr(self, 'gnd_denoise_val', 0.0) * 100.0)),
                int(getattr(self, 'gnd_antihalo_val', 0) * 100),
                int(getattr(self, 'gnd_edge_shift_val', 0))
            )
        self.update_stretch_preview()

    def on_gnd_antihalo_changed(self, val: int):
        self.gnd_antihalo_val = val / 100.0
        if getattr(self, '_is_slider_pressed', False):
            self._is_dragging_antihalo = True
        if hasattr(self, 'panel_ground_sky') and self.panel_ground_sky is not None:
            if hasattr(self.panel_ground_sky, 'slider_gnd_antihalo') and self.panel_ground_sky.slider_gnd_antihalo.value() != val:
                self.panel_ground_sky.slider_gnd_antihalo.setValue(val)
        if hasattr(self, 'mask_layers_widget'):
            if hasattr(self.mask_layers_widget, 'slider_gnd_special_antihalo') and self.mask_layers_widget.slider_gnd_special_antihalo.value() != val:
                self.mask_layers_widget.slider_gnd_special_antihalo.setValue(val)
        self.update_stretch_preview()

    def on_gnd_edge_shift_changed(self, val: int):
        self.gnd_edge_shift_val = val
        if hasattr(self, 'panel_ground_sky') and self.panel_ground_sky is not None:
            if hasattr(self.panel_ground_sky, 'slider_gnd_edge_shift') and self.panel_ground_sky.slider_gnd_edge_shift.value() != val:
                self.panel_ground_sky.slider_gnd_edge_shift.setValue(val)
        if hasattr(self, 'mask_layers_widget'):
            if hasattr(self.mask_layers_widget, 'slider_gnd_special_edge_shift') and self.mask_layers_widget.slider_gnd_special_edge_shift.value() != val:
                self.mask_layers_widget.slider_gnd_special_edge_shift.setValue(val)
        if hasattr(self, 'spin_mask_shift') and self.spin_mask_shift.value() != val:
            self.spin_mask_shift.setValue(val)
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

