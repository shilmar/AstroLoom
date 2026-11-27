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
from gui.widgets.export_button import ExportSplitButton
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
    STYLE_RUN_IDLE, STYLE_SPECIAL_ACTION_BTN, STYLE_SECONDARY_BTN,
    BG_PANEL, BG_CARD, BG_INPUT, BORDER_BASE, BORDER_HOVER, ACCENT_COBALT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED
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
from gui.developer.tile_engine import AsyncTileManager
from core.buffer_cache import PipelineBufferCache
from core.display_transfer import (
    compute_combined_stf_lut,
    apply_display_lut16
)


from gui.developer.history_presets import HistoryPresetsMixin
from gui.developer.io_project_manager import ProjectIOMixin
from gui.developer.canvas_tools import CanvasToolsMixin
from gui.developer.layer_sync_bridge import LayerSyncMixin
from gui.developer.ai_workers_controller import AIWorkersMixin
from gui.developer.render_engine import RenderMixin



class DeveloperTab(RenderMixin, HistoryPresetsMixin, ProjectIOMixin, CanvasToolsMixin, LayerSyncMixin, AIWorkersMixin, QWidget):
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
        self._pan_refine_timer.setInterval(50)
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

        # Motor asíncrono de teselas fijas (512x512) con caché LRU y prefetch para Zoom 1:1
        self.tile_manager = AsyncTileManager(self)
        self.tile_manager.mosaic_ready.connect(self._on_tile_mosaic_ready)

        # Memoria de tamaños para paneles ocultables 
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
        self.proj_container.setStyleSheet(f"background: {BG_CARD}; border: 1px solid {BORDER_BASE}; border-radius: 4px;")
        proj_vbox = QVBoxLayout(self.proj_container)
        proj_vbox.setContentsMargins(6, 6, 6, 6)
        proj_vbox.setSpacing(5)

        self.lbl_project_title = QLabel("Proyecto: Sin guardar")
        self.lbl_project_title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: bold; border: none; background: transparent;")
        proj_vbox.addWidget(self.lbl_project_title)

        proj_btn_row = QHBoxLayout()
        proj_btn_row.setContentsMargins(0, 0, 0, 0)
        proj_btn_row.setSpacing(4)

        self.btn_open_proj = QPushButton("Abrir")
        self.btn_open_proj.setIcon(get_icon("folder", size=13, color="#a1a1aa"))
        self.btn_open_proj.setToolTip("Abrir proyecto de revelado .devloom (Atajo: Ctrl+O)")
        self.btn_open_proj.setCursor(Qt.PointingHandCursor)
        self.btn_open_proj.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_open_proj.clicked.connect(self.open_project_dialog)
        proj_btn_row.addWidget(self.btn_open_proj)

        self.btn_save_proj = QPushButton("Guardar")
        self.btn_save_proj.setIcon(get_icon("save", size=13, color="#a1a1aa"))
        self.btn_save_proj.setToolTip("Guardar sesión de revelado actual (Atajo: Ctrl+S)")
        self.btn_save_proj.setCursor(Qt.PointingHandCursor)
        self.btn_save_proj.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_save_proj.clicked.connect(self.save_project)
        proj_btn_row.addWidget(self.btn_save_proj)

        self.btn_save_as_proj = QPushButton("Guardar Como...")
        self.btn_save_as_proj.setIcon(get_icon("copy", size=13, color="#a1a1aa"))
        self.btn_save_as_proj.setToolTip("Guardar sesión de revelado en un archivo nuevo (Atajo: Ctrl+Shift+S)")
        self.btn_save_as_proj.setCursor(Qt.PointingHandCursor)
        self.btn_save_as_proj.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_save_as_proj.clicked.connect(self.save_project_as_dialog)
        proj_btn_row.addWidget(self.btn_save_as_proj)

        proj_vbox.addLayout(proj_btn_row)
        left_container_layout.addWidget(self.proj_container, 0)

        # Carga externa (Siempre fijada arriba)
        self.btn_open_img = QPushButton("Abrir Imagen (TIFF / FITS / RAW)...")
        self.btn_open_img.setIcon(get_icon("folder", color="#e4e4e7", size=16))
        self.btn_open_img.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_open_img.setCursor(Qt.PointingHandCursor)
        self.btn_open_img.clicked.connect(self.open_image_file)
        left_container_layout.addWidget(self.btn_open_img, 0)

        # Botón de Arranque Rápido: 1-Click Start Astro-Reveal
        self.btn_start_reveal = QPushButton("✨ 1-Click Start Astro-Reveal")
        self.btn_start_reveal.setIcon(get_icon("sparkles", color="#8B5CF6", hover_color="#A78BFA", size=16))
        self.btn_start_reveal.setToolTip("Arranque en 1 clic para paisaje: detecta horizonte por IA, neutraliza el fondo celestial, auto-estira el cielo y evalúa el suelo con Smart Ground Lift.")
        self.btn_start_reveal.setCursor(Qt.PointingHandCursor)
        self.btn_start_reveal.setFixedHeight(34)
        self.btn_start_reveal.setStyleSheet("""
            QPushButton {
                font-weight: 600;
                font-size: 11px;
                background-color: #1E2433;
                color: #E4E4E7;
                border: 1px solid #3B82F6;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #273147;
                border-color: #60A5FA;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #161B26;
                border-color: #2563EB;
            }
            QPushButton:disabled {
                background-color: #1A1A1E;
                color: #71717A;
                border-color: #2E2E35;
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

        btn_curve_style = (
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #2c2c32; color: #a1a1aa; border: 1px solid #2e2e35; }"
            "QPushButton:hover { background: #3f3f46; color: #ffffff; }"
            "QPushButton:checked { background: rgba(47, 129, 247, 0.15); color: #2f81f7; border: 1px solid #2f81f7; }"
        )

        self.btn_curve_rgb = QPushButton("Tonos (RGB)")
        self.btn_curve_rgb.setIcon(get_icon("curve", color="#a1a1aa", active_color="#2f81f7", size=12))
        self.btn_curve_rgb.setCheckable(True)
        self.btn_curve_rgb.setChecked(True)
        self.btn_curve_rgb.setStyleSheet(btn_curve_style)
        self.btn_curve_rgb.setToolTip("Curva tonal RGB continua con histograma logarítmico y avisos de recorte.")
        self.btn_curve_rgb.clicked.connect(lambda: self._set_curve_mode("rgb"))

        self.btn_curve_hue = QPushButton("Tono vs Sat")
        self.btn_curve_hue.setIcon(get_icon("palette", color="#a1a1aa", active_color="#2f81f7", size=12))
        self.btn_curve_hue.setCheckable(True)
        self.btn_curve_hue.setStyleSheet(btn_curve_style)
        self.btn_curve_hue.setToolTip("Curva selectiva de saturación en función del tono (Hue vs Sat).\nPermite saturar selectivamente H-alfa / nebulosas o neutralizar tonalidades no deseadas.")
        self.btn_curve_hue.clicked.connect(lambda: self._set_curve_mode("hue_sat"))

        self.btn_curve_luma = QPushButton("Luma vs Sat")
        self.btn_curve_luma.setIcon(get_icon("chart", color="#a1a1aa", active_color="#2f81f7", size=12))
        self.btn_curve_luma.setCheckable(True)
        self.btn_curve_luma.setStyleSheet(btn_curve_style)
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
        self.btn_reset_curve.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_reset_curve.setToolTip("Restablece únicamente la curva actualmente visible a su valor neutro.")
        self.btn_reset_curve.clicked.connect(lambda: self.curve_widget.reset_curve())

        self.btn_reset_all_curves = QPushButton("Resetear Todas")
        self.btn_reset_all_curves.setStyleSheet(STYLE_SECONDARY_BTN)
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
        self.mask_layers_widget.sky_annotations_requested.connect(self._on_generate_sky_annotations)
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
                self.sec_masks.set_status_badge(f"● {count} {'capas' if count > 1 else 'capa'}", "#3B82F6")
            else:
                self.sec_masks.set_status_badge("")
        self.mask_layers_widget.layers_changed.connect(_update_masks_badge)

        self.sec_masks.set_content_widget(self.mask_layers_widget)
        # Nota: self.sec_masks se añade a la columna derecha (right_layout)

        left_layout.addStretch(1)

        # Asignar panel de acordeones al área de scroll
        self.left_scroll_area.setWidget(left_panel)
        left_container_layout.addWidget(self.left_scroll_area, 1)

        # Botón Exportar dividido: Clic para maestro (TIFF/PSD/PNG/JPEG) y desplegable para formatos y Web
        self.btn_export = ExportSplitButton(
            text="EXPORTAR IMAGEN REVELADA...",
            on_master_export=self.export_image,
            on_web_export=self.export_web_image,
            parent=self
        )
        self.btn_export.setup_standard_menu(
            on_format_export=lambda fmt: self.export_image(force_format=fmt),
            psd_label="Guardar Photoshop por Capas (PSD)..."
        )
        self.btn_export.setFixedHeight(40)
        self.btn_export.setToolTip(
            "Exporta la imagen revelada a disco.\n"
            "• Clic principal: Formatos maestros (TIFF 16/32-bit, PSD multicapa, PNG, JPEG directo).\n"
            "• Desplegable (▼) o Ctrl+Alt+W: Acceso directo a formatos y Exportar para Web / Redes Sociales (sRGB 8-bit, Lanczos, WebP/JPEG/PNG)."
        )
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
        lbl_right_title.setStyleSheet(f"font-weight: 700; font-size: 11px; color: {TEXT_SECONDARY}; letter-spacing: 0.6px;")
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

        # Atajo global 'Tab' para alternar ambos paneles laterales (Modo Inmersivo / Cine )
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

        # Atajo global 'J' para alternar indicadores de recorte 
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
                background-color: {BG_INPUT};
                border: 1px solid {BORDER_BASE};
                border-radius: 4px;
                padding: 4px 10px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
            }}
            #canvas_toolbar QPushButton:hover {{
                background-color: {BORDER_HOVER};
                border-color: {BORDER_HOVER};
            }}
            #canvas_toolbar QPushButton:checked {{
                background-color: rgba(47, 129, 247, 0.2);
                color: #ffffff;
                border-color: {ACCENT_COBALT};
            }}
            #canvas_toolbar QPushButton:disabled {{
                background-color: {BG_PANEL};
                color: {TEXT_MUTED};
                border-color: {BORDER_BASE};
            }}
            #crop_toolbar {{
                background-color: {BG_PANEL};
                border-bottom: 1px solid {BORDER_BASE};
            }}
            #crop_toolbar QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
            }}
            #crop_toolbar QComboBox {{
                background-color: {BG_INPUT};
                border: 1px solid {BORDER_BASE};
                border-radius: 4px;
                padding: 2px 8px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
            }}
            #crop_toolbar QPushButton {{
                background-color: {BG_INPUT};
                border: 1px solid {BORDER_BASE};
                border-radius: 4px;
                padding: 3px 8px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
            }}
            #crop_toolbar QPushButton:hover {{
                background-color: {BORDER_HOVER};
                border-color: {BORDER_HOVER};
            }}
            #mask_toolbar {{
                background-color: {BG_PANEL};
                border-bottom: 1px solid {BORDER_BASE};
            }}
            #mask_toolbar QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
            }}
            #mask_toolbar QSpinBox {{
                background-color: {BG_INPUT};
                border: 1px solid {BORDER_BASE};
                border-radius: 4px;
                padding: 2px 4px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
            }}
            #mask_toolbar QPushButton {{
                background-color: {BG_INPUT};
                border: 1px solid {BORDER_BASE};
                border-radius: 4px;
                padding: 3px 8px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
            }}
            #mask_toolbar QPushButton:hover {{
                background-color: {BORDER_HOVER};
                border-color: {BORDER_HOVER};
            }}
        """)

    def log_message(self, text: str):
        log_hub.log("Revelador", text)



    def _set_curve_mode(self, mode: str) -> None:
        """Cambia el modo activo de la curva y sincroniza los botones de selección."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_mode(mode)
            self.btn_curve_rgb.setChecked(mode == "rgb")
            self.btn_curve_hue.setChecked(mode == "hue_sat")
            self.btn_curve_luma.setChecked(mode == "luma_sat")

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

    def reset_sliders(self):
        """Restablece los controles de estirado y contraste delegando en el panel."""
        if hasattr(self, 'panel_stretch') and self.panel_stretch is not None:
            self.panel_stretch.reset_sliders()
        else:
            self.sync_controls(0.0, 0.10)
            


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

