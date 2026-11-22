# gui/widgets/mask_layers/mask_definition_panels.py
"""
Sub-paneles de definición de máscara (Pincel, Luminancia, Color HSV, Suelo/Cielo y Gradiente).
Encapsula la creación visual y la interacción de grp_mask_def.
"""
from typing import Optional, Callable, Any, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QGroupBox, QMessageBox, QMenu, QApplication
)

from core.mask_layers import (
    MaskLayer,
    compute_effective_layer_mask,
    get_canvas_below_layer,
    MASK_TYPE_BRUSH,
    MASK_TYPE_GROUND,
    MASK_TYPE_SKY,
    MASK_TYPE_LUMA,
    MASK_TYPE_COLOR,
    MASK_TYPE_GRADIENT_LINEAR,
    MASK_TYPE_GRADIENT_RADIAL,
    MASK_TYPE_IMAGE_RASTER,
    LIMIT_MASK_NONE,
    LIMIT_MASK_SKY,
    LIMIT_MASK_GROUND,
)
from gui.icons import get_icon
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.widgets.mask_layers.common import make_resettable_label, make_box_qss


class MaskDefinitionWidget(QGroupBox):
    """
    Panel agrupado para la definición del origen de la máscara de capa:
    Pincel, Luminancia, Color HSV, Suelo/Cielo e IA, y Gradientes Analíticos.
    """
    brush_params_changed = Signal(int, float, float, str)
    brush_tool_toggled = Signal(bool, object)
    color_eyedropper_requested = Signal(bool)
    ai_ground_mask_requested = Signal()
    edit_ground_mask_requested = Signal()
    gradient_tool_toggled = Signal(bool, object)
    preview_mask_toggled = Signal(bool)
    preview_mask_requested = Signal(object)
    mask_param_changed = Signal()
    action_committed = Signal()
    master_mask_updated = Signal(object)
    mask_structure_changed = Signal()
    interaction_finished = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Definición de Máscara", parent)
        self.setObjectName("mask_def_group")

        self._updating_ui: bool = False
        self._last_brush_radius: int = 35
        self._last_brush_hardness: float = 0.5
        self._last_brush_flow: float = 0.8
        self._last_brush_mode: str = "paint"

        self._image_reference_getter: Optional[Callable[[], Any]] = None
        self._master_mask_getter: Optional[Callable[[], Any]] = None
        self._stars_getter: Optional[Callable[[], Any]] = None
        self._starless_getter: Optional[Callable[[], Any]] = None
        self._layers_getter: Optional[Callable[[], list]] = None
        self._selected_layer_getter: Optional[Callable[[], Optional[MaskLayer]]] = None

        self._setup_ui()

    def set_getters(
        self,
        img_getter: Optional[Callable[[], Any]] = None,
        mask_getter: Optional[Callable[[], Any]] = None,
        stars_getter: Optional[Callable[[], Any]] = None,
        starless_getter: Optional[Callable[[], Any]] = None,
        layers_getter: Optional[Callable[[], list]] = None,
        selected_layer_getter: Optional[Callable[[], Optional[MaskLayer]]] = None,
    ) -> None:
        self._image_reference_getter = img_getter
        self._master_mask_getter = mask_getter
        self._stars_getter = stars_getter
        self._starless_getter = starless_getter
        self._layers_getter = layers_getter
        self._selected_layer_getter = selected_layer_getter

    def get_selected_layer(self) -> Optional[MaskLayer]:
        if self._selected_layer_getter:
            return self._selected_layer_getter()
        return None

    def _setup_ui(self) -> None:
        self.layout_mask_def = QVBoxLayout(self)
        self.layout_mask_def.setContentsMargins(8, 10, 8, 8)
        self.layout_mask_def.setSpacing(6)

        # 1. Sub-panel Pincel
        self._setup_brush_panel()

        # 2. Sub-panel Luminancia
        self._setup_luma_panel()

        # 3. Sub-panel Color (HSV)
        self._setup_color_panel()

        # 4. Sub-panel Suelo / Cielo
        self._setup_ground_panel()

        # 5. Sub-panel Gradiente Lineal y Radial
        self._setup_gradient_panel()

        # 6. Fila Refinar Bordes (Guided Cross / LightCraft)
        row_refine = QHBoxLayout()
        self.chk_layer_refine = QCheckBox("Refinar Bordes (Guided Cross)")
        self.chk_layer_refine.setToolTip(
            "Imanta los bordes de la máscara a las aristas reales de la escena usando el Filtro Guiado Cruzado de LightCraft"
        )
        self.chk_layer_refine.toggled.connect(self._on_layer_refine_toggled)
        row_refine.addWidget(self.chk_layer_refine)

        self.slider_refine_strength = QSlider(Qt.Horizontal)
        self.slider_refine_strength.setRange(0, 100)
        self.slider_refine_strength.setValue(100)
        self.slider_refine_strength.setEnabled(False)
        self.slider_refine_strength.valueChanged.connect(self._on_layer_refine_strength_changed)
        self.slider_refine_strength.sliderReleased.connect(self.interaction_finished.emit)
        row_refine.addWidget(self.slider_refine_strength, stretch=1)

        self.lbl_refine_val = QLabel("100%")
        self.lbl_refine_val.setStyleSheet("color: #80d8ff; font-weight: bold; min-width: 35px;")
        row_refine.addWidget(self.lbl_refine_val)
        self.layout_mask_def.addLayout(row_refine)

        # 7. Botón de previsualización en lienzo
        self.btn_preview_mask = QPushButton(" Previsualizar Máscara en Lienzo")
        self.btn_preview_mask.setIcon(get_icon("eye", color="#80d8ff", size=13))
        self.btn_preview_mask.setCheckable(True)
        self.btn_preview_mask.setCursor(Qt.PointingHandCursor)
        self.btn_preview_mask.setStyleSheet("""
            QPushButton {
                background-color: #1a2736;
                color: #80d8ff;
                border: 1px solid #0288d1;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #0288d1;
                color: #ffffff;
            }
        """)
        self.btn_preview_mask.toggled.connect(self._on_preview_mask_toggled)
        self.layout_mask_def.addWidget(self.btn_preview_mask)

    def _setup_brush_panel(self) -> None:
        self.panel_brush = QWidget()
        brush_layout = QVBoxLayout(self.panel_brush)
        brush_layout.setContentsMargins(0, 0, 0, 0)
        brush_layout.setSpacing(6)

        # Acciones de Máscara de Capa para capas Ráster
        self.container_raster_mask_actions = QWidget()
        layout_rma = QVBoxLayout(self.container_raster_mask_actions)
        layout_rma.setContentsMargins(0, 0, 0, 4)
        layout_rma.setSpacing(4)

        self.row_rma_basic = QWidget()
        row_rma_1 = QHBoxLayout(self.row_rma_basic)
        row_rma_1.setContentsMargins(0, 0, 0, 0)
        row_rma_1.setSpacing(6)

        self.btn_raster_add_mask = QPushButton(" + Revelar Todo")
        self.btn_raster_add_mask.setIcon(get_icon("plus", color="#80d8ff", size=12))
        self.btn_raster_add_mask.setCursor(Qt.PointingHandCursor)
        self.btn_raster_add_mask.setToolTip("Crea una máscara de capa blanca (100% visible) para ocultar zonas con el pincel en Borrar")
        self.btn_raster_add_mask.clicked.connect(self._add_raster_layer_mask_white)
        row_rma_1.addWidget(self.btn_raster_add_mask)

        self.btn_raster_add_black = QPushButton(" + Ocultar Todo")
        self.btn_raster_add_black.setIcon(get_icon("plus", color="#a5d6a7", size=12))
        self.btn_raster_add_black.setCursor(Qt.PointingHandCursor)
        self.btn_raster_add_black.setToolTip("Crea una máscara de capa negra (capa oculta) para descubrir con el pincel en Pintar")
        self.btn_raster_add_black.clicked.connect(self._add_raster_layer_mask_black)
        row_rma_1.addWidget(self.btn_raster_add_black)
        layout_rma.addWidget(self.row_rma_basic)

        self.row_rma_horizon = QWidget()
        row_rma_2 = QHBoxLayout(self.row_rma_horizon)
        row_rma_2.setContentsMargins(0, 0, 0, 0)
        row_rma_2.setSpacing(6)

        self.btn_raster_add_ground = QPushButton(" + Máscara Suelo")
        self.btn_raster_add_ground.setIcon(get_icon("mountain", color="#a5d6a7", size=12))
        self.btn_raster_add_ground.setCursor(Qt.PointingHandCursor)
        self.btn_raster_add_ground.setToolTip("Crea una máscara de suelo (1.0 en suelo, 0.0 en cielo) para tomas en trípode.")
        self.btn_raster_add_ground.clicked.connect(lambda: self._add_raster_layer_mask_ground())
        row_rma_2.addWidget(self.btn_raster_add_ground)

        self.btn_raster_add_sky = QPushButton(" + Máscara Cielo")
        self.btn_raster_add_sky.setIcon(get_icon("sparkles", color="#80d8ff", size=12))
        self.btn_raster_add_sky.setCursor(Qt.PointingHandCursor)
        self.btn_raster_add_sky.setToolTip("Crea una máscara de cielo (1.0 en cielo, 0.0 en suelo) preservando el suelo.")
        self.btn_raster_add_sky.clicked.connect(lambda: self._add_raster_layer_mask_sky())
        row_rma_2.addWidget(self.btn_raster_add_sky)
        layout_rma.addWidget(self.row_rma_horizon)

        self.btn_raster_del_mask = QPushButton(" Quitar Máscara")
        self.btn_raster_del_mask.setIcon(get_icon("trash", color="#ef9a9a", size=12))
        self.btn_raster_del_mask.setCursor(Qt.PointingHandCursor)
        self.btn_raster_del_mask.setToolTip("Elimina la máscara de capa (hace la capa 100% visible)")
        self.btn_raster_del_mask.setStyleSheet("color: #ef9a9a;")
        self.btn_raster_del_mask.clicked.connect(self._remove_raster_layer_mask)
        layout_rma.addWidget(self.btn_raster_del_mask)

        brush_layout.addWidget(self.container_raster_mask_actions)

        # Fila Radio / Tamaño
        row_b_radius = QHBoxLayout()
        row_b_radius.addWidget(make_resettable_label(QLabel("Radio (Tamaño):"), lambda: self.slider_b_radius.setValue(35), "35 px"))
        self.lbl_b_radius = QLabel("35 px")
        self.lbl_b_radius.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_b_radius.addWidget(self.lbl_b_radius)
        self.slider_b_radius = QSlider(Qt.Horizontal)
        self.slider_b_radius.setRange(5, 250)
        self.slider_b_radius.setValue(35)
        self.slider_b_radius.valueChanged.connect(self._on_brush_param_changed)
        row_b_radius.addWidget(self.slider_b_radius, stretch=1)
        brush_layout.addLayout(row_b_radius)

        # Fila Dureza
        row_b_hard = QHBoxLayout()
        row_b_hard.addWidget(make_resettable_label(QLabel("Dureza:"), lambda: self.slider_b_hardness.setValue(50), "50%"))
        self.lbl_b_hardness = QLabel("50%")
        self.lbl_b_hardness.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_b_hard.addWidget(self.lbl_b_hardness)
        self.slider_b_hardness = QSlider(Qt.Horizontal)
        self.slider_b_hardness.setRange(0, 100)
        self.slider_b_hardness.setValue(50)
        self.slider_b_hardness.valueChanged.connect(self._on_brush_param_changed)
        row_b_hard.addWidget(self.slider_b_hardness, stretch=1)
        brush_layout.addLayout(row_b_hard)

        # Fila Flujo
        row_b_flow = QHBoxLayout()
        row_b_flow.addWidget(make_resettable_label(QLabel("Flujo / Opacidad:"), lambda: self.slider_b_flow.setValue(80), "80%"))
        self.lbl_b_flow = QLabel("80%")
        self.lbl_b_flow.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_b_flow.addWidget(self.lbl_b_flow)
        self.slider_b_flow = QSlider(Qt.Horizontal)
        self.slider_b_flow.setRange(5, 100)
        self.slider_b_flow.setValue(80)
        self.slider_b_flow.valueChanged.connect(self._on_brush_param_changed)
        row_b_flow.addWidget(self.slider_b_flow, stretch=1)
        brush_layout.addLayout(row_b_flow)

        # Fila Modos de pincel
        row_b_mode = QHBoxLayout()
        self.btn_b_paint = QPushButton(" Pintar (+)")
        self.btn_b_paint.setIcon(get_icon("brush", color="#80d8ff", size=13))
        self.btn_b_paint.setCheckable(True)
        self.btn_b_paint.setChecked(True)
        self.btn_b_paint.setCursor(Qt.PointingHandCursor)
        self.btn_b_paint.clicked.connect(lambda: self._set_brush_mode("paint"))
        row_b_mode.addWidget(self.btn_b_paint)

        self.btn_b_erase = QPushButton(" Borrar (-)")
        self.btn_b_erase.setIcon(get_icon("x", color="#ff8a80", size=13))
        self.btn_b_erase.setCheckable(True)
        self.btn_b_erase.setCursor(Qt.PointingHandCursor)
        self.btn_b_erase.clicked.connect(lambda: self._set_brush_mode("erase"))
        row_b_mode.addWidget(self.btn_b_erase)

        self.btn_b_invert = QPushButton(" Invertir")
        self.btn_b_invert.setIcon(get_icon("rotate_cw", color="#ce93d8", size=13))
        self.btn_b_invert.setCursor(Qt.PointingHandCursor)
        self.btn_b_invert.setToolTip("Invierte la máscara actual [Ctrl+I]")
        self.btn_b_invert.clicked.connect(self.invert_selected_brush_mask)
        row_b_mode.addWidget(self.btn_b_invert)

        self.btn_b_clear = QPushButton(" Limpiar")
        self.btn_b_clear.setIcon(get_icon("trash", color="#ef9a9a", size=13))
        self.btn_b_clear.setCursor(Qt.PointingHandCursor)
        self.btn_b_clear.setStyleSheet("color: #ef9a9a;")
        self.btn_b_clear.clicked.connect(self._clear_brush_mask)
        row_b_mode.addWidget(self.btn_b_clear)

        brush_layout.addLayout(row_b_mode)
        self.layout_mask_def.addWidget(self.panel_brush)

    def _setup_luma_panel(self) -> None:
        self.panel_luma = QWidget()
        luma_layout = QVBoxLayout(self.panel_luma)
        luma_layout.setContentsMargins(0, 0, 0, 0)
        luma_layout.setSpacing(6)

        row_luma_thresh = QHBoxLayout()
        row_luma_thresh.addWidget(QLabel("Mín:"))
        self.spin_luma_min = QDoubleSpinBox()
        self.spin_luma_min.setRange(0.0, 1.0)
        self.spin_luma_min.setSingleStep(0.05)
        self.spin_luma_min.setValue(0.0)
        self.spin_luma_min.valueChanged.connect(self._on_luma_param_changed)
        self.spin_luma_min.editingFinished.connect(self.interaction_finished.emit)
        row_luma_thresh.addWidget(self.spin_luma_min)

        row_luma_thresh.addWidget(QLabel("Máx:"))
        self.spin_luma_max = QDoubleSpinBox()
        self.spin_luma_max.setRange(0.0, 1.0)
        self.spin_luma_max.setSingleStep(0.05)
        self.spin_luma_max.setValue(1.0)
        self.spin_luma_max.valueChanged.connect(self._on_luma_param_changed)
        self.spin_luma_max.editingFinished.connect(self.interaction_finished.emit)
        row_luma_thresh.addWidget(self.spin_luma_max)

        row_luma_thresh.addWidget(QLabel("Borde:"))
        self.spin_luma_feather = QDoubleSpinBox()
        self.spin_luma_feather.setRange(0.001, 0.40)
        self.spin_luma_feather.setSingleStep(0.02)
        self.spin_luma_feather.setValue(0.08)
        self.spin_luma_feather.valueChanged.connect(self._on_luma_param_changed)
        self.spin_luma_feather.editingFinished.connect(self.interaction_finished.emit)
        row_luma_thresh.addWidget(self.spin_luma_feather)

        luma_layout.addLayout(row_luma_thresh)

        row_luma_presets = QHBoxLayout()
        btn_pre_shadows = QPushButton("Sombras [0-0.3]")
        btn_pre_shadows.clicked.connect(lambda: self._set_luma_preset(0.0, 0.30))
        row_luma_presets.addWidget(btn_pre_shadows)

        btn_pre_mid = QPushButton("Medios [0.25-0.75]")
        btn_pre_mid.clicked.connect(lambda: self._set_luma_preset(0.25, 0.75))
        row_luma_presets.addWidget(btn_pre_mid)

        btn_pre_high = QPushButton("Luces [0.7-1.0]")
        btn_pre_high.clicked.connect(lambda: self._set_luma_preset(0.70, 1.0))
        row_luma_presets.addWidget(btn_pre_high)

        luma_layout.addLayout(row_luma_presets)
        self.layout_mask_def.addWidget(self.panel_luma)

    def _setup_color_panel(self) -> None:
        self.panel_color = QWidget()
        color_layout = QVBoxLayout(self.panel_color)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(6)

        row_hue = QHBoxLayout()
        row_hue.addWidget(make_resettable_label(QLabel("Matiz (Hue):"), lambda: self.slider_hue.setValue(0), "0°"))
        self.lbl_hue_val = QLabel("0°")
        self.lbl_hue_val.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_hue.addWidget(self.lbl_hue_val)

        self.lbl_color_swatch = QLabel()
        self.lbl_color_swatch.setFixedSize(22, 16)
        self.lbl_color_swatch.setStyleSheet("background-color: #ff0000; border: 1px solid #ffffff; border-radius: 3px;")
        row_hue.addWidget(self.lbl_color_swatch)

        self.slider_hue = QSlider(Qt.Horizontal)
        self.slider_hue.setRange(0, 360)
        self.slider_hue.setValue(0)
        self.slider_hue.valueChanged.connect(self._on_hue_slider_changed)
        self.slider_hue.sliderReleased.connect(self.interaction_finished.emit)
        row_hue.addWidget(self.slider_hue, stretch=1)
        color_layout.addLayout(row_hue)

        row_pick = QHBoxLayout()
        self.btn_pick_color = QPushButton(" Muestrear Color en Imagen")
        self.btn_pick_color.setIcon(get_icon("pipette", color="#80d8ff", size=13))
        self.btn_pick_color.setCheckable(True)
        self.btn_pick_color.setCursor(Qt.PointingHandCursor)
        self.btn_pick_color.setToolTip("Activa la pipeta para hacer clic en el visor y definir automáticamente el matiz")
        self.btn_pick_color.setStyleSheet("""
            QPushButton {
                background-color: #262933;
                color: #e0e0e0;
                border: 1px solid #4c566a;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3b4252; border-color: #ffd54f; color: #ffffff; }
            QPushButton:checked { background-color: #d97706; color: #ffffff; border: 1px solid #f59e0b; font-weight: bold; }
        """)
        self.btn_pick_color.toggled.connect(self._on_pick_color_toggled)
        row_pick.addWidget(self.btn_pick_color)
        color_layout.addLayout(row_pick)

        row_tol_sat = QHBoxLayout()
        row_tol_sat.addWidget(QLabel("Tolerancia:"))
        self.spin_color_tol = QSpinBox()
        self.spin_color_tol.setRange(5, 120)
        self.spin_color_tol.setValue(30)
        self.spin_color_tol.setSuffix("°")
        self.spin_color_tol.valueChanged.connect(self._on_color_param_changed)
        self.spin_color_tol.editingFinished.connect(self.interaction_finished.emit)
        row_tol_sat.addWidget(self.spin_color_tol)

        row_tol_sat.addWidget(QLabel("Sat. Mín:"))
        self.spin_color_sat = QDoubleSpinBox()
        self.spin_color_sat.setRange(0.01, 0.90)
        self.spin_color_sat.setSingleStep(0.05)
        self.spin_color_sat.setValue(0.08)
        self.spin_color_sat.valueChanged.connect(self._on_color_param_changed)
        self.spin_color_sat.editingFinished.connect(self.interaction_finished.emit)
        row_tol_sat.addWidget(self.spin_color_sat)
        color_layout.addLayout(row_tol_sat)

        row_color_presets = QHBoxLayout()
        btn_pre_halpha = QPushButton("● H-Alfa (0°)")
        btn_pre_halpha.setStyleSheet("color: #ff5252; font-weight: 500;")
        btn_pre_halpha.clicked.connect(lambda: self._set_color_preset(0, 30))
        row_color_presets.addWidget(btn_pre_halpha)

        btn_pre_oiii = QPushButton("● OIII (190°)")
        btn_pre_oiii.setStyleSheet("color: #40c4ff; font-weight: 500;")
        btn_pre_oiii.clicked.connect(lambda: self._set_color_preset(190, 35))
        row_color_presets.addWidget(btn_pre_oiii)

        btn_pre_sodium = QPushButton("● Sodio (38°)")
        btn_pre_sodium.setStyleSheet("color: #ffb74d; font-weight: 500;")
        btn_pre_sodium.clicked.connect(lambda: self._set_color_preset(38, 25))
        row_color_presets.addWidget(btn_pre_sodium)

        btn_pre_airglow = QPushButton("● Airglow (115°)")
        btn_pre_airglow.setStyleSheet("color: #69f0ae; font-weight: 500;")
        btn_pre_airglow.clicked.connect(lambda: self._set_color_preset(115, 30))
        row_color_presets.addWidget(btn_pre_airglow)

        color_layout.addLayout(row_color_presets)
        self.layout_mask_def.addWidget(self.panel_color)

    def _setup_ground_panel(self) -> None:
        self.panel_ground = QWidget()
        gnd_layout = QVBoxLayout(self.panel_ground)
        gnd_layout.setContentsMargins(0, 0, 0, 0)
        gnd_layout.setSpacing(6)

        row_gnd_info = QHBoxLayout()
        self.lbl_gnd_desc = QLabel("Vinculada a la máscara de segmentación de suelo.")
        self.lbl_gnd_desc.setStyleSheet("color: #a5d6a7; font-size: 11px; font-style: italic;")
        row_gnd_info.addWidget(self.lbl_gnd_desc, stretch=1)

        self.btn_ai_gnd_mask = QPushButton(" Detección IA")
        self.btn_ai_gnd_mask.setIcon(get_icon("sparkles", color="#ffffff", size=13))
        self.btn_ai_gnd_mask.setCursor(Qt.PointingHandCursor)
        self.btn_ai_gnd_mask.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a1b9a, stop:1 #8e24aa);
                color: #ffffff;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px 8px;
                border: 1px solid #ba68c8;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b1fa2, stop:1 #9c27b0);
                border-color: #ce93d8;
            }
        """)
        self.btn_ai_gnd_mask.setToolTip("Detecta automáticamente el horizonte con IA ligera U²-Net")
        self.btn_ai_gnd_mask.clicked.connect(self.ai_ground_mask_requested.emit)
        row_gnd_info.addWidget(self.btn_ai_gnd_mask)

        self.btn_edit_gnd_mask = QPushButton(" Pintar Trazos...")
        self.btn_edit_gnd_mask.setIcon(get_icon("brush", color="#a5d6a7", size=13))
        self.btn_edit_gnd_mask.setCursor(Qt.PointingHandCursor)
        self.btn_edit_gnd_mask.clicked.connect(self.edit_ground_mask_requested.emit)
        row_gnd_info.addWidget(self.btn_edit_gnd_mask)
        gnd_layout.addLayout(row_gnd_info)

        self.layout_mask_def.addWidget(self.panel_ground)

    def _setup_gradient_panel(self) -> None:
        self.panel_gradient = QWidget()
        grad_layout = QVBoxLayout(self.panel_gradient)
        grad_layout.setContentsMargins(0, 0, 0, 0)
        grad_layout.setSpacing(6)

        row_limit = QHBoxLayout()
        row_limit.addWidget(QLabel("Limitar a:"))
        self.combo_grad_limit = QComboBox()
        self.combo_grad_limit.addItem("Sin límite (Toda la escena)", LIMIT_MASK_NONE)
        self.combo_grad_limit.addItem("Imantar / Solo Cielo", LIMIT_MASK_SKY)
        self.combo_grad_limit.addItem("Imantar / Solo Suelo", LIMIT_MASK_GROUND)
        self.combo_grad_limit.currentIndexChanged.connect(self._on_grad_limit_changed)
        row_limit.addWidget(self.combo_grad_limit, stretch=1)
        grad_layout.addLayout(row_limit)

        # Gradiente Lineal
        self.container_linear_props = QWidget()
        linear_vbox = QVBoxLayout(self.container_linear_props)
        linear_vbox.setContentsMargins(0, 0, 0, 0)
        linear_vbox.setSpacing(6)

        row_lin_profile = QHBoxLayout()
        row_lin_profile.addWidget(QLabel("Perfil:"))
        self.combo_lin_profile = QComboBox()
        self.combo_lin_profile.addItem("Suave (Smoothstep Cúbico)", "smoothstep")
        self.combo_lin_profile.addItem("Lineal", "linear")
        self.combo_lin_profile.addItem("Exponencial (Polución)", "exponential")
        self.combo_lin_profile.addItem("Banda Reflejada (0 ➔ 1 ➔ 0)", "reflected")
        self.combo_lin_profile.currentIndexChanged.connect(self._on_linear_profile_changed)
        row_lin_profile.addWidget(self.combo_lin_profile, stretch=1)
        linear_vbox.addLayout(row_lin_profile)

        self.row_lin_midpoint = PrecisionSliderRow("Punto Medio:", 10, 90, 50, scale=1.0, unit="%", label_width=110)
        self.slider_lin_midpoint = self.row_lin_midpoint.slider
        self.slider_lin_midpoint.valueChanged.connect(self._on_linear_midpoint_changed)
        self.row_lin_midpoint.sliderReleased.connect(self.interaction_finished.emit)
        linear_vbox.addWidget(self.row_lin_midpoint)

        self.container_reflected_props = QWidget()
        refl_vbox = QVBoxLayout(self.container_reflected_props)
        refl_vbox.setContentsMargins(0, 0, 0, 0)
        refl_vbox.setSpacing(4)

        self.chk_lin_symmetric = QCheckBox("Banda Simétrica (Lado A = Lado B)")
        self.chk_lin_symmetric.setChecked(True)
        self.chk_lin_symmetric.setStyleSheet("color: #80d8ff; font-size: 11px;")
        self.chk_lin_symmetric.toggled.connect(self._on_linear_symmetric_toggled)
        refl_vbox.addWidget(self.chk_lin_symmetric)

        self.row_lin_span_b = PrecisionSliderRow("Ancho Lado B:", 10, 300, 100, scale=1.0, unit="%", label_width=110)
        self.slider_lin_span_b = self.row_lin_span_b.slider
        self.slider_lin_span_b.valueChanged.connect(self._on_linear_span_b_changed)
        self.row_lin_span_b.sliderReleased.connect(self.interaction_finished.emit)
        refl_vbox.addWidget(self.row_lin_span_b)

        self.container_reflected_props.setVisible(False)
        linear_vbox.addWidget(self.container_reflected_props)
        grad_layout.addWidget(self.container_linear_props)

        # Gradiente Radial
        self.container_radial_props = QWidget()
        radial_vbox = QVBoxLayout(self.container_radial_props)
        radial_vbox.setContentsMargins(0, 0, 0, 0)
        radial_vbox.setSpacing(6)

        self.row_rad_feather = PrecisionSliderRow("Desvanecimiento:", 5, 100, 50, scale=1.0, unit="%", label_width=110)
        self.slider_rad_feather = self.row_rad_feather.slider
        self.slider_rad_feather.valueChanged.connect(self._on_radial_feather_changed)
        self.row_rad_feather.sliderReleased.connect(self.interaction_finished.emit)
        radial_vbox.addWidget(self.row_rad_feather)

        self.row_rad_angle = PrecisionSliderRow("Inclinación:", -180, 180, 0, scale=1.0, unit="°", force_sign=True, label_width=110)
        self.slider_rad_angle = self.row_rad_angle.slider
        self.slider_rad_angle.valueChanged.connect(self._on_radial_angle_changed)
        self.row_rad_angle.sliderReleased.connect(self.interaction_finished.emit)
        radial_vbox.addWidget(self.row_rad_angle)

        self.row_rad_rx = PrecisionSliderRow("Semieje X (Ancho):", 5, 150, 25, scale=1.0, unit="%", label_width=110)
        self.slider_rad_rx = self.row_rad_rx.slider
        self.slider_rad_rx.valueChanged.connect(self._on_radial_rx_changed)
        self.row_rad_rx.sliderReleased.connect(self.interaction_finished.emit)
        radial_vbox.addWidget(self.row_rad_rx)

        self.row_rad_ry = PrecisionSliderRow("Semieje Y (Alto):", 5, 150, 35, scale=1.0, unit="%", label_width=110)
        self.slider_rad_ry = self.row_rad_ry.slider
        self.slider_rad_ry.valueChanged.connect(self._on_radial_ry_changed)
        self.row_rad_ry.sliderReleased.connect(self.interaction_finished.emit)
        radial_vbox.addWidget(self.row_rad_ry)

        grad_layout.addWidget(self.container_radial_props)

        row_grad_reset = QHBoxLayout()
        self.btn_reset_grad_geom = QPushButton(" Restablecer Geometría")
        self.btn_reset_grad_geom.setIcon(get_icon("rotate_ccw", color="#80d8ff", size=12))
        self.btn_reset_grad_geom.setCursor(Qt.PointingHandCursor)
        self.btn_reset_grad_geom.setToolTip("Restablece el gradiente a su posición, tamaño y orientación por defecto")
        self.btn_reset_grad_geom.clicked.connect(self._reset_gradient_geometry)
        row_grad_reset.addWidget(self.btn_reset_grad_geom)
        grad_layout.addLayout(row_grad_reset)

        self.layout_mask_def.addWidget(self.panel_gradient)

    def is_preview_active(self) -> bool:
        return self.btn_preview_mask.isChecked()

    def set_preview_active(self, active: bool) -> None:
        if self.btn_preview_mask.isChecked() != active:
            self.btn_preview_mask.setChecked(active)

    def apply_theme(self, border_col: str, title_col: str, bg_col: str, title: str) -> None:
        self.setStyleSheet(make_box_qss("mask_def_group", border_col, title_col, bg_col))
        self.setTitle(title)

    def update_for_layer(self, layer: Optional[MaskLayer]) -> None:
        if not layer:
            self.setVisible(False)
            return

        self.setVisible(not layer.is_system_layer)
        if layer.is_system_layer:
            return

        self._updating_ui = True
        mtype = layer.mask_type

        # Visibilidad por tipo
        is_brush = getattr(layer, 'supports_brush', False) or (mtype == MASK_TYPE_BRUSH)
        is_raster = (mtype == MASK_TYPE_IMAGE_RASTER)
        self.panel_brush.setVisible(is_brush)
        self.panel_luma.setVisible(mtype == MASK_TYPE_LUMA)
        self.panel_color.setVisible(mtype == MASK_TYPE_COLOR)
        self.panel_ground.setVisible(mtype in (MASK_TYPE_GROUND, MASK_TYPE_SKY))
        self.panel_gradient.setVisible(getattr(layer, 'is_gradient', False))

        # Acciones ráster
        self.container_raster_mask_actions.setVisible(is_raster)
        if is_raster:
            has_mask = (layer.brush_mask is not None)
            self.btn_raster_del_mask.setVisible(has_mask)
            self.btn_b_paint.setVisible(has_mask)
            self.btn_b_erase.setVisible(has_mask)
            self.btn_b_invert.setVisible(has_mask)
            self.btn_b_clear.setVisible(has_mask)
            self.row_rma_basic.setVisible(not has_mask)
            self.row_rma_horizon.setVisible(not has_mask)

        # Actualizar valores de controles según el tipo
        if is_brush:
            rad = getattr(layer, 'brush_radius', self._last_brush_radius)
            hard = int(getattr(layer, 'brush_hardness', self._last_brush_hardness) * 100)
            flw = int(getattr(layer, 'brush_flow', self._last_brush_flow) * 100)
            self.slider_b_radius.setValue(rad)
            self.lbl_b_radius.setText(f"{rad} px")
            self.slider_b_hardness.setValue(hard)
            self.lbl_b_hardness.setText(f"{hard}%")
            self.slider_b_flow.setValue(flw)
            self.lbl_b_flow.setText(f"{flw}%")
            b_mode = getattr(layer, 'brush_mode', "paint")
            self.btn_b_paint.setChecked(b_mode == "paint")
            self.btn_b_erase.setChecked(b_mode == "erase")

        if mtype == MASK_TYPE_LUMA:
            self.spin_luma_min.setValue(getattr(layer, 'luma_min', 0.0))
            self.spin_luma_max.setValue(getattr(layer, 'luma_max', 1.0))
            self.spin_luma_feather.setValue(getattr(layer, 'feather', 0.08))

        if mtype == MASK_TYPE_COLOR:
            hue = int(getattr(layer, 'color_hue', 0.0))
            self.slider_hue.setValue(hue)
            self.lbl_hue_val.setText(f"{hue}°")
            self._update_color_swatch(hue)
            self.spin_color_tol.setValue(int(getattr(layer, 'color_tolerance', 30.0)))
            self.spin_color_sat.setValue(getattr(layer, 'color_sat_min', 0.08))

        if mtype in (MASK_TYPE_GROUND, MASK_TYPE_SKY):
            self.lbl_gnd_desc.setText(
                "Vinculada a la máscara de cielo." if mtype == MASK_TYPE_SKY
                else "Vinculada a la máscara de segmentación de suelo."
            )

        if getattr(layer, 'is_gradient', False):
            self.update_gradient_ui_from_layer(layer)

        # Refinar bordes
        refine = getattr(layer, 'refine_edges', False)
        strength = int(getattr(layer, 'refine_strength', 1.0) * 100)
        self.chk_layer_refine.setChecked(refine)
        self.slider_refine_strength.setEnabled(refine)
        self.slider_refine_strength.setValue(strength)
        self.lbl_refine_val.setText(f"{strength}%")

        self._updating_ui = False

    # --- Pincel ---
    def _on_brush_param_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and getattr(layer, 'supports_brush', False):
            layer.brush_radius = self.slider_b_radius.value()
            layer.brush_hardness = self.slider_b_hardness.value() / 100.0
            layer.brush_flow = self.slider_b_flow.value() / 100.0
            self._last_brush_radius = layer.brush_radius
            self._last_brush_hardness = layer.brush_hardness
            self._last_brush_flow = layer.brush_flow
            self.lbl_b_radius.setText(f"{layer.brush_radius} px")
            self.lbl_b_hardness.setText(f"{self.slider_b_hardness.value()}%")
            self.lbl_b_flow.setText(f"{self.slider_b_flow.value()}%")
            self.brush_params_changed.emit(
                layer.brush_radius, layer.brush_hardness, layer.brush_flow, layer.brush_mode
            )

    def _set_brush_mode(self, mode: str) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and getattr(layer, 'supports_brush', False):
            layer.brush_mode = mode
            self._last_brush_mode = mode
            self.btn_b_paint.setChecked(mode == "paint")
            self.btn_b_erase.setChecked(mode == "erase")
            self.brush_params_changed.emit(
                layer.brush_radius, layer.brush_hardness, layer.brush_flow, layer.brush_mode
            )

    def _clear_brush_mask(self) -> None:
        layer = self.get_selected_layer()
        if layer and getattr(layer, 'supports_brush', False):
            if hasattr(layer, "set_brush_mask"):
                layer.set_brush_mask(None)
            else:
                layer.brush_mask = None
            if layer.mask_type == MASK_TYPE_IMAGE_RASTER:
                layer.active_target = "image"
            self.brush_tool_toggled.emit(True, layer)
            self.set_preview_active(False)
            self.preview_mask_requested.emit(None)
            self.mask_param_changed.emit()
            self.action_committed.emit()

    def set_brush_radius_from_canvas(self, radius: int) -> None:
        self._last_brush_radius = radius
        layer = self.get_selected_layer()
        if layer and getattr(layer, 'supports_brush', False):
            layer.brush_radius = radius
            self._updating_ui = True
            self.slider_b_radius.setValue(radius)
            self.lbl_b_radius.setText(f"{radius} px")
            self._updating_ui = False

    def invert_selected_brush_mask(self) -> None:
        layer = self.get_selected_layer()
        if not layer or not getattr(layer, 'supports_brush', False):
            return

        h_m, w_m = self._get_layer_image_shape(layer)
        if layer.brush_mask is None:
            new_m = np.zeros((h_m, w_m), dtype=np.float32)
        else:
            new_m = np.clip(1.0 - layer.brush_mask, 0.0, 1.0).astype(np.float32)

        if hasattr(layer, "set_brush_mask"):
            layer.set_brush_mask(new_m)
        else:
            layer.brush_mask = new_m

        new_mode = "paint" if layer.brush_mode == "erase" else "erase"
        layer.brush_mode = new_mode
        self._last_brush_mode = new_mode
        self._updating_ui = True
        self.btn_b_paint.setChecked(new_mode == "paint")
        self.btn_b_erase.setChecked(new_mode == "erase")
        self._updating_ui = False

        self.mask_param_changed.emit()
        self.action_committed.emit()
        if self.btn_preview_mask.isChecked():
            self._emit_mask_preview()

    # --- Acciones Ráster ---
    def _get_layer_image_shape(self, layer: MaskLayer) -> Tuple[int, int]:
        if getattr(layer, 'image_data', None) is not None and hasattr(layer.image_data, 'shape') and len(layer.image_data.shape) >= 2:
            return layer.image_data.shape[:2]
        if self._image_reference_getter:
            try:
                base = self._image_reference_getter()
                if base is not None and hasattr(base, 'shape') and len(base.shape) >= 2:
                    return base.shape[:2]
            except Exception:
                pass
        return 1000, 1500

    def _add_raster_layer_mask_white(self) -> None:
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_IMAGE_RASTER:
            return
        h, w = self._get_layer_image_shape(layer)
        layer.set_brush_mask(np.ones((h, w), dtype=np.float32))
        layer.active_target = "mask"
        layer.brush_mode = "erase"
        self._last_brush_mode = "erase"
        self.btn_b_erase.setChecked(True)
        self.btn_b_paint.setChecked(False)
        self.update_for_layer(layer)
        self.brush_tool_toggled.emit(True, layer)
        self.mask_param_changed.emit()
        self.mask_structure_changed.emit()
        self.action_committed.emit()

    def _add_raster_layer_mask_black(self) -> None:
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_IMAGE_RASTER:
            return
        h, w = self._get_layer_image_shape(layer)
        layer.set_brush_mask(np.zeros((h, w), dtype=np.float32))
        layer.active_target = "mask"
        layer.brush_mode = "paint"
        self._last_brush_mode = "paint"
        self.btn_b_paint.setChecked(True)
        self.btn_b_erase.setChecked(False)
        self.update_for_layer(layer)
        self.brush_tool_toggled.emit(True, layer)
        self.mask_param_changed.emit()
        self.mask_structure_changed.emit()
        self.action_committed.emit()

    def _detect_horizon_from_image(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        if QApplication.instance():
            QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from core.ai_horizon import auto_detect_sky_mask
            mask, method = auto_detect_sky_mask(img, force_algorithmic=False)
            return mask, method
        except Exception:
            try:
                from core.ai_horizon import detect_sky_mask_algorithmic
                mask = detect_sky_mask_algorithmic(img)
                return mask, "Algorítmico (Fallback)"
            except Exception as e2:
                if QApplication.instance():
                    QMessageBox.warning(self, "Detección de Horizonte", f"No se pudo detectar el horizonte automáticamente:\n{e2}")
                return None, ""
        finally:
            if QApplication.instance():
                QApplication.restoreOverrideCursor()

    def _apply_raster_horizon_mask(self, is_ground: bool = True, source_choice: Optional[str] = None) -> None:
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_IMAGE_RASTER:
            return

        existing_master = self._master_mask_getter() if self._master_mask_getter else None
        has_existing_master = (existing_master is not None)
        raster_img = getattr(layer, 'image_data', None)
        base_img = self._image_reference_getter() if self._image_reference_getter else None

        chosen_sky_mask = None
        detected_fresh = False

        if source_choice == "master":
            chosen_sky_mask = existing_master
        elif source_choice == "raster":
            target = raster_img if raster_img is not None else base_img
            if target is not None:
                chosen_sky_mask, _ = self._detect_horizon_from_image(target)
                detected_fresh = True
        elif source_choice == "base":
            target = base_img if base_img is not None else raster_img
            if target is not None:
                chosen_sky_mask, _ = self._detect_horizon_from_image(target)
                detected_fresh = True
        else:
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu { background-color: #1a1e29; color: #e2e8f0; border: 1px solid #33384c; padding: 4px; }
                QMenu::item { padding: 6px 20px; font-size: 11px; }
                QMenu::item:selected { background-color: #2b3245; color: #80d8ff; }
            """)
            act_master = None
            act_base = None
            act_raster = None

            if has_existing_master:
                act_master = menu.addAction("Aprovechar Horizonte de la Escena (Recomendado)")
            if base_img is not None:
                title_base = "Detectar Horizonte en Imagen Base con IA" if not has_existing_master else "Redetectar Horizonte en Imagen Base con IA"
                act_base = menu.addAction(title_base)
            if raster_img is not None:
                act_raster = menu.addAction("Detectar Horizonte en esta Capa Ráster con IA")

            actions = [a for a in (act_master, act_base, act_raster) if a is not None]
            if len(actions) == 0:
                return
            elif len(actions) == 1:
                chosen = actions[0]
            else:
                sender_btn = getattr(self, 'btn_raster_add_ground' if is_ground else 'btn_raster_add_sky', None)
                if sender_btn and sender_btn.isVisible():
                    pos = sender_btn.mapToGlobal(sender_btn.rect().bottomLeft())
                else:
                    pos = QCursor.pos()
                chosen = menu.exec(pos)
                if chosen is None:
                    return

            if act_master and chosen == act_master:
                chosen_sky_mask = existing_master
            elif act_base and chosen == act_base:
                chosen_sky_mask, _ = self._detect_horizon_from_image(base_img)
                detected_fresh = True
            elif act_raster and chosen == act_raster:
                chosen_sky_mask, _ = self._detect_horizon_from_image(raster_img)
                detected_fresh = True
            else:
                return

        if chosen_sky_mask is None:
            return

        if chosen_sky_mask.ndim == 3:
            chosen_sky_mask = chosen_sky_mask[..., 0]

        h_l, w_l = self._get_layer_image_shape(layer)
        if chosen_sky_mask.shape[:2] != (h_l, w_l):
            chosen_sky_mask = cv2.resize(chosen_sky_mask, (w_l, h_l), interpolation=cv2.INTER_LINEAR)

        final_mask = (1.0 - chosen_sky_mask) if is_ground else chosen_sky_mask
        final_mask = np.ascontiguousarray(np.clip(final_mask, 0.0, 1.0), dtype=np.float32)

        layer.set_brush_mask(final_mask)
        layer.active_target = "mask"
        layer.brush_mode = "erase"
        self._last_brush_mode = "erase"
        self.btn_b_erase.setChecked(True)
        self.btn_b_paint.setChecked(False)

        if detected_fresh and not has_existing_master:
            self.master_mask_updated.emit(chosen_sky_mask)

        self.update_for_layer(layer)
        self.brush_tool_toggled.emit(True, layer)
        self.mask_param_changed.emit()
        self.mask_structure_changed.emit()
        self.action_committed.emit()

    def _add_raster_layer_mask_ground(self, source_choice: Optional[str] = None) -> None:
        self._apply_raster_horizon_mask(is_ground=True, source_choice=source_choice)

    def _add_raster_layer_mask_sky(self, source_choice: Optional[str] = None) -> None:
        self._apply_raster_horizon_mask(is_ground=False, source_choice=source_choice)

    def _remove_raster_layer_mask(self) -> None:
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_IMAGE_RASTER:
            return
        layer.set_brush_mask(None)
        layer.active_target = "image"
        self.update_for_layer(layer)
        self.brush_tool_toggled.emit(False, layer)
        self.mask_param_changed.emit()
        self.mask_structure_changed.emit()
        self.action_committed.emit()

    # --- Luminancia ---
    def _on_luma_param_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_LUMA:
            layer.luma_min = self.spin_luma_min.value()
            layer.luma_max = self.spin_luma_max.value()
            layer.feather = self.spin_luma_feather.value()
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.mask_param_changed.emit()

    def _set_luma_preset(self, l_min: float, l_max: float) -> None:
        self.spin_luma_min.setValue(l_min)
        self.spin_luma_max.setValue(l_max)
        self.interaction_finished.emit()

    # --- Color HSV ---
    def _on_hue_slider_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_hue_val.setText(f"{val}°")
        self._update_color_swatch(val)
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_COLOR:
            layer.color_hue = float(val)
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.mask_param_changed.emit()

    def _on_color_param_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_COLOR:
            layer.color_tolerance = float(self.spin_color_tol.value())
            layer.color_sat_min = float(self.spin_color_sat.value())
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.mask_param_changed.emit()

    def _set_color_preset(self, hue: int, tol: int) -> None:
        self.slider_hue.setValue(hue)
        self.spin_color_tol.setValue(tol)
        self.interaction_finished.emit()

    def _update_color_swatch(self, hue: int) -> None:
        col = QColor.fromHsv(int(round(hue)) % 360, 240, 240)
        self.lbl_color_swatch.setStyleSheet(
            f"background-color: {col.name()}; border: 1px solid #ffffff; border-radius: 3px;"
        )

    def _on_pick_color_toggled(self, checked: bool) -> None:
        self.color_eyedropper_requested.emit(checked)

    def set_color_picker_active(self, active: bool) -> None:
        if hasattr(self, 'btn_pick_color'):
            self.btn_pick_color.blockSignals(True)
            self.btn_pick_color.setChecked(active)
            self.btn_pick_color.blockSignals(False)

    def apply_sampled_color(self, h: float, s: float) -> None:
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_COLOR:
            return
        self._updating_ui = True
        hue_deg = int(round(h * 360.0)) % 360
        self.slider_hue.setValue(hue_deg)
        self.lbl_hue_val.setText(f"{hue_deg}°")
        self._update_color_swatch(hue_deg)
        sat_val = max(0.01, min(0.90, round(s * 0.75, 2)))
        self.spin_color_sat.setValue(sat_val)
        layer.color_hue = float(hue_deg)
        layer.color_sat_min = float(sat_val)
        self._updating_ui = False
        self.set_color_picker_active(False)
        self.mask_param_changed.emit()
        self.action_committed.emit()
        if self.btn_preview_mask.isChecked():
            self._emit_mask_preview()

    # --- Gradientes ---
    def _on_grad_limit_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or not getattr(layer, 'is_gradient', False):
            return
        layer.limit_mask = self.combo_grad_limit.currentData()
        self.mask_param_changed.emit()
        self.action_committed.emit()

    def _on_linear_profile_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_LINEAR:
            return
        layer.linear_profile = self.combo_lin_profile.currentData()
        is_refl = layer.linear_profile == "reflected"
        self.row_lin_midpoint.setVisible(not is_refl)
        self.container_reflected_props.setVisible(is_refl)
        self.row_lin_span_b.setVisible(is_refl and not getattr(layer, 'linear_symmetric', True))
        self.mask_param_changed.emit()
        self.action_committed.emit()

    def _on_linear_symmetric_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_LINEAR:
            return
        layer.linear_symmetric = checked
        self.row_lin_span_b.setVisible(not checked)
        self.mask_param_changed.emit()
        self.action_committed.emit()

    def _on_linear_span_b_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_LINEAR:
            return
        layer.linear_span_b = float(self.slider_lin_span_b.value()) / 100.0
        self.mask_param_changed.emit()

    def _on_linear_midpoint_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_LINEAR:
            return
        layer.linear_midpoint = float(self.slider_lin_midpoint.value()) / 100.0
        self.mask_param_changed.emit()

    def _on_radial_feather_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_RADIAL:
            return
        layer.radial_feather = float(self.slider_rad_feather.value()) / 100.0
        self.mask_param_changed.emit()

    def _on_radial_angle_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_RADIAL:
            return
        layer.radial_angle = float(self.slider_rad_angle.value())
        self.mask_param_changed.emit()

    def _on_radial_rx_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_RADIAL:
            return
        layer.radial_rx = float(self.slider_rad_rx.value()) / 100.0
        self.mask_param_changed.emit()

    def _on_radial_ry_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if not layer or layer.mask_type != MASK_TYPE_GRADIENT_RADIAL:
            return
        layer.radial_ry = float(self.slider_rad_ry.value()) / 100.0
        self.mask_param_changed.emit()

    def _reset_gradient_geometry(self) -> None:
        layer = self.get_selected_layer()
        if not layer or not getattr(layer, 'is_gradient', False):
            return
        if layer.mask_type == MASK_TYPE_GRADIENT_LINEAR:
            layer.linear_start = (0.5, 0.8)
            layer.linear_end = (0.5, 0.2)
            layer.linear_midpoint = 0.5
            layer.linear_profile = "smoothstep"
            layer.linear_symmetric = True
            layer.linear_span_b = 1.0
        elif layer.mask_type == MASK_TYPE_GRADIENT_RADIAL:
            layer.radial_center = (0.5, 0.5)
            layer.radial_rx = 0.25
            layer.radial_ry = 0.35
            layer.radial_angle = 0.0
            layer.radial_feather = 0.5
        self.update_gradient_ui_from_layer(layer)
        self.mask_param_changed.emit()
        self.action_committed.emit()
        self.gradient_tool_toggled.emit(True, layer)

    def update_gradient_ui_from_layer(self, layer: MaskLayer) -> None:
        is_lin = layer.mask_type == MASK_TYPE_GRADIENT_LINEAR
        is_rad = layer.mask_type == MASK_TYPE_GRADIENT_RADIAL
        self.container_linear_props.setVisible(is_lin)
        self.container_radial_props.setVisible(is_rad)

        idx = self.combo_grad_limit.findData(getattr(layer, 'limit_mask', LIMIT_MASK_NONE))
        if idx >= 0:
            self.combo_grad_limit.setCurrentIndex(idx)

        if is_lin:
            prof = getattr(layer, 'linear_profile', "smoothstep")
            idx_p = self.combo_lin_profile.findData(prof)
            if idx_p >= 0:
                self.combo_lin_profile.setCurrentIndex(idx_p)
            is_refl = prof == "reflected"
            self.row_lin_midpoint.setVisible(not is_refl)
            self.container_reflected_props.setVisible(is_refl)
            symm = getattr(layer, 'linear_symmetric', True)
            self.chk_lin_symmetric.setChecked(symm)
            self.row_lin_span_b.setVisible(is_refl and not symm)
            self.slider_lin_midpoint.setValue(int(round(getattr(layer, 'linear_midpoint', 0.5) * 100)))
            self.slider_lin_span_b.setValue(int(round(getattr(layer, 'linear_span_b', 1.0) * 100)))
        elif is_rad:
            self.slider_rad_feather.setValue(int(round(getattr(layer, 'radial_feather', 0.5) * 100)))
            self.slider_rad_angle.setValue(int(round(getattr(layer, 'radial_angle', 0.0))))
            self.slider_rad_rx.setValue(int(round(getattr(layer, 'radial_rx', 0.25) * 100)))
            self.slider_rad_ry.setValue(int(round(getattr(layer, 'radial_ry', 0.35) * 100)))

    # --- Refinar y Previsualización ---
    def _on_layer_refine_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.refine_edges = checked
            self.slider_refine_strength.setEnabled(checked)
            self.mask_param_changed.emit()
            self.action_committed.emit()
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()

    def _on_layer_refine_strength_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_refine_val.setText(f"{val}%")
        layer = self.get_selected_layer()
        if layer:
            layer.refine_strength = val / 100.0
            self.mask_param_changed.emit()
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()

    def _on_preview_mask_toggled(self, checked: bool) -> None:
        self.preview_mask_toggled.emit(checked)
        if checked:
            self._emit_mask_preview()
        else:
            self.preview_mask_requested.emit(None)

    def _emit_mask_preview(self) -> None:
        layer = self.get_selected_layer()
        if not layer or not layer.enabled or layer.is_system_layer or not self._image_reference_getter:
            self.preview_mask_requested.emit(None)
            return

        if layer.mask_type == MASK_TYPE_BRUSH and (layer.brush_mask is None or np.max(layer.brush_mask) <= 1e-4):
            self.preview_mask_requested.emit(None)
            return

        base_img = self._image_reference_getter()
        if base_img is None:
            self.preview_mask_requested.emit(None)
            return

        master_mask = self._master_mask_getter() if self._master_mask_getter else None
        stars_img = self._stars_getter() if self._stars_getter else None
        starless_img = self._starless_getter() if self._starless_getter else None
        layers = self._layers_getter() if self._layers_getter else [layer]

        canvas_below = get_canvas_below_layer(
            base_img,
            layers,
            layer.id,
            master_sky_mask=master_mask,
            starless_image=starless_img,
            stars_image=stars_img
        )

        m = compute_effective_layer_mask(
            layer, canvas_below,
            master_sky_mask=master_mask,
            starless_image=starless_img,
            stars_image=stars_img
        )
        if m is None or np.max(m) <= 1e-4:
            self.preview_mask_requested.emit(None)
        else:
            self.preview_mask_requested.emit(m)
