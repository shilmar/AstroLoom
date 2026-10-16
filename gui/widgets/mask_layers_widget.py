# gui/widgets/mask_layers_widget.py
"""
gui/widgets/mask_layers_widget.py - Panel interactivo de gestión de capas de máscaras y ajustes locales.
Permite crear, ordenar, editar y previsualizar múltiples máscaras (Pincel, Luminancia, Color y Suelo/Cielo),
con ajustes locales paramétricos independientes y conciencia de capas StarNet.
"""
from typing import Optional, List, Dict, Any, Callable
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QSlider, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QLineEdit, QGroupBox,
    QFrame, QSizePolicy, QMessageBox
)

from core.mask_layers import (
    MaskLayer, compute_layer_mask, compute_effective_layer_mask, get_canvas_below_layer,
    MASK_TYPE_BRUSH, MASK_TYPE_GROUND, MASK_TYPE_SKY, MASK_TYPE_LUMA, MASK_TYPE_COLOR, MASK_TYPE_CUSTOM,
    LAYER_KIND_ADJUSTMENT, LAYER_KIND_IMAGE,
    MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS, MASK_TYPE_IMAGE_GROUND,
    SCOPE_TARGET_ALL, SCOPE_TARGET_GROUND, SCOPE_TARGET_STARLESS, SCOPE_TARGET_STARS
)
from gui.theme import (
    BG_PANEL, BORDER_BASE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    TEXT_CYAN, ACCENT_CYAN
)


class MaskLayersWidget(QWidget):
    """
    Widget contenedor para la gestión de capas de máscaras y ajustes locales.
    """
    layers_changed = Signal()                    # Emitido al modificar cualquier capa o parámetro
    preview_mask_requested = Signal(object)       # Emitido con mask (np.ndarray) o None para el lienzo
    edit_ground_mask_requested = Signal()        # Solicitud de abrir el pincel interactivo de suelo
    ai_ground_mask_requested = Signal()          # Solicitud de auto-detección de horizonte con IA
    brush_tool_toggled = Signal(bool, object)    # (activo, MaskLayer) para activar el pincel en el lienzo
    brush_params_changed = Signal(int, float, float, str) # (radio, dureza, flujo, modo)
    color_eyedropper_requested = Signal(bool)     # Solicitud de pipeta para muestrear color en el lienzo
    starnet_opacity_changed = Signal(float)       # (opacidad 0.0 - 1.0) para sincronizar con slider_stars de StarNet
    starnet_visibility_changed = Signal(bool)     # (visibilidad) para sincronizar con combo_layer de StarNet
    ground_ev_changed = Signal(int)              # (val) sincronizar slider_gnd_ev (-200 a 300)
    ground_shadows_changed = Signal(int)         # (val) sincronizar slider_gnd_sh (0 a 100)
    ground_bp_changed = Signal(int)              # (val) sincronizar slider_gnd_bp (-50 a 100)
    ground_reset_requested = Signal()            # reset tono suelo
    ground_slider_released = Signal()            # commit de deshacer tras soltar deslizador de suelo

    # Sincronización contextual de Estrellas
    stars_intensity_changed = Signal(int)        # (val 0 a 200) sincronizar slider_stars
    stars_reset_requested = Signal()             # restablecer estrellas a 100%
    stars_diff_strength_changed = Signal(int)    # (val 0 a 100)
    stars_diff_radius_changed = Signal(int)      # (val 5 a 60)
    stars_diff_threshold_changed = Signal(int)   # (val 15 a 85)
    stars_diff_color_boost_changed = Signal(int) # (val 0 a 100)
    stars_diff_mode_changed = Signal(str)        # "screen", "soft_light", "additive"

    # Sincronización contextual de Fondo Nebular (Starless)
    starless_clarity_changed = Signal(int)       # (val -200 a 200)
    starless_dehaze_changed = Signal(int)        # (val -200 a 200)
    starless_ond_est_changed = Signal(int)       # (val -100 a 100)
    starless_ond_bg_changed = Signal(int)        # (val 0 a 100)
    starless_contrast_changed = Signal(int)      # (val -100 a 100)
    starless_denoise_changed = Signal(int)       # (val 0 a 100)
    starless_orton_strength_changed = Signal(int)   # (val 0 a 100)
    starless_orton_radius_changed = Signal(int)     # (val 10 a 150)
    starless_orton_threshold_changed = Signal(int)  # (val 0 a 50)
    starless_orton_tint_changed = Signal(str)       # "warm_gold", "neutral", "cool_blue", "halpha"
    starless_orton_mode_changed = Signal(str)       # "soft_light", "screen", "additive"
    starless_reset_requested = Signal()          # restablecer ajustes de fondo

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layers: List[MaskLayer] = []
        self._selected_layer_id: Optional[str] = None
        self._updating_ui: bool = False
        self._image_reference_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._master_mask_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._stars_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._starless_getter: Optional[Callable[[], Optional[np.ndarray]]] = None

        # Memoria de herramienta para Pincel y Máscaras
        self._last_brush_radius: int = 35
        self._last_brush_hardness: float = 0.50
        self._last_brush_flow: float = 0.80
        self._last_brush_mode: str = "paint"
        self._last_brush_opacity: float = 1.00

        self._setup_ui()

    def set_getters(
        self,
        img_getter: Callable[[], Optional[np.ndarray]],
        mask_getter: Callable[[], Optional[np.ndarray]],
        stars_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        starless_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
    ) -> None:
        """Configura los proveedores de imagen base, máscara maestra de suelo y capas StarNet."""
        self._image_reference_getter = img_getter
        self._master_mask_getter = mask_getter
        self._stars_getter = stars_getter
        self._starless_getter = starless_getter

    @staticmethod
    def _make_resettable_label(lbl: QLabel, reset_action, default_desc: str = "") -> QLabel:
        """Asigna comportamiento de doble clic para restablecer el control asociado."""
        lbl.setCursor(Qt.PointingHandCursor)
        tt = f"Doble clic para restablecer a {default_desc}" if default_desc else "Doble clic para restablecer"
        lbl.setToolTip(tt)
        lbl.mouseDoubleClickEvent = lambda e: reset_action()
        return lbl

    def set_preview_active(self, active: bool) -> None:
        """Sincroniza el botón de previsualización sin entrar en bucles de emisión redundantes."""
        if hasattr(self, 'btn_preview_mask') and self.btn_preview_mask.isChecked() != active:
            self.btn_preview_mask.setChecked(active)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # 1. Barra de creación rápida de máscaras
        row_create = QHBoxLayout()
        row_create.setSpacing(5)

        self.btn_add_brush = QPushButton("🖌️ Pincel")
        self.btn_add_brush.setCursor(Qt.PointingHandCursor)
        self.btn_add_brush.setToolTip("Crea una capa con máscara pintada a mano con dureza y opacidad de trazo")
        self.btn_add_brush.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_BRUSH))
        row_create.addWidget(self.btn_add_brush)

        self.btn_add_luma = QPushButton("➕ Luminancia")
        self.btn_add_luma.setCursor(Qt.PointingHandCursor)
        self.btn_add_luma.setToolTip("Crea una máscara basada en el rango de brillo (Sombras, Medios, Luces)")
        self.btn_add_luma.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_LUMA))
        row_create.addWidget(self.btn_add_luma)

        self.btn_add_color = QPushButton("➕ Color (HSV)")
        self.btn_add_color.setCursor(Qt.PointingHandCursor)
        self.btn_add_color.setToolTip("Crea una máscara basada en tono cromático (H-Alfa, OIII, Sodio, etc.)")
        self.btn_add_color.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_COLOR))
        row_create.addWidget(self.btn_add_color)

        self.btn_add_gnd = QPushButton("➕ Suelo / Cielo")
        self.btn_add_gnd.setCursor(Qt.PointingHandCursor)
        self.btn_add_gnd.setToolTip("Crea una capa vinculada a la segmentación del terreno")
        self.btn_add_gnd.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_GROUND))
        row_create.addWidget(self.btn_add_gnd)

        main_layout.addLayout(row_create)

        # 2. Lista de capas de máscaras
        self.list_layers = QListWidget()
        self.list_layers.setFixedHeight(120)
        self.list_layers.setStyleSheet("""
            QListWidget {
                background-color: #12151e;
                border: 1px solid #282e42;
                border-radius: 5px;
                padding: 2px;
            }
            QListWidget::item {
                background-color: #1a1e2b;
                border: 1px solid #2a3147;
                border-radius: 4px;
                padding: 4px 6px;
                margin-bottom: 2px;
                color: #e0e6f0;
            }
            QListWidget::item:selected {
                background-color: #1a3a5a;
                border-color: #00acc1;
                color: #ffffff;
            }
            QListWidget::item:hover {
                border-color: #4fc3f7;
            }
        """)
        self.list_layers.currentRowChanged.connect(self._on_layer_selection_changed)
        main_layout.addWidget(self.list_layers)

        # 3. Acciones sobre la capa seleccionada (Subir, Bajar, Duplicar, Eliminar)
        row_actions = QHBoxLayout()
        row_actions.setSpacing(6)

        self.btn_move_up = QPushButton("▲ Subir")
        self.btn_move_up.setToolTip("Mueve la capa seleccionada hacia arriba en la pila")
        self.btn_move_up.clicked.connect(self._move_selected_up)
        row_actions.addWidget(self.btn_move_up)

        self.btn_move_down = QPushButton("▼ Bajar")
        self.btn_move_down.setToolTip("Mueve la capa seleccionada hacia abajo en la pila")
        self.btn_move_down.clicked.connect(self._move_selected_down)
        row_actions.addWidget(self.btn_move_down)

        self.btn_duplicate = QPushButton("📋 Duplicar")
        self.btn_duplicate.setToolTip("Duplica la capa seleccionada con todos sus parámetros")
        self.btn_duplicate.clicked.connect(self._duplicate_selected)
        row_actions.addWidget(self.btn_duplicate)

        self.btn_delete = QPushButton("🗑️ Eliminar")
        self.btn_delete.setToolTip("Elimina la capa seleccionada")
        self.btn_delete.setStyleSheet("color: #ef9a9a;")
        self.btn_delete.clicked.connect(self._delete_selected)
        row_actions.addWidget(self.btn_delete)

        main_layout.addLayout(row_actions)

        # 4. Contenedor de edición de la capa seleccionada
        self.container_layer_props = QWidget()
        props_layout = QVBoxLayout(self.container_layer_props)
        props_layout.setContentsMargins(0, 4, 0, 0)
        props_layout.setSpacing(8)

        # Fila A: Nombre + Activa + Invertir
        row_name_flags = QHBoxLayout()
        self.txt_layer_name = QLineEdit()
        self.txt_layer_name.setPlaceholderText("Nombre de la máscara...")
        self.txt_layer_name.textChanged.connect(self._on_layer_name_changed)
        row_name_flags.addWidget(self.txt_layer_name, stretch=1)

        self.chk_layer_enabled = QCheckBox("Activa")
        self.chk_layer_enabled.setChecked(True)
        self.chk_layer_enabled.toggled.connect(self._on_layer_enabled_toggled)
        row_name_flags.addWidget(self.chk_layer_enabled)

        self.chk_layer_invert = QCheckBox("Invertir")
        self.chk_layer_invert.setToolTip("Invierte la selección (1.0 - máscara)")
        self.chk_layer_invert.toggled.connect(self._on_layer_invert_toggled)
        row_name_flags.addWidget(self.chk_layer_invert)

        props_layout.addLayout(row_name_flags)


        # Fila C: Opacidad de la capa
        row_op = QHBoxLayout()
        row_op.addWidget(self._make_resettable_label(QLabel("Opacidad:"), lambda: self.slider_opacity.setValue(100), "100%"))
        self.lbl_opacity = QLabel("100%")
        self.lbl_opacity.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_op.addWidget(self.lbl_opacity)
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        row_op.addWidget(self.slider_opacity, stretch=1)
        props_layout.addLayout(row_op)

        # 5. Caja dinámica de definición de máscara (Pincel / Luminancia / Color / Suelo)
        self.grp_mask_def = QGroupBox("Definición de Máscara")
        self.grp_mask_def.setObjectName("mask_def_group")
        self.layout_mask_def = QVBoxLayout(self.grp_mask_def)
        self.layout_mask_def.setContentsMargins(8, 10, 8, 8)
        self.layout_mask_def.setSpacing(6)

        # Sub-panel Pincel (Ajuste Local con Dureza y Opacidad)
        self.panel_brush = QWidget()
        brush_layout = QVBoxLayout(self.panel_brush)
        brush_layout.setContentsMargins(0, 0, 0, 0)
        brush_layout.setSpacing(6)

        # Fila Radio / Tamaño
        row_b_radius = QHBoxLayout()
        row_b_radius.addWidget(self._make_resettable_label(QLabel("Radio (Tamaño):"), lambda: self.slider_b_radius.setValue(35), "35 px"))
        self.lbl_b_radius = QLabel("35 px")
        self.lbl_b_radius.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_b_radius.addWidget(self.lbl_b_radius)
        self.slider_b_radius = QSlider(Qt.Horizontal)
        self.slider_b_radius.setRange(5, 250)
        self.slider_b_radius.setValue(35)
        self.slider_b_radius.valueChanged.connect(self._on_brush_param_changed)
        row_b_radius.addWidget(self.slider_b_radius, stretch=1)
        brush_layout.addLayout(row_b_radius)

        # Fila Dureza (Hardness / Feather)
        row_b_hard = QHBoxLayout()
        row_b_hard.addWidget(self._make_resettable_label(QLabel("Dureza:"), lambda: self.slider_b_hardness.setValue(50), "50%"))
        self.lbl_b_hardness = QLabel("50%")
        self.lbl_b_hardness.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_b_hard.addWidget(self.lbl_b_hardness)
        self.slider_b_hardness = QSlider(Qt.Horizontal)
        self.slider_b_hardness.setRange(0, 100)
        self.slider_b_hardness.setValue(50)
        self.slider_b_hardness.valueChanged.connect(self._on_brush_param_changed)
        row_b_hard.addWidget(self.slider_b_hardness, stretch=1)
        brush_layout.addLayout(row_b_hard)

        # Fila Flujo / Opacidad por trazo
        row_b_flow = QHBoxLayout()
        row_b_flow.addWidget(self._make_resettable_label(QLabel("Flujo / Opacidad:"), lambda: self.slider_b_flow.setValue(80), "80%"))
        self.lbl_b_flow = QLabel("80%")
        self.lbl_b_flow.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_b_flow.addWidget(self.lbl_b_flow)
        self.slider_b_flow = QSlider(Qt.Horizontal)
        self.slider_b_flow.setRange(5, 100)
        self.slider_b_flow.setValue(80)
        self.slider_b_flow.valueChanged.connect(self._on_brush_param_changed)
        row_b_flow.addWidget(self.slider_b_flow, stretch=1)
        brush_layout.addLayout(row_b_flow)

        # Fila Modo de Trazo: Pintar (+) vs Borrar (-) y Limpiar
        row_b_mode = QHBoxLayout()
        self.btn_b_paint = QPushButton("🖌️ Pintar (+)")
        self.btn_b_paint.setCheckable(True)
        self.btn_b_paint.setChecked(True)
        self.btn_b_paint.setCursor(Qt.PointingHandCursor)
        self.btn_b_paint.clicked.connect(lambda: self._set_brush_mode("paint"))
        row_b_mode.addWidget(self.btn_b_paint)

        self.btn_b_erase = QPushButton("🧹 Borrar (-)")
        self.btn_b_erase.setCheckable(True)
        self.btn_b_erase.setCursor(Qt.PointingHandCursor)
        self.btn_b_erase.clicked.connect(lambda: self._set_brush_mode("erase"))
        row_b_mode.addWidget(self.btn_b_erase)

        self.btn_b_clear = QPushButton("🗑️ Limpiar")
        self.btn_b_clear.setCursor(Qt.PointingHandCursor)
        self.btn_b_clear.setStyleSheet("color: #ef9a9a;")
        self.btn_b_clear.clicked.connect(self._clear_brush_mask)
        row_b_mode.addWidget(self.btn_b_clear)

        brush_layout.addLayout(row_b_mode)
        self.layout_mask_def.addWidget(self.panel_brush)

        # Sub-panel Luminancia
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
        row_luma_thresh.addWidget(self.spin_luma_min)

        row_luma_thresh.addWidget(QLabel("Máx:"))
        self.spin_luma_max = QDoubleSpinBox()
        self.spin_luma_max.setRange(0.0, 1.0)
        self.spin_luma_max.setSingleStep(0.05)
        self.spin_luma_max.setValue(1.0)
        self.spin_luma_max.valueChanged.connect(self._on_luma_param_changed)
        row_luma_thresh.addWidget(self.spin_luma_max)

        row_luma_thresh.addWidget(QLabel("Borde:"))
        self.spin_luma_feather = QDoubleSpinBox()
        self.spin_luma_feather.setRange(0.001, 0.40)
        self.spin_luma_feather.setSingleStep(0.02)
        self.spin_luma_feather.setValue(0.08)
        self.spin_luma_feather.valueChanged.connect(self._on_luma_param_changed)
        row_luma_thresh.addWidget(self.spin_luma_feather)

        luma_layout.addLayout(row_luma_thresh)

        # Presets rápidos de luminancia
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

        # Sub-panel Color (HSV)
        self.panel_color = QWidget()
        color_layout = QVBoxLayout(self.panel_color)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(6)

        row_hue = QHBoxLayout()
        row_hue.addWidget(self._make_resettable_label(QLabel("Matiz (Hue):"), lambda: self.slider_hue.setValue(0), "0°"))
        self.lbl_hue_val = QLabel("0°")
        self.lbl_hue_val.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_hue.addWidget(self.lbl_hue_val)

        # Muestra visual de color (swatch)
        self.lbl_color_swatch = QLabel()
        self.lbl_color_swatch.setFixedSize(22, 16)
        self.lbl_color_swatch.setStyleSheet("background-color: #ff0000; border: 1px solid #ffffff; border-radius: 3px;")
        row_hue.addWidget(self.lbl_color_swatch)

        self.slider_hue = QSlider(Qt.Horizontal)
        self.slider_hue.setRange(0, 360)
        self.slider_hue.setValue(0)
        self.slider_hue.valueChanged.connect(self._on_hue_slider_changed)
        row_hue.addWidget(self.slider_hue, stretch=1)
        color_layout.addLayout(row_hue)

        # Fila con Pipeta Muestreadora de Color
        row_pick = QHBoxLayout()
        self.btn_pick_color = QPushButton("🎯 Muestrear Color en Imagen")
        self.btn_pick_color.setCheckable(True)
        self.btn_pick_color.setCursor(Qt.PointingHandCursor)
        self.btn_pick_color.setToolTip(
            "Activa la pipeta para hacer clic sobre cualquier zona de la imagen en el visor\n"
            "y definir automáticamente el Matiz (Hue) y Saturación de esta máscara."
        )
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
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #ffd54f;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #d97706;
                color: #ffffff;
                border: 1px solid #f59e0b;
                font-weight: bold;
            }
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
        row_tol_sat.addWidget(self.spin_color_tol)

        row_tol_sat.addWidget(QLabel("Sat. Mín:"))
        self.spin_color_sat = QDoubleSpinBox()
        self.spin_color_sat.setRange(0.01, 0.90)
        self.spin_color_sat.setSingleStep(0.05)
        self.spin_color_sat.setValue(0.08)
        self.spin_color_sat.valueChanged.connect(self._on_color_param_changed)
        row_tol_sat.addWidget(self.spin_color_sat)

        color_layout.addLayout(row_tol_sat)

        # Presets astronómicos de color
        row_color_presets = QHBoxLayout()
        btn_pre_halpha = QPushButton("🔴 H-Alfa (0°)")
        btn_pre_halpha.clicked.connect(lambda: self._set_color_preset(0, 30))
        row_color_presets.addWidget(btn_pre_halpha)

        btn_pre_oiii = QPushButton("🔵 OIII (190°)")
        btn_pre_oiii.clicked.connect(lambda: self._set_color_preset(190, 35))
        row_color_presets.addWidget(btn_pre_oiii)

        btn_pre_sodium = QPushButton("🟠 Sodio (38°)")
        btn_pre_sodium.clicked.connect(lambda: self._set_color_preset(38, 25))
        row_color_presets.addWidget(btn_pre_sodium)

        btn_pre_airglow = QPushButton("🟢 Airglow (115°)")
        btn_pre_airglow.clicked.connect(lambda: self._set_color_preset(115, 30))
        row_color_presets.addWidget(btn_pre_airglow)

        color_layout.addLayout(row_color_presets)
        self.layout_mask_def.addWidget(self.panel_color)

        # Sub-panel Suelo / Cielo
        self.panel_ground = QWidget()
        gnd_layout = QVBoxLayout(self.panel_ground)
        gnd_layout.setContentsMargins(0, 0, 0, 0)
        gnd_layout.setSpacing(6)

        row_gnd_info = QHBoxLayout()
        self.lbl_gnd_desc = QLabel("Vinculada a la máscara de segmentación de suelo.")
        self.lbl_gnd_desc.setStyleSheet("color: #a5d6a7; font-size: 11px; font-style: italic;")
        row_gnd_info.addWidget(self.lbl_gnd_desc, stretch=1)

        self.btn_ai_gnd_mask = QPushButton("🧠 Detección IA")
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
        self.btn_ai_gnd_mask.setToolTip("Detecta automáticamente el horizonte con IA ligera U²-Net (1-Click)")
        self.btn_ai_gnd_mask.clicked.connect(self.ai_ground_mask_requested.emit)
        row_gnd_info.addWidget(self.btn_ai_gnd_mask)

        self.btn_edit_gnd_mask = QPushButton("🖌️ Pintar Trazos...")
        self.btn_edit_gnd_mask.setCursor(Qt.PointingHandCursor)
        self.btn_edit_gnd_mask.clicked.connect(self.edit_ground_mask_requested.emit)
        row_gnd_info.addWidget(self.btn_edit_gnd_mask)
        gnd_layout.addLayout(row_gnd_info)

        self.layout_mask_def.addWidget(self.panel_ground)

        # Botón de visualización en lienzo
        self.btn_preview_mask = QPushButton("👁️ Previsualizar Máscara en Lienzo")
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

        props_layout.addWidget(self.grp_mask_def)

        # 6. Controles de Ajustes Locales de Revelado
        self.grp_local_adj = QGroupBox("Ajustes Locales de Revelado")
        self.grp_local_adj.setObjectName("local_adj_group")
        adj_layout = QVBoxLayout(self.grp_local_adj)
        adj_layout.setContentsMargins(8, 12, 8, 8)
        adj_layout.setSpacing(6)

        # Exposición (EV)
        row_adj_ev = QHBoxLayout()
        row_adj_ev.addWidget(self._make_resettable_label(QLabel("Exposición:"), lambda: self.slider_adj_ev.setValue(0), "0.00 EV"))
        self.lbl_adj_ev = QLabel("0.00 EV")
        self.lbl_adj_ev.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_adj_ev.addWidget(self.lbl_adj_ev)
        self.slider_adj_ev = QSlider(Qt.Horizontal)
        self.slider_adj_ev.setRange(-300, 300)
        self.slider_adj_ev.setValue(0)
        self.slider_adj_ev.valueChanged.connect(self._on_adj_ev_changed)
        row_adj_ev.addWidget(self.slider_adj_ev, stretch=1)
        adj_layout.addLayout(row_adj_ev)

        # Contraste y Sombras (en 2 columnas compactas)
        row_pair1 = QHBoxLayout()
        row_pair1.setSpacing(10)

        col_c = QVBoxLayout()
        col_c.setSpacing(2)
        row_c_lbl = QHBoxLayout()
        row_c_lbl.addWidget(self._make_resettable_label(QLabel("Contraste:"), lambda: self.slider_adj_contrast.setValue(0), "0.00"))
        self.lbl_adj_contrast = QLabel("0.00")
        self.lbl_adj_contrast.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_c_lbl.addWidget(self.lbl_adj_contrast)
        col_c.addLayout(row_c_lbl)
        self.slider_adj_contrast = QSlider(Qt.Horizontal)
        self.slider_adj_contrast.setRange(-100, 100)
        self.slider_adj_contrast.setValue(0)
        self.slider_adj_contrast.valueChanged.connect(self._on_adj_contrast_changed)
        col_c.addWidget(self.slider_adj_contrast)
        row_pair1.addLayout(col_c)

        col_sh = QVBoxLayout()
        col_sh.setSpacing(2)
        row_sh_lbl = QHBoxLayout()
        row_sh_lbl.addWidget(self._make_resettable_label(QLabel("Sombras:"), lambda: self.slider_adj_shadows.setValue(0), "0.00"))
        self.lbl_adj_shadows = QLabel("0.00")
        self.lbl_adj_shadows.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_sh_lbl.addWidget(self.lbl_adj_shadows)
        col_sh.addLayout(row_sh_lbl)
        self.slider_adj_shadows = QSlider(Qt.Horizontal)
        self.slider_adj_shadows.setRange(-100, 100)
        self.slider_adj_shadows.setValue(0)
        self.slider_adj_shadows.valueChanged.connect(self._on_adj_shadows_changed)
        col_sh.addWidget(self.slider_adj_shadows)
        row_pair1.addLayout(col_sh)

        adj_layout.addLayout(row_pair1)

        # Luces y Saturación (en 2 columnas compactas)
        row_pair2 = QHBoxLayout()
        row_pair2.setSpacing(10)

        col_hl = QVBoxLayout()
        col_hl.setSpacing(2)
        row_hl_lbl = QHBoxLayout()
        row_hl_lbl.addWidget(self._make_resettable_label(QLabel("Luces:"), lambda: self.slider_adj_highlights.setValue(0), "0.00"))
        self.lbl_adj_highlights = QLabel("0.00")
        self.lbl_adj_highlights.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_hl_lbl.addWidget(self.lbl_adj_highlights)
        col_hl.addLayout(row_hl_lbl)
        self.slider_adj_highlights = QSlider(Qt.Horizontal)
        self.slider_adj_highlights.setRange(-100, 100)
        self.slider_adj_highlights.setValue(0)
        self.slider_adj_highlights.valueChanged.connect(self._on_adj_highlights_changed)
        col_hl.addWidget(self.slider_adj_highlights)
        row_pair2.addLayout(col_hl)

        col_sat = QVBoxLayout()
        col_sat.setSpacing(2)
        row_sat_lbl = QHBoxLayout()
        row_sat_lbl.addWidget(self._make_resettable_label(QLabel("Saturación:"), lambda: self.slider_adj_sat.setValue(100), "1.00x"))
        self.lbl_adj_sat = QLabel("1.00x")
        self.lbl_adj_sat.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_sat_lbl.addWidget(self.lbl_adj_sat)
        col_sat.addLayout(row_sat_lbl)
        self.slider_adj_sat = QSlider(Qt.Horizontal)
        self.slider_adj_sat.setRange(0, 250)
        self.slider_adj_sat.setValue(100)
        self.slider_adj_sat.valueChanged.connect(self._on_adj_sat_changed)
        col_sat.addWidget(self.slider_adj_sat)
        row_pair2.addLayout(col_sat)

        adj_layout.addLayout(row_pair2)

        # Temperatura y Tinte (WB local de alta precisión)
        row_pair3 = QHBoxLayout()
        row_pair3.setSpacing(10)

        # Temp WB
        col_t = QVBoxLayout()
        col_t.setSpacing(2)
        row_t_lbl = QHBoxLayout()
        lbl_t_title = QLabel("Temp WB:")
        lbl_t_title.setToolTip("Doble clic para restablecer a 0.0000")
        lbl_t_title.mouseDoubleClickEvent = lambda e: self.spin_adj_temp.setValue(0.0)
        row_t_lbl.addWidget(lbl_t_title)

        self.spin_adj_temp = QDoubleSpinBox()
        self.spin_adj_temp.setDecimals(4)
        self.spin_adj_temp.setRange(-1.0000, 1.0000)
        self.spin_adj_temp.setSingleStep(0.0050)
        self.spin_adj_temp.setValue(0.0000)
        self.spin_adj_temp.setFixedWidth(94)
        self.spin_adj_temp.setStyleSheet("color: #ffd54f; font-weight: bold;")
        self.spin_adj_temp.valueChanged.connect(self._on_spin_adj_temp_changed)
        row_t_lbl.addWidget(self.spin_adj_temp)
        col_t.addLayout(row_t_lbl)

        self.slider_adj_temp = QSlider(Qt.Horizontal)
        self.slider_adj_temp.setRange(-10000, 10000)
        self.slider_adj_temp.setValue(0)
        self.slider_adj_temp.valueChanged.connect(self._on_slider_adj_temp_changed)
        col_t.addWidget(self.slider_adj_temp)
        row_pair3.addLayout(col_t)

        # Tinte WB
        col_ti = QVBoxLayout()
        col_ti.setSpacing(2)
        row_ti_lbl = QHBoxLayout()
        lbl_ti_title = QLabel("Tinte WB:")
        lbl_ti_title.setToolTip("Doble clic para restablecer a 0.0000")
        lbl_ti_title.mouseDoubleClickEvent = lambda e: self.spin_adj_tint.setValue(0.0)
        row_ti_lbl.addWidget(lbl_ti_title)

        self.spin_adj_tint = QDoubleSpinBox()
        self.spin_adj_tint.setDecimals(4)
        self.spin_adj_tint.setRange(-1.0000, 1.0000)
        self.spin_adj_tint.setSingleStep(0.0050)
        self.spin_adj_tint.setValue(0.0000)
        self.spin_adj_tint.setFixedWidth(94)
        self.spin_adj_tint.setStyleSheet("color: #ce93d8; font-weight: bold;")
        self.spin_adj_tint.valueChanged.connect(self._on_spin_adj_tint_changed)
        row_ti_lbl.addWidget(self.spin_adj_tint)
        col_ti.addLayout(row_ti_lbl)

        self.slider_adj_tint = QSlider(Qt.Horizontal)
        self.slider_adj_tint.setRange(-10000, 10000)
        self.slider_adj_tint.setValue(0)
        self.slider_adj_tint.valueChanged.connect(self._on_slider_adj_tint_changed)
        col_ti.addWidget(self.slider_adj_tint)
        row_pair3.addLayout(col_ti)

        adj_layout.addLayout(row_pair3)

        # Botón restablecer ajustes de la capa
        btn_reset_local = QPushButton("↺ Restablecer Ajustes Locales")
        btn_reset_local.setCursor(Qt.PointingHandCursor)
        btn_reset_local.clicked.connect(self._reset_local_adjustments)
        adj_layout.addWidget(btn_reset_local)

        props_layout.addWidget(self.grp_local_adj)

        # 7. Controles Especiales de Tono de Suelo (Visibles SOLO cuando se selecciona una capa de Suelo)
        self.grp_gnd_special = QGroupBox("Ajustes Tonales Especiales de Suelo")
        self.grp_gnd_special.setObjectName("gnd_special_group")
        gnd_spec_layout = QVBoxLayout(self.grp_gnd_special)
        gnd_spec_layout.setContentsMargins(8, 12, 8, 8)
        gnd_spec_layout.setSpacing(6)

        lbl_gnd_notice = QLabel("Ajustes dedicados para la superficie terrestre.")
        lbl_gnd_notice.setStyleSheet("color: #a5d6a7; font-size: 11px; font-style: italic;")
        gnd_spec_layout.addWidget(lbl_gnd_notice)

        # Exposición Suelo
        row_gnd_sp_ev = QHBoxLayout()
        row_gnd_sp_ev.addWidget(self._make_resettable_label(QLabel("Exposición Suelo:"), lambda: self.slider_gnd_special_ev.setValue(0), "0.00 EV"))
        self.lbl_gnd_special_ev = QLabel("0.00 EV")
        self.lbl_gnd_special_ev.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_gnd_sp_ev.addWidget(self.lbl_gnd_special_ev)
        row_gnd_sp_ev.addSpacing(6)
        self.slider_gnd_special_ev = QSlider(Qt.Horizontal)
        self.slider_gnd_special_ev.setRange(-200, 300)
        self.slider_gnd_special_ev.setValue(0)
        self.slider_gnd_special_ev.valueChanged.connect(self._on_gnd_special_ev_changed)
        self.slider_gnd_special_ev.sliderReleased.connect(self.ground_slider_released.emit)
        row_gnd_sp_ev.addWidget(self.slider_gnd_special_ev, stretch=1)
        gnd_spec_layout.addLayout(row_gnd_sp_ev)

        # Par Sombras / Punto Negro en 2 columnas compactas
        row_gnd_sp_pair = QHBoxLayout()
        row_gnd_sp_pair.setSpacing(10)

        col_gnd_sp_sh = QVBoxLayout()
        col_gnd_sp_sh.setSpacing(2)
        row_gnd_sp_sh_lbl = QHBoxLayout()
        row_gnd_sp_sh_lbl.addWidget(self._make_resettable_label(QLabel("Sombras:"), lambda: self.slider_gnd_special_sh.setValue(0), "0%"))
        self.lbl_gnd_special_sh = QLabel("0%")
        self.lbl_gnd_special_sh.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_gnd_sp_sh_lbl.addWidget(self.lbl_gnd_special_sh)
        col_gnd_sp_sh.addLayout(row_gnd_sp_sh_lbl)
        self.slider_gnd_special_sh = QSlider(Qt.Horizontal)
        self.slider_gnd_special_sh.setRange(0, 100)
        self.slider_gnd_special_sh.setValue(0)
        self.slider_gnd_special_sh.valueChanged.connect(self._on_gnd_special_sh_changed)
        self.slider_gnd_special_sh.sliderReleased.connect(self.ground_slider_released.emit)
        col_gnd_sp_sh.addWidget(self.slider_gnd_special_sh)
        row_gnd_sp_pair.addLayout(col_gnd_sp_sh)

        col_gnd_sp_bp = QVBoxLayout()
        col_gnd_sp_bp.setSpacing(2)
        row_gnd_sp_bp_lbl = QHBoxLayout()
        row_gnd_sp_bp_lbl.addWidget(self._make_resettable_label(QLabel("Punto Negro:"), lambda: self.slider_gnd_special_bp.setValue(0), "0.000"))
        self.lbl_gnd_special_bp = QLabel("0.000")
        self.lbl_gnd_special_bp.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_gnd_sp_bp_lbl.addWidget(self.lbl_gnd_special_bp)
        col_gnd_sp_bp.addLayout(row_gnd_sp_bp_lbl)
        self.slider_gnd_special_bp = QSlider(Qt.Horizontal)
        self.slider_gnd_special_bp.setRange(-50, 100)
        self.slider_gnd_special_bp.setValue(0)
        self.slider_gnd_special_bp.valueChanged.connect(self._on_gnd_special_bp_changed)
        self.slider_gnd_special_bp.sliderReleased.connect(self.ground_slider_released.emit)
        col_gnd_sp_bp.addWidget(self.slider_gnd_special_bp)
        row_gnd_sp_pair.addLayout(col_gnd_sp_bp)

        gnd_spec_layout.addLayout(row_gnd_sp_pair)

        self.btn_reset_gnd_special = QPushButton("↺ Restablecer Tono Suelo")
        self.btn_reset_gnd_special.setCursor(Qt.PointingHandCursor)
        self.btn_reset_gnd_special.clicked.connect(self.ground_reset_requested.emit)
        gnd_spec_layout.addWidget(self.btn_reset_gnd_special)

        props_layout.addWidget(self.grp_gnd_special)

        # 8. Controles Especiales de Capa de Estrellas (Visibles SOLO en Capa de Estrellas)
        self.grp_stars_special = QGroupBox("Ajustes Especiales de Estrellas")
        self.grp_stars_special.setObjectName("stars_special_group")
        stars_spec_layout = QVBoxLayout(self.grp_stars_special)
        stars_spec_layout.setContentsMargins(8, 12, 8, 8)
        stars_spec_layout.setSpacing(6)

        lbl_stars_notice = QLabel("Ajuste de densidad y presencia estelar separada.")
        lbl_stars_notice.setStyleSheet("color: #80d8ff; font-size: 11px; font-style: italic;")
        stars_spec_layout.addWidget(lbl_stars_notice)

        row_stars_slider = QHBoxLayout()
        row_stars_slider.addWidget(self._make_resettable_label(QLabel("Intensidad Estrellas:"), lambda: self.slider_stars_special.setValue(100), "100%"))
        self.lbl_stars_special = QLabel("100%")
        self.lbl_stars_special.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_stars_slider.addWidget(self.lbl_stars_special)
        row_stars_slider.addSpacing(6)
        self.slider_stars_special = QSlider(Qt.Horizontal)
        self.slider_stars_special.setRange(0, 200)
        self.slider_stars_special.setValue(100)
        self.slider_stars_special.valueChanged.connect(self._on_stars_special_changed)
        self.slider_stars_special.sliderReleased.connect(self.ground_slider_released.emit)
        row_stars_slider.addWidget(self.slider_stars_special, stretch=1)
        stars_spec_layout.addLayout(row_stars_slider)

        # Filtro Difusor y Realce de Constelaciones (Star Mist)
        self.box_sl_star_diff = QGroupBox("🌟 Filtro Difusor y Realce de Constelaciones")
        self.box_sl_star_diff.setStyleSheet("""
            QGroupBox {
                border: 1px solid #fbc02d;
                border-radius: 6px;
                margin-top: 4px;
                padding-top: 8px;
                background-color: #221f14;
                font-weight: bold;
                color: #fff9c4;
            }
        """)
        sl_sd_layout = QVBoxLayout(self.box_sl_star_diff)
        sl_sd_layout.setContentsMargins(6, 4, 6, 6)
        sl_sd_layout.setSpacing(6)

        # Intensidad Difusor
        row_sd_int = QHBoxLayout()
        row_sd_int.addWidget(self._make_resettable_label(QLabel("Intensidad Difusor:"), lambda: self.slider_sl_star_diff_strength.setValue(0), "0%"))
        self.lbl_sl_star_diff_strength = QLabel("0%")
        self.lbl_sl_star_diff_strength.setStyleSheet("color: #ffd54f; font-weight: bold;")
        self.lbl_sl_star_diff_strength.setFixedWidth(34)
        row_sd_int.addWidget(self.lbl_sl_star_diff_strength)
        self.slider_sl_star_diff_strength = QSlider(Qt.Horizontal)
        self.slider_sl_star_diff_strength.setRange(0, 100)
        self.slider_sl_star_diff_strength.setValue(0)
        self.slider_sl_star_diff_strength.valueChanged.connect(self._on_sl_star_diff_strength_changed)
        self.slider_sl_star_diff_strength.sliderReleased.connect(self.ground_slider_released.emit)
        row_sd_int.addWidget(self.slider_sl_star_diff_strength, stretch=1)
        sl_sd_layout.addLayout(row_sd_int)

        # Fila 2: Radio (Halo) y Umbral de Magnitud
        row_sd_params = QHBoxLayout()
        row_sd_params.setSpacing(10)

        col_sd_rad = QVBoxLayout()
        col_sd_rad.setSpacing(2)
        row_sd_rad_lbl = QHBoxLayout()
        row_sd_rad_lbl.addWidget(self._make_resettable_label(QLabel("Radio (Halo):"), lambda: self.slider_sl_star_diff_radius.setValue(20), "20 px"))
        self.lbl_sl_star_diff_radius = QLabel("20 px")
        self.lbl_sl_star_diff_radius.setStyleSheet("color: #ffe082; font-weight: bold;")
        row_sd_rad_lbl.addWidget(self.lbl_sl_star_diff_radius)
        col_sd_rad.addLayout(row_sd_rad_lbl)
        self.slider_sl_star_diff_radius = QSlider(Qt.Horizontal)
        self.slider_sl_star_diff_radius.setRange(5, 60)
        self.slider_sl_star_diff_radius.setValue(20)
        self.slider_sl_star_diff_radius.valueChanged.connect(self._on_sl_star_diff_radius_changed)
        self.slider_sl_star_diff_radius.sliderReleased.connect(self.ground_slider_released.emit)
        col_sd_rad.addWidget(self.slider_sl_star_diff_radius)
        row_sd_params.addLayout(col_sd_rad)

        col_sd_thr = QVBoxLayout()
        col_sd_thr.setSpacing(2)
        row_sd_thr_lbl = QHBoxLayout()
        row_sd_thr_lbl.addWidget(self._make_resettable_label(QLabel("Umbral Magnitud:"), lambda: self.slider_sl_star_diff_threshold.setValue(45), "0.45"))
        self.lbl_sl_star_diff_threshold = QLabel("0.45")
        self.lbl_sl_star_diff_threshold.setStyleSheet("color: #ffe082; font-weight: bold;")
        row_sd_thr_lbl.addWidget(self.lbl_sl_star_diff_threshold)
        col_sd_thr.addLayout(row_sd_thr_lbl)
        self.slider_sl_star_diff_threshold = QSlider(Qt.Horizontal)
        self.slider_sl_star_diff_threshold.setRange(15, 85)
        self.slider_sl_star_diff_threshold.setValue(45)
        self.slider_sl_star_diff_threshold.valueChanged.connect(self._on_sl_star_diff_threshold_changed)
        self.slider_sl_star_diff_threshold.sliderReleased.connect(self.ground_slider_released.emit)
        col_sd_thr.addWidget(self.slider_sl_star_diff_threshold)
        row_sd_params.addLayout(col_sd_thr)
        sl_sd_layout.addLayout(row_sd_params)

        # Fila 3: Color Espectral y Modo de Fusión
        row_sd_combos = QHBoxLayout()
        row_sd_combos.setSpacing(8)

        col_sd_col = QVBoxLayout()
        col_sd_col.setSpacing(2)
        row_sd_col_lbl = QHBoxLayout()
        row_sd_col_lbl.addWidget(self._make_resettable_label(QLabel("Color Espectral:"), lambda: self.slider_sl_star_diff_color.setValue(50), "50%"))
        self.lbl_sl_star_diff_color = QLabel("50%")
        self.lbl_sl_star_diff_color.setStyleSheet("color: #ffe082; font-weight: bold;")
        row_sd_col_lbl.addWidget(self.lbl_sl_star_diff_color)
        col_sd_col.addLayout(row_sd_col_lbl)
        self.slider_sl_star_diff_color = QSlider(Qt.Horizontal)
        self.slider_sl_star_diff_color.setRange(0, 100)
        self.slider_sl_star_diff_color.setValue(50)
        self.slider_sl_star_diff_color.valueChanged.connect(self._on_sl_star_diff_color_changed)
        self.slider_sl_star_diff_color.sliderReleased.connect(self.ground_slider_released.emit)
        col_sd_col.addWidget(self.slider_sl_star_diff_color)
        row_sd_combos.addLayout(col_sd_col, stretch=1)

        col_sd_mode = QVBoxLayout()
        col_sd_mode.setSpacing(2)
        col_sd_mode.addWidget(QLabel("Fusión:"))
        self.combo_sl_star_diff_mode = QComboBox()
        self.combo_sl_star_diff_mode.addItem("Trama (Velo Etéreo)", "screen")
        self.combo_sl_star_diff_mode.addItem("Luz Suave (Aterciopelado)", "soft_light")
        self.combo_sl_star_diff_mode.addItem("Aditivo (Luminoso)", "additive")
        self.combo_sl_star_diff_mode.currentIndexChanged.connect(self._on_sl_star_diff_mode_changed)
        col_sd_mode.addWidget(self.combo_sl_star_diff_mode)
        row_sd_combos.addLayout(col_sd_mode, stretch=1)
        sl_sd_layout.addLayout(row_sd_combos)

        stars_spec_layout.addWidget(self.box_sl_star_diff)

        self.btn_reset_stars_special = QPushButton("↺ Restablecer Estrellas")
        self.btn_reset_stars_special.setCursor(Qt.PointingHandCursor)
        self.btn_reset_stars_special.clicked.connect(self.stars_reset_requested.emit)
        stars_spec_layout.addWidget(self.btn_reset_stars_special)

        props_layout.addWidget(self.grp_stars_special)

        # 9. Controles Especiales de Fondo Nebular (Visibles SOLO en Fondo Nebular Starless)
        self.grp_starless_special = QGroupBox("Ajustes Especiales de Fondo Nebular")
        self.grp_starless_special.setObjectName("starless_special_group")
        sl_layout = QVBoxLayout(self.grp_starless_special)
        sl_layout.setContentsMargins(8, 12, 8, 8)
        sl_layout.setSpacing(6)

        lbl_sl_notice = QLabel("Ajustes avanzados para resaltar estructuras nebulares y contraste galáctico.")
        lbl_sl_notice.setStyleSheet("color: #ce93d8; font-size: 11px; font-style: italic;")
        sl_layout.addWidget(lbl_sl_notice)

        # Par 1: Claridad Fondo y Borrar Neblina
        row_sl_p1 = QHBoxLayout()
        row_sl_p1.setSpacing(10)

        col_cl = QVBoxLayout()
        col_cl.setSpacing(2)
        row_cl_lbl = QHBoxLayout()
        row_cl_lbl.addWidget(self._make_resettable_label(QLabel("Claridad Fondo:"), lambda: self.slider_sl_clarity.setValue(0), "0.00"))
        self.lbl_sl_clarity = QLabel("0.00")
        self.lbl_sl_clarity.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_cl_lbl.addWidget(self.lbl_sl_clarity)
        col_cl.addLayout(row_cl_lbl)
        self.slider_sl_clarity = QSlider(Qt.Horizontal)
        self.slider_sl_clarity.setRange(-200, 200)
        self.slider_sl_clarity.setValue(0)
        self.slider_sl_clarity.valueChanged.connect(self._on_sl_clarity_changed)
        self.slider_sl_clarity.sliderReleased.connect(self.ground_slider_released.emit)
        col_cl.addWidget(self.slider_sl_clarity)
        row_sl_p1.addLayout(col_cl)

        col_dh = QVBoxLayout()
        col_dh.setSpacing(2)
        row_dh_lbl = QHBoxLayout()
        row_dh_lbl.addWidget(self._make_resettable_label(QLabel("Borrar Neblina:"), lambda: self.slider_sl_dehaze.setValue(0), "0.00"))
        self.lbl_sl_dehaze = QLabel("0.00")
        self.lbl_sl_dehaze.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_dh_lbl.addWidget(self.lbl_sl_dehaze)
        col_dh.addLayout(row_dh_lbl)
        self.slider_sl_dehaze = QSlider(Qt.Horizontal)
        self.slider_sl_dehaze.setRange(-200, 200)
        self.slider_sl_dehaze.setValue(0)
        self.slider_sl_dehaze.valueChanged.connect(self._on_sl_dehaze_changed)
        self.slider_sl_dehaze.sliderReleased.connect(self.ground_slider_released.emit)
        col_dh.addWidget(self.slider_sl_dehaze)
        row_sl_p1.addLayout(col_dh)
        sl_layout.addLayout(row_sl_p1)

        # Par 2: Estructura Ondículas y Atenuar Fondo
        row_sl_p2 = QHBoxLayout()
        row_sl_p2.setSpacing(10)

        col_ond = QVBoxLayout()
        col_ond.setSpacing(2)
        row_ond_lbl = QHBoxLayout()
        row_ond_lbl.addWidget(self._make_resettable_label(QLabel("Estructura:"), lambda: self.slider_sl_ond_est.setValue(0), "0.00"))
        self.lbl_sl_ond_est = QLabel("0.00")
        self.lbl_sl_ond_est.setStyleSheet("color: #b388ff; font-weight: bold;")
        row_ond_lbl.addWidget(self.lbl_sl_ond_est)
        col_ond.addLayout(row_ond_lbl)
        self.slider_sl_ond_est = QSlider(Qt.Horizontal)
        self.slider_sl_ond_est.setRange(-100, 100)
        self.slider_sl_ond_est.setValue(0)
        self.slider_sl_ond_est.valueChanged.connect(self._on_sl_ond_est_changed)
        self.slider_sl_ond_est.sliderReleased.connect(self.ground_slider_released.emit)
        col_ond.addWidget(self.slider_sl_ond_est)
        row_sl_p2.addLayout(col_ond)

        col_bg = QVBoxLayout()
        col_bg.setSpacing(2)
        row_bg_lbl = QHBoxLayout()
        row_bg_lbl.addWidget(self._make_resettable_label(QLabel("Atenuar Fondo:"), lambda: self.slider_sl_ond_bg.setValue(0), "0%"))
        self.lbl_sl_ond_bg = QLabel("0%")
        self.lbl_sl_ond_bg.setStyleSheet("color: #b388ff; font-weight: bold;")
        row_bg_lbl.addWidget(self.lbl_sl_ond_bg)
        col_bg.addLayout(row_bg_lbl)
        self.slider_sl_ond_bg = QSlider(Qt.Horizontal)
        self.slider_sl_ond_bg.setRange(0, 100)
        self.slider_sl_ond_bg.setValue(0)
        self.slider_sl_ond_bg.valueChanged.connect(self._on_sl_ond_bg_changed)
        self.slider_sl_ond_bg.sliderReleased.connect(self.ground_slider_released.emit)
        col_bg.addWidget(self.slider_sl_ond_bg)
        row_sl_p2.addLayout(col_bg)
        sl_layout.addLayout(row_sl_p2)

        # Par 3: Contraste Fondo y Reducción Ruido
        row_sl_p3 = QHBoxLayout()
        row_sl_p3.setSpacing(10)

        col_cs = QVBoxLayout()
        col_cs.setSpacing(2)
        row_cs_lbl = QHBoxLayout()
        row_cs_lbl.addWidget(self._make_resettable_label(QLabel("Contraste Fondo:"), lambda: self.slider_sl_contrast.setValue(0), "0.00"))
        self.lbl_sl_contrast = QLabel("0.00")
        self.lbl_sl_contrast.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_cs_lbl.addWidget(self.lbl_sl_contrast)
        col_cs.addLayout(row_cs_lbl)
        self.slider_sl_contrast = QSlider(Qt.Horizontal)
        self.slider_sl_contrast.setRange(-100, 100)
        self.slider_sl_contrast.setValue(0)
        self.slider_sl_contrast.valueChanged.connect(self._on_sl_contrast_changed)
        self.slider_sl_contrast.sliderReleased.connect(self.ground_slider_released.emit)
        col_cs.addWidget(self.slider_sl_contrast)
        row_sl_p3.addLayout(col_cs)

        col_dn = QVBoxLayout()
        col_dn.setSpacing(2)
        row_dn_lbl = QHBoxLayout()
        row_dn_lbl.addWidget(self._make_resettable_label(QLabel("Reducción Ruido:"), lambda: self.slider_sl_denoise.setValue(0), "0%"))
        self.lbl_sl_denoise = QLabel("0%")
        self.lbl_sl_denoise.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_dn_lbl.addWidget(self.lbl_sl_denoise)
        col_dn.addLayout(row_dn_lbl)
        self.slider_sl_denoise = QSlider(Qt.Horizontal)
        self.slider_sl_denoise.setRange(0, 100)
        self.slider_sl_denoise.setValue(0)
        self.slider_sl_denoise.valueChanged.connect(self._on_sl_denoise_changed)
        self.slider_sl_denoise.sliderReleased.connect(self.ground_slider_released.emit)
        col_dn.addWidget(self.slider_sl_denoise)
        row_sl_p3.addLayout(col_dn)
        sl_layout.addLayout(row_sl_p3)

        # Par 4: Efecto Orton Astronómico y Glow de Polvo Galáctico
        self.box_sl_orton = QGroupBox("✨ Efecto Orton y Glow de Polvo Galáctico")
        self.box_sl_orton.setStyleSheet("""
            QGroupBox {
                border: 1px solid #7b1fa2;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #1a1625;
                font-weight: bold;
                color: #e1bee7;
            }
        """)
        orton_layout = QVBoxLayout(self.box_sl_orton)
        orton_layout.setContentsMargins(6, 6, 6, 6)
        orton_layout.setSpacing(6)

        # Fila 1: Deslizador de Intensidad Orton
        row_ort_int = QHBoxLayout()
        row_ort_int.addWidget(self._make_resettable_label(QLabel("Intensidad Glow:"), lambda: self.slider_sl_orton_strength.setValue(0), "0%"))
        self.lbl_sl_orton_strength = QLabel("0%")
        self.lbl_sl_orton_strength.setStyleSheet("color: #ffb74d; font-weight: bold;")
        self.lbl_sl_orton_strength.setFixedWidth(34)
        row_ort_int.addWidget(self.lbl_sl_orton_strength)
        self.slider_sl_orton_strength = QSlider(Qt.Horizontal)
        self.slider_sl_orton_strength.setRange(0, 100)
        self.slider_sl_orton_strength.setValue(0)
        self.slider_sl_orton_strength.valueChanged.connect(self._on_sl_orton_strength_changed)
        self.slider_sl_orton_strength.sliderReleased.connect(self.ground_slider_released.emit)
        row_ort_int.addWidget(self.slider_sl_orton_strength, stretch=1)
        orton_layout.addLayout(row_ort_int)

        # Fila 2: Radio (Difusión) y Umbral de Polvo
        row_ort_params = QHBoxLayout()
        row_ort_params.setSpacing(10)

        col_rad = QVBoxLayout()
        col_rad.setSpacing(2)
        row_rad_lbl = QHBoxLayout()
        row_rad_lbl.addWidget(self._make_resettable_label(QLabel("Radio:"), lambda: self.slider_sl_orton_radius.setValue(40), "40 px"))
        self.lbl_sl_orton_radius = QLabel("40 px")
        self.lbl_sl_orton_radius.setStyleSheet("color: #ce93d8; font-weight: bold;")
        row_rad_lbl.addWidget(self.lbl_sl_orton_radius)
        col_rad.addLayout(row_rad_lbl)
        self.slider_sl_orton_radius = QSlider(Qt.Horizontal)
        self.slider_sl_orton_radius.setRange(10, 150)
        self.slider_sl_orton_radius.setValue(40)
        self.slider_sl_orton_radius.valueChanged.connect(self._on_sl_orton_radius_changed)
        self.slider_sl_orton_radius.sliderReleased.connect(self.ground_slider_released.emit)
        col_rad.addWidget(self.slider_sl_orton_radius)
        row_ort_params.addLayout(col_rad)

        col_thr = QVBoxLayout()
        col_thr.setSpacing(2)
        row_thr_lbl = QHBoxLayout()
        row_thr_lbl.addWidget(self._make_resettable_label(QLabel("Umbral Polvo:"), lambda: self.slider_sl_orton_threshold.setValue(5), "0.05"))
        self.lbl_sl_orton_threshold = QLabel("0.05")
        self.lbl_sl_orton_threshold.setStyleSheet("color: #ce93d8; font-weight: bold;")
        row_thr_lbl.addWidget(self.lbl_sl_orton_threshold)
        col_thr.addLayout(row_thr_lbl)
        self.slider_sl_orton_threshold = QSlider(Qt.Horizontal)
        self.slider_sl_orton_threshold.setRange(0, 50)
        self.slider_sl_orton_threshold.setValue(5)
        self.slider_sl_orton_threshold.valueChanged.connect(self._on_sl_orton_threshold_changed)
        self.slider_sl_orton_threshold.sliderReleased.connect(self.ground_slider_released.emit)
        col_thr.addWidget(self.slider_sl_orton_threshold)
        row_ort_params.addLayout(col_thr)
        orton_layout.addLayout(row_ort_params)

        # Fila 3: Tinte Cromático y Modo de Fusión
        row_ort_combos = QHBoxLayout()
        row_ort_combos.setSpacing(8)

        col_tint = QVBoxLayout()
        col_tint.setSpacing(2)
        col_tint.addWidget(QLabel("Tinte:"))
        self.combo_sl_orton_tint = QComboBox()
        self.combo_sl_orton_tint.addItem("Dorado Cálido (Núcleo Vía Láctea)", "warm_gold")
        self.combo_sl_orton_tint.addItem("Luminancia Neutra", "neutral")
        self.combo_sl_orton_tint.addItem("Azul Frío (Reflexión)", "cool_blue")
        self.combo_sl_orton_tint.addItem("H-Alfa (Rubí Cósmico)", "halpha")
        self.combo_sl_orton_tint.currentIndexChanged.connect(self._on_sl_orton_tint_changed)
        col_tint.addWidget(self.combo_sl_orton_tint)
        row_ort_combos.addLayout(col_tint, stretch=1)

        col_mode = QVBoxLayout()
        col_mode.setSpacing(2)
        col_mode.addWidget(QLabel("Fusión:"))
        self.combo_sl_orton_mode = QComboBox()
        self.combo_sl_orton_mode.addItem("Luz Suave (Profundidad)", "soft_light")
        self.combo_sl_orton_mode.addItem("Trama (Velo Etéreo)", "screen")
        self.combo_sl_orton_mode.addItem("Aditivo (Luminoso)", "additive")
        self.combo_sl_orton_mode.currentIndexChanged.connect(self._on_sl_orton_mode_changed)
        col_mode.addWidget(self.combo_sl_orton_mode)
        row_ort_combos.addLayout(col_mode, stretch=1)

        orton_layout.addLayout(row_ort_combos)
        sl_layout.addWidget(self.box_sl_orton)

        self.btn_reset_sl_special = QPushButton("↺ Restablecer Ajustes de Fondo")
        self.btn_reset_sl_special.setCursor(Qt.PointingHandCursor)
        self.btn_reset_sl_special.clicked.connect(self.starless_reset_requested.emit)
        sl_layout.addWidget(self.btn_reset_sl_special)

        props_layout.addWidget(self.grp_starless_special)

        main_layout.addWidget(self.container_layer_props)

        # Estado inicial vacío
        self._update_list_and_props_state()

    # =========================================================================
    # GESTIÓN DE CAPAS (AÑADIR, DUPLICAR, ELIMINAR, REORDENAR)
    # =========================================================================

    def has_image_layers(self) -> bool:
        """Indica si la lista contiene capas especiales de imagen (StarNet / Suelo / Base)."""
        return any(l.layer_kind == LAYER_KIND_IMAGE for l in self.layers)

    def ensure_base_layer(self) -> None:
        """
        Garantiza que exista una capa base '🖼️ Fondo (Imagen Base)' al cargar una imagen.
        Si la pila no contiene una capa base ni capas de StarNet, la añade al final ([BOTTOM]).
        """
        has_base_or_starnet = any(
            l.mask_type in (MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS)
            for l in self.layers
        )
        if not has_base_or_starnet:
            base_layer = MaskLayer(
                name="Fondo (Imagen Base)",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_BASE,
                target_scope=SCOPE_TARGET_ALL,
                is_system_layer=True
            )
            self.layers.append(base_layer)
            self._rebuild_list_items(select_layer_id=base_layer.id)
            self.layers_changed.emit()

    def ensure_starnet_layers(self, has_ground: bool = False) -> None:
        """
        Asegura que las capas de imagen producidas por StarNet existan en la pila (estilo Photoshop).
        Si existía una capa base previa 'MASK_TYPE_IMAGE_BASE', se desacopla y reemplaza por
        las capas de StarNet (preservando los ajustes locales de la base en el Fondo Starless).
        Si no existían capas de StarNet, las inicializa en el orden por defecto:
          [TOP]    Máscaras de ajuste del usuario
                   ⭐ Capa de Estrellas
                   🏞️ Capa de Suelo (si has_ground=True)
          [BOTTOM] 🌌 Fondo Nebular (Starless)
        Si ya existían, preserva fielmente el orden personalizado en que el usuario las haya movido.
        """
        existing_sys = {l.mask_type: l for l in self.layers if l.is_system_layer}
        has_starnet = (MASK_TYPE_IMAGE_STARS in existing_sys or MASK_TYPE_IMAGE_STARLESS in existing_sys)

        if not has_starnet:
            # Buscar si existía capa Fondo Base previa para migrar sus ajustes
            base_l = existing_sys.get(MASK_TYPE_IMAGE_BASE)

            user_layers = [l for l in self.layers if not l.is_system_layer]
            stars_layer = MaskLayer(
                name="Capa de Estrellas",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_STARS,
                target_scope=SCOPE_TARGET_STARS,
                is_system_layer=True
            )
            starless_layer = MaskLayer(
                name="Fondo Nebular (Starless)",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_STARLESS,
                target_scope=SCOPE_TARGET_STARLESS,
                is_system_layer=True
            )
            if base_l is not None:
                # Traspasar posibles ajustes hechos sobre la capa base previa
                starless_layer.exposure = base_l.exposure
                starless_layer.contrast = base_l.contrast
                starless_layer.shadows = base_l.shadows
                starless_layer.highlights = base_l.highlights
                starless_layer.saturation = base_l.saturation
                starless_layer.temp = base_l.temp
                starless_layer.tint = base_l.tint
                starless_layer.opacity = base_l.opacity

            new_layers: List[MaskLayer] = list(user_layers)
            new_layers.append(stars_layer)
            if has_ground:
                ground_layer = MaskLayer(
                    name="Capa de Suelo (Terreno)",
                    layer_kind=LAYER_KIND_IMAGE,
                    mask_type=MASK_TYPE_IMAGE_GROUND,
                    target_scope=SCOPE_TARGET_GROUND,
                    is_system_layer=True
                )
                new_layers.append(ground_layer)
            new_layers.append(starless_layer)

            self.layers = new_layers
            self._rebuild_list_items(select_layer_id=stars_layer.id)
            self.layers_changed.emit()
            return

        # Si ya existían capas del sistema, comprobar si el estado de suelo ha cambiado
        has_gnd_layer = MASK_TYPE_IMAGE_GROUND in existing_sys
        modified = False

        if has_ground and not has_gnd_layer:
            # Crear e insertar capa de suelo justo encima del fondo starless
            gnd = MaskLayer(
                name="Capa de Suelo (Terreno)",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_GROUND,
                target_scope=SCOPE_TARGET_GROUND,
                is_system_layer=True
            )
            starless_idx = next((i for i, l in enumerate(self.layers) if l.mask_type == MASK_TYPE_IMAGE_STARLESS), len(self.layers))
            self.layers.insert(starless_idx, gnd)
            modified = True
        elif not has_ground and has_gnd_layer:
            # Quitar la capa de suelo porque se eliminó la máscara
            self.layers = [l for l in self.layers if l.mask_type != MASK_TYPE_IMAGE_GROUND]
            modified = True

        if modified:
            self._rebuild_list_items(select_layer_id=self._selected_layer_id)
            self.layers_changed.emit()

    def remove_starnet_layers(self) -> None:
        """Elimina las capas de sistema generadas por StarNet y restaura la capa base si hay imagen."""
        self.layers = [l for l in self.layers if l.mask_type not in (MASK_TYPE_IMAGE_STARS, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_GROUND)]
        if self._image_reference_getter and self._image_reference_getter() is not None:
            self.ensure_base_layer()
        else:
            self._rebuild_list_items()
            self.layers_changed.emit()

    def set_starnet_stars_opacity(self, opacity: float) -> None:
        """Sincroniza la opacidad de la capa de estrellas con controles externos."""
        for l in self.layers:
            if l.mask_type == MASK_TYPE_IMAGE_STARS:
                l.opacity = float(np.clip(opacity, 0.0, 1.0))
                if self._selected_layer_id == l.id:
                    self._updating_ui = True
                    op_pct = int(round(l.opacity * 100))
                    self.slider_opacity.setValue(op_pct)
                    self.lbl_opacity.setText(f"{op_pct}%")
                    self._updating_ui = False
                break
        self._rebuild_list_items(select_layer_id=self._selected_layer_id)

    def set_starnet_stars_enabled(self, enabled: bool) -> None:
        """Sincroniza la visibilidad de la capa de estrellas con controles externos."""
        for l in self.layers:
            if l.mask_type == MASK_TYPE_IMAGE_STARS:
                l.enabled = enabled
                if self._selected_layer_id == l.id:
                    self._updating_ui = True
                    self.chk_layer_enabled.setChecked(enabled)
                    self._updating_ui = False
                break
        self._rebuild_list_items(select_layer_id=self._selected_layer_id)

    def add_new_layer(self, mask_type: str) -> MaskLayer:
        """Crea una nueva capa de ajuste del tipo indicado y la sitúa al inicio (TOP) de la pila."""
        count = len(self.layers) + 1
        if mask_type == MASK_TYPE_BRUSH:
            name = f"Pincel {count}"
            layer = MaskLayer(
                name=name,
                mask_type=mask_type,
                opacity=self._last_brush_opacity,
                brush_radius=self._last_brush_radius,
                brush_hardness=self._last_brush_hardness,
                brush_flow=self._last_brush_flow,
                brush_mode=self._last_brush_mode
            )
        elif mask_type == MASK_TYPE_LUMA:
            name = f"Luminancia {count}"
            layer = MaskLayer(name=name, mask_type=mask_type)
        elif mask_type == MASK_TYPE_COLOR:
            name = f"Color {count}"
            layer = MaskLayer(name=name, mask_type=mask_type)
        elif mask_type == MASK_TYPE_GROUND:
            name = f"Suelo {count}"
            layer = MaskLayer(name=name, mask_type=mask_type)
        elif mask_type == MASK_TYPE_SKY:
            name = f"Cielo {count}"
            layer = MaskLayer(name=name, mask_type=mask_type)
        else:
            name = f"Máscara {count}"
            layer = MaskLayer(name=name, mask_type=mask_type)

        # En Photoshop las capas nuevas se insertan arriba del todo
        self.layers.insert(0, layer)
        self._rebuild_list_items(select_layer_id=layer.id)
        self.layers_changed.emit()
        return layer

    def _delete_selected(self) -> None:
        """Elimina la capa actualmente seleccionada protegiendo las capas del sistema."""
        idx = self.list_layers.currentRow()
        if 0 <= idx < len(self.layers):
            layer = self.layers[idx]
            if layer.is_system_layer:
                return  # No se permite borrar capas de sistema

            # Apagar herramienta de pincel y limpiar previsualización en lienzo de la capa eliminada
            self.brush_tool_toggled.emit(False, None)
            self.set_preview_active(False)
            self.preview_mask_requested.emit(None)

            self.layers.pop(idx)
            next_select = self.layers[idx].id if idx < len(self.layers) else (self.layers[-1].id if self.layers else None)
            self._rebuild_list_items(select_layer_id=next_select)
            self.layers_changed.emit()

    def _duplicate_selected(self) -> None:
        """Duplica la capa seleccionada si no es de sistema."""
        layer = self.get_selected_layer()
        if layer and not layer.is_system_layer:
            d = layer.to_dict()
            new_layer = MaskLayer.from_dict(d)
            if layer.brush_mask is not None:
                new_layer.brush_mask = layer.brush_mask.copy()
            new_layer.name = f"{layer.name} (Copia)"
            idx = self.layers.index(layer)
            self.layers.insert(idx + 1, new_layer)
            self._rebuild_list_items(select_layer_id=new_layer.id)
            self.layers_changed.emit()

    def _move_selected_up(self) -> None:
        """Mueve la capa seleccionada hacia arriba."""
        idx = self.list_layers.currentRow()
        if idx > 0:
            self.layers[idx - 1], self.layers[idx] = self.layers[idx], self.layers[idx - 1]
            self._rebuild_list_items(select_layer_id=self.layers[idx - 1].id)
            self.layers_changed.emit()

    def _move_selected_down(self) -> None:
        """Mueve la capa seleccionada hacia abajo."""
        idx = self.list_layers.currentRow()
        if 0 <= idx < len(self.layers) - 1:
            self.layers[idx + 1], self.layers[idx] = self.layers[idx], self.layers[idx + 1]
            self._rebuild_list_items(select_layer_id=self.layers[idx + 1].id)
            self.layers_changed.emit()

    def get_selected_layer(self) -> Optional[MaskLayer]:
        """Obtiene la instancia de MaskLayer actualmente seleccionada."""
        if not self._selected_layer_id:
            return None
        for l in self.layers:
            if l.id == self._selected_layer_id:
                return l
        return None

    def _rebuild_list_items(self, select_layer_id: Optional[str] = None) -> None:
        """Regenera los elementos de la lista QListWidget con iconos, botón de ojo y estilos acordes a su tipo."""
        self._updating_ui = True
        self.list_layers.clear()

        select_row = -1
        for i, layer in enumerate(self.layers):
            if layer.mask_type == MASK_TYPE_IMAGE_BASE:
                icon = "🖼️"
                type_tag = "Fondo Base"
                fg_col = "#ffd54f" if layer.enabled else "#607d8b"
            elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
                icon = "⭐"
                type_tag = "StarNet AI"
                fg_col = "#80d8ff" if layer.enabled else "#607d8b"
            elif layer.mask_type == MASK_TYPE_IMAGE_STARLESS:
                icon = "🌌"
                type_tag = "StarNet AI"
                fg_col = "#ce93d8" if layer.enabled else "#607d8b"
            elif layer.mask_type == MASK_TYPE_IMAGE_GROUND:
                icon = "🏞️"
                type_tag = "Paisaje Suelo"
                fg_col = "#a5d6a7" if layer.enabled else "#607d8b"
            else:
                type_tag = {
                    MASK_TYPE_BRUSH: "Pincel",
                    MASK_TYPE_LUMA: "Luminancia",
                    MASK_TYPE_COLOR: "Color HSV",
                    MASK_TYPE_GROUND: "Máscara Suelo",
                    MASK_TYPE_SKY: "Máscara Cielo",
                }.get(layer.mask_type, "Ajuste")

                icon = {
                    MASK_TYPE_BRUSH: "🖌️",
                    MASK_TYPE_LUMA: "💡",
                    MASK_TYPE_COLOR: "🎨",
                    MASK_TYPE_GROUND: "🏞️",
                    MASK_TYPE_SKY: "🌌",
                }.get(layer.mask_type, "📑")
                fg_col = "#e0e6f0" if layer.enabled else "#607d8b"

            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 30))
            self.list_layers.addItem(item)

            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 2, 4, 2)
            row_layout.setSpacing(6)

            bold_css = "font-weight: bold; " if layer.is_system_layer else ""
            lbl_title = QLabel(f"{icon}  {layer.name}  [{type_tag}]")
            lbl_title.setStyleSheet(f"color: {fg_col}; {bold_css}background: transparent;")
            lbl_title.setCursor(Qt.PointingHandCursor)
            row_layout.addWidget(lbl_title, stretch=1)

            btn_eye = QPushButton("👁️" if layer.enabled else "🚫")
            btn_eye.setFixedSize(26, 22)
            btn_eye.setCursor(Qt.PointingHandCursor)
            btn_eye.setToolTip("Capa visible (clic para ocultar)" if layer.enabled else "Capa oculta (clic para mostrar)")
            btn_eye.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    font-size: 13px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #2a3147;
                    border-radius: 3px;
                }
            """)

            def make_click_handler(r):
                return lambda e: self.list_layers.setCurrentRow(r)

            def make_eye_handler(r):
                return lambda: self._toggle_layer_enabled_at(r)

            row_widget.mousePressEvent = make_click_handler(i)
            lbl_title.mousePressEvent = make_click_handler(i)
            btn_eye.clicked.connect(make_eye_handler(i))

            row_layout.addWidget(btn_eye)
            self.list_layers.setItemWidget(item, row_widget)

            if select_layer_id and layer.id == select_layer_id:
                select_row = i

        if select_row >= 0:
            self.list_layers.setCurrentRow(select_row)
            self._selected_layer_id = select_layer_id
        elif self.layers:
            self.list_layers.setCurrentRow(0)
            self._selected_layer_id = self.layers[0].id
        else:
            self._selected_layer_id = None

        self._updating_ui = False
        self._update_list_and_props_state()

    def _toggle_layer_enabled_at(self, idx: int) -> None:
        """Alterna el estado enabled de la capa en el índice indicado al hacer clic en el ojo."""
        if 0 <= idx < len(self.layers):
            layer = self.layers[idx]
            new_state = not layer.enabled
            layer.enabled = new_state

            if not new_state:
                if layer.mask_type == MASK_TYPE_BRUSH:
                    self.brush_tool_toggled.emit(False, None)
                self.set_preview_active(False)
                self.preview_mask_requested.emit(None)
            else:
                if layer.mask_type == MASK_TYPE_BRUSH:
                    self.brush_tool_toggled.emit(True, layer)
                elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
                    self.starnet_visibility_changed.emit(True)

            if not new_state and layer.mask_type == MASK_TYPE_IMAGE_STARS:
                self.starnet_visibility_changed.emit(False)

            self._rebuild_list_items(select_layer_id=layer.id)
            self.layers_changed.emit()

    def _on_layer_selection_changed(self, row: int) -> None:
        """Manejador al cambiar la fila seleccionada."""
        if self._updating_ui:
            return
        if 0 <= row < len(self.layers):
            self._selected_layer_id = self.layers[row].id
        else:
            self._selected_layer_id = None
        self._update_list_and_props_state()

    def _update_list_and_props_state(self) -> None:
        """Actualiza la visibilidad y los valores de los controles según la capa activa."""
        layer = self.get_selected_layer()
        has_sel = layer is not None
        is_sys = layer.is_system_layer if layer else False
        is_ground = (layer is not None and layer.mask_type in (MASK_TYPE_IMAGE_GROUND, MASK_TYPE_GROUND))
        is_stars = (layer is not None and layer.mask_type == MASK_TYPE_IMAGE_STARS)
        is_starless = (layer is not None and layer.mask_type == MASK_TYPE_IMAGE_STARLESS)

        self.container_layer_props.setVisible(has_sel)
        self.btn_move_up.setEnabled(has_sel)
        self.btn_move_down.setEnabled(has_sel)
        self.btn_duplicate.setEnabled(has_sel and not is_sys)
        self.btn_delete.setEnabled(has_sel and not is_sys)

        if hasattr(self, 'grp_gnd_special'):
            self.grp_gnd_special.setVisible(is_ground)
        if hasattr(self, 'grp_stars_special'):
            self.grp_stars_special.setVisible(is_stars)
        if hasattr(self, 'grp_starless_special'):
            self.grp_starless_special.setVisible(is_starless)

        # Aplicar identidad temática y cromática dinámica según el tipo de capa activa
        self._apply_layer_theme(layer)

        if not layer or layer.mask_type != MASK_TYPE_COLOR:
            self.set_color_picker_active(False)

        if not layer:
            self.brush_tool_toggled.emit(False, None)
            self.preview_mask_requested.emit(None)
            return

        self._updating_ui = True

        # Fila A
        self.txt_layer_name.setText(layer.name)
        self.txt_layer_name.setEnabled(not is_sys)
        self.chk_layer_enabled.setChecked(layer.enabled)
        self.chk_layer_invert.setChecked(layer.inverted)
        self.chk_layer_invert.setEnabled(not is_sys)

        # Fila C: Opacidad
        op_pct = int(round(layer.opacity * 100))
        self.slider_opacity.setValue(op_pct)
        self.lbl_opacity.setText(f"{op_pct}%")

        # Grupo de definición de máscara (Ocultar si es capa de sistema de StarNet)
        self.grp_mask_def.setVisible(not is_sys)

        if not is_sys:
            # Sub-paneles de tipo de máscara
            self.panel_brush.setVisible(layer.mask_type == MASK_TYPE_BRUSH)
            self.panel_luma.setVisible(layer.mask_type == MASK_TYPE_LUMA)
            self.panel_color.setVisible(layer.mask_type == MASK_TYPE_COLOR)
            self.panel_ground.setVisible(layer.mask_type in (MASK_TYPE_GROUND, MASK_TYPE_SKY))
        else:
            self.panel_brush.setVisible(False)
            self.panel_luma.setVisible(False)
            self.panel_color.setVisible(False)
            self.panel_ground.setVisible(False)

        if layer.mask_type == MASK_TYPE_BRUSH:
            self.slider_b_radius.setValue(layer.brush_radius)
            self.lbl_b_radius.setText(f"{layer.brush_radius} px")
            self.slider_b_hardness.setValue(int(round(layer.brush_hardness * 100)))
            self.lbl_b_hardness.setText(f"{int(round(layer.brush_hardness * 100))}%")
            self.slider_b_flow.setValue(int(round(layer.brush_flow * 100)))
            self.lbl_b_flow.setText(f"{int(round(layer.brush_flow * 100))}%")
            self.btn_b_paint.setChecked(layer.brush_mode == "paint")
            self.btn_b_erase.setChecked(layer.brush_mode == "erase")

        elif layer.mask_type == MASK_TYPE_LUMA:
            self.spin_luma_min.setValue(layer.luma_min)
            self.spin_luma_max.setValue(layer.luma_max)
            self.spin_luma_feather.setValue(layer.feather)

        elif layer.mask_type == MASK_TYPE_COLOR:
            hue_int = int(round(layer.color_hue))
            self.slider_hue.setValue(hue_int)
            self.lbl_hue_val.setText(f"{hue_int}°")
            self._update_color_swatch(hue_int)
            self.spin_color_tol.setValue(int(layer.color_tolerance))
            self.spin_color_sat.setValue(layer.color_min_sat)

        # Ajustes locales
        self.slider_adj_ev.setValue(int(round(layer.exposure * 100)))
        self.lbl_adj_ev.setText(f"{layer.exposure:+.2f} EV")

        self.slider_adj_contrast.setValue(int(round(layer.contrast * 100)))
        self.lbl_adj_contrast.setText(f"{layer.contrast:+.2f}")

        self.slider_adj_shadows.setValue(int(round(layer.shadows * 100)))
        self.lbl_adj_shadows.setText(f"{layer.shadows:+.2f}")

        self.slider_adj_highlights.setValue(int(round(layer.highlights * 100)))
        self.lbl_adj_highlights.setText(f"{layer.highlights:+.2f}")

        sat_pct = int(round(layer.saturation * 100))
        self.slider_adj_sat.setValue(sat_pct)
        self.lbl_adj_sat.setText(f"{layer.saturation:.2f}x")

        # Temperatura y Tinte WB (-1.0000 a +1.0000 con 4 decimales)
        self.spin_adj_temp.setValue(layer.temp)
        self.slider_adj_temp.setValue(int(round(layer.temp * 10000.0)))

        self.spin_adj_tint.setValue(layer.tint)
        self.slider_adj_tint.setValue(int(round(layer.tint * 10000.0)))

        self._updating_ui = False

        if not layer.enabled or is_sys:
            self.brush_tool_toggled.emit(False, None)
            self.preview_mask_requested.emit(None)
        elif layer.mask_type == MASK_TYPE_BRUSH:
            self.brush_tool_toggled.emit(True, layer)
            self.brush_params_changed.emit(
                layer.brush_radius, layer.brush_hardness, layer.brush_flow, layer.brush_mode
            )
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            else:
                self.preview_mask_requested.emit(None)
        else:
            self.brush_tool_toggled.emit(False, None)
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            else:
                self.preview_mask_requested.emit(None)

    def _apply_layer_theme(self, layer: Optional[MaskLayer]) -> None:
        """Aplica un estilo cromático identificativo a los grupos de controles según el tipo de capa activa."""
        if not hasattr(self, 'grp_local_adj'):
            return

        def _make_box_qss(obj_name: str, border_col: str, title_col: str, bg_col: str) -> str:
            return f"""
                QGroupBox#{obj_name} {{
                    font-weight: bold;
                    border: 1px solid {border_col};
                    border-radius: 6px;
                    margin-top: 12px;
                    padding-top: 14px;
                    background-color: {bg_col};
                    color: #dce1eb;
                }}
                QGroupBox#{obj_name}::title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 12px;
                    padding: 0 6px;
                    color: {title_col};
                    font-weight: bold;
                    font-size: 11px;
                }}
            """

        if layer is None:
            self.grp_local_adj.setStyleSheet(_make_box_qss("local_adj_group", "#2b3044", "#00e5ff", "#171a26"))
            self.grp_local_adj.setTitle("Ajustes Locales de Revelado")
            if hasattr(self, 'grp_mask_def'):
                self.grp_mask_def.setStyleSheet(_make_box_qss("mask_def_group", "#2b3044", "#00e5ff", "#171a26"))
                self.grp_mask_def.setTitle("Definición de Máscara")
            return

        mtype = layer.mask_type
        if mtype in (MASK_TYPE_IMAGE_GROUND, MASK_TYPE_GROUND):
            border_c = "#43a047"
            title_c = "#81c784"
            bg_c = "#141f17"
            adj_title = "🏞️ Ajustes de Capa de Suelo"
            mask_def_title = "🏞️ Definición de Máscara de Suelo"
        elif mtype == MASK_TYPE_IMAGE_STARLESS:
            border_c = "#8e24aa"
            title_c = "#ce93d8"
            bg_c = "#1c1424"
            adj_title = "🌌 Ajustes de Capa de Fondo Nebular"
            mask_def_title = "Definición de Máscara"
        elif mtype == MASK_TYPE_IMAGE_STARS:
            border_c = "#fbc02d"
            title_c = "#ffd54f"
            bg_c = "#221f14"
            adj_title = "⭐ Ajustes de Capa de Estrellas"
            mask_def_title = "Definición de Máscara"
        elif mtype == MASK_TYPE_BRUSH:
            border_c = "#e53935"
            title_c = "#ef5350"
            bg_c = "#221415"
            adj_title = "🖌️ Ajustes Locales de Pincel"
            mask_def_title = "🖌️ Trazo y Parámetros de Pincel"
        elif mtype == MASK_TYPE_LUMA:
            border_c = "#fdd835"
            title_c = "#fff59d"
            bg_c = "#202018"
            adj_title = "💡 Ajustes Locales de Luminancia"
            mask_def_title = "💡 Rango de Luminancia"
        elif mtype == MASK_TYPE_COLOR:
            border_c = "#00acc1"
            title_c = "#80d8ff"
            bg_c = "#141f24"
            adj_title = "🎨 Ajustes Locales de Color HSV"
            mask_def_title = "🎨 Tono y Tolerancia de Color"
        elif mtype == MASK_TYPE_SKY:
            border_c = "#1976d2"
            title_c = "#90caf9"
            bg_c = "#141924"
            adj_title = "☁️ Ajustes de Capa de Cielo"
            mask_def_title = "☁️ Definición de Máscara de Cielo"
        elif mtype == MASK_TYPE_IMAGE_BASE:
            border_c = "#ffa000"
            title_c = "#ffe082"
            bg_c = "#1f1c14"
            adj_title = "🖼️ Ajustes de Imagen Base"
            mask_def_title = "Definición de Máscara"
        else:
            border_c = "#2b3044"
            title_c = "#00e5ff"
            bg_c = "#171a26"
            adj_title = "Ajustes Locales de Revelado"
            mask_def_title = "Definición de Máscara"

        self.grp_local_adj.setStyleSheet(_make_box_qss("local_adj_group", border_c, title_c, bg_c))
        self.grp_local_adj.setTitle(adj_title)

        if hasattr(self, 'grp_mask_def'):
            self.grp_mask_def.setStyleSheet(_make_box_qss("mask_def_group", border_c, title_c, bg_c))
            self.grp_mask_def.setTitle(mask_def_title)

        # Tematizar de manera fija y distinguible los paneles contextuales dedicados
        if hasattr(self, 'grp_gnd_special'):
            self.grp_gnd_special.setStyleSheet(_make_box_qss("gnd_special_group", "#43a047", "#81c784", "#141f17"))
            self.grp_gnd_special.setTitle("🏞️ Ajustes Tonales Especiales de Suelo")
        if hasattr(self, 'grp_stars_special'):
            self.grp_stars_special.setStyleSheet(_make_box_qss("stars_special_group", "#fbc02d", "#ffd54f", "#221f14"))
            self.grp_stars_special.setTitle("⭐ Ajustes Especiales de Estrellas")
        if hasattr(self, 'grp_starless_special'):
            self.grp_starless_special.setStyleSheet(_make_box_qss("starless_special_group", "#8e24aa", "#ce93d8", "#1c1424"))
            self.grp_starless_special.setTitle("🌌 Ajustes Especiales de Fondo Nebular")

    def select_layer_by_id(self, layer_id: str) -> bool:
        """Selecciona la capa con el ID especificado."""
        for layer in self.layers:
            if layer.id == layer_id:
                self._rebuild_list_items(select_layer_id=layer.id)
                return True
        return False

    def select_layer_by_type(self, mask_type: str) -> bool:
        """Selecciona la primera capa que coincida con el tipo especificado."""
        for layer in self.layers:
            if layer.mask_type == mask_type:
                self._rebuild_list_items(select_layer_id=layer.id)
                return True
        return False

    # =========================================================================
    # ACTUALIZACIÓN DE PARÁMETROS DE LA CAPA
    # =========================================================================

    def _on_layer_name_changed(self, text: str) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            if layer.is_system_layer:
                return  # No se permite renombrar capas de sistema
            layer.name = text.strip() or "Máscara"
            self._rebuild_list_items(select_layer_id=layer.id)

    def _on_layer_enabled_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.enabled = checked
            if not checked:
                # Al desactivar la capa, apagar de inmediato el pincel y limpiar previsualización en lienzo
                if layer.mask_type == MASK_TYPE_BRUSH:
                    self.brush_tool_toggled.emit(False, None)
                self.set_preview_active(False)
                self.preview_mask_requested.emit(None)
            self._rebuild_list_items(select_layer_id=layer.id)
            if checked:
                if layer.mask_type == MASK_TYPE_BRUSH:
                    self.brush_tool_toggled.emit(True, layer)
                elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
                    self.starnet_visibility_changed.emit(True)
            elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
                self.starnet_visibility_changed.emit(False)
            self.layers_changed.emit()

    def _on_layer_invert_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.inverted = checked
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _on_opacity_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.opacity = val / 100.0
            if layer.mask_type == MASK_TYPE_BRUSH:
                self._last_brush_opacity = layer.opacity
            self.lbl_opacity.setText(f"{val}%")
            self.layers_changed.emit()

    # Controles de Pincel interactivo
    def _on_brush_param_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_BRUSH:
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
        if layer and layer.mask_type == MASK_TYPE_BRUSH:
            layer.brush_mode = mode
            self._last_brush_mode = mode
            self.btn_b_paint.setChecked(mode == "paint")
            self.btn_b_erase.setChecked(mode == "erase")
            self.brush_params_changed.emit(
                layer.brush_radius, layer.brush_hardness, layer.brush_flow, layer.brush_mode
            )

    def _clear_brush_mask(self) -> None:
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_BRUSH:
            layer.brush_mask = None
            self.brush_tool_toggled.emit(True, layer)
            self.set_preview_active(False)
            self.preview_mask_requested.emit(None)
            self.layers_changed.emit()

    def set_brush_radius_from_canvas(self, radius: int) -> None:
        """Actualiza el control deslizante de radio cuando se redimensiona con la rueda del ratón."""
        self._last_brush_radius = radius
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_BRUSH:
            layer.brush_radius = radius
            self._updating_ui = True
            self.slider_b_radius.setValue(radius)
            self.lbl_b_radius.setText(f"{radius} px")
            self._updating_ui = False

    # Controles de Luminancia
    def _on_luma_param_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_LUMA:
            layer.luma_min = self.spin_luma_min.value()
            layer.luma_max = self.spin_luma_max.value()
            layer.feather = self.spin_luma_feather.value()
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _set_luma_preset(self, l_min: float, l_max: float) -> None:
        self.spin_luma_min.setValue(l_min)
        self.spin_luma_max.setValue(l_max)

    # Controles de Rango de Color
    def _on_hue_slider_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_hue_val.setText(f"{val}°")
        self._update_color_swatch(val)
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_COLOR:
            layer.color_hue = float(val)
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _on_color_param_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_COLOR:
            layer.color_tolerance = float(self.spin_color_tol.value())
            layer.color_min_sat = float(self.spin_color_sat.value())
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _set_color_preset(self, hue: int, tol: int) -> None:
        self.slider_hue.setValue(hue)
        self.spin_color_tol.setValue(tol)

    def _update_color_swatch(self, hue: int) -> None:
        """Actualiza la insignia de color HSV de muestra según el matiz."""
        c = QColor.fromHsv(int(hue % 360), 240, 240)
        self.lbl_color_swatch.setStyleSheet(
            f"background-color: {c.name()}; border: 1px solid #ffffff; border-radius: 3px;"
        )

    def _on_pick_color_toggled(self, checked: bool) -> None:
        """Emite señal para activar o cancelar la pipeta de color en el lienzo."""
        self.color_eyedropper_requested.emit(checked)

    def set_color_picker_active(self, active: bool) -> None:
        """Sincroniza el botón de pipeta de color con el estado del lienzo."""
        if hasattr(self, 'btn_pick_color') and self.btn_pick_color.isChecked() != active:
            self.btn_pick_color.blockSignals(True)
            self.btn_pick_color.setChecked(active)
            self.btn_pick_color.blockSignals(False)

    def apply_sampled_color(self, hue_deg: int, sat_val: float) -> None:
        """Aplica el matiz y saturación muestreados con la pipeta a la capa de color activa."""
        self._updating_ui = True
        self.slider_hue.setValue(int(hue_deg))
        self.lbl_hue_val.setText(f"{int(hue_deg)}°")
        self._update_color_swatch(int(hue_deg))
        # Umbral adaptativo razonable basado en la saturación de la muestra
        suggested_sat = float(np.clip(sat_val * 0.45, 0.03, 0.40))
        self.spin_color_sat.setValue(suggested_sat)
        self._updating_ui = False

        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_COLOR:
            layer.color_hue = float(hue_deg)
            layer.color_min_sat = suggested_sat
            if hasattr(self, 'btn_preview_mask'):
                if not self.btn_preview_mask.isChecked():
                    self.btn_preview_mask.setChecked(True)
                else:
                    self._emit_mask_preview()
            self.layers_changed.emit()

    # Controles de Ajustes Locales
    def _on_adj_ev_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.exposure = val / 100.0
            self.lbl_adj_ev.setText(f"{layer.exposure:+.2f} EV")
            self.layers_changed.emit()

    def _on_adj_contrast_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.contrast = val / 100.0
            self.lbl_adj_contrast.setText(f"{layer.contrast:+.2f}")
            self.layers_changed.emit()

    def _on_adj_shadows_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.shadows = val / 100.0
            self.lbl_adj_shadows.setText(f"{layer.shadows:+.2f}")
            self.layers_changed.emit()

    def _on_adj_highlights_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.highlights = val / 100.0
            self.lbl_adj_highlights.setText(f"{layer.highlights:+.2f}")
            self.layers_changed.emit()

    def _on_adj_sat_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.saturation = val / 100.0
            self.lbl_adj_sat.setText(f"{layer.saturation:.2f}x")
            self.layers_changed.emit()

    def _on_spin_adj_temp_changed(self, val: float) -> None:
        if self._updating_ui: return
        self._updating_ui = True
        slider_val = int(round(val * 10000.0))
        self.slider_adj_temp.setValue(slider_val)
        self._updating_ui = False
        layer = self.get_selected_layer()
        if layer:
            layer.temp = float(val)
            self.layers_changed.emit()

    def _on_slider_adj_temp_changed(self, val: int) -> None:
        if self._updating_ui: return
        self._updating_ui = True
        float_val = val / 10000.0
        self.spin_adj_temp.setValue(float_val)
        self._updating_ui = False
        layer = self.get_selected_layer()
        if layer:
            layer.temp = float_val
            self.layers_changed.emit()

    def _on_spin_adj_tint_changed(self, val: float) -> None:
        if self._updating_ui: return
        self._updating_ui = True
        slider_val = int(round(val * 10000.0))
        self.slider_adj_tint.setValue(slider_val)
        self._updating_ui = False
        layer = self.get_selected_layer()
        if layer:
            layer.tint = float(val)
            self.layers_changed.emit()

    def _on_slider_adj_tint_changed(self, val: int) -> None:
        if self._updating_ui: return
        self._updating_ui = True
        float_val = val / 10000.0
        self.spin_adj_tint.setValue(float_val)
        self._updating_ui = False
        layer = self.get_selected_layer()
        if layer:
            layer.tint = float_val
            self.layers_changed.emit()

    def _reset_local_adjustments(self) -> None:
        """Restablece los controles de revelado local de la capa actual a valores neutros."""
        layer = self.get_selected_layer()
        if not layer: return
        layer.exposure = 0.0
        layer.contrast = 0.0
        layer.shadows = 0.0
        layer.highlights = 0.0
        layer.saturation = 1.0
        layer.temp = 0.0
        layer.tint = 0.0
        self._update_list_and_props_state()
        self.layers_changed.emit()

    # Controles Especiales de Tono de Suelo
    def _on_gnd_special_ev_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_gnd_special_ev.setText(f"{val / 100.0:+.2f} EV")
        self.ground_ev_changed.emit(val)

    def _on_gnd_special_sh_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_gnd_special_sh.setText(f"{val}%")
        self.ground_shadows_changed.emit(val)

    def _on_gnd_special_bp_changed(self, val: int) -> None:
        if self._updating_ui: return
        display_val = val / 1000.0
        self.lbl_gnd_special_bp.setText(f"{display_val:+.3f}")
        self.ground_bp_changed.emit(val)

    def set_ground_tone_values(self, ev_val: int, sh_val: int, bp_val: int) -> None:
        """Sincroniza los valores de los controles tonales de suelo desde tab_developer."""
        self._updating_ui = True
        if hasattr(self, 'slider_gnd_special_ev'):
            self.slider_gnd_special_ev.setValue(ev_val)
            self.lbl_gnd_special_ev.setText(f"{ev_val / 100.0:+.2f} EV")
        if hasattr(self, 'slider_gnd_special_sh'):
            self.slider_gnd_special_sh.setValue(sh_val)
            self.lbl_gnd_special_sh.setText(f"{sh_val}%")
        if hasattr(self, 'slider_gnd_special_bp'):
            self.slider_gnd_special_bp.setValue(bp_val)
            self.lbl_gnd_special_bp.setText(f"{bp_val / 1000.0:+.3f}")
        self._updating_ui = False

    # Controles Especiales de Estrellas
    def _on_stars_special_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_stars_special.setText(f"{val}%")
        self.stars_intensity_changed.emit(val)

    def _on_sl_star_diff_strength_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_star_diff_strength.setText(f"{val}%")
        self.stars_diff_strength_changed.emit(val)

    def _on_sl_star_diff_radius_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_star_diff_radius.setText(f"{val} px")
        self.stars_diff_radius_changed.emit(val)

    def _on_sl_star_diff_threshold_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_star_diff_threshold.setText(f"{val / 100.0:.2f}")
        self.stars_diff_threshold_changed.emit(val)

    def _on_sl_star_diff_color_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_star_diff_color.setText(f"{val}%")
        self.stars_diff_color_boost_changed.emit(val)

    def _on_sl_star_diff_mode_changed(self, idx: int) -> None:
        if self._updating_ui: return
        modes = ["screen", "soft_light", "additive"]
        mode_key = modes[idx] if 0 <= idx < len(modes) else "screen"
        self.stars_diff_mode_changed.emit(mode_key)

    def set_stars_special_values(
        self,
        presence: Optional[int] = None,
        diff_strength: Optional[int] = None,
        diff_radius: Optional[int] = None,
        diff_threshold: Optional[int] = None,
        diff_color: Optional[int] = None,
        diff_mode_idx: Optional[int] = None
    ) -> None:
        """Sincroniza los controles especiales de estrellas desde tab_developer."""
        self._updating_ui = True
        if presence is not None and hasattr(self, 'slider_stars_special'):
            self.slider_stars_special.setValue(presence)
            self.lbl_stars_special.setText(f"{presence}%")
        if diff_strength is not None and hasattr(self, 'slider_sl_star_diff_strength'):
            self.slider_sl_star_diff_strength.setValue(diff_strength)
            self.lbl_sl_star_diff_strength.setText(f"{diff_strength}%")
        if diff_radius is not None and hasattr(self, 'slider_sl_star_diff_radius'):
            self.slider_sl_star_diff_radius.setValue(diff_radius)
            self.lbl_sl_star_diff_radius.setText(f"{diff_radius} px")
        if diff_threshold is not None and hasattr(self, 'slider_sl_star_diff_threshold'):
            self.slider_sl_star_diff_threshold.setValue(diff_threshold)
            self.lbl_sl_star_diff_threshold.setText(f"{diff_threshold / 100.0:.2f}")
        if diff_color is not None and hasattr(self, 'slider_sl_star_diff_color'):
            self.slider_sl_star_diff_color.setValue(diff_color)
            self.lbl_sl_star_diff_color.setText(f"{diff_color}%")
        if diff_mode_idx is not None and hasattr(self, 'combo_sl_star_diff_mode'):
            self.combo_sl_star_diff_mode.setCurrentIndex(diff_mode_idx)
        self._updating_ui = False

    def set_stars_special_value(self, val: int) -> None:
        """Sincroniza el control especial de intensidad de estrellas desde tab_developer (compatibilidad)."""
        self.set_stars_special_values(presence=val)

    # Controles Especiales de Fondo Nebular (Starless)
    def _on_sl_clarity_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_clarity.setText(f"{val / 100.0:+.2f}")
        self.starless_clarity_changed.emit(val)

    def _on_sl_dehaze_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_dehaze.setText(f"{val / 100.0:+.2f}")
        self.starless_dehaze_changed.emit(val)

    def _on_sl_ond_est_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_ond_est.setText(f"{val / 100.0:+.2f}")
        self.starless_ond_est_changed.emit(val)

    def _on_sl_ond_bg_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_ond_bg.setText(f"{val}%")
        self.starless_ond_bg_changed.emit(val)

    def _on_sl_contrast_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_contrast.setText(f"{val / 100.0:+.2f}")
        self.starless_contrast_changed.emit(val)

    def _on_sl_denoise_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_denoise.setText(f"{val}%")
        self.starless_denoise_changed.emit(val)

    def _on_sl_orton_strength_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_orton_strength.setText(f"{val}%")
        self.starless_orton_strength_changed.emit(val)

    def _on_sl_orton_radius_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_orton_radius.setText(f"{val} px")
        self.starless_orton_radius_changed.emit(val)

    def _on_sl_orton_threshold_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_orton_threshold.setText(f"{val / 100.0:.2f}")
        self.starless_orton_threshold_changed.emit(val)

    def _on_sl_orton_tint_changed(self, idx: int) -> None:
        if self._updating_ui: return
        tint_key = self.combo_sl_orton_tint.currentData()
        self.starless_orton_tint_changed.emit(tint_key)

    def _on_sl_orton_mode_changed(self, idx: int) -> None:
        if self._updating_ui: return
        mode_key = self.combo_sl_orton_mode.currentData()
        self.starless_orton_mode_changed.emit(mode_key)

    def set_starless_tone_values(
        self, clarity: int, dehaze: int, ond_est: int, ond_bg: int, contrast: int, denoise: int,
        orton_strength: int = 0, orton_radius: int = 40, orton_threshold: int = 5,
        orton_tint_idx: int = 0, orton_mode_idx: int = 0
    ) -> None:
        """Sincroniza los controles especiales de fondo nebular desde tab_developer."""
        self._updating_ui = True
        if hasattr(self, 'slider_sl_clarity'):
            self.slider_sl_clarity.setValue(clarity)
            self.lbl_sl_clarity.setText(f"{clarity / 100.0:+.2f}")
        if hasattr(self, 'slider_sl_dehaze'):
            self.slider_sl_dehaze.setValue(dehaze)
            self.lbl_sl_dehaze.setText(f"{dehaze / 100.0:+.2f}")
        if hasattr(self, 'slider_sl_ond_est'):
            self.slider_sl_ond_est.setValue(ond_est)
            self.lbl_sl_ond_est.setText(f"{ond_est / 100.0:+.2f}")
        if hasattr(self, 'slider_sl_ond_bg'):
            self.slider_sl_ond_bg.setValue(ond_bg)
            self.lbl_sl_ond_bg.setText(f"{ond_bg}%")
        if hasattr(self, 'slider_sl_contrast'):
            self.slider_sl_contrast.setValue(contrast)
            self.lbl_sl_contrast.setText(f"{contrast / 100.0:+.2f}")
        if hasattr(self, 'slider_sl_denoise'):
            self.slider_sl_denoise.setValue(denoise)
            self.lbl_sl_denoise.setText(f"{denoise}%")
        if hasattr(self, 'slider_sl_orton_strength'):
            self.slider_sl_orton_strength.setValue(orton_strength)
            self.lbl_sl_orton_strength.setText(f"{orton_strength}%")
        if hasattr(self, 'slider_sl_orton_radius'):
            self.slider_sl_orton_radius.setValue(orton_radius)
            self.lbl_sl_orton_radius.setText(f"{orton_radius} px")
        if hasattr(self, 'slider_sl_orton_threshold'):
            self.slider_sl_orton_threshold.setValue(orton_threshold)
            self.lbl_sl_orton_threshold.setText(f"{orton_threshold / 100.0:.2f}")
        if hasattr(self, 'combo_sl_orton_tint'):
            self.combo_sl_orton_tint.setCurrentIndex(orton_tint_idx)
        if hasattr(self, 'combo_sl_orton_mode'):
            self.combo_sl_orton_mode.setCurrentIndex(orton_mode_idx)
        self._updating_ui = False

    # Previsualización interactiva de la máscara en el lienzo
    def _on_preview_mask_toggled(self, checked: bool) -> None:
        if checked:
            self._emit_mask_preview()
        else:
            self.preview_mask_requested.emit(None)

    def _emit_mask_preview(self) -> None:
        """Calcula y envía la máscara de la capa actual para renderizado en el lienzo."""
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

        # En el modelo de Photoshop, la máscara y ajustes se calculan sobre el estado
        # del lienzo resultante de componer todas las capas que quedan por debajo de esta
        canvas_below = get_canvas_below_layer(
            base_img,
            self.layers,
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

    # Serialización y deserialización para proyectos y presets
    def get_state(self) -> List[Dict[str, Any]]:
        """Devuelve una lista serializable de todas las capas configuradas."""
        return [layer.to_dict() for layer in self.layers]

    def set_state(self, state: Optional[List[Dict[str, Any]]]) -> None:
        """Restaura las capas desde una lista serializada."""
        self.layers.clear()
        if state:
            for item in state:
                try:
                    self.layers.append(MaskLayer.from_dict(item))
                except Exception:
                    pass
        self._rebuild_list_items(select_layer_id=self.layers[0].id if self.layers else None)
        self.layers_changed.emit()
