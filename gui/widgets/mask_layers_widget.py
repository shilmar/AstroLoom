# gui/widgets/mask_layers_widget.py
"""
gui/widgets/mask_layers_widget.py - Panel interactivo de gestión de capas de ajuste local (Lightroom style).
Permite crear, ordenar, editar y previsualizar múltiples máscaras de luminancia, color y suelo/cielo,
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
    MaskLayer, compute_layer_mask, compute_effective_layer_mask,
    MASK_TYPE_GROUND, MASK_TYPE_SKY, MASK_TYPE_LUMA, MASK_TYPE_COLOR, MASK_TYPE_CUSTOM,
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

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layers: List[MaskLayer] = []
        self._selected_layer_id: Optional[str] = None
        self._updating_ui: bool = False
        self._image_reference_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._master_mask_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._stars_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._starless_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
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

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # 1. Barra de creación rápida de máscaras
        row_create = QHBoxLayout()
        row_create.setSpacing(6)

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

        # Fila B: Ámbito de Aplicación (Target Scope)
        row_scope = QHBoxLayout()
        row_scope.addWidget(QLabel("Ámbito / Destino:"))
        self.combo_target_scope = QComboBox()
        self.combo_target_scope.addItem("Toda la Imagen (Compuesta)", SCOPE_TARGET_ALL)
        self.combo_target_scope.addItem("Solo Suelo (Paisaje)", SCOPE_TARGET_GROUND)
        self.combo_target_scope.addItem("Solo Fondo (Starless)", SCOPE_TARGET_STARLESS)
        self.combo_target_scope.addItem("Solo Estrellas (Stars)", SCOPE_TARGET_STARS)
        self.combo_target_scope.setToolTip("Define sobre qué capa o región de la imagen actúa este ajuste local")
        self.combo_target_scope.currentIndexChanged.connect(self._on_target_scope_changed)
        row_scope.addWidget(self.combo_target_scope, stretch=1)
        props_layout.addLayout(row_scope)

        # Fila C: Opacidad de la capa
        row_op = QHBoxLayout()
        row_op.addWidget(QLabel("Opacidad:"))
        self.lbl_opacity = QLabel("100%")
        self.lbl_opacity.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_op.addWidget(self.lbl_opacity)
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        row_op.addWidget(self.slider_opacity, stretch=1)
        props_layout.addLayout(row_op)

        # 5. Caja dinámica de definición de máscara (Luminancia / Color / Suelo)
        self.grp_mask_def = QGroupBox("Definición de Máscara")
        self.layout_mask_def = QVBoxLayout(self.grp_mask_def)
        self.layout_mask_def.setContentsMargins(8, 10, 8, 8)
        self.layout_mask_def.setSpacing(6)

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
        row_hue.addWidget(QLabel("Matiz (Hue):"))
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
        grp_local_adj = QGroupBox("Ajustes Locales de Revelado")
        adj_layout = QVBoxLayout(grp_local_adj)
        adj_layout.setContentsMargins(8, 12, 8, 8)
        adj_layout.setSpacing(6)

        # Exposición (EV)
        row_adj_ev = QHBoxLayout()
        row_adj_ev.addWidget(QLabel("Exposición:"))
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
        row_c_lbl.addWidget(QLabel("Contraste:"))
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
        row_sh_lbl.addWidget(QLabel("Sombras:"))
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
        row_hl_lbl.addWidget(QLabel("Luces:"))
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
        row_sat_lbl.addWidget(QLabel("Saturación:"))
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

        # Temperatura y Tinte (WB local)
        row_pair3 = QHBoxLayout()
        row_pair3.setSpacing(10)

        col_t = QVBoxLayout()
        col_t.setSpacing(2)
        row_t_lbl = QHBoxLayout()
        row_t_lbl.addWidget(QLabel("Temp WB:"))
        self.lbl_adj_temp = QLabel("0.0")
        self.lbl_adj_temp.setStyleSheet("color: #ffb74d; font-weight: bold;")
        row_t_lbl.addWidget(self.lbl_adj_temp)
        col_t.addLayout(row_t_lbl)
        self.slider_adj_temp = QSlider(Qt.Horizontal)
        self.slider_adj_temp.setRange(-50, 50)
        self.slider_adj_temp.setValue(0)
        self.slider_adj_temp.valueChanged.connect(self._on_adj_temp_changed)
        col_t.addWidget(self.slider_adj_temp)
        row_pair3.addLayout(col_t)

        col_ti = QVBoxLayout()
        col_ti.setSpacing(2)
        row_ti_lbl = QHBoxLayout()
        row_ti_lbl.addWidget(QLabel("Tinte WB:"))
        self.lbl_adj_tint = QLabel("0.0")
        self.lbl_adj_tint.setStyleSheet("color: #ce93d8; font-weight: bold;")
        row_ti_lbl.addWidget(self.lbl_adj_tint)
        col_ti.addLayout(row_ti_lbl)
        self.slider_adj_tint = QSlider(Qt.Horizontal)
        self.slider_adj_tint.setRange(-50, 50)
        self.slider_adj_tint.setValue(0)
        self.slider_adj_tint.valueChanged.connect(self._on_adj_tint_changed)
        col_ti.addWidget(self.slider_adj_tint)
        row_pair3.addLayout(col_ti)

        adj_layout.addLayout(row_pair3)

        # Botón restablecer ajustes de la capa
        btn_reset_local = QPushButton("↺ Restablecer Ajustes Locales")
        btn_reset_local.setCursor(Qt.PointingHandCursor)
        btn_reset_local.clicked.connect(self._reset_local_adjustments)
        adj_layout.addWidget(btn_reset_local)

        props_layout.addWidget(grp_local_adj)

        main_layout.addWidget(self.container_layer_props)

        # Estado inicial vacío
        self._update_list_and_props_state()

    # =========================================================================
    # GESTIÓN DE CAPAS (AÑADIR, DUPLICAR, ELIMINAR, REORDENAR)
    # =========================================================================

    def add_new_layer(self, mask_type: str) -> MaskLayer:
        """Crea una nueva capa del tipo indicado y la añade a la pila."""
        count = len(self.layers) + 1
        if mask_type == MASK_TYPE_LUMA:
            name = f"Luminancia {count}"
        elif mask_type == MASK_TYPE_COLOR:
            name = f"Color {count}"
        elif mask_type == MASK_TYPE_GROUND:
            name = f"Suelo {count}"
        elif mask_type == MASK_TYPE_SKY:
            name = f"Cielo {count}"
        else:
            name = f"Máscara {count}"

        layer = MaskLayer(name=name, mask_type=mask_type)
        self.layers.append(layer)
        self._rebuild_list_items(select_layer_id=layer.id)
        self.layers_changed.emit()
        return layer

    def _delete_selected(self) -> None:
        """Elimina la capa actualmente seleccionada."""
        idx = self.list_layers.currentRow()
        if 0 <= idx < len(self.layers):
            layer = self.layers.pop(idx)
            next_select = self.layers[idx].id if idx < len(self.layers) else (self.layers[-1].id if self.layers else None)
            self._rebuild_list_items(select_layer_id=next_select)
            self.layers_changed.emit()

    def _duplicate_selected(self) -> None:
        """Duplica la capa seleccionada."""
        layer = self.get_selected_layer()
        if layer:
            d = layer.to_dict()
            new_layer = MaskLayer.from_dict(d)
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
        """Regenera los elementos de la lista QListWidget."""
        self._updating_ui = True
        self.list_layers.clear()

        select_row = -1
        for i, layer in enumerate(self.layers):
            scope_tag = {
                SCOPE_TARGET_ALL: "Compuesta",
                SCOPE_TARGET_GROUND: "Suelo",
                SCOPE_TARGET_STARLESS: "Starless",
                SCOPE_TARGET_STARS: "Estrellas",
            }.get(layer.target_scope, "Todo")

            icon = {
                MASK_TYPE_LUMA: "💡",
                MASK_TYPE_COLOR: "🎨",
                MASK_TYPE_GROUND: "🏞️",
                MASK_TYPE_SKY: "🌌",
            }.get(layer.mask_type, "📑")

            status = "✓" if layer.enabled else "✕"
            txt = f"{status} {icon} {layer.name}  [{scope_tag}]"
            item = QListWidgetItem(txt)
            if not layer.enabled:
                item.setForeground(QColor("#78909c"))
            self.list_layers.addItem(item)

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

        self.container_layer_props.setVisible(has_sel)
        self.btn_move_up.setEnabled(has_sel)
        self.btn_move_down.setEnabled(has_sel)
        self.btn_duplicate.setEnabled(has_sel)
        self.btn_delete.setEnabled(has_sel)

        if not layer:
            return

        self._updating_ui = True

        # Fila A
        self.txt_layer_name.setText(layer.name)
        self.chk_layer_enabled.setChecked(layer.enabled)
        self.chk_layer_invert.setChecked(layer.inverted)

        # Fila B: Scope
        idx = self.combo_target_scope.findData(layer.target_scope)
        if idx >= 0:
            self.combo_target_scope.setCurrentIndex(idx)

        # Fila C: Opacidad
        op_pct = int(round(layer.opacity * 100))
        self.slider_opacity.setValue(op_pct)
        self.lbl_opacity.setText(f"{op_pct}%")

        # Sub-paneles de tipo de máscara
        self.panel_luma.setVisible(layer.mask_type == MASK_TYPE_LUMA)
        self.panel_color.setVisible(layer.mask_type == MASK_TYPE_COLOR)
        self.panel_ground.setVisible(layer.mask_type in (MASK_TYPE_GROUND, MASK_TYPE_SKY))

        if layer.mask_type == MASK_TYPE_LUMA:
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

        self.slider_adj_temp.setValue(int(round(layer.temp)))
        self.lbl_adj_temp.setText(f"{layer.temp:+.1f}")

        self.slider_adj_tint.setValue(int(round(layer.tint)))
        self.lbl_adj_tint.setText(f"{layer.tint:+.1f}")

        self._updating_ui = False

        # Si el botón de previsualización de máscara en lienzo estaba activo, refrescarlo
        if self.btn_preview_mask.isChecked():
            self._emit_mask_preview()

    # =========================================================================
    # ACTUALIZACIÓN DE PARÁMETROS DE LA CAPA
    # =========================================================================

    def _on_layer_name_changed(self, text: str) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.name = text.strip() or "Máscara"
            # Actualizar texto del item en la lista sin reconstruir todo
            row = self.list_layers.currentRow()
            if row >= 0:
                scope_tag = {
                    SCOPE_TARGET_ALL: "Compuesta",
                    SCOPE_TARGET_GROUND: "Suelo",
                    SCOPE_TARGET_STARLESS: "Starless",
                    SCOPE_TARGET_STARS: "Estrellas",
                }.get(layer.target_scope, "Todo")
                icon = "💡" if layer.mask_type == MASK_TYPE_LUMA else ("🎨" if layer.mask_type == MASK_TYPE_COLOR else "🏞️")
                status = "✓" if layer.enabled else "✕"
                self.list_layers.item(row).setText(f"{status} {icon} {layer.name}  [{scope_tag}]")

    def _on_layer_enabled_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.enabled = checked
            self._rebuild_list_items(select_layer_id=layer.id)
            self.layers_changed.emit()

    def _on_layer_invert_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.inverted = checked
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _on_target_scope_changed(self, idx: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.target_scope = self.combo_target_scope.currentData()
            self._rebuild_list_items(select_layer_id=layer.id)
            self.layers_changed.emit()

    def _on_opacity_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.opacity = val / 100.0
            self.lbl_opacity.setText(f"{val}%")
            self.layers_changed.emit()

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

    def _on_adj_temp_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.temp = float(val)
            self.lbl_adj_temp.setText(f"{layer.temp:+.1f}")
            self.layers_changed.emit()

    def _on_adj_tint_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.tint = float(val)
            self.lbl_adj_tint.setText(f"{layer.tint:+.1f}")
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

    # Previsualización interactiva de la máscara en el lienzo
    def _on_preview_mask_toggled(self, checked: bool) -> None:
        if checked:
            self._emit_mask_preview()
        else:
            self.preview_mask_requested.emit(None)

    def _emit_mask_preview(self) -> None:
        """Calcula y envía la máscara de la capa actual para renderizado en el lienzo."""
        layer = self.get_selected_layer()
        if not layer or not self._image_reference_getter:
            self.preview_mask_requested.emit(None)
            return

        base_img = self._image_reference_getter()
        if base_img is None:
            self.preview_mask_requested.emit(None)
            return

        master_mask = self._master_mask_getter() if self._master_mask_getter else None
        stars_img = self._stars_getter() if self._stars_getter else None
        starless_img = self._starless_getter() if self._starless_getter else None

        m = compute_effective_layer_mask(
            layer, base_img,
            master_sky_mask=master_mask,
            starless_image=starless_img,
            stars_image=stars_img
        )
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
