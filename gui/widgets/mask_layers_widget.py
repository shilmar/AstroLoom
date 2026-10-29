# gui/widgets/mask_layers_widget.py
"""
gui/widgets/mask_layers_widget.py - Panel interactivo de gestión de capas de máscaras y ajustes locales.
Permite crear, ordenar, editar y previsualizar múltiples máscaras (Pincel, Luminancia, Color y Suelo/Cielo),
con ajustes locales paramétricos independientes y conciencia de capas StarNet.
"""
import uuid
from typing import Optional, List, Dict, Any, Callable, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QColor, QFont, QCursor, QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QSlider, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QLineEdit, QGroupBox,
    QFrame, QSizePolicy, QMessageBox, QSplitter, QButtonGroup
)

from core.mask_layers import (
    MaskLayer, compute_layer_mask, compute_effective_layer_mask, get_canvas_below_layer,
    MASK_TYPE_BRUSH, MASK_TYPE_GROUND, MASK_TYPE_SKY, MASK_TYPE_LUMA, MASK_TYPE_COLOR, MASK_TYPE_CUSTOM,
    LAYER_KIND_ADJUSTMENT, LAYER_KIND_IMAGE,
    MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS, MASK_TYPE_IMAGE_GROUND,
    SCOPE_TARGET_ALL, SCOPE_TARGET_GROUND, SCOPE_TARGET_STARLESS, SCOPE_TARGET_STARS
)

# Calibración de dimensiones para los 3 modos de miniaturas de capa (con soporte para horizontal y vertical)
# Nueva escala generosa: S equivale a la M anterior, M a la L anterior y L es una nueva XL detallada
THUMB_SPECS = {
    "S": {"width": 42, "height": 28, "max_w": 42, "max_h": 34, "row_height": 42, "icon_size": 12, "title_pt": 9.5},
    "M": {"width": 56, "height": 38, "max_w": 56, "max_h": 48, "row_height": 56, "icon_size": 14, "title_pt": 10},
    "L": {"width": 75, "height": 50, "max_w": 75, "max_h": 66, "row_height": 74, "icon_size": 16, "title_pt": 11},
}
from gui.theme import (
    BG_PANEL, BORDER_BASE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    TEXT_CYAN, ACCENT_CYAN
)
from gui.icons import get_icon, get_svg_pixmap
from gui.widgets.precision_slider import PrecisionSliderRow, PrecisionSpinBox
from gui.widgets.curve_widget import CurveWidget
from gui.widgets.collapsible_section import CollapsibleSection


class LayerListResizeHandle(QFrame):
    """
    Tirador interactivo y elegante para ajustar la altura de la lista de capas.
    Permite arrastrar hacia arriba/abajo sin restricciones de divisores rígidos,
    y realizar doble clic para volver al modo auto-ajuste adaptativo.
    """
    height_adjusted = Signal(int)
    reset_auto_fit = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(12)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip("Arrastra para ajustar la altura de la lista de capas.\nDoble clic para auto-ajustar a todas las capas.")
        self._dragging = False
        self._drag_start_y = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        
        # Línea divisoria tenue
        painter.setPen(QColor("#1a2030"))
        painter.drawLine(8, h // 2, w - 8, h // 2)

        # Píldora de agarre central
        pill_w = 42
        pill_h = 4
        pill_x = (w - pill_w) // 2
        pill_y = (h - pill_h) // 2
        color = QColor("#38bdf8") if self.underMouse() or self._dragging else QColor("#334155")
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 2, 2)
        painter.end()

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_y = event.globalPosition().y()
            self.update()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            curr_y = event.globalPosition().y()
            delta = int(curr_y - self._drag_start_y)
            self._drag_start_y = curr_y
            self.height_adjusted.emit(delta)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.update()
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset_auto_fit.emit()
            event.accept()


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
    ground_denoise_changed = Signal(int)         # (val 0 a 100) sincronizar slider de ruido de suelo
    ground_antihalo_changed = Signal(int)        # (val 0 a 100) fuerza anti-halo en la frontera
    ground_edge_shift_changed = Signal(int)      # (val -15 a 15) desplazamiento de borde en px
    ground_ai_denoise_requested = Signal()       # Solicitud de alternar o calcular Denoise IA para Suelo
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
    starless_denoise_color_changed = Signal(int) # (val 0 a 100)
    starless_denoise_mask_changed = Signal(int)  # (val 0 a 100)
    starless_ai_denoise_requested = Signal()
    starless_orton_strength_changed = Signal(int)   # (val 0 a 100)
    starless_orton_radius_changed = Signal(int)     # (val 10 a 150)
    starless_orton_threshold_changed = Signal(int)  # (val 0 a 50)
    starless_orton_tint_changed = Signal(str)       # "warm_gold", "neutral", "cool_blue", "halpha"
    starless_orton_mode_changed = Signal(str)       # "soft_light", "screen", "additive"
    starless_reset_requested = Signal()          # restablecer ajustes de fondo
    auto_stretch_starless_requested = Signal()   # Estirado Adaptativo RGB para capa starless
    interaction_started = Signal()
    interaction_finished = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layers: List[MaskLayer] = []
        self._selected_layer_id: Optional[str] = None
        self._updating_ui: bool = False
        self._is_scrubbing: bool = False
        self._image_reference_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._master_mask_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._stars_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._starless_getter: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._layer_image_provider: Optional[Callable[[Any], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]] = None

        # Memoria de herramienta para Pincel y Máscaras
        self._last_brush_radius: int = 35
        self._last_brush_hardness: float = 0.50
        self._last_brush_flow: float = 0.80
        self._last_brush_mode: str = "paint"
        self._last_brush_opacity: float = 1.00

        # Modo de tamaño de miniaturas: "S" (Compacto), "M" (Estándar), "L" (Grande)
        self._thumb_size_mode: str = "M"
        self._custom_list_height: bool = False
        self._user_list_height: Optional[int] = None

        # Temporizador para reconstrucción diferida de miniaturas en lista tras interacción continua
        self._thumb_rebuild_timer = QTimer(self)
        self._thumb_rebuild_timer.setSingleShot(True)
        self._thumb_rebuild_timer.setInterval(250)
        self._thumb_rebuild_timer.timeout.connect(self._on_mask_param_interaction_finished)

        self._setup_ui()

    def set_getters(
        self,
        img_getter: Callable[[], Optional[np.ndarray]],
        mask_getter: Callable[[], Optional[np.ndarray]],
        stars_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        starless_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        layer_image_provider: Optional[Callable[[Any], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]] = None
    ) -> None:
        """Configura los proveedores de imagen base, máscara maestra de suelo y capas StarNet."""
        self._image_reference_getter = img_getter
        self._master_mask_getter = mask_getter
        self._stars_getter = stars_getter
        self._starless_getter = starless_getter
        self._layer_image_provider = layer_image_provider

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

        self.btn_add_brush = QPushButton(" Pincel")
        self.btn_add_brush.setIcon(get_icon("brush", color="#e0e6f0", size=13))
        self.btn_add_brush.setCursor(Qt.PointingHandCursor)
        self.btn_add_brush.setToolTip("Crea una capa con máscara pintada a mano con dureza y opacidad de trazo")
        self.btn_add_brush.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_BRUSH))
        row_create.addWidget(self.btn_add_brush)

        self.btn_add_luma = QPushButton(" Luminancia")
        self.btn_add_luma.setIcon(get_icon("sparkles", color="#e0e6f0", size=13))
        self.btn_add_luma.setCursor(Qt.PointingHandCursor)
        self.btn_add_luma.setToolTip("Crea una máscara basada en el rango de brillo (Sombras, Medios, Luces)")
        self.btn_add_luma.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_LUMA))
        row_create.addWidget(self.btn_add_luma)

        self.btn_add_color = QPushButton(" Color (HSV)")
        self.btn_add_color.setIcon(get_icon("palette", color="#e0e6f0", size=13))
        self.btn_add_color.setCursor(Qt.PointingHandCursor)
        self.btn_add_color.setToolTip("Crea una máscara basada en tono cromático (H-Alfa, OIII, Sodio, etc.)")
        self.btn_add_color.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_COLOR))
        row_create.addWidget(self.btn_add_color)

        self.btn_add_gnd = QPushButton(" Suelo / Cielo")
        self.btn_add_gnd.setIcon(get_icon("mountain", color="#e0e6f0", size=13))
        self.btn_add_gnd.setCursor(Qt.PointingHandCursor)
        self.btn_add_gnd.setToolTip("Crea una capa vinculada a la segmentación del terreno")
        self.btn_add_gnd.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_GROUND))
        row_create.addWidget(self.btn_add_gnd)

        main_layout.addLayout(row_create)

        # Panel superior: Cabecera con selector S/M/L + Lista de capas + Acciones
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        # Fila de cabecera: Título de la pila + Micro-conmutador de miniaturas [ S | M | L ]
        row_header = QHBoxLayout()
        row_header.setContentsMargins(0, 2, 0, 2)
        row_header.setSpacing(4)
        
        lbl_pila = QLabel("Pila de Capas")
        lbl_pila.setStyleSheet("font-size: 11px; font-weight: bold; color: #cbd5e1;")
        row_header.addWidget(lbl_pila)
        row_header.addStretch(1)

        lbl_vista = QLabel("Miniaturas:")
        lbl_vista.setStyleSheet("font-size: 10px; color: #64748b;")
        row_header.addWidget(lbl_vista)

        self.btn_group_size = QButtonGroup(self)
        self.btn_size_s = QPushButton("S")
        self.btn_size_m = QPushButton("M")
        self.btn_size_l = QPushButton("L")
        for btn in (self.btn_size_s, self.btn_size_m, self.btn_size_l):
            btn.setCheckable(True)
            btn.setFixedSize(22, 20)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #12151e;
                    color: #94a3b8;
                    border: 1px solid #202638;
                    border-radius: 3px;
                    font-size: 10px;
                    font-weight: 600;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #1a2234;
                    color: #f1f5f9;
                }
                QPushButton:checked {
                    background-color: #0284c7;
                    color: #ffffff;
                    border-color: #38bdf8;
                    font-weight: bold;
                }
            """)
            self.btn_group_size.addButton(btn)
            row_header.addWidget(btn)

        self.btn_size_m.setChecked(True)
        self.btn_size_s.clicked.connect(lambda: self.set_thumbnail_size_mode("S"))
        self.btn_size_m.clicked.connect(lambda: self.set_thumbnail_size_mode("M"))
        self.btn_size_l.clicked.connect(lambda: self.set_thumbnail_size_mode("L"))
        top_layout.addLayout(row_header)

        # 2. Lista de capas de máscaras
        self.list_layers = QListWidget()
        self.list_layers.setMinimumHeight(70)
        self.list_layers.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_layers.setStyleSheet("""
            QListWidget {
                background-color: #0f121a;
                border: 1px solid #1e2538;
                border-radius: 5px;
                padding: 2px;
            }
            QListWidget::item {
                background-color: #131722;
                border: 1px solid #1c2333;
                border-radius: 4px;
                margin-bottom: 2px;
                padding: 0px;
            }
            QListWidget::item:selected {
                background-color: #162033;
                border: 1px solid #0284c7;
            }
            QListWidget::item:hover:!selected {
                background-color: #161a26;
                border-color: #28324a;
            }
        """)
        self.list_layers.currentRowChanged.connect(self._on_layer_selection_changed)
        self.list_layers.itemClicked.connect(lambda item: self._on_layer_selection_changed(self.list_layers.row(item)))
        top_layout.addWidget(self.list_layers, stretch=1)

        # 3. Acciones sobre la capa seleccionada (Subir, Bajar, Duplicar, Eliminar)
        row_actions = QHBoxLayout()
        row_actions.setSpacing(6)

        self.btn_move_up = QPushButton(" Subir")
        self.btn_move_up.setIcon(get_icon("chevron_up", color="#b0bec5", size=13))
        self.btn_move_up.setToolTip("Mueve la capa seleccionada hacia arriba en la pila")
        self.btn_move_up.clicked.connect(self._move_selected_up)
        row_actions.addWidget(self.btn_move_up)

        self.btn_move_down = QPushButton(" Bajar")
        self.btn_move_down.setIcon(get_icon("chevron_down", color="#b0bec5", size=13))
        self.btn_move_down.setToolTip("Mueve la capa seleccionada hacia abajo en la pila")
        self.btn_move_down.clicked.connect(self._move_selected_down)
        row_actions.addWidget(self.btn_move_down)

        self.btn_duplicate = QPushButton(" Duplicar")
        self.btn_duplicate.setIcon(get_icon("copy", color="#b0bec5", size=13))
        self.btn_duplicate.setToolTip("Duplica la capa seleccionada con todos sus parámetros")
        self.btn_duplicate.clicked.connect(self._duplicate_selected)
        row_actions.addWidget(self.btn_duplicate)

        self.btn_delete = QPushButton(" Eliminar")
        self.btn_delete.setIcon(get_icon("trash", color="#ef9a9a", size=13))
        self.btn_delete.setToolTip("Elimina la capa seleccionada")
        self.btn_delete.setStyleSheet("color: #ef9a9a;")
        self.btn_delete.clicked.connect(self._delete_selected)
        row_actions.addWidget(self.btn_delete)

        top_layout.addLayout(row_actions)
        main_layout.addWidget(top_container)

        # 3. Tirador interactivo de redimensionado de altura de lista
        self.resize_handle = LayerListResizeHandle(self)
        self.resize_handle.height_adjusted.connect(self._on_list_resize_dragged)
        self.resize_handle.reset_auto_fit.connect(self._on_list_auto_fit_requested)
        main_layout.addWidget(self.resize_handle)

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
        self.slider_opacity.sliderReleased.connect(self.ground_slider_released.emit)
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

        self.btn_b_clear = QPushButton(" Limpiar")
        self.btn_b_clear.setIcon(get_icon("trash", color="#ef9a9a", size=13))
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
        self.spin_luma_min.editingFinished.connect(self._on_mask_param_interaction_finished)
        row_luma_thresh.addWidget(self.spin_luma_min)

        row_luma_thresh.addWidget(QLabel("Máx:"))
        self.spin_luma_max = QDoubleSpinBox()
        self.spin_luma_max.setRange(0.0, 1.0)
        self.spin_luma_max.setSingleStep(0.05)
        self.spin_luma_max.setValue(1.0)
        self.spin_luma_max.valueChanged.connect(self._on_luma_param_changed)
        self.spin_luma_max.editingFinished.connect(self._on_mask_param_interaction_finished)
        row_luma_thresh.addWidget(self.spin_luma_max)

        row_luma_thresh.addWidget(QLabel("Borde:"))
        self.spin_luma_feather = QDoubleSpinBox()
        self.spin_luma_feather.setRange(0.001, 0.40)
        self.spin_luma_feather.setSingleStep(0.02)
        self.spin_luma_feather.setValue(0.08)
        self.spin_luma_feather.valueChanged.connect(self._on_luma_param_changed)
        self.spin_luma_feather.editingFinished.connect(self._on_mask_param_interaction_finished)
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
        self.slider_hue.sliderReleased.connect(self._on_mask_param_interaction_finished)
        row_hue.addWidget(self.slider_hue, stretch=1)
        color_layout.addLayout(row_hue)

        # Fila con Pipeta Muestreadora de Color
        row_pick = QHBoxLayout()
        self.btn_pick_color = QPushButton(" Muestrear Color en Imagen")
        self.btn_pick_color.setIcon(get_icon("pipette", color="#80d8ff", size=13))
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
        self.spin_color_tol.editingFinished.connect(self._on_mask_param_interaction_finished)
        row_tol_sat.addWidget(self.spin_color_tol)

        row_tol_sat.addWidget(QLabel("Sat. Mín:"))
        self.spin_color_sat = QDoubleSpinBox()
        self.spin_color_sat.setRange(0.01, 0.90)
        self.spin_color_sat.setSingleStep(0.05)
        self.spin_color_sat.setValue(0.08)
        self.spin_color_sat.valueChanged.connect(self._on_color_param_changed)
        self.spin_color_sat.editingFinished.connect(self._on_mask_param_interaction_finished)
        row_tol_sat.addWidget(self.spin_color_sat)

        color_layout.addLayout(row_tol_sat)

        # Presets astronómicos de color
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

        # Sub-panel Suelo / Cielo
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
        self.btn_ai_gnd_mask.setToolTip("Detecta automáticamente el horizonte con IA ligera U²-Net (1-Click)")
        self.btn_ai_gnd_mask.clicked.connect(self.ai_ground_mask_requested.emit)
        row_gnd_info.addWidget(self.btn_ai_gnd_mask)

        self.btn_edit_gnd_mask = QPushButton(" Pintar Trazos...")
        self.btn_edit_gnd_mask.setIcon(get_icon("brush", color="#a5d6a7", size=13))
        self.btn_edit_gnd_mask.setCursor(Qt.PointingHandCursor)
        self.btn_edit_gnd_mask.clicked.connect(self.edit_ground_mask_requested.emit)
        row_gnd_info.addWidget(self.btn_edit_gnd_mask)
        gnd_layout.addLayout(row_gnd_info)

        self.layout_mask_def.addWidget(self.panel_ground)

        # Botón de visualización en lienzo
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

        props_layout.addWidget(self.grp_mask_def)

        # 6. Controles de Ajustes Locales de Revelado
        self.grp_local_adj = QGroupBox("Ajustes Locales de Revelado")
        self.grp_local_adj.setObjectName("local_adj_group")
        adj_layout = QVBoxLayout(self.grp_local_adj)
        adj_layout.setContentsMargins(8, 12, 8, 8)
        adj_layout.setSpacing(6)

        # Exposición (EV)
        self.row_adj_ev = PrecisionSliderRow("Exposición:", -300, 300, 0, scale=100.0, decimals=2, unit=" EV", force_sign=True, label_width=115)
        self.slider_adj_ev = self.row_adj_ev.slider
        self.lbl_adj_ev = self.row_adj_ev.lbl_value
        self.slider_adj_ev.valueChanged.connect(self._on_adj_ev_changed)
        self.row_adj_ev.sliderReleased.connect(self.ground_slider_released.emit)
        adj_layout.addWidget(self.row_adj_ev)

        # Contraste
        self.row_adj_contrast = PrecisionSliderRow("Contraste:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_contrast = self.row_adj_contrast.slider
        self.lbl_adj_contrast = self.row_adj_contrast.lbl_value
        self.slider_adj_contrast.valueChanged.connect(self._on_adj_contrast_changed)
        self.row_adj_contrast.sliderReleased.connect(self.ground_slider_released.emit)
        adj_layout.addWidget(self.row_adj_contrast)

        # Sombras
        self.row_adj_shadows = PrecisionSliderRow("Sombras:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_shadows = self.row_adj_shadows.slider
        self.lbl_adj_shadows = self.row_adj_shadows.lbl_value
        self.slider_adj_shadows.valueChanged.connect(self._on_adj_shadows_changed)
        self.row_adj_shadows.sliderReleased.connect(self.ground_slider_released.emit)
        adj_layout.addWidget(self.row_adj_shadows)

        # Luces
        self.row_adj_highlights = PrecisionSliderRow("Luces:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_highlights = self.row_adj_highlights.slider
        self.lbl_adj_highlights = self.row_adj_highlights.lbl_value
        self.slider_adj_highlights.valueChanged.connect(self._on_adj_highlights_changed)
        self.row_adj_highlights.sliderReleased.connect(self.ground_slider_released.emit)
        adj_layout.addWidget(self.row_adj_highlights)

        # Saturación
        self.row_adj_sat = PrecisionSliderRow("Saturación:", 0, 250, 100, scale=100.0, decimals=2, unit="x", label_width=115)
        self.slider_adj_sat = self.row_adj_sat.slider
        self.lbl_adj_sat = self.row_adj_sat.lbl_value
        self.slider_adj_sat.valueChanged.connect(self._on_adj_sat_changed)
        self.row_adj_sat.sliderReleased.connect(self.ground_slider_released.emit)
        adj_layout.addWidget(self.row_adj_sat)

        # Temperatura WB
        self.row_adj_temp = PrecisionSliderRow("Temp WB:", -10000, 10000, 0, scale=10000.0, decimals=4, force_sign=True, label_width=115)
        self.slider_adj_temp = self.row_adj_temp.slider
        self.spin_adj_temp = self.row_adj_temp.spin
        self.lbl_adj_temp = self.row_adj_temp.lbl_value
        self.slider_adj_temp.valueChanged.connect(self._on_slider_adj_temp_changed)
        self.row_adj_temp.sliderReleased.connect(self.ground_slider_released.emit)
        adj_layout.addWidget(self.row_adj_temp)

        # Tinte WB
        self.row_adj_tint = PrecisionSliderRow("Tinte WB:", -10000, 10000, 0, scale=10000.0, decimals=4, force_sign=True, label_width=115)
        self.slider_adj_tint = self.row_adj_tint.slider
        self.spin_adj_tint = self.row_adj_tint.spin
        self.lbl_adj_tint = self.row_adj_tint.lbl_value
        self.slider_adj_tint.valueChanged.connect(self._on_slider_adj_tint_changed)
        self.row_adj_tint.sliderReleased.connect(self.ground_slider_released.emit)
        adj_layout.addWidget(self.row_adj_tint)

        # Curva Tonal de Capa Interactiva
        self.sec_layer_curve = CollapsibleSection(
            "Curva Tonal de Capa",
            is_expanded=False,
            icon_name="activity",
            on_reset=self._reset_layer_curve
        )
        w_curve = QWidget()
        curve_layout = QVBoxLayout(w_curve)
        curve_layout.setContentsMargins(2, 4, 2, 4)
        curve_layout.setSpacing(4)

        self.layer_curve_widget = CurveWidget()
        self.layer_curve_widget.setFixedHeight(160)
        self.layer_curve_widget.curveChanged.connect(self._on_layer_curve_changed)
        self.layer_curve_widget.curveCommitted.connect(self.ground_slider_released.emit)
        curve_layout.addWidget(self.layer_curve_widget)

        self.sec_layer_curve.set_content_widget(w_curve)
        self.sec_layer_curve.toggled.connect(lambda exp: self.update_active_layer_histogram() if exp else None)
        adj_layout.addWidget(self.sec_layer_curve)

        # Botón restablecer ajustes de la capa
        btn_reset_local = QPushButton("Restablecer Ajustes Locales")
        btn_reset_local.setIcon(get_icon("rotate_ccw", color="#94a3b8", size=13))
        btn_reset_local.setCursor(Qt.PointingHandCursor)
        btn_reset_local.clicked.connect(self._reset_local_adjustments)
        adj_layout.addWidget(btn_reset_local)

        props_layout.addWidget(self.grp_local_adj)

        # 7. Controles Especiales de Tono de Suelo (Visibles SOLO cuando se selecciona una capa de Suelo)
        self.grp_gnd_special = QGroupBox("Ajustes Tonales Especiales de Suelo (Pre-Estirado Lineal)")
        self.grp_gnd_special.setObjectName("gnd_special_group")
        gnd_spec_layout = QVBoxLayout(self.grp_gnd_special)
        gnd_spec_layout.setContentsMargins(8, 12, 8, 8)
        gnd_spec_layout.setSpacing(6)

        lbl_gnd_notice = QLabel("Ajustes lineales pre-estirado (EV lineal, sombras, anti-halo y reducción de ruido).")
        lbl_gnd_notice.setStyleSheet("color: #a5d6a7; font-size: 11px; font-style: italic;")
        gnd_spec_layout.addWidget(lbl_gnd_notice)

        # Exposición Suelo
        self.row_gnd_special_ev = PrecisionSliderRow("Exposición Suelo:", -200, 300, 0, scale=100.0, decimals=2, unit=" EV", force_sign=True, label_width=120)
        self.slider_gnd_special_ev = self.row_gnd_special_ev.slider
        self.lbl_gnd_special_ev = self.row_gnd_special_ev.lbl_value
        self.slider_gnd_special_ev.valueChanged.connect(self._on_gnd_special_ev_changed)
        self.slider_gnd_special_ev.sliderReleased.connect(self.ground_slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_ev)

        # Sombras
        self.row_gnd_special_sh = PrecisionSliderRow("Sombras:", 0, 100, 0, scale=1.0, decimals=0, unit="%", label_width=120)
        self.slider_gnd_special_sh = self.row_gnd_special_sh.slider
        self.lbl_gnd_special_sh = self.row_gnd_special_sh.lbl_value
        self.slider_gnd_special_sh.valueChanged.connect(self._on_gnd_special_sh_changed)
        self.slider_gnd_special_sh.sliderReleased.connect(self.ground_slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_sh)

        # Punto Negro
        self.row_gnd_special_bp = PrecisionSliderRow("Punto Negro:", -50, 100, 0, scale=1000.0, decimals=3, force_sign=True, label_width=120)
        self.slider_gnd_special_bp = self.row_gnd_special_bp.slider
        self.lbl_gnd_special_bp = self.row_gnd_special_bp.lbl_value
        self.slider_gnd_special_bp.valueChanged.connect(self._on_gnd_special_bp_changed)
        self.slider_gnd_special_bp.sliderReleased.connect(self.ground_slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_bp)

        # Reducción Ruido Suelo (Clásico + IA NAFNet)
        self.btn_gnd_ai_denoise = QPushButton("✨ IA")
        self.btn_gnd_ai_denoise.setCheckable(True)
        self.btn_gnd_ai_denoise.setChecked(False)
        self.btn_gnd_ai_denoise.setCursor(Qt.PointingHandCursor)
        self.btn_gnd_ai_denoise.setFixedHeight(18)
        self.btn_gnd_ai_denoise.setToolTip("Reducción de Ruido Profunda con IA (NAFNet) para la superficie terrestre")
        self.btn_gnd_ai_denoise.setStyleSheet("""
            QPushButton {
                background-color: #1b3822;
                color: #a5d6a7;
                border: 1px solid #43a047;
                border-radius: 3px;
                padding: 1px 6px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #254d2f;
                border-color: #81c784;
                color: #ffffff;
            }
        """)
        self.btn_gnd_ai_denoise.clicked.connect(self.ground_ai_denoise_requested.emit)

        self.row_gnd_special_dn = PrecisionSliderRow(
            "Reducción Ruido:", 0, 100, 0, unit="%", label_width=120, action_button=self.btn_gnd_ai_denoise
        )
        self.slider_gnd_special_dn = self.row_gnd_special_dn.slider
        self.lbl_gnd_special_dn = self.row_gnd_special_dn.lbl_value
        self.slider_gnd_special_dn.valueChanged.connect(self._on_gnd_special_dn_changed)
        self.slider_gnd_special_dn.sliderReleased.connect(self.ground_slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_dn)

        # Anti-Halo y Desplazamiento de Borde en Frontera
        self.row_gnd_special_antihalo = PrecisionSliderRow(
            "Anti-Halo Frontera:", 0, 100, 0, scale=1.0, decimals=0, unit="%", label_width=120
        )
        self.slider_gnd_special_antihalo = self.row_gnd_special_antihalo.slider
        self.lbl_gnd_special_antihalo = self.row_gnd_special_antihalo.lbl_value
        self.slider_gnd_special_antihalo.valueChanged.connect(self._on_gnd_special_antihalo_changed)
        self.slider_gnd_special_antihalo.sliderReleased.connect(self.ground_slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_antihalo)

        self.row_gnd_special_edge_shift = PrecisionSliderRow(
            "Desplazar Borde:", -15, 15, 0, scale=1.0, decimals=0, unit=" px", force_sign=True, label_width=120
        )
        self.slider_gnd_special_edge_shift = self.row_gnd_special_edge_shift.slider
        self.lbl_gnd_special_edge_shift = self.row_gnd_special_edge_shift.lbl_value
        self.slider_gnd_special_edge_shift.valueChanged.connect(self._on_gnd_special_edge_shift_changed)
        self.slider_gnd_special_edge_shift.sliderReleased.connect(self.ground_slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_edge_shift)

        self.btn_reset_gnd_special = QPushButton("Restablecer Tono Suelo")
        self.btn_reset_gnd_special.setIcon(get_icon("rotate_ccw", color="#a5d6a7", size=13))
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

        self.row_stars_special = PrecisionSliderRow("Intensidad Estrellas:", 0, 200, 100, unit="%", label_width=130)
        self.slider_stars_special = self.row_stars_special.slider
        self.lbl_stars_special = self.row_stars_special.lbl_value
        self.row_stars_special.valueChanged.connect(self._on_stars_special_changed)
        self.row_stars_special.resetRequested.connect(lambda: self._on_stars_special_changed(self.row_stars_special.raw_default))
        self.row_stars_special.sliderReleased.connect(self.ground_slider_released.emit)
        self.row_stars_special.sliderReleased.connect(self._on_slider_interaction_ended)
        self.row_stars_special.sliderPressed.connect(self._on_slider_interaction_started)
        stars_spec_layout.addWidget(self.row_stars_special)

        # Filtro Difusor y Realce de Constelaciones (Star Mist)
        self.box_sl_star_diff = QGroupBox("Filtro Difusor y Realce de Constelaciones")
        self.box_sl_star_diff.setObjectName("sl_star_diff_box")
        self.box_sl_star_diff.setStyleSheet("""
            QGroupBox#sl_star_diff_box {
                font-weight: bold;
                border: 1px solid #fbc02d;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 14px;
                background-color: #1e1b12;
                color: #ffe082;
            }
            QGroupBox#sl_star_diff_box::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 6px;
                color: #fff9c4;
                font-weight: bold;
                font-size: 11px;
                background-color: #16130c;
                border-radius: 3px;
            }
        """)
        sl_sd_layout = QVBoxLayout(self.box_sl_star_diff)
        sl_sd_layout.setContentsMargins(8, 8, 8, 8)
        sl_sd_layout.setSpacing(6)

        # Intensidad Difusor
        self.row_sl_star_diff_strength = PrecisionSliderRow("Intensidad Difusor:", 0, 100, 0, unit="%", label_width=130)
        self.slider_sl_star_diff_strength = self.row_sl_star_diff_strength.slider
        self.lbl_sl_star_diff_strength = self.row_sl_star_diff_strength.lbl_value
        self.row_sl_star_diff_strength.valueChanged.connect(self._on_sl_star_diff_strength_changed)
        self.row_sl_star_diff_strength.resetRequested.connect(lambda: self._on_sl_star_diff_strength_changed(self.row_sl_star_diff_strength.raw_default))
        self.row_sl_star_diff_strength.sliderReleased.connect(self.ground_slider_released.emit)
        self.row_sl_star_diff_strength.sliderReleased.connect(self._on_slider_interaction_ended)
        self.row_sl_star_diff_strength.sliderPressed.connect(self._on_slider_interaction_started)
        sl_sd_layout.addWidget(self.row_sl_star_diff_strength)

        # Radio (Halo)
        self.row_sl_star_diff_radius = PrecisionSliderRow("Radio (Halo):", 5, 60, 20, unit=" px", label_width=130)
        self.slider_sl_star_diff_radius = self.row_sl_star_diff_radius.slider
        self.lbl_sl_star_diff_radius = self.row_sl_star_diff_radius.lbl_value
        self.row_sl_star_diff_radius.valueChanged.connect(self._on_sl_star_diff_radius_changed)
        self.row_sl_star_diff_radius.resetRequested.connect(lambda: self._on_sl_star_diff_radius_changed(self.row_sl_star_diff_radius.raw_default))
        self.row_sl_star_diff_radius.sliderReleased.connect(self.ground_slider_released.emit)
        self.row_sl_star_diff_radius.sliderReleased.connect(self._on_slider_interaction_ended)
        self.row_sl_star_diff_radius.sliderPressed.connect(self._on_slider_interaction_started)
        sl_sd_layout.addWidget(self.row_sl_star_diff_radius)

        # Umbral Magnitud
        self.row_sl_star_diff_threshold = PrecisionSliderRow("Umbral:", 15, 85, 45, scale=100.0, decimals=2, label_width=130)
        self.slider_sl_star_diff_threshold = self.row_sl_star_diff_threshold.slider
        self.lbl_sl_star_diff_threshold = self.row_sl_star_diff_threshold.lbl_value
        self.row_sl_star_diff_threshold.valueChanged.connect(self._on_sl_star_diff_threshold_changed)
        self.row_sl_star_diff_threshold.resetRequested.connect(lambda: self._on_sl_star_diff_threshold_changed(self.row_sl_star_diff_threshold.raw_default))
        self.row_sl_star_diff_threshold.sliderReleased.connect(self.ground_slider_released.emit)
        self.row_sl_star_diff_threshold.sliderReleased.connect(self._on_slider_interaction_ended)
        self.row_sl_star_diff_threshold.sliderPressed.connect(self._on_slider_interaction_started)
        sl_sd_layout.addWidget(self.row_sl_star_diff_threshold)

        # Color Espectral
        self.row_sl_star_diff_color = PrecisionSliderRow("Color Espectral:", 0, 100, 50, unit="%", label_width=130)
        self.slider_sl_star_diff_color = self.row_sl_star_diff_color.slider
        self.lbl_sl_star_diff_color = self.row_sl_star_diff_color.lbl_value
        self.row_sl_star_diff_color.valueChanged.connect(self._on_sl_star_diff_color_changed)
        self.row_sl_star_diff_color.resetRequested.connect(lambda: self._on_sl_star_diff_color_changed(self.row_sl_star_diff_color.raw_default))
        self.row_sl_star_diff_color.sliderReleased.connect(self.ground_slider_released.emit)
        self.row_sl_star_diff_color.sliderReleased.connect(self._on_slider_interaction_ended)
        self.row_sl_star_diff_color.sliderPressed.connect(self._on_slider_interaction_started)
        sl_sd_layout.addWidget(self.row_sl_star_diff_color)

        # Modo de Fusión
        row_sd_mode = QHBoxLayout()
        row_sd_mode.setContentsMargins(14, 0, 0, 0)
        lbl_sd_mode = QLabel("Fusión:")
        lbl_sd_mode.setFixedWidth(120)
        lbl_sd_mode.setStyleSheet("color: #94a3b8; font-size: 11px;")
        row_sd_mode.addWidget(lbl_sd_mode)
        self.combo_sl_star_diff_mode = QComboBox()
        self.combo_sl_star_diff_mode.addItem("Trama (Velo Etéreo)", "screen")
        self.combo_sl_star_diff_mode.addItem("Luz Suave (Aterciopelado)", "soft_light")
        self.combo_sl_star_diff_mode.addItem("Aditivo (Luminoso)", "additive")
        self.combo_sl_star_diff_mode.currentIndexChanged.connect(self._on_sl_star_diff_mode_changed)
        row_sd_mode.addWidget(self.combo_sl_star_diff_mode, stretch=1)
        sl_sd_layout.addLayout(row_sd_mode)

        stars_spec_layout.addWidget(self.box_sl_star_diff)

        self.btn_reset_stars_special = QPushButton("Restablecer Estrellas")
        self.btn_reset_stars_special.setIcon(get_icon("rotate_ccw", color="#80d8ff", size=13))
        self.btn_reset_stars_special.setCursor(Qt.PointingHandCursor)
        self.btn_reset_stars_special.clicked.connect(self.stars_reset_requested.emit)
        self.btn_reset_stars_special.clicked.connect(self.ground_slider_released.emit)
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

        # Claridad Fondo
        self.row_sl_clarity = PrecisionSliderRow("Claridad Fondo:", -200, 200, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_clarity = self.row_sl_clarity.slider
        self.lbl_sl_clarity = self.row_sl_clarity.lbl_value
        self.slider_sl_clarity.valueChanged.connect(self._on_sl_clarity_changed)
        self.slider_sl_clarity.sliderReleased.connect(self.ground_slider_released.emit)
        sl_layout.addWidget(self.row_sl_clarity)

        # Borrar Neblina
        self.row_sl_dehaze = PrecisionSliderRow("Borrar Neblina:", -200, 200, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_dehaze = self.row_sl_dehaze.slider
        self.lbl_sl_dehaze = self.row_sl_dehaze.lbl_value
        self.slider_sl_dehaze.valueChanged.connect(self._on_sl_dehaze_changed)
        self.slider_sl_dehaze.sliderReleased.connect(self.ground_slider_released.emit)
        sl_layout.addWidget(self.row_sl_dehaze)

        # Estructura Ondículas
        self.row_sl_ond_est = PrecisionSliderRow("Estructura:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_ond_est = self.row_sl_ond_est.slider
        self.lbl_sl_ond_est = self.row_sl_ond_est.lbl_value
        self.slider_sl_ond_est.valueChanged.connect(self._on_sl_ond_est_changed)
        self.slider_sl_ond_est.sliderReleased.connect(self.ground_slider_released.emit)
        sl_layout.addWidget(self.row_sl_ond_est)

        # Atenuar Fondo
        self.row_sl_ond_bg = PrecisionSliderRow("Atenuar Fondo:", 0, 100, 0, unit="%", label_width=130)
        self.slider_sl_ond_bg = self.row_sl_ond_bg.slider
        self.lbl_sl_ond_bg = self.row_sl_ond_bg.lbl_value
        self.slider_sl_ond_bg.valueChanged.connect(self._on_sl_ond_bg_changed)
        self.slider_sl_ond_bg.sliderReleased.connect(self.ground_slider_released.emit)
        sl_layout.addWidget(self.row_sl_ond_bg)

        # Contraste Fondo
        self.row_sl_contrast = PrecisionSliderRow("Contraste Fondo:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_contrast = self.row_sl_contrast.slider
        self.lbl_sl_contrast = self.row_sl_contrast.lbl_value
        self.slider_sl_contrast.valueChanged.connect(self._on_sl_contrast_changed)
        self.slider_sl_contrast.sliderReleased.connect(self.ground_slider_released.emit)
        sl_layout.addWidget(self.row_sl_contrast)

        # Reducción Ruido
        self.btn_sl_denoise_expand = QPushButton()
        self.btn_sl_denoise_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#ce93d8", size=11))
        self.btn_sl_denoise_expand.setCheckable(True)
        self.btn_sl_denoise_expand.setChecked(False)
        self.btn_sl_denoise_expand.setFixedSize(18, 18)
        self.btn_sl_denoise_expand.setStyleSheet("""
            QPushButton { border: none; background: transparent; padding: 0px; border-radius: 3px; }
            QPushButton:hover { background: #262a3a; }
            QPushButton:checked { background: #3b284c; }
        """)
        self.btn_sl_denoise_expand.setToolTip("Ajustes avanzados: Ruido de Color (Croma) y Máscara Bordes/Sombras")

        self.btn_sl_ai_denoise = QPushButton("✨ IA")
        self.btn_sl_ai_denoise.setCheckable(True)
        self.btn_sl_ai_denoise.setChecked(False)
        self.btn_sl_ai_denoise.setCursor(Qt.PointingHandCursor)
        self.btn_sl_ai_denoise.setFixedHeight(18)
        self.btn_sl_ai_denoise.setToolTip("Ejecutar Reducción de Ruido Profunda con IA (NAFNet)")
        self.btn_sl_ai_denoise.setStyleSheet("""
            QPushButton {
                background-color: #2b1f3d;
                color: #ce93d8;
                border: 1px solid #7b1fa2;
                border-radius: 3px;
                padding: 1px 6px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #3b2854;
                border-color: #ba68c8;
                color: #ffffff;
            }
        """)
        self.btn_sl_ai_denoise.clicked.connect(self.starless_ai_denoise_requested.emit)

        w_sl_dn_btns = QWidget()
        h_sl_dn_btns = QHBoxLayout(w_sl_dn_btns)
        h_sl_dn_btns.setContentsMargins(0, 0, 0, 0)
        h_sl_dn_btns.setSpacing(4)
        h_sl_dn_btns.addWidget(self.btn_sl_ai_denoise)
        h_sl_dn_btns.addWidget(self.btn_sl_denoise_expand)

        self.row_sl_denoise = PrecisionSliderRow(
            "Reducción Ruido:", 0, 100, 0, unit="%", label_width=130, action_button=w_sl_dn_btns
        )
        self.slider_sl_denoise = self.row_sl_denoise.slider
        self.lbl_sl_denoise = self.row_sl_denoise.lbl_value
        self.slider_sl_denoise.valueChanged.connect(self._on_sl_denoise_changed)
        self.slider_sl_denoise.sliderReleased.connect(self.ground_slider_released.emit)
        sl_layout.addWidget(self.row_sl_denoise)

        # Contenedor avanzado de Reducción de Ruido (Color y Máscara)
        self.w_sl_denoise_advanced = QWidget()
        self.w_sl_denoise_advanced.setVisible(False)
        layout_sl_dn_adv = QVBoxLayout(self.w_sl_denoise_advanced)
        layout_sl_dn_adv.setContentsMargins(8, 6, 8, 6)
        layout_sl_dn_adv.setSpacing(4)
        self.w_sl_denoise_advanced.setStyleSheet("background-color: #171420; border-radius: 5px; border: 1px solid #3b284c;")

        self.row_sl_denoise_color = PrecisionSliderRow("Ruido Color (Croma):", 0, 100, 50, unit="%", label_width=130)
        self.slider_sl_denoise_color = self.row_sl_denoise_color.slider
        self.lbl_sl_denoise_color = self.row_sl_denoise_color.lbl_value
        self.slider_sl_denoise_color.valueChanged.connect(self._on_sl_denoise_color_changed)
        self.slider_sl_denoise_color.sliderReleased.connect(self.ground_slider_released.emit)
        layout_sl_dn_adv.addWidget(self.row_sl_denoise_color)

        self.row_sl_denoise_mask = PrecisionSliderRow("Máscara Bordes/Sombras:", 0, 100, 50, unit="%", label_width=130)
        self.slider_sl_denoise_mask = self.row_sl_denoise_mask.slider
        self.lbl_sl_denoise_mask = self.row_sl_denoise_mask.lbl_value
        self.slider_sl_denoise_mask.valueChanged.connect(self._on_sl_denoise_mask_changed)
        self.slider_sl_denoise_mask.sliderReleased.connect(self.ground_slider_released.emit)
        layout_sl_dn_adv.addWidget(self.row_sl_denoise_mask)

        self.btn_sl_denoise_expand.toggled.connect(self.w_sl_denoise_advanced.setVisible)
        sl_layout.addWidget(self.w_sl_denoise_advanced)

        # Par 4: Efecto Orton Astronómico y Glow de Polvo Galáctico
        self.box_sl_orton = QGroupBox("Efecto Orton y Glow de Polvo Galáctico")
        self.box_sl_orton.setObjectName("sl_orton_box")
        self.box_sl_orton.setStyleSheet("""
            QGroupBox#sl_orton_box {
                font-weight: bold;
                border: 1px solid #8e24aa;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 14px;
                background-color: #191424;
                color: #ce93d8;
            }
            QGroupBox#sl_orton_box::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 6px;
                color: #f3e5f5;
                font-weight: bold;
                font-size: 11px;
                background-color: #130f1b;
                border-radius: 3px;
            }
        """)
        orton_layout = QVBoxLayout(self.box_sl_orton)
        orton_layout.setContentsMargins(8, 8, 8, 8)
        orton_layout.setSpacing(6)

        # Fila 1: Deslizador de Intensidad Orton
        self.row_sl_orton_strength = PrecisionSliderRow("Intensidad Glow:", 0, 100, 0, unit="%", label_width=130)
        self.slider_sl_orton_strength = self.row_sl_orton_strength.slider
        self.lbl_sl_orton_strength = self.row_sl_orton_strength.lbl_value
        self.slider_sl_orton_strength.valueChanged.connect(self._on_sl_orton_strength_changed)
        self.slider_sl_orton_strength.sliderReleased.connect(self.ground_slider_released.emit)
        orton_layout.addWidget(self.row_sl_orton_strength)

        # Radio (Difusión)
        self.row_sl_orton_radius = PrecisionSliderRow("Radio Difusión:", 10, 150, 40, unit=" px", label_width=130)
        self.slider_sl_orton_radius = self.row_sl_orton_radius.slider
        self.lbl_sl_orton_radius = self.row_sl_orton_radius.lbl_value
        self.slider_sl_orton_radius.valueChanged.connect(self._on_sl_orton_radius_changed)
        self.slider_sl_orton_radius.sliderReleased.connect(self.ground_slider_released.emit)
        orton_layout.addWidget(self.row_sl_orton_radius)

        # Umbral Polvo
        self.row_sl_orton_threshold = PrecisionSliderRow("Umbral Polvo:", 0, 50, 5, scale=100.0, decimals=2, label_width=130)
        self.slider_sl_orton_threshold = self.row_sl_orton_threshold.slider
        self.lbl_sl_orton_threshold = self.row_sl_orton_threshold.lbl_value
        self.slider_sl_orton_threshold.valueChanged.connect(self._on_sl_orton_threshold_changed)
        self.slider_sl_orton_threshold.sliderReleased.connect(self.ground_slider_released.emit)
        orton_layout.addWidget(self.row_sl_orton_threshold)

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

        # Fila Botón de Acción + Engranaje de opciones avanzadas de Estirado Adaptativo RGB
        row_as_header = QHBoxLayout()
        row_as_header.setSpacing(4)

        self.btn_sl_autostretch = QPushButton("Estirado Adaptativo RGB")
        self.btn_sl_autostretch.setIcon(get_icon("sparkles", color="#ce93d8", size=13))
        self.btn_sl_autostretch.setCursor(Qt.PointingHandCursor)
        self.btn_sl_autostretch.setToolTip(
            "Aplica el algoritmo de estirado no lineal adaptativo MTF sobre la capa sin estrellas,\n"
            "aislando el cálculo estadístico del fondo celeste respecto al suelo."
        )
        self.btn_sl_autostretch.setStyleSheet("""
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
        self.btn_sl_autostretch.clicked.connect(self.auto_stretch_starless_requested.emit)
        row_as_header.addWidget(self.btn_sl_autostretch, stretch=1)

        self.btn_sl_autostretch_expand = QPushButton()
        self.btn_sl_autostretch_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#ce93d8", size=12))
        self.btn_sl_autostretch_expand.setCheckable(True)
        self.btn_sl_autostretch_expand.setChecked(False)
        self.btn_sl_autostretch_expand.setFixedSize(28, 28)
        self.btn_sl_autostretch_expand.setToolTip("Despliega las opciones avanzadas de estirado: Fondo objetivo, corte sombras, saturación y canales.")
        self.btn_sl_autostretch_expand.setStyleSheet("""
            QPushButton { font-size: 11px; padding: 2px; border-radius: 4px; background: #262933; color: #cfd8dc; border: 1px solid #3b4252; }
            QPushButton:hover { background: #323846; border-color: #7b1fa2; }
            QPushButton:checked { background: #4a148c; color: #e1bee7; border: 1px solid #ba68c8; }
        """)
        row_as_header.addWidget(self.btn_sl_autostretch_expand)
        sl_layout.addLayout(row_as_header)

        # Panel colapsable de opciones avanzadas de Estirado Adaptativo RGB
        self.w_sl_autostretch_advanced = QWidget()
        self.w_sl_autostretch_advanced.setVisible(False)
        w_as_layout = QVBoxLayout(self.w_sl_autostretch_advanced)
        w_as_layout.setContentsMargins(4, 2, 4, 4)
        w_as_layout.setSpacing(4)

        self.row_sl_as_target_bg = PrecisionSliderRow("Fondo Objetivo:", 10, 30, 15, scale=100.0, decimals=2, unit="%", label_width=130)
        w_as_layout.addWidget(self.row_sl_as_target_bg)

        self.row_sl_as_sigma = PrecisionSliderRow("Corte Sombras:", 10, 50, 20, scale=10.0, decimals=1, unit=" σ", label_width=130)
        w_as_layout.addWidget(self.row_sl_as_sigma)

        self.row_sl_as_sat = PrecisionSliderRow("Saturación:", 10, 40, 25, scale=10.0, decimals=1, unit="x", label_width=130)
        w_as_layout.addWidget(self.row_sl_as_sat)

        self.row_sl_as_feather = PrecisionSliderRow("Suavizado Borde:", 0, 10, 2, unit=" px", label_width=130)
        w_as_layout.addWidget(self.row_sl_as_feather)

        self.chk_sl_as_unlinked = QCheckBox("Canales RGB independientes (neutraliza dominante)")
        self.chk_sl_as_unlinked.setChecked(True)
        self.chk_sl_as_unlinked.setToolTip(
            "Activado: Calcula MTF independiente por canal R, G y B (neutraliza dominantes en nebulosas).\n"
            "Desactivado: Usa curva conjunta compartida (conserva balance de color y gradientes del horizonte)."
        )
        self.chk_sl_as_unlinked.setStyleSheet("color: #ce93d8; font-size: 11px; margin-left: 12px; margin-top: 2px;")
        w_as_layout.addWidget(self.chk_sl_as_unlinked)

        self.btn_sl_autostretch_expand.toggled.connect(self.w_sl_autostretch_advanced.setVisible)
        sl_layout.addWidget(self.w_sl_autostretch_advanced)

        self.btn_reset_sl_special = QPushButton("Restablecer Ajustes de Fondo")
        self.btn_reset_sl_special.setIcon(get_icon("rotate_ccw", color="#ce93d8", size=13))
        self.btn_reset_sl_special.setCursor(Qt.PointingHandCursor)
        self.btn_reset_sl_special.clicked.connect(self.starless_reset_requested.emit)
        sl_layout.addWidget(self.btn_reset_sl_special)

        props_layout.addWidget(self.grp_starless_special)
        main_layout.addWidget(self.container_layer_props)

        # Conectar todos los deslizadores para coordinación fluida de interacción
        self._connect_all_slider_interactions()

        # Estado inicial vacío
        self._update_list_height()
        self._update_list_and_props_state()

    def _update_list_height(self) -> None:
        """Ajusta la altura del list widget de capas según el modo de miniatura y número de capas."""
        if self._custom_list_height and self._user_list_height is not None:
            self.list_layers.setFixedHeight(self._user_list_height)
            return
        spec = THUMB_SPECS.get(self._thumb_size_mode, THUMB_SPECS["M"])
        row_h = spec["row_height"]
        n_layers = max(1, len(self.layers))
        needed_h = n_layers * row_h + 8
        auto_h = max(80, min(500, needed_h))
        self.list_layers.setFixedHeight(auto_h)

    def _on_list_resize_dragged(self, delta: int) -> None:
        """Manejador para el arrastre del tirador de redimensionado de la lista de capas."""
        self._custom_list_height = True
        cur_h = self.list_layers.height()
        new_h = max(70, min(850, cur_h + delta))
        self._user_list_height = new_h
        self.list_layers.setFixedHeight(new_h)

    def _on_list_auto_fit_requested(self) -> None:
        """Restaura el tamaño automático para que quepan todas las capas tras hacer doble clic."""
        self._custom_list_height = False
        self._user_list_height = None
        self._update_list_height()

    def get_all_sliders(self) -> List[QSlider]:
        """Retorna todos los controles deslizantes activos del panel de capas."""
        sliders = []
        slider_attrs = [
            'slider_opacity', 'slider_b_radius', 'slider_b_hardness', 'slider_b_flow',
            'slider_hue', 'slider_adj_ev', 'slider_adj_contrast', 'slider_adj_shadows',
            'slider_adj_highlights', 'slider_adj_sat', 'slider_adj_temp', 'slider_adj_tint',
            'slider_gnd_special_ev', 'slider_gnd_special_sh', 'slider_gnd_special_bp',
            'slider_gnd_special_dn', 'slider_gnd_special_antihalo', 'slider_gnd_special_edge_shift',
            'slider_stars_special', 'slider_sl_star_diff_strength', 'slider_sl_star_diff_radius',
            'slider_sl_star_diff_threshold', 'slider_sl_star_diff_color', 'slider_sl_clarity',
            'slider_sl_dehaze', 'slider_sl_ond_est', 'slider_sl_ond_bg', 'slider_sl_contrast',
            'slider_sl_denoise', 'slider_sl_denoise_color', 'slider_sl_denoise_mask',
            'slider_sl_orton_strength', 'slider_sl_orton_radius',
            'slider_sl_orton_threshold'
        ]
        for attr in slider_attrs:
            s = getattr(self, attr, None)
            if s is not None and isinstance(s, QSlider):
                sliders.append(s)
        return sliders

    def _connect_all_slider_interactions(self) -> None:
        """Conecta eventos de pulsación y liberación para coordinar modo draft e histogramas."""
        for s in self.get_all_sliders():
            s.sliderPressed.connect(self._on_slider_interaction_started)
            s.sliderReleased.connect(self._on_slider_interaction_ended)

    def _on_slider_interaction_started(self) -> None:
        self._is_scrubbing = True
        self.interaction_started.emit()

    def _on_slider_interaction_ended(self) -> None:
        self._is_scrubbing = False
        layer = self.get_selected_layer()
        if layer:
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=True)
        self.interaction_finished.emit()
        self.update_active_layer_histogram()

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
            new_layer.id = str(uuid.uuid4())[:8]
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

    def set_thumbnail_size_mode(self, mode: str) -> None:
        """Cambia el tamaño de las miniaturas de capa ('S', 'M', 'L')."""
        if mode not in THUMB_SPECS:
            return
        if self._thumb_size_mode == mode:
            return
        self._thumb_size_mode = mode
        if hasattr(self, 'btn_size_s'):
            self.btn_size_s.setChecked(mode == "S")
            self.btn_size_m.setChecked(mode == "M")
            self.btn_size_l.setChecked(mode == "L")
        
        if not self._custom_list_height:
            self._update_list_height()

        self._rebuild_list_items(select_layer_id=self._selected_layer_id)

    def invalidate_layer_thumbnail(self, layer_id: Optional[str] = None, rebuild_now: bool = True) -> None:
        """Invalida la caché de miniaturas de una capa específica o de todas las capas."""
        for layer in self.layers:
            if layer_id is None or layer.id == layer_id:
                if hasattr(layer, "_thumb_pixmap_cache"):
                    layer._thumb_pixmap_cache.clear()
        if rebuild_now and not self._updating_ui:
            if hasattr(self, '_thumb_rebuild_timer') and self._thumb_rebuild_timer.isActive():
                self._thumb_rebuild_timer.stop()
            self._rebuild_list_items(select_layer_id=self._selected_layer_id)
        elif not rebuild_now and not self._updating_ui:
            if hasattr(self, '_thumb_rebuild_timer'):
                self._thumb_rebuild_timer.start(250)

    def _on_mask_param_interaction_finished(self) -> None:
        """Al finalizar el arrastre del ratón o la edición de parámetros de máscara, actualiza la miniatura."""
        if hasattr(self, '_thumb_rebuild_timer') and self._thumb_rebuild_timer.isActive():
            self._thumb_rebuild_timer.stop()
        if not self._updating_ui:
            self._rebuild_list_items(select_layer_id=self._selected_layer_id)

    def _on_thumb_double_clicked(self, layer: MaskLayer) -> None:
        """Al hacer doble clic en la miniatura, conmuta la previsualización de su máscara en el lienzo."""
        if hasattr(self, 'btn_preview_mask'):
            self.btn_preview_mask.toggle()

    def _get_image_aspect_ratio(self) -> float:
        """
        Determina la relación de aspecto real (ancho / alto) de la imagen de trabajo.
        Retorna 1.5 (3:2 horizontal) por defecto si aún no hay imagen cargada.
        """
        raw = None
        if self._image_reference_getter:
            try:
                raw = self._image_reference_getter()
            except Exception:
                raw = None
        if raw is None and self._starless_getter:
            try:
                raw = self._starless_getter()
            except Exception:
                raw = None
        if raw is None and self._stars_getter:
            try:
                raw = self._stars_getter()
            except Exception:
                raw = None
        if raw is None and self._master_mask_getter:
            try:
                raw = self._master_mask_getter()
            except Exception:
                raw = None
        if raw is not None and hasattr(raw, "shape") and len(raw.shape) >= 2:
            h, w = raw.shape[:2]
            if h > 0 and w > 0:
                return float(w) / float(h)
        return 1.5

    def _get_thumbnail_dimensions(self) -> Tuple[int, int]:
        """
        Calcula las dimensiones (ancho, alto) de la miniatura respetando la orientación
        y relación de aspecto real de la imagen (horizontal 3:2, vertical 2:3, cuadrada 1:1, etc.).
        """
        spec = THUMB_SPECS.get(self._thumb_size_mode, THUMB_SPECS["M"])
        max_w = spec.get("max_w", spec.get("width", 42))
        max_h = spec.get("max_h", spec.get("height", 34))
        ar = self._get_image_aspect_ratio()

        box_ar = max_w / float(max_h)
        if ar >= box_ar:
            # Orientación horizontal / panorámica
            tw = max_w
            th = max(12, int(round(tw / ar)))
        else:
            # Orientación vertical / cuadrada
            th = max_h
            tw = max(12, int(round(th * ar)))

        return tw, th

    def _get_layer_thumbnail_pixmap(self, layer: MaskLayer) -> QPixmap:
        """
        Genera o recupera de la caché perezosa la miniatura de la capa según el modo actual (S/M/L)
        y la orientación real de la imagen.
        Capas de imagen: RGB real a escala.
        Capas de ajuste/máscaras: Máscara en escala de grises [0..255] (B/N).
        """
        mode = self._thumb_size_mode
        tw, th = self._get_thumbnail_dimensions()

        cache = getattr(layer, "_thumb_pixmap_cache", None)
        if cache is None:
            layer._thumb_pixmap_cache = cache = {}

        cache_key = (mode, tw, th, layer.enabled, layer.inverted)
        if cache_key in cache:
            return cache[cache_key]

        pix = None
        # --- CASO 1: CAPAS DE IMAGEN (RGB) ---
        if layer.layer_kind == LAYER_KIND_IMAGE:
            raw_img = None
            mask_img = None
            if self._layer_image_provider is not None:
                try:
                    img_prov, m_prov, _ = self._layer_image_provider(layer)
                    if img_prov is not None:
                        raw_img = img_prov
                        mask_img = m_prov
                except Exception:
                    raw_img = None

            if raw_img is None:
                if layer.mask_type == MASK_TYPE_IMAGE_STARS:
                    raw_img = self._stars_getter() if self._stars_getter else None
                elif layer.mask_type == MASK_TYPE_IMAGE_STARLESS:
                    raw_img = self._starless_getter() if self._starless_getter else None
                    if raw_img is None and self._image_reference_getter:
                        raw_img = self._image_reference_getter()
                else:
                    raw_img = self._image_reference_getter() if self._image_reference_getter else None

            if raw_img is not None:
                try:
                    # En la capa de suelo, aislar el terreno usando la máscara para evitar que el cielo sature la miniatura
                    if mask_img is not None and layer.mask_type == MASK_TYPE_IMAGE_GROUND:
                        m_3d = mask_img if mask_img.ndim == 3 else mask_img[..., np.newaxis]
                        if m_3d.shape[:2] != raw_img.shape[:2]:
                            m_3d = cv2.resize(m_3d, (raw_img.shape[1], raw_img.shape[0]))
                            if m_3d.ndim == 2:
                                m_3d = m_3d[..., np.newaxis]
                        raw_img = raw_img * m_3d

                    small = cv2.resize(raw_img, (tw, th), interpolation=cv2.INTER_AREA)
                    # Si es una capa de imagen (que no sean solo estrellas ni suelo aislado) y sus valores son lineales oscuros, estirar para visibilidad
                    if layer.mask_type not in (MASK_TYPE_IMAGE_STARS, MASK_TYPE_IMAGE_GROUND) and small.ndim == 3 and np.percentile(small, 98) < 0.12:
                        from core.stretch import manual_stretch
                        small = manual_stretch(small, black_point=0.0, midtone=0.15)
                    u8 = (np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)
                    u8_contig = np.ascontiguousarray(u8)
                    qimg = QImage(u8_contig.data, tw, th, tw * 3, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg)
                except Exception:
                    pix = None

        # --- CASO 2: CAPAS DE AJUSTE / MÁSCARAS (B/N) ---
        else:
            m = None
            if layer.mask_type == MASK_TYPE_BRUSH:
                m = layer.brush_mask
            else:
                base_img = self._image_reference_getter() if self._image_reference_getter else None
                if base_img is not None:
                    try:
                        m = compute_effective_layer_mask(
                            layer,
                            base_img,
                            master_sky_mask=self._master_mask_getter() if self._master_mask_getter else None,
                            starless_image=self._starless_getter() if self._starless_getter else None,
                            stars_image=self._stars_getter() if self._stars_getter else None
                        )
                    except Exception:
                        m = None

            if m is not None:
                try:
                    if layer.mask_type == MASK_TYPE_BRUSH and layer.inverted:
                        m_eval = 1.0 - m
                    else:
                        m_eval = m

                    small = cv2.resize(m_eval, (tw, th), interpolation=cv2.INTER_AREA)
                    u8 = (np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)
                    u8_contig = np.ascontiguousarray(u8)
                    qimg = QImage(u8_contig.data, tw, th, tw, QImage.Format_Grayscale8)
                    pix = QPixmap.fromImage(qimg)
                except Exception:
                    pix = None
            else:
                # Placeholder para máscara de pincel vacía
                pix = QPixmap(tw, th)
                pix.fill(QColor("#0d1017"))
                painter = QPainter(pix)
                painter.setPen(QColor("#222a3d"))
                painter.drawRect(0, 0, tw - 1, th - 1)
                b_ico = get_svg_pixmap("brush", color="#3b4866", size=max(10, min(tw, th) - 8))
                if not b_ico.isNull():
                    painter.drawPixmap((tw - b_ico.width()) // 2, (th - b_ico.height()) // 2, b_ico)
                painter.end()

        # Fallback genérico si pix sigue siendo None
        if pix is None or pix.isNull():
            pix = QPixmap(tw, th)
            pix.fill(QColor("#0a0c10"))

        # Si la capa está deshabilitada (ojo apagado), atenuar con opacidad reducida
        if not layer.enabled:
            dimmed = QPixmap(pix.size())
            dimmed.fill(Qt.transparent)
            p = QPainter(dimmed)
            p.setOpacity(0.35)
            p.drawPixmap(0, 0, pix)
            p.end()
            pix = dimmed

        cache[cache_key] = pix
        return pix

    def _rebuild_list_items(self, select_layer_id: Optional[str] = None) -> None:
        """Regenera los elementos de la lista QListWidget con miniaturas, iconos, botón de ojo y estilos acordes a su tipo."""
        self._updating_ui = True
        self.list_layers.clear()
        self._update_list_height()

        spec = THUMB_SPECS.get(self._thumb_size_mode, THUMB_SPECS["M"])
        tw, th = self._get_thumbnail_dimensions()
        row_h = spec["row_height"]
        ico_sz = spec["icon_size"]
        title_pt = spec["title_pt"]

        select_row = -1
        for i, layer in enumerate(self.layers):
            if layer.mask_type == MASK_TYPE_IMAGE_BASE:
                icon_name = "file"
                type_tag = "Fondo Base"
                fg_col = "#ffd54f" if layer.enabled else "#607d8b"
            elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
                icon_name = "star"
                type_tag = "StarNet AI"
                fg_col = "#80d8ff" if layer.enabled else "#607d8b"
            elif layer.mask_type == MASK_TYPE_IMAGE_STARLESS:
                icon_name = "sparkles"
                type_tag = "StarNet AI"
                fg_col = "#ce93d8" if layer.enabled else "#607d8b"
            elif layer.mask_type == MASK_TYPE_IMAGE_GROUND:
                icon_name = "mountain"
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

                icon_name = {
                    MASK_TYPE_BRUSH: "brush",
                    MASK_TYPE_LUMA: "sparkles",
                    MASK_TYPE_COLOR: "palette",
                    MASK_TYPE_GROUND: "mountain",
                    MASK_TYPE_SKY: "sparkles",
                }.get(layer.mask_type, "layers")
                fg_col = "#e0e6f0" if layer.enabled else "#607d8b"

            item = QListWidgetItem()
            item.setSizeHint(QSize(0, row_h))
            self.list_layers.addItem(item)

            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 2, 4, 2)
            row_layout.setSpacing(6)

            # 1. Miniatura visual (RGB para imagen, B/N para máscaras)
            pix = self._get_layer_thumbnail_pixmap(layer)
            lbl_thumb = QLabel()
            lbl_thumb.setFixedSize(tw + 2, th + 2)
            border_col = "#28324a" if layer.enabled else "#161b26"
            lbl_thumb.setStyleSheet(f"""
                QLabel {{
                    background-color: #07090e;
                    border: 1px solid {border_col};
                    border-radius: 3px;
                }}
            """)
            lbl_thumb.setAlignment(Qt.AlignCenter)
            lbl_thumb.setPixmap(pix)
            lbl_thumb.setCursor(Qt.PointingHandCursor)
            lbl_thumb.setToolTip("Miniatura de capa.\nHaz doble clic para previsualizar la máscara en el lienzo.")
            row_layout.addWidget(lbl_thumb)

            # 2. Micro-icono de tipo
            lbl_icon = QLabel()
            lbl_icon.setStyleSheet("background: transparent;")
            pix_ico = get_svg_pixmap(icon_name, color=fg_col, size=ico_sz)
            if not pix_ico.isNull():
                lbl_icon.setPixmap(pix_ico)
            lbl_icon.setFixedSize(ico_sz + 2, ico_sz + 2)
            row_layout.addWidget(lbl_icon)

            # 3. Título de la capa
            bold_css = "font-weight: bold; " if layer.is_system_layer else ""
            lbl_title = QLabel(f"{layer.name}  [{type_tag}]")
            f = lbl_title.font()
            f.setPointSize(title_pt)
            if layer.is_system_layer:
                f.setBold(True)
            lbl_title.setFont(f)
            lbl_title.setStyleSheet(f"color: {fg_col}; {bold_css}background: transparent;")
            lbl_title.setCursor(Qt.PointingHandCursor)
            row_layout.addWidget(lbl_title, stretch=1)

            # 4. Botón de visibilidad (Ojo)
            btn_eye = QPushButton()
            eye_color = fg_col if layer.enabled else "#78909c"
            btn_eye.setIcon(get_icon("eye" if layer.enabled else "eye_off", color=eye_color, size=14))
            btn_eye.setFixedSize(26, 22)
            btn_eye.setCursor(Qt.PointingHandCursor)
            btn_eye.setToolTip("Capa visible (clic para ocultar)" if layer.enabled else "Capa oculta (clic para mostrar)")
            btn_eye.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    padding: 0px;
                }
                QPushButton:hover {
                    background-color: #2a3147;
                    border-radius: 3px;
                }
            """)

            def make_click_handler(layer_id, r):
                def _handler(e):
                    self.list_layers.setCurrentRow(r)
                    if self._selected_layer_id != layer_id:
                        self._selected_layer_id = layer_id
                        self._update_list_and_props_state()
                return _handler

            def make_eye_handler(r):
                return lambda: self._toggle_layer_enabled_at(r)

            def make_thumb_double_click_handler(l):
                return lambda e: self._on_thumb_double_clicked(l)

            h = make_click_handler(layer.id, i)
            row_widget.mousePressEvent = h
            lbl_title.mousePressEvent = h
            lbl_thumb.mousePressEvent = h
            lbl_icon.mousePressEvent = h
            lbl_thumb.mouseDoubleClickEvent = make_thumb_double_click_handler(layer)
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
            self.invalidate_layer_thumbnail(layer.id)

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
            self.update_active_layer_histogram()
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
        if hasattr(self, 'resize_handle'):
            self.resize_handle.setVisible(len(self.layers) > 0)
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
        try:
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

            # Ocultar Temperatura y Tinte en Capa de Estrellas para preservar color espectral intrínseco
            if hasattr(self, 'row_adj_temp'):
                self.row_adj_temp.setVisible(not is_stars)
            if hasattr(self, 'row_adj_tint'):
                self.row_adj_tint.setVisible(not is_stars)

            if is_stars and layer:
                layer.temp = 0.0
                layer.tint = 0.0

            # Temperatura y Tinte WB (-1.0000 a +1.0000 con 4 decimales)
            self.spin_adj_temp.setValue(layer.temp)
            self.slider_adj_temp.setValue(int(round(layer.temp * 10000.0)))

            self.spin_adj_tint.setValue(layer.tint)
            self.slider_adj_tint.setValue(int(round(layer.tint * 10000.0)))

            # Curva tonal interactiva de capa
            if hasattr(self, 'layer_curve_widget'):
                pts = getattr(layer, 'curve_points', None)
                if pts:
                    self.layer_curve_widget.set_state({"rgb": pts})
                else:
                    self.layer_curve_widget.reset_curve("rgb")
                self.update_active_layer_histogram()

        finally:
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
            adj_title = "Ajustes de Capa de Suelo (Post-Estirado)"
            mask_def_title = "Definición de Máscara de Suelo"
        elif mtype == MASK_TYPE_IMAGE_STARLESS:
            border_c = "#8e24aa"
            title_c = "#ce93d8"
            bg_c = "#1c1424"
            adj_title = "Ajustes de Capa de Fondo Nebular"
            mask_def_title = "Definición de Máscara"
        elif mtype == MASK_TYPE_IMAGE_STARS:
            border_c = "#fbc02d"
            title_c = "#ffd54f"
            bg_c = "#221f14"
            adj_title = "Ajustes de Capa de Estrellas"
            mask_def_title = "Definición de Máscara"
        elif mtype == MASK_TYPE_BRUSH:
            border_c = "#e53935"
            title_c = "#ef5350"
            bg_c = "#221415"
            adj_title = "Ajustes Locales de Pincel"
            mask_def_title = "Trazo y Parámetros de Pincel"
        elif mtype == MASK_TYPE_LUMA:
            border_c = "#fdd835"
            title_c = "#fff59d"
            bg_c = "#202018"
            adj_title = "Ajustes Locales de Luminancia"
            mask_def_title = "Rango de Luminancia"
        elif mtype == MASK_TYPE_COLOR:
            border_c = "#00acc1"
            title_c = "#80d8ff"
            bg_c = "#141f24"
            adj_title = "Ajustes Locales de Color HSV"
            mask_def_title = "Tono y Tolerancia de Color"
        elif mtype == MASK_TYPE_SKY:
            border_c = "#1976d2"
            title_c = "#90caf9"
            bg_c = "#141924"
            adj_title = "Ajustes de Capa de Cielo"
            mask_def_title = "Definición de Máscara de Cielo"
        elif mtype == MASK_TYPE_IMAGE_BASE:
            border_c = "#ffa000"
            title_c = "#ffe082"
            bg_c = "#1f1c14"
            adj_title = "Ajustes de Imagen Base"
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
            self.grp_gnd_special.setTitle("Ajustes Tonales Especiales de Suelo (Pre-Estirado Lineal)")
        if hasattr(self, 'grp_stars_special'):
            self.grp_stars_special.setStyleSheet(_make_box_qss("stars_special_group", "#fbc02d", "#ffd54f", "#221f14"))
            self.grp_stars_special.setTitle("Ajustes Especiales de Estrellas")
        if hasattr(self, 'grp_starless_special'):
            self.grp_starless_special.setStyleSheet(_make_box_qss("starless_special_group", "#8e24aa", "#ce93d8", "#1c1424"))
            self.grp_starless_special.setTitle("Ajustes Especiales de Fondo Nebular")

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
            self.invalidate_layer_thumbnail(layer.id)
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
            self.update_active_layer_histogram()
            self.layers_changed.emit()

    def _on_layer_invert_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.inverted = checked
            self.invalidate_layer_thumbnail(layer.id)
            self._rebuild_list_items(select_layer_id=layer.id)
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
            elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
                self.starnet_opacity_changed.emit(layer.opacity)
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
            self.invalidate_layer_thumbnail(layer.id)
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
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _set_luma_preset(self, l_min: float, l_max: float) -> None:
        self.spin_luma_min.setValue(l_min)
        self.spin_luma_max.setValue(l_max)
        self._on_mask_param_interaction_finished()

    # Controles de Rango de Color
    def _on_hue_slider_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_hue_val.setText(f"{val}°")
        self._update_color_swatch(val)
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_COLOR:
            layer.color_hue = float(val)
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _on_color_param_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and layer.mask_type == MASK_TYPE_COLOR:
            layer.color_tolerance = float(self.spin_color_tol.value())
            layer.color_min_sat = float(self.spin_color_sat.value())
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            if self.btn_preview_mask.isChecked():
                self._emit_mask_preview()
            self.layers_changed.emit()

    def _set_color_preset(self, hue: int, tol: int) -> None:
        self.slider_hue.setValue(hue)
        self.spin_color_tol.setValue(tol)
        self._on_mask_param_interaction_finished()

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
            self.invalidate_layer_thumbnail(layer.id)
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
            self.update_active_layer_histogram()
            self.layers_changed.emit()

    def _on_adj_contrast_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.contrast = val / 100.0
            self.lbl_adj_contrast.setText(f"{layer.contrast:+.2f}")
            self.update_active_layer_histogram()
            self.layers_changed.emit()

    def _on_adj_shadows_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.shadows = val / 100.0
            self.lbl_adj_shadows.setText(f"{layer.shadows:+.2f}")
            self.update_active_layer_histogram()
            self.layers_changed.emit()

    def _on_adj_highlights_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.highlights = val / 100.0
            self.lbl_adj_highlights.setText(f"{layer.highlights:+.2f}")
            self.update_active_layer_histogram()
            self.layers_changed.emit()

    def _on_adj_sat_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.saturation = val / 100.0
            self.lbl_adj_sat.setText(f"{layer.saturation:.2f}x")
            self.update_active_layer_histogram()
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
            self.update_active_layer_histogram()
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
            self.update_active_layer_histogram()
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
            self.update_active_layer_histogram()
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
            self.update_active_layer_histogram()
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
        layer.curve_points = [(0.0, 0.0), (1.0, 1.0)]
        layer.curve_lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        if hasattr(self, 'layer_curve_widget'):
            self._updating_ui = True
            self.layer_curve_widget.reset_curve("rgb")
            self._updating_ui = False
        self._update_list_and_props_state()
        self.update_active_layer_histogram()
        self.ground_slider_released.emit()
        self.layers_changed.emit()

    def update_active_layer_histogram(self) -> None:
        """Actualiza el histograma de fondo de la curva según la capa activa."""
        if getattr(self, '_is_scrubbing', False):
            return
        if not hasattr(self, 'layer_curve_widget') or not hasattr(self, 'sec_layer_curve') or not self.sec_layer_curve.is_expanded:
            return
        layer = self.get_selected_layer()
        if not layer:
            self.layer_curve_widget.set_histogram_from_image(None)
            return

        if getattr(self, '_layer_image_provider', None) is not None:
            img, mask, is_stars = self._layer_image_provider(layer)
            self.layer_curve_widget.set_histogram_from_image(img, mask=mask, is_stars_layer=is_stars)
            return

        img = None
        if layer.mask_type == MASK_TYPE_IMAGE_STARS and self._stars_getter:
            img = self._stars_getter()
        elif layer.mask_type == MASK_TYPE_IMAGE_STARLESS and self._starless_getter:
            img = self._starless_getter()
        elif self._image_reference_getter:
            img = self._image_reference_getter()

        self.layer_curve_widget.set_histogram_from_image(img)

    def _on_layer_curve_changed(self) -> None:
        """Actualiza los puntos y LUT de la curva de la capa seleccionada."""
        if self._updating_ui:
            return
        layer = self.get_selected_layer()
        if layer and hasattr(self, 'layer_curve_widget'):
            layer.curve_points = [tuple(p) for p in self.layer_curve_widget.points]
            layer.curve_lut = self.layer_curve_widget.lut.copy()
            self.layers_changed.emit()

    def _reset_layer_curve(self) -> None:
        """Restablece la curva tonal de la capa activa a su estado lineal neutro."""
        layer = self.get_selected_layer()
        if layer:
            layer.curve_points = [(0.0, 0.0), (1.0, 1.0)]
            layer.curve_lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            if hasattr(self, 'layer_curve_widget'):
                self._updating_ui = True
                self.layer_curve_widget.reset_curve("rgb")
                self._updating_ui = False
            self.layers_changed.emit()
            self.ground_slider_released.emit()

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

    def set_ground_tone_values(self, ev_val: int, sh_val: int, bp_val: int, dn_val: int = 0, antihalo_val: int = 0, edge_shift_val: int = 0) -> None:
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
        if hasattr(self, 'slider_gnd_special_dn'):
            self.slider_gnd_special_dn.setValue(dn_val)
            self.lbl_gnd_special_dn.setText(f"{dn_val}%")
        if hasattr(self, 'slider_gnd_special_antihalo'):
            self.slider_gnd_special_antihalo.setValue(antihalo_val)
            self.lbl_gnd_special_antihalo.setText(f"{antihalo_val}%")
        if hasattr(self, 'slider_gnd_special_edge_shift'):
            self.slider_gnd_special_edge_shift.setValue(edge_shift_val)
            self.lbl_gnd_special_edge_shift.setText(f"{edge_shift_val:+d} px" if edge_shift_val != 0 else "0 px")
        self._updating_ui = False

    def _on_gnd_special_dn_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_gnd_special_dn.setText(f"{val}%")
        self.ground_denoise_changed.emit(val)

    def _on_gnd_special_antihalo_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_gnd_special_antihalo.setText(f"{val}%")
        self.ground_antihalo_changed.emit(val)

    def _on_gnd_special_edge_shift_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_gnd_special_edge_shift.setText(f"{val:+d} px" if val != 0 else "0 px")
        self.ground_edge_shift_changed.emit(val)

    def set_ground_ai_denoise_active(self, active: bool):
        """Actualiza el botón IA de la capa de suelo."""
        if not hasattr(self, 'btn_gnd_ai_denoise'):
            return
        self.btn_gnd_ai_denoise.setChecked(active)
        if active:
            self.btn_gnd_ai_denoise.setText("✓ IA")
            self.btn_gnd_ai_denoise.setStyleSheet("""
                QPushButton {
                    background-color: #1b3822;
                    color: #a5d6a7;
                    border: 1px solid #43a047;
                    border-radius: 3px;
                    padding: 1px 6px;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #2e593e;
                    color: #ffffff;
                }
            """)
            self.btn_gnd_ai_denoise.setToolTip("Modo IA ACTIVO en Suelo. Haz clic para alternar al modo Clásico.")
        else:
            self.btn_gnd_ai_denoise.setText("✨ IA")
            self.btn_gnd_ai_denoise.setStyleSheet("""
                QPushButton {
                    background-color: #1b3822;
                    color: #a5d6a7;
                    border: 1px solid #43a047;
                    border-radius: 3px;
                    padding: 1px 6px;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #254d2f;
                    border-color: #81c784;
                    color: #ffffff;
                }
            """)
            self.btn_gnd_ai_denoise.setToolTip("Modo Clásico ACTIVO en Suelo. Haz clic para activar o calcular Modo IA (NAFNet).")

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
        try:
            if presence is not None and hasattr(self, 'row_stars_special'):
                self.row_stars_special.setValue(presence)
                self.lbl_stars_special.setText(f"{presence}%")
            if diff_strength is not None and hasattr(self, 'row_sl_star_diff_strength'):
                self.row_sl_star_diff_strength.setValue(diff_strength)
                self.lbl_sl_star_diff_strength.setText(f"{diff_strength}%")
            if diff_radius is not None and hasattr(self, 'row_sl_star_diff_radius'):
                self.row_sl_star_diff_radius.setValue(diff_radius)
                self.lbl_sl_star_diff_radius.setText(f"{diff_radius} px")
            if diff_threshold is not None and hasattr(self, 'row_sl_star_diff_threshold'):
                self.row_sl_star_diff_threshold.setValue(diff_threshold)
                self.lbl_sl_star_diff_threshold.setText(f"{diff_threshold / 100.0:.2f}")
            if diff_color is not None and hasattr(self, 'row_sl_star_diff_color'):
                self.row_sl_star_diff_color.setValue(diff_color)
                self.lbl_sl_star_diff_color.setText(f"{diff_color}%")
            if diff_mode_idx is not None and hasattr(self, 'combo_sl_star_diff_mode'):
                self.combo_sl_star_diff_mode.setCurrentIndex(diff_mode_idx)
        finally:
            self._updating_ui = False

    def set_stars_special_value(self, val: int) -> None:
        """Sincroniza el control especial de intensidad de estrellas desde tab_developer (compatibilidad)."""
        self.set_stars_special_values(presence=val)

    def get_autostretch_params(self) -> dict:
        """Retorna los parámetros actuales configurados para el Estirado Adaptativo RGB."""
        return {
            "target_bg": self.row_sl_as_target_bg.value() / 100.0 if hasattr(self, "row_sl_as_target_bg") else 0.15,
            "shadow_clip_sigma": self.row_sl_as_sigma.value() / 10.0 if hasattr(self, "row_sl_as_sigma") else 2.0,
            "saturation_boost": self.row_sl_as_sat.value() / 10.0 if hasattr(self, "row_sl_as_sat") else 2.5,
            "feather_radius": self.row_sl_as_feather.value() if hasattr(self, "row_sl_as_feather") else 2,
            "unlinked": self.chk_sl_as_unlinked.isChecked() if hasattr(self, "chk_sl_as_unlinked") else True,
        }

    def set_autostretch_params(self, params: dict) -> None:
        """Configura los controles del panel de Estirado Adaptativo RGB."""
        if "target_bg" in params and hasattr(self, "row_sl_as_target_bg"):
            self.row_sl_as_target_bg.setValue(int(round(params["target_bg"] * 100)))
        if "shadow_clip_sigma" in params and hasattr(self, "row_sl_as_sigma"):
            self.row_sl_as_sigma.setValue(int(round(params["shadow_clip_sigma"] * 10)))
        if "saturation_boost" in params and hasattr(self, "row_sl_as_sat"):
            self.row_sl_as_sat.setValue(int(round(params["saturation_boost"] * 10)))
        if "feather_radius" in params and hasattr(self, "row_sl_as_feather"):
            self.row_sl_as_feather.setValue(int(params["feather_radius"]))
        if "unlinked" in params and hasattr(self, "chk_sl_as_unlinked"):
            self.chk_sl_as_unlinked.setChecked(bool(params["unlinked"]))

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

    def set_ai_denoise_active(self, active: bool):
        """Actualiza el aspecto visual del botón según si hay un buffer IA activo."""
        if not hasattr(self, 'btn_sl_ai_denoise'):
            return
        self.btn_sl_ai_denoise.setChecked(active)
        if active:
            self.btn_sl_ai_denoise.setText("✓ IA")
            self.btn_sl_ai_denoise.setStyleSheet("""
                QPushButton {
                    background-color: #1b3a2b;
                    color: #a5d6a7;
                    border: 1px solid #2e7d32;
                    border-radius: 3px;
                    padding: 1px 6px;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #2e593e;
                    color: #ffffff;
                }
            """)
            self.btn_sl_ai_denoise.setToolTip("Reducción de ruido por IA calculada y activa. Desliza para graduar fuerza.")
        else:
            self.btn_sl_ai_denoise.setText("✨ IA")
            self.btn_sl_ai_denoise.setStyleSheet("""
                QPushButton {
                    background-color: #2b1f3d;
                    color: #ce93d8;
                    border: 1px solid #7b1fa2;
                    border-radius: 3px;
                    padding: 1px 6px;
                    font-weight: bold;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #3b2854;
                    border-color: #ba68c8;
                    color: #ffffff;
                }
            """)
            self.btn_sl_ai_denoise.setToolTip("Ejecutar Reducción de Ruido Profunda con IA (NAFNet)")

    def _on_sl_denoise_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_denoise.setText(f"{val}%")
        self.starless_denoise_changed.emit(val)

    def _on_sl_denoise_color_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_denoise_color.setText(f"{val}%")
        self.starless_denoise_color_changed.emit(val)

    def _on_sl_denoise_mask_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_denoise_mask.setText(f"{val}%")
        self.starless_denoise_mask_changed.emit(val)

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
        orton_tint_idx: int = 0, orton_mode_idx: int = 0,
        denoise_color: int = 50, denoise_mask: int = 50
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
        if hasattr(self, 'slider_sl_denoise_color'):
            self.slider_sl_denoise_color.setValue(denoise_color)
            self.lbl_sl_denoise_color.setText(f"{denoise_color}%")
        if hasattr(self, 'slider_sl_denoise_mask'):
            self.slider_sl_denoise_mask.setValue(denoise_mask)
            self.lbl_sl_denoise_mask.setText(f"{denoise_mask}%")
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
