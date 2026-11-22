# gui/widgets/mask_layers_widget.py
"""
gui/widgets/mask_layers_widget.py - Coordinador interactivo de gestión de capas de máscaras y ajustes locales.
Permite crear, ordenar, editar y previsualizar múltiples capas y máscaras
(Pincel, Luminancia, Color, Gradientes, Suelo, Cielo y Ráster), coordinando sub-widgets
especializados del paquete `gui.widgets.mask_layers`.
"""
import uuid
from typing import Optional, List, Dict, Any, Callable, Tuple
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QRect
from PySide6.QtGui import QColor, QFont, QCursor, QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QSlider, QCheckBox, QComboBox,
    QLineEdit, QGroupBox, QFrame, QSizePolicy, QMessageBox,
    QButtonGroup, QApplication
)

from core.mask_layers import (
    MaskLayer, compute_layer_mask, compute_effective_layer_mask, get_canvas_below_layer,
    clear_subcomposite_cache,
    MASK_TYPE_BRUSH, MASK_TYPE_GROUND, MASK_TYPE_SKY, MASK_TYPE_LUMA, MASK_TYPE_COLOR, MASK_TYPE_CUSTOM,
    MASK_TYPE_GRADIENT_LINEAR, MASK_TYPE_GRADIENT_RADIAL,
    LIMIT_MASK_NONE, LIMIT_MASK_SKY, LIMIT_MASK_GROUND,
    LAYER_KIND_ADJUSTMENT, LAYER_KIND_IMAGE,
    MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS, MASK_TYPE_IMAGE_GROUND,
    MASK_TYPE_IMAGE_RASTER, BLEND_MODE_NORMAL,
    SCOPE_TARGET_ALL, SCOPE_TARGET_GROUND, SCOPE_TARGET_STARLESS, SCOPE_TARGET_STARS
)

from gui.theme import (
    BG_PANEL, BORDER_BASE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    TEXT_CYAN, ACCENT_CYAN
)
from gui.icons import get_icon, get_svg_pixmap
from gui.widgets.mask_layers import (
    THUMB_SPECS,
    LayerListResizeHandle,
    get_image_aspect_ratio,
    get_thumbnail_dimensions,
    get_layer_thumbnail_pixmap,
    get_raster_mask_thumbnail_pixmap,
    invalidate_layer_thumbnail_cache,
    MaskDefinitionWidget,
    RasterTransformPanel,
    GroundSpecialPanel,
    StarsSpecialPanel,
    StarlessSpecialPanel,
    LocalAdjustmentsPanel,
    make_resettable_label,
    make_box_qss,
)


class MaskLayersWidget(QWidget):
    """
    Widget coordinador para la gestión de la pila de capas de máscaras y ajustes locales.
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
    action_committed = Signal()                  # commit de deshacer consolidado para historial

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
    starless_texture_clarity_changed = Signal(int) # (val -200 a 200)
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
    raster_layer_import_requested = Signal()     # Petición para importar capa de imagen ráster externa
    layer_move_tool_toggled = Signal(bool)       # (activo) conmuta el modo interactivo de mover en el lienzo
    gradient_tool_toggled = Signal(bool, object) # (activo, layer) activa manipulación de gradiente en el lienzo
    master_mask_updated = Signal(object)        # (np.ndarray float32) máscara maestra de horizonte sincronizada desde capa ráster

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layers: List[MaskLayer] = []
        self._selected_layer_id: Optional[str] = None
        self._updating_ui: bool = False
        self._is_scrubbing: bool = False

        self._image_reference_getter: Optional[Callable[[], Any]] = None
        self._master_mask_getter: Optional[Callable[[], Any]] = None
        self._stars_getter: Optional[Callable[[], Any]] = None
        self._starless_getter: Optional[Callable[[], Any]] = None
        self._layer_image_provider: Optional[Callable[..., Any]] = None

        self._thumb_size_mode: str = "M"
        self._custom_list_height: Optional[int] = None

        self._thumb_rebuild_timer = QTimer(self)
        self._thumb_rebuild_timer.setSingleShot(True)
        self._thumb_rebuild_timer.timeout.connect(lambda: self._rebuild_list_items(select_layer_id=self._selected_layer_id))

        self._setup_ui()
        self._connect_all_slider_interactions()

    def set_getters(
        self,
        img_getter: Optional[Callable[[], Any]] = None,
        mask_getter: Optional[Callable[[], Any]] = None,
        stars_getter: Optional[Callable[[], Any]] = None,
        starless_getter: Optional[Callable[[], Any]] = None,
        layer_image_provider: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Configura los proveedores de imagen base, máscara maestra de suelo y capas StarNet."""
        self._image_reference_getter = img_getter
        self._master_mask_getter = mask_getter
        self._stars_getter = stars_getter
        self._starless_getter = starless_getter
        self._layer_image_provider = layer_image_provider

        # Pasar getters a sub-paneles
        self.grp_mask_def.set_getters(
            img_getter=img_getter,
            mask_getter=mask_getter,
            stars_getter=stars_getter,
            starless_getter=starless_getter,
            layers_getter=lambda: self.layers,
            selected_layer_getter=self.get_selected_layer
        )
        self.grp_local_adj.set_getters(
            img_getter=img_getter,
            mask_getter=mask_getter,
            stars_getter=stars_getter,
            starless_getter=starless_getter,
            layer_img_provider=layer_image_provider,
            layers_getter=lambda: self.layers,
            selected_layer_getter=self.get_selected_layer
        )
        self.grp_transform.set_layer_getter(self.get_selected_layer)

    @staticmethod
    def _make_resettable_label(lbl: QLabel, reset_action, default_desc: str = "") -> QLabel:
        return make_resettable_label(lbl, reset_action, default_desc)

    def is_mask_preview_active(self) -> bool:
        """Indica si el botón de previsualización de máscara en lienzo está activo."""
        return self.grp_mask_def.is_preview_active()

    def set_mask_preview_active(self, active: bool) -> None:
        """Sincroniza el botón de previsualización sin entrar en bucles de emisión redundantes."""
        self.grp_mask_def.set_preview_active(active)

    def set_preview_active(self, active: bool) -> None:
        """Alias para set_mask_preview_active."""
        self.set_mask_preview_active(active)

    def set_ground_denoise_value(self, val: int) -> None:
        """Asigna el valor del deslizador de reducción de ruido de suelo."""
        self.grp_gnd_special.slider_gnd_special_dn.setValue(val)

    def set_ground_antihalo_value(self, val: int) -> None:
        """Asigna el valor del deslizador de anti-halo de frontera de suelo."""
        self.grp_gnd_special.slider_gnd_special_antihalo.setValue(val)

    def set_ground_edge_shift_value(self, val: int) -> None:
        """Asigna el valor del deslizador de desplazamiento de borde de suelo."""
        self.grp_gnd_special.slider_gnd_special_edge_shift.setValue(val)

    def set_starless_denoise_value(self, val: int) -> None:
        """Asigna el valor del deslizador de reducción de ruido de fondo nebular."""
        self.grp_starless_special.slider_sl_denoise.setValue(val)

    def is_curve_editor_expanded(self) -> bool:
        """Indica si la sección colapsable de la curva de capa está desplegada."""
        return self.grp_local_adj.is_curve_expanded()

    # --- Propiedades de compatibilidad hacia atrás ---
    @property
    def btn_preview_mask(self):
        return self.grp_mask_def.btn_preview_mask

    @property
    def panel_brush(self):
        return self.grp_mask_def.panel_brush

    @property
    def slider_gnd_special_dn(self):
        return self.grp_gnd_special.slider_gnd_special_dn

    @property
    def slider_gnd_special_antihalo(self):
        return self.grp_gnd_special.slider_gnd_special_antihalo

    @property
    def slider_gnd_special_edge_shift(self):
        return self.grp_gnd_special.slider_gnd_special_edge_shift

    @property
    def slider_sl_denoise(self):
        return self.grp_starless_special.slider_sl_denoise

    @property
    def slider_sl_clarity(self):
        return self.grp_starless_special.slider_sl_clarity

    @property
    def lbl_sl_clarity(self):
        return self.grp_starless_special.lbl_sl_clarity

    @property
    def slider_sl_texture_clarity(self):
        return self.grp_starless_special.slider_sl_texture_clarity

    @property
    def lbl_sl_texture_clarity(self):
        return self.grp_starless_special.lbl_sl_texture_clarity

    @property
    def slider_adj_clarity(self):
        return self.grp_local_adj.slider_adj_clarity

    @property
    def slider_adj_texture_clarity(self):
        return self.grp_local_adj.slider_adj_texture_clarity

    @property
    def lbl_adj_clarity(self):
        return self.grp_local_adj.lbl_adj_clarity

    @property
    def lbl_adj_texture_clarity(self):
        return self.grp_local_adj.lbl_adj_texture_clarity

    @property
    def row_adj_clarity(self):
        return self.grp_local_adj.row_adj_clarity

    @property
    def row_adj_texture_clarity(self):
        return self.grp_local_adj.row_adj_texture_clarity

    @property
    def slider_trans_scale_x(self):
        return self.grp_transform.slider_trans_scale_x

    @property
    def slider_trans_scale_y(self):
        return self.grp_transform.slider_trans_scale_y

    @property
    def slider_trans_x(self):
        return self.grp_transform.slider_trans_x

    @property
    def slider_trans_y(self):
        return self.grp_transform.slider_trans_y

    @property
    def chk_link_scales(self):
        return self.grp_transform.chk_link_scales

    @property
    def slider_b_radius(self):
        return self.grp_mask_def.slider_b_radius

    @property
    def slider_b_hardness(self):
        return self.grp_mask_def.slider_b_hardness

    @property
    def slider_b_flow(self):
        return self.grp_mask_def.slider_b_flow

    @property
    def btn_b_paint(self):
        return self.grp_mask_def.btn_b_paint

    @property
    def btn_b_erase(self):
        return self.grp_mask_def.btn_b_erase

    @property
    def btn_sharp_mask_preview(self):
        return self.grp_local_adj.btn_sharp_mask_preview

    @property
    def sec_layer_curve(self):
        return self.grp_local_adj.sec_layer_curve

    @property
    def layer_curve_widget(self):
        return self.grp_local_adj.layer_curve_widget

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # 1. Barra de creación rápida de máscaras (en 2 filas compactas y ergonómicas)
        layout_create = QVBoxLayout()
        layout_create.setSpacing(4)

        row_create_1 = QHBoxLayout()
        row_create_1.setSpacing(4)

        self.btn_add_brush = QPushButton(" Pincel")
        self.btn_add_brush.setIcon(get_icon("brush", color="#e0e6f0", size=13))
        self.btn_add_brush.setCursor(Qt.PointingHandCursor)
        self.btn_add_brush.setToolTip("Crea una capa con máscara pintada a mano con dureza y opacidad de trazo")
        self.btn_add_brush.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_BRUSH))
        row_create_1.addWidget(self.btn_add_brush)

        self.btn_add_linear = QPushButton(" Lineal")
        self.btn_add_linear.setIcon(get_icon("sliders", color="#80d8ff", size=13))
        self.btn_add_linear.setCursor(Qt.PointingHandCursor)
        self.btn_add_linear.setToolTip("Filtro degradado lineal vectorial analítico (ideal para cúpulas de polución lumínica)")
        self.btn_add_linear.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_GRADIENT_LINEAR))
        row_create_1.addWidget(self.btn_add_linear)

        self.btn_add_radial = QPushButton(" Radial")
        self.btn_add_radial.setIcon(get_icon("target", color="#80d8ff", size=13))
        self.btn_add_radial.setCursor(Qt.PointingHandCursor)
        self.btn_add_radial.setToolTip("Filtro graduado radial elíptico analítico (ideal para el núcleo de la Vía Láctea)")
        self.btn_add_radial.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_GRADIENT_RADIAL))
        row_create_1.addWidget(self.btn_add_radial)

        self.btn_add_raster = QPushButton(" Ráster")
        self.btn_add_raster.setIcon(get_icon("layers", color="#ffb74d", size=13))
        self.btn_add_raster.setCursor(Qt.PointingHandCursor)
        self.btn_add_raster.setToolTip("Carga una imagen externa (H-Alfa, Suelo hora azul, cometa) como capa ráster con modo de fusión")
        self.btn_add_raster.clicked.connect(self._on_add_raster_clicked)
        row_create_1.addWidget(self.btn_add_raster)

        layout_create.addLayout(row_create_1)

        row_create_2 = QHBoxLayout()
        row_create_2.setSpacing(4)

        self.btn_add_luma = QPushButton(" Luminancia")
        self.btn_add_luma.setIcon(get_icon("sparkles", color="#e0e6f0", size=13))
        self.btn_add_luma.setCursor(Qt.PointingHandCursor)
        self.btn_add_luma.setToolTip("Crea una máscara basada en el rango de brillo (Sombras, Medios, Luces)")
        self.btn_add_luma.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_LUMA))
        row_create_2.addWidget(self.btn_add_luma)

        self.btn_add_color = QPushButton(" Color")
        self.btn_add_color.setIcon(get_icon("palette", color="#e0e6f0", size=13))
        self.btn_add_color.setCursor(Qt.PointingHandCursor)
        self.btn_add_color.setToolTip("Crea una máscara basada en tono cromático (H-Alfa, OIII, Sodio, etc.)")
        self.btn_add_color.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_COLOR))
        row_create_2.addWidget(self.btn_add_color)

        self.btn_add_gnd = QPushButton(" Suelo")
        self.btn_add_gnd.setIcon(get_icon("mountain", color="#e0e6f0", size=13))
        self.btn_add_gnd.setCursor(Qt.PointingHandCursor)
        self.btn_add_gnd.setToolTip("Crea una capa vinculada a la segmentación del terreno")
        self.btn_add_gnd.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_GROUND))
        row_create_2.addWidget(self.btn_add_gnd)

        self.btn_add_sky = QPushButton(" Cielo")
        self.btn_add_sky.setIcon(get_icon("cloud", color="#80d8ff", size=13))
        self.btn_add_sky.setCursor(Qt.PointingHandCursor)
        self.btn_add_sky.setToolTip("Crea una capa vinculada a la segmentación del cielo / bóveda celeste")
        self.btn_add_sky.clicked.connect(lambda: self.add_new_layer(MASK_TYPE_SKY))
        row_create_2.addWidget(self.btn_add_sky)

        layout_create.addLayout(row_create_2)
        main_layout.addLayout(layout_create)

        # Panel superior: Cabecera con selector S/M/L + Lista de capas + Acciones
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

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
                    background-color: #12151e; color: #94a3b8; border: 1px solid #202638;
                    border-radius: 3px; font-size: 10px; font-weight: 600; padding: 0px;
                }
                QPushButton:hover { background-color: #1a2234; color: #f1f5f9; }
                QPushButton:checked { background-color: #0284c7; color: #ffffff; border-color: #38bdf8; font-weight: bold; }
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
                background-color: #0f121a; border: 1px solid #1e2538; border-radius: 5px; padding: 2px;
            }
            QListWidget::item {
                background-color: #131722; border: 1px solid #1c2333; border-radius: 4px; margin-bottom: 2px; padding: 0px;
            }
            QListWidget::item:selected { background-color: #162033; border: 1px solid #0284c7; }
            QListWidget::item:hover:!selected { background-color: #161a26; border-color: #28324a; }
        """)
        self.list_layers.currentRowChanged.connect(self._on_layer_selection_changed)
        self.list_layers.itemClicked.connect(lambda item: self._on_layer_selection_changed(self.list_layers.row(item)))
        top_layout.addWidget(self.list_layers, stretch=1)

        # 3. Acciones sobre la capa seleccionada
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

        # Tirador interactivo de altura de lista
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

        # Fila B: Modo de fusión (para capas de imagen ráster)
        self.container_blend_mode = QWidget()
        row_blend = QHBoxLayout(self.container_blend_mode)
        row_blend.setContentsMargins(0, 0, 0, 0)
        row_blend.setSpacing(6)

        lbl_blend = QLabel("Modo de Fusión:")
        lbl_blend.setStyleSheet("color: #b0bec5; font-size: 11px; font-weight: 600;")
        row_blend.addWidget(lbl_blend)

        self.combo_blend_mode = QComboBox()
        self.combo_blend_mode.addItem("Normal (Opacidad Ponderada)", "normal")
        self.combo_blend_mode.addItem("Luminosidad (MGN / WOW / NAFE)", "luminosity")
        self.combo_blend_mode.addItem("Trama / Screen (Protuberancias)", "screen")
        self.combo_blend_mode.addItem("Luz Suave / Soft Light (Contraste)", "soft_light")
        self.combo_blend_mode.addItem("Superponer / Overlay (Estructuras)", "overlay")
        self.combo_blend_mode.addItem("Multiplicar / Multiply (Densidad)", "multiply")
        self.combo_blend_mode.currentIndexChanged.connect(self._on_blend_mode_changed)
        row_blend.addWidget(self.combo_blend_mode, stretch=1)
        props_layout.addWidget(self.container_blend_mode)

        # Fila C: Opacidad
        row_op = QHBoxLayout()
        row_op.addWidget(make_resettable_label(QLabel("Opacidad:"), lambda: self.slider_opacity.setValue(100), "100%"))
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

        # 5. Transformación espacial 2D (capas ráster)
        self.grp_transform = RasterTransformPanel(self)
        self.grp_transform.set_layer_getter(self.get_selected_layer)
        self.grp_transform.layer_move_tool_toggled.connect(self.layer_move_tool_toggled)
        self.grp_transform.transform_changed.connect(lambda: self.invalidate_layer_thumbnail(None, rebuild_now=False))
        self.grp_transform.transform_changed.connect(self.layers_changed)
        self.grp_transform.action_committed.connect(self._notify_action_committed)
        self.grp_transform.slider_released.connect(self.ground_slider_released)
        props_layout.addWidget(self.grp_transform)

        # 6. Definición de Máscara (Pincel, Luminancia, Color HSV, Suelo/Cielo, Gradiente)
        self.grp_mask_def = MaskDefinitionWidget(self)
        self.grp_mask_def.set_getters(
            img_getter=lambda: self._image_reference_getter() if self._image_reference_getter else None,
            mask_getter=lambda: self._master_mask_getter() if self._master_mask_getter else None,
            stars_getter=lambda: self._stars_getter() if self._stars_getter else None,
            starless_getter=lambda: self._starless_getter() if self._starless_getter else None,
            layers_getter=lambda: self.layers,
            selected_layer_getter=self.get_selected_layer
        )
        self.grp_mask_def.brush_params_changed.connect(self.brush_params_changed)
        self.grp_mask_def.brush_tool_toggled.connect(self.brush_tool_toggled)
        self.grp_mask_def.color_eyedropper_requested.connect(self.color_eyedropper_requested)
        self.grp_mask_def.ai_ground_mask_requested.connect(self.ai_ground_mask_requested)
        self.grp_mask_def.edit_ground_mask_requested.connect(self.edit_ground_mask_requested)
        self.grp_mask_def.gradient_tool_toggled.connect(self.gradient_tool_toggled)
        self.grp_mask_def.preview_mask_requested.connect(self.preview_mask_requested)
        self.grp_mask_def.master_mask_updated.connect(self.master_mask_updated)
        self.grp_mask_def.mask_structure_changed.connect(lambda: self.invalidate_layer_thumbnail(None, rebuild_now=True))
        self.grp_mask_def.mask_param_changed.connect(lambda: self.invalidate_layer_thumbnail(None, rebuild_now=False))
        self.grp_mask_def.mask_param_changed.connect(self.layers_changed)
        self.grp_mask_def.action_committed.connect(self._notify_action_committed)
        self.grp_mask_def.interaction_finished.connect(self._on_mask_param_interaction_finished)
        props_layout.addWidget(self.grp_mask_def)

        # 7. Ajustes Locales de Revelado
        self.grp_local_adj = LocalAdjustmentsPanel(self)
        self.grp_local_adj.set_getters(
            img_getter=lambda: self._image_reference_getter() if self._image_reference_getter else None,
            mask_getter=lambda: self._master_mask_getter() if self._master_mask_getter else None,
            stars_getter=lambda: self._stars_getter() if self._stars_getter else None,
            starless_getter=lambda: self._starless_getter() if self._starless_getter else None,
            layer_img_provider=self._layer_image_provider,
            layers_getter=lambda: self.layers,
            selected_layer_getter=self.get_selected_layer
        )
        self.grp_local_adj.adjustments_changed.connect(self.layers_changed)
        self.grp_local_adj.slider_released.connect(self.ground_slider_released)
        self.grp_local_adj.preview_mask_requested.connect(self.preview_mask_requested)
        self.grp_local_adj.interaction_started.connect(self._on_slider_interaction_started)
        self.grp_local_adj.interaction_ended.connect(self._on_slider_interaction_ended)
        props_layout.addWidget(self.grp_local_adj)

        # 8. Ajustes Especiales de Suelo
        self.grp_gnd_special = GroundSpecialPanel(self)
        self.grp_gnd_special.ground_ev_changed.connect(self.ground_ev_changed)
        self.grp_gnd_special.ground_shadows_changed.connect(self.ground_shadows_changed)
        self.grp_gnd_special.ground_bp_changed.connect(self.ground_bp_changed)
        self.grp_gnd_special.ground_denoise_changed.connect(self.ground_denoise_changed)
        self.grp_gnd_special.ground_antihalo_changed.connect(self.ground_antihalo_changed)
        self.grp_gnd_special.ground_edge_shift_changed.connect(self.ground_edge_shift_changed)
        self.grp_gnd_special.ground_ai_denoise_requested.connect(self.ground_ai_denoise_requested)
        self.grp_gnd_special.ground_reset_requested.connect(self.ground_reset_requested)
        self.grp_gnd_special.slider_released.connect(self.ground_slider_released)
        props_layout.addWidget(self.grp_gnd_special)

        # 9. Ajustes Especiales de Estrellas
        self.grp_stars_special = StarsSpecialPanel(self)
        self.grp_stars_special.stars_intensity_changed.connect(self.stars_intensity_changed)
        self.grp_stars_special.stars_diff_strength_changed.connect(self.stars_diff_strength_changed)
        self.grp_stars_special.stars_diff_radius_changed.connect(self.stars_diff_radius_changed)
        self.grp_stars_special.stars_diff_threshold_changed.connect(self.stars_diff_threshold_changed)
        self.grp_stars_special.stars_diff_color_boost_changed.connect(self.stars_diff_color_boost_changed)
        self.grp_stars_special.stars_diff_mode_changed.connect(self.stars_diff_mode_changed)
        self.grp_stars_special.stars_reset_requested.connect(self.stars_reset_requested)
        self.grp_stars_special.slider_released.connect(self.ground_slider_released)
        props_layout.addWidget(self.grp_stars_special)

        # 10. Ajustes Especiales de Fondo Nebular (Starless)
        self.grp_starless_special = StarlessSpecialPanel(self)
        self.grp_starless_special.starless_clarity_changed.connect(self.starless_clarity_changed)
        self.grp_starless_special.starless_texture_clarity_changed.connect(self.starless_texture_clarity_changed)
        self.grp_starless_special.starless_dehaze_changed.connect(self.starless_dehaze_changed)
        self.grp_starless_special.starless_ond_est_changed.connect(self.starless_ond_est_changed)
        self.grp_starless_special.starless_ond_bg_changed.connect(self.starless_ond_bg_changed)
        self.grp_starless_special.starless_contrast_changed.connect(self.starless_contrast_changed)
        self.grp_starless_special.starless_denoise_changed.connect(self.starless_denoise_changed)
        self.grp_starless_special.starless_denoise_color_changed.connect(self.starless_denoise_color_changed)
        self.grp_starless_special.starless_denoise_mask_changed.connect(self.starless_denoise_mask_changed)
        self.grp_starless_special.starless_ai_denoise_requested.connect(self.starless_ai_denoise_requested)
        self.grp_starless_special.starless_orton_strength_changed.connect(self.starless_orton_strength_changed)
        self.grp_starless_special.starless_orton_radius_changed.connect(self.starless_orton_radius_changed)
        self.grp_starless_special.starless_orton_threshold_changed.connect(self.starless_orton_threshold_changed)
        self.grp_starless_special.starless_orton_tint_changed.connect(self.starless_orton_tint_changed)
        self.grp_starless_special.starless_orton_mode_changed.connect(self.starless_orton_mode_changed)
        self.grp_starless_special.starless_reset_requested.connect(self.starless_reset_requested)
        self.grp_starless_special.auto_stretch_starless_requested.connect(self.auto_stretch_starless_requested)
        self.grp_starless_special.slider_released.connect(self.ground_slider_released)
        props_layout.addWidget(self.grp_starless_special)

        main_layout.addWidget(self.container_layer_props)

    def _update_list_height(self) -> None:
        if self._custom_list_height is not None:
            self.list_layers.setFixedHeight(self._custom_list_height)
            return
        count = len(self.layers)
        spec = THUMB_SPECS.get(self._thumb_size_mode, THUMB_SPECS["M"])
        row_h = spec["row_height"]
        pad = 12
        target_h = max(70, min(count * (row_h + 3) + pad, 250))
        self.list_layers.setFixedHeight(target_h)

    def _on_list_resize_dragged(self, dy: int) -> None:
        current_h = self.list_layers.height()
        new_h = max(55, min(current_h + dy, 480))
        self._custom_list_height = new_h
        self.list_layers.setFixedHeight(new_h)

    def _on_list_auto_fit_requested(self) -> None:
        self._custom_list_height = None
        self._update_list_height()

    def get_all_sliders(self) -> List[QSlider]:
        """Retorna todos los controles deslizantes activos del panel de capas."""
        return self.findChildren(QSlider)

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

    def has_image_layers(self) -> bool:
        return any(layer.layer_kind == LAYER_KIND_IMAGE for layer in self.layers)

    def ensure_base_layer(self) -> None:
        has_base_or_starnet = any(
            l.mask_type in (MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS)
            for l in self.layers
        )
        if not has_base_or_starnet:
            base_layer = MaskLayer(
                name="Fondo Base (RGB)",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_BASE,
                target_scope=SCOPE_TARGET_ALL,
                is_system_layer=True,
                enabled=True,
                opacity=1.0,
                blend_mode=BLEND_MODE_NORMAL
            )
            self.layers.append(base_layer)
            self._rebuild_list_items(select_layer_id=base_layer.id)
            self.layers_changed.emit()

    def ensure_starnet_layers(self, has_ground: bool = False) -> None:
        """Crea o sincroniza las capas de StarNet (Fondo Nebular, Estrellas y Suelo)."""
        existing_sys = {l.mask_type: l for l in self.layers if l.is_system_layer}
        has_starnet = (MASK_TYPE_IMAGE_STARS in existing_sys or MASK_TYPE_IMAGE_STARLESS in existing_sys)

        if not has_starnet:
            base_l = existing_sys.get(MASK_TYPE_IMAGE_BASE)

            layer_starless = MaskLayer(
                name="Fondo Nebular (Starless)",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_STARLESS,
                target_scope=SCOPE_TARGET_STARLESS,
                is_system_layer=True,
                enabled=True,
                opacity=1.0,
                blend_mode=BLEND_MODE_NORMAL
            )

            layer_stars = MaskLayer(
                name="Estrellas",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_STARS,
                target_scope=SCOPE_TARGET_STARS,
                is_system_layer=True,
                enabled=True,
                opacity=1.0,
                blend_mode="screen"
            )

            if base_l is not None:
                layer_starless.exposure = base_l.exposure
                layer_starless.contrast = base_l.contrast
                layer_starless.shadows = base_l.shadows
                layer_starless.highlights = base_l.highlights
                layer_starless.saturation = base_l.saturation
                layer_starless.temp = base_l.temp
                layer_starless.tint = base_l.tint
                layer_starless.opacity = base_l.opacity
                if hasattr(base_l, 'curve_points') and base_l.curve_points:
                    layer_starless.curve_points = list(base_l.curve_points)
                if hasattr(base_l, 'curve_lut') and base_l.curve_lut is not None:
                    layer_starless.curve_lut = base_l.curve_lut.copy()

            user_layers = [l for l in self.layers if not l.is_system_layer]
            new_layers = list(user_layers)
            new_layers.append(layer_stars)

            if has_ground:
                layer_ground = MaskLayer(
                    name="Paisaje (Suelo)",
                    layer_kind=LAYER_KIND_IMAGE,
                    mask_type=MASK_TYPE_IMAGE_GROUND,
                    target_scope=SCOPE_TARGET_GROUND,
                    is_system_layer=True,
                    enabled=True,
                    opacity=1.0,
                    blend_mode=BLEND_MODE_NORMAL
                )
                new_layers.append(layer_ground)

            new_layers.append(layer_starless)
            self.layers = new_layers

            self._rebuild_list_items(select_layer_id=layer_stars.id)
            self.layers_changed.emit()
            return

        has_gnd_layer = MASK_TYPE_IMAGE_GROUND in existing_sys
        modified = False

        if has_ground and not has_gnd_layer:
            layer_ground = MaskLayer(
                name="Paisaje (Suelo)",
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=MASK_TYPE_IMAGE_GROUND,
                target_scope=SCOPE_TARGET_GROUND,
                is_system_layer=True,
                enabled=True,
                opacity=1.0,
                blend_mode=BLEND_MODE_NORMAL
            )
            starless_idx = next((i for i, l in enumerate(self.layers) if l.mask_type == MASK_TYPE_IMAGE_STARLESS), len(self.layers))
            self.layers.insert(starless_idx, layer_ground)
            modified = True
        elif not has_ground and has_gnd_layer:
            self.layers = [l for l in self.layers if l.mask_type != MASK_TYPE_IMAGE_GROUND]
            modified = True

        if modified:
            self._rebuild_list_items(select_layer_id=self._selected_layer_id)
            self.layers_changed.emit()

    def set_starnet_stars_enabled(self, enabled: bool) -> None:
        changed = False
        for l in self.layers:
            if l.mask_type == MASK_TYPE_IMAGE_STARS:
                if l.enabled != enabled:
                    l.enabled = enabled
                    changed = True
                break
        if changed:
            self._rebuild_list_items(select_layer_id=self._selected_layer_id)
            self.layers_changed.emit()

    def add_new_layer(self, mask_type: str = MASK_TYPE_BRUSH) -> MaskLayer:
        """Crea una nueva capa de ajuste del tipo indicado y la sitúa al inicio (TOP) de la pila."""
        count = len(self.layers) + 1
        if mask_type == MASK_TYPE_BRUSH:
            name = f"Pincel {count}"
            layer = MaskLayer(
                name=name,
                mask_type=mask_type,
                opacity=getattr(self.grp_mask_def, '_last_brush_opacity', 1.0),
                brush_radius=getattr(self.grp_mask_def, '_last_brush_radius', 35),
                brush_hardness=getattr(self.grp_mask_def, '_last_brush_hardness', 0.5),
                brush_flow=getattr(self.grp_mask_def, '_last_brush_flow', 1.0),
                brush_mode=getattr(self.grp_mask_def, '_last_brush_mode', "paint")
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
        elif mask_type == MASK_TYPE_GRADIENT_LINEAR:
            name = f"Grad. Lineal {count}"
            layer = MaskLayer(name=name, mask_type=mask_type, limit_mask=LIMIT_MASK_NONE)
        elif mask_type == MASK_TYPE_GRADIENT_RADIAL:
            name = f"Grad. Radial {count}"
            layer = MaskLayer(name=name, mask_type=mask_type, limit_mask=LIMIT_MASK_NONE)
        elif mask_type == MASK_TYPE_IMAGE_RASTER:
            name = f"Ráster {count}"
            layer = MaskLayer(
                name=name,
                layer_kind=LAYER_KIND_IMAGE,
                mask_type=mask_type,
                blend_mode=BLEND_MODE_NORMAL,
                brush_radius=getattr(self.grp_mask_def, '_last_brush_radius', 35),
                brush_hardness=getattr(self.grp_mask_def, '_last_brush_hardness', 0.5),
                brush_flow=getattr(self.grp_mask_def, '_last_brush_flow', 1.0),
                brush_mode=getattr(self.grp_mask_def, '_last_brush_mode', "paint")
            )
        else:
            name = f"Máscara {count}"
            layer = MaskLayer(name=name, mask_type=mask_type)

        self.layers.insert(0, layer)
        self._rebuild_list_items(select_layer_id=layer.id)
        self.layers_changed.emit()
        self._notify_action_committed()
        return layer

    def add_raster_layer(
        self,
        name: str = "Capa Ráster",
        image_data: Optional[np.ndarray] = None,
        blend_mode: str = "normal",
        opacity: float = 1.0,
        position: int = 0
    ) -> MaskLayer:
        """Añade una capa de imagen ráster (filtro coronal, composición, etc.) a la pila."""
        layer = MaskLayer(
            name=name,
            layer_kind=LAYER_KIND_IMAGE,
            mask_type=MASK_TYPE_IMAGE_RASTER,
            image_data=image_data,
            blend_mode=blend_mode,
            opacity=opacity,
            brush_radius=getattr(self.grp_mask_def, '_last_brush_radius', 35),
            brush_hardness=getattr(self.grp_mask_def, '_last_brush_hardness', 0.5),
            brush_flow=getattr(self.grp_mask_def, '_last_brush_flow', 1.0),
            brush_mode=getattr(self.grp_mask_def, '_last_brush_mode', "paint")
        )
        pos = max(0, min(position, len(self.layers)))
        self.layers.insert(pos, layer)
        self._rebuild_list_items(select_layer_id=layer.id)
        self.layers_changed.emit()
        self._notify_action_committed()
        return layer

    def _delete_selected(self) -> None:
        layer = self.get_selected_layer()
        if not layer or layer.is_system_layer:
            return
        if hasattr(self, 'grp_transform') and self.grp_transform.btn_toggle_move.isChecked():
            self.grp_transform.btn_toggle_move.setChecked(False)
        row = self.list_layers.currentRow()
        if hasattr(layer, "release_heavy_data"):
            layer.release_heavy_data()
        self.layers = [l for l in self.layers if l.id != layer.id]
        clear_subcomposite_cache()

        next_sel_id = None
        if self.layers:
            new_row = min(row, len(self.layers) - 1)
            next_sel_id = self.layers[new_row].id

        self._rebuild_list_items(select_layer_id=next_sel_id)
        self.layers_changed.emit()
        self._notify_action_committed()

    def _duplicate_selected(self) -> None:
        layer = self.get_selected_layer()
        if not layer or layer.is_system_layer:
            return
        dup = layer.shallow_snapshot()
        dup.id = str(uuid.uuid4())
        dup.name = f"{layer.name} (Copia)"
        row = self.list_layers.currentRow()
        self.layers.insert(row + 1, dup)
        self._rebuild_list_items(select_layer_id=dup.id)
        self.layers_changed.emit()
        self._notify_action_committed()

    def _move_selected_up(self) -> None:
        row = self.list_layers.currentRow()
        if row > 0:
            self.layers[row], self.layers[row - 1] = self.layers[row - 1], self.layers[row]
            clear_subcomposite_cache()
            self._rebuild_list_items(select_layer_id=self.layers[row - 1].id)
            self.layers_changed.emit()
            self._notify_action_committed()

    def _move_selected_down(self) -> None:
        row = self.list_layers.currentRow()
        if 0 <= row < len(self.layers) - 1:
            self.layers[row], self.layers[row + 1] = self.layers[row + 1], self.layers[row]
            clear_subcomposite_cache()
            self._rebuild_list_items(select_layer_id=self.layers[row + 1].id)
            self.layers_changed.emit()
            self._notify_action_committed()

    def get_selected_layer(self) -> Optional[MaskLayer]:
        """Obtiene la instancia de MaskLayer actualmente seleccionada."""
        if self._selected_layer_id:
            for l in self.layers:
                if l.id == self._selected_layer_id:
                    return l
        row = self.list_layers.currentRow()
        if 0 <= row < len(self.layers):
            return self.layers[row]
        return None

    def set_thumbnail_size_mode(self, mode: str) -> None:
        if mode not in THUMB_SPECS or self._thumb_size_mode == mode:
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
                invalidate_layer_thumbnail_cache(layer)
        if rebuild_now and not self._updating_ui:
            if self._thumb_rebuild_timer.isActive():
                self._thumb_rebuild_timer.stop()
            self._rebuild_list_items(select_layer_id=self._selected_layer_id)
        elif not rebuild_now and not self._updating_ui:
            self._thumb_rebuild_timer.start(250)

    def _on_mask_param_interaction_finished(self) -> None:
        if self._thumb_rebuild_timer.isActive():
            self._thumb_rebuild_timer.stop()
        if not self._updating_ui:
            self._rebuild_list_items(select_layer_id=self._selected_layer_id)

    def _on_thumb_double_clicked(self, layer: MaskLayer) -> None:
        if hasattr(self, 'btn_preview_mask'):
            self.btn_preview_mask.toggle()

    def _get_image_aspect_ratio(self) -> float:
        return get_image_aspect_ratio(
            image_reference_getter=self._image_reference_getter,
            starless_getter=self._starless_getter,
            stars_getter=self._stars_getter,
            master_mask_getter=self._master_mask_getter
        )

    def _get_thumbnail_dimensions(self) -> Tuple[int, int]:
        return get_thumbnail_dimensions(self._thumb_size_mode, self._get_image_aspect_ratio())

    def _get_layer_thumbnail_pixmap(self, layer: MaskLayer) -> QPixmap:
        tw, th = self._get_thumbnail_dimensions()
        return get_layer_thumbnail_pixmap(
            layer, self._thumb_size_mode, tw, th,
            image_reference_getter=self._image_reference_getter,
            starless_getter=self._starless_getter,
            stars_getter=self._stars_getter,
            master_mask_getter=self._master_mask_getter,
            layer_image_provider=self._layer_image_provider
        )

    def _on_add_raster_clicked(self) -> None:
        self.raster_layer_import_requested.emit()

    def _get_layer_image_shape(self, layer: MaskLayer) -> Tuple[int, int]:
        if self._image_reference_getter:
            try:
                base = self._image_reference_getter()
                if base is not None and hasattr(base, 'shape') and len(base.shape) >= 2:
                    return base.shape[:2]
            except Exception:
                pass
        if getattr(layer, 'image_data', None) is not None and hasattr(layer.image_data, 'shape') and len(layer.image_data.shape) >= 2:
            return layer.image_data.shape[:2]
        return 1000, 1500

    def _get_raster_mask_thumbnail_pixmap(self, layer: MaskLayer) -> QPixmap:
        tw, th = self._get_thumbnail_dimensions()
        return get_raster_mask_thumbnail_pixmap(layer, self._thumb_size_mode, tw, th)

    def _rebuild_list_items(self, select_layer_id: Optional[str] = None) -> None:
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
            elif layer.mask_type == MASK_TYPE_IMAGE_RASTER:
                icon_name = "sun"
                b_labels = {
                    "normal": "Normal", "luminosity": "Luminosidad", "screen": "Trama",
                    "soft_light": "Luz Suave", "overlay": "Superponer", "multiply": "Multiplicar",
                }
                b_mode = getattr(layer, 'blend_mode', 'normal').lower()
                b_str = b_labels.get(b_mode, b_mode.capitalize())
                type_tag = f"Ráster ({b_str})"
                fg_col = "#ffb74d" if layer.enabled else "#607d8b"
            else:
                type_tag = {
                    MASK_TYPE_BRUSH: "Pincel", MASK_TYPE_LUMA: "Luminancia", MASK_TYPE_COLOR: "Color HSV",
                    MASK_TYPE_GROUND: "Máscara Suelo", MASK_TYPE_SKY: "Máscara Cielo",
                    MASK_TYPE_GRADIENT_LINEAR: "Grad. Lineal", MASK_TYPE_GRADIENT_RADIAL: "Grad. Radial",
                }.get(layer.mask_type, "Ajuste")

                icon_name = {
                    MASK_TYPE_BRUSH: "brush", MASK_TYPE_LUMA: "sparkles", MASK_TYPE_COLOR: "palette",
                    MASK_TYPE_GROUND: "mountain", MASK_TYPE_SKY: "sparkles",
                    MASK_TYPE_GRADIENT_LINEAR: "sliders", MASK_TYPE_GRADIENT_RADIAL: "target",
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

            is_layer_selected = (select_layer_id and layer.id == select_layer_id) or (not select_layer_id and i == 0)
            target = getattr(layer, 'active_target', 'image')

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

            if layer.mask_type == MASK_TYPE_IMAGE_RASTER:
                pix_img = self._get_layer_thumbnail_pixmap(layer)
                lbl_thumb_img = QLabel()
                lbl_thumb_img.setFixedSize(tw + 2, th + 2)
                is_img_active = is_layer_selected and (target == "image")
                border_img = "#38bdf8" if is_img_active else ("#28324a" if layer.enabled else "#161b26")
                lbl_thumb_img.setStyleSheet(f"""
                    QLabel {{
                        background-color: #07090e; border: {2 if is_img_active else 1}px solid {border_img}; border-radius: 3px;
                    }}
                """)
                lbl_thumb_img.setAlignment(Qt.AlignCenter)
                lbl_thumb_img.setPixmap(pix_img)
                lbl_thumb_img.setCursor(Qt.PointingHandCursor)
                lbl_thumb_img.setToolTip("Contenido de Imagen Ráster.\nHaz clic para seleccionarlo.")
                row_layout.addWidget(lbl_thumb_img)

                pix_mask = self._get_raster_mask_thumbnail_pixmap(layer)
                lbl_thumb_mask = QLabel()
                lbl_thumb_mask.setFixedSize(tw + 2, th + 2)
                has_mask = layer.brush_mask is not None
                is_mask_active = is_layer_selected and (target == "mask")
                border_mask = "#38bdf8" if is_mask_active else ("#28324a" if (has_mask and layer.enabled) else "#2d3748")
                bstyle_mask = "solid" if (has_mask or is_mask_active) else "dashed"
                lbl_thumb_mask.setStyleSheet(f"""
                    QLabel {{
                        background-color: #07090e; border: {2 if is_mask_active else 1}px {bstyle_mask} {border_mask}; border-radius: 3px;
                    }}
                """)
                lbl_thumb_mask.setAlignment(Qt.AlignCenter)
                lbl_thumb_mask.setPixmap(pix_mask)
                lbl_thumb_mask.setCursor(Qt.PointingHandCursor)
                if has_mask:
                    lbl_thumb_mask.setToolTip("Máscara de Capa (B/N).\nHaz clic para seleccionarla y pintar.\nDoble clic para previsualizar.")
                else:
                    lbl_thumb_mask.setToolTip("Sin Máscara de Capa.\nHaz clic para añadir una máscara pintable.")
                row_layout.addWidget(lbl_thumb_mask)

                def make_img_click_handler(l_id, r):
                    def _h(e):
                        l = self.layers[r] if 0 <= r < len(self.layers) else None
                        if l:
                            l.active_target = "image"
                        self.list_layers.setCurrentRow(r)
                        self._selected_layer_id = l_id
                        self._update_list_and_props_state()
                        self._rebuild_list_items(select_layer_id=l_id)
                    return _h

                def make_mask_click_handler(l_id, r):
                    def _h(e):
                        l = self.layers[r] if 0 <= r < len(self.layers) else None
                        if l:
                            if l.brush_mask is None:
                                h_m, w_m = self._get_layer_image_shape(l)
                                l.brush_mask = np.ones((h_m, w_m), dtype=np.float32)
                                l.brush_mode = "erase"
                                self.invalidate_layer_thumbnail(l.id, rebuild_now=False)
                            l.active_target = "mask"
                        self.list_layers.setCurrentRow(r)
                        self._selected_layer_id = l_id
                        self._update_list_and_props_state()
                        self._rebuild_list_items(select_layer_id=l_id)
                    return _h

                lbl_thumb_img.mousePressEvent = make_img_click_handler(layer.id, i)
                lbl_thumb_mask.mousePressEvent = make_mask_click_handler(layer.id, i)
                lbl_thumb_img.mouseDoubleClickEvent = make_thumb_double_click_handler(layer)
                lbl_thumb_mask.mouseDoubleClickEvent = make_thumb_double_click_handler(layer)
            else:
                pix = self._get_layer_thumbnail_pixmap(layer)
                lbl_thumb = QLabel()
                lbl_thumb.setFixedSize(tw + 2, th + 2)
                border_col = "#28324a" if layer.enabled else "#161b26"
                lbl_thumb.setStyleSheet(f"""
                    QLabel {{
                        background-color: #07090e; border: 1px solid {border_col}; border-radius: 3px;
                    }}
                """)
                lbl_thumb.setAlignment(Qt.AlignCenter)
                lbl_thumb.setPixmap(pix)
                lbl_thumb.setCursor(Qt.PointingHandCursor)
                lbl_thumb.setToolTip("Miniatura de capa.\nHaz doble clic para previsualizar la máscara en el lienzo.")
                row_layout.addWidget(lbl_thumb)
                lbl_thumb.mousePressEvent = make_click_handler(layer.id, i)
                lbl_thumb.mouseDoubleClickEvent = make_thumb_double_click_handler(layer)

            lbl_icon = QLabel()
            lbl_icon.setStyleSheet("background: transparent;")
            pix_ico = get_svg_pixmap(icon_name, color=fg_col, size=ico_sz)
            if not pix_ico.isNull():
                lbl_icon.setPixmap(pix_ico)
            lbl_icon.setFixedSize(ico_sz + 2, ico_sz + 2)
            row_layout.addWidget(lbl_icon)

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

            btn_eye = QPushButton()
            eye_color = fg_col if layer.enabled else "#78909c"
            btn_eye.setIcon(get_icon("eye" if layer.enabled else "eye_off", color=eye_color, size=14))
            btn_eye.setFixedSize(26, 22)
            btn_eye.setCursor(Qt.PointingHandCursor)
            btn_eye.setToolTip("Capa visible (clic para ocultar)" if layer.enabled else "Capa oculta (clic para mostrar)")
            btn_eye.setStyleSheet("""
                QPushButton { background-color: transparent; border: none; padding: 0px; }
                QPushButton:hover { background-color: #2a3147; border-radius: 3px; }
            """)

            h = make_click_handler(layer.id, i)
            row_widget.mousePressEvent = h
            lbl_title.mousePressEvent = h
            lbl_icon.mousePressEvent = h
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
        if 0 <= idx < len(self.layers):
            layer = self.layers[idx]
            layer.enabled = not layer.enabled
            clear_subcomposite_cache()

            if layer.mask_type == MASK_TYPE_IMAGE_STARS:
                self.starnet_visibility_changed.emit(layer.enabled)

            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            self.layers_changed.emit()
            self._notify_action_committed()

            item = self.list_layers.item(idx)
            if item:
                w = self.list_layers.itemWidget(item)
                if w:
                    btn_eye = w.findChild(QPushButton)
                    if btn_eye:
                        fg_col = "#e0e6f0" if layer.enabled else "#78909c"
                        btn_eye.setIcon(get_icon("eye" if layer.enabled else "eye_off", color=fg_col, size=14))

            if layer.id == self._selected_layer_id:
                self._update_list_and_props_state()

    def _on_layer_selection_changed(self, row: int) -> None:
        if self._updating_ui: return
        if 0 <= row < len(self.layers):
            new_id = self.layers[row].id
            if new_id != self._selected_layer_id:
                if hasattr(self, 'grp_transform') and self.grp_transform.btn_toggle_move.isChecked():
                    self.grp_transform.btn_toggle_move.setChecked(False)
            self._selected_layer_id = new_id
            self._update_list_and_props_state()

    def _update_list_and_props_state(self) -> None:
        layer = self.get_selected_layer()
        has_sel = layer is not None
        is_sys = layer.is_system_layer if layer else False
        is_ground = (layer is not None and layer.mask_type in (MASK_TYPE_IMAGE_GROUND, MASK_TYPE_GROUND))
        is_stars = (layer is not None and layer.mask_type == MASK_TYPE_IMAGE_STARS)
        is_starless = (layer is not None and layer.mask_type == MASK_TYPE_IMAGE_STARLESS)
        is_raster = (layer is not None and getattr(layer, 'is_raster_image', False))

        self.container_layer_props.setVisible(has_sel)
        self.container_blend_mode.setVisible(is_raster)
        self.resize_handle.setVisible(len(self.layers) > 0)

        self.btn_move_up.setEnabled(has_sel)
        self.btn_move_down.setEnabled(has_sel)
        self.btn_duplicate.setEnabled(has_sel and not is_sys)
        self.btn_delete.setEnabled(has_sel and not is_sys)

        # Visibilidad de paneles especiales
        self.grp_gnd_special.setVisible(is_ground)
        self.grp_stars_special.setVisible(is_stars)
        self.grp_starless_special.setVisible(is_starless)

        # Actualizar sub-paneles con la capa activa
        self.grp_transform.update_for_layer(layer)
        self.grp_mask_def.update_for_layer(layer)
        self.grp_local_adj.update_for_layer(layer)

        # Aplicar estilo cromático e identidad
        self._apply_layer_theme(layer)

        if not layer:
            self.brush_tool_toggled.emit(False, None)
            self.layer_move_tool_toggled.emit(False)
            self.preview_mask_requested.emit(None)
            return

        self._updating_ui = True
        try:
            self.txt_layer_name.setText(layer.name)
            self.txt_layer_name.setEnabled(not is_sys)
            self.chk_layer_enabled.setChecked(layer.enabled)
            self.chk_layer_invert.setChecked(layer.inverted)
            self.chk_layer_invert.setEnabled(not is_sys)

            if is_raster:
                b_idx = self.combo_blend_mode.findData(getattr(layer, 'blend_mode', 'normal'))
                if b_idx >= 0:
                    self.combo_blend_mode.setCurrentIndex(b_idx)

            op_pct = int(round(layer.opacity * 100))
            self.slider_opacity.setValue(op_pct)
            self.lbl_opacity.setText(f"{op_pct}%")
        finally:
            self._updating_ui = False

        if not layer.enabled or is_sys:
            self.brush_tool_toggled.emit(False, None)
            self.gradient_tool_toggled.emit(False, None)
            self.layer_move_tool_toggled.emit(False)
            self.preview_mask_requested.emit(None)
        elif getattr(layer, 'is_gradient', False):
            self.brush_tool_toggled.emit(False, None)
            self.gradient_tool_toggled.emit(True, layer)
            self.layer_move_tool_toggled.emit(False)
            if self.is_mask_preview_active():
                self.grp_mask_def._emit_mask_preview()
            else:
                self.preview_mask_requested.emit(None)
        elif getattr(layer, 'supports_brush', False):
            self.gradient_tool_toggled.emit(False, None)
            if layer.mask_type == MASK_TYPE_IMAGE_RASTER:
                is_mask_target = (getattr(layer, 'active_target', 'image') == "mask" and layer.brush_mask is not None)
                self.brush_tool_toggled.emit(is_mask_target, layer if is_mask_target else None)
                if is_mask_target:
                    self.layer_move_tool_toggled.emit(False)
            else:
                self.brush_tool_toggled.emit(True, layer)
                self.layer_move_tool_toggled.emit(False)

            self.brush_params_changed.emit(
                layer.brush_radius, layer.brush_hardness, layer.brush_flow, layer.brush_mode
            )
            if self.is_mask_preview_active():
                self.grp_mask_def._emit_mask_preview()
            else:
                self.preview_mask_requested.emit(None)
        else:
            self.brush_tool_toggled.emit(False, None)
            self.gradient_tool_toggled.emit(False, None)
            self.layer_move_tool_toggled.emit(False)
            if self.is_mask_preview_active():
                self.grp_mask_def._emit_mask_preview()
            else:
                self.preview_mask_requested.emit(None)

    def _apply_layer_theme(self, layer: Optional[MaskLayer]) -> None:
        if layer is None:
            self.grp_local_adj.apply_theme("#2b3044", "#00e5ff", "#171a26", "Ajustes Locales de Revelado")
            self.grp_mask_def.apply_theme("#2b3044", "#00e5ff", "#171a26", "Definición de Máscara")
            return

        mtype = layer.mask_type
        if mtype in (MASK_TYPE_IMAGE_GROUND, MASK_TYPE_GROUND):
            border_c, title_c, bg_c = "#43a047", "#81c784", "#141f17"
            adj_title = "Ajustes de Capa de Suelo (Post-Estirado)"
            mask_def_title = "Definición de Máscara de Suelo"
        elif mtype == MASK_TYPE_IMAGE_STARLESS:
            border_c, title_c, bg_c = "#8e24aa", "#ce93d8", "#1c1424"
            adj_title = "Ajustes de Capa de Fondo Nebular"
            mask_def_title = "Definición de Máscara"
        elif mtype == MASK_TYPE_IMAGE_STARS:
            border_c, title_c, bg_c = "#fbc02d", "#ffd54f", "#221f14"
            adj_title = "Ajustes de Capa de Estrellas"
            mask_def_title = "Definición de Máscara"
        elif mtype == MASK_TYPE_BRUSH:
            border_c, title_c, bg_c = "#e53935", "#ef5350", "#221415"
            adj_title = "Ajustes Locales de Pincel"
            mask_def_title = "Trazo y Parámetros de Pincel"
        elif mtype == MASK_TYPE_LUMA:
            border_c, title_c, bg_c = "#fdd835", "#fff59d", "#202018"
            adj_title = "Ajustes Locales de Luminancia"
            mask_def_title = "Rango de Luminancia"
        elif mtype == MASK_TYPE_COLOR:
            border_c, title_c, bg_c = "#00acc1", "#80d8ff", "#141f24"
            adj_title = "Ajustes Locales de Color HSV"
            mask_def_title = "Tono y Tolerancia de Color"
        elif mtype == MASK_TYPE_SKY:
            border_c, title_c, bg_c = "#1976d2", "#90caf9", "#141924"
            adj_title = "Ajustes de Capa de Cielo"
            mask_def_title = "Definición de Máscara de Cielo"
        elif mtype == MASK_TYPE_IMAGE_BASE:
            border_c, title_c, bg_c = "#ffa000", "#ffe082", "#1f1c14"
            adj_title = "Ajustes de Imagen Base"
            mask_def_title = "Definición de Máscara"
        elif mtype == MASK_TYPE_IMAGE_RASTER:
            border_c, title_c, bg_c = "#ff9800", "#ffb74d", "#241914"
            adj_title = "Ajustes de Capa Ráster / Filtro Coronal"
            mask_def_title = "Máscara Espacial de Capa (Pincel)"
        elif mtype == MASK_TYPE_GRADIENT_LINEAR:
            border_c, title_c, bg_c = "#00bcd4", "#80deea", "#141f22"
            adj_title = "Ajustes Locales de Gradiente Lineal"
            mask_def_title = "Propiedades de Gradiente Lineal"
        elif mtype == MASK_TYPE_GRADIENT_RADIAL:
            border_c, title_c, bg_c = "#00acc1", "#80deea", "#141f22"
            adj_title = "Ajustes Locales de Gradiente Radial"
            mask_def_title = "Propiedades de Gradiente Radial"
        else:
            border_c, title_c, bg_c = "#2b3044", "#00e5ff", "#171a26"
            adj_title = "Ajustes Locales de Revelado"
            mask_def_title = "Definición de Máscara"

        self.grp_local_adj.apply_theme(border_c, title_c, bg_c, adj_title)
        self.grp_mask_def.apply_theme(border_c, title_c, bg_c, mask_def_title)
        self.grp_gnd_special.apply_theme("#43a047", "#81c784", "#141f17", "Ajustes Tonales Especiales de Suelo (Pre-Estirado Lineal)")
        self.grp_stars_special.apply_theme("#fbc02d", "#ffd54f", "#221f14", "Ajustes Especiales de Estrellas")
        self.grp_starless_special.apply_theme("#8e24aa", "#ce93d8", "#1c1424", "Ajustes Especiales de Fondo Nebular")

    def select_layer_by_id(self, layer_id: str) -> bool:
        for layer in self.layers:
            if layer.id == layer_id:
                self._rebuild_list_items(select_layer_id=layer.id)
                return True
        return False

    def select_layer_by_type(self, mask_type: str) -> bool:
        for layer in self.layers:
            if layer.mask_type == mask_type:
                self._rebuild_list_items(select_layer_id=layer.id)
                return True
        return False

    def _on_layer_name_changed(self, text: str) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and not layer.is_system_layer:
            layer.name = text.strip() or "Capa"
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            self.layers_changed.emit()

    def _on_layer_enabled_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.enabled = checked
            clear_subcomposite_cache()
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            self.layers_changed.emit()
            self._notify_action_committed()
            row = self.list_layers.currentRow()
            if 0 <= row < self.list_layers.count():
                item = self.list_layers.item(row)
                if item:
                    w = self.list_layers.itemWidget(item)
                    if w:
                        btn_eye = w.findChild(QPushButton)
                        if btn_eye:
                            fg_col = "#e0e6f0" if layer.enabled else "#78909c"
                            btn_eye.setIcon(get_icon("eye" if layer.enabled else "eye_off", color=fg_col, size=14))

    def _on_layer_invert_toggled(self, checked: bool) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and not layer.is_system_layer:
            layer.inverted = checked
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            self.layers_changed.emit()
            self._notify_action_committed()
            if self.is_mask_preview_active():
                self.grp_mask_def._emit_mask_preview()

    def _on_blend_mode_changed(self, idx: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and getattr(layer, 'is_raster_image', False):
            layer.blend_mode = self.combo_blend_mode.currentData()
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            self.layers_changed.emit()
            self._notify_action_committed()

    def _on_opacity_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.opacity = val / 100.0
            self.lbl_opacity.setText(f"{val}%")
            if layer.mask_type == MASK_TYPE_IMAGE_STARS:
                self.starnet_opacity_changed.emit(layer.opacity)
            self.invalidate_layer_thumbnail(layer.id, rebuild_now=False)
            self.layers_changed.emit()

    # --- Métodos delegados a sub-paneles ---
    def set_layer_transform_from_canvas(self, dx_norm: float, dy_norm: float, angle: float = 0.0) -> None:
        self.grp_transform.set_layer_transform_from_canvas(dx_norm, dy_norm, angle)

    def set_layer_scale_from_canvas(self, factor: float) -> None:
        self.grp_transform.set_layer_scale_from_canvas(factor)

    def _reset_layer_transform(self) -> None:
        self.grp_transform._reset_layer_transform()

    def _add_raster_layer_mask_white(self) -> None:
        self.grp_mask_def._add_raster_layer_mask_white()

    def _add_raster_layer_mask_black(self) -> None:
        self.grp_mask_def._add_raster_layer_mask_black()

    def _add_raster_layer_mask_ground(self, source_choice: Optional[str] = None) -> None:
        self.grp_mask_def._add_raster_layer_mask_ground(source_choice=source_choice)

    def _add_raster_layer_mask_sky(self, source_choice: Optional[str] = None) -> None:
        self.grp_mask_def._add_raster_layer_mask_sky(source_choice=source_choice)

    def _remove_raster_layer_mask(self) -> None:
        self.grp_mask_def._remove_raster_layer_mask()

    def _clear_brush_mask(self) -> None:
        self.grp_mask_def._clear_brush_mask()

    def _set_brush_mode(self, mode: str) -> None:
        self.grp_mask_def._set_brush_mode(mode)

    def _on_brush_param_changed(self) -> None:
        self.grp_mask_def._on_brush_param_changed()

    def set_brush_radius_from_canvas(self, radius: int) -> None:
        self.grp_mask_def.set_brush_radius_from_canvas(radius)

    def set_color_picker_active(self, active: bool) -> None:
        self.grp_mask_def.set_color_picker_active(active)

    def apply_sampled_color(self, h: float, s: float) -> None:
        self.grp_mask_def.apply_sampled_color(h, s)

    def invert_selected_brush_mask(self) -> None:
        self.grp_mask_def.invert_selected_brush_mask()

    def update_gradient_ui_from_layer(self, layer: MaskLayer) -> None:
        self.grp_mask_def.update_gradient_ui_from_layer(layer)

    def update_active_layer_histogram(self) -> None:
        self.grp_local_adj.update_active_layer_histogram()

    def _reset_local_adjustments(self) -> None:
        self.grp_local_adj._reset_local_adjustments()

    def set_ground_tone_values(self, *args, **kwargs) -> None:
        self.grp_gnd_special.set_ground_tone_values(*args, **kwargs)

    def set_ground_ai_denoise_active(self, active: bool) -> None:
        self.grp_gnd_special.set_ai_denoise_active(active)

    def set_stars_special_values(self, *args, **kwargs) -> None:
        self.grp_stars_special.set_special_values(*args, **kwargs)

    def set_starless_tone_values(self, *args, **kwargs) -> None:
        self.grp_starless_special.set_starless_tone_values(*args, **kwargs)

    def set_ai_denoise_active(self, active: bool) -> None:
        self.grp_starless_special.set_ai_denoise_active(active)

    def get_autostretch_params(self) -> Dict[str, Any]:
        return self.grp_starless_special.get_autostretch_params()

    def _notify_action_committed(self) -> None:
        self.action_committed.emit()
        self.ground_slider_released.emit()

    def clear_all_layers(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "release_heavy_data"):
                layer.release_heavy_data()
        self.layers.clear()
        clear_subcomposite_cache()
        self._rebuild_list_items(None)
        self.layers_changed.emit()

    # Serialización y deserialización para proyectos, historial y presets
    def get_state(self, encode_masks: bool = False) -> List[Any]:
        """Devuelve una lista de instantáneas de las capas para el historial o serialización."""
        if encode_masks:
            return [layer.to_dict(encode_mask=True) for layer in self.layers]
        return [layer.shallow_snapshot() for layer in self.layers]

    def set_state(self, state: Optional[List[Any]]) -> None:
        """Restaura las capas desde una lista de capas o diccionarios serializados."""
        self.layers.clear()
        if state:
            for item in state:
                try:
                    if isinstance(item, MaskLayer):
                        self.layers.append(item.shallow_snapshot())
                    elif isinstance(item, dict):
                        self.layers.append(MaskLayer.from_dict(item))
                except Exception:
                    pass
        self._rebuild_list_items(select_layer_id=self.layers[0].id if self.layers else None)
        self.layers_changed.emit()
