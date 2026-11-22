# gui/widgets/mask_layers/special_layers_panels.py
"""
Paneles contextuales para capas especiales del sistema y transformación de ráster:
- RasterTransformPanel: Mover, Escalar y Rotar capas ráster.
- GroundSpecialPanel: Controles tonales lineales de suelo pre-estirado, anti-halo y Denoise IA.
- StarsSpecialPanel: Intensidad de estrellas y filtro difusor Star Mist.
- StarlessSpecialPanel: Claridad Gauss/Textura, Ondículas, Denoise NAFNet, Efecto Orton y Estirado Adaptativo RGB.
"""
from typing import Optional, Dict, Any, Callable
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QComboBox, QGroupBox
)

from core.mask_layers import MaskLayer
from gui.icons import get_icon
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.widgets.mask_layers.common import make_box_qss


class RasterTransformPanel(QGroupBox):
    """Grupo de transformación espacial 2D para capas ráster (Mover, Escalar, Rotar)."""
    layer_move_tool_toggled = Signal(bool)
    transform_changed = Signal()
    action_committed = Signal()
    slider_released = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Transformación Espacial (Mover y Redimensionar)", parent)
        self.setObjectName("layer_transform_group")
        self._updating_ui: bool = False
        self._selected_layer_getter: Optional[Callable[[], Optional[MaskLayer]]] = None
        self._current_layer_id: Optional[str] = None
        self._setup_ui()

    def set_layer_getter(self, getter: Callable[[], Optional[MaskLayer]]) -> None:
        self._selected_layer_getter = getter

    def get_selected_layer(self) -> Optional[MaskLayer]:
        if self._selected_layer_getter:
            return self._selected_layer_getter()
        return None

    def _setup_ui(self) -> None:
        trans_layout = QVBoxLayout(self)
        trans_layout.setContentsMargins(8, 10, 8, 8)
        trans_layout.setSpacing(6)

        self.btn_toggle_move = QPushButton(" Mover y Redimensionar en Lienzo")
        self.btn_toggle_move.setIcon(get_icon("maximize", color="#80d8ff", size=13))
        self.btn_toggle_move.setCheckable(True)
        self.btn_toggle_move.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_move.setToolTip(
            "Activa el modo interactivo en el visor:\n"
            "• Arrastra con clic izquierdo para desplazar la posición (Mover X/Y).\n"
            "• Gira la rueda del ratón o arrastra con Shift/clic derecho para redimensionar (Escala)."
        )
        self.btn_toggle_move.setStyleSheet("""
            QPushButton {
                background-color: #1a2736;
                color: #80d8ff;
                border: 1px solid #0288d1;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #243447; border-color: #03a9f4; }
            QPushButton:checked { background-color: #0288d1; color: #ffffff; }
        """)
        self.btn_toggle_move.toggled.connect(self._on_toggle_move_mode)
        trans_layout.addWidget(self.btn_toggle_move)

        self.row_trans_x = PrecisionSliderRow("Mover X:", -100, 100, 0, scale=1.0, unit=" %", force_sign=True, label_width=115)
        self.slider_trans_x = self.row_trans_x.slider
        self.slider_trans_x.valueChanged.connect(self._on_transform_slider_changed)
        self.row_trans_x.sliderReleased.connect(self.slider_released.emit)
        trans_layout.addWidget(self.row_trans_x)

        self.row_trans_y = PrecisionSliderRow("Mover Y:", -100, 100, 0, scale=1.0, unit=" %", force_sign=True, label_width=115)
        self.slider_trans_y = self.row_trans_y.slider
        self.slider_trans_y.valueChanged.connect(self._on_transform_slider_changed)
        self.row_trans_y.sliderReleased.connect(self.slider_released.emit)
        trans_layout.addWidget(self.row_trans_y)

        self.chk_link_scales = QCheckBox("Vincular relación de aspecto (Escala X = Y)")
        self.chk_link_scales.setChecked(True)
        self.chk_link_scales.setStyleSheet("color: #80d8ff; font-size: 11px; margin-top: 2px; margin-bottom: 2px;")
        trans_layout.addWidget(self.chk_link_scales)

        self.row_trans_scale_x = PrecisionSliderRow("Escala X:", 10, 400, 100, scale=1.0, unit=" %", decimals=0, label_width=115)
        self.slider_trans_scale_x = self.row_trans_scale_x.slider
        self.slider_trans_scale_x.valueChanged.connect(self._on_scale_x_changed)
        self.row_trans_scale_x.sliderReleased.connect(self.slider_released.emit)
        trans_layout.addWidget(self.row_trans_scale_x)

        self.row_trans_scale_y = PrecisionSliderRow("Escala Y:", 10, 400, 100, scale=1.0, unit=" %", decimals=0, label_width=115)
        self.slider_trans_scale_y = self.row_trans_scale_y.slider
        self.slider_trans_scale_y.valueChanged.connect(self._on_scale_y_changed)
        self.row_trans_scale_y.sliderReleased.connect(self.slider_released.emit)
        trans_layout.addWidget(self.row_trans_scale_y)

        # Alias de conveniencia
        self.row_trans_scale = self.row_trans_scale_x
        self.slider_trans_scale = self.slider_trans_scale_x

        self.row_trans_angle = PrecisionSliderRow("Rotación:", -180, 180, 0, scale=1.0, unit="°", force_sign=True, label_width=115)
        self.slider_trans_angle = self.row_trans_angle.slider
        self.slider_trans_angle.valueChanged.connect(self._on_transform_slider_changed)
        self.row_trans_angle.sliderReleased.connect(self.slider_released.emit)
        trans_layout.addWidget(self.row_trans_angle)

        self.btn_reset_transform = QPushButton(" Restablecer Posición y Escala")
        self.btn_reset_transform.setIcon(get_icon("rotate-ccw", color="#80d8ff", size=12))
        self.btn_reset_transform.setCursor(Qt.PointingHandCursor)
        self.btn_reset_transform.setStyleSheet("color: #b0bec5; font-size: 11px;")
        self.btn_reset_transform.clicked.connect(self._reset_layer_transform)
        trans_layout.addWidget(self.btn_reset_transform)

    def _on_toggle_move_mode(self, checked: bool) -> None:
        if checked:
            layer = self.get_selected_layer()
            if not layer or not getattr(layer, 'is_raster_image', False) or not getattr(layer, 'enabled', True):
                self.btn_toggle_move.setChecked(False)
                return
            if getattr(layer, 'active_target', 'image') == 'mask':
                layer.active_target = 'image'
        self.layer_move_tool_toggled.emit(checked)

    def _on_scale_x_changed(self, val: int) -> None:
        if self._updating_ui: return
        if self.chk_link_scales.isChecked():
            self._updating_ui = True
            self.slider_trans_scale_y.setValue(val)
            self._updating_ui = False
        self._on_transform_slider_changed()

    def _on_scale_y_changed(self, val: int) -> None:
        if self._updating_ui: return
        if self.chk_link_scales.isChecked():
            self._updating_ui = True
            self.slider_trans_scale_x.setValue(val)
            self._updating_ui = False
        self._on_transform_slider_changed()

    def _on_transform_slider_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and getattr(layer, 'is_raster_image', False):
            layer.offset_x = self.slider_trans_x.value() / 100.0
            layer.offset_y = self.slider_trans_y.value() / 100.0
            layer.scale_x = self.slider_trans_scale_x.value() / 100.0
            layer.scale_y = self.slider_trans_scale_y.value() / 100.0
            layer.scale = layer.scale_x
            layer.angle = float(self.slider_trans_angle.value())
            self.transform_changed.emit()

    def _reset_layer_transform(self) -> None:
        layer = self.get_selected_layer()
        if layer and getattr(layer, 'is_raster_image', False):
            self._updating_ui = True
            try:
                layer.offset_x = 0.0
                layer.offset_y = 0.0
                layer.scale_x = 1.0
                layer.scale_y = 1.0
                layer.scale = 1.0
                layer.angle = 0.0
                self.slider_trans_x.setValue(0)
                self.slider_trans_y.setValue(0)
                self.slider_trans_scale_x.setValue(100)
                self.slider_trans_scale_y.setValue(100)
                self.slider_trans_angle.setValue(0)
                if hasattr(self, 'chk_link_scales'):
                    self.chk_link_scales.setChecked(True)
            finally:
                self._updating_ui = False
            self.transform_changed.emit()
            self.action_committed.emit()
            self.slider_released.emit()

    def update_for_layer(self, layer: Optional[MaskLayer]) -> None:
        layer_id = getattr(layer, "id", None) if layer is not None else None
        if layer_id != self._current_layer_id:
            self._current_layer_id = layer_id
            if self.btn_toggle_move.isChecked():
                self.btn_toggle_move.setChecked(False)

        if (
            not layer
            or not getattr(layer, 'is_raster_image', False)
            or layer.is_system_layer
            or not getattr(layer, 'enabled', True)
            or getattr(layer, 'active_target', 'image') == 'mask'
        ):
            if self.btn_toggle_move.isChecked():
                self.btn_toggle_move.setChecked(False)
            self.setVisible(False)
            return

        self.setVisible(True)
        self._updating_ui = True
        self.slider_trans_x.setValue(int(round(getattr(layer, 'offset_x', 0.0) * 100)))
        self.slider_trans_y.setValue(int(round(getattr(layer, 'offset_y', 0.0) * 100)))
        sx = int(round(getattr(layer, 'scale_x', getattr(layer, 'scale', 1.0)) * 100))
        sy = int(round(getattr(layer, 'scale_y', getattr(layer, 'scale', 1.0)) * 100))
        self.chk_link_scales.setChecked(abs(sx - sy) <= 1)
        self.slider_trans_scale_x.setValue(sx)
        self.slider_trans_scale_y.setValue(sy)
        self.slider_trans_angle.setValue(int(round(getattr(layer, 'angle', 0.0))))
        self._updating_ui = False

    def set_layer_transform_from_canvas(self, dx_norm: float, dy_norm: float, angle: float = 0.0) -> None:
        """Añade desplazamiento relativo arrastrado interactivamente desde el lienzo."""
        layer = self.get_selected_layer()
        if not layer or not getattr(layer, 'is_raster_image', False):
            if self.btn_toggle_move.isChecked():
                self.btn_toggle_move.setChecked(False)
            return
        layer.offset_x = float(np.clip(layer.offset_x + dx_norm, -1.0, 1.0))
        layer.offset_y = float(np.clip(layer.offset_y + dy_norm, -1.0, 1.0))
        if angle != 0.0:
            layer.angle = float(np.clip(layer.angle + angle, -180.0, 180.0))
        self._updating_ui = True
        try:
            self.slider_trans_x.setValue(int(round(layer.offset_x * 100)))
            self.slider_trans_y.setValue(int(round(layer.offset_y * 100)))
            self.slider_trans_angle.setValue(int(round(layer.angle)))
        finally:
            self._updating_ui = False
        self.transform_changed.emit()

    def set_layer_scale_from_canvas(self, factor: float) -> None:
        """Aplica factor de escala multiplicativo desde la rueda del ratón en el lienzo."""
        layer = self.get_selected_layer()
        if not layer or not getattr(layer, 'is_raster_image', False):
            if self.btn_toggle_move.isChecked():
                self.btn_toggle_move.setChecked(False)
            return
        cur_sx = getattr(layer, 'scale_x', getattr(layer, 'scale', 1.0))
        cur_sy = getattr(layer, 'scale_y', getattr(layer, 'scale', 1.0))
        new_sx = float(np.clip(cur_sx * factor, 0.10, 4.0))
        new_sy = float(np.clip(cur_sy * factor, 0.10, 4.0))
        layer.scale_x = new_sx
        layer.scale_y = new_sy
        layer.scale = new_sx
        self._updating_ui = True
        try:
            self.slider_trans_scale_x.setValue(int(round(new_sx * 100)))
            self.slider_trans_scale_y.setValue(int(round(new_sy * 100)))
        finally:
            self._updating_ui = False
        self.transform_changed.emit()


class GroundSpecialPanel(QGroupBox):
    """Panel de ajustes tonales especiales de Suelo (Pre-Estirado Lineal)."""
    ground_ev_changed = Signal(int)
    ground_shadows_changed = Signal(int)
    ground_bp_changed = Signal(int)
    ground_denoise_changed = Signal(int)
    ground_antihalo_changed = Signal(int)
    ground_edge_shift_changed = Signal(int)
    ground_ai_denoise_requested = Signal()
    ground_reset_requested = Signal()
    slider_released = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Ajustes Tonales Especiales de Suelo (Pre-Estirado Lineal)", parent)
        self.setObjectName("gnd_special_group")
        self._updating_ui: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        gnd_spec_layout = QVBoxLayout(self)
        gnd_spec_layout.setContentsMargins(8, 12, 8, 8)
        gnd_spec_layout.setSpacing(6)

        lbl_gnd_notice = QLabel("Ajustes lineales pre-estirado (EV lineal, sombras, anti-halo y reducción de ruido).")
        lbl_gnd_notice.setStyleSheet("color: #a5d6a7; font-size: 11px; font-style: italic;")
        gnd_spec_layout.addWidget(lbl_gnd_notice)

        self.row_gnd_special_ev = PrecisionSliderRow("Exposición Suelo:", -200, 300, 0, scale=100.0, decimals=2, unit=" EV", force_sign=True, label_width=120)
        self.slider_gnd_special_ev = self.row_gnd_special_ev.slider
        self.lbl_gnd_special_ev = self.row_gnd_special_ev.lbl_value
        self.slider_gnd_special_ev.valueChanged.connect(self._on_gnd_special_ev_changed)
        self.slider_gnd_special_ev.sliderReleased.connect(self.slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_ev)

        self.row_gnd_special_sh = PrecisionSliderRow("Sombras:", 0, 100, 0, scale=1.0, decimals=0, unit="%", label_width=120)
        self.slider_gnd_special_sh = self.row_gnd_special_sh.slider
        self.lbl_gnd_special_sh = self.row_gnd_special_sh.lbl_value
        self.slider_gnd_special_sh.valueChanged.connect(self._on_gnd_special_sh_changed)
        self.slider_gnd_special_sh.sliderReleased.connect(self.slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_sh)

        self.row_gnd_special_bp = PrecisionSliderRow("Punto Negro:", -50, 100, 0, scale=1000.0, decimals=3, force_sign=True, label_width=120)
        self.slider_gnd_special_bp = self.row_gnd_special_bp.slider
        self.lbl_gnd_special_bp = self.row_gnd_special_bp.lbl_value
        self.slider_gnd_special_bp.valueChanged.connect(self._on_gnd_special_bp_changed)
        self.slider_gnd_special_bp.sliderReleased.connect(self.slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_bp)

        self.btn_gnd_ai_denoise = QPushButton("✨ IA")
        self.btn_gnd_ai_denoise.setCheckable(True)
        self.btn_gnd_ai_denoise.setChecked(False)
        self.btn_gnd_ai_denoise.setCursor(Qt.PointingHandCursor)
        self.btn_gnd_ai_denoise.setFixedHeight(18)
        self.btn_gnd_ai_denoise.setToolTip("Reducción de Ruido Profunda con IA (NAFNet) para la superficie terrestre")
        self.btn_gnd_ai_denoise.setStyleSheet("""
            QPushButton {
                background-color: #1b3822; color: #a5d6a7; border: 1px solid #43a047;
                border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 10px;
            }
            QPushButton:hover { background-color: #254d2f; border-color: #81c784; color: #ffffff; }
        """)
        self.btn_gnd_ai_denoise.clicked.connect(self.ground_ai_denoise_requested.emit)

        self.row_gnd_special_dn = PrecisionSliderRow(
            "Reducción Ruido:", 0, 100, 0, unit="%", label_width=120, action_button=self.btn_gnd_ai_denoise
        )
        self.slider_gnd_special_dn = self.row_gnd_special_dn.slider
        self.lbl_gnd_special_dn = self.row_gnd_special_dn.lbl_value
        self.slider_gnd_special_dn.valueChanged.connect(self._on_gnd_special_dn_changed)
        self.slider_gnd_special_dn.sliderReleased.connect(self.slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_dn)

        self.row_gnd_special_antihalo = PrecisionSliderRow(
            "Anti-Halo Frontera:", 0, 100, 0, scale=1.0, decimals=0, unit="%", label_width=120
        )
        self.slider_gnd_special_antihalo = self.row_gnd_special_antihalo.slider
        self.lbl_gnd_special_antihalo = self.row_gnd_special_antihalo.lbl_value
        self.slider_gnd_special_antihalo.valueChanged.connect(self._on_gnd_special_antihalo_changed)
        self.slider_gnd_special_antihalo.sliderReleased.connect(self.slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_antihalo)

        self.row_gnd_special_edge_shift = PrecisionSliderRow(
            "Desplazar Borde:", -15, 15, 0, scale=1.0, decimals=0, unit=" px", force_sign=True, label_width=120
        )
        self.slider_gnd_special_edge_shift = self.row_gnd_special_edge_shift.slider
        self.lbl_gnd_special_edge_shift = self.row_gnd_special_edge_shift.lbl_value
        self.slider_gnd_special_edge_shift.valueChanged.connect(self._on_gnd_special_edge_shift_changed)
        self.slider_gnd_special_edge_shift.sliderReleased.connect(self.slider_released.emit)
        gnd_spec_layout.addWidget(self.row_gnd_special_edge_shift)

        self.btn_reset_gnd_special = QPushButton("Restablecer Tono Suelo")
        self.btn_reset_gnd_special.setIcon(get_icon("rotate_ccw", color="#a5d6a7", size=13))
        self.btn_reset_gnd_special.setCursor(Qt.PointingHandCursor)
        self.btn_reset_gnd_special.clicked.connect(self.ground_reset_requested.emit)
        gnd_spec_layout.addWidget(self.btn_reset_gnd_special)

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
        self.lbl_gnd_special_bp.setText(f"{val / 1000.0:+.3f}")
        self.ground_bp_changed.emit(val)

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

    def set_ground_tone_values(
        self, ev_val: int = 0, sh_val: int = 0, bp_val: int = 0,
        dn_val: int = 0, antihalo_val: int = 0, edge_shift_val: int = 0,
        **kwargs
    ) -> None:
        self._updating_ui = True
        ev = kwargs.get('ev', ev_val)
        sh = kwargs.get('sh', kwargs.get('shadows', sh_val))
        bp = kwargs.get('bp', kwargs.get('black_point', bp_val))
        dn = kwargs.get('dn', kwargs.get('denoise', dn_val))
        ah = kwargs.get('antihalo', antihalo_val)
        es = kwargs.get('edge_shift', edge_shift_val)

        self.slider_gnd_special_ev.setValue(ev)
        self.lbl_gnd_special_ev.setText(f"{ev / 100.0:+.2f} EV")
        self.slider_gnd_special_sh.setValue(sh)
        self.lbl_gnd_special_sh.setText(f"{sh}%")
        self.slider_gnd_special_bp.setValue(bp)
        self.lbl_gnd_special_bp.setText(f"{bp / 1000.0:+.3f}")
        self.slider_gnd_special_dn.setValue(dn)
        self.lbl_gnd_special_dn.setText(f"{dn}%")
        self.slider_gnd_special_antihalo.setValue(ah)
        self.lbl_gnd_special_antihalo.setText(f"{ah}%")
        self.slider_gnd_special_edge_shift.setValue(es)
        self.lbl_gnd_special_edge_shift.setText(f"{es:+d} px" if es != 0 else "0 px")
        self._updating_ui = False

    def set_ai_denoise_active(self, active: bool) -> None:
        self.btn_gnd_ai_denoise.setChecked(active)
        if active:
            self.btn_gnd_ai_denoise.setText("✓ IA")
            self.btn_gnd_ai_denoise.setStyleSheet("""
                QPushButton { background-color: #2e7d32; color: #ffffff; border: 1px solid #81c784; border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 10px; }
            """)
        else:
            self.btn_gnd_ai_denoise.setText("✨ IA")
            self.btn_gnd_ai_denoise.setStyleSheet("""
                QPushButton { background-color: #1b3822; color: #a5d6a7; border: 1px solid #43a047; border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 10px; }
                QPushButton:hover { background-color: #254d2f; border-color: #81c784; color: #ffffff; }
            """)

    def apply_theme(self, border_col: str, title_col: str, bg_col: str, title: str) -> None:
        self.setStyleSheet(make_box_qss("gnd_special_group", border_col, title_col, bg_col))
        self.setTitle(title)


class StarsSpecialPanel(QGroupBox):
    """Panel de ajustes especiales de Estrellas y filtro difusor Star Mist."""
    stars_intensity_changed = Signal(int)
    stars_diff_strength_changed = Signal(int)
    stars_diff_radius_changed = Signal(int)
    stars_diff_threshold_changed = Signal(int)
    stars_diff_color_boost_changed = Signal(int)
    stars_diff_mode_changed = Signal(str)
    stars_reset_requested = Signal()
    slider_released = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Ajustes Especiales de Estrellas", parent)
        self.setObjectName("stars_special_group")
        self._updating_ui: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        stars_spec_layout = QVBoxLayout(self)
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
        self.row_stars_special.sliderReleased.connect(self.slider_released.emit)
        stars_spec_layout.addWidget(self.row_stars_special)

        # Star Mist
        self.box_sl_star_diff = QGroupBox("Filtro Difusor y Realce de Constelaciones")
        self.box_sl_star_diff.setObjectName("sl_star_diff_box")
        self.box_sl_star_diff.setStyleSheet("""
            QGroupBox#sl_star_diff_box {
                font-weight: bold; border: 1px solid #fbc02d; border-radius: 6px;
                margin-top: 14px; padding-top: 14px; background-color: #1e1b12; color: #ffe082;
            }
            QGroupBox#sl_star_diff_box::title {
                subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 6px;
                color: #fff9c4; font-weight: bold; font-size: 11px; background-color: #16130c; border-radius: 3px;
            }
        """)
        sl_sd_layout = QVBoxLayout(self.box_sl_star_diff)
        sl_sd_layout.setContentsMargins(8, 8, 8, 8)
        sl_sd_layout.setSpacing(6)

        self.row_sl_star_diff_strength = PrecisionSliderRow("Intensidad Difusor:", 0, 100, 0, unit="%", label_width=130)
        self.slider_sl_star_diff_strength = self.row_sl_star_diff_strength.slider
        self.lbl_sl_star_diff_strength = self.row_sl_star_diff_strength.lbl_value
        self.row_sl_star_diff_strength.valueChanged.connect(self._on_sl_star_diff_strength_changed)
        self.row_sl_star_diff_strength.resetRequested.connect(lambda: self._on_sl_star_diff_strength_changed(self.row_sl_star_diff_strength.raw_default))
        self.row_sl_star_diff_strength.sliderReleased.connect(self.slider_released.emit)
        sl_sd_layout.addWidget(self.row_sl_star_diff_strength)

        self.row_sl_star_diff_radius = PrecisionSliderRow("Radio (Halo):", 5, 60, 20, unit=" px", label_width=130)
        self.slider_sl_star_diff_radius = self.row_sl_star_diff_radius.slider
        self.lbl_sl_star_diff_radius = self.row_sl_star_diff_radius.lbl_value
        self.row_sl_star_diff_radius.valueChanged.connect(self._on_sl_star_diff_radius_changed)
        self.row_sl_star_diff_radius.resetRequested.connect(lambda: self._on_sl_star_diff_radius_changed(self.row_sl_star_diff_radius.raw_default))
        self.row_sl_star_diff_radius.sliderReleased.connect(self.slider_released.emit)
        sl_sd_layout.addWidget(self.row_sl_star_diff_radius)

        self.row_sl_star_diff_threshold = PrecisionSliderRow("Umbral:", 15, 85, 45, scale=100.0, decimals=2, label_width=130)
        self.slider_sl_star_diff_threshold = self.row_sl_star_diff_threshold.slider
        self.lbl_sl_star_diff_threshold = self.row_sl_star_diff_threshold.lbl_value
        self.row_sl_star_diff_threshold.valueChanged.connect(self._on_sl_star_diff_threshold_changed)
        self.row_sl_star_diff_threshold.resetRequested.connect(lambda: self._on_sl_star_diff_threshold_changed(self.row_sl_star_diff_threshold.raw_default))
        self.row_sl_star_diff_threshold.sliderReleased.connect(self.slider_released.emit)
        sl_sd_layout.addWidget(self.row_sl_star_diff_threshold)

        self.row_sl_star_diff_color = PrecisionSliderRow("Color Espectral:", 0, 100, 50, unit="%", label_width=130)
        self.slider_sl_star_diff_color = self.row_sl_star_diff_color.slider
        self.lbl_sl_star_diff_color = self.row_sl_star_diff_color.lbl_value
        self.row_sl_star_diff_color.valueChanged.connect(self._on_sl_star_diff_color_changed)
        self.row_sl_star_diff_color.resetRequested.connect(lambda: self._on_sl_star_diff_color_changed(self.row_sl_star_diff_color.raw_default))
        self.row_sl_star_diff_color.sliderReleased.connect(self.slider_released.emit)
        sl_sd_layout.addWidget(self.row_sl_star_diff_color)

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
        self.btn_reset_stars_special.clicked.connect(self.slider_released.emit)
        stars_spec_layout.addWidget(self.btn_reset_stars_special)

    def _on_stars_special_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.stars_intensity_changed.emit(val)

    def _on_sl_star_diff_strength_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.stars_diff_strength_changed.emit(val)

    def _on_sl_star_diff_radius_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.stars_diff_radius_changed.emit(val)

    def _on_sl_star_diff_threshold_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.stars_diff_threshold_changed.emit(val)

    def _on_sl_star_diff_color_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.stars_diff_color_boost_changed.emit(val)

    def _on_sl_star_diff_mode_changed(self) -> None:
        if self._updating_ui: return
        self.stars_diff_mode_changed.emit(self.combo_sl_star_diff_mode.currentData())

    def set_special_values(
        self,
        presence: Optional[int] = None,
        diff_strength: Optional[int] = None,
        diff_radius: Optional[int] = None,
        diff_threshold: Optional[int] = None,
        diff_color: Optional[int] = None,
        diff_mode_idx: Optional[int] = None,
        diff_mode: Optional[str] = None,
        intensity_val: Optional[int] = None,
        **kwargs
    ) -> None:
        self._updating_ui = True
        val = presence if presence is not None else intensity_val
        if val is None and 'intensity' in kwargs:
            val = kwargs['intensity']
        if val is not None and hasattr(self, 'row_stars_special'):
            self.row_stars_special.setValue(val)
            self.lbl_stars_special.setText(f"{val}%")
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
        if diff_mode_idx is not None and hasattr(self, 'combo_sl_star_diff_mode'):
            self.combo_sl_star_diff_mode.setCurrentIndex(diff_mode_idx)
        elif diff_mode is not None and hasattr(self, 'combo_sl_star_diff_mode'):
            idx = self.combo_sl_star_diff_mode.findData(diff_mode)
            if idx >= 0:
                self.combo_sl_star_diff_mode.setCurrentIndex(idx)
        self._updating_ui = False

    def set_stars_special_values(self, *args, **kwargs) -> None:
        self.set_special_values(*args, **kwargs)

    def apply_theme(self, border_col: str, title_col: str, bg_col: str, title: str) -> None:
        self.setStyleSheet(make_box_qss("stars_special_group", border_col, title_col, bg_col))
        self.setTitle(title)


class StarlessSpecialPanel(QGroupBox):
    """Panel de ajustes especiales de Fondo Nebular (Starless), Ondículas, Orton y Estirado."""
    starless_clarity_changed = Signal(int)
    starless_texture_clarity_changed = Signal(int)
    starless_dehaze_changed = Signal(int)
    starless_ond_est_changed = Signal(int)
    starless_ond_bg_changed = Signal(int)
    starless_contrast_changed = Signal(int)
    starless_denoise_changed = Signal(int)
    starless_denoise_color_changed = Signal(int)
    starless_denoise_mask_changed = Signal(int)
    starless_ai_denoise_requested = Signal()
    starless_orton_strength_changed = Signal(int)
    starless_orton_radius_changed = Signal(int)
    starless_orton_threshold_changed = Signal(int)
    starless_orton_tint_changed = Signal(str)
    starless_orton_mode_changed = Signal(str)
    starless_reset_requested = Signal()
    auto_stretch_starless_requested = Signal()
    slider_released = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Ajustes Especiales de Fondo Nebular", parent)
        self.setObjectName("starless_special_group")
        self._updating_ui: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        sl_layout = QVBoxLayout(self)
        sl_layout.setContentsMargins(8, 12, 8, 8)
        sl_layout.setSpacing(6)

        lbl_sl_notice = QLabel("Ajustes avanzados para resaltar estructuras nebulares y contraste galáctico.")
        lbl_sl_notice.setStyleSheet("color: #ce93d8; font-size: 11px; font-style: italic;")
        sl_layout.addWidget(lbl_sl_notice)

        self.row_sl_clarity = PrecisionSliderRow("Claridad (Gauss):", -200, 200, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_clarity = self.row_sl_clarity.slider
        self.lbl_sl_clarity = self.row_sl_clarity.lbl_value
        self.slider_sl_clarity.valueChanged.connect(self._on_sl_clarity_changed)
        self.slider_sl_clarity.sliderReleased.connect(self.slider_released.emit)
        sl_layout.addWidget(self.row_sl_clarity)

        self.row_sl_texture_clarity = PrecisionSliderRow("Claridad Textura:", -200, 200, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_texture_clarity = self.row_sl_texture_clarity.slider
        self.lbl_sl_texture_clarity = self.row_sl_texture_clarity.lbl_value
        self.slider_sl_texture_clarity.valueChanged.connect(self._on_sl_texture_clarity_changed)
        self.slider_sl_texture_clarity.sliderReleased.connect(self.slider_released.emit)
        sl_layout.addWidget(self.row_sl_texture_clarity)

        self.row_sl_dehaze = PrecisionSliderRow("Borrar Neblina:", -200, 200, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_dehaze = self.row_sl_dehaze.slider
        self.lbl_sl_dehaze = self.row_sl_dehaze.lbl_value
        self.slider_sl_dehaze.valueChanged.connect(self._on_sl_dehaze_changed)
        self.slider_sl_dehaze.sliderReleased.connect(self.slider_released.emit)
        sl_layout.addWidget(self.row_sl_dehaze)

        self.row_sl_ond_est = PrecisionSliderRow("Estructura:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_ond_est = self.row_sl_ond_est.slider
        self.lbl_sl_ond_est = self.row_sl_ond_est.lbl_value
        self.slider_sl_ond_est.valueChanged.connect(self._on_sl_ond_est_changed)
        self.slider_sl_ond_est.sliderReleased.connect(self.slider_released.emit)
        sl_layout.addWidget(self.row_sl_ond_est)

        self.row_sl_ond_bg = PrecisionSliderRow("Atenuar Fondo:", 0, 100, 0, unit="%", label_width=130)
        self.slider_sl_ond_bg = self.row_sl_ond_bg.slider
        self.lbl_sl_ond_bg = self.row_sl_ond_bg.lbl_value
        self.slider_sl_ond_bg.valueChanged.connect(self._on_sl_ond_bg_changed)
        self.slider_sl_ond_bg.sliderReleased.connect(self.slider_released.emit)
        sl_layout.addWidget(self.row_sl_ond_bg)

        self.row_sl_contrast = PrecisionSliderRow("Contraste Fondo:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=130)
        self.slider_sl_contrast = self.row_sl_contrast.slider
        self.lbl_sl_contrast = self.row_sl_contrast.lbl_value
        self.slider_sl_contrast.valueChanged.connect(self._on_sl_contrast_changed)
        self.slider_sl_contrast.sliderReleased.connect(self.slider_released.emit)
        sl_layout.addWidget(self.row_sl_contrast)

        # Reducción Ruido con botón IA y expandir
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

        self.btn_sl_ai_denoise = QPushButton("✨ IA")
        self.btn_sl_ai_denoise.setCheckable(True)
        self.btn_sl_ai_denoise.setChecked(False)
        self.btn_sl_ai_denoise.setCursor(Qt.PointingHandCursor)
        self.btn_sl_ai_denoise.setFixedHeight(18)
        self.btn_sl_ai_denoise.setToolTip("Ejecutar Reducción de Ruido Profunda con IA (NAFNet)")
        self.btn_sl_ai_denoise.setStyleSheet("""
            QPushButton {
                background-color: #2b1f3d; color: #ce93d8; border: 1px solid #7b1fa2;
                border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 10px;
            }
            QPushButton:hover { background-color: #3b2854; border-color: #ba68c8; color: #ffffff; }
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
        self.slider_sl_denoise.sliderReleased.connect(self.slider_released.emit)
        sl_layout.addWidget(self.row_sl_denoise)

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
        self.slider_sl_denoise_color.sliderReleased.connect(self.slider_released.emit)
        layout_sl_dn_adv.addWidget(self.row_sl_denoise_color)

        self.row_sl_denoise_mask = PrecisionSliderRow("Máscara Bordes/Sombras:", 0, 100, 50, unit="%", label_width=130)
        self.slider_sl_denoise_mask = self.row_sl_denoise_mask.slider
        self.lbl_sl_denoise_mask = self.row_sl_denoise_mask.lbl_value
        self.slider_sl_denoise_mask.valueChanged.connect(self._on_sl_denoise_mask_changed)
        self.slider_sl_denoise_mask.sliderReleased.connect(self.slider_released.emit)
        layout_sl_dn_adv.addWidget(self.row_sl_denoise_mask)

        self.btn_sl_denoise_expand.toggled.connect(self.w_sl_denoise_advanced.setVisible)
        sl_layout.addWidget(self.w_sl_denoise_advanced)

        # Orton Effect
        self.box_sl_orton = QGroupBox("Efecto Orton y Glow de Polvo Galáctico")
        self.box_sl_orton.setObjectName("sl_orton_box")
        self.box_sl_orton.setStyleSheet("""
            QGroupBox#sl_orton_box {
                font-weight: bold; border: 1px solid #8e24aa; border-radius: 6px;
                margin-top: 14px; padding-top: 14px; background-color: #191424; color: #ce93d8;
            }
            QGroupBox#sl_orton_box::title {
                subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 6px;
                color: #f3e5f5; font-weight: bold; font-size: 11px; background-color: #130f1b; border-radius: 3px;
            }
        """)
        orton_layout = QVBoxLayout(self.box_sl_orton)
        orton_layout.setContentsMargins(8, 8, 8, 8)
        orton_layout.setSpacing(6)

        self.row_sl_orton_strength = PrecisionSliderRow("Intensidad Glow:", 0, 100, 0, unit="%", label_width=130)
        self.slider_sl_orton_strength = self.row_sl_orton_strength.slider
        self.lbl_sl_orton_strength = self.row_sl_orton_strength.lbl_value
        self.slider_sl_orton_strength.valueChanged.connect(self._on_sl_orton_strength_changed)
        self.slider_sl_orton_strength.sliderReleased.connect(self.slider_released.emit)
        orton_layout.addWidget(self.row_sl_orton_strength)

        self.row_sl_orton_radius = PrecisionSliderRow("Radio Difusión:", 10, 150, 40, unit=" px", label_width=130)
        self.slider_sl_orton_radius = self.row_sl_orton_radius.slider
        self.lbl_sl_orton_radius = self.row_sl_orton_radius.lbl_value
        self.slider_sl_orton_radius.valueChanged.connect(self._on_sl_orton_radius_changed)
        self.slider_sl_orton_radius.sliderReleased.connect(self.slider_released.emit)
        orton_layout.addWidget(self.row_sl_orton_radius)

        self.row_sl_orton_threshold = PrecisionSliderRow("Umbral Polvo:", 0, 50, 5, scale=100.0, decimals=2, label_width=130)
        self.slider_sl_orton_threshold = self.row_sl_orton_threshold.slider
        self.lbl_sl_orton_threshold = self.row_sl_orton_threshold.lbl_value
        self.slider_sl_orton_threshold.valueChanged.connect(self._on_sl_orton_threshold_changed)
        self.slider_sl_orton_threshold.sliderReleased.connect(self.slider_released.emit)
        orton_layout.addWidget(self.row_sl_orton_threshold)

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

        # Estirado Adaptativo RGB
        row_as_header = QHBoxLayout()
        row_as_header.setSpacing(4)

        self.btn_sl_autostretch = QPushButton("Estirado Adaptativo RGB")
        self.btn_sl_autostretch.setIcon(get_icon("sparkles", color="#ce93d8", size=13))
        self.btn_sl_autostretch.setCursor(Qt.PointingHandCursor)
        self.btn_sl_autostretch.setStyleSheet("""
            QPushButton {
                background-color: #2b1f3d; color: #ce93d8; border: 1px solid #7b1fa2;
                border-radius: 4px; padding: 6px 12px; font-weight: bold; font-size: 11px;
            }
            QPushButton:hover { background-color: #3b2854; border-color: #ba68c8; color: #ffffff; }
        """)
        self.btn_sl_autostretch.clicked.connect(self.auto_stretch_starless_requested.emit)
        row_as_header.addWidget(self.btn_sl_autostretch, stretch=1)

        self.btn_sl_autostretch_expand = QPushButton()
        self.btn_sl_autostretch_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#ce93d8", size=12))
        self.btn_sl_autostretch_expand.setCheckable(True)
        self.btn_sl_autostretch_expand.setChecked(False)
        self.btn_sl_autostretch_expand.setFixedSize(28, 28)
        self.btn_sl_autostretch_expand.setStyleSheet("""
            QPushButton { font-size: 11px; padding: 2px; border-radius: 4px; background: #262933; color: #cfd8dc; border: 1px solid #3b4252; }
            QPushButton:hover { background: #323846; border-color: #7b1fa2; }
            QPushButton:checked { background: #4a148c; color: #e1bee7; border: 1px solid #ba68c8; }
        """)
        row_as_header.addWidget(self.btn_sl_autostretch_expand)
        sl_layout.addLayout(row_as_header)

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
        self.chk_sl_as_unlinked.setStyleSheet("color: #ce93d8; font-size: 11px; margin-left: 12px; margin-top: 2px;")
        w_as_layout.addWidget(self.chk_sl_as_unlinked)

        self.btn_sl_autostretch_expand.toggled.connect(self.w_sl_autostretch_advanced.setVisible)
        sl_layout.addWidget(self.w_sl_autostretch_advanced)

        self.btn_reset_sl_special = QPushButton("Restablecer Ajustes de Fondo")
        self.btn_reset_sl_special.setIcon(get_icon("rotate_ccw", color="#ce93d8", size=13))
        self.btn_reset_sl_special.setCursor(Qt.PointingHandCursor)
        self.btn_reset_sl_special.clicked.connect(self.starless_reset_requested.emit)
        sl_layout.addWidget(self.btn_reset_sl_special)

    def _on_sl_clarity_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_clarity.setText(f"{val / 100.0:+.2f}")
        self.starless_clarity_changed.emit(val)

    def _on_sl_texture_clarity_changed(self, val: int) -> None:
        if self._updating_ui: return
        self.lbl_sl_texture_clarity.setText(f"{val / 100.0:+.2f}")
        self.starless_texture_clarity_changed.emit(val)

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

    def _on_sl_orton_tint_changed(self) -> None:
        if self._updating_ui: return
        self.starless_orton_tint_changed.emit(self.combo_sl_orton_tint.currentData())

    def _on_sl_orton_mode_changed(self) -> None:
        if self._updating_ui: return
        self.starless_orton_mode_changed.emit(self.combo_sl_orton_mode.currentData())

    def set_ai_denoise_active(self, active: bool) -> None:
        self.btn_sl_ai_denoise.setChecked(active)
        if active:
            self.btn_sl_ai_denoise.setText("✓ IA")
            self.btn_sl_ai_denoise.setStyleSheet("""
                QPushButton { background-color: #4a148c; color: #ffffff; border: 1px solid #ba68c8; border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 10px; }
            """)
        else:
            self.btn_sl_ai_denoise.setText("✨ IA")
            self.btn_sl_ai_denoise.setStyleSheet("""
                QPushButton { background-color: #2b1f3d; color: #ce93d8; border: 1px solid #7b1fa2; border-radius: 3px; padding: 1px 6px; font-weight: bold; font-size: 10px; }
                QPushButton:hover { background-color: #3b2854; border-color: #ba68c8; color: #ffffff; }
            """)

    def set_starless_tone_values(
        self,
        clarity: Optional[int] = None,
        dehaze: Optional[int] = None,
        ond_est: Optional[int] = None,
        ond_bg: Optional[int] = None,
        contrast: Optional[int] = None,
        denoise: Optional[int] = None,
        orton_strength: Optional[int] = None,
        orton_radius: Optional[int] = None,
        orton_threshold: Optional[int] = None,
        orton_tint: Any = None,
        orton_mode: Any = None,
        *args,
        **kwargs
    ) -> None:
        self._updating_ui = True
        c = clarity if clarity is not None else kwargs.get('clarity_val', kwargs.get('clarity'))
        tc = kwargs.get('texture_clarity', kwargs.get('texture_clarity_val'))
        dh = dehaze if dehaze is not None else kwargs.get('dehaze_val', kwargs.get('dehaze'))
        oe = ond_est if ond_est is not None else kwargs.get('ond_est_val', kwargs.get('ond_est'))
        ob = ond_bg if ond_bg is not None else kwargs.get('ond_bg_val', kwargs.get('ond_bg'))
        ct = contrast if contrast is not None else kwargs.get('contrast_val', kwargs.get('contrast'))
        dn = denoise if denoise is not None else kwargs.get('denoise_val', kwargs.get('denoise'))
        dc = kwargs.get('denoise_color', kwargs.get('denoise_color_val'))
        dm = kwargs.get('denoise_mask', kwargs.get('denoise_mask_val'))
        os_ = orton_strength if orton_strength is not None else kwargs.get('orton_strength_val', kwargs.get('orton_strength'))
        or_ = orton_radius if orton_radius is not None else kwargs.get('orton_radius_val', kwargs.get('orton_radius'))
        ot = orton_threshold if orton_threshold is not None else kwargs.get('orton_threshold_val', kwargs.get('orton_threshold'))
        otint = orton_tint if orton_tint is not None else kwargs.get('orton_tint_val', kwargs.get('orton_tint_idx', kwargs.get('orton_tint')))
        omode = orton_mode if orton_mode is not None else kwargs.get('orton_mode_val', kwargs.get('orton_mode_idx', kwargs.get('orton_mode')))

        if c is not None and hasattr(self, 'slider_sl_clarity'):
            self.slider_sl_clarity.setValue(c)
            self.lbl_sl_clarity.setText(f"{c / 100.0:+.2f}")
        if tc is not None and hasattr(self, 'slider_sl_texture_clarity'):
            self.slider_sl_texture_clarity.setValue(tc)
            self.lbl_sl_texture_clarity.setText(f"{tc / 100.0:+.2f}")
        if dh is not None and hasattr(self, 'slider_sl_dehaze'):
            self.slider_sl_dehaze.setValue(dh)
            self.lbl_sl_dehaze.setText(f"{dh / 100.0:+.2f}")
        if oe is not None and hasattr(self, 'slider_sl_ond_est'):
            self.slider_sl_ond_est.setValue(oe)
            self.lbl_sl_ond_est.setText(f"{oe / 100.0:+.2f}")
        if ob is not None and hasattr(self, 'slider_sl_ond_bg'):
            self.slider_sl_ond_bg.setValue(ob)
            self.lbl_sl_ond_bg.setText(f"{ob}%")
        if ct is not None and hasattr(self, 'slider_sl_contrast'):
            self.slider_sl_contrast.setValue(ct)
            self.lbl_sl_contrast.setText(f"{ct / 100.0:+.2f}")
        if dn is not None and hasattr(self, 'slider_sl_denoise'):
            self.slider_sl_denoise.setValue(dn)
            self.lbl_sl_denoise.setText(f"{dn}%")
        if dc is not None and hasattr(self, 'slider_sl_denoise_color'):
            self.slider_sl_denoise_color.setValue(dc)
            self.lbl_sl_denoise_color.setText(f"{dc}%")
        if dm is not None and hasattr(self, 'slider_sl_denoise_mask'):
            self.slider_sl_denoise_mask.setValue(dm)
            self.lbl_sl_denoise_mask.setText(f"{dm}%")
        if os_ is not None and hasattr(self, 'slider_sl_orton_strength'):
            self.slider_sl_orton_strength.setValue(os_)
            self.lbl_sl_orton_strength.setText(f"{os_}%")
        if or_ is not None and hasattr(self, 'slider_sl_orton_radius'):
            self.slider_sl_orton_radius.setValue(or_)
            self.lbl_sl_orton_radius.setText(f"{or_} px")
        if ot is not None and hasattr(self, 'slider_sl_orton_threshold'):
            self.slider_sl_orton_threshold.setValue(ot)
            self.lbl_sl_orton_threshold.setText(f"{ot / 100.0:.2f}")
        if otint is not None and hasattr(self, 'combo_sl_orton_tint'):
            if isinstance(otint, int):
                self.combo_sl_orton_tint.setCurrentIndex(otint)
            else:
                idx = self.combo_sl_orton_tint.findData(otint)
                if idx >= 0:
                    self.combo_sl_orton_tint.setCurrentIndex(idx)
        if omode is not None and hasattr(self, 'combo_sl_orton_mode'):
            if isinstance(omode, int):
                self.combo_sl_orton_mode.setCurrentIndex(omode)
            else:
                idx = self.combo_sl_orton_mode.findData(omode)
                if idx >= 0:
                    self.combo_sl_orton_mode.setCurrentIndex(idx)
        self._updating_ui = False

    def get_autostretch_params(self) -> Dict[str, Any]:
        return {
            "target_b": float(self.row_sl_as_target_bg.slider.value()) / 100.0,
            "shadows_clip_sigma": float(self.row_sl_as_sigma.slider.value()) / 10.0,
            "sat_boost": float(self.row_sl_as_sat.slider.value()) / 10.0,
            "feather_px": int(self.row_sl_as_feather.slider.value()),
            "unlinked": self.chk_sl_as_unlinked.isChecked()
        }

    def apply_theme(self, border_col: str, title_col: str, bg_col: str, title: str) -> None:
        self.setStyleSheet(make_box_qss("starless_special_group", border_col, title_col, bg_col))
        self.setTitle(title)
