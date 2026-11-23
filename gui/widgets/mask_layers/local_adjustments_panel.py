# gui/widgets/mask_layers/local_adjustments_panel.py
"""
Panel de ajustes locales de revelado (EV, Contraste, Sombras, Luces, Claridad, Curva Tonal y Enfoque).
Encapsula grp_local_adj.
"""
from typing import Optional, Callable, Any, Tuple, List
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox, QApplication
)

from core.mask_layers import (
    MaskLayer,
    compute_effective_layer_mask,
    get_canvas_below_layer,
    MASK_TYPE_IMAGE_BASE,
    MASK_TYPE_IMAGE_GROUND,
    MASK_TYPE_IMAGE_STARLESS,
    MASK_TYPE_IMAGE_STARS,
    MASK_TYPE_IMAGE_RASTER,
)
from gui.icons import get_icon
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.widgets.curve_widget import CurveWidget
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.mask_layers.common import make_box_qss


class LocalAdjustmentsPanel(QGroupBox):
    """Panel de ajustes locales de revelado y máscara de enfoque de la capa."""
    adjustments_changed = Signal()
    slider_released = Signal()
    preview_mask_requested = Signal(object)
    interaction_started = Signal()
    interaction_ended = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("Ajustes Locales de Revelado", parent)
        self.setObjectName("local_adj_group")

        self._updating_ui: bool = False
        self._is_scrubbing: bool = False
        self._alt_scrubbing_sharp: bool = False

        self._image_reference_getter: Optional[Callable[[], Any]] = None
        self._master_mask_getter: Optional[Callable[[], Any]] = None
        self._stars_getter: Optional[Callable[[], Any]] = None
        self._starless_getter: Optional[Callable[[], Any]] = None
        self._layer_image_provider: Optional[Callable[..., Any]] = None
        self._layers_getter: Optional[Callable[[], list]] = None
        self._selected_layer_getter: Optional[Callable[[], Optional[MaskLayer]]] = None

        self._setup_ui()

    def set_getters(
        self,
        img_getter: Optional[Callable[[], Any]] = None,
        mask_getter: Optional[Callable[[], Any]] = None,
        stars_getter: Optional[Callable[[], Any]] = None,
        starless_getter: Optional[Callable[[], Any]] = None,
        layer_img_provider: Optional[Callable[..., Any]] = None,
        layers_getter: Optional[Callable[[], list]] = None,
        selected_layer_getter: Optional[Callable[[], Optional[MaskLayer]]] = None,
    ) -> None:
        self._image_reference_getter = img_getter
        self._master_mask_getter = mask_getter
        self._stars_getter = stars_getter
        self._starless_getter = starless_getter
        self._layer_image_provider = layer_img_provider
        self._layers_getter = layers_getter
        self._selected_layer_getter = selected_layer_getter

    def get_selected_layer(self) -> Optional[MaskLayer]:
        if self._selected_layer_getter:
            return self._selected_layer_getter()
        return None

    def _setup_ui(self) -> None:
        adj_layout = QVBoxLayout(self)
        adj_layout.setContentsMargins(8, 12, 8, 8)
        adj_layout.setSpacing(6)

        # Exposición (EV)
        self.row_adj_ev = PrecisionSliderRow("Exposición:", -300, 300, 0, scale=100.0, decimals=2, unit=" EV", force_sign=True, label_width=115)
        self.slider_adj_ev = self.row_adj_ev.slider
        self.lbl_adj_ev = self.row_adj_ev.lbl_value
        self.slider_adj_ev.valueChanged.connect(self._on_adj_ev_changed)
        self.row_adj_ev.sliderReleased.connect(self.slider_released.emit)
        adj_layout.addWidget(self.row_adj_ev)

        # Contraste
        self.row_adj_contrast = PrecisionSliderRow("Contraste:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_contrast = self.row_adj_contrast.slider
        self.lbl_adj_contrast = self.row_adj_contrast.lbl_value
        self.slider_adj_contrast.valueChanged.connect(self._on_adj_contrast_changed)
        self.row_adj_contrast.sliderReleased.connect(self.slider_released.emit)
        adj_layout.addWidget(self.row_adj_contrast)

        # Sombras
        self.row_adj_shadows = PrecisionSliderRow("Sombras:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_shadows = self.row_adj_shadows.slider
        self.lbl_adj_shadows = self.row_adj_shadows.lbl_value
        self.slider_adj_shadows.valueChanged.connect(self._on_adj_shadows_changed)
        self.row_adj_shadows.sliderReleased.connect(self.slider_released.emit)
        adj_layout.addWidget(self.row_adj_shadows)

        # Luces
        self.row_adj_highlights = PrecisionSliderRow("Luces:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_highlights = self.row_adj_highlights.slider
        self.lbl_adj_highlights = self.row_adj_highlights.lbl_value
        self.slider_adj_highlights.valueChanged.connect(self._on_adj_highlights_changed)
        self.row_adj_highlights.sliderReleased.connect(self.slider_released.emit)
        adj_layout.addWidget(self.row_adj_highlights)

        # Claridad Clásica (Gauss)
        self.row_adj_clarity = PrecisionSliderRow("Claridad (Gauss):", -200, 200, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_clarity = self.row_adj_clarity.slider
        self.lbl_adj_clarity = self.row_adj_clarity.lbl_value
        self.slider_adj_clarity.valueChanged.connect(self._on_adj_clarity_changed)
        self.row_adj_clarity.sliderReleased.connect(self.slider_released.emit)
        self.row_adj_clarity.setToolTip("Claridad clásica de frecuencias medias basada en paso alto gaussiano.")
        adj_layout.addWidget(self.row_adj_clarity)

        # Claridad Textura (Seti Astro)
        self.row_adj_texture_clarity = PrecisionSliderRow("Claridad Textura:", -200, 200, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_adj_texture_clarity = self.row_adj_texture_clarity.slider
        self.lbl_adj_texture_clarity = self.row_adj_texture_clarity.lbl_value
        self.slider_adj_texture_clarity.valueChanged.connect(self._on_adj_texture_clarity_changed)
        self.row_adj_texture_clarity.sliderReleased.connect(self.slider_released.emit)
        self.row_adj_texture_clarity.setToolTip("Claridad de textura bilateral modulada por campana de medios tonos 4L(1-L) (Seti Astro).")
        adj_layout.addWidget(self.row_adj_texture_clarity)

        # Saturación con doble capa)
        self.row_adj_sat = PrecisionSliderRow("Saturación:", 0, 250, 100, scale=100.0, decimals=2, unit="x", label_width=115)
        self.row_adj_sat.set_saturation_style()
        self.slider_adj_sat = self.row_adj_sat.slider
        self.lbl_adj_sat = self.row_adj_sat.lbl_value
        self.slider_adj_sat.valueChanged.connect(self._on_adj_sat_changed)
        self.row_adj_sat.sliderReleased.connect(self.slider_released.emit)
        adj_layout.addWidget(self.row_adj_sat)

        # Temperatura WB
        self.row_adj_temp = PrecisionSliderRow("Temp WB:", -10000, 10000, 0, scale=10000.0, decimals=4, force_sign=True, label_width=115)
        self.row_adj_temp.set_track_gradient("qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4B739B, stop:0.5 #2C2C32, stop:1 #9B794B)")
        self.slider_adj_temp = self.row_adj_temp.slider
        self.spin_adj_temp = self.row_adj_temp.spin
        self.lbl_adj_temp = self.row_adj_temp.lbl_value
        self.slider_adj_temp.valueChanged.connect(self._on_slider_adj_temp_changed)
        self.row_adj_temp.sliderReleased.connect(self.slider_released.emit)
        adj_layout.addWidget(self.row_adj_temp)

        # Tinte WB
        self.row_adj_tint = PrecisionSliderRow("Tinte WB:", -10000, 10000, 0, scale=10000.0, decimals=4, force_sign=True, label_width=115)
        self.row_adj_tint.set_track_gradient("qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4B9B65, stop:0.5 #2C2C32, stop:1 #9B4B8E)")
        self.slider_adj_tint = self.row_adj_tint.slider
        self.spin_adj_tint = self.row_adj_tint.spin
        self.lbl_adj_tint = self.row_adj_tint.lbl_value
        self.slider_adj_tint.valueChanged.connect(self._on_slider_adj_tint_changed)
        self.row_adj_tint.sliderReleased.connect(self.slider_released.emit)
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
        self.layer_curve_widget.curveCommitted.connect(
            lambda: self.slider_released.emit() if not self._updating_ui else None
        )
        curve_layout.addWidget(self.layer_curve_widget)

        self.sec_layer_curve.set_content_widget(w_curve)
        self.sec_layer_curve.toggled.connect(lambda exp: self.update_active_layer_histogram() if exp else None)
        adj_layout.addWidget(self.sec_layer_curve)

        # Máscara de Enfoque y Nitidez
        self.sec_layer_sharp = CollapsibleSection(
            "Máscara de Enfoque y Nitidez",
            is_expanded=False,
            icon_name="zap",
            on_reset=self._reset_layer_sharpening
        )
        w_sharp = QWidget()
        sharp_layout = QVBoxLayout(w_sharp)
        sharp_layout.setContentsMargins(4, 6, 4, 6)
        sharp_layout.setSpacing(6)

        self.row_sharp_amount = PrecisionSliderRow("Cantidad:", 0, 150, 0, scale=1.0, decimals=0, unit="%", label_width=95)
        self.slider_adj_sharp_amount = self.row_sharp_amount.slider
        self.lbl_adj_sharp_amount = self.row_sharp_amount.lbl_value
        self.slider_adj_sharp_amount.valueChanged.connect(self._on_adj_sharp_amount_changed)
        self.row_sharp_amount.sliderReleased.connect(self.slider_released.emit)
        sharp_layout.addWidget(self.row_sharp_amount)

        self.row_sharp_radius = PrecisionSliderRow("Radio:", 5, 30, 10, scale=10.0, decimals=1, unit=" px", label_width=95)
        self.slider_adj_sharp_radius = self.row_sharp_radius.slider
        self.lbl_adj_sharp_radius = self.row_sharp_radius.lbl_value
        self.slider_adj_sharp_radius.valueChanged.connect(self._on_adj_sharp_radius_changed)
        self.row_sharp_radius.sliderReleased.connect(self.slider_released.emit)
        sharp_layout.addWidget(self.row_sharp_radius)

        self.row_sharp_detail = PrecisionSliderRow("Detalle:", 0, 100, 25, scale=1.0, decimals=0, unit="%", label_width=95)
        self.slider_adj_sharp_detail = self.row_sharp_detail.slider
        self.lbl_adj_sharp_detail = self.row_sharp_detail.lbl_value
        self.slider_adj_sharp_detail.valueChanged.connect(self._on_adj_sharp_detail_changed)
        self.row_sharp_detail.sliderReleased.connect(self.slider_released.emit)
        sharp_layout.addWidget(self.row_sharp_detail)

        self.btn_sharp_mask_preview = QPushButton()
        self.btn_sharp_mask_preview.setIcon(get_icon("eye", color="#94a3b8", active_color="#38bdf8", size=13))
        self.btn_sharp_mask_preview.setCheckable(True)
        self.btn_sharp_mask_preview.setFixedSize(20, 20)
        self.btn_sharp_mask_preview.setCursor(Qt.PointingHandCursor)
        self.btn_sharp_mask_preview.setToolTip("Ver máscara de bordes en blanco y negro (o mantén pulsado Alt mientras arrastras el deslizador)")
        self.btn_sharp_mask_preview.setStyleSheet("""
            QPushButton { background-color: #1e2430; border: 1px solid #334155; border-radius: 3px; padding: 1px; }
            QPushButton:hover { border-color: #38bdf8; background-color: #242c3d; }
            QPushButton:checked { background-color: #0284c7; border-color: #38bdf8; }
        """)
        self.btn_sharp_mask_preview.toggled.connect(self._on_sharp_mask_preview_toggled)

        self.row_sharp_masking = PrecisionSliderRow(
            "Máscara:", 0, 100, 0, scale=1.0, decimals=0, unit="%", label_width=95,
            action_button=self.btn_sharp_mask_preview
        )
        self.slider_adj_sharp_masking = self.row_sharp_masking.slider
        self.lbl_adj_sharp_masking = self.row_sharp_masking.lbl_value
        self.slider_adj_sharp_masking.valueChanged.connect(self._on_adj_sharp_masking_changed)
        self.slider_adj_sharp_masking.sliderPressed.connect(self._on_sharp_masking_slider_pressed)
        self.row_sharp_masking.sliderReleased.connect(self._on_sharp_masking_slider_released)
        sharp_layout.addWidget(self.row_sharp_masking)

        self.sec_layer_sharp.set_content_widget(w_sharp)
        adj_layout.addWidget(self.sec_layer_sharp)

        # Botón restablecer ajustes de la capa
        self.btn_reset_local = QPushButton("Restablecer Ajustes Locales")
        self.btn_reset_local.setIcon(get_icon("rotate_ccw", color="#94a3b8", size=13))
        self.btn_reset_local.setCursor(Qt.PointingHandCursor)
        self.btn_reset_local.clicked.connect(self._reset_local_adjustments)
        adj_layout.addWidget(self.btn_reset_local)

    def is_curve_expanded(self) -> bool:
        return bool(hasattr(self, 'sec_layer_curve') and self.sec_layer_curve.is_expanded)

    def apply_theme(self, border_col: str, title_col: str, bg_col: str, title: str) -> None:
        self.setStyleSheet(make_box_qss("local_adj_group", border_col, title_col, bg_col))
        self.setTitle(title)

    def update_for_layer(self, layer: Optional[MaskLayer]) -> None:
        if not layer:
            self.setVisible(False)
            return
        self.setVisible(True)
        self._updating_ui = True

        is_stars = (layer.mask_type == MASK_TYPE_IMAGE_STARS)
        is_starless = (layer.mask_type == MASK_TYPE_IMAGE_STARLESS)

        self.slider_adj_ev.setValue(int(round(getattr(layer, 'exposure', 0.0) * 100)))
        self.lbl_adj_ev.setText(f"{layer.exposure:+.2f} EV")

        self.slider_adj_contrast.setValue(int(round(getattr(layer, 'contrast', 0.0) * 100)))
        self.lbl_adj_contrast.setText(f"{layer.contrast:+.2f}")

        self.slider_adj_shadows.setValue(int(round(getattr(layer, 'shadows', 0.0) * 100)))
        self.lbl_adj_shadows.setText(f"{layer.shadows:+.2f}")

        self.slider_adj_highlights.setValue(int(round(getattr(layer, 'highlights', 0.0) * 100)))
        self.lbl_adj_highlights.setText(f"{layer.highlights:+.2f}")

        self.slider_adj_clarity.setValue(int(round(getattr(layer, 'clarity', 0.0) * 100)))
        self.lbl_adj_clarity.setText(f"{getattr(layer, 'clarity', 0.0):+.2f}")

        self.slider_adj_texture_clarity.setValue(int(round(getattr(layer, 'texture_clarity', 0.0) * 100)))
        self.lbl_adj_texture_clarity.setText(f"{getattr(layer, 'texture_clarity', 0.0):+.2f}")

        hide_adj_clarity = (is_stars or is_starless)
        self.row_adj_clarity.setVisible(not hide_adj_clarity)
        self.row_adj_texture_clarity.setVisible(not hide_adj_clarity)

        self.slider_adj_sat.setValue(int(round(getattr(layer, 'saturation', 1.0) * 100)))
        self.lbl_adj_sat.setText(f"{layer.saturation:.2f}x")

        self.row_adj_temp.setVisible(not is_stars)
        self.row_adj_tint.setVisible(not is_stars)
        if is_stars:
            layer.temp = 0.0
            layer.tint = 0.0

        self.slider_adj_temp.setValue(int(round(getattr(layer, 'temp', 0.0) * 10000)))
        self.spin_adj_temp.setValue(getattr(layer, 'temp', 0.0))
        self.lbl_adj_temp.setText(f"{layer.temp:+.4f}")

        self.slider_adj_tint.setValue(int(round(getattr(layer, 'tint', 0.0) * 10000)))
        self.spin_adj_tint.setValue(getattr(layer, 'tint', 0.0))
        self.lbl_adj_tint.setText(f"{layer.tint:+.4f}")

        self.sec_layer_sharp.setVisible(not is_stars)

        # Curva
        if hasattr(self, 'layer_curve_widget'):
            pts = getattr(layer, 'curve_points', None)
            if pts:
                self.layer_curve_widget.set_state({"rgb": pts})
            else:
                self.layer_curve_widget.reset_curve("rgb")

        # Nitidez
        amt = int(round(getattr(layer, 'sharp_amount', 0.0)))
        rad = int(round(getattr(layer, 'sharp_radius', 1.0) * 10))
        det = int(round(getattr(layer, 'sharp_detail', 25.0)))
        msq = int(round(getattr(layer, 'sharp_masking', 0.0)))

        self.slider_adj_sharp_amount.setValue(amt)
        self.lbl_adj_sharp_amount.setText(f"{amt}%")
        self.slider_adj_sharp_radius.setValue(rad)
        self.lbl_adj_sharp_radius.setText(f"{rad / 10.0:.1f} px")
        self.slider_adj_sharp_detail.setValue(det)
        self.lbl_adj_sharp_detail.setText(f"{det}%")
        self.slider_adj_sharp_masking.setValue(msq)
        self.lbl_adj_sharp_masking.setText(f"{msq}%")

        if self.btn_sharp_mask_preview.isChecked():
            self._emit_sharp_edge_mask_preview()

        self._updating_ui = False
        self.update_active_layer_histogram()

    def update_active_layer_histogram(self) -> None:
        if self._is_scrubbing:
            return
        if not hasattr(self, 'layer_curve_widget') or not self.is_curve_expanded():
            return
        layer = self.get_selected_layer()
        if not layer:
            self.layer_curve_widget.set_histogram_from_image(None)
            return

        if self._layer_image_provider is not None:
            img, mask, is_stars = self._layer_image_provider(layer)
            self.layer_curve_widget.set_histogram_from_image(img, mask=mask, is_stars_layer=is_stars)
            return

        img = None
        if layer.mask_type == MASK_TYPE_IMAGE_STARS and self._stars_getter:
            img = self._stars_getter()
        elif layer.mask_type == MASK_TYPE_IMAGE_STARLESS and self._starless_getter:
            img = self._starless_getter()
        elif layer.mask_type == MASK_TYPE_IMAGE_RASTER and getattr(layer, 'image_data', None) is not None:
            img = layer.image_data
            if img is not None and img.ndim == 2:
                img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        elif self._image_reference_getter:
            img = self._image_reference_getter()

        self.layer_curve_widget.set_histogram_from_image(img)

    def _on_adj_ev_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.exposure = val / 100.0
            self.lbl_adj_ev.setText(f"{layer.exposure:+.2f} EV")
            self.update_active_layer_histogram()
            self.adjustments_changed.emit()

    def _on_adj_contrast_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.contrast = val / 100.0
            self.lbl_adj_contrast.setText(f"{layer.contrast:+.2f}")
            self.update_active_layer_histogram()
            self.adjustments_changed.emit()

    def _on_adj_shadows_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.shadows = val / 100.0
            self.lbl_adj_shadows.setText(f"{layer.shadows:+.2f}")
            self.update_active_layer_histogram()
            self.adjustments_changed.emit()

    def _on_adj_highlights_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.highlights = val / 100.0
            self.lbl_adj_highlights.setText(f"{layer.highlights:+.2f}")
            self.update_active_layer_histogram()
            self.adjustments_changed.emit()

    def _on_adj_clarity_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.clarity = val / 100.0
            self.lbl_adj_clarity.setText(f"{layer.clarity:+.2f}")
            self.update_active_layer_histogram()
            self.adjustments_changed.emit()

    def _on_adj_texture_clarity_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.texture_clarity = val / 100.0
            self.lbl_adj_texture_clarity.setText(f"{layer.texture_clarity:+.2f}")
            self.update_active_layer_histogram()
            self.adjustments_changed.emit()

    def _on_adj_sat_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.saturation = val / 100.0
            self.lbl_adj_sat.setText(f"{layer.saturation:.2f}x")
            self.update_active_layer_histogram()
            self.adjustments_changed.emit()

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
            self.adjustments_changed.emit()

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
            self.adjustments_changed.emit()

    def _on_layer_curve_changed(self) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer and hasattr(self, 'layer_curve_widget'):
            layer.curve_points = [tuple(p) for p in self.layer_curve_widget.points]
            layer.curve_lut = self.layer_curve_widget.lut.copy()
            self.adjustments_changed.emit()

    def _reset_layer_curve(self) -> None:
        layer = self.get_selected_layer()
        if layer:
            layer.curve_points = [(0.0, 0.0), (1.0, 1.0)]
            layer.curve_lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            if hasattr(self, 'layer_curve_widget'):
                self._updating_ui = True
                self.layer_curve_widget.reset_curve("rgb")
                self._updating_ui = False
            self.adjustments_changed.emit()
            self.slider_released.emit()

    def _on_adj_sharp_amount_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.sharp_amount = float(val)
            self.lbl_adj_sharp_amount.setText(f"{val}%")
            self.adjustments_changed.emit()

    def _on_adj_sharp_radius_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.sharp_radius = val / 10.0
            self.lbl_adj_sharp_radius.setText(f"{layer.sharp_radius:.1f} px")
            self.adjustments_changed.emit()

    def _on_adj_sharp_detail_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.sharp_detail = float(val)
            self.lbl_adj_sharp_detail.setText(f"{val}%")
            self.adjustments_changed.emit()

    def _on_adj_sharp_masking_changed(self, val: int) -> None:
        if self._updating_ui: return
        layer = self.get_selected_layer()
        if layer:
            layer.sharp_masking = float(val)
            self.lbl_adj_sharp_masking.setText(f"{val}%")
            if self._alt_scrubbing_sharp or (hasattr(self, 'btn_sharp_mask_preview') and self.btn_sharp_mask_preview.isChecked()):
                self._emit_sharp_edge_mask_preview()
            else:
                self.adjustments_changed.emit()

    def _on_sharp_masking_slider_pressed(self) -> None:
        mods = QApplication.keyboardModifiers()
        if mods & Qt.AltModifier:
            self._alt_scrubbing_sharp = True
            self._emit_sharp_edge_mask_preview()
        self.interaction_started.emit()

    def _on_sharp_masking_slider_released(self) -> None:
        if self._alt_scrubbing_sharp:
            self._alt_scrubbing_sharp = False
            if not (hasattr(self, 'btn_sharp_mask_preview') and self.btn_sharp_mask_preview.isChecked()):
                self.preview_mask_requested.emit(None)
                self.adjustments_changed.emit()
        self.slider_released.emit()
        self.interaction_ended.emit()

    def _on_sharp_mask_preview_toggled(self, checked: bool) -> None:
        if checked:
            self._emit_sharp_edge_mask_preview()
        else:
            self.preview_mask_requested.emit(None)
            self.adjustments_changed.emit()

    def _emit_sharp_edge_mask_preview(self) -> None:
        layer = self.get_selected_layer()
        if not layer or not layer.enabled or not self._image_reference_getter:
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

        from core.sharpening import compute_edge_mask
        masking_val = float(getattr(layer, "sharp_masking", 0.0))

        if getattr(layer, 'is_raster_image', False) and getattr(layer, 'image_data', None) is not None:
            flt_img = layer.image_data
            if flt_img.shape[:2] != canvas_below.shape[:2]:
                flt_img = cv2.resize(flt_img, (canvas_below.shape[1], canvas_below.shape[0]), interpolation=cv2.INTER_LINEAR)
            edge_m = compute_edge_mask(flt_img, masking=masking_val, scope="global")
        else:
            scope = "ground" if layer.mask_type == MASK_TYPE_IMAGE_GROUND else "global"
            edge_m = compute_edge_mask(
                canvas_below,
                masking=masking_val,
                sky_mask=master_mask,
                scope=scope
            )

        if layer.mask_type not in (MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_GROUND, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS):
            eff_m = compute_effective_layer_mask(
                layer, canvas_below,
                master_sky_mask=master_mask,
                starless_image=starless_img,
                stars_image=stars_img
            )
            if eff_m is not None:
                if eff_m.shape[:2] != edge_m.shape[:2]:
                    eff_m = cv2.resize(eff_m, (edge_m.shape[1], edge_m.shape[0]), interpolation=cv2.INTER_LINEAR)
                edge_m = edge_m * eff_m

        self.preview_mask_requested.emit((edge_m, "bw"))

    def _reset_layer_sharpening(self) -> None:
        layer = self.get_selected_layer()
        if not layer: return
        layer.sharp_amount = 0.0
        layer.sharp_radius = 1.0
        layer.sharp_detail = 25.0
        layer.sharp_masking = 0.0
        self._updating_ui = True
        self.slider_adj_sharp_amount.setValue(0)
        self.lbl_adj_sharp_amount.setText("0%")
        self.slider_adj_sharp_radius.setValue(10)
        self.lbl_adj_sharp_radius.setText("1.0 px")
        self.slider_adj_sharp_detail.setValue(25)
        self.lbl_adj_sharp_detail.setText("25%")
        self.slider_adj_sharp_masking.setValue(0)
        self.lbl_adj_sharp_masking.setText("0%")
        if self.btn_sharp_mask_preview.isChecked():
            self.btn_sharp_mask_preview.setChecked(False)
            self.preview_mask_requested.emit(None)
        self._updating_ui = False
        self.slider_released.emit()
        self.adjustments_changed.emit()

    def _reset_local_adjustments(self) -> None:
        layer = self.get_selected_layer()
        if not layer: return
        layer.exposure = 0.0
        layer.contrast = 0.0
        layer.shadows = 0.0
        layer.highlights = 0.0
        layer.saturation = 1.0
        layer.temp = 0.0
        layer.tint = 0.0
        layer.clarity = 0.0
        layer.texture_clarity = 0.0
        layer.sharp_amount = 0.0
        layer.sharp_radius = 1.0
        layer.sharp_detail = 25.0
        layer.sharp_masking = 0.0
        layer.curve_points = [(0.0, 0.0), (1.0, 1.0)]
        layer.curve_lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        if hasattr(self, 'layer_curve_widget'):
            self._updating_ui = True
            self.layer_curve_widget.reset_curve("rgb")
            self._updating_ui = False
        self._updating_ui = True
        self.slider_adj_ev.setValue(0)
        self.lbl_adj_ev.setText("+0.00 EV")
        self.slider_adj_contrast.setValue(0)
        self.lbl_adj_contrast.setText("+0.00")
        self.slider_adj_shadows.setValue(0)
        self.lbl_adj_shadows.setText("+0.00")
        self.slider_adj_highlights.setValue(0)
        self.lbl_adj_highlights.setText("+0.00")
        self.slider_adj_clarity.setValue(0)
        self.lbl_adj_clarity.setText("+0.00")
        self.slider_adj_texture_clarity.setValue(0)
        self.lbl_adj_texture_clarity.setText("+0.00")
        self.slider_adj_sat.setValue(100)
        self.lbl_adj_sat.setText("1.00x")
        self.slider_adj_temp.setValue(0)
        self.spin_adj_temp.setValue(0.0)
        self.lbl_adj_temp.setText("+0.0000")
        self.slider_adj_tint.setValue(0)
        self.spin_adj_tint.setValue(0.0)
        self.lbl_adj_tint.setText("+0.0000")
        self.slider_adj_sharp_amount.setValue(0)
        self.lbl_adj_sharp_amount.setText("0%")
        self.slider_adj_sharp_radius.setValue(10)
        self.lbl_adj_sharp_radius.setText("1.0 px")
        self.slider_adj_sharp_detail.setValue(25)
        self.lbl_adj_sharp_detail.setText("25%")
        self.slider_adj_sharp_masking.setValue(0)
        self.lbl_adj_sharp_masking.setText("0%")
        self._updating_ui = False
        if self.btn_sharp_mask_preview.isChecked():
            self.btn_sharp_mask_preview.setChecked(False)
            self.preview_mask_requested.emit(None)
        self.update_active_layer_histogram()
        self.slider_released.emit()
        self.adjustments_changed.emit()
