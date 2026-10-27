"""Color and White Balance collapsible panels for AstroLoom Developer."""
from __future__ import annotations

from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PySide6.QtCore import Qt, Signal as pyqtSignal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.icons import get_icon


class ColorWbPanel(QWidget):
    """Encapsulates White Balance and Saturation / Vibrance controls."""

    # Signals for reactive pipeline updates
    parameter_changed = pyqtSignal()
    pipette_toggled = pyqtSignal(bool)
    action_committed = pyqtSignal()
    interaction_started = pyqtSignal()
    interaction_finished = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.temp_val: float = 0.0
        self.tint_val: float = 0.0
        self.sat_sky_val: float = 1.0
        self.sat_gnd_val: float = 1.0
        self.vibrance_val: float = 0.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- SECCIÓN: Balance de Blancos ---
        self.sec_wb = CollapsibleSection(
            "Balance de Blancos y Calibración",
            is_expanded=True,
            on_reset=self.reset_wb,
            icon_name="thermometer"
        )
        w_wb = QWidget()
        wb_layout = QVBoxLayout(w_wb)
        wb_layout.setContentsMargins(6, 4, 6, 6)
        wb_layout.setSpacing(8)

        # Sliders de alta precisión para Balance de Blancos
        self.row_temp = PrecisionSliderRow("Temperatura:", -1000, 1000, 0, scale=1000.0, decimals=3, force_sign=True, label_width=115)
        self.slider_temp = self.row_temp.slider
        self.spin_temp = self.row_temp.spin
        self.lbl_temp = self.row_temp.lbl_value
        self.slider_temp.valueChanged.connect(self._on_temp_changed)
        self.spin_temp.valueChanged.connect(self._on_spin_temp_changed)
        self.slider_temp.sliderPressed.connect(self.interaction_started.emit)
        self.slider_temp.sliderReleased.connect(self._on_slider_released)
        wb_layout.addWidget(self.row_temp)

        self.row_tint = PrecisionSliderRow("Tinte (V/M):", -1000, 1000, 0, scale=1000.0, decimals=3, force_sign=True, label_width=115)
        self.slider_tint = self.row_tint.slider
        self.spin_tint = self.row_tint.spin
        self.lbl_tint = self.row_tint.lbl_value
        self.slider_tint.valueChanged.connect(self._on_tint_changed)
        self.spin_tint.valueChanged.connect(self._on_spin_tint_changed)
        self.slider_tint.sliderPressed.connect(self.interaction_started.emit)
        self.slider_tint.sliderReleased.connect(self._on_slider_released)
        wb_layout.addWidget(self.row_tint)

        row_wb_btn = QHBoxLayout()
        self.btn_pipette = QPushButton("Pipeta WB (W)")
        self.btn_pipette.setIcon(get_icon("pipette", color="#80d8ff", size=13))
        self.btn_pipette.setCheckable(True)
        self.btn_pipette.setCursor(Qt.PointingHandCursor)
        self.btn_pipette.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_pipette.setToolTip("Activa la pipeta para calibrar el balance de blancos haciendo clic sobre un fondo neutro en la imagen (Atajo: W)")
        self.btn_pipette.toggled.connect(self._on_pipette_toggled)
        row_wb_btn.addWidget(self.btn_pipette)
        self.btn_panel_pipette = self.btn_pipette

        btn_reset_wb = QPushButton("Restablecer")
        btn_reset_wb.clicked.connect(self.reset_wb)
        row_wb_btn.addWidget(btn_reset_wb)
        wb_layout.addLayout(row_wb_btn)

        self.sec_wb.set_content_widget(w_wb)
        layout.addWidget(self.sec_wb)

        # --- SECCIÓN: Color y Saturación ---
        self.sec_color = CollapsibleSection(
            "Color y Saturación",
            is_expanded=True,
            on_reset=self.reset_saturation,
            icon_name="palette"
        )
        w_sat = QWidget()
        sat_layout = QVBoxLayout(w_sat)
        sat_layout.setContentsMargins(6, 4, 6, 6)
        sat_layout.setSpacing(8)

        # Sliders de alta precisión para Saturación y Vibranza
        self.row_sat_sky = PrecisionSliderRow("Sat. Cielo:", 0, 250, 100, scale=100.0, decimals=2, unit="x", label_width=115)
        self.slider_sat_sky = self.row_sat_sky.slider
        self.lbl_sat_sky = self.row_sat_sky.lbl_value
        self.slider_sat_sky.valueChanged.connect(self._on_sat_sky_changed)
        self.slider_sat_sky.sliderPressed.connect(self.interaction_started.emit)
        self.slider_sat_sky.sliderReleased.connect(self._on_slider_released)
        sat_layout.addWidget(self.row_sat_sky)

        self.row_sat_gnd = PrecisionSliderRow("Sat. Suelo:", 0, 250, 100, scale=100.0, decimals=2, unit="x", label_width=115)
        self.slider_sat_gnd = self.row_sat_gnd.slider
        self.lbl_sat_gnd = self.row_sat_gnd.lbl_value
        self.slider_sat_gnd.valueChanged.connect(self._on_sat_gnd_changed)
        self.slider_sat_gnd.sliderPressed.connect(self.interaction_started.emit)
        self.slider_sat_gnd.sliderReleased.connect(self._on_slider_released)
        sat_layout.addWidget(self.row_sat_gnd)

        self.row_vibrance = PrecisionSliderRow("Vibranza:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_vibrance = self.row_vibrance.slider
        self.lbl_vibrance = self.row_vibrance.lbl_value
        self.slider_vibrance.valueChanged.connect(self._on_vibrance_changed)
        self.slider_vibrance.sliderPressed.connect(self.interaction_started.emit)
        self.slider_vibrance.sliderReleased.connect(self._on_slider_released)

        btn_reset_sat = QPushButton("Restablecer")
        btn_reset_sat.setToolTip("Restablecer Saturación y Vibranza")
        btn_reset_sat.clicked.connect(self.reset_saturation)
        self.row_vibrance.layout().addWidget(btn_reset_sat)
        sat_layout.addWidget(self.row_vibrance)

        self.sec_color.set_content_widget(w_sat)
        layout.addWidget(self.sec_color)

    # -------------------------------------------------------------
    # Sliders & Value synchronization
    # -------------------------------------------------------------
    def _on_slider_released(self) -> None:
        self.interaction_finished.emit()
        self.action_committed.emit()

    def _on_temp_changed(self, val: int) -> None:
        new_val = val / 1000.0
        if abs(self.temp_val - new_val) < 1e-6:
            return
        self.temp_val = new_val
        self.parameter_changed.emit()

    def _on_spin_temp_changed(self, val: float) -> None:
        new_val = float(val)
        if abs(self.temp_val - new_val) < 1e-6:
            return
        self.temp_val = new_val
        self.parameter_changed.emit()

    def _on_tint_changed(self, val: int) -> None:
        new_val = val / 1000.0
        if abs(self.tint_val - new_val) < 1e-6:
            return
        self.tint_val = new_val
        self.parameter_changed.emit()

    def _on_spin_tint_changed(self, val: float) -> None:
        new_val = float(val)
        if abs(self.tint_val - new_val) < 1e-6:
            return
        self.tint_val = new_val
        self.parameter_changed.emit()

    def _on_sat_sky_changed(self, val: int) -> None:
        new_val = val / 100.0
        if abs(self.sat_sky_val - new_val) < 1e-6:
            return
        self.sat_sky_val = new_val
        self.lbl_sat_sky.setText(f"{self.sat_sky_val:.2f}x")
        self.parameter_changed.emit()

    def _on_sat_gnd_changed(self, val: int) -> None:
        new_val = val / 100.0
        if abs(self.sat_gnd_val - new_val) < 1e-6:
            return
        self.sat_gnd_val = new_val
        self.lbl_sat_gnd.setText(f"{self.sat_gnd_val:.2f}x")
        self.parameter_changed.emit()

    def _on_vibrance_changed(self, val: int) -> None:
        new_val = val / 100.0
        if abs(self.vibrance_val - new_val) < 1e-6:
            return
        self.vibrance_val = new_val
        self.lbl_vibrance.setText(f"{self.vibrance_val:+.2f}")
        self.parameter_changed.emit()

    def _on_pipette_toggled(self, checked: bool) -> None:
        self.pipette_toggled.emit(checked)

    # -------------------------------------------------------------
    # Public controls API
    # -------------------------------------------------------------
    def reset_wb(self) -> None:
        self.slider_temp.blockSignals(True)
        self.slider_tint.blockSignals(True)
        self.spin_temp.blockSignals(True)
        self.spin_tint.blockSignals(True)

        self.spin_temp.setValue(0.0)
        self.spin_tint.setValue(0.0)
        self.slider_temp.setValue(0)
        self.slider_tint.setValue(0)
        self.temp_val = 0.0
        self.tint_val = 0.0

        self.spin_temp.blockSignals(False)
        self.spin_tint.blockSignals(False)
        self.slider_temp.blockSignals(False)
        self.slider_tint.blockSignals(False)

        self.parameter_changed.emit()
        self.action_committed.emit()

    def reset_saturation(self) -> None:
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
        self.lbl_vibrance.setText("+0.00")

        self.slider_sat_sky.blockSignals(False)
        self.slider_sat_gnd.blockSignals(False)
        self.slider_vibrance.blockSignals(False)

        self.parameter_changed.emit()
        self.action_committed.emit()

    def reset_all(self) -> None:
        self.reset_wb()
        self.reset_saturation()

    def get_state(self) -> dict:
        return {
            "temp_val": self.slider_temp.value(),
            "tint_val": self.slider_tint.value(),
            "sat_sky_val": self.slider_sat_sky.value(),
            "sat_gnd_val": self.slider_sat_gnd.value(),
            "vibrance_val": self.slider_vibrance.value(),
        }

    def set_state(self, state: dict) -> None:
        self.block_signals(True)
        try:
            if "temp_val" in state:
                val = state["temp_val"]
                self.slider_temp.setValue(val)
                self.temp_val = val / 1000.0
                self.spin_temp.setValue(self.temp_val)
            if "tint_val" in state:
                val = state["tint_val"]
                self.slider_tint.setValue(val)
                self.tint_val = val / 1000.0
                self.spin_tint.setValue(self.tint_val)
            if "sat_sky_val" in state:
                val = state["sat_sky_val"]
                self.slider_sat_sky.setValue(val)
                self.sat_sky_val = val / 100.0
                self.lbl_sat_sky.setText(f"{self.sat_sky_val:.2f}x")
            if "sat_gnd_val" in state:
                val = state["sat_gnd_val"]
                self.slider_sat_gnd.setValue(val)
                self.sat_gnd_val = val / 100.0
                self.lbl_sat_gnd.setText(f"{self.sat_gnd_val:.2f}x")
            if "vibrance_val" in state:
                val = state["vibrance_val"]
                self.slider_vibrance.setValue(val)
                self.vibrance_val = val / 100.0
                self.lbl_vibrance.setText(f"{self.vibrance_val:+.2f}")
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.slider_temp, self.spin_temp,
            self.slider_tint, self.spin_tint,
            self.slider_sat_sky, self.slider_sat_gnd,
            self.slider_vibrance,
        ):
            w.blockSignals(blocked)
