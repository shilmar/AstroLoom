"""Color, White Balance, Photometric Calibration and Dual-Band collapsible panels for AstroLoom Developer."""
from __future__ import annotations

from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
)
from PySide6.QtCore import Qt, Signal as pyqtSignal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.icons import get_icon


class ColorWbPanel(QWidget):
    """Encapsulates White Balance, PCC, Saturation, and Dual-Band Narrowband controls."""

    # Signals for reactive pipeline updates
    parameter_changed = pyqtSignal()
    pipette_toggled = pyqtSignal(bool)
    pcc_requested = pyqtSignal()
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

        # Dual-band & Stellar correction
        self.dualband_mode: str = "none"
        self.dualband_oiii_boost: int = 100
        self.unclip_stars_val: int = 0
        self.unpurple_val: int = 0
        self.pcc_kr: float = 1.0
        self.pcc_kb: float = 1.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- SECCIÓN: Balance de Blancos y Calibración ---
        self.sec_wb = CollapsibleSection(
            "2. Balance de Blancos y Calibración",
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

        # Fila Calibración Fotométrica del Color (PCC Real)
        row_pcc = QHBoxLayout()
        self.btn_pcc = QPushButton("Calibración PCC (Gaia)")
        self.btn_pcc.setIcon(get_icon("star", color="#ffd54f", size=13))
        self.btn_pcc.setCursor(Qt.PointingHandCursor)
        self.btn_pcc.setToolTip("Calibra el color físico exacto de las estrellas contra el estándar solar Gaia G2V (BP - RP = 0.82)")
        self.btn_pcc.setStyleSheet("font-weight: bold; color: #ffd54f; margin-top: 2px;")
        self.btn_pcc.clicked.connect(self._on_pcc_clicked)

        self.btn_reset_pcc = QPushButton("Restablecer")
        self.btn_reset_pcc.setToolTip("Restablecer ganancias fotométricas a 1.000")
        self.btn_reset_pcc.clicked.connect(self.reset_pcc)

        row_pcc.addWidget(self.btn_pcc, stretch=3)
        row_pcc.addWidget(self.btn_reset_pcc, stretch=1)
        wb_layout.addLayout(row_pcc)

        self.lbl_pcc_status = QLabel("PCC: Inactivo (Ganancias 1.000)")
        self.lbl_pcc_status.setStyleSheet("color: #888888; font-size: 11px;")
        wb_layout.addWidget(self.lbl_pcc_status)

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

        # --- SECCIÓN: Paletas Dual-Band OSC y Reparación Estelar (Fase 1) ---
        self.sec_dualband = CollapsibleSection(
            "Paletas Dual-Band y Corrección Estelar",
            is_expanded=False,
            on_reset=self.reset_dualband,
            icon_name="sparkles"
        )
        w_db = QWidget()
        db_layout = QVBoxLayout(w_db)
        db_layout.setContentsMargins(6, 4, 6, 6)
        db_layout.setSpacing(8)

        row_pal = QHBoxLayout()
        lbl_pal = QLabel("Paleta OSC:")
        lbl_pal.setFixedWidth(115)
        self.combo_dualband = QComboBox()
        self.combo_dualband.addItems([
            "Desactivado (RGB)",
            "HOO Bicolor",
            "SHO Hubble Sintético",
            "Foraxx Dinámico",
            "Cannistra Bicolor"
        ])
        self.combo_dualband.currentIndexChanged.connect(self._on_dualband_combo_changed)
        row_pal.addWidget(lbl_pal)
        row_pal.addWidget(self.combo_dualband)
        db_layout.addLayout(row_pal)

        lbl_db_hint = QLabel(
            "Uso exclusivo para tomas capturadas con filtros físicos de doble banda estrecha (Hα/[OIII]).\n"
            "Para fotografía de paisaje y Vía Láctea RGB estándar, mantener en 'Desactivado (RGB)'."
        )
        lbl_db_hint.setWordWrap(True)
        lbl_db_hint.setStyleSheet("color: #78909c; font-size: 10px; line-height: 1.3;")
        db_layout.addWidget(lbl_db_hint)

        self.row_oiii_boost = PrecisionSliderRow("Boost [OIII]:", 10, 300, 100, scale=100.0, decimals=2, unit="x", label_width=115)
        self.slider_oiii_boost = self.row_oiii_boost.slider
        self.lbl_oiii_boost = self.row_oiii_boost.lbl_value
        self.slider_oiii_boost.valueChanged.connect(self._on_oiii_boost_changed)
        self.slider_oiii_boost.sliderPressed.connect(self.interaction_started.emit)
        self.slider_oiii_boost.sliderReleased.connect(self._on_slider_released)
        db_layout.addWidget(self.row_oiii_boost)

        self.row_unclip = PrecisionSliderRow("Rescate Núcleo:", 0, 100, 0, scale=1.0, decimals=0, unit="%", label_width=115)
        self.slider_unclip = self.row_unclip.slider
        self.lbl_unclip = self.row_unclip.lbl_value
        self.slider_unclip.valueChanged.connect(self._on_unclip_changed)
        self.slider_unclip.sliderPressed.connect(self.interaction_started.emit)
        self.slider_unclip.sliderReleased.connect(self._on_slider_released)
        db_layout.addWidget(self.row_unclip)

        self.row_unpurple = PrecisionSliderRow("Atenuar Magenta:", 0, 100, 0, scale=1.0, decimals=0, unit="%", label_width=115)
        self.slider_unpurple = self.row_unpurple.slider
        self.lbl_unpurple = self.row_unpurple.lbl_value
        self.slider_unpurple.valueChanged.connect(self._on_unpurple_changed)
        self.slider_unpurple.sliderPressed.connect(self.interaction_started.emit)
        self.slider_unpurple.sliderReleased.connect(self._on_slider_released)
        db_layout.addWidget(self.row_unpurple)

        self.sec_dualband.set_content_widget(w_db)
        layout.addWidget(self.sec_dualband)

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

    def _on_pcc_clicked(self) -> None:
        self.pcc_requested.emit()

    def set_pcc_gains(self, kr: float, kb: float) -> None:
        self.pcc_kr = float(kr)
        self.pcc_kb = float(kb)
        if hasattr(self, 'lbl_pcc_status'):
            if abs(self.pcc_kr - 1.0) > 1e-4 or abs(self.pcc_kb - 1.0) > 1e-4:
                self.lbl_pcc_status.setText(f"✓ PCC Activo: R={self.pcc_kr:.3f}, B={self.pcc_kb:.3f}")
                self.lbl_pcc_status.setStyleSheet("color: #ffd54f; font-size: 11px; font-weight: bold;")
            else:
                self.lbl_pcc_status.setText("PCC: Inactivo (Ganancias 1.000)")
                self.lbl_pcc_status.setStyleSheet("color: #888888; font-size: 11px;")
        self.parameter_changed.emit()

    def reset_pcc(self) -> None:
        self.set_pcc_gains(1.0, 1.0)
        self.action_committed.emit()

    def _on_dualband_combo_changed(self, idx: int) -> None:
        modes = ["none", "hoo", "sho", "foraxx", "cannistra"]
        self.dualband_mode = modes[idx] if 0 <= idx < len(modes) else "none"
        self.parameter_changed.emit()
        self.action_committed.emit()

    def _on_oiii_boost_changed(self, val: int) -> None:
        self.dualband_oiii_boost = val
        self.lbl_oiii_boost.setText(f"{val / 100.0:.2f}x")
        self.parameter_changed.emit()

    def _on_unclip_changed(self, val: int) -> None:
        self.unclip_stars_val = val
        self.lbl_unclip.setText(f"{val}%")
        self.parameter_changed.emit()

    def _on_unpurple_changed(self, val: int) -> None:
        self.unpurple_val = val
        self.lbl_unpurple.setText(f"{val}%")
        self.parameter_changed.emit()

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
        self.set_pcc_gains(1.0, 1.0)

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

    def reset_dualband(self) -> None:
        self.combo_dualband.blockSignals(True)
        self.slider_oiii_boost.blockSignals(True)
        self.slider_unclip.blockSignals(True)
        self.slider_unpurple.blockSignals(True)

        self.combo_dualband.setCurrentIndex(0)
        self.slider_oiii_boost.setValue(100)
        self.slider_unclip.setValue(0)
        self.slider_unpurple.setValue(0)
        self.dualband_mode = "none"
        self.dualband_oiii_boost = 100
        self.unclip_stars_val = 0
        self.unpurple_val = 0
        self.lbl_oiii_boost.setText("1.00x")
        self.lbl_unclip.setText("0%")
        self.lbl_unpurple.setText("0%")

        self.combo_dualband.blockSignals(False)
        self.slider_oiii_boost.blockSignals(False)
        self.slider_unclip.blockSignals(False)
        self.slider_unpurple.blockSignals(False)

        self.parameter_changed.emit()
        self.action_committed.emit()

    def reset_all(self) -> None:
        self.reset_wb()
        self.reset_saturation()
        self.reset_dualband()

    def get_state(self) -> dict:
        return {
            "temp_val": self.slider_temp.value(),
            "tint_val": self.slider_tint.value(),
            "sat_sky_val": self.slider_sat_sky.value(),
            "sat_gnd_val": self.slider_sat_gnd.value(),
            "vibrance_val": self.slider_vibrance.value(),
            "dualband_mode": self.dualband_mode,
            "dualband_oiii_boost": self.slider_oiii_boost.value(),
            "unclip_stars_val": self.slider_unclip.value(),
            "unpurple_val": self.slider_unpurple.value(),
            "pcc_kr": self.pcc_kr,
            "pcc_kb": self.pcc_kb,
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
            if "dualband_mode" in state:
                mode_str = str(state["dualband_mode"]).lower()
                modes = ["none", "hoo", "sho", "foraxx", "cannistra"]
                idx = modes.index(mode_str) if mode_str in modes else 0
                self.combo_dualband.setCurrentIndex(idx)
                self.dualband_mode = modes[idx]
            if "dualband_oiii_boost" in state:
                val = state["dualband_oiii_boost"]
                self.slider_oiii_boost.setValue(val)
                self.dualband_oiii_boost = val
                self.lbl_oiii_boost.setText(f"{val / 100.0:.2f}x")
            if "unclip_stars_val" in state:
                val = state["unclip_stars_val"]
                self.slider_unclip.setValue(val)
                self.unclip_stars_val = val
                self.lbl_unclip.setText(f"{val}%")
            if "unpurple_val" in state:
                val = state["unpurple_val"]
                self.slider_unpurple.setValue(val)
                self.unpurple_val = val
                self.lbl_unpurple.setText(f"{val}%")
            if "pcc_kr" in state or "pcc_kb" in state:
                kr = float(state.get("pcc_kr", 1.0))
                kb = float(state.get("pcc_kb", 1.0))
                self.set_pcc_gains(kr, kb)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.slider_temp, self.spin_temp,
            self.slider_tint, self.spin_tint,
            self.slider_sat_sky, self.slider_sat_gnd,
            self.slider_vibrance,
            self.combo_dualband, self.slider_oiii_boost,
            self.slider_unclip, self.slider_unpurple
        ):
            w.blockSignals(blocked)
