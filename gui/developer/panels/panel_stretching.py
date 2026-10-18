"""Dynamic range stretching (MTF / GHS) panel for AstroLoom Developer."""
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QButtonGroup
)
from PySide6.QtCore import Qt, Signal as pyqtSignal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.icons import get_icon


class StretchingPanel(QWidget):
    """Encapsulates Midtone Transfer Function (MTF) and Generalized Hyperbolic Stretch (GHS) controls."""

    parameter_changed = pyqtSignal()
    action_committed = pyqtSignal()
    auto_mtf_requested = pyqtSignal()
    auto_ghs_requested = pyqtSignal()
    ghs_pipette_toggled = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.stretch_mode: str = "mtf"
        self.ghs_color_mode: str = "luma"
        self.bp_val: float = 0.0
        self.mtf_val: float = 0.15
        self.contrast_val: float = 0.0

        self.ghs_sp_val: float = 0.02
        self.ghs_d_val: float = 5.0
        self.ghs_b_val: float = 2.0
        self.ghs_hp_val: float = 0.0
        self.ghs_lp_val: float = 0.0
        self.ghs_bp_val: float = 0.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- SECCIÓN: Estirado Tonal y Contraste (MTF / GHS) ---
        self.sec_stretch = CollapsibleSection(
            "Estirado Tonal y Contraste",
            is_expanded=True,
            on_reset=self.reset_sliders,
            icon_name="chart"
        )
        w_stretch = QWidget()
        stretch_layout = QVBoxLayout(w_stretch)
        stretch_layout.setContentsMargins(6, 4, 6, 6)
        stretch_layout.setSpacing(8)

        # 1. Selector de Modo de Estirado: MTF vs GHS
        row_stretch_mode = QHBoxLayout()
        row_stretch_mode.setSpacing(4)

        self.btn_mode_mtf = QPushButton("MTF Estándar")
        self.btn_mode_mtf.setIcon(get_icon("sliders", color="#80d8ff", size=12))
        self.btn_mode_mtf.setCheckable(True)
        self.btn_mode_mtf.setChecked(True)
        self.btn_mode_mtf.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #80d8ff; border: 1px solid #80d8ff; }"
        )
        self.btn_mode_mtf.setToolTip("Estirado logarítmico rápido mediante Función de Transferencia de Medios Tonos (MTF)")
        self.btn_mode_mtf.clicked.connect(lambda: self.set_stretch_mode("mtf"))

        self.btn_mode_ghs = QPushButton("Hiperbólico (GHS)")
        self.btn_mode_ghs.setIcon(get_icon("chart", color="#ffb74d", size=12))
        self.btn_mode_ghs.setCheckable(True)
        self.btn_mode_ghs.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #ffb74d; border: 1px solid #ffb74d; }"
        )
        self.btn_mode_ghs.setToolTip("Estirado hiperbólico avanzado (GHS): máximo contraste en nebulosas protegiendo estrellas")
        self.btn_mode_ghs.clicked.connect(lambda: self.set_stretch_mode("ghs"))

        self.grp_stretch_mode = QButtonGroup(self)
        self.grp_stretch_mode.setExclusive(True)
        self.grp_stretch_mode.addButton(self.btn_mode_mtf)
        self.grp_stretch_mode.addButton(self.btn_mode_ghs)

        row_stretch_mode.addWidget(self.btn_mode_mtf)
        row_stretch_mode.addWidget(self.btn_mode_ghs)
        stretch_layout.addLayout(row_stretch_mode)

        # 2. Contenedor MTF
        self.w_mtf_container = QWidget()
        mtf_layout = QVBoxLayout(self.w_mtf_container)
        mtf_layout.setContentsMargins(0, 0, 0, 0)
        mtf_layout.setSpacing(6)

        btn_row_mtf = QHBoxLayout()
        btn_auto_mtf = QPushButton("Auto-Estirado MTF")
        btn_auto_mtf.setIcon(get_icon("sparkles", color="#80d8ff", size=12))
        btn_auto_mtf.setStyleSheet("font-weight: bold; color: #80d8ff;")
        btn_auto_mtf.clicked.connect(self.auto_mtf_requested.emit)
        btn_reset_tone = QPushButton("Restablecer MTF")
        btn_reset_tone.clicked.connect(self.reset_sliders)
        btn_row_mtf.addWidget(btn_auto_mtf)
        btn_row_mtf.addWidget(btn_reset_tone)
        mtf_layout.addLayout(btn_row_mtf)

        self.row_bp = PrecisionSliderRow("P. Negro:", 0, 2000, 0, scale=10000.0, decimals=5, label_width=115)
        self.slider_bp = self.row_bp.slider
        self.spin_bp = self.row_bp.spin
        self.spin_bp.setRange(0.0, 0.20000)
        self.spin_bp.setSingleStep(0.00010)
        self.slider_bp.valueChanged.connect(self._on_slider_bp_changed)
        self.spin_bp.valueChanged.connect(self._on_spin_bp_changed)
        self.slider_bp.sliderReleased.connect(self.action_committed.emit)
        mtf_layout.addWidget(self.row_bp)

        self.row_mtf = PrecisionSliderRow("Medios (m):", 1, 2000, 400, scale=4000.0, decimals=5, label_width=115)
        self.slider_mtf = self.row_mtf.slider
        self.spin_mtf = self.row_mtf.spin
        self.spin_mtf.setRange(0.00010, 0.50000)
        self.spin_mtf.setSingleStep(0.00050)
        self.slider_mtf.valueChanged.connect(self._on_slider_mtf_changed)
        self.spin_mtf.valueChanged.connect(self._on_spin_mtf_changed)
        self.slider_mtf.sliderReleased.connect(self.action_committed.emit)
        mtf_layout.addWidget(self.row_mtf)

        stretch_layout.addWidget(self.w_mtf_container)

        # 3. Contenedor GHS
        self.w_ghs_container = QWidget()
        self.w_ghs_container.setVisible(False)
        ghs_layout = QVBoxLayout(self.w_ghs_container)
        ghs_layout.setContentsMargins(0, 0, 0, 0)
        ghs_layout.setSpacing(6)

        btn_row_ghs = QHBoxLayout()
        self.btn_auto_ghs = QPushButton("Auto-GHS")
        self.btn_auto_ghs.setIcon(get_icon("sparkles", color="#ffb74d", size=12))
        self.btn_auto_ghs.setStyleSheet("font-weight: bold; color: #ffb74d;")
        self.btn_auto_ghs.clicked.connect(self.auto_ghs_requested.emit)
        self.btn_auto_ghs.setToolTip("Calcula y aplica automáticamente los parámetros óptimos de GHS según el histograma")

        self.btn_ghs_pipette = QPushButton("Cuentagotas SP")
        self.btn_ghs_pipette.setIcon(get_icon("pipette", color="#80d8ff", size=12))
        self.btn_ghs_pipette.setCheckable(True)
        self.btn_ghs_pipette.setCursor(Qt.PointingHandCursor)
        self.btn_ghs_pipette.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_ghs_pipette.setToolTip("Haz clic aquí y luego pincha en el fondo de cielo o estructura del lienzo para centrar el Punto de Simetría (SP)")
        self.btn_ghs_pipette.toggled.connect(self.ghs_pipette_toggled.emit)

        self.btn_reset_ghs = QPushButton("Restablecer")
        self.btn_reset_ghs.clicked.connect(self.reset_ghs)

        btn_row_ghs.addWidget(self.btn_auto_ghs)
        btn_row_ghs.addWidget(self.btn_ghs_pipette)
        btn_row_ghs.addWidget(self.btn_reset_ghs)
        ghs_layout.addLayout(btn_row_ghs)

        # Selector de Modo de Color GHS
        row_ghs_color = QHBoxLayout()
        row_ghs_color.setSpacing(4)
        self.btn_ghs_luma = QPushButton("Luminancia (Color)")
        self.btn_ghs_luma.setIcon(get_icon("chart", color="#80d8ff", size=11))
        self.btn_ghs_luma.setCheckable(True)
        self.btn_ghs_luma.setChecked(True)
        self.btn_ghs_luma.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px; font-weight: bold; border-radius: 3px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #80d8ff; border: 1px solid #80d8ff; }"
        )
        self.btn_ghs_luma.setToolTip("Preserva la saturación natural y matices estirando solo la luminancia Rec.709")
        self.btn_ghs_luma.clicked.connect(lambda: self.set_ghs_color_mode("luma"))

        self.btn_ghs_rgb = QPushButton("RGB Enlazado")
        self.btn_ghs_rgb.setIcon(get_icon("palette", color="#00e676", size=11))
        self.btn_ghs_rgb.setCheckable(True)
        self.btn_ghs_rgb.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px; font-weight: bold; border-radius: 3px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #00e676; border: 1px solid #00e676; }"
        )
        self.btn_ghs_rgb.setToolTip("Aplica la curva GHS uniformemente sobre los canales R, G y B")
        self.btn_ghs_rgb.clicked.connect(lambda: self.set_ghs_color_mode("rgb"))

        self.grp_ghs_color = QButtonGroup(self)
        self.grp_ghs_color.setExclusive(True)
        self.grp_ghs_color.addButton(self.btn_ghs_luma)
        self.grp_ghs_color.addButton(self.btn_ghs_rgb)
        row_ghs_color.addWidget(self.btn_ghs_luma)
        row_ghs_color.addWidget(self.btn_ghs_rgb)
        ghs_layout.addLayout(row_ghs_color)

        # Fila SP: Punto de Simetría
        self.row_ghs_sp = PrecisionSliderRow("P. Simetría (SP):", 0, 10000, 200, scale=10000.0, decimals=5, label_width=115)
        self.slider_ghs_sp = self.row_ghs_sp.slider
        self.spin_ghs_sp = self.row_ghs_sp.spin
        self.lbl_ghs_sp = self.row_ghs_sp.lbl_value
        self.slider_ghs_sp.valueChanged.connect(self._on_slider_ghs_sp_changed)
        self.spin_ghs_sp.valueChanged.connect(self._on_spin_ghs_sp_changed)
        self.slider_ghs_sp.sliderReleased.connect(self.action_committed.emit)
        ghs_layout.addWidget(self.row_ghs_sp)

        # Fila Factor D
        self.row_ghs_d = PrecisionSliderRow("Factor D:", 0, 2000, 500, scale=100.0, decimals=2, label_width=115)
        self.slider_ghs_d = self.row_ghs_d.slider
        self.spin_ghs_d = self.row_ghs_d.spin
        self.lbl_ghs_d = self.row_ghs_d.lbl_value
        self.slider_ghs_d.valueChanged.connect(self._on_slider_ghs_d_changed)
        self.spin_ghs_d.valueChanged.connect(self._on_spin_ghs_d_changed)
        self.slider_ghs_d.sliderReleased.connect(self.action_committed.emit)
        ghs_layout.addWidget(self.row_ghs_d)

        # Fila Forma (b)
        self.row_ghs_b = PrecisionSliderRow("Forma (b):", 0, 1500, 200, scale=100.0, decimals=2, label_width=115)
        self.slider_ghs_b = self.row_ghs_b.slider
        self.spin_ghs_b = self.row_ghs_b.spin
        self.lbl_ghs_b = self.row_ghs_b.lbl_value
        self.slider_ghs_b.valueChanged.connect(self._on_slider_ghs_b_changed)
        self.spin_ghs_b.valueChanged.connect(self._on_spin_ghs_b_changed)
        self.slider_ghs_b.sliderReleased.connect(self.action_committed.emit)
        ghs_layout.addWidget(self.row_ghs_b)

        # Fila Prot. Estrellas (HP)
        self.row_ghs_hp = PrecisionSliderRow("Prot. Estrellas:", 0, 1000, 0, scale=1000.0, decimals=2, label_width=115)
        self.slider_ghs_hp = self.row_ghs_hp.slider
        self.spin_ghs_hp = self.row_ghs_hp.spin
        self.lbl_ghs_hp = self.row_ghs_hp.lbl_value
        self.slider_ghs_hp.valueChanged.connect(self._on_slider_ghs_hp_changed)
        self.spin_ghs_hp.valueChanged.connect(self._on_spin_ghs_hp_changed)
        self.slider_ghs_hp.sliderReleased.connect(self.action_committed.emit)
        ghs_layout.addWidget(self.row_ghs_hp)

        # Fila Prot. Sombras (LP)
        self.row_ghs_lp = PrecisionSliderRow("Prot. Sombras:", 0, 1000, 0, scale=1000.0, decimals=2, label_width=115)
        self.slider_ghs_lp = self.row_ghs_lp.slider
        self.spin_ghs_lp = self.row_ghs_lp.spin
        self.lbl_ghs_lp = self.row_ghs_lp.lbl_value
        self.slider_ghs_lp.valueChanged.connect(self._on_slider_ghs_lp_changed)
        self.spin_ghs_lp.valueChanged.connect(self._on_spin_ghs_lp_changed)
        self.slider_ghs_lp.sliderReleased.connect(self.action_committed.emit)
        ghs_layout.addWidget(self.row_ghs_lp)

        # Fila Punto Negro GHS (BP)
        self.row_ghs_bp = PrecisionSliderRow("P. Negro GHS:", 0, 2000, 0, scale=10000.0, decimals=5, label_width=115)
        self.slider_ghs_bp = self.row_ghs_bp.slider
        self.spin_ghs_bp = self.row_ghs_bp.spin
        self.lbl_ghs_bp = self.row_ghs_bp.lbl_value
        self.slider_ghs_bp.valueChanged.connect(self._on_slider_ghs_bp_changed)
        self.spin_ghs_bp.valueChanged.connect(self._on_spin_ghs_bp_changed)
        self.slider_ghs_bp.sliderReleased.connect(self.action_committed.emit)
        ghs_layout.addWidget(self.row_ghs_bp)

        stretch_layout.addWidget(self.w_ghs_container)

        # 4. Contraste General (común para MTF y GHS)
        self.row_contrast = PrecisionSliderRow("Contraste:", -100, 100, 0, scale=100.0, decimals=2, force_sign=True, label_width=115)
        self.slider_contrast = self.row_contrast.slider
        self.lbl_contrast = self.row_contrast.lbl_value
        self.slider_contrast.valueChanged.connect(self._on_contrast_changed)
        self.slider_contrast.sliderReleased.connect(self.action_committed.emit)
        stretch_layout.addWidget(self.row_contrast)

        self.sec_stretch.set_content_widget(w_stretch)
        layout.addWidget(self.sec_stretch)

    # -------------------------------------------------------------
    # Sliders & Value synchronization
    # -------------------------------------------------------------
    def _on_slider_bp_changed(self, val: int) -> None:
        bp = val / 10000.0
        if abs(self.bp_val - bp) < 1e-6:
            return
        self.bp_val = bp
        self.spin_bp.blockSignals(True)
        self.spin_bp.setValue(bp)
        self.spin_bp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_bp_changed(self, val: float) -> None:
        bp = float(val)
        if abs(self.bp_val - bp) < 1e-6:
            return
        self.bp_val = bp
        self.slider_bp.blockSignals(True)
        self.slider_bp.setValue(int(round(bp * 10000.0)))
        self.slider_bp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_mtf_changed(self, val: int) -> None:
        m = val / 4000.0
        if abs(self.mtf_val - m) < 1e-6:
            return
        self.mtf_val = m
        self.spin_mtf.blockSignals(True)
        self.spin_mtf.setValue(m)
        self.spin_mtf.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_mtf_changed(self, val: float) -> None:
        m = float(val)
        if abs(self.mtf_val - m) < 1e-6:
            return
        self.mtf_val = m
        self.slider_mtf.blockSignals(True)
        self.slider_mtf.setValue(int(round(m * 4000.0)))
        self.slider_mtf.blockSignals(False)
        self.parameter_changed.emit()

    def _on_contrast_changed(self, val: int) -> None:
        c = val / 100.0
        if abs(self.contrast_val - c) < 1e-4:
            return
        self.contrast_val = c
        self.parameter_changed.emit()

    def _on_slider_ghs_sp_changed(self, val: int) -> None:
        sp = val / 10000.0
        if abs(self.ghs_sp_val - sp) < 1e-6:
            return
        self.ghs_sp_val = sp
        self.spin_ghs_sp.blockSignals(True)
        self.spin_ghs_sp.setValue(sp)
        self.spin_ghs_sp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_ghs_sp_changed(self, val: float) -> None:
        sp = float(val)
        if abs(self.ghs_sp_val - sp) < 1e-6:
            return
        self.ghs_sp_val = sp
        self.slider_ghs_sp.blockSignals(True)
        self.slider_ghs_sp.setValue(int(round(sp * 10000.0)))
        self.slider_ghs_sp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_ghs_d_changed(self, val: int) -> None:
        d = val / 100.0
        if abs(self.ghs_d_val - d) < 1e-4:
            return
        self.ghs_d_val = d
        self.spin_ghs_d.blockSignals(True)
        self.spin_ghs_d.setValue(d)
        self.spin_ghs_d.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_ghs_d_changed(self, val: float) -> None:
        d = float(val)
        if abs(self.ghs_d_val - d) < 1e-4:
            return
        self.ghs_d_val = d
        self.slider_ghs_d.blockSignals(True)
        self.slider_ghs_d.setValue(int(round(d * 100.0)))
        self.slider_ghs_d.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_ghs_b_changed(self, val: int) -> None:
        b = val / 100.0
        if abs(self.ghs_b_val - b) < 1e-4:
            return
        self.ghs_b_val = b
        self.spin_ghs_b.blockSignals(True)
        self.spin_ghs_b.setValue(b)
        self.spin_ghs_b.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_ghs_b_changed(self, val: float) -> None:
        b = float(val)
        if abs(self.ghs_b_val - b) < 1e-4:
            return
        self.ghs_b_val = b
        self.slider_ghs_b.blockSignals(True)
        self.slider_ghs_b.setValue(int(round(b * 100.0)))
        self.slider_ghs_b.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_ghs_hp_changed(self, val: int) -> None:
        hp = val / 1000.0
        if abs(self.ghs_hp_val - hp) < 1e-5:
            return
        self.ghs_hp_val = hp
        self.spin_ghs_hp.blockSignals(True)
        self.spin_ghs_hp.setValue(hp)
        self.spin_ghs_hp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_ghs_hp_changed(self, val: float) -> None:
        hp = float(val)
        if abs(self.ghs_hp_val - hp) < 1e-5:
            return
        self.ghs_hp_val = hp
        self.slider_ghs_hp.blockSignals(True)
        self.slider_ghs_hp.setValue(int(round(hp * 1000.0)))
        self.slider_ghs_hp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_ghs_lp_changed(self, val: int) -> None:
        lp = val / 1000.0
        if abs(self.ghs_lp_val - lp) < 1e-5:
            return
        self.ghs_lp_val = lp
        self.spin_ghs_lp.blockSignals(True)
        self.spin_ghs_lp.setValue(lp)
        self.spin_ghs_lp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_ghs_lp_changed(self, val: float) -> None:
        lp = float(val)
        if abs(self.ghs_lp_val - lp) < 1e-5:
            return
        self.ghs_lp_val = lp
        self.slider_ghs_lp.blockSignals(True)
        self.slider_ghs_lp.setValue(int(round(lp * 1000.0)))
        self.slider_ghs_lp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_ghs_bp_changed(self, val: int) -> None:
        bp = val / 10000.0
        if abs(self.ghs_bp_val - bp) < 1e-6:
            return
        self.ghs_bp_val = bp
        self.spin_ghs_bp.blockSignals(True)
        self.spin_ghs_bp.setValue(bp)
        self.spin_ghs_bp.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_ghs_bp_changed(self, val: float) -> None:
        bp = float(val)
        if abs(self.ghs_bp_val - bp) < 1e-6:
            return
        self.ghs_bp_val = bp
        self.slider_ghs_bp.blockSignals(True)
        self.slider_ghs_bp.setValue(int(round(bp * 10000.0)))
        self.slider_ghs_bp.blockSignals(False)
        self.parameter_changed.emit()

    # -------------------------------------------------------------
    # Modes & Configuration
    # -------------------------------------------------------------
    def set_stretch_mode(self, mode: str) -> None:
        if mode not in ("mtf", "ghs"):
            mode = "mtf"
        self.stretch_mode = mode
        if mode == "mtf":
            self.btn_mode_mtf.setChecked(True)
            self.w_mtf_container.setVisible(True)
            self.w_ghs_container.setVisible(False)
        else:
            self.btn_mode_ghs.setChecked(True)
            self.w_mtf_container.setVisible(False)
            self.w_ghs_container.setVisible(True)
        self.parameter_changed.emit()
        self.action_committed.emit()

    def set_ghs_color_mode(self, mode: str) -> None:
        if mode not in ("luma", "rgb"):
            mode = "luma"
        self.ghs_color_mode = mode
        if mode == "luma":
            self.btn_ghs_luma.setChecked(True)
        else:
            self.btn_ghs_rgb.setChecked(True)
        self.parameter_changed.emit()
        self.action_committed.emit()

    def sync_controls(self, bp: float, m: float) -> None:
        self.block_signals(True)
        try:
            self.bp_val = float(bp)
            self.mtf_val = float(m)
            self.spin_bp.setValue(bp)
            self.slider_bp.setValue(int(round(bp * 10000.0)))
            self.spin_mtf.setValue(m)
            self.slider_mtf.setValue(int(round(m * 4000.0)))
        finally:
            self.block_signals(False)
        self.parameter_changed.emit()

    def reset_sliders(self) -> None:
        self.block_signals(True)
        try:
            self.slider_bp.setValue(0)
            self.spin_bp.setValue(0.0)
            self.slider_mtf.setValue(400)
            self.spin_mtf.setValue(0.10)
            self.slider_contrast.setValue(0)
            self.bp_val = 0.0
            self.mtf_val = 0.10
            self.contrast_val = 0.0
        finally:
            self.block_signals(False)
        self.parameter_changed.emit()
        self.action_committed.emit()

    def reset_ghs(self) -> None:
        self.block_signals(True)
        try:
            self.spin_ghs_sp.setValue(0.02)
            self.slider_ghs_sp.setValue(200)
            self.spin_ghs_d.setValue(5.00)
            self.slider_ghs_d.setValue(500)
            self.spin_ghs_b.setValue(2.00)
            self.slider_ghs_b.setValue(200)
            self.spin_ghs_hp.setValue(0.00)
            self.slider_ghs_hp.setValue(0)
            self.spin_ghs_lp.setValue(0.00)
            self.slider_ghs_lp.setValue(0)
            self.spin_ghs_bp.setValue(0.00000)
            self.slider_ghs_bp.setValue(0)

            self.ghs_sp_val = 0.02
            self.ghs_d_val = 5.0
            self.ghs_b_val = 2.0
            self.ghs_hp_val = 0.0
            self.ghs_lp_val = 0.0
            self.ghs_bp_val = 0.0
            self.set_ghs_color_mode("luma")
        finally:
            self.block_signals(False)
        self.parameter_changed.emit()
        self.action_committed.emit()

    def get_state(self) -> Dict[str, Any]:
        return {
            "stretch_mode": self.stretch_mode,
            "ghs_color_mode": self.ghs_color_mode,
            "ghs_sp_val": float(self.spin_ghs_sp.value()),
            "ghs_d_val": float(self.spin_ghs_d.value()),
            "ghs_b_val": float(self.spin_ghs_b.value()),
            "ghs_hp_val": float(self.spin_ghs_hp.value()),
            "ghs_lp_val": float(self.spin_ghs_lp.value()),
            "ghs_bp_val": float(self.spin_ghs_bp.value()),
            "bp_val": float(self.spin_bp.value()),
            "mtf_val": float(self.spin_mtf.value()),
            "contrast_val": self.slider_contrast.value(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.block_signals(True)
        try:
            if "stretch_mode" in state:
                self.set_stretch_mode(state["stretch_mode"])
            if "ghs_color_mode" in state:
                self.set_ghs_color_mode(state["ghs_color_mode"])
            if "ghs_sp_val" in state:
                val = float(state["ghs_sp_val"])
                self.ghs_sp_val = val
                self.spin_ghs_sp.setValue(val)
                self.slider_ghs_sp.setValue(int(np.clip(val * 10000.0, 0, 10000)))
            if "ghs_d_val" in state:
                val = float(state["ghs_d_val"])
                self.ghs_d_val = val
                self.spin_ghs_d.setValue(val)
                self.slider_ghs_d.setValue(int(np.clip(val * 100.0, 0, 2000)))
            if "ghs_b_val" in state:
                val = float(state["ghs_b_val"])
                self.ghs_b_val = val
                self.spin_ghs_b.setValue(val)
                self.slider_ghs_b.setValue(int(np.clip(val * 100.0, 0, 1500)))
            if "ghs_hp_val" in state:
                val = float(state["ghs_hp_val"])
                self.ghs_hp_val = val
                self.spin_ghs_hp.setValue(val)
                self.slider_ghs_hp.setValue(int(np.clip(val * 1000.0, 0, 1000)))
            if "ghs_lp_val" in state:
                val = float(state["ghs_lp_val"])
                self.ghs_lp_val = val
                self.spin_ghs_lp.setValue(val)
                self.slider_ghs_lp.setValue(int(np.clip(val * 1000.0, 0, 1000)))
            if "ghs_bp_val" in state:
                val = float(state["ghs_bp_val"])
                self.ghs_bp_val = val
                self.spin_ghs_bp.setValue(val)
                self.slider_ghs_bp.setValue(int(np.clip(val * 10000.0, 0, 2000)))
            if "bp_val" in state:
                val = float(state["bp_val"])
                self.bp_val = val
                self.spin_bp.setValue(val)
                self.slider_bp.setValue(int(np.clip(val * 10000.0, 0, 2000)))
            if "mtf_val" in state:
                val = float(state["mtf_val"])
                self.mtf_val = val
                self.spin_mtf.setValue(val)
                self.slider_mtf.setValue(int(np.clip(val * 4000.0, 1, 2000)))
            if "contrast_val" in state:
                val = int(state["contrast_val"])
                self.slider_contrast.setValue(val)
                self.contrast_val = val / 100.0
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.slider_bp, self.spin_bp,
            self.slider_mtf, self.spin_mtf,
            self.slider_contrast,
            self.slider_ghs_sp, self.spin_ghs_sp,
            self.slider_ghs_d, self.spin_ghs_d,
            self.slider_ghs_b, self.spin_ghs_b,
            self.slider_ghs_hp, self.spin_ghs_hp,
            self.slider_ghs_lp, self.spin_ghs_lp,
            self.slider_ghs_bp, self.spin_ghs_bp,
            self.btn_mode_mtf, self.btn_mode_ghs,
            self.btn_ghs_luma, self.btn_ghs_rgb
        ):
            if w is not None:
                w.blockSignals(blocked)
