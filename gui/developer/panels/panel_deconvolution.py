"""Richardson-Lucy deconvolution collapsible panel for AstroLoom Developer."""
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal as pyqtSignal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.icons import get_icon


class DeconvolutionPanel(QWidget):
    """Encapsulates Richardson-Lucy optical deconvolution and PSF configuration controls."""

    parameter_changed = pyqtSignal()
    action_committed = pyqtSignal()
    auto_psf_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.deconv_iterations: int = 0
        self.deconv_fwhm: float = 3.0
        self.deconv_deringing: float = 0.008
        self.deconv_damping: float = 0.25
        self.deconv_aspect_ratio: float = 1.0
        self.deconv_angle_deg: float = 0.0
        self.deconv_model: str = "moffat"
        self.deconv_scope: str = "all"
        self.deconv_luma_only: bool = True

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- SECCIÓN: Deconvolución Óptica y Enfoque (R-L) ---
        self.sec_deconv = CollapsibleSection(
            "Deconvolución Óptica y Enfoque (R-L)",
            is_expanded=False,
            on_reset=self.reset_deconv,
            icon_name="microscope"
        )
        w_deconv = QWidget()
        deconv_layout = QVBoxLayout(w_deconv)
        deconv_layout.setContentsMargins(6, 4, 6, 6)
        deconv_layout.setSpacing(6)

        # 1. Botón de Estimación Automática de PSF y Estado
        self.btn_auto_psf = QPushButton("Auto-Estimar PSF de Estrellas")
        self.btn_auto_psf.setIcon(get_icon("crosshair", color="#90caf9", size=14))
        self.btn_auto_psf.setToolTip("Analiza estrellas aisladas no saturadas de la imagen para determinar el diámetro FWHM y relación de aspecto de la PSF.")
        self.btn_auto_psf.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; background: #1c2836; border: 1px solid #1e88e5; border-radius: 4px; color: #90caf9; }"
            "QPushButton:hover { background: #22374e; border-color: #42a5f5; }"
        )
        self.btn_auto_psf.setCursor(Qt.PointingHandCursor)
        self.btn_auto_psf.clicked.connect(self.auto_psf_requested.emit)
        deconv_layout.addWidget(self.btn_auto_psf)

        self.lbl_psf_status = QLabel("PSF: Moffat 2D (FWHM: 3.0 px | Aspecto: 1.00)")
        self.lbl_psf_status.setWordWrap(True)
        self.lbl_psf_status.setStyleSheet("color: #90caf9; font-size: 11px; padding-left: 2px;")
        deconv_layout.addWidget(self.lbl_psf_status)

        # 2. Control Principal: Iteraciones R-L
        self.row_deconv_iter = PrecisionSliderRow("Iteraciones (R-L):", 0, 40, 0, special_zero_text="0 (Off)", label_width=115)
        self.slider_deconv_iter = self.row_deconv_iter.slider
        self.lbl_deconv_iter = self.row_deconv_iter.lbl_value

        self.btn_deconv_expand = QPushButton("")
        self.btn_deconv_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#e3f2fd", size=13))
        self.btn_deconv_expand.setCheckable(True)
        self.btn_deconv_expand.setChecked(False)
        self.btn_deconv_expand.setToolTip("Despliega controles avanzados de FWHM, antianillos (deringing), amortiguación de ruido, elipticidad y ámbito.")
        self.btn_deconv_expand.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 6px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #1976d2; color: #e3f2fd; border: 1px solid #42a5f5; }"
        )
        self.btn_deconv_expand.toggled.connect(self._on_toggle_deconv_expand)
        self.row_deconv_iter.layout().addWidget(self.btn_deconv_expand)

        self.row_deconv_iter.valueChanged.connect(self._on_iterations_changed)
        self.row_deconv_iter.sliderReleased.connect(self.action_committed.emit)
        deconv_layout.addWidget(self.row_deconv_iter)

        # 3. Control Principal: FWHM
        self.row_deconv_fwhm = PrecisionSliderRow("FWHM (PSF):", 10, 80, 30, scale=10.0, decimals=1, unit=" px", label_width=115)
        self.slider_deconv_fwhm = self.row_deconv_fwhm.slider
        self.lbl_deconv_fwhm = self.row_deconv_fwhm.lbl_value
        self.row_deconv_fwhm.valueChanged.connect(self._on_fwhm_changed)
        self.row_deconv_fwhm.sliderReleased.connect(self.action_committed.emit)
        deconv_layout.addWidget(self.row_deconv_fwhm)

        # 4. Panel Avanzado Desplegable
        self.w_deconv_advanced = QWidget()
        self.w_deconv_advanced.setVisible(False)
        w_dadv_layout = QVBoxLayout(self.w_deconv_advanced)
        w_dadv_layout.setContentsMargins(0, 4, 0, 2)
        w_dadv_layout.setSpacing(6)

        self.row_deconv_deringing = PrecisionSliderRow("Antianillos:", 0, 40, 8, scale=1000.0, decimals=3, label_width=115)
        self.slider_deconv_deringing = self.row_deconv_deringing.slider
        self.lbl_deconv_deringing = self.row_deconv_deringing.lbl_value
        self.row_deconv_deringing.valueChanged.connect(self._on_deringing_changed)
        self.row_deconv_deringing.sliderReleased.connect(self.action_committed.emit)
        w_dadv_layout.addWidget(self.row_deconv_deringing)

        self.row_deconv_damping = PrecisionSliderRow("Amortiguación:", 0, 100, 25, unit="%", label_width=115)
        self.slider_deconv_damping = self.row_deconv_damping.slider
        self.lbl_deconv_damping = self.row_deconv_damping.lbl_value
        self.row_deconv_damping.valueChanged.connect(self._on_damping_changed)
        self.row_deconv_damping.sliderReleased.connect(self.action_committed.emit)
        w_dadv_layout.addWidget(self.row_deconv_damping)

        self.row_deconv_aspect = PrecisionSliderRow("Aspecto PSF:", 100, 250, 100, scale=100.0, decimals=2, unit="x", label_width=115)
        self.slider_deconv_aspect = self.row_deconv_aspect.slider
        self.lbl_deconv_aspect = self.row_deconv_aspect.lbl_value
        self.row_deconv_aspect.valueChanged.connect(self._on_aspect_changed)
        self.row_deconv_aspect.sliderReleased.connect(self.action_committed.emit)
        w_dadv_layout.addWidget(self.row_deconv_aspect)

        self.row_deconv_angle = PrecisionSliderRow("Ángulo PSF:", -90, 90, 0, unit="°", force_sign=True, label_width=115)
        self.slider_deconv_angle = self.row_deconv_angle.slider
        self.lbl_deconv_angle = self.row_deconv_angle.lbl_value
        self.row_deconv_angle.valueChanged.connect(self._on_angle_changed)
        self.row_deconv_angle.sliderReleased.connect(self.action_committed.emit)
        w_dadv_layout.addWidget(self.row_deconv_angle)

        # Fila Modelo PSF y Ámbito de Aplicación
        row_mod_scp = QHBoxLayout()
        row_mod_scp.setSpacing(10)

        col_mod = QVBoxLayout()
        col_mod.setSpacing(2)
        col_mod.addWidget(QLabel("Perfil PSF:"))
        self.combo_deconv_model = QComboBox()
        self.combo_deconv_model.addItem("Moffat 2D (Astrofísica)", "moffat")
        self.combo_deconv_model.addItem("Gaussiana 2D", "gaussian")
        self.combo_deconv_model.currentIndexChanged.connect(self._on_model_changed)
        col_mod.addWidget(self.combo_deconv_model)
        row_mod_scp.addLayout(col_mod)

        col_scp = QVBoxLayout()
        col_scp.setSpacing(2)
        col_scp.addWidget(QLabel("Ámbito:"))
        self.combo_deconv_scope = QComboBox()
        self.combo_deconv_scope.addItem("Toda la Imagen", "all")
        self.combo_deconv_scope.addItem("Solo Estrellas", "stars_only")
        self.combo_deconv_scope.addItem("Solo Fondo / Nebulosa", "background_only")
        self.combo_deconv_scope.currentIndexChanged.connect(self._on_scope_changed)
        col_scp.addWidget(self.combo_deconv_scope)
        row_mod_scp.addLayout(col_scp)
        w_dadv_layout.addLayout(row_mod_scp)

        # Checkbox Solo Luminancia
        self.chk_deconv_luma_only = QCheckBox("Solo Luminancia (Preservar color)")
        self.chk_deconv_luma_only.setChecked(True)
        self.chk_deconv_luma_only.toggled.connect(self._on_luma_only_toggled)
        self.chk_deconv_luma_only.setStyleSheet("color: #b0bec5; font-size: 11px;")
        w_dadv_layout.addWidget(self.chk_deconv_luma_only)

        deconv_layout.addWidget(self.w_deconv_advanced)

        self.sec_deconv.set_content_widget(w_deconv)
        layout.addWidget(self.sec_deconv)

    # -------------------------------------------------------------
    # Sliders & Value synchronization
    # -------------------------------------------------------------
    def _on_toggle_deconv_expand(self, checked: bool) -> None:
        self.w_deconv_advanced.setVisible(checked)

    def _on_iterations_changed(self, val: int) -> None:
        if self.deconv_iterations == int(val):
            return
        self.deconv_iterations = int(val)
        txt = f"{val}" if val > 0 else "0 (Off)"
        self.lbl_deconv_iter.setText(txt)
        self.parameter_changed.emit()

    def _on_fwhm_changed(self, val: int) -> None:
        fwhm = val / 10.0
        if abs(self.deconv_fwhm - fwhm) < 1e-4:
            return
        self.deconv_fwhm = float(fwhm)
        self.lbl_deconv_fwhm.setText(f"{fwhm:.1f} px")
        self.parameter_changed.emit()

    def _on_deringing_changed(self, val: int) -> None:
        der = val / 1000.0
        if abs(self.deconv_deringing - der) < 1e-5:
            return
        self.deconv_deringing = float(der)
        self.lbl_deconv_deringing.setText(f"{der:.3f}")
        self.parameter_changed.emit()

    def _on_damping_changed(self, val: int) -> None:
        damp = val / 100.0
        if abs(self.deconv_damping - damp) < 1e-4:
            return
        self.deconv_damping = float(damp)
        self.lbl_deconv_damping.setText(f"{val}%")
        self.parameter_changed.emit()

    def _on_aspect_changed(self, val: int) -> None:
        aspect = val / 100.0
        if abs(self.deconv_aspect_ratio - aspect) < 1e-4:
            return
        self.deconv_aspect_ratio = float(aspect)
        self.lbl_deconv_aspect.setText(f"{aspect:.2f}x")
        self.parameter_changed.emit()

    def _on_angle_changed(self, val: int) -> None:
        angle = float(val)
        if abs(self.deconv_angle_deg - angle) < 1e-4:
            return
        self.deconv_angle_deg = angle
        self.lbl_deconv_angle.setText(f"{val}°")
        self.parameter_changed.emit()

    def _on_model_changed(self, idx: int) -> None:
        model = self.combo_deconv_model.itemData(idx) or ("moffat" if idx == 0 else "gaussian")
        if self.deconv_model == model:
            return
        self.deconv_model = model
        self.parameter_changed.emit()

    def _on_scope_changed(self, idx: int) -> None:
        scope = self.combo_deconv_scope.itemData(idx) or "all"
        if self.deconv_scope == scope:
            return
        self.deconv_scope = scope
        self.parameter_changed.emit()

    def _on_luma_only_toggled(self, checked: bool) -> None:
        if self.deconv_luma_only == checked:
            return
        self.deconv_luma_only = checked
        self.parameter_changed.emit()

    # -------------------------------------------------------------
    # Public controls API
    # -------------------------------------------------------------
    def reset_deconv(self) -> None:
        self.block_signals(True)
        try:
            self.slider_deconv_iter.setValue(0)
            self.slider_deconv_fwhm.setValue(30)
            self.slider_deconv_deringing.setValue(8)
            self.slider_deconv_damping.setValue(25)
            self.slider_deconv_aspect.setValue(100)
            self.slider_deconv_angle.setValue(0)
            self.combo_deconv_model.setCurrentIndex(0)
            self.combo_deconv_scope.setCurrentIndex(0)
            self.chk_deconv_luma_only.setChecked(True)

            self.deconv_iterations = 0
            self.deconv_fwhm = 3.0
            self.deconv_deringing = 0.008
            self.deconv_damping = 0.25
            self.deconv_aspect_ratio = 1.0
            self.deconv_angle_deg = 0.0
            self.deconv_model = "moffat"
            self.deconv_scope = "all"
            self.deconv_luma_only = True

            self.lbl_deconv_iter.setText("0 (Off)")
            self.lbl_deconv_fwhm.setText("3.0 px")
            self.lbl_deconv_deringing.setText("0.008")
            self.lbl_deconv_damping.setText("25%")
            self.lbl_deconv_aspect.setText("1.00x")
            self.lbl_deconv_angle.setText("0°")
            self.lbl_psf_status.setText("PSF: Moffat 2D (FWHM: 3.0 px | Aspecto: 1.00)")
        finally:
            self.block_signals(False)

        self.parameter_changed.emit()
        self.action_committed.emit()

    def update_psf_status(self, text: str) -> None:
        self.lbl_psf_status.setText(text)

    def get_state(self) -> Dict[str, Any]:
        return {
            "deconv_iterations": self.slider_deconv_iter.value(),
            "deconv_fwhm": self.slider_deconv_fwhm.value(),
            "deconv_deringing": self.slider_deconv_deringing.value(),
            "deconv_damping": self.slider_deconv_damping.value(),
            "deconv_aspect": self.slider_deconv_aspect.value(),
            "deconv_angle": self.slider_deconv_angle.value(),
            "deconv_model_idx": self.combo_deconv_model.currentIndex(),
            "deconv_scope_idx": self.combo_deconv_scope.currentIndex(),
            "deconv_luma_only": self.chk_deconv_luma_only.isChecked(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.block_signals(True)
        try:
            if "deconv_iterations" in state:
                val = state["deconv_iterations"]
                self.slider_deconv_iter.setValue(val)
                self.deconv_iterations = int(val)
                self.lbl_deconv_iter.setText(f"{val}" if val > 0 else "0 (Off)")
            if "deconv_fwhm" in state:
                val = state["deconv_fwhm"]
                self.slider_deconv_fwhm.setValue(val)
                self.deconv_fwhm = val / 10.0
                self.lbl_deconv_fwhm.setText(f"{self.deconv_fwhm:.1f} px")
            if "deconv_deringing" in state:
                val = state["deconv_deringing"]
                self.slider_deconv_deringing.setValue(val)
                self.deconv_deringing = val / 1000.0
                self.lbl_deconv_deringing.setText(f"{self.deconv_deringing:.3f}")
            if "deconv_damping" in state:
                val = state["deconv_damping"]
                self.slider_deconv_damping.setValue(val)
                self.deconv_damping = val / 100.0
                self.lbl_deconv_damping.setText(f"{val}%")
            if "deconv_aspect" in state:
                val = state["deconv_aspect"]
                self.slider_deconv_aspect.setValue(val)
                self.deconv_aspect_ratio = val / 100.0
                self.lbl_deconv_aspect.setText(f"{self.deconv_aspect_ratio:.2f}x")
            if "deconv_angle" in state:
                val = state["deconv_angle"]
                self.slider_deconv_angle.setValue(val)
                self.deconv_angle_deg = float(val)
                self.lbl_deconv_angle.setText(f"{val}°")
            if "deconv_model_idx" in state:
                idx = state["deconv_model_idx"]
                self.combo_deconv_model.setCurrentIndex(idx)
                self.deconv_model = self.combo_deconv_model.itemData(idx) or "moffat"
            if "deconv_scope_idx" in state:
                idx = state["deconv_scope_idx"]
                self.combo_deconv_scope.setCurrentIndex(idx)
                self.deconv_scope = self.combo_deconv_scope.itemData(idx) or "all"
            if "deconv_luma_only" in state:
                self.chk_deconv_luma_only.setChecked(state["deconv_luma_only"])
                self.deconv_luma_only = state["deconv_luma_only"]
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.slider_deconv_iter, self.slider_deconv_fwhm,
            self.slider_deconv_deringing, self.slider_deconv_damping,
            self.slider_deconv_aspect, self.slider_deconv_angle,
            self.combo_deconv_model, self.combo_deconv_scope,
            self.chk_deconv_luma_only
        ):
            if w is not None:
                w.blockSignals(blocked)
