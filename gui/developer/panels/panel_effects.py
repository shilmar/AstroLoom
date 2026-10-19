"""Orton Glow and Star Diffusion visual effects panels for AstroLoom Developer."""
from __future__ import annotations

from typing import Optional, Callable, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSlider, QComboBox
)
from PySide6.QtCore import Qt, Signal as pyqtSignal

from gui.icons import get_icon


class EffectsPanel(QWidget):
    """Encapsulates Orton Glow and Star Diffusion (Star Mist) effects."""

    # Reactive signals
    parameter_changed = pyqtSignal()
    action_committed = pyqtSignal()
    orton_changed = pyqtSignal(int, int, int, str, str)       # strength, radius, threshold, tint_key, mode_key
    star_diff_changed = pyqtSignal(int, int, int, int, str)   # strength, radius, threshold, color, mode_key

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Internal parameter values
        self._orton_strength: float = 0.0
        self._orton_radius: float = 40.0
        self._orton_threshold: float = 0.05
        self._orton_tint_type: str = "warm_gold"
        self._orton_mode: str = "soft_light"

        self._star_diff_strength: float = 0.0
        self._star_diff_radius: float = 20.0
        self._star_diff_threshold: float = 0.45
        self._star_diff_color_boost: float = 0.50
        self._star_diff_mode: str = "screen"

        self._guard: bool = False
        self._setup_ui()

    @staticmethod
    def _make_resettable_label(lbl: QLabel, reset_action: Callable[[], None], default_desc: str = "") -> QLabel:
        """Assign double click behavior to reset associated control."""
        lbl.setCursor(Qt.PointingHandCursor)
        tt = f"Doble clic para restablecer a {default_desc}" if default_desc else "Doble clic para restablecer"
        lbl.setToolTip(tt)
        lbl.mouseDoubleClickEvent = lambda e: reset_action()
        return lbl

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # =====================================================================
        # 1. Efecto Orton y Resplandor de Polvo Galáctico
        # =====================================================================
        self.box_orton_tools = QGroupBox("Efecto Orton y Glow Galáctico")
        self.box_orton_tools.setStyleSheet("""
            QGroupBox {
                border: 1px solid #7b1fa2;
                border-radius: 6px;
                margin-top: 4px;
                padding-top: 8px;
                background-color: #1a1625;
                font-weight: bold;
                color: #e1bee7;
            }
        """)
        box_orton_layout = QVBoxLayout(self.box_orton_tools)
        box_orton_layout.setContentsMargins(6, 4, 6, 6)
        box_orton_layout.setSpacing(6)

        # Intensidad Orton
        row_ort_int = QHBoxLayout()
        row_ort_int.addWidget(self._make_resettable_label(QLabel("Intensidad Glow:"), lambda: self.slider_orton.setValue(0), "0%"))
        self.lbl_orton = QLabel("0%")
        self.lbl_orton.setStyleSheet("color: #ffb74d; font-weight: bold;")
        self.lbl_orton.setFixedWidth(34)
        row_ort_int.addWidget(self.lbl_orton)
        self.slider_orton = QSlider(Qt.Horizontal)
        self.slider_orton.setRange(0, 100)
        self.slider_orton.setValue(0)
        self.slider_orton.valueChanged.connect(self._on_orton_strength_changed)
        self.slider_orton.sliderReleased.connect(self.action_committed.emit)
        row_ort_int.addWidget(self.slider_orton, stretch=1)

        self.btn_orton_expand = QPushButton("")
        self.btn_orton_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#e1bee7", size=13))
        self.btn_orton_expand.setCheckable(True)
        self.btn_orton_expand.setChecked(False)
        self.btn_orton_expand.setToolTip("Despliega los controles avanzados de radio, umbral de polvo, tinte cromático y modo de fusión.")
        self.btn_orton_expand.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 6px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #4a148c; color: #e1bee7; border: 1px solid #ba68c8; }"
        )
        self.btn_orton_expand.toggled.connect(self._on_toggle_orton_expand)
        row_ort_int.addWidget(self.btn_orton_expand)
        box_orton_layout.addLayout(row_ort_int)

        # Parámetros avanzados colapsables Orton
        self.w_orton_advanced = QWidget()
        self.w_orton_advanced.setVisible(False)
        w_ort_layout = QVBoxLayout(self.w_orton_advanced)
        w_ort_layout.setContentsMargins(0, 2, 0, 0)
        w_ort_layout.setSpacing(6)

        # Fila Radio y Umbral Orton
        row_ort_ru = QHBoxLayout()
        row_ort_ru.setSpacing(10)

        col_rad = QVBoxLayout()
        col_rad.setSpacing(2)
        row_rad_lbl = QHBoxLayout()
        row_rad_lbl.addWidget(self._make_resettable_label(QLabel("Radio (Difusión):"), lambda: self.slider_orton_radius.setValue(40), "40 px"))
        self.lbl_orton_radius = QLabel("40 px")
        self.lbl_orton_radius.setStyleSheet("color: #ce93d8; font-weight: bold;")
        row_rad_lbl.addWidget(self.lbl_orton_radius)
        col_rad.addLayout(row_rad_lbl)
        self.slider_orton_radius = QSlider(Qt.Horizontal)
        self.slider_orton_radius.setRange(10, 150)
        self.slider_orton_radius.setValue(40)
        self.slider_orton_radius.valueChanged.connect(self._on_orton_radius_changed)
        self.slider_orton_radius.sliderReleased.connect(self.action_committed.emit)
        col_rad.addWidget(self.slider_orton_radius)
        row_ort_ru.addLayout(col_rad)

        col_thr = QVBoxLayout()
        col_thr.setSpacing(2)
        row_thr_lbl = QHBoxLayout()
        row_thr_lbl.addWidget(self._make_resettable_label(QLabel("Umbral Polvo:"), lambda: self.slider_orton_threshold.setValue(5), "0.05"))
        self.lbl_orton_threshold = QLabel("0.05")
        self.lbl_orton_threshold.setStyleSheet("color: #ce93d8; font-weight: bold;")
        row_thr_lbl.addWidget(self.lbl_orton_threshold)
        col_thr.addLayout(row_thr_lbl)
        self.slider_orton_threshold = QSlider(Qt.Horizontal)
        self.slider_orton_threshold.setRange(0, 50)
        self.slider_orton_threshold.setValue(5)
        self.slider_orton_threshold.valueChanged.connect(self._on_orton_threshold_changed)
        self.slider_orton_threshold.sliderReleased.connect(self.action_committed.emit)
        col_thr.addWidget(self.slider_orton_threshold)
        row_ort_ru.addLayout(col_thr)
        w_ort_layout.addLayout(row_ort_ru)

        # Fila Tinte y Fusión Orton
        row_ort_tf = QHBoxLayout()
        row_ort_tf.setSpacing(8)

        col_tint = QVBoxLayout()
        col_tint.setSpacing(2)
        col_tint.addWidget(QLabel("Tinte:"))
        self.combo_orton_tint = QComboBox()
        self.combo_orton_tint.addItem("Dorado Cálido (Núcleo)", "warm_gold")
        self.combo_orton_tint.addItem("Luminancia Neutra", "neutral")
        self.combo_orton_tint.addItem("Azul Frío (Reflexión)", "cool_blue")
        self.combo_orton_tint.addItem("H-Alfa (Rubí Cósmico)", "halpha")
        self.combo_orton_tint.currentIndexChanged.connect(self._on_orton_tint_changed)
        col_tint.addWidget(self.combo_orton_tint)
        row_ort_tf.addLayout(col_tint, stretch=1)

        col_mode = QVBoxLayout()
        col_mode.setSpacing(2)
        col_mode.addWidget(QLabel("Fusión:"))
        self.combo_orton_mode = QComboBox()
        self.combo_orton_mode.addItem("Luz Suave (Profundidad)", "soft_light")
        self.combo_orton_mode.addItem("Trama (Velo Etéreo)", "screen")
        self.combo_orton_mode.addItem("Aditivo (Luminoso)", "additive")
        self.combo_orton_mode.currentIndexChanged.connect(self._on_orton_mode_changed)
        col_mode.addWidget(self.combo_orton_mode)
        row_ort_tf.addLayout(col_mode, stretch=1)

        w_ort_layout.addLayout(row_ort_tf)
        box_orton_layout.addWidget(self.w_orton_advanced)
        layout.addWidget(self.box_orton_tools)

        # =====================================================================
        # 2. Filtro Difusor y Realce de Constelaciones (Star Mist)
        # =====================================================================
        self.box_star_diff_tools = QGroupBox("Filtro Difusor y Realce de Constelaciones")
        self.box_star_diff_tools.setStyleSheet("""
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
        box_sd_layout = QVBoxLayout(self.box_star_diff_tools)
        box_sd_layout.setContentsMargins(6, 4, 6, 6)
        box_sd_layout.setSpacing(6)

        # Intensidad Difusor
        row_sd_int = QHBoxLayout()
        row_sd_int.addWidget(self._make_resettable_label(QLabel("Intensidad Difusor:"), lambda: self.slider_star_diff.setValue(0), "0%"))
        self.lbl_star_diff = QLabel("0%")
        self.lbl_star_diff.setStyleSheet("color: #ffd54f; font-weight: bold;")
        self.lbl_star_diff.setFixedWidth(34)
        row_sd_int.addWidget(self.lbl_star_diff)
        self.slider_star_diff = QSlider(Qt.Horizontal)
        self.slider_star_diff.setRange(0, 100)
        self.slider_star_diff.setValue(0)
        self.slider_star_diff.valueChanged.connect(self._on_star_diff_strength_changed)
        self.slider_star_diff.sliderReleased.connect(self.action_committed.emit)
        row_sd_int.addWidget(self.slider_star_diff, stretch=1)

        self.btn_star_diff_expand = QPushButton("")
        self.btn_star_diff_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#fff9c4", size=13))
        self.btn_star_diff_expand.setCheckable(True)
        self.btn_star_diff_expand.setChecked(False)
        self.btn_star_diff_expand.setToolTip("Despliega controles avanzados de radio de halo, umbral de magnitud, saturación espectral y modo de fusión.")
        self.btn_star_diff_expand.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 6px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #f57f17; color: #fffde7; border: 1px solid #ffd54f; }"
        )
        self.btn_star_diff_expand.toggled.connect(self._on_toggle_star_diff_expand)
        row_sd_int.addWidget(self.btn_star_diff_expand)
        box_sd_layout.addLayout(row_sd_int)

        # Opciones avanzadas colapsables Difusor
        self.w_star_diff_advanced = QWidget()
        self.w_star_diff_advanced.setVisible(False)
        w_sd_layout = QVBoxLayout(self.w_star_diff_advanced)
        w_sd_layout.setContentsMargins(0, 2, 0, 0)
        w_sd_layout.setSpacing(6)

        # Fila Radio y Umbral Difusor
        row_sd_ru = QHBoxLayout()
        row_sd_ru.setSpacing(10)

        col_sd_rad = QVBoxLayout()
        col_sd_rad.setSpacing(2)
        row_sd_rad_lbl = QHBoxLayout()
        row_sd_rad_lbl.addWidget(self._make_resettable_label(QLabel("Radio (Halo):"), lambda: self.slider_star_diff_radius.setValue(20), "20 px"))
        self.lbl_star_diff_radius = QLabel("20 px")
        self.lbl_star_diff_radius.setStyleSheet("color: #ffe082; font-weight: bold;")
        row_sd_rad_lbl.addWidget(self.lbl_star_diff_radius)
        col_sd_rad.addLayout(row_sd_rad_lbl)
        self.slider_star_diff_radius = QSlider(Qt.Horizontal)
        self.slider_star_diff_radius.setRange(5, 60)
        self.slider_star_diff_radius.setValue(20)
        self.slider_star_diff_radius.valueChanged.connect(self._on_star_diff_radius_changed)
        self.slider_star_diff_radius.sliderReleased.connect(self.action_committed.emit)
        col_sd_rad.addWidget(self.slider_star_diff_radius)
        row_sd_ru.addLayout(col_sd_rad)

        col_sd_thr = QVBoxLayout()
        col_sd_thr.setSpacing(2)
        row_sd_thr_lbl = QHBoxLayout()
        row_sd_thr_lbl.addWidget(self._make_resettable_label(QLabel("Umbral Magnitud:"), lambda: self.slider_star_diff_threshold.setValue(45), "0.45"))
        self.lbl_star_diff_threshold = QLabel("0.45")
        self.lbl_star_diff_threshold.setStyleSheet("color: #ffe082; font-weight: bold;")
        row_sd_thr_lbl.addWidget(self.lbl_star_diff_threshold)
        col_sd_thr.addLayout(row_sd_thr_lbl)
        self.slider_star_diff_threshold = QSlider(Qt.Horizontal)
        self.slider_star_diff_threshold.setRange(15, 85)
        self.slider_star_diff_threshold.setValue(45)
        self.slider_star_diff_threshold.valueChanged.connect(self._on_star_diff_threshold_changed)
        self.slider_star_diff_threshold.sliderReleased.connect(self.action_committed.emit)
        col_sd_thr.addWidget(self.slider_star_diff_threshold)
        row_sd_ru.addLayout(col_sd_thr)
        w_sd_layout.addLayout(row_sd_ru)

        # Fila Color Espectral y Modo Difusor
        row_sd_cm = QHBoxLayout()
        row_sd_cm.setSpacing(10)

        col_sd_col = QVBoxLayout()
        col_sd_col.setSpacing(2)
        row_sd_col_lbl = QHBoxLayout()
        row_sd_col_lbl.addWidget(self._make_resettable_label(QLabel("Color Espectral:"), lambda: self.slider_star_diff_color.setValue(50), "50%"))
        self.lbl_star_diff_color = QLabel("50%")
        self.lbl_star_diff_color.setStyleSheet("color: #ffe082; font-weight: bold;")
        row_sd_col_lbl.addWidget(self.lbl_star_diff_color)
        col_sd_col.addLayout(row_sd_col_lbl)
        self.slider_star_diff_color = QSlider(Qt.Horizontal)
        self.slider_star_diff_color.setRange(0, 100)
        self.slider_star_diff_color.setValue(50)
        self.slider_star_diff_color.valueChanged.connect(self._on_star_diff_color_changed)
        self.slider_star_diff_color.sliderReleased.connect(self.action_committed.emit)
        col_sd_col.addWidget(self.slider_star_diff_color)
        row_sd_cm.addLayout(col_sd_col)

        col_sd_mode = QVBoxLayout()
        col_sd_mode.setSpacing(2)
        col_sd_mode.addWidget(QLabel("Fusión:"))
        self.combo_star_diff_mode = QComboBox()
        self.combo_star_diff_mode.addItem("Trama (Velo Etéreo)", "screen")
        self.combo_star_diff_mode.addItem("Luz Suave (Aterciopelado)", "soft_light")
        self.combo_star_diff_mode.addItem("Aditivo (Luminoso)", "additive")
        self.combo_star_diff_mode.currentIndexChanged.connect(self._on_star_diff_mode_changed)
        col_sd_mode.addWidget(self.combo_star_diff_mode)
        row_sd_cm.addLayout(col_sd_mode)
        w_sd_layout.addLayout(row_sd_cm)

        box_sd_layout.addWidget(self.w_star_diff_advanced)
        layout.addWidget(self.box_star_diff_tools)

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def orton_strength(self) -> float:
        return self._orton_strength

    @orton_strength.setter
    def orton_strength(self, val: float):
        self._orton_strength = float(val)

    @property
    def orton_radius(self) -> float:
        return self._orton_radius

    @orton_radius.setter
    def orton_radius(self, val: float):
        self._orton_radius = float(val)

    @property
    def orton_threshold(self) -> float:
        return self._orton_threshold

    @orton_threshold.setter
    def orton_threshold(self, val: float):
        self._orton_threshold = float(val)

    @property
    def orton_tint_type(self) -> str:
        return self._orton_tint_type

    @orton_tint_type.setter
    def orton_tint_type(self, val: str):
        self._orton_tint_type = str(val)

    @property
    def orton_mode(self) -> str:
        return self._orton_mode

    @orton_mode.setter
    def orton_mode(self, val: str):
        self._orton_mode = str(val)

    @property
    def star_diff_strength(self) -> float:
        return self._star_diff_strength

    @star_diff_strength.setter
    def star_diff_strength(self, val: float):
        self._star_diff_strength = float(val)

    @property
    def star_diff_radius(self) -> float:
        return self._star_diff_radius

    @star_diff_radius.setter
    def star_diff_radius(self, val: float):
        self._star_diff_radius = float(val)

    @property
    def star_diff_threshold(self) -> float:
        return self._star_diff_threshold

    @star_diff_threshold.setter
    def star_diff_threshold(self, val: float):
        self._star_diff_threshold = float(val)

    @property
    def star_diff_color_boost(self) -> float:
        return self._star_diff_color_boost

    @star_diff_color_boost.setter
    def star_diff_color_boost(self, val: float):
        self._star_diff_color_boost = float(val)

    @property
    def star_diff_mode(self) -> str:
        return self._star_diff_mode

    @star_diff_mode.setter
    def star_diff_mode(self, val: str):
        self._star_diff_mode = str(val)

    # =========================================================================
    # Internal Slots / Event Handlers
    # =========================================================================
    def _on_toggle_orton_expand(self, checked: bool) -> None:
        self.w_orton_advanced.setVisible(checked)

    def _on_orton_strength_changed(self, val: int) -> None:
        self._orton_strength = val / 100.0
        self.lbl_orton.setText(f"{val}%")
        if not self._guard:
            self._emit_orton_changed()

    def _on_orton_radius_changed(self, val: int) -> None:
        self._orton_radius = float(val)
        self.lbl_orton_radius.setText(f"{val} px")
        if not self._guard:
            self._emit_orton_changed()

    def _on_orton_threshold_changed(self, val: int) -> None:
        self._orton_threshold = val / 100.0
        self.lbl_orton_threshold.setText(f"{self._orton_threshold:.2f}")
        if not self._guard:
            self._emit_orton_changed()

    def _on_orton_tint_changed(self, idx: int) -> None:
        self._orton_tint_type = self.combo_orton_tint.currentData() or "warm_gold"
        if not self._guard:
            self._emit_orton_changed()
            self.action_committed.emit()

    def _on_orton_mode_changed(self, idx: int) -> None:
        self._orton_mode = self.combo_orton_mode.currentData() or "soft_light"
        if not self._guard:
            self._emit_orton_changed()
            self.action_committed.emit()

    def _emit_orton_changed(self) -> None:
        self.orton_changed.emit(
            self.slider_orton.value(),
            self.slider_orton_radius.value(),
            self.slider_orton_threshold.value(),
            self._orton_tint_type,
            self._orton_mode
        )
        self.parameter_changed.emit()

    def _on_toggle_star_diff_expand(self, checked: bool) -> None:
        self.w_star_diff_advanced.setVisible(checked)

    def _on_star_diff_strength_changed(self, val: int) -> None:
        self._star_diff_strength = val / 100.0
        self.lbl_star_diff.setText(f"{val}%")
        if not self._guard:
            self._emit_star_diff_changed()

    def _on_star_diff_radius_changed(self, val: int) -> None:
        self._star_diff_radius = float(val)
        self.lbl_star_diff_radius.setText(f"{val} px")
        if not self._guard:
            self._emit_star_diff_changed()

    def _on_star_diff_threshold_changed(self, val: int) -> None:
        self._star_diff_threshold = val / 100.0
        self.lbl_star_diff_threshold.setText(f"{val / 100.0:.2f}")
        if not self._guard:
            self._emit_star_diff_changed()

    def _on_star_diff_color_changed(self, val: int) -> None:
        self._star_diff_color_boost = val / 100.0
        self.lbl_star_diff_color.setText(f"{val}%")
        if not self._guard:
            self._emit_star_diff_changed()

    def _on_star_diff_mode_changed(self, idx: int) -> None:
        modes = ["screen", "soft_light", "additive"]
        self._star_diff_mode = modes[idx] if 0 <= idx < len(modes) else "screen"
        if not self._guard:
            self._emit_star_diff_changed()
            self.action_committed.emit()

    def _emit_star_diff_changed(self) -> None:
        self.star_diff_changed.emit(
            self.slider_star_diff.value(),
            self.slider_star_diff_radius.value(),
            self.slider_star_diff_threshold.value(),
            self.slider_star_diff_color.value(),
            self._star_diff_mode
        )
        self.parameter_changed.emit()

    # =========================================================================
    # Public Control API
    # =========================================================================
    def set_orton_values(self, strength: int, radius: int, threshold: int, tint: str | int = 0, mode: str | int = 0) -> None:
        """Batch set Orton values without duplicate signal emissions."""
        self._guard = True
        try:
            self.slider_orton.setValue(strength)
            self._orton_strength = strength / 100.0
            self.lbl_orton.setText(f"{strength}%")

            self.slider_orton_radius.setValue(radius)
            self._orton_radius = float(radius)
            self.lbl_orton_radius.setText(f"{radius} px")

            self.slider_orton_threshold.setValue(threshold)
            self._orton_threshold = threshold / 100.0
            self.lbl_orton_threshold.setText(f"{self._orton_threshold:.2f}")

            if isinstance(tint, int):
                self.combo_orton_tint.setCurrentIndex(tint)
            else:
                idx = self.combo_orton_tint.findData(tint)
                if idx >= 0:
                    self.combo_orton_tint.setCurrentIndex(idx)
            self._orton_tint_type = self.combo_orton_tint.currentData() or "warm_gold"

            if isinstance(mode, int):
                self.combo_orton_mode.setCurrentIndex(mode)
            else:
                idx = self.combo_orton_mode.findData(mode)
                if idx >= 0:
                    self.combo_orton_mode.setCurrentIndex(idx)
            self._orton_mode = self.combo_orton_mode.currentData() or "soft_light"
        finally:
            self._guard = False

    def reset_orton(self) -> None:
        """Reset Orton effect to defaults."""
        self.set_orton_values(0, 40, 5, 0, 0)
        self._emit_orton_changed()

    def set_star_diff_values(self, strength: int, radius: int, threshold: int, color: int, mode: str | int = 0) -> None:
        """Batch set Star Diffusion values without duplicate signal emissions."""
        self._guard = True
        try:
            self.slider_star_diff.setValue(strength)
            self._star_diff_strength = strength / 100.0
            self.lbl_star_diff.setText(f"{strength}%")

            self.slider_star_diff_radius.setValue(radius)
            self._star_diff_radius = float(radius)
            self.lbl_star_diff_radius.setText(f"{radius} px")

            self.slider_star_diff_threshold.setValue(threshold)
            self._star_diff_threshold = threshold / 100.0
            self.lbl_star_diff_threshold.setText(f"{threshold / 100.0:.2f}")

            self.slider_star_diff_color.setValue(color)
            self._star_diff_color_boost = color / 100.0
            self.lbl_star_diff_color.setText(f"{color}%")

            if isinstance(mode, int):
                self.combo_star_diff_mode.setCurrentIndex(mode)
            else:
                idx = self.combo_star_diff_mode.findData(mode)
                if idx >= 0:
                    self.combo_star_diff_mode.setCurrentIndex(idx)
            modes = ["screen", "soft_light", "additive"]
            idx = self.combo_star_diff_mode.currentIndex()
            self._star_diff_mode = modes[idx] if 0 <= idx < len(modes) else "screen"
        finally:
            self._guard = False

    def reset_star_diff(self) -> None:
        """Reset Star Diffusion effect to defaults."""
        self.set_star_diff_values(0, 20, 45, 50, 0)
        self._emit_star_diff_changed()

    def get_state(self) -> Dict[str, Any]:
        """Captura el estado paramétrico de los efectos Orton y Difusión de estrellas."""
        return {
            "orton_strength": self.slider_orton.value(),
            "orton_radius": self.slider_orton_radius.value(),
            "orton_threshold": self.slider_orton_threshold.value(),
            "orton_tint_idx": self.combo_orton_tint.currentIndex(),
            "orton_mode_idx": self.combo_orton_mode.currentIndex(),
            "star_diff_strength": self.slider_star_diff.value(),
            "star_diff_radius": self.slider_star_diff_radius.value(),
            "star_diff_threshold": self.slider_star_diff_threshold.value(),
            "star_diff_color": self.slider_star_diff_color.value(),
            "star_diff_mode_idx": self.combo_star_diff_mode.currentIndex(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaura los controles de efectos visuales sin disparar eventos redundantes."""
        self.block_signals(True)
        try:
            if "orton_strength" in state:
                val = state["orton_strength"]
                self.slider_orton.setValue(val)
                self._orton_strength = val / 100.0
                self.lbl_orton.setText(f"{val}%")
            if "orton_radius" in state:
                val = state["orton_radius"]
                self.slider_orton_radius.setValue(val)
                self._orton_radius = float(val)
                self.lbl_orton_radius.setText(f"{val} px")
            if "orton_threshold" in state:
                val = state["orton_threshold"]
                self.slider_orton_threshold.setValue(val)
                self._orton_threshold = val / 100.0
                self.lbl_orton_threshold.setText(f"{self._orton_threshold:.2f}")
            if "orton_tint_idx" in state:
                idx = state["orton_tint_idx"]
                self.combo_orton_tint.setCurrentIndex(idx)
                self._orton_tint_type = self.combo_orton_tint.currentData() or "warm_gold"
            if "orton_mode_idx" in state:
                idx = state["orton_mode_idx"]
                self.combo_orton_mode.setCurrentIndex(idx)
                self._orton_mode = self.combo_orton_mode.currentData() or "soft_light"

            if "star_diff_strength" in state:
                val = state["star_diff_strength"]
                self.slider_star_diff.setValue(val)
                self._star_diff_strength = val / 100.0
                self.lbl_star_diff.setText(f"{val}%")
            if "star_diff_radius" in state:
                val = state["star_diff_radius"]
                self.slider_star_diff_radius.setValue(val)
                self._star_diff_radius = float(val)
                self.lbl_star_diff_radius.setText(f"{val} px")
            if "star_diff_threshold" in state:
                val = state["star_diff_threshold"]
                self.slider_star_diff_threshold.setValue(val)
                self._star_diff_threshold = val / 100.0
                self.lbl_star_diff_threshold.setText(f"{self._star_diff_threshold:.2f}")
            if "star_diff_color" in state:
                val = state["star_diff_color"]
                self.slider_star_diff_color.setValue(val)
                self._star_diff_color_boost = val / 100.0
                self.lbl_star_diff_color.setText(f"{val}%")
            if "star_diff_mode_idx" in state:
                idx = state["star_diff_mode_idx"]
                self.combo_star_diff_mode.setCurrentIndex(idx)
                modes = ["screen", "soft_light", "additive"]
                self._star_diff_mode = modes[idx] if 0 <= idx < len(modes) else "screen"
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """Bloquea las señales de los widgets y la emisión reactiva interna."""
        self._guard = blocked
        for w in (
            self.slider_orton, self.slider_orton_radius, self.slider_orton_threshold,
            self.combo_orton_tint, self.combo_orton_mode,
            self.slider_star_diff, self.slider_star_diff_radius,
            self.slider_star_diff_threshold, self.slider_star_diff_color,
            self.combo_star_diff_mode
        ):
            if w is not None:
                w.blockSignals(blocked)

