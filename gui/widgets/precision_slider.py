# gui/widgets/precision_slider.py
"""
gui/widgets/precision_slider.py - Control deslizante de alta precisión con alineación estándar y edición numérica.
Inspirado en los estándares de interfaces de edición creativa profesional (Adobe Lightroom Classic, DaVinci Resolve).
"""
import re
from typing import Optional, Callable
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox, QAbstractSpinBox
)
from PySide6.QtGui import QCursor


class PrecisionSpinBox(QDoubleSpinBox):
    """
    Pastilla interactiva de lectura y edición de valores numéricos estilo pro suite.
    Hereda de QDoubleSpinBox (sin botones de flechas para máxima elegancia) y añade
    compatibilidad con .setText(...) para permitir que callbacks legacy sigan funcionando.
    """
    def __init__(self, decimals: int = 0, force_sign: bool = False, parent: QWidget = None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._decimals = decimals
        self._force_sign = force_sign
        self._is_modified = False
        self.setCursor(Qt.IBeamCursor)
        self.setToolTip("Haz clic o doble clic para teclear un valor exacto con el teclado.")
        self._apply_style()

    def set_modified(self, modified: bool) -> None:
        if self._is_modified != modified:
            self._is_modified = modified
            self._apply_style()

    def _apply_style(self) -> None:
        if self._is_modified:
            self.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #121826;
                    border: 1px solid #0288d1;
                    border-radius: 4px;
                    color: #38bdf8;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 1px 4px;
                }
                QDoubleSpinBox:hover {
                    border-color: #00e5ff;
                    background-color: #172338;
                }
                QDoubleSpinBox:focus {
                    border-color: #00e5ff;
                    background-color: #19263e;
                    color: #ffffff;
                }
            """)
        else:
            self.setStyleSheet("""
                QDoubleSpinBox {
                    background-color: #141722;
                    border: 1px solid #232738;
                    border-radius: 4px;
                    color: #e2e8f0;
                    font-size: 11px;
                    font-weight: 500;
                    padding: 1px 4px;
                }
                QDoubleSpinBox:hover {
                    border-color: #38bdf8;
                    background-color: #181d2c;
                }
                QDoubleSpinBox:focus {
                    border-color: #00e5ff;
                    background-color: #19263e;
                    color: #ffffff;
                }
            """)

    def textFromValue(self, val: float) -> str:
        if self.specialValueText() and abs(val - self.minimum()) < 1e-6:
            return self.specialValueText()
        if self._decimals == 0:
            txt = f"{int(round(val))}"
        else:
            txt = f"{val:.{self._decimals}f}"
        if self._force_sign and val > 0:
            return f"+{txt}"
        return txt

    def setText(self, text: str) -> None:
        """Permite compatibilidad total cuando se invoca self.lbl_*.setText(...)."""
        m = re.search(r"[-+]?\d*\.?\d+", text.replace(",", "."))
        if m:
            try:
                self.setValue(float(m.group(0)))
            except ValueError:
                pass


class PrecisionSliderRow(QWidget):
    """
    Fila homogénea de 3 columnas para ajuste de precisión:
    [ ● Nombre (fijo) | Slider ultra-fino | Pastilla de valor editable ]
    """
    valueChanged = Signal(int)
    sliderPressed = Signal()
    sliderReleased = Signal()
    resetRequested = Signal()

    def __init__(
        self,
        title: str = "",
        min_val: int = 0,
        max_val: int = 100,
        default_val: int = 0,
        scale: float = 1.0,
        unit: str = "",
        decimals: int = 0,
        label_width: int = 120,
        value_width: int = 56,
        special_zero_text: str = "",
        force_sign: bool = False,
        parent: QWidget = None,
        label: Optional[str] = None,
        step: int = 1,
    ):
        super().__init__(parent)
        if label is not None:
            title = label
        self.raw_default = int(default_val)
        self.scale = float(scale) if scale != 0 else 1.0
        self.step = step
        self.unit = unit
        self.decimals = decimals
        self.special_zero_text = special_zero_text
        self.force_sign = force_sign
        if decimals >= 4 and value_width <= 60:
            value_width = 68
        elif decimals >= 2 and len(unit) > 2 and value_width <= 56:
            value_width = 64
        self._syncing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        # 1. Indicador dot + Etiqueta
        self.lbl_dot = QLabel("●")
        self.lbl_dot.setFixedWidth(8)
        self.lbl_dot.setStyleSheet("color: transparent; font-size: 8px;")
        layout.addWidget(self.lbl_dot)

        self.lbl_title = QLabel(title)
        self.lbl_title.setFixedWidth(max(40, label_width - 10))
        self.lbl_title.setCursor(Qt.PointingHandCursor)
        self.lbl_title.setToolTip(f"Doble clic para restablecer (por defecto: {self._format_value(self.raw_default)})")
        self.lbl_title.mouseDoubleClickEvent = lambda e: self.reset()
        layout.addWidget(self.lbl_title)

        # 2. Slider ultra-fino
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val), int(max_val))
        self.slider.setSingleStep(int(self.step))
        self.slider.setPageStep(int(self.step))
        self.slider.setValue(self.raw_default)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self.sliderPressed.emit)
        self.slider.sliderReleased.connect(self.sliderReleased.emit)
        self.slider.mouseDoubleClickEvent = lambda e: self.reset()
        layout.addWidget(self.slider, stretch=1)

        # 3. Pastilla de lectura / edición
        self.spin = PrecisionSpinBox(decimals=decimals, force_sign=force_sign)
        self.spin.setFixedWidth(value_width)
        self.spin.setDecimals(decimals)
        self.spin.setRange(min_val / self.scale, max_val / self.scale)
        self.spin.setSingleStep(1.0 / self.scale if self.scale > 1 else 1.0)
        if unit:
            self.spin.setSuffix(unit)
        if special_zero_text and min_val == 0:
            self.spin.setSpecialValueText(special_zero_text)
        self.spin.setValue(self.raw_default / self.scale)
        self.spin.valueChanged.connect(self._on_spin_changed)
        self.spin.editingFinished.connect(self.sliderReleased.emit)
        layout.addWidget(self.spin)

        # Alias para compatibilidad hacia atrás
        self.lbl_value = self.spin
        self.spin_val = self.spin

        self._update_styles()

    def _format_value(self, raw_val: int) -> str:
        if self.special_zero_text and raw_val == 0:
            return self.special_zero_text
        fval = raw_val / self.scale
        if self.decimals == 0:
            txt = f"{int(round(fval))}"
        else:
            txt = f"{fval:.{self.decimals}f}"
        if self.force_sign and fval > 0:
            txt = f"+{txt}"
        return f"{txt}{self.unit}"

    def _on_slider_changed(self, raw_val: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(raw_val / self.scale)
        self._update_styles()
        self._syncing = False
        self.valueChanged.emit(raw_val)

    def _on_spin_changed(self, fval: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        raw_val = int(round(fval * self.scale))
        raw_val = max(self.slider.minimum(), min(self.slider.maximum(), raw_val))
        self.slider.setValue(raw_val)
        self._update_styles()
        self._syncing = False
        self.valueChanged.emit(raw_val)

    def _update_styles(self) -> None:
        modified = (self.slider.value() != self.raw_default)
        if modified:
            self.lbl_dot.setStyleSheet("color: #38bdf8; font-size: 8px;")
            self.lbl_title.setStyleSheet("color: #f1f5f9; font-weight: 600; font-size: 11px;")
            self.spin.set_modified(True)
        else:
            self.lbl_dot.setStyleSheet("color: transparent; font-size: 8px;")
            self.lbl_title.setStyleSheet("color: #94a3b8; font-weight: normal; font-size: 11px;")
            self.spin.set_modified(False)

    def value(self) -> int:
        return self.slider.value()

    def float_value(self) -> float:
        return self.slider.value() / self.scale

    def setValue(self, val: int | float) -> None:
        raw_val = int(round(val * self.scale)) if isinstance(val, float) else int(val)
        self.slider.setValue(raw_val)
        self._update_styles()

    def setRange(self, min_val: int, max_val: int) -> None:
        self.slider.setRange(min_val, max_val)
        self.spin.setRange(min_val / self.scale, max_val / self.scale)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.lbl_title.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.spin.setEnabled(enabled)

    def blockSignals(self, block: bool) -> bool:
        prev = self.slider.blockSignals(block)
        self.spin.blockSignals(block)
        return prev

    def reset(self) -> None:
        self.setValue(self.raw_default)
        self.resetRequested.emit()
        self.sliderReleased.emit()

    def is_modified(self) -> bool:
        return self.slider.value() != self.raw_default

    def set_title(self, text: str) -> None:
        self.lbl_title.setText(text)

    def set_default(self, val: int) -> None:
        self.raw_default = int(val)
        self.lbl_title.setToolTip(f"Doble clic para restablecer (por defecto: {self._format_value(self.raw_default)})")
        self._update_styles()
