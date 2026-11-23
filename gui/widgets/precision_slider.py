# gui/widgets/precision_slider.py
"""
gui/widgets/precision_slider.py - Control deslizante de alta precisión con alineación estándar y edición numérica.
Inspirado en los estándares de interfaces de edición creativa profesional.
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
        # No usamos NoButtons porque en Qt 6 con QSS colapsa el edit-field a 1px; ocultamos botones por CSS.
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._decimals = decimals
        self._force_sign = force_sign
        self._is_modified = False
        self.setCursor(Qt.IBeamCursor)
        self.setToolTip("Haz clic o doble clic para teclear un valor exacto con el teclado.")
        self._apply_style()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        le = self.lineEdit()
        if le:
            le.setGeometry(self.rect().adjusted(2, 1, -2, -1))

    def set_modified(self, modified: bool) -> None:
        if self._is_modified != modified:
            self._is_modified = modified
            self._apply_style()

    def _apply_style(self) -> None:
        btn_css = "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0px; height: 0px; border: none; }"
        if self._is_modified:
            self.setStyleSheet(f"""
                QDoubleSpinBox {{
                    background-color: #2c2c32;
                    border: 1px solid #2f81f7;
                    border-radius: 4px;
                    color: #2f81f7;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 1px 4px;
                }}
                QDoubleSpinBox:hover {{
                    border-color: #388bfd;
                    background-color: #383840;
                }}
                QDoubleSpinBox:focus {{
                    border-color: #2f81f7;
                    background-color: #2c2c32;
                    color: #ffffff;
                }}
                {btn_css}
            """)
        else:
            self.setStyleSheet(f"""
                QDoubleSpinBox {{
                    background-color: #2c2c32;
                    border: 1px solid #2e2e35;
                    border-radius: 4px;
                    color: #a1a1aa;
                    font-size: 11px;
                    font-weight: 500;
                    padding: 1px 4px;
                }}
                QDoubleSpinBox:hover {{
                    border-color: #3f3f46;
                    background-color: #383840;
                    color: #e4e4e7;
                }}
                QDoubleSpinBox:focus {{
                    border-color: #2f81f7;
                    background-color: #2c2c32;
                    color: #ffffff;
                }}
                {btn_css}
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
        action_button: Optional[QWidget] = None,
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
        layout.setAlignment(Qt.AlignVCenter)

        # 1. Indicador dot + Etiqueta
        self.lbl_dot = QLabel("●")
        self.lbl_dot.setFixedWidth(8)
        self.lbl_dot.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.lbl_dot.setStyleSheet("color: transparent; font-size: 8px;")
        layout.addWidget(self.lbl_dot, alignment=Qt.AlignVCenter)

        self.lbl_title = QLabel(title)
        self.lbl_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_title.setCursor(Qt.PointingHandCursor)
        self.lbl_title.setToolTip(f"Doble clic para restablecer (por defecto: {self._format_value(self.raw_default)})")
        self.lbl_title.mouseDoubleClickEvent = lambda e: self.reset()

        if action_button is not None:
            btn_w = action_button.sizeHint().width() if action_button.sizeHint().width() > 0 else 18
            self.lbl_title.setFixedWidth(max(40, label_width - 10 - btn_w - 4))
            layout.addWidget(self.lbl_title, alignment=Qt.AlignVCenter)
            layout.addWidget(action_button, alignment=Qt.AlignVCenter)
        else:
            self.lbl_title.setFixedWidth(max(40, label_width - 10))
            layout.addWidget(self.lbl_title, alignment=Qt.AlignVCenter)

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
        layout.addWidget(self.slider, stretch=1, alignment=Qt.AlignVCenter)

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
            self.lbl_dot.setStyleSheet("color: #3b82f6; font-size: 8px;")
            self.lbl_title.setStyleSheet("color: #e4e4e7; font-weight: 600; font-size: 11px;")
            self.spin.set_modified(True)
        else:
            self.lbl_dot.setStyleSheet("color: transparent; font-size: 8px;")
            self.lbl_title.setStyleSheet("color: #a1a1aa; font-weight: normal; font-size: 11px;")
            self.spin.set_modified(False)

    def set_track_gradient(self, gradient_css: str, add_page_css: Optional[str] = None) -> None:
        """
        Aplica un degradado fotográfico a la pista del deslizador (groove) manteniendo
        el tirador en pastilla blanco hueso #F4F4F5 estilizado (e.g. Balance de Blancos, SCNR, Halos).
        """
        add_page_rule = f"background: {add_page_css}; border-radius: 2.5px;" if add_page_css else "background: transparent;"
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid #2e2e35;
                height: 5px;
                background: {gradient_css};
                border-radius: 2.5px;
            }}
            QSlider::sub-page:horizontal {{
                background: transparent;
            }}
            QSlider::add-page:horizontal {{
                {add_page_rule}
            }}
            QSlider::handle:horizontal {{
                background: #F4F4F5;
                width: 8px;
                height: 16px;
                margin: -6px 0;
                border-radius: 4px;
                border: 1px solid #18181B;
            }}
            QSlider::handle:horizontal:hover {{
                background: #FFFFFF;
                border: 1px solid #09090B;
            }}
            QSlider::handle:horizontal:pressed {{
                background: #E4E4E7;
                border: 1px solid #09090B;
            }}
            QSlider::handle:horizontal:disabled {{
                background: #52525B;
                border: 1px solid #27272A;
            }}
        """)

    def set_saturation_style(self) -> None:
        """
        Aplica el estilo Lightroom para deslizadores de saturación a:
        Pista que va desde tonos grises (sin saturación, izquierda) hasta los colores
        vivos del espectro hacia la saturación (derecha), donde el tirador blanco hueso
        genera la diferencia entre la capa activa luminosa y la capa velada neutra.
        """
        sat_gradient = (
            "qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #33333A, stop:0.20 #4A4A55, stop:0.38 #5E5E6E, "
            "stop:0.52 #0EA5E9, stop:0.68 #8B5CF6, stop:0.82 #EC4899, stop:0.92 #EF4444, stop:1.0 #F59E0B)"
        )
        self.set_track_gradient(sat_gradient, add_page_css="rgba(18, 18, 20, 0.45)")

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

    def set_title(self, text: str) -> None:
        self.lbl_title.setText(text)

