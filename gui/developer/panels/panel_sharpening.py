# gui/developer/panels/panel_sharpening.py
"""
gui/developer/panels/panel_sharpening.py - Panel de Máscara de Enfoque y Nitidez.
Proporciona controles de altas frecuencias espaciales (USM), radio, umbralización de bordes
y previsualización monocroma interactiva (mediante botón o arrastre con tecla Alt).
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QApplication
)
from PySide6.QtCore import Qt, Signal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.icons import get_icon


class SharpeningPanel(QWidget):
    """
    Panel de interfaz modular para Máscara de Enfoque y Realce de Detalle.
    """
    parameter_changed = Signal()
    action_committed = Signal()
    interaction_started = Signal()
    interaction_finished = Signal()
    mask_preview_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._alt_scrubbing = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- SECCIÓN: Máscara de Enfoque y Nitidez ---
        self.sec_sharpening = CollapsibleSection(
            "Enfoque y Detalle (Wavelets / USM)",
            is_expanded=False,
            on_reset=self.reset_sharpening,
            icon_name="zap",
        )
        w_content = QWidget()
        content_layout = QVBoxLayout(w_content)
        content_layout.setContentsMargins(6, 4, 6, 6)
        content_layout.setSpacing(6)

        # 1. Cantidad (Amount)
        self.row_amount = PrecisionSliderRow(
            "Cantidad:", 0, 150, 0,
            special_zero_text="0 (Off)", unit="%", label_width=100
        )
        self.row_amount.valueChanged.connect(self._on_slider_changed)
        self.row_amount.sliderPressed.connect(self._on_amount_slider_pressed)
        self.row_amount.sliderReleased.connect(self._on_slider_released)
        content_layout.addWidget(self.row_amount)

        # 2. Radio (Radius)
        self.row_radius = PrecisionSliderRow(
            "Radio:", 5, 30, 10,
            scale=10.0, decimals=1, unit=" px", label_width=100
        )
        self.row_radius.valueChanged.connect(self._on_slider_changed)
        self.row_radius.sliderPressed.connect(self._on_slider_pressed)
        self.row_radius.sliderReleased.connect(self._on_slider_released)
        content_layout.addWidget(self.row_radius)

        # 3. Detalle (Detail / Halo suppression)
        self.row_detail = PrecisionSliderRow(
            "Detalle:", 0, 100, 25,
            unit="%", label_width=100
        )
        self.row_detail.valueChanged.connect(self._on_slider_changed)
        self.row_detail.sliderPressed.connect(self._on_slider_pressed)
        self.row_detail.sliderReleased.connect(self._on_slider_released)
        content_layout.addWidget(self.row_detail)

        # 4. Máscara (Masking) + Botón Ver Máscara
        self.btn_mask_preview = QPushButton()
        self.btn_mask_preview.setIcon(get_icon("eye", color="#94a3b8", active_color="#38bdf8", size=13))
        self.btn_mask_preview.setCheckable(True)
        self.btn_mask_preview.setFixedSize(22, 22)
        self.btn_mask_preview.setCursor(Qt.PointingHandCursor)
        self.btn_mask_preview.setToolTip("Ver máscara de bordes en blanco y negro (o mantén pulsado Alt mientras arrastras el deslizador)")
        self.btn_mask_preview.setStyleSheet("""
            QPushButton {
                background-color: #1e2430;
                border: 1px solid #334155;
                border-radius: 3px;
                padding: 1px;
            }
            QPushButton:hover {
                border-color: #38bdf8;
                background-color: #242c3d;
            }
            QPushButton:checked {
                background-color: #0284c7;
                border-color: #38bdf8;
            }
        """)
        self.btn_mask_preview.toggled.connect(self._on_mask_preview_toggled)

        self.row_masking = PrecisionSliderRow(
            "Máscara:", 0, 100, 0,
            unit="%", label_width=100,
            action_button=self.btn_mask_preview
        )
        self.row_masking.valueChanged.connect(self._on_slider_changed)
        self.row_masking.sliderPressed.connect(self._on_masking_slider_pressed)
        self.row_masking.sliderReleased.connect(self._on_slider_released)
        content_layout.addWidget(self.row_masking)

        # 5. Selector de Ámbito (Scope)
        row_scope = QHBoxLayout()
        row_scope.setSpacing(6)
        lbl_scope = QLabel("Ámbito:")
        lbl_scope.setFixedWidth(60)
        lbl_scope.setStyleSheet("color: #94a3b8; font-size: 11px;")
        row_scope.addWidget(lbl_scope)

        self.combo_scope = QComboBox()
        self.combo_scope.addItem("Solo Suelo (Recomendado)", "ground")
        self.combo_scope.addItem("Cielo sin Estrellas", "sky_starless")
        self.combo_scope.addItem("Todo el Fotograma (Global)", "global")
        self.combo_scope.currentIndexChanged.connect(self._on_scope_changed)
        row_scope.addWidget(self.combo_scope, stretch=1)
        content_layout.addLayout(row_scope)

        # --- Subsección: Ondículas Multiescala (Wavelets B3-Spline) ---
        sep_wavelets = QLabel("Realce Multiescala por Ondículas (B3-Spline):")
        sep_wavelets.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600; margin-top: 8px; margin-bottom: 2px;")
        content_layout.addWidget(sep_wavelets)

        # 6. Estructura Ondículas (-100 a +100 -> -1.00 a +1.00)
        self.row_ond_est = PrecisionSliderRow(
            "Estructura:", -100, 100, 0,
            scale=100.0, decimals=2, force_sign=True, special_zero_text="0.00", label_width=100
        )
        self.row_ond_est.setToolTip("Realce multiescala de filamentos de gas y bandas de polvo de la Vía Láctea (escalas intermedias) sin inflar estrellas ni ruido.")
        self.row_ond_est.valueChanged.connect(self._on_slider_changed)
        self.row_ond_est.sliderPressed.connect(self._on_slider_pressed)
        self.row_ond_est.sliderReleased.connect(self._on_slider_released)
        content_layout.addWidget(self.row_ond_est)

        # 7. Atenuar Fondo (0 a 100 -> 0% a 100%)
        self.row_ond_bg = PrecisionSliderRow(
            "Atenuar Fondo:", 0, 100, 0,
            unit="%", special_zero_text="0%", label_width=100
        )
        self.row_ond_bg.setToolTip("Atenúa la capa residual de frecuencia ultra-baja (pedestal y cúpula de fondo) profundizando los negros del cielo.")
        self.row_ond_bg.valueChanged.connect(self._on_slider_changed)
        self.row_ond_bg.sliderPressed.connect(self._on_slider_pressed)
        self.row_ond_bg.sliderReleased.connect(self._on_slider_released)
        content_layout.addWidget(self.row_ond_bg)

        self.sec_sharpening.set_content_widget(w_content)
        layout.addWidget(self.sec_sharpening)

    # -------------------------------------------------------------
    # Eventos de interacción y Alt-drag
    # -------------------------------------------------------------
    def _on_slider_changed(self, _val: int) -> None:
        self.parameter_changed.emit()

    def _on_scope_changed(self, _idx: int) -> None:
        self.parameter_changed.emit()
        self.action_committed.emit()

    def _on_slider_pressed(self) -> None:
        self.interaction_started.emit()

    def _on_amount_slider_pressed(self) -> None:
        self._check_alt_modifier()
        self.interaction_started.emit()

    def _on_masking_slider_pressed(self) -> None:
        self._check_alt_modifier()
        self.interaction_started.emit()

    def _check_alt_modifier(self) -> None:
        mods = QApplication.keyboardModifiers()
        if mods & Qt.AltModifier:
            self._alt_scrubbing = True
            self.mask_preview_toggled.emit(True)

    def _on_slider_released(self) -> None:
        if self._alt_scrubbing:
            self._alt_scrubbing = False
            # Si el botón toggle explícito no estaba activo, desactivar la máscara
            self.mask_preview_toggled.emit(self.btn_mask_preview.isChecked())
        self.interaction_finished.emit()
        self.action_committed.emit()

    def _on_mask_preview_toggled(self, checked: bool) -> None:
        self.mask_preview_toggled.emit(checked)
        self.parameter_changed.emit()

    # -------------------------------------------------------------
    # API de Estado
    # -------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Devuelve el estado paramétrico actual del panel."""
        return {
            "sharp_amount": self.row_amount.slider.value(),
            "sharp_radius": self.row_radius.slider.value(),
            "sharp_detail": self.row_detail.slider.value(),
            "sharp_masking": self.row_masking.slider.value(),
            "sharp_scope_idx": self.combo_scope.currentIndex(),
            "sharp_mask_preview": self.btn_mask_preview.isChecked() or self._alt_scrubbing,
            "ond_est_val": self.row_ond_est.slider.value(),
            "ond_bg_val": self.row_ond_bg.slider.value(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Actualiza los controles visuales desde un diccionario de estado."""
        self.block_signals(True)
        try:
            if "sharp_amount" in state:
                self.row_amount.slider.setValue(int(state["sharp_amount"]))
            if "sharp_radius" in state:
                self.row_radius.slider.setValue(int(state["sharp_radius"]))
            if "sharp_detail" in state:
                self.row_detail.slider.setValue(int(state["sharp_detail"]))
            if "sharp_masking" in state:
                self.row_masking.slider.setValue(int(state["sharp_masking"]))
            if "sharp_scope_idx" in state:
                idx = max(0, min(self.combo_scope.count() - 1, int(state["sharp_scope_idx"])))
                self.combo_scope.setCurrentIndex(idx)
            if "sharp_mask_preview" in state:
                self.btn_mask_preview.setChecked(bool(state["sharp_mask_preview"]))
            if "ond_est_val" in state:
                self.row_ond_est.slider.setValue(int(state["ond_est_val"]))
            if "ond_bg_val" in state:
                self.row_ond_bg.slider.setValue(int(state["ond_bg_val"]))
        finally:
            self.block_signals(False)

    def reset_sharpening(self) -> None:
        """Restablece los parámetros a sus valores por defecto neutros."""
        self.block_signals(True)
        try:
            self.row_amount.slider.setValue(0)
            self.row_radius.slider.setValue(10)
            self.row_detail.slider.setValue(25)
            self.row_masking.slider.setValue(0)
            self.combo_scope.setCurrentIndex(0)
            self.btn_mask_preview.setChecked(False)
            self._alt_scrubbing = False
            self.row_ond_est.slider.setValue(0)
            self.row_ond_bg.slider.setValue(0)
        finally:
            self.block_signals(False)

        self.parameter_changed.emit()
        self.action_committed.emit()

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.row_amount.slider, self.row_radius.slider,
            self.row_detail.slider, self.row_masking.slider,
            self.combo_scope, self.btn_mask_preview,
            getattr(self, 'row_ond_est', None) and self.row_ond_est.slider,
            getattr(self, 'row_ond_bg', None) and self.row_ond_bg.slider,
        ):
            if w is not None:
                w.blockSignals(blocked)
