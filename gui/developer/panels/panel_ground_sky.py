# gui/developer/panels/panel_ground_sky.py
"""
gui/developer/panels/panel_ground_sky.py - Panel desacoplado de Ajuste Tonal de Suelo y Máscaras de Horizonte.
Gestiona el estado de la máscara de suelo, acceso a edición interactiva y controles tonales del paisaje.
"""
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QFrame
)
from PySide6.QtCore import Qt, Signal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.icons import get_icon


class GroundSkyPanel(QWidget):
    """
    Panel modular para Ajuste Tonal Suelo (Paisaje) en DeveloperTab.
    """
    edit_mask_requested = Signal()
    go_to_layer_requested = Signal()
    reset_requested = Signal()
    ev_changed = Signal(int)
    shadows_changed = Signal(int)
    bp_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SECCIÓN 8: Ajuste Tonal Suelo (Paisaje) ---
        self.sec_gnd_tone = CollapsibleSection(
            "Ajuste Tonal Suelo (Paisaje)",
            is_expanded=False,
            on_reset=self.reset_requested.emit,
            icon_name="mountain"
        )
        w_gnd_tone = QWidget()
        gnd_tone_layout = QVBoxLayout(w_gnd_tone)
        gnd_tone_layout.setContentsMargins(6, 4, 6, 6)
        gnd_tone_layout.setSpacing(8)

        # Acceso y estado de la Máscara de Suelo
        row_mask_status = QHBoxLayout()
        self.lbl_gnd_mask_status = QLabel("Sin máscara de suelo activa")
        self.lbl_gnd_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
        row_mask_status.addWidget(self.lbl_gnd_mask_status, stretch=1)

        self.btn_edit_ground_mask = QPushButton("Crear / Editar Máscara...")
        self.btn_edit_ground_mask.setIcon(get_icon("brush", color="#80d8ff", size=13))
        self.btn_edit_ground_mask.setCursor(Qt.PointingHandCursor)
        self.btn_edit_ground_mask.setToolTip("Abre la herramienta de trazado interactivo de máscara en el visor (Atajo: M)")
        self.btn_edit_ground_mask.clicked.connect(self.edit_mask_requested.emit)
        row_mask_status.addWidget(self.btn_edit_ground_mask)
        gnd_tone_layout.addLayout(row_mask_status)

        # Tarjeta informativa y acceso directo a la Capa de Suelo
        self.card_gnd_info = QFrame()
        self.card_gnd_info.setStyleSheet("""
            QFrame {
                background-color: #152219;
                border: 1px solid #2e7d32;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout_card_gnd = QVBoxLayout(self.card_gnd_info)
        layout_card_gnd.setContentsMargins(8, 8, 8, 8)
        layout_card_gnd.setSpacing(8)

        self.lbl_gnd_info = QLabel(
            "Los ajustes tonales de la superficie terrestre (Exposición, Sombras, Punto Negro y Balance de Blancos dedicado) "
            "se gestionan ahora de forma centralizada e intuitiva en la Capa de Suelo dentro del Gestor de Capas (Sección 9)."
        )
        self.lbl_gnd_info.setWordWrap(True)
        self.lbl_gnd_info.setStyleSheet("color: #c8e6c9; font-size: 11px; line-height: 1.4;")
        layout_card_gnd.addWidget(self.lbl_gnd_info)

        self.btn_go_to_ground_layer = QPushButton("Abrir Capa de Suelo en Sección 9")
        self.btn_go_to_ground_layer.setIcon(get_icon("mountain", color="#a5d6a7", size=13))
        self.btn_go_to_ground_layer.setCursor(Qt.PointingHandCursor)
        self.btn_go_to_ground_layer.setStyleSheet("""
            QPushButton {
                background-color: #1b3822;
                color: #a5d6a7;
                border: 1px solid #43a047;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #254d2f;
                border-color: #81c784;
                color: #ffffff;
            }
        """)
        self.btn_go_to_ground_layer.clicked.connect(self.go_to_layer_requested.emit)
        layout_card_gnd.addWidget(self.btn_go_to_ground_layer)

        gnd_tone_layout.addWidget(self.card_gnd_info)

        # Controles en memoria (ocultos de la interfaz para mantener compatibilidad con presets)
        self._hidden_gnd_container = QWidget()
        self._hidden_gnd_container.setVisible(False)
        hidden_gnd_layout = QVBoxLayout(self._hidden_gnd_container)

        self.lbl_gnd_ev = QLabel("0.00 EV")
        self.slider_gnd_ev = QSlider(Qt.Horizontal)
        self.slider_gnd_ev.setRange(-200, 300)
        self.slider_gnd_ev.setValue(0)
        self.slider_gnd_ev.valueChanged.connect(self.ev_changed.emit)
        hidden_gnd_layout.addWidget(self.slider_gnd_ev)

        self.lbl_gnd_sh = QLabel("0%")
        self.slider_gnd_sh = QSlider(Qt.Horizontal)
        self.slider_gnd_sh.setRange(0, 100)
        self.slider_gnd_sh.setValue(0)
        self.slider_gnd_sh.valueChanged.connect(self.shadows_changed.emit)
        hidden_gnd_layout.addWidget(self.slider_gnd_sh)

        self.lbl_gnd_bp = QLabel("0.000")
        self.slider_gnd_bp = QSlider(Qt.Horizontal)
        self.slider_gnd_bp.setRange(-50, 100)
        self.slider_gnd_bp.setValue(0)
        self.slider_gnd_bp.valueChanged.connect(self.bp_changed.emit)
        hidden_gnd_layout.addWidget(self.slider_gnd_bp)

        self.btn_reset_gnd = QPushButton("Restablecer Tono Suelo")
        self.btn_reset_gnd.clicked.connect(self.reset_requested.emit)
        hidden_gnd_layout.addWidget(self.btn_reset_gnd)

        gnd_tone_layout.addWidget(self._hidden_gnd_container)

        self.sec_gnd_tone.set_content_widget(w_gnd_tone)
        main_layout.addWidget(self.sec_gnd_tone)

    def get_state(self) -> Dict[str, Any]:
        """Captura el estado de los controles tonales de suelo/paisaje."""
        return {
            "gnd_ev_val": self.slider_gnd_ev.value(),
            "gnd_sh_val": self.slider_gnd_sh.value(),
            "gnd_bp_val": self.slider_gnd_bp.value(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaura los controles tonales de suelo sin emisiones de señal redundantes."""
        self.block_signals(True)
        try:
            if "gnd_ev_val" in state:
                val = state["gnd_ev_val"]
                self.slider_gnd_ev.setValue(val)
                self.lbl_gnd_ev.setText(f"{val / 100.0:+.2f} EV")
            if "gnd_sh_val" in state:
                val = state["gnd_sh_val"]
                self.slider_gnd_sh.setValue(val)
                self.lbl_gnd_sh.setText(f"{val}%")
            if "gnd_bp_val" in state:
                val = state["gnd_bp_val"]
                self.slider_gnd_bp.setValue(val)
                self.lbl_gnd_bp.setText(f"{val / 1000.0:+.3f}")
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """Bloquea las señales de los controles tonales de suelo."""
        for w in (self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp):
            if w is not None:
                w.blockSignals(blocked)

