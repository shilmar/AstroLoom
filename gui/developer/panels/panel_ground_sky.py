# gui/developer/panels/panel_ground_sky.py
"""
gui/developer/panels/panel_ground_sky.py - Panel de Paisaje y Máscara de Suelo (Paso 1).
Gestiona el estado de la máscara de suelo, detección automática con IA y controles del paisaje.
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
    Panel modular para Paso 1: Paisaje y Máscara de Suelo (Horizonte IA).
    """
    edit_mask_requested = Signal()
    ai_mask_requested = Signal()
    overlay_toggled = Signal(bool)
    delete_mask_requested = Signal()
    go_to_layer_requested = Signal()
    reset_requested = Signal()
    ev_changed = Signal(int)
    shadows_changed = Signal(int)
    bp_changed = Signal(int)
    denoise_changed = Signal(int)
    antihalo_changed = Signal(int)
    edge_shift_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SECCIÓN 1: Paisaje y Máscara de Suelo (Horizonte IA) ---
        self.sec_gnd_tone = CollapsibleSection(
            "1. Paisaje y Máscara de Suelo (Horizonte IA)",
            is_expanded=True,
            on_reset=self.reset_requested.emit,
            icon_name="mountain"
        )
        w_gnd_tone = QWidget()
        gnd_tone_layout = QVBoxLayout(w_gnd_tone)
        gnd_tone_layout.setContentsMargins(6, 4, 6, 6)
        gnd_tone_layout.setSpacing(8)

        # 1. Estado de la Máscara de Suelo
        row_mask_status = QHBoxLayout()
        self.lbl_gnd_mask_status = QLabel("Sin máscara de suelo activa")
        self.lbl_gnd_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
        row_mask_status.addWidget(self.lbl_gnd_mask_status, stretch=1)
        gnd_tone_layout.addLayout(row_mask_status)

        # 2. Botón principal: Detección Automática con IA
        self.btn_ai_mask = QPushButton("✨ Detectar Suelo con IA")
        self.btn_ai_mask.setIcon(get_icon("sparkles", color="#ffffff", size=14))
        self.btn_ai_mask.setCursor(Qt.PointingHandCursor)
        self.btn_ai_mask.setFixedHeight(32)
        self.btn_ai_mask.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a1b9a, stop:1 #8e24aa);
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #ba68c8;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b1fa2, stop:1 #9c27b0);
                border-color: #ce93d8;
            }
            QPushButton:pressed {
                background-color: #4a148c;
            }
        """)
        self.btn_ai_mask.setToolTip("Detecta automáticamente el horizonte y suelo mediante IA ligera U²-Net con 1 solo clic")
        self.btn_ai_mask.clicked.connect(self.ai_mask_requested.emit)
        gnd_tone_layout.addWidget(self.btn_ai_mask)

        # 3. Fila de herramientas secundarias (Pintar en lienzo, Ver máscara, Borrar)
        row_actions = QHBoxLayout()
        row_actions.setSpacing(4)

        self.btn_edit_ground_mask = QPushButton("Pintar en Lienzo (M)")
        self.btn_edit_ground_mask.setIcon(get_icon("brush", color="#80d8ff", size=13))
        self.btn_edit_ground_mask.setCursor(Qt.PointingHandCursor)
        self.btn_edit_ground_mask.setStyleSheet("""
            QPushButton {
                background-color: #1a2736;
                color: #80d8ff;
                border: 1px solid #2a3b50;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #24354a;
                border-color: #38bdf8;
            }
        """)
        self.btn_edit_ground_mask.setToolTip("Abre la herramienta de trazado interactivo de máscara en el visor (Atajo: M)")
        self.btn_edit_ground_mask.clicked.connect(self.edit_mask_requested.emit)
        row_actions.addWidget(self.btn_edit_ground_mask, stretch=2)

        self.btn_overlay_mask = QPushButton("Ver")
        self.btn_overlay_mask.setIcon(get_icon("layers", color="#cfd8dc", size=13))
        self.btn_overlay_mask.setCheckable(True)
        self.btn_overlay_mask.setCursor(Qt.PointingHandCursor)
        self.btn_overlay_mask.setStyleSheet("""
            QPushButton {
                background-color: #222733;
                color: #cfd8dc;
                border: 1px solid #333d4d;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #1b5e20;
                color: #a5d6a7;
                border-color: #4caf50;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2d3545;
            }
        """)
        self.btn_overlay_mask.setToolTip("Alterna la visualización superpuesta (overlay verde/rojo) de la máscara en el lienzo")
        self.btn_overlay_mask.toggled.connect(self.overlay_toggled.emit)
        row_actions.addWidget(self.btn_overlay_mask, stretch=1)

        self.btn_delete_mask = QPushButton("Borrar")
        self.btn_delete_mask.setIcon(get_icon("trash", color="#ef9a9a", size=13))
        self.btn_delete_mask.setCursor(Qt.PointingHandCursor)
        self.btn_delete_mask.setStyleSheet("""
            QPushButton {
                background-color: #2a1b1b;
                color: #ef9a9a;
                border: 1px solid #5a2c2c;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3d2323;
                border-color: #ef5350;
            }
            QPushButton:disabled {
                background-color: #1a1c23;
                color: #4c566a;
                border: 1px solid #2a2e39;
            }
        """)
        self.btn_delete_mask.setToolTip("Elimina la máscara actual y devuelve el cielo a cobertura completa")
        self.btn_delete_mask.clicked.connect(self.delete_mask_requested.emit)
        row_actions.addWidget(self.btn_delete_mask, stretch=1)

        gnd_tone_layout.addLayout(row_actions)

        # 4. Consejo de flujo
        self.lbl_gnd_tip = QLabel(
            "Separa el suelo para protegerlo de la extracción de polución y del estirado. "
            "Si tu toma no contiene paisaje (solo cielo / cielo profundo), puedes ignorar este paso."
        )
        self.lbl_gnd_tip.setWordWrap(True)
        self.lbl_gnd_tip.setStyleSheet("color: #78909c; font-size: 10px; line-height: 1.3;")
        gnd_tone_layout.addWidget(self.lbl_gnd_tip)

        # 5. Tarjeta informativa y acceso directo a la Capa de Suelo en el gestor
        self.card_gnd_info = QFrame()
        self.card_gnd_info.setStyleSheet("""
            QFrame {
                background-color: #152219;
                border: 1px solid #2e7d32;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        layout_card_gnd = QVBoxLayout(self.card_gnd_info)
        layout_card_gnd.setContentsMargins(6, 6, 6, 6)
        layout_card_gnd.setSpacing(6)

        self.lbl_gnd_info = QLabel(
            "Una vez estirada la imagen, podrás regular la exposición (EV), sombras y reducción de ruido del terreno "
            "en la Capa de Suelo dentro del Gestor de Capas."
        )
        self.lbl_gnd_info.setWordWrap(True)
        self.lbl_gnd_info.setStyleSheet("color: #c8e6c9; font-size: 10px; line-height: 1.3;")
        layout_card_gnd.addWidget(self.lbl_gnd_info)

        self.btn_go_to_ground_layer = QPushButton("Ajustar Exposición del Suelo en Capas")
        self.btn_go_to_ground_layer.setIcon(get_icon("mountain", color="#a5d6a7", size=13))
        self.btn_go_to_ground_layer.setCursor(Qt.PointingHandCursor)
        self.btn_go_to_ground_layer.setStyleSheet("""
            QPushButton {
                background-color: #1b3822;
                color: #a5d6a7;
                border: 1px solid #43a047;
                border-radius: 4px;
                padding: 4px 10px;
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

        self.lbl_gnd_dn = QLabel("0%")
        self.slider_gnd_dn = QSlider(Qt.Horizontal)
        self.slider_gnd_dn.setRange(0, 100)
        self.slider_gnd_dn.setValue(0)
        self.slider_gnd_dn.valueChanged.connect(self.denoise_changed.emit)
        hidden_gnd_layout.addWidget(self.slider_gnd_dn)

        self.slider_gnd_antihalo = QSlider(Qt.Horizontal)
        self.slider_gnd_antihalo.setRange(0, 100)
        self.slider_gnd_antihalo.setValue(0)
        self.slider_gnd_antihalo.valueChanged.connect(self.antihalo_changed.emit)
        hidden_gnd_layout.addWidget(self.slider_gnd_antihalo)

        self.slider_gnd_edge_shift = QSlider(Qt.Horizontal)
        self.slider_gnd_edge_shift.setRange(-15, 15)
        self.slider_gnd_edge_shift.setValue(0)
        self.slider_gnd_edge_shift.valueChanged.connect(self.edge_shift_changed.emit)
        hidden_gnd_layout.addWidget(self.slider_gnd_edge_shift)

        self.btn_reset_gnd = QPushButton("Restablecer Tono Suelo")
        self.btn_reset_gnd.clicked.connect(self.reset_requested.emit)
        hidden_gnd_layout.addWidget(self.btn_reset_gnd)

        gnd_tone_layout.addWidget(self._hidden_gnd_container)

        self.sec_gnd_tone.set_content_widget(w_gnd_tone)
        main_layout.addWidget(self.sec_gnd_tone)

    def update_mask_ui_state(self, has_mask: bool, source_name: str = "", overlay_visible: bool = False):
        """Actualiza textos y botones en función de si hay máscara de suelo activa."""
        if has_mask:
            desc = f"Máscara activa ({source_name})" if source_name else "Máscara de suelo activa"
            self.lbl_gnd_mask_status.setText(f"● {desc}")
            self.lbl_gnd_mask_status.setStyleSheet("color: #a5d6a7; font-size: 11px; font-weight: bold;")
            self.btn_edit_ground_mask.setText("Refinar Pincel (M)")
            self.btn_delete_mask.setEnabled(True)
            self.btn_overlay_mask.setEnabled(True)
        else:
            self.lbl_gnd_mask_status.setText("Sin máscara de suelo (Todo cielo)")
            self.lbl_gnd_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
            self.btn_edit_ground_mask.setText("Pintar en Lienzo (M)")
            self.btn_delete_mask.setEnabled(False)
            self.btn_overlay_mask.setEnabled(False)
        self.btn_overlay_mask.blockSignals(True)
        self.btn_overlay_mask.setChecked(overlay_visible and has_mask)
        self.btn_overlay_mask.blockSignals(False)

    def set_overlay_state(self, is_active: bool):
        """Sincroniza el botón de ver máscara con el estado del lienzo."""
        self.btn_overlay_mask.blockSignals(True)
        self.btn_overlay_mask.setChecked(is_active)
        self.btn_overlay_mask.blockSignals(False)

    def get_state(self) -> Dict[str, Any]:
        """Captura el estado de los controles tonales de suelo/paisaje."""
        return {
            "gnd_ev_val": self.slider_gnd_ev.value(),
            "gnd_sh_val": self.slider_gnd_sh.value(),
            "gnd_bp_val": self.slider_gnd_bp.value(),
            "gnd_dn_val": self.slider_gnd_dn.value() if hasattr(self, 'slider_gnd_dn') else 0,
            "gnd_antihalo_val": self.slider_gnd_antihalo.value() if hasattr(self, 'slider_gnd_antihalo') else 0,
            "gnd_edge_shift_val": self.slider_gnd_edge_shift.value() if hasattr(self, 'slider_gnd_edge_shift') else 0,
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
            if "gnd_dn_val" in state and hasattr(self, 'slider_gnd_dn'):
                val = state["gnd_dn_val"]
                self.slider_gnd_dn.setValue(val)
                self.lbl_gnd_dn.setText(f"{val}%")
            if "gnd_antihalo_val" in state and hasattr(self, 'slider_gnd_antihalo'):
                self.slider_gnd_antihalo.setValue(state["gnd_antihalo_val"])
            if "gnd_edge_shift_val" in state and hasattr(self, 'slider_gnd_edge_shift'):
                self.slider_gnd_edge_shift.setValue(state["gnd_edge_shift_val"])
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """Bloquea las señales de los controles tonales de suelo."""
        for w in (self.slider_gnd_ev, self.slider_gnd_sh, self.slider_gnd_bp, getattr(self, 'slider_gnd_dn', None), getattr(self, 'slider_gnd_antihalo', None), getattr(self, 'slider_gnd_edge_shift', None)):
            if w is not None:
                w.blockSignals(blocked)
