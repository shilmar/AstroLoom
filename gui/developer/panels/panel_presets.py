# gui/developer/panels/panel_presets.py
"""
gui/developer/panels/panel_presets.py - Panel de Gestión de Presets y Estilos de Revelado.
Proporciona acceso directo a la Galería Visual y al guardado de ajustes actuales.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Signal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.icons import get_icon


class PresetsPanel(QWidget):
    """
    Panel modular para Presets y Estilos de Revelado en DeveloperTab.
    """
    save_requested = Signal()
    manage_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sec_presets = CollapsibleSection("Presets y Estilos de Revelado", is_expanded=False, icon_name="sliders")
        w_presets = QWidget()
        presets_layout = QHBoxLayout(w_presets)
        presets_layout.setContentsMargins(6, 6, 6, 6)
        presets_layout.setSpacing(6)

        # Botón 1: Galería Visual de Presets (Destacado)
        self.btn_manage_presets = QPushButton("Galería de Presets...")
        self.btn_manage_presets.setIcon(get_icon("sparkles", color="#38bdf8", size=13))
        self.btn_manage_presets.setToolTip("Abre la galería visual con miniaturas en vivo sobre tu imagen, modulación de intensidad y previsualización en tiempo real")
        self.btn_manage_presets.setStyleSheet("""
            QPushButton {
                font-weight: 600;
                color: #e2e8f0;
                background-color: #161e2e;
                border: 1px solid #2b3954;
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                border-color: #38bdf8;
                color: #38bdf8;
                background-color: #1a253a;
            }
        """)
        self.btn_manage_presets.clicked.connect(self.manage_requested.emit)
        presets_layout.addWidget(self.btn_manage_presets, stretch=3)

        # Botón 2: Guardar Ajustes Actuales
        self.btn_save_preset = QPushButton("Guardar...")
        self.btn_save_preset.setIcon(get_icon("save", color="#94a3b8", hover_color="#38bdf8", size=12))
        self.btn_save_preset.setToolTip("Guarda los ajustes y deslizadores actuales como un nuevo preset reutilizable")
        self.btn_save_preset.setStyleSheet("""
            QPushButton {
                font-weight: 500;
                color: #cbd5e1;
                background-color: #121622;
                border: 1px solid #222a3d;
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                border-color: #38bdf8;
                color: #f8fafc;
                background-color: #182030;
            }
        """)
        self.btn_save_preset.clicked.connect(self.save_requested.emit)
        presets_layout.addWidget(self.btn_save_preset, stretch=2)

        self.sec_presets.set_content_widget(w_presets)
        main_layout.addWidget(self.sec_presets)
