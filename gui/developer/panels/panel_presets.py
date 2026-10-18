# gui/developer/panels/panel_presets.py
"""
gui/developer/panels/panel_presets.py - Panel desacoplado de Gestión de Presets y Estilos de Revelado.
Permite seleccionar, aplicar por ámbito (Todo, Color, Detalles, Tono), guardar y administrar presets astronómicos.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.icons import get_icon
from core.developer_presets import (
    SCOPE_ALL, SCOPE_TONE, SCOPE_COLOR, SCOPE_DETAILS
)


class PresetsPanel(QWidget):
    """
    Panel modular para Presets y Estilos de Revelado en DeveloperTab.
    """
    preset_changed = Signal(int)
    apply_requested = Signal()
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
        presets_layout = QVBoxLayout(w_presets)
        presets_layout.setContentsMargins(6, 4, 6, 6)
        presets_layout.setSpacing(6)

        # Fila 1: Selector de Preset (ancho completo)
        self.combo_presets = QComboBox()
        self.combo_presets.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_presets.setMinimumContentsLength(8)
        self.combo_presets.setToolTip("Selecciona un estilo astronómico preestablecido oficial o personalizado")
        self.combo_presets.currentIndexChanged.connect(self.preset_changed.emit)
        presets_layout.addWidget(self.combo_presets)

        # Fila 2: Descripción corta del preset seleccionado
        self.lbl_preset_desc = QLabel("")
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color: #90a4ae; font-size: 11px; font-style: italic;")
        presets_layout.addWidget(self.lbl_preset_desc)

        # Fila 3: Modo de Aplicación + Botón Aplicar en la misma línea (ahorro vertical)
        row_mode_apply = QHBoxLayout()
        row_mode_apply.setSpacing(6)

        self.combo_preset_scope = QComboBox()
        self.combo_preset_scope.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_preset_scope.setMinimumContentsLength(8)
        self.combo_preset_scope.addItem("Todo (Sin estirado)", SCOPE_ALL)
        self.combo_preset_scope.addItem("Solo Color y WB", SCOPE_COLOR)
        self.combo_preset_scope.addItem("Solo Estrellas y Filtros", SCOPE_DETAILS)
        self.combo_preset_scope.addItem("Solo Contraste y Suelo", SCOPE_TONE)
        self.combo_preset_scope.setToolTip("Elige qué controles aplicar del preset (el estirado MTF y el punto negro de la imagen se respetan siempre)")
        row_mode_apply.addWidget(self.combo_preset_scope, stretch=1)

        self.btn_apply_preset = QPushButton("Aplicar")
        self.btn_apply_preset.setIcon(get_icon("sparkles", color="#ffffff", size=12))
        self.btn_apply_preset.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_apply_preset.setFixedWidth(84)
        self.btn_apply_preset.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 11px;
                background-color: #00838f;
                color: #ffffff;
                border: 1px solid #00acc1;
                border-radius: 5px;
                padding: 4px 6px;
            }
            QPushButton:hover {
                background-color: #0097a7;
                border-color: #26c6da;
            }
            QPushButton:pressed {
                background-color: #006064;
            }
            QPushButton:disabled {
                background-color: #152528;
                color: #43656b;
                border-color: #1f373b;
            }
        """)
        self.btn_apply_preset.setToolTip("Aplica el preset seleccionado a la imagen actual según el modo elegido")
        self.btn_apply_preset.clicked.connect(self.apply_requested.emit)
        row_mode_apply.addWidget(self.btn_apply_preset)
        presets_layout.addLayout(row_mode_apply)

        # Fila 4: Botones Guardar Actual y Administrar
        row_preset_btns = QHBoxLayout()
        self.btn_save_preset = QPushButton("Guardar...")
        self.btn_save_preset.setIcon(get_icon("save", color="#94a3b8", hover_color="#38bdf8", size=12))
        self.btn_save_preset.setToolTip("Guarda los ajustes y deslizadores actuales como un nuevo preset")
        self.btn_save_preset.clicked.connect(self.save_requested.emit)
        row_preset_btns.addWidget(self.btn_save_preset)

        self.btn_manage_presets = QPushButton("Gestor Presets...")
        self.btn_manage_presets.setIcon(get_icon("settings", color="#94a3b8", hover_color="#38bdf8", size=12))
        self.btn_manage_presets.setToolTip("Abre el administrador para importar, exportar, inspeccionar o eliminar presets")
        self.btn_manage_presets.clicked.connect(self.manage_requested.emit)
        row_preset_btns.addWidget(self.btn_manage_presets)
        presets_layout.addLayout(row_preset_btns)

        self.sec_presets.set_content_widget(w_presets)
        main_layout.addWidget(self.sec_presets)
