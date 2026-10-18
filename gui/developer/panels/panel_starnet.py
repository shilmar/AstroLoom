# gui/developer/panels/panel_starnet.py
"""
gui/developer/panels/panel_starnet.py - Panel modular de StarNet++ AI y Desacoplo de Estrellas.
Encapsula la configuración de stride, ejecución de StarNet, controles de detalle de capa de estrellas
y fondo cósmico, e integra el submódulo de efectos visuales (Orton y Star Diffusion).
"""
from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QFrame, QComboBox, QSlider
)
from PySide6.QtCore import Qt, Signal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.icons import get_icon
from gui.developer.panels.panel_effects import EffectsPanel


class StarNetPanel(QWidget):
    """
    Panel modular para StarNet++ AI y Desacoplo de Estrellas en DeveloperTab.
    """
    run_starnet_requested = Signal()
    jump_to_masks_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, resettable_label_factory: Optional[Callable] = None):
        super().__init__(parent)
        self._make_resettable_label = resettable_label_factory or (lambda lbl, *_: lbl)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SECCIÓN 5: StarNet++ AI - Desacoplo y Procesado de Capas ---
        self.sec_starnet = CollapsibleSection("StarNet++ AI — Desacoplo de Estrellas", is_expanded=False, icon_name="star")
        w_starnet = QWidget()
        layout_starnet = QVBoxLayout(w_starnet)
        layout_starnet.setContentsMargins(6, 4, 6, 6)
        layout_starnet.setSpacing(8)

        # 1. Disparador de Ejecución
        row_sn = QHBoxLayout()
        row_sn.addWidget(QLabel("Paso (Stride):"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(64, 512)
        self.spin_stride.setSingleStep(64)
        self.spin_stride.setValue(256)
        self.spin_stride.setFixedWidth(65)
        row_sn.addWidget(self.spin_stride)

        self.btn_starnet = QPushButton("Separar Estrellas con StarNet")
        self.btn_starnet.setIcon(get_icon("star", color="#b388ff", size=13))
        self.btn_starnet.setStyleSheet("font-weight: bold; color: #b388ff;")
        self.btn_starnet.clicked.connect(self.run_starnet_requested.emit)
        row_sn.addWidget(self.btn_starnet)
        layout_starnet.addLayout(row_sn)

        # 2. Tarjeta informativa y acceso directo al Gestor de Capas
        self.card_starnet_info = QFrame()
        self.card_starnet_info.setStyleSheet("""
            QFrame {
                background-color: #1a1e2d;
                border: 1px solid #33384c;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout_card_sn = QVBoxLayout(self.card_starnet_info)
        layout_card_sn.setContentsMargins(8, 8, 8, 8)
        layout_card_sn.setSpacing(8)

        self.lbl_starnet_info = QLabel(
            "Ejecuta StarNet para desacoplar las estrellas del fondo cósmico.\n"
            "Se generarán automáticamente las capas 'Fondo Nebular' y 'Estrellas' en el Gestor de Capas (Sección 9), "
            "donde podrás ajustar de forma contextual su presencia, claridad, contraste y reducción de ruido."
        )
        self.lbl_starnet_info.setWordWrap(True)
        self.lbl_starnet_info.setStyleSheet("color: #b0bec5; font-size: 11px; line-height: 1.4;")
        layout_card_sn.addWidget(self.lbl_starnet_info)

        self.btn_go_to_masks_starnet = QPushButton("Gestionar Capas en Columna 3")
        self.btn_go_to_masks_starnet.setIcon(get_icon("layers", color="#b388ff", size=13))
        self.btn_go_to_masks_starnet.setCursor(Qt.PointingHandCursor)
        self.btn_go_to_masks_starnet.setStyleSheet("""
            QPushButton {
                background-color: #2b1f3d;
                color: #ce93d8;
                border: 1px solid #7b1fa2;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b2854;
                border-color: #ba68c8;
                color: #ffffff;
            }
        """)
        self.btn_go_to_masks_starnet.clicked.connect(lambda: self.jump_to_masks_requested.emit("image_starless"))
        self.btn_go_to_masks_starnet.setVisible(False)
        layout_card_sn.addWidget(self.btn_go_to_masks_starnet)

        layout_starnet.addWidget(self.card_starnet_info)

        # 3. Contenedor de herramientas dependientes en memoria
        self.container_starnet_tools = QWidget()
        self.container_starnet_tools.setVisible(False)
        tools_layout = QVBoxLayout(self.container_starnet_tools)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(8)

        # Capas y Estrellas
        row_view = QHBoxLayout()
        row_view.addWidget(QLabel("Capa visible:"))
        self.combo_layer = QComboBox()
        self.combo_layer.addItems(["Compuesta (Normal)", "Solo Fondo (Starless)", "Solo Estrellas"])
        row_view.addWidget(self.combo_layer)

        row_view.addSpacing(10)
        row_view.addWidget(self._make_resettable_label(QLabel("Estrellas:"), lambda: self.slider_stars.setValue(100), "100%"))
        self.lbl_stars = QLabel("100%")
        self.lbl_stars.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_view.addWidget(self.lbl_stars)
        tools_layout.addLayout(row_view)

        self.slider_stars = QSlider(Qt.Horizontal)
        self.slider_stars.setRange(0, 200)
        self.slider_stars.setValue(100)
        tools_layout.addWidget(self.slider_stars)

        # Par 1: Claridad Fondo (izq) y Borrar Neblina (der)
        row_pair1 = QHBoxLayout()
        row_pair1.setSpacing(14)

        col_clarity = QVBoxLayout()
        col_clarity.setSpacing(4)
        row_cl_lbl = QHBoxLayout()
        row_cl_lbl.addWidget(self._make_resettable_label(QLabel("Claridad Fondo:"), lambda: self.slider_clarity.setValue(0), "0.00"))
        self.lbl_clarity = QLabel("0.00")
        self.lbl_clarity.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_cl_lbl.addWidget(self.lbl_clarity)
        col_clarity.addLayout(row_cl_lbl)
        self.slider_clarity = QSlider(Qt.Horizontal)
        self.slider_clarity.setRange(-200, 200)
        self.slider_clarity.setValue(0)
        col_clarity.addWidget(self.slider_clarity)
        row_pair1.addLayout(col_clarity)

        col_dehaze = QVBoxLayout()
        col_dehaze.setSpacing(4)
        row_dh_lbl = QHBoxLayout()
        row_dh_lbl.addWidget(self._make_resettable_label(QLabel("Borrar Neblina:"), lambda: self.slider_dehaze.setValue(0), "0.00"))
        self.lbl_dehaze = QLabel("0.00")
        self.lbl_dehaze.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_dh_lbl.addWidget(self.lbl_dehaze)
        col_dehaze.addLayout(row_dh_lbl)
        self.slider_dehaze = QSlider(Qt.Horizontal)
        self.slider_dehaze.setRange(-200, 200)
        self.slider_dehaze.setValue(0)
        col_dehaze.addWidget(self.slider_dehaze)
        row_pair1.addLayout(col_dehaze)
        tools_layout.addLayout(row_pair1)

        # Par 2: Estructura Ondículas (izq) y Atenuar Fondo (der)
        row_pair2 = QHBoxLayout()
        row_pair2.setSpacing(14)

        col_ond = QVBoxLayout()
        col_ond.setSpacing(4)
        row_ond_lbl = QHBoxLayout()
        row_ond_lbl.addWidget(self._make_resettable_label(QLabel("Estructura (Ondículas):"), lambda: self.slider_ond_est.setValue(0), "0.00"))
        self.lbl_ond_est = QLabel("0.00")
        self.lbl_ond_est.setStyleSheet("color: #b388ff; font-weight: bold;")
        row_ond_lbl.addWidget(self.lbl_ond_est)
        col_ond.addLayout(row_ond_lbl)
        self.slider_ond_est = QSlider(Qt.Horizontal)
        self.slider_ond_est.setRange(-100, 100)
        self.slider_ond_est.setValue(0)
        col_ond.addWidget(self.slider_ond_est)
        row_pair2.addLayout(col_ond)

        col_ond_bg = QVBoxLayout()
        col_ond_bg.setSpacing(4)
        row_ondbg_lbl = QHBoxLayout()
        row_ondbg_lbl.addWidget(self._make_resettable_label(QLabel("Atenuar Fondo:"), lambda: self.slider_ond_bg.setValue(0), "0%"))
        self.lbl_ond_bg = QLabel("0%")
        self.lbl_ond_bg.setStyleSheet("color: #b388ff; font-weight: bold;")
        row_ondbg_lbl.addWidget(self.lbl_ond_bg)
        col_ond_bg.addLayout(row_ondbg_lbl)
        self.slider_ond_bg = QSlider(Qt.Horizontal)
        self.slider_ond_bg.setRange(0, 100)
        self.slider_ond_bg.setValue(0)
        col_ond_bg.addWidget(self.slider_ond_bg)
        row_pair2.addLayout(col_ond_bg)
        tools_layout.addLayout(row_pair2)

        # Par 3: Contraste Starless (izq) y Reducción de Ruido (der)
        row_pair3 = QHBoxLayout()
        row_pair3.setSpacing(14)

        col_cs = QVBoxLayout()
        col_cs.setSpacing(4)
        row_cs_lbl = QHBoxLayout()
        row_cs_lbl.addWidget(self._make_resettable_label(QLabel("Contraste Fondo:"), lambda: self.slider_contrast_starless.setValue(0), "0.00"))
        self.lbl_contrast_starless = QLabel("0.00")
        self.lbl_contrast_starless.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_cs_lbl.addWidget(self.lbl_contrast_starless)
        col_cs.addLayout(row_cs_lbl)
        self.slider_contrast_starless = QSlider(Qt.Horizontal)
        self.slider_contrast_starless.setRange(-100, 100)
        self.slider_contrast_starless.setValue(0)
        col_cs.addWidget(self.slider_contrast_starless)
        row_pair3.addLayout(col_cs)

        col_dn = QVBoxLayout()
        col_dn.setSpacing(4)
        row_dn_lbl = QHBoxLayout()
        row_dn_lbl.addWidget(self._make_resettable_label(QLabel("Reducción Ruido:"), lambda: self.slider_denoise.setValue(0), "0%"))
        self.lbl_denoise = QLabel("0%")
        self.lbl_denoise.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        row_dn_lbl.addWidget(self.lbl_denoise)
        col_dn.addLayout(row_dn_lbl)
        self.slider_denoise = QSlider(Qt.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(0)
        col_dn.addWidget(self.slider_denoise)
        row_pair3.addLayout(col_dn)
        tools_layout.addLayout(row_pair3)

        # Efectos Visuales: Orton y Star Diffusion
        self.panel_effects = EffectsPanel(self)
        self.box_orton_tools = self.panel_effects.box_orton_tools
        self.slider_orton = self.panel_effects.slider_orton
        self.lbl_orton = self.panel_effects.lbl_orton
        self.btn_orton_expand = self.panel_effects.btn_orton_expand
        self.w_orton_advanced = self.panel_effects.w_orton_advanced
        self.slider_orton_radius = self.panel_effects.slider_orton_radius
        self.lbl_orton_radius = self.panel_effects.lbl_orton_radius
        self.slider_orton_threshold = self.panel_effects.slider_orton_threshold
        self.lbl_orton_threshold = self.panel_effects.lbl_orton_threshold
        self.combo_orton_tint = self.panel_effects.combo_orton_tint
        self.combo_orton_mode = self.panel_effects.combo_orton_mode

        self.box_star_diff_tools = self.panel_effects.box_star_diff_tools
        self.slider_star_diff = self.panel_effects.slider_star_diff
        self.lbl_star_diff = self.panel_effects.lbl_star_diff
        self.btn_star_diff_expand = self.panel_effects.btn_star_diff_expand
        self.w_star_diff_advanced = self.panel_effects.w_star_diff_advanced
        self.slider_star_diff_radius = self.panel_effects.slider_star_diff_radius
        self.lbl_star_diff_radius = self.panel_effects.lbl_star_diff_radius
        self.slider_star_diff_threshold = self.panel_effects.slider_star_diff_threshold
        self.lbl_star_diff_threshold = self.panel_effects.lbl_star_diff_threshold
        self.slider_star_diff_color = self.panel_effects.slider_star_diff_color
        self.lbl_star_diff_color = self.panel_effects.lbl_star_diff_color
        self.combo_star_diff_mode = self.panel_effects.combo_star_diff_mode

        tools_layout.addWidget(self.box_orton_tools)
        tools_layout.addWidget(self.box_star_diff_tools)

        layout_starnet.addWidget(self.container_starnet_tools)

        self.sec_starnet.set_content_widget(w_starnet)
        main_layout.addWidget(self.sec_starnet)

    def set_tools_enabled(self, enabled: bool):
        """Habilita o deshabilita los controles dependientes de StarNet."""
        if enabled:
            self.sec_starnet.set_status_badge("✨ Separadas", "#00e5ff")
        else:
            self.sec_starnet.set_status_badge("")

        self.btn_go_to_masks_starnet.setVisible(enabled)

        controls = [
            self.combo_layer,
            self.slider_stars,
            self.slider_contrast_starless,
            self.slider_clarity,
            self.slider_dehaze,
            self.slider_ond_est,
            self.slider_ond_bg,
            self.slider_denoise,
        ]
        for ctrl in controls:
            if ctrl is not None:
                ctrl.setEnabled(enabled)
