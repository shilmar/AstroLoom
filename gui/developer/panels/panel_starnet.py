# gui/developer/panels/panel_starnet.py
"""
gui/developer/panels/panel_starnet.py - Panel modular de StarNet++ AI y Desacoplo de Estrellas.
Encapsula la configuración de stride, ejecución de StarNet, controles de detalle de capa de estrellas
y fondo cósmico, e integra el submódulo de efectos visuales (Orton y Star Diffusion).
"""
from typing import Optional, Callable, Dict, Any
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

        self.btn_denoise_expand = QPushButton()
        self.btn_denoise_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#a5d6a7", size=11))
        self.btn_denoise_expand.setCheckable(True)
        self.btn_denoise_expand.setChecked(False)
        self.btn_denoise_expand.setFixedSize(18, 18)
        self.btn_denoise_expand.setStyleSheet("""
            QPushButton { border: none; background: transparent; padding: 0px; border-radius: 3px; }
            QPushButton:hover { background: #262a3a; }
            QPushButton:checked { background: #2e3846; }
        """)
        self.btn_denoise_expand.setToolTip("Ajustes avanzados: Ruido de Color (Croma) y Máscara de Enfoque en Sombras / Protección de Bordes")
        row_dn_lbl.addWidget(self.btn_denoise_expand)

        col_dn.addLayout(row_dn_lbl)
        self.slider_denoise = QSlider(Qt.Horizontal)
        self.slider_denoise.setRange(0, 100)
        self.slider_denoise.setValue(0)
        col_dn.addWidget(self.slider_denoise)
        row_pair3.addLayout(col_dn)
        tools_layout.addLayout(row_pair3)

        # Contenedor avanzado de Reducción de Ruido (Color y Máscara de Bordes/Sombras)
        self.w_denoise_advanced = QWidget()
        self.w_denoise_advanced.setVisible(False)
        layout_dn_adv = QVBoxLayout(self.w_denoise_advanced)
        layout_dn_adv.setContentsMargins(6, 4, 6, 6)
        layout_dn_adv.setSpacing(6)
        self.w_denoise_advanced.setStyleSheet("background-color: #171a25; border-radius: 4px; border: 1px solid #232738;")

        # Fila 1: Ruido Cromático (Color)
        row_dnc = QHBoxLayout()
        row_dnc_lbl = self._make_resettable_label(QLabel("Ruido Color (Croma):"), lambda: self.slider_denoise_color.setValue(50), "50%")
        row_dnc_lbl.setStyleSheet("font-size: 11px; color: #cfd8dc;")
        row_dnc.addWidget(row_dnc_lbl)
        self.lbl_denoise_color = QLabel("50%")
        self.lbl_denoise_color.setStyleSheet("font-size: 11px; color: #80cbc4; font-weight: bold;")
        row_dnc.addWidget(self.lbl_denoise_color)
        layout_dn_adv.addLayout(row_dnc)

        self.slider_denoise_color = QSlider(Qt.Horizontal)
        self.slider_denoise_color.setRange(0, 100)
        self.slider_denoise_color.setValue(50)
        self.slider_denoise_color.valueChanged.connect(lambda v: self.lbl_denoise_color.setText(f"{v}%"))
        layout_dn_adv.addWidget(self.slider_denoise_color)

        # Fila 2: Máscara de Sombras / Protección de Bordes
        row_dnm = QHBoxLayout()
        row_dnm_lbl = self._make_resettable_label(QLabel("Máscara Bordes/Sombras:"), lambda: self.slider_denoise_mask.setValue(50), "50%")
        row_dnm_lbl.setStyleSheet("font-size: 11px; color: #cfd8dc;")
        row_dnm.addWidget(row_dnm_lbl)
        self.lbl_denoise_mask = QLabel("50%")
        self.lbl_denoise_mask.setStyleSheet("font-size: 11px; color: #80cbc4; font-weight: bold;")
        row_dnm.addWidget(self.lbl_denoise_mask)
        layout_dn_adv.addLayout(row_dnm)

        self.slider_denoise_mask = QSlider(Qt.Horizontal)
        self.slider_denoise_mask.setRange(0, 100)
        self.slider_denoise_mask.setValue(50)
        self.slider_denoise_mask.valueChanged.connect(lambda v: self.lbl_denoise_mask.setText(f"{v}%"))
        layout_dn_adv.addWidget(self.slider_denoise_mask)

        self.btn_denoise_expand.toggled.connect(self.w_denoise_advanced.setVisible)
        tools_layout.addWidget(self.w_denoise_advanced)

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
            self.slider_denoise_color,
            self.slider_denoise_mask,
        ]
        for ctrl in controls:
            if ctrl is not None:
                ctrl.setEnabled(enabled)

    def get_state(self) -> Dict[str, Any]:
        """Captura el estado de desacoplo de estrellas y fondo cósmico, incluyendo efectos."""
        state = {
            "layer_idx": self.combo_layer.currentIndex(),
            "stars_val": self.slider_stars.value(),
            "contrast_starless_val": self.slider_contrast_starless.value(),
            "clarity_val": self.slider_clarity.value(),
            "dehaze_val": self.slider_dehaze.value(),
            "ond_est_val": self.slider_ond_est.value(),
            "ond_bg_val": self.slider_ond_bg.value(),
            "denoise_val": self.slider_denoise.value(),
            "denoise_color_val": self.slider_denoise_color.value(),
            "denoise_mask_val": self.slider_denoise_mask.value(),
        }
        if hasattr(self, 'panel_effects'):
            state.update(self.panel_effects.get_state())
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaura los controles de StarNet y efectos delegados sin emisiones redundantes."""
        self.block_signals(True)
        try:
            if "layer_idx" in state:
                self.combo_layer.setCurrentIndex(state["layer_idx"])
            if "stars_val" in state:
                val = state["stars_val"]
                self.slider_stars.setValue(val)
                self.lbl_stars.setText(f"{val}%")
            if "contrast_starless_val" in state:
                val = state["contrast_starless_val"]
                self.slider_contrast_starless.setValue(val)
                self.lbl_contrast_starless.setText(f"{val / 100.0:+.2f}")
            if "clarity_val" in state:
                val = state["clarity_val"]
                self.slider_clarity.setValue(val)
                self.lbl_clarity.setText(f"{val / 100.0:+.2f}")
            if "dehaze_val" in state:
                val = state["dehaze_val"]
                self.slider_dehaze.setValue(val)
                self.lbl_dehaze.setText(f"{val / 100.0:+.2f}")
            if "ond_est_val" in state:
                val = state["ond_est_val"]
                self.slider_ond_est.setValue(val)
                self.lbl_ond_est.setText(f"{val / 100.0:+.2f}")
            if "ond_bg_val" in state:
                val = state["ond_bg_val"]
                self.slider_ond_bg.setValue(val)
                self.lbl_ond_bg.setText(f"{val}%")
            if "denoise_val" in state:
                val = state["denoise_val"]
                self.slider_denoise.setValue(val)
                self.lbl_denoise.setText(f"{val}%")
            if "denoise_color_val" in state:
                val = state["denoise_color_val"]
                self.slider_denoise_color.setValue(val)
                self.lbl_denoise_color.setText(f"{val}%")
            if "denoise_mask_val" in state:
                val = state["denoise_mask_val"]
                self.slider_denoise_mask.setValue(val)
                self.lbl_denoise_mask.setText(f"{val}%")
            if hasattr(self, 'panel_effects'):
                self.panel_effects.set_state(state)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """Bloquea las señales de los widgets dependientes de StarNet y sus efectos."""
        for w in (
            self.combo_layer, self.slider_stars,
            self.slider_contrast_starless, self.slider_clarity,
            self.slider_dehaze, self.slider_ond_est,
            self.slider_ond_bg, self.slider_denoise,
            self.slider_denoise_color, self.slider_denoise_mask
        ):
            if w is not None:
                w.blockSignals(blocked)
        if hasattr(self, 'panel_effects'):
            self.panel_effects.block_signals(blocked)

