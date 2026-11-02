"""Light Pollution and Background Gradient Reduction panel for AstroLoom Developer."""
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal as pyqtSignal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.icons import get_icon


class LightPollutionPanel(QWidget):
    """Encapsulates background gradient reduction and 2D light pollution dome controls."""

    parameter_changed = pyqtSignal()
    action_committed = pyqtSignal()
    interaction_started = pyqtSignal()
    interaction_finished = pyqtSignal()
    extract_bg_requested = pyqtSignal()
    auto_detect_lp_requested = pyqtSignal()
    center_pipette_toggled = pyqtSignal(bool)
    color_pipette_toggled = pyqtSignal(bool)
    sample_tool_toggled = pyqtSignal(bool)
    auto_grid_requested = pyqtSignal()
    clear_points_requested = pyqtSignal()
    sample_box_size_changed = pyqtSignal(int)
    rbf_mode_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.lp_reduction: float = 0.0
        self.lp_center_x: float = 0.50
        self.lp_center_y: float = 0.90
        self.lp_width: float = 0.60
        self.lp_height: float = 0.45
        self.lp_tint_type: str = "sodium"
        self.lp_custom_tint: Tuple[float, float, float] = (1.0, 0.68, 0.22)
        self.lp_tint_strength: float = 0.50
        self.lp_mode: str = "attenuate"
        self.lp_legacy_1d: bool = False
        self.rbf_mode: str = "subtraction"
        self.rbf_box_size: int = 25

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- SECCIÓN: Gradientes y Contaminación Lumínica ---
        self.sec_bg = CollapsibleSection(
            "3. Gradientes y Contaminación Lumínica",
            is_expanded=False,
            on_reset=self.reset_gradients,
            icon_name="sparkles"
        )
        w_bg = QWidget()
        layout_bg = QVBoxLayout(w_bg)
        layout_bg.setContentsMargins(6, 4, 6, 6)
        layout_bg.setSpacing(8)

        # Selector de Modelo de Fondo (Astrofotografía de Paisaje)
        row_engine = QHBoxLayout()
        row_engine.addWidget(QLabel("Modelo:"))
        self.combo_poly_degree = QComboBox()
        self.combo_poly_degree.addItems([
            "Muestreo RBF (Puntos 2D)",
            "Cuadrático (Grado 2)",
            "Lineal (Grado 1)"
        ])
        self.combo_poly_degree.setToolTip(
            "Muestreo RBF: Interpolación 2D multiescala por muestras de cielo libre (DBE / GraXpert).\n"
            "Cuadrático (Grado 2): Modela curvaturas y viñeteo suave.\n"
            "Lineal (Grado 1): Modela rampas y gradientes planos."
        )
        self.combo_poly_degree.currentIndexChanged.connect(self._on_model_changed)
        row_engine.addWidget(self.combo_poly_degree)
        self.combo_bg_engine = self.combo_poly_degree  # Alias de compatibilidad
        layout_bg.addLayout(row_engine)

        # Controles interactivos de Muestreo RBF (DBE)
        self.w_rbf_controls = QWidget()
        self.w_rbf_controls.setStyleSheet("background-color: #1e2129; border-radius: 6px;")
        layout_rbf = QVBoxLayout(self.w_rbf_controls)
        layout_rbf.setContentsMargins(6, 6, 6, 6)
        layout_rbf.setSpacing(6)

        # Fila 1: Botones de herramientas de puntos
        row_rbf_btns = QHBoxLayout()
        row_rbf_btns.setSpacing(4)

        self.btn_sample_tool = QPushButton("Puntos")
        self.btn_sample_tool.setIcon(get_icon("crosshair", color="#94a3b8", active_color="#80d8ff", size=12))
        self.btn_sample_tool.setCheckable(True)
        self.btn_sample_tool.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #80d8ff; border: 1px solid #80d8ff; }"
        )
        self.btn_sample_tool.setToolTip("Activa la edición interactiva en el lienzo:\n- Clic izquierdo: Añadir o arrastrar punto.\n- Clic derecho: Eliminar punto.")
        self.btn_sample_tool.toggled.connect(self.sample_tool_toggled.emit)
        row_rbf_btns.addWidget(self.btn_sample_tool)

        self.btn_auto_grid = QPushButton("Rejilla Auto")
        self.btn_auto_grid.setIcon(get_icon("sparkles", color="#80d8ff", size=12))
        self.btn_auto_grid.setStyleSheet("font-size: 11px; padding: 4px; font-weight: bold; color: #80d8ff;")
        self.btn_auto_grid.setToolTip("Genera automáticamente una malla de muestras en el cielo excluyendo suelo y estrellas.")
        self.btn_auto_grid.clicked.connect(self.auto_grid_requested.emit)
        row_rbf_btns.addWidget(self.btn_auto_grid)

        self.btn_clear_points = QPushButton("Borrar")
        self.btn_clear_points.setStyleSheet("font-size: 11px; padding: 4px; color: #ffb74d;")
        self.btn_clear_points.setToolTip("Elimina todos los puntos de muestreo del lienzo.")
        self.btn_clear_points.clicked.connect(self.clear_points_requested.emit)
        row_rbf_btns.addWidget(self.btn_clear_points)

        layout_rbf.addLayout(row_rbf_btns)

        # Fila 2: Contador de puntos y modo (Sustracción / División)
        row_rbf_mode = QHBoxLayout()
        self.lbl_points_count = QLabel("Puntos: 0")
        self.lbl_points_count.setStyleSheet("font-size: 11px; font-weight: bold; color: #80d8ff;")
        row_rbf_mode.addWidget(self.lbl_points_count)

        self.combo_rbf_mode = QComboBox()
        self.combo_rbf_mode.addItem("Sustracción", "subtraction")
        self.combo_rbf_mode.addItem("División", "division")
        self.combo_rbf_mode.setToolTip("Sustracción: Para resplandor aditivo (polución/luna).\nDivisión: Para viñeteo óptico o gradientes multiplicativos.")
        self.combo_rbf_mode.currentIndexChanged.connect(self._on_rbf_mode_changed)
        row_rbf_mode.addWidget(self.combo_rbf_mode, stretch=1)
        layout_rbf.addLayout(row_rbf_mode)

        # Fila 3: Tamaño de caja de muestreo
        self.row_box_size = PrecisionSliderRow("Caja Muestra:", 10, 60, 25, unit="px", label_width=115)
        self.slider_box_size = self.row_box_size.slider
        self.slider_box_size.valueChanged.connect(self._on_box_size_changed)
        layout_rbf.addWidget(self.row_box_size)

        layout_bg.addWidget(self.w_rbf_controls)

        self.btn_extract_bg = QPushButton("Eliminar Gradientes")
        self.btn_extract_bg.setIcon(get_icon("sparkles", color="#80d8ff", size=13))
        self.btn_extract_bg.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_extract_bg.setToolTip("Ajusta una superficie analítica inmune a la silueta del suelo y neutraliza el gradiente.")
        self.btn_extract_bg.clicked.connect(self.extract_bg_requested.emit)
        layout_bg.addWidget(self.btn_extract_bg)

        # Cúpula de luz (Contaminación Lumínica) integrada
        self.row_lp = PrecisionSliderRow("Atenuar Cúpula:", 0, 100, 0, unit="%", label_width=115)
        self.slider_lp = self.row_lp.slider
        self.lbl_lp = self.row_lp.lbl_value
        self.slider_lp.valueChanged.connect(self._on_lp_changed)
        self.slider_lp.sliderPressed.connect(self.interaction_started.emit)
        self.slider_lp.sliderReleased.connect(self._on_slider_released)
        layout_bg.addWidget(self.row_lp)

        # Botonera de herramientas de Cúpula 2D
        row_lp_tools = QHBoxLayout()
        row_lp_tools.setSpacing(4)
        self.btn_lp_expand = QPushButton("Cúpula 2D")
        self.btn_lp_expand.setIcon(get_icon("settings", color="#94a3b8", active_color="#ffd54f", size=12))
        self.btn_lp_expand.setCheckable(True)
        self.btn_lp_expand.setChecked(False)
        self.btn_lp_expand.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #ffd54f; border: 1px solid #ffd54f; }"
        )
        self.btn_lp_expand.setToolTip("Despliega los controles analíticos avanzados de posición 2D, forma y tinte cromático de la cúpula.")
        self.btn_lp_expand.toggled.connect(self._on_toggle_lp_expand)
        row_lp_tools.addWidget(self.btn_lp_expand)

        self.btn_lp_auto = QPushButton("Auto-Detectar")
        self.btn_lp_auto.setIcon(get_icon("sparkles", color="#80d8ff", size=12))
        self.btn_lp_auto.setStyleSheet("font-size: 11px; padding: 4px; font-weight: bold; color: #80d8ff;")
        self.btn_lp_auto.setToolTip("Analiza el resplandor difuso del cielo y sitúa automáticamente el centro de la cúpula urbana.")
        self.btn_lp_auto.clicked.connect(self.auto_detect_lp_requested.emit)
        row_lp_tools.addWidget(self.btn_lp_auto)

        self.btn_lp_center_pipette = QPushButton("Pinchar Centro")
        self.btn_lp_center_pipette.setIcon(get_icon("crosshair", color="#94a3b8", active_color="#ffd54f", size=12))
        self.btn_lp_center_pipette.setCheckable(True)
        self.btn_lp_center_pipette.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #ffd54f; border: 1px solid #ffd54f; }"
        )
        self.btn_lp_center_pipette.setToolTip("Activa la pipeta interactiva en el lienzo para situar el foco de la cúpula de polución con un clic.")
        self.btn_lp_center_pipette.toggled.connect(self.center_pipette_toggled.emit)
        row_lp_tools.addWidget(self.btn_lp_center_pipette)
        layout_bg.addLayout(row_lp_tools)

        # Contenedor desplegable de parámetros 2D
        self.w_lp_params = QWidget()
        self.w_lp_params.setVisible(False)
        self.w_lp_params.setStyleSheet("background-color: #1e2129; border-radius: 6px;")
        layout_lp_params = QVBoxLayout(self.w_lp_params)
        layout_lp_params.setContentsMargins(8, 6, 8, 8)
        layout_lp_params.setSpacing(6)

        # Fila 1: Centro X y Centro Y
        self.row_lp_cx = PrecisionSliderRow("Centro X:", 0, 100, 50, scale=100.0, decimals=2, label_width=115)
        self.slider_lp_cx = self.row_lp_cx.slider
        self.spin_lp_cx = self.row_lp_cx.spin
        self.lbl_lp_cx = self.row_lp_cx.lbl_value
        self.slider_lp_cx.valueChanged.connect(self._on_slider_lp_cx_changed)
        self.spin_lp_cx.valueChanged.connect(self._on_spin_lp_cx_changed)
        self.slider_lp_cx.sliderPressed.connect(self.interaction_started.emit)
        self.slider_lp_cx.sliderReleased.connect(self._on_slider_released)
        layout_lp_params.addWidget(self.row_lp_cx)

        self.row_lp_cy = PrecisionSliderRow("Centro Y:", 0, 100, 90, scale=100.0, decimals=2, label_width=115)
        self.slider_lp_cy = self.row_lp_cy.slider
        self.spin_lp_cy = self.row_lp_cy.spin
        self.lbl_lp_cy = self.row_lp_cy.lbl_value
        self.slider_lp_cy.valueChanged.connect(self._on_slider_lp_cy_changed)
        self.spin_lp_cy.valueChanged.connect(self._on_spin_lp_cy_changed)
        self.slider_lp_cy.sliderPressed.connect(self.interaction_started.emit)
        self.slider_lp_cy.sliderReleased.connect(self._on_slider_released)
        layout_lp_params.addWidget(self.row_lp_cy)

        # Fila 2: Ancho W y Alto H
        self.row_lp_w = PrecisionSliderRow("Ancho:", 5, 200, 60, scale=100.0, decimals=2, label_width=115)
        self.slider_lp_w = self.row_lp_w.slider
        self.spin_lp_w = self.row_lp_w.spin
        self.lbl_lp_w = self.row_lp_w.lbl_value
        self.slider_lp_w.valueChanged.connect(self._on_slider_lp_w_changed)
        self.spin_lp_w.valueChanged.connect(self._on_spin_lp_w_changed)
        self.slider_lp_w.sliderPressed.connect(self.interaction_started.emit)
        self.slider_lp_w.sliderReleased.connect(self._on_slider_released)
        layout_lp_params.addWidget(self.row_lp_w)

        self.row_lp_h = PrecisionSliderRow("Alto:", 5, 150, 45, scale=100.0, decimals=2, label_width=115)
        self.slider_lp_h = self.row_lp_h.slider
        self.spin_lp_h = self.row_lp_h.spin
        self.lbl_lp_h = self.row_lp_h.lbl_value
        self.slider_lp_h.valueChanged.connect(self._on_slider_lp_h_changed)
        self.spin_lp_h.valueChanged.connect(self._on_spin_lp_h_changed)
        self.slider_lp_h.sliderPressed.connect(self.interaction_started.emit)
        self.slider_lp_h.sliderReleased.connect(self._on_slider_released)
        layout_lp_params.addWidget(self.row_lp_h)

        # Fila 3: Tinte cromático y botón cuentagotas de color
        row_lp_tint = QHBoxLayout()
        row_lp_tint.addWidget(QLabel("Tinte:"))
        self.combo_lp_tint = QComboBox()
        self.combo_lp_tint.addItem("Vapor de Sodio (Ámbar)", "sodium")
        self.combo_lp_tint.addItem("LED Urbano Cálido", "warm_led")
        self.combo_lp_tint.addItem("LED Frío / Mercurio", "cold_led")
        self.combo_lp_tint.addItem("Atenuación Neutra", "neutral")
        self.combo_lp_tint.addItem("Personalizado", "custom")
        self.combo_lp_tint.currentIndexChanged.connect(self._on_lp_tint_combo_changed)
        row_lp_tint.addWidget(self.combo_lp_tint, stretch=1)

        self.btn_lp_swatch = QPushButton()
        self.btn_lp_swatch.setFixedSize(22, 22)
        self.btn_lp_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_lp_swatch.setToolTip("Muestra visual del color del tinte. Clic para seleccionar manualmente.")
        self.btn_lp_swatch.clicked.connect(self._on_swatch_clicked)
        self.lbl_tint_swatch = self.btn_lp_swatch
        row_lp_tint.addWidget(self.btn_lp_swatch)

        self.btn_lp_color_pipette = QPushButton("Muestrear Color")
        self.btn_lp_color_pipette.setIcon(get_icon("pipette", color="#94a3b8", active_color="#ffd54f", size=12))
        self.btn_lp_color_pipette.setCheckable(True)
        self.btn_lp_color_pipette.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px; font-weight: bold; border-radius: 4px; background: #262933; color: #cfd8dc; }"
            "QPushButton:checked { background: #3b4252; color: #ffd54f; border: 1px solid #ffd54f; }"
        )
        self.btn_lp_color_pipette.setToolTip("Muestrea el tinte cromático exacto pinchando sobre el resplandor de la cúpula en el lienzo.")
        self.btn_lp_color_pipette.toggled.connect(self.color_pipette_toggled.emit)
        row_lp_tint.addWidget(self.btn_lp_color_pipette)
        layout_lp_params.addLayout(row_lp_tint)

        # Fila 4: Neutralización cromática
        self.row_lp_tint_strength = PrecisionSliderRow("Neutralización:", 0, 100, 50, unit="%", label_width=115)
        self.slider_lp_tint_strength = self.row_lp_tint_strength.slider
        self.lbl_lp_tint_strength = self.row_lp_tint_strength.lbl_value
        self.slider_lp_tint_strength.valueChanged.connect(self._on_lp_tint_strength_changed)
        self.slider_lp_tint_strength.sliderPressed.connect(self.interaction_started.emit)
        self.slider_lp_tint_strength.sliderReleased.connect(self._on_slider_released)
        layout_lp_params.addWidget(self.row_lp_tint_strength)

        # Fila 5: Modo de reducción y Compatibilidad 1D
        row_lp_mode = QHBoxLayout()
        row_lp_mode.addWidget(QLabel("Modo:"))
        self.combo_lp_mode = QComboBox()
        self.combo_lp_mode.addItem("Atenuación Suave", "attenuate")
        self.combo_lp_mode.addItem("Sustracción Física", "subtractive")
        self.combo_lp_mode.currentIndexChanged.connect(self._on_lp_mode_combo_changed)
        row_lp_mode.addWidget(self.combo_lp_mode, stretch=1)

        self.chk_lp_legacy_1d = QCheckBox("Modo Clásico 1D")
        self.chk_lp_legacy_1d.setToolTip("Aplica un gradiente lineal vertical clásico en lugar del modelo elíptico 2D.")
        self.chk_lp_legacy_1d.toggled.connect(self._on_lp_legacy_1d_toggled)
        row_lp_mode.addWidget(self.chk_lp_legacy_1d)
        layout_lp_params.addLayout(row_lp_mode)

        # Botón de Restablecer Cúpula 2D
        btn_reset_lp = QPushButton("Restablecer Cúpula")
        btn_reset_lp.setStyleSheet("font-size: 11px; padding: 4px; color: #ffb74d;")
        btn_reset_lp.clicked.connect(self.reset_gradients)
        layout_lp_params.addWidget(btn_reset_lp)

        layout_bg.addWidget(self.w_lp_params)

        self._update_swatch()
        self.sec_bg.set_content_widget(w_bg)
        layout.addWidget(self.sec_bg)

    # -------------------------------------------------------------
    # Sliders & Value synchronization
    # -------------------------------------------------------------
    def _on_model_changed(self, idx: int) -> None:
        is_rbf = ("Muestreo" in self.combo_poly_degree.currentText())
        self.w_rbf_controls.setVisible(is_rbf)
        if not is_rbf and self.btn_sample_tool.isChecked():
            self.btn_sample_tool.setChecked(False)

    def _on_rbf_mode_changed(self, idx: int) -> None:
        self.rbf_mode = self.combo_rbf_mode.currentData() or "subtraction"
        self.rbf_mode_changed.emit(self.rbf_mode)

    def _on_box_size_changed(self, val: int) -> None:
        self.rbf_box_size = int(val)
        self.sample_box_size_changed.emit(self.rbf_box_size)

    def set_points_count(self, count: int) -> None:
        self.lbl_points_count.setText(f"Puntos: {count}")

    def set_sample_tool_active(self, active: bool) -> None:
        self.btn_sample_tool.blockSignals(True)
        self.btn_sample_tool.setChecked(active)
        self.btn_sample_tool.blockSignals(False)

    def _on_slider_released(self) -> None:
        self.interaction_finished.emit()
        self.action_committed.emit()

    def _on_toggle_lp_expand(self, checked: bool) -> None:
        self.w_lp_params.setVisible(checked)

    def _on_lp_changed(self, val: int) -> None:
        red = val / 100.0
        if abs(self.lp_reduction - red) < 1e-4:
            return
        self.lp_reduction = red
        self.lbl_lp.setText(f"{val}%")
        self.parameter_changed.emit()

    def _on_slider_lp_cx_changed(self, val: int) -> None:
        cx = val / 100.0
        if abs(self.lp_center_x - cx) < 1e-4:
            return
        self.lp_center_x = cx
        self.spin_lp_cx.blockSignals(True)
        self.spin_lp_cx.setValue(cx)
        self.spin_lp_cx.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_lp_cx_changed(self, val: float) -> None:
        cx = float(val)
        if abs(self.lp_center_x - cx) < 1e-4:
            return
        self.lp_center_x = cx
        self.slider_lp_cx.blockSignals(True)
        self.slider_lp_cx.setValue(int(round(cx * 100.0)))
        self.slider_lp_cx.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_lp_cy_changed(self, val: int) -> None:
        cy = val / 100.0
        if abs(self.lp_center_y - cy) < 1e-4:
            return
        self.lp_center_y = cy
        self.spin_lp_cy.blockSignals(True)
        self.spin_lp_cy.setValue(cy)
        self.spin_lp_cy.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_lp_cy_changed(self, val: float) -> None:
        cy = float(val)
        if abs(self.lp_center_y - cy) < 1e-4:
            return
        self.lp_center_y = cy
        self.slider_lp_cy.blockSignals(True)
        self.slider_lp_cy.setValue(int(round(cy * 100.0)))
        self.slider_lp_cy.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_lp_w_changed(self, val: int) -> None:
        w = val / 100.0
        if abs(self.lp_width - w) < 1e-4:
            return
        self.lp_width = w
        self.spin_lp_w.blockSignals(True)
        self.spin_lp_w.setValue(w)
        self.spin_lp_w.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_lp_w_changed(self, val: float) -> None:
        w = float(val)
        if abs(self.lp_width - w) < 1e-4:
            return
        self.lp_width = w
        self.slider_lp_w.blockSignals(True)
        self.slider_lp_w.setValue(int(round(w * 100.0)))
        self.slider_lp_w.blockSignals(False)
        self.parameter_changed.emit()

    def _on_slider_lp_h_changed(self, val: int) -> None:
        h = val / 100.0
        if abs(self.lp_height - h) < 1e-4:
            return
        self.lp_height = h
        self.spin_lp_h.blockSignals(True)
        self.spin_lp_h.setValue(h)
        self.spin_lp_h.blockSignals(False)
        self.parameter_changed.emit()

    def _on_spin_lp_h_changed(self, val: float) -> None:
        h = float(val)
        if abs(self.lp_height - h) < 1e-4:
            return
        self.lp_height = h
        self.slider_lp_h.blockSignals(True)
        self.slider_lp_h.setValue(int(round(h * 100.0)))
        self.slider_lp_h.blockSignals(False)
        self.parameter_changed.emit()

    def _update_swatch(self) -> None:
        if not hasattr(self, "btn_lp_swatch"):
            return
        from core.light_pollution import LP_PRESET_TINTS
        tint_type = self.combo_lp_tint.currentData() or self.lp_tint_type
        if tint_type == "custom":
            rgb = self.lp_custom_tint
        else:
            rgb = LP_PRESET_TINTS.get(tint_type, (1.0, 1.0, 1.0))
        r_u8 = int(np.clip(rgb[0] * 255.0, 0, 255))
        g_u8 = int(np.clip(rgb[1] * 255.0, 0, 255))
        b_u8 = int(np.clip(rgb[2] * 255.0, 0, 255))
        self.btn_lp_swatch.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r_u8}, {g_u8}, {b_u8}); border: 1px solid #90a4ae; border-radius: 4px; }}"
            f"QPushButton:hover {{ border: 1.5px solid #80d8ff; }}"
        )
        self.btn_lp_swatch.setToolTip(f"Tinte activo: RGB({r_u8}, {g_u8}, {b_u8}). Clic para elegir color.")

    def _on_swatch_clicked(self) -> None:
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        from core.light_pollution import LP_PRESET_TINTS
        tint_type = self.combo_lp_tint.currentData() or self.lp_tint_type
        if tint_type == "custom":
            rgb = self.lp_custom_tint
        else:
            rgb = LP_PRESET_TINTS.get(tint_type, (1.0, 1.0, 1.0))
        cur_col = QColor(int(np.clip(rgb[0] * 255.0, 0, 255)), int(np.clip(rgb[1] * 255.0, 0, 255)), int(np.clip(rgb[2] * 255.0, 0, 255)))
        col = QColorDialog.getColor(cur_col, self, "Seleccionar Tinte de Contaminación Lumínica")
        if col.isValid():
            r, g, b = col.redF(), col.greenF(), col.blueF()
            max_c = max(1e-4, max(r, g, b))
            self.set_custom_tint(r / max_c, g / max_c, b / max_c)
            self.action_committed.emit()

    def _on_lp_tint_combo_changed(self, idx: int) -> None:
        tint_type = self.combo_lp_tint.currentData() or "sodium"
        if self.lp_tint_type == tint_type:
            return
        self.lp_tint_type = tint_type
        self._update_swatch()
        self.parameter_changed.emit()
        self.action_committed.emit()

    def _on_lp_tint_strength_changed(self, val: int) -> None:
        str_val = val / 100.0
        if abs(self.lp_tint_strength - str_val) < 1e-4:
            return
        self.lp_tint_strength = str_val
        self.lbl_lp_tint_strength.setText(f"{val}%")
        self.parameter_changed.emit()

    def _on_lp_mode_combo_changed(self, idx: int) -> None:
        mode = self.combo_lp_mode.currentData() or "attenuate"
        if self.lp_mode == mode:
            return
        self.lp_mode = mode
        self.parameter_changed.emit()
        self.action_committed.emit()

    def _on_lp_legacy_1d_toggled(self, checked: bool) -> None:
        if self.lp_legacy_1d == checked:
            return
        self.lp_legacy_1d = checked
        self.parameter_changed.emit()
        self.action_committed.emit()

    # -------------------------------------------------------------
    # Public controls API
    # -------------------------------------------------------------
    def reset_gradients(self) -> None:
        self.block_signals(True)
        try:
            self.slider_lp.setValue(0)
            self.lbl_lp.setText("0%")
            self.slider_lp_cx.setValue(50)
            self.spin_lp_cx.setValue(0.50)
            self.slider_lp_cy.setValue(90)
            self.spin_lp_cy.setValue(0.90)
            self.slider_lp_w.setValue(60)
            self.spin_lp_w.setValue(0.60)
            self.slider_lp_h.setValue(45)
            self.spin_lp_h.setValue(0.45)
            self.combo_lp_tint.setCurrentIndex(0)
            self.slider_lp_tint_strength.setValue(50)
            self.lbl_lp_tint_strength.setText("50%")
            self.combo_lp_mode.setCurrentIndex(0)
            self.chk_lp_legacy_1d.setChecked(False)

            self.lp_reduction = 0.0
            self.lp_center_x = 0.50
            self.lp_center_y = 0.90
            self.lp_width = 0.60
            self.lp_height = 0.45
            self.lp_tint_type = "sodium"
            self.lp_custom_tint = (1.0, 0.68, 0.22)
            self.lp_tint_strength = 0.50
            self.lp_mode = "attenuate"
            self.lp_legacy_1d = False
        finally:
            self.block_signals(False)

        self._update_swatch()
        self.parameter_changed.emit()
        self.action_committed.emit()

    def set_center(self, cx: float, cy: float) -> None:
        """Sets the dome center programmatically (e.g. from canvas click or auto-detect)."""
        self.lp_center_x = float(cx)
        self.lp_center_y = float(cy)
        self.spin_lp_cx.setValue(cx)
        self.slider_lp_cx.setValue(int(round(cx * 100.0)))
        self.spin_lp_cy.setValue(cy)
        self.slider_lp_cy.setValue(int(round(cy * 100.0)))
        self.parameter_changed.emit()

    def set_custom_tint(self, r: float, g: float, b: float) -> None:
        """Sets custom sampled tint and activates custom mode."""
        self.lp_custom_tint = (float(r), float(g), float(b))
        self.lp_tint_type = "custom"
        custom_idx = self.combo_lp_tint.findData("custom")
        if custom_idx >= 0:
            self.combo_lp_tint.blockSignals(True)
            self.combo_lp_tint.setCurrentIndex(custom_idx)
            self.combo_lp_tint.blockSignals(False)
        self._update_swatch()
        self.parameter_changed.emit()

    def get_state(self) -> Dict[str, Any]:
        return {
            "lp_val": self.slider_lp.value(),
            "lp_cx": float(self.spin_lp_cx.value()),
            "lp_cy": float(self.spin_lp_cy.value()),
            "lp_w": float(self.spin_lp_w.value()),
            "lp_h": float(self.spin_lp_h.value()),
            "lp_tint_idx": self.combo_lp_tint.currentIndex(),
            "lp_custom_tint": self.lp_custom_tint,
            "lp_tint_strength": self.slider_lp_tint_strength.value(),
            "lp_mode_idx": self.combo_lp_mode.currentIndex(),
            "lp_legacy_1d": self.chk_lp_legacy_1d.isChecked(),
            "bg_model_idx": self.combo_poly_degree.currentIndex(),
            "rbf_mode_idx": self.combo_rbf_mode.currentIndex(),
            "rbf_box_size": self.rbf_box_size,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.block_signals(True)
        try:
            if "bg_model_idx" in state:
                idx = int(state["bg_model_idx"])
                if 0 <= idx < self.combo_poly_degree.count():
                    self.combo_poly_degree.setCurrentIndex(idx)
                    self._on_model_changed(idx)
            if "rbf_mode_idx" in state:
                idx = int(state["rbf_mode_idx"])
                if 0 <= idx < self.combo_rbf_mode.count():
                    self.combo_rbf_mode.setCurrentIndex(idx)
                    self.rbf_mode = self.combo_rbf_mode.itemData(idx) or "subtraction"
            if "rbf_box_size" in state:
                val = int(state["rbf_box_size"])
                self.rbf_box_size = val
                self.slider_box_size.setValue(val)
            if "lp_val" in state:
                val = state["lp_val"]
                self.slider_lp.setValue(val)
                self.lp_reduction = val / 100.0
                self.lbl_lp.setText(f"{val}%")
            if "lp_cx" in state:
                val = float(state["lp_cx"])
                self.lp_center_x = val
                self.spin_lp_cx.setValue(val)
                self.slider_lp_cx.setValue(int(round(val * 100.0)))
            if "lp_cy" in state:
                val = float(state["lp_cy"])
                self.lp_center_y = val
                self.spin_lp_cy.setValue(val)
                self.slider_lp_cy.setValue(int(round(val * 100.0)))
            if "lp_w" in state:
                val = float(state["lp_w"])
                self.lp_width = val
                self.spin_lp_w.setValue(val)
                self.slider_lp_w.setValue(int(round(val * 100.0)))
            if "lp_h" in state:
                val = float(state["lp_h"])
                self.lp_height = val
                self.spin_lp_h.setValue(val)
                self.slider_lp_h.setValue(int(round(val * 100.0)))
            if "lp_tint_idx" in state:
                idx = state["lp_tint_idx"]
                self.combo_lp_tint.setCurrentIndex(idx)
                self.lp_tint_type = self.combo_lp_tint.itemData(idx) or "sodium"
            if "lp_custom_tint" in state:
                self.lp_custom_tint = tuple(state["lp_custom_tint"])
            if "lp_tint_strength" in state:
                val = state["lp_tint_strength"]
                self.slider_lp_tint_strength.setValue(val)
                self.lp_tint_strength = val / 100.0
                self.lbl_lp_tint_strength.setText(f"{val}%")
            if "lp_mode_idx" in state:
                idx = state["lp_mode_idx"]
                self.combo_lp_mode.setCurrentIndex(idx)
                self.lp_mode = self.combo_lp_mode.itemData(idx) or "attenuate"
            if "lp_legacy_1d" in state:
                val = bool(state["lp_legacy_1d"])
                self.chk_lp_legacy_1d.setChecked(val)
                self.lp_legacy_1d = val
        finally:
            self.block_signals(False)
        self._update_swatch()

    def block_signals(self, blocked: bool) -> None:
        for w in (
            self.slider_lp, self.slider_lp_cx, self.spin_lp_cx,
            self.slider_lp_cy, self.spin_lp_cy, self.slider_lp_w,
            self.spin_lp_w, self.slider_lp_h, self.spin_lp_h,
            self.combo_lp_tint, self.slider_lp_tint_strength,
            self.combo_lp_mode, self.chk_lp_legacy_1d,
            self.combo_poly_degree, self.combo_rbf_mode, self.slider_box_size
        ):
            if w is not None:
                w.blockSignals(blocked)
