"""gui/developer/panels/panel_dust.py - Sensor Dust Spot Detection, Spot Healing and Visualization panel."""
from __future__ import annotations

from typing import Optional, Callable, List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox, QFrame
)
from PySide6.QtCore import Qt, Signal as pyqtSignal

from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.icons import get_icon
from gui.theme import STYLE_RUN_IDLE, STYLE_TRANSFER_BTN


class DustPanel(QWidget):
    """Panel para Detección y Curación Automática de Manchas de Polvo del Sensor.

    Inspirado en la arquitectura clásica y no destructiva de Lightcraft:
    - Búsqueda automática basada en gradiente de luminancia y supresión por textura.
    - Curación continua por aproximación de Poisson (convolución normalizada) / inpainting.
    - Modo de diagnóstico de manchas por paso alto (Visualize Spots, atajo A).
    """

    # Señales reactivas
    parameter_changed = pyqtSignal()
    auto_detect_requested = pyqtSignal(int)          # sensitivity (0..100)
    visualize_toggled = pyqtSignal(bool, int)        # is_active, threshold (0..100)
    spot_mode_toggled = pyqtSignal(bool)             # modo colocación manual en canvas
    clear_spots_requested = pyqtSignal()
    action_committed = pyqtSignal()
    interaction_started = pyqtSignal()
    interaction_finished = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._sensitivity: int = 50
        self._visualize: bool = False
        self._vis_threshold: int = 50
        self._mode: str = "heal"
        self._spot_count: int = 0
        self._is_spot_mode_active: bool = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Sección colapsable
        self.sec_dust = CollapsibleSection(
            "Corrección de Manchas y Polvo",
            is_expanded=False,
            on_reset=self.reset_dust,
            icon_name="target",
        )

        w_content = QWidget()
        c_layout = QVBoxLayout(w_content)
        c_layout.setContentsMargins(6, 4, 6, 6)
        c_layout.setSpacing(8)

        # 1. Slider de Sensibilidad de Detección
        self.row_sensitivity = PrecisionSliderRow(
            "Sensibilidad:", 0, 100, 50, scale=1.0, decimals=0, label_width=110
        )
        self.slider_sens = self.row_sensitivity.slider
        self.spin_sens = self.row_sensitivity.spin
        self.lbl_sens = self.row_sensitivity.lbl_value
        self.slider_sens.setToolTip("Mayor sensibilidad detecta manchas más tenues y difusas en el cielo.")
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        self.spin_sens.valueChanged.connect(self._on_spin_sens_changed)
        self.slider_sens.sliderPressed.connect(self.interaction_started.emit)
        self.slider_sens.sliderReleased.connect(self._on_slider_released)
        c_layout.addWidget(self.row_sensitivity)

        # 2. Selector de Modo de Curación
        row_mode = QHBoxLayout()
        lbl_m = QLabel("Modo:")
        lbl_m.setStyleSheet("color: #c0caf5; font-size: 11px;")
        lbl_m.setFixedWidth(60)
        row_mode.addWidget(lbl_m)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Curar (Tonal Seamless)",
            "Inpainting Puro",
            "Clonar (Directo)",
        ])
        self.combo_mode.setToolTip(
            "Curar: mezcla de gradiente tonal y textura sin costuras (Lightcraft/Poisson).\n"
            "Inpainting: reconstrucción fluida sin parche donante.\n"
            "Clonar: copia directa de píxeles desde el parche donante."
        )
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        row_mode.addWidget(self.combo_mode)
        c_layout.addLayout(row_mode)

        # 3. Botón de Búsqueda Automática
        self.btn_auto_detect = QPushButton("Buscar Manchas de Polvo")
        self.btn_auto_detect.setIcon(get_icon("search", size=13))
        self.btn_auto_detect.setStyleSheet(STYLE_TRANSFER_BTN)
        self.btn_auto_detect.setCursor(Qt.PointingHandCursor)
        self.btn_auto_detect.setToolTip("Escanea áreas homogéneas y añade parches de curación automáticos.")
        self.btn_auto_detect.clicked.connect(self._on_auto_detect_clicked)
        c_layout.addWidget(self.btn_auto_detect)

        # Separador tenue
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #2e3440; margin: 2px 0;")
        c_layout.addWidget(sep)

        # 4. Diagnóstico "Visualizar Manchas (A)"
        row_vis = QHBoxLayout()
        self.chk_visualize = QCheckBox("Visualizar Manchas (Paso Alto)")
        self.chk_visualize.setStyleSheet("color: #80d8ff; font-weight: bold; font-size: 11px;")
        self.chk_visualize.setToolTip("Modo de inspección en blanco y negro de alto contraste para detectar motas invisibles a simple vista.")
        self.chk_visualize.toggled.connect(self._on_visualize_toggled)
        row_vis.addWidget(self.chk_visualize)
        c_layout.addLayout(row_vis)

        # Slider de umbral de visualización
        self.row_threshold = PrecisionSliderRow(
            "Umbral Vis.:", 0, 100, 50, scale=1.0, decimals=0, label_width=110
        )
        self.slider_thresh = self.row_threshold.slider
        self.spin_thresh = self.row_threshold.spin
        self.slider_thresh.setToolTip("Ajusta el umbral de corte del filtro paso alto (valores más altos muestran motas más sutiles).")
        self.slider_thresh.valueChanged.connect(self._on_thresh_changed)
        self.spin_thresh.valueChanged.connect(self._on_spin_thresh_changed)
        self.slider_thresh.sliderPressed.connect(self.interaction_started.emit)
        self.slider_thresh.sliderReleased.connect(self._on_slider_released)
        c_layout.addWidget(self.row_threshold)

        # 5. Estado y controles de manchas
        self.lbl_status = QLabel("0 manchas en la imagen")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")
        c_layout.addWidget(self.lbl_status)

        row_actions = QHBoxLayout()
        self.btn_manual_spot = QPushButton("Añadir Manual")
        self.btn_manual_spot.setIcon(get_icon("plus", size=12))
        self.btn_manual_spot.setCheckable(True)
        self.btn_manual_spot.setCursor(Qt.PointingHandCursor)
        self.btn_manual_spot.setToolTip("Activa el puntero en el lienzo para añadir o ajustar motas manualmente con un clic.")
        self.btn_manual_spot.toggled.connect(self._on_manual_spot_toggled)
        row_actions.addWidget(self.btn_manual_spot)

        self.btn_clear_spots = QPushButton("Limpiar Todas")
        self.btn_clear_spots.setIcon(get_icon("trash", size=12))
        self.btn_clear_spots.setCursor(Qt.PointingHandCursor)
        self.btn_clear_spots.setToolTip("Elimina todas las manchas y parches de curación de la imagen.")
        self.btn_clear_spots.clicked.connect(self._on_clear_clicked)
        row_actions.addWidget(self.btn_clear_spots)
        c_layout.addLayout(row_actions)

        self.sec_dust.set_content_widget(w_content)
        layout.addWidget(self.sec_dust)

    # --- Handlers y Callbacks ---

    def _on_sens_changed(self, val: int) -> None:
        self._sensitivity = val
        self.parameter_changed.emit()

    def _on_spin_sens_changed(self, val: float) -> None:
        self.slider_sens.setValue(int(val))

    def _on_thresh_changed(self, val: int) -> None:
        self._vis_threshold = val
        if self._visualize:
            self.visualize_toggled.emit(True, self._vis_threshold)
            self.parameter_changed.emit()

    def _on_spin_thresh_changed(self, val: float) -> None:
        self.slider_thresh.setValue(int(val))

    def _on_mode_changed(self, idx: int) -> None:
        modes = ["heal", "inpaint", "clone"]
        self._mode = modes[idx] if 0 <= idx < len(modes) else "heal"
        self.parameter_changed.emit()
        self.action_committed.emit()

    def _on_auto_detect_clicked(self) -> None:
        self.auto_detect_requested.emit(self._sensitivity)

    def _on_visualize_toggled(self, checked: bool) -> None:
        self._visualize = checked
        self.visualize_toggled.emit(checked, self._vis_threshold)
        self.parameter_changed.emit()

    def _on_manual_spot_toggled(self, checked: bool) -> None:
        self._is_spot_mode_active = checked
        self.spot_mode_toggled.emit(checked)

    def _on_clear_clicked(self) -> None:
        self.clear_spots_requested.emit()

    def _on_slider_released(self) -> None:
        self.interaction_finished.emit()
        self.action_committed.emit()

    def update_spot_count(self, count: int) -> None:
        """Actualiza la etiqueta con la cantidad de manchas curadas."""
        self._spot_count = count
        if count == 0:
            self.lbl_status.setText("0 manchas en la imagen")
        elif count == 1:
            self.lbl_status.setText("1 mancha curada")
        else:
            self.lbl_status.setText(f"{count} manchas curadas")

    def reset_dust(self) -> None:
        """Restablece los controles a sus valores predeterminados."""
        self.slider_sens.setValue(50)
        self.slider_thresh.setValue(50)
        self.chk_visualize.setChecked(False)
        self.combo_mode.setCurrentIndex(0)
        self.btn_manual_spot.setChecked(False)
        self.clear_spots_requested.emit()

    @property
    def sensitivity(self) -> int:
        return self._sensitivity

    @property
    def visualize(self) -> bool:
        return self._visualize

    @property
    def visualize_threshold(self) -> int:
        return self._vis_threshold

    @property
    def mode(self) -> str:
        return self._mode
