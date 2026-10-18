# gui/developer/panels/panel_crop.py
"""
gui/developer/panels/panel_crop.py - Barra contextual y herramientas de recorte, rotación y nivelación de horizonte.
"""
from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QSlider, QFrame
)
from PySide6.QtCore import Qt, Signal

from gui.icons import get_icon


class CropToolbar(QWidget):
    """
    Barra de herramientas contextual de Recorte y Nivelación de Horizonte.
    """
    ratio_changed = Signal(int)
    flip_ratio_clicked = Signal()
    angle_changed = Signal(int)
    guides_changed = Signal(int)
    reset_clicked = Signal()
    cancel_clicked = Signal()
    apply_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None, resettable_label_factory: Optional[Callable] = None):
        super().__init__(parent)
        self._make_resettable_label = resettable_label_factory or (lambda lbl, *_: lbl)
        self._setup_ui()

    def _setup_ui(self):
        tb_layout = QHBoxLayout(self)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(8)

        # 1. Indicador / Icono de modo
        lbl_icon = QLabel()
        lbl_icon.setPixmap(get_icon("crop", color="#80d8ff", size=15).pixmap(15, 15))
        tb_layout.addWidget(lbl_icon)

        lbl_title = QLabel("RECORTE:")
        lbl_title.setStyleSheet("color: #80d8ff; font-weight: bold; font-size: 11px;")
        tb_layout.addWidget(lbl_title)

        # 2. Selector de Proporción (Aspect Ratio)
        lbl_prop = QLabel("Proporción:")
        lbl_prop.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_prop)
        self.combo_crop_ratio = QComboBox()
        self.combo_crop_ratio.setToolTip("Define la relación de aspecto del marco de recorte")
        self.combo_crop_ratio.addItems([
            "Libre",
            "Original",
            "1:1 (Cuadrado)",
            "16:9 (Panorámico)",
            "3:2 (Fotografía)",
            "4:3 (Estándar)",
            "5:4 (Gran Formato)",
            "2:1 (Panorámico 2:1)",
            "9:16 (Vertical)",
            "2:3 (Vertical 35mm)",
            "4:5 (Vertical IG)",
        ])
        self.combo_crop_ratio.currentIndexChanged.connect(self.ratio_changed.emit)
        tb_layout.addWidget(self.combo_crop_ratio)

        # 3. Botón Invertir orientación de la proporción
        self.btn_crop_flip_ratio = QPushButton("Invertir (X)")
        self.btn_crop_flip_ratio.setIcon(get_icon("sliders", size=12))
        self.btn_crop_flip_ratio.setToolTip("Invierte la orientación de la proporción (Horizontal ⇄ Vertical)\nAtajo: X")
        self.btn_crop_flip_ratio.setCursor(Qt.PointingHandCursor)
        self.btn_crop_flip_ratio.clicked.connect(self.flip_ratio_clicked.emit)
        tb_layout.addWidget(self.btn_crop_flip_ratio)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        sep1.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep1)

        # 4. Nivelación de Horizonte / Ángulo de rotación
        lbl_ang = self._make_resettable_label(QLabel("Ángulo:"), lambda: self.slider_crop_angle.setValue(0), "0.0°")
        lbl_ang.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_ang)
        self.slider_crop_angle = QSlider(Qt.Horizontal)
        self.slider_crop_angle.setRange(-450, 450)  # -45.0° a +45.0°
        self.slider_crop_angle.setValue(0)
        self.slider_crop_angle.setFixedWidth(120)
        self.slider_crop_angle.setToolTip("Nivela el horizonte rotando la imagen en tiempo real (-45° a +45°)")
        self.slider_crop_angle.valueChanged.connect(self.angle_changed.emit)
        tb_layout.addWidget(self.slider_crop_angle)

        self.lbl_crop_angle_val = QLabel("0.0°")
        self.lbl_crop_angle_val.setStyleSheet("color: #88c0d0; font-weight: bold; min-width: 38px; font-size: 11px;")
        tb_layout.addWidget(self.lbl_crop_angle_val)

        self.btn_crop_reset_angle = QPushButton("0°")
        self.btn_crop_reset_angle.setFixedWidth(28)
        self.btn_crop_reset_angle.setToolTip("Restablece el ángulo de nivelación a 0°")
        self.btn_crop_reset_angle.setCursor(Qt.PointingHandCursor)
        self.btn_crop_reset_angle.clicked.connect(lambda: self.slider_crop_angle.setValue(0))
        tb_layout.addWidget(self.btn_crop_reset_angle)

        # Separador vertical
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep2)

        # 5. Guías de composición
        lbl_gui = QLabel("Guías:")
        lbl_gui.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_gui)
        self.combo_crop_guides = QComboBox()
        self.combo_crop_guides.addItems([
            "Regla de Tercios",
            "Cuadrícula 6×6",
            "Sin Guías"
        ])
        self.combo_crop_guides.currentIndexChanged.connect(self.guides_changed.emit)
        tb_layout.addWidget(self.combo_crop_guides)

        # 6. Etiqueta informativa de dimensiones
        self.lbl_crop_dimensions = QLabel("0 × 0 px")
        self.lbl_crop_dimensions.setStyleSheet("color: #ffd54f; font-weight: bold; font-size: 11px; margin-left: 4px;")
        tb_layout.addWidget(self.lbl_crop_dimensions)

        tb_layout.addStretch()

        # 7. Botones de acción: Restablecer, Cancelar, Aplicar
        self.btn_crop_reset = QPushButton("Restablecer")
        self.btn_crop_reset.setIcon(get_icon("refresh", size=13))
        self.btn_crop_reset.setToolTip("Restablece el marco de recorte al tamaño completo de la imagen")
        self.btn_crop_reset.setCursor(Qt.PointingHandCursor)
        self.btn_crop_reset.clicked.connect(self.reset_clicked.emit)
        tb_layout.addWidget(self.btn_crop_reset)

        self.btn_crop_cancel = QPushButton("Cancelar (Esc)")
        self.btn_crop_cancel.setIcon(get_icon("x", color="#f87171", size=13))
        self.btn_crop_cancel.setToolTip("Cancela el recorte y descarta los cambios (Esc)")
        self.btn_crop_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_crop_cancel.clicked.connect(self.cancel_clicked.emit)
        tb_layout.addWidget(self.btn_crop_cancel)

        self.btn_crop_apply = QPushButton("Aplicar Recorte (Intro)")
        self.btn_crop_apply.setIcon(get_icon("check", color="#4ade80", size=13))
        self.btn_crop_apply.setToolTip("Aplica permanentemente el recorte y la rotación seleccionados (Intro)")
        self.btn_crop_apply.setCursor(Qt.PointingHandCursor)
        self.btn_crop_apply.setStyleSheet("""
            QPushButton {
                background-color: #14532d;
                color: #86efac;
                font-weight: bold;
                border: 1px solid #16a34a;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #166534;
                color: #ffffff;
            }
        """)
        self.btn_crop_apply.clicked.connect(self.apply_clicked.emit)
        tb_layout.addWidget(self.btn_crop_apply)
