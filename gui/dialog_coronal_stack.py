# gui/dialog_coronal_stack.py
"""
gui/dialog_coronal_stack.py - Diálogo de selección y configuración de la multicapa coronal.
Permite elegir qué filtros astronómicos de eclipse (MGN, WOW, NAFE, NRGF, etc.)
se transferirán como capas ráster independientes con modos de fusión y opacidades al Revelador.
"""
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QSpinBox, QFrame, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt
from gui.icons import get_svg_pixmap, get_icon
from core.mask_layers import (
    BLEND_MODE_NORMAL, BLEND_MODE_SCREEN, BLEND_MODE_LUMINOSITY,
    BLEND_MODE_SOFT_LIGHT, BLEND_MODE_OVERLAY, BLEND_MODE_MULTIPLY
)

# Definición estándar de filtros coronales en orden de apilado físico (base -> cima)
DEFAULT_CORONAL_FILTERS = [
    {
        "id": "nrgf",
        "name": "01 - NRGF (Gradiente Radial)",
        "desc": "Normalized Radial Gradient Filter para atenuar la caída radial 1/r².",
        "blend_mode": BLEND_MODE_SCREEN,
        "opacity": 15,
        "checked": True,
    },
    {
        "id": "swap",
        "name": "16 - Filtro SWAP PROBA2",
        "desc": "Filtro no lineal multiescala para protuberancias y cromosfera.",
        "blend_mode": BLEND_MODE_NORMAL,
        "opacity": 25,
        "checked": False,
    },
    {
        "id": "larson_sekanina",
        "name": "15 - Larson-Sekanina",
        "desc": "Diferenciación angular rotacional para chorros polares y streamers.",
        "blend_mode": BLEND_MODE_SOFT_LIGHT,
        "opacity": 20,
        "checked": False,
    },
    {
        "id": "steerable_g2",
        "name": "11 - Steerable G2",
        "desc": "Filtro orientable de segunda derivada para arcos y bucles cerrados.",
        "blend_mode": BLEND_MODE_SOFT_LIGHT,
        "opacity": 20,
        "checked": False,
    },
    {
        "id": "nafe",
        "name": "09 - NAFE Difuso",
        "desc": "Normalizing-Adaptive Filter para contraste en gradientes de baja densidad.",
        "blend_mode": BLEND_MODE_SOFT_LIGHT,
        "opacity": 20,
        "checked": True,
    },
    {
        "id": "wow",
        "name": "02 - WOW Bilateral Guiado",
        "desc": "Filtro edge-aware que preserva bordes sutiles sin producir halos.",
        "blend_mode": BLEND_MODE_LUMINOSITY,
        "opacity": 25,
        "checked": True,
    },
    {
        "id": "mgn",
        "name": "03 - MGN Multiescala",
        "desc": "Morgan & Druckmüller. Realce multiescala de serpentinas y detalles finos.",
        "blend_mode": BLEND_MODE_LUMINOSITY,
        "opacity": 40,
        "checked": True,
    },
]

BLEND_MODES_DISPLAY = [
    ("Luminosidad", BLEND_MODE_LUMINOSITY),
    ("Luz Suave (Soft Light)", BLEND_MODE_SOFT_LIGHT),
    ("Trama (Screen)", BLEND_MODE_SCREEN),
    ("Superponer (Overlay)", BLEND_MODE_OVERLAY),
    ("Multiplicar", BLEND_MODE_MULTIPLY),
    ("Normal", BLEND_MODE_NORMAL),
]


class CoronalStackDialog(QDialog):
    """Diálogo modal para configurar la exportación multicapa de eclipse al Revelador."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Enviar Multicapa Coronal al Revelador")
        self.resize(580, 520)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f121d;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
            }
            QComboBox, QSpinBox {
                background-color: #131724;
                border: 1px solid #273044;
                border-radius: 4px;
                color: #f1f5f9;
                padding: 4px 6px;
                font-size: 11px;
            }
            QComboBox:focus, QSpinBox:focus {
                border: 1px solid #ffb74d;
            }
            QCheckBox {
                color: #e2e8f0;
                font-weight: bold;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)

        self._row_controls: List[Dict[str, Any]] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Encabezado
        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        lbl_icon = QLabel()
        lbl_icon.setPixmap(get_svg_pixmap("layers", color="#ffb74d", size=24))
        header_row.addWidget(lbl_icon)

        header_text = QVBoxLayout()
        header_text.setSpacing(2)
        lbl_title = QLabel("Composición Multicapa Coronal en el Revelador")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffb74d;")
        lbl_sub = QLabel(
            "Selecciona qué filtros astronómicos calcular y transferir a la pila de capas ráster.\n"
            "La imagen Fondo HDR Base siempre se transfiere en la base con opacidad al 100%."
        )
        lbl_sub.setStyleSheet("font-size: 11px; color: #94a3b8;")
        header_text.addWidget(lbl_title)
        header_text.addWidget(lbl_sub)
        header_row.addLayout(header_text, stretch=1)
        layout.addLayout(header_row)

        # Botones rápidos de selección
        row_quick = QHBoxLayout()
        row_quick.setSpacing(8)
        btn_rec = QPushButton("Selección Recomendada")
        btn_rec.setCursor(Qt.PointingHandCursor)
        btn_rec.setStyleSheet("background: #1e293b; color: #ffb74d; border: 1px solid #334155; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
        btn_rec.clicked.connect(self._select_recommended)
        row_quick.addWidget(btn_rec)

        btn_all = QPushButton("Seleccionar Todo")
        btn_all.setCursor(Qt.PointingHandCursor)
        btn_all.setStyleSheet("background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
        btn_all.clicked.connect(self._select_all)
        row_quick.addWidget(btn_all)

        btn_none = QPushButton("Desmarcar Todo")
        btn_none.setCursor(Qt.PointingHandCursor)
        btn_none.setStyleSheet("background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 8px; border-radius: 4px; font-size: 11px;")
        btn_none.clicked.connect(self._select_none)
        row_quick.addWidget(btn_none)
        row_quick.addStretch()
        layout.addLayout(row_quick)

        # Contenedor con scroll para las capas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #1e293b; border-radius: 6px; background-color: #0b0e17; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(8)

        # Crear filas de filtros
        for item in DEFAULT_CORONAL_FILTERS:
            row_frame = QFrame()
            row_frame.setStyleSheet("""
                QFrame {
                    background-color: #131724;
                    border: 1px solid #1e293b;
                    border-radius: 5px;
                    padding: 4px;
                }
            """)
            r_lay = QHBoxLayout(row_frame)
            r_lay.setContentsMargins(8, 6, 8, 6)
            r_lay.setSpacing(10)

            chk = QCheckBox(item["name"])
            chk.setChecked(item["checked"])
            chk.setToolTip(item["desc"])

            info_col = QVBoxLayout()
            info_col.setSpacing(2)
            info_col.addWidget(chk)
            lbl_desc = QLabel(item["desc"])
            lbl_desc.setStyleSheet("color: #64748b; font-size: 10px; margin-left: 20px;")
            lbl_desc.setWordWrap(True)
            info_col.addWidget(lbl_desc)
            r_lay.addLayout(info_col, stretch=1)

            # Selector de modo de fusión
            combo_blend = QComboBox()
            combo_blend.setFixedWidth(130)
            for disp_name, b_val in BLEND_MODES_DISPLAY:
                combo_blend.addItem(disp_name, b_val)
            # Buscar índice por defecto
            for idx in range(combo_blend.count()):
                if combo_blend.itemData(idx) == item["blend_mode"]:
                    combo_blend.setCurrentIndex(idx)
                    break
            combo_blend.setEnabled(item["checked"])
            r_lay.addWidget(combo_blend)

            # Opacidad
            spin_op = QSpinBox()
            spin_op.setRange(0, 100)
            spin_op.setSuffix("%")
            spin_op.setValue(item["opacity"])
            spin_op.setFixedWidth(65)
            spin_op.setEnabled(item["checked"])
            r_lay.addWidget(spin_op)

            # Sincronizar habilitación de controles con el checkbox
            chk.toggled.connect(combo_blend.setEnabled)
            chk.toggled.connect(spin_op.setEnabled)

            scroll_layout.addWidget(row_frame)
            self._row_controls.append({
                "id": item["id"],
                "name": item["name"],
                "chk": chk,
                "combo": combo_blend,
                "spin": spin_op,
            })

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)

        # Botones de acción inferiores
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)

        lbl_hint = QLabel("💡 En el Revelador podrás pintar máscaras con pincel en cada capa.")
        lbl_hint.setStyleSheet("color: #80cbc4; font-size: 11px;")
        btn_box.addWidget(lbl_hint)
        btn_box.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.setStyleSheet("background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 6px 14px; border-radius: 4px;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_accept = QPushButton("Enviar al Revelador")
        btn_accept.setIcon(get_icon("arrow_right", color="#ffffff"))
        btn_accept.setCursor(Qt.PointingHandCursor)
        btn_accept.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d97706, stop:1 #f59e0b);
                color: #ffffff;
                font-weight: bold;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #b45309, stop:1 #d97706);
            }
        """)
        btn_accept.clicked.connect(self.accept)
        btn_box.addWidget(btn_accept)

        layout.addLayout(btn_box)

    def _select_recommended(self) -> None:
        rec_ids = {"nrgf", "nafe", "wow", "mgn"}
        for ctrl in self._row_controls:
            ctrl["chk"].setChecked(ctrl["id"] in rec_ids)

    def _select_all(self) -> None:
        for ctrl in self._row_controls:
            ctrl["chk"].setChecked(True)

    def _select_none(self) -> None:
        for ctrl in self._row_controls:
            ctrl["chk"].setChecked(False)

    def get_selected_layers(self) -> List[Dict[str, Any]]:
        """Retorna las configuraciones de filtros seleccionados en orden de apilado físico (base -> cima)."""
        selected = []
        for ctrl in self._row_controls:
            if ctrl["chk"].isChecked():
                selected.append({
                    "id": ctrl["id"],
                    "name": ctrl["name"],
                    "blend_mode": ctrl["combo"].currentData(),
                    "opacity": float(ctrl["spin"].value()) / 100.0,
                })
        return selected
