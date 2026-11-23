# gui/widgets/mask_layers/common.py
"""
Utilidades compartidas para los paneles de capas de máscaras.
"""
from typing import Callable, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


def make_resettable_label(lbl: QLabel, reset_action: Callable[[], None], default_desc: str = "") -> QLabel:
    """Asigna comportamiento de doble clic para restablecer el control asociado."""
    lbl.setCursor(Qt.PointingHandCursor)
    tt = f"Doble clic para restablecer a {default_desc}" if default_desc else "Doble clic para restablecer"
    lbl.setToolTip(tt)
    lbl.mouseDoubleClickEvent = lambda e: reset_action()
    return lbl


def make_box_qss(obj_name: str, border_col: str, title_col: str, bg_col: str) -> str:
    """Genera la hoja de estilos QSS para los QGroupBox tematizados de capas (Neutral Studio 60-30-10 con relieve DaVinci Resolve)."""
    return f"""
        QGroupBox#{obj_name} {{
            font-weight: bold;
            border: 1px solid #2e2e35;
            border-radius: 6px;
            margin-top: 14px;
            padding-top: 14px;
            background-color: #222226;
            color: #e4e4e7;
        }}
        QGroupBox#{obj_name}::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 2px;
            padding: 2px 8px;
            background-color: #18181b;
            border: 1px solid #2e2e35;
            border-radius: 4px;
            color: {title_col};
            font-weight: bold;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
    """
