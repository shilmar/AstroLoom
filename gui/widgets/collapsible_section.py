# gui/widgets/collapsible_section.py
"""
gui/widgets/collapsible_section.py - Sección Plegable (Acordeón) Pro para AstroLoom.

Proporciona un contenedor colapsable reutilizable con cabecera interactiva,
iconografía vectorial SVG integrada (chevron, icono temático de sección y reset),
título limpio de alta legibilidad, badge de estado flotante tipo píldora
y superficies de tarjeta elevadas acordes a AstroLoom Studio Pro.
"""

import re
from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QCursor, QFont

from gui.icons import get_svg_pixmap, get_icon, ICON_COLOR_DEFAULT, ICON_COLOR_ACTIVE, ICON_COLOR_MUTED

# Mapeo inteligente de palabras clave del título a iconos vectoriales SVG
SECTION_ICON_KEYWORDS = {
    "preset": "sliders",
    "estilo": "sliders",
    "style": "sliders",
    "calibracion": "sparkles",
    "calibración": "sparkles",
    "gradiente": "sparkles",
    "gradient": "sparkles",
    "graxpert": "sparkles",
    "estirado": "chart",
    "stretch": "chart",
    "curva": "curve",
    "curve": "curve",
    "starnet": "star",
    "estrella": "star",
    "star": "star",
    "deconv": "microscope",
    "enfoque": "microscope",
    "sharp": "microscope",
    "balance": "thermometer",
    "wb": "thermometer",
    "color": "palette",
    "saturacion": "palette",
    "saturación": "palette",
    "suelo": "mountain",
    "ground": "mountain",
    "paisaje": "mountain",
    "mascara": "layers",
    "máscara": "layers",
    "mask": "layers",
    "capa": "layers",
    "layer": "layers",
}


class CollapsibleSection(QWidget):
    """
    Contenedor de sección plegable estilo acordeón con iconografía SVG.
    Permite expandir y contraer su contenido al hacer clic en la cabecera.
    """
    toggled = Signal(bool)  # Emite True si se expande, False si se contrae
    switchToggled = Signal(bool)  # Emite el estado On/Off del switch de cabecera

    def __init__(
        self,
        title: str,
        is_expanded: bool = True,
        on_reset: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
        icon_name: Optional[str] = None,
        icon: Optional[str] = None
    ) -> None:
        super().__init__(parent)
        self._is_expanded: bool = is_expanded
        self._on_reset_callback: Optional[Callable[[], None]] = on_reset
        self._explicit_icon: Optional[str] = icon or icon_name

        clean_title, detected_icon = self._clean_title_and_detect_icon(title, self._explicit_icon)
        self._title_text: str = clean_title
        self._icon_name: Optional[str] = detected_icon

        self._setup_ui()
        self.set_expanded(self._is_expanded)

    @classmethod
    def _clean_title_and_detect_icon(
        cls,
        raw_title: str,
        explicit_icon: Optional[str] = None
    ) -> tuple[str, Optional[str]]:
        """
        Limpia emojis hardcodeados del título y determina el icono vectorial correspondiente.
        """
        icon = explicit_icon
        if not icon:
            low = raw_title.lower()
            for kw, ic in SECTION_ICON_KEYWORDS.items():
                if kw in low:
                    icon = ic
                    break

        # Limpiar emojis o símbolos decorativos iniciales
        cleaned = re.sub(r'^[^\w\s\(\)\-\.]+', '', raw_title).strip()
        if not cleaned:
            cleaned = raw_title.strip()

        return cleaned, icon

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 2, 0, 4)
        main_layout.setSpacing(0)

        # 1. Cabecera interactiva
        self.header = QFrame()
        self.header.setFixedHeight(34)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setObjectName("CollapsibleHeader")

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 0, 8, 0)
        header_layout.setSpacing(7)

        # Flecha indicadora vectorial SVG
        self.lbl_arrow = QLabel()
        self.lbl_arrow.setFixedSize(14, 14)
        self.lbl_arrow.setAlignment(Qt.AlignCenter)
        self._update_arrow_pixmap()
        header_layout.addWidget(self.lbl_arrow)

        # Icono temático de sección vectorial SVG
        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(16, 16)
        self.lbl_icon.setAlignment(Qt.AlignCenter)
        if self._icon_name:
            self.lbl_icon.setPixmap(get_svg_pixmap(self._icon_name, color="#38bdf8", size=15))
            self.lbl_icon.setVisible(True)
        else:
            self.lbl_icon.setVisible(False)
        header_layout.addWidget(self.lbl_icon)

        # Título de la sección
        self.lbl_title = QLabel(self._title_text)
        self.lbl_title.setStyleSheet("font-weight: 600; color: #eceff4; font-size: 12px;")
        self.lbl_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        header_layout.addWidget(self.lbl_title)

        header_layout.addStretch(1)

        # Badge / Indicador de estado flotante (ej: '● ON', '● Separadas')
        self.lbl_badge = QLabel("")
        self.lbl_badge.setMaximumWidth(135)
        self.lbl_badge.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.lbl_badge.setStyleSheet(
            "background-color: #121622; border: 1px solid #2b3648; border-radius: 4px; "
            "padding: 2px 6px; font-size: 10px; font-weight: bold; color: #80d8ff; margin-right: 4px;"
        )
        self.lbl_badge.setVisible(False)
        header_layout.addWidget(self.lbl_badge)

        # Botón de reinicio rápido en cabecera con icono SVG vectorial
        self.btn_reset = QPushButton()
        self.btn_reset.setFixedSize(22, 22)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setToolTip("Restablecer controles de esta sección")
        self.btn_reset.setIcon(get_icon("refresh", color="#64748b", hover_color="#38bdf8", active_color="#00e5ff", size=13))
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                border-radius: 11px;
                padding: 0px;
            }
            QPushButton:hover {
                background: #22293d;
            }
            QPushButton:pressed {
                background: #333d59;
            }
        """)
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        if not self._on_reset_callback:
            self.btn_reset.setVisible(False)
        header_layout.addWidget(self.btn_reset)

        # Switch deslizante On/Off en cabecera
        self.btn_switch = QPushButton()
        self.btn_switch.setObjectName("SectionSwitch")
        self.btn_switch.setCheckable(True)
        self.btn_switch.setChecked(True)
        self.btn_switch.setFixedSize(30, 16)
        self.btn_switch.setCursor(Qt.PointingHandCursor)
        self.btn_switch.setText("●")
        self.btn_switch.setStyleSheet("""
            QPushButton#SectionSwitch {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
                border-radius: 8px;
                color: #ffffff;
                font-size: 10px;
                text-align: right;
                padding-right: 3px;
                padding-bottom: 1px;
            }
            QPushButton#SectionSwitch:!checked {
                background-color: #1a1e2c;
                border: 1px solid #2e354a;
                border-radius: 8px;
                color: #64748b;
                text-align: left;
                padding-left: 3px;
                padding-bottom: 1px;
            }
        """)
        self.btn_switch.toggled.connect(self._on_switch_toggled)
        self.btn_switch.setVisible(False)
        header_layout.addWidget(self.btn_switch)

        main_layout.addWidget(self.header)

        # 2. Contenedor del contenido (cuerpo)
        self.content_area = QWidget()
        self.content_area.setObjectName("CollapsibleContentArea")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        self.content_layout.setSpacing(6)
        main_layout.addWidget(self.content_area)

        # Evento de clic en toda la cabecera
        self.header.mousePressEvent = self._on_header_clicked
        self._update_styles()

    def _update_arrow_pixmap(self) -> None:
        """Actualiza el chevron SVG según el estado desplegado/colapsado."""
        if self._is_expanded:
            self.lbl_arrow.setPixmap(get_svg_pixmap("chevron_down", color="#38bdf8", size=11))
        else:
            self.lbl_arrow.setPixmap(get_svg_pixmap("chevron_right", color="#64748b", size=11))

    def _update_styles(self) -> None:
        """Aplica estilos acordes a AstroLoom Studio Pro."""
        bg_col = "#1c2030" if self._is_expanded else "#151824"
        border_col = "#2a3146" if self._is_expanded else "#222738"
        border_radius = "6px 6px 0px 0px" if self._is_expanded else "6px"

        self.header.setStyleSheet(f"""
            QFrame#CollapsibleHeader {{
                background-color: {bg_col};
                border: 1px solid {border_col};
                border-radius: {border_radius};
            }}
            QFrame#CollapsibleHeader:hover {{
                background-color: #222738;
                border-color: #3b4563;
            }}
        """)

        self.content_area.setStyleSheet("""
            QWidget#CollapsibleContentArea {
                background-color: #12141e;
                border-left: 1px solid #2a3146;
                border-right: 1px solid #2a3146;
                border-bottom: 1px solid #2a3146;
                border-radius: 0px 0px 6px 6px;
            }
        """)

    def _on_header_clicked(self, event) -> None:
        """Manejador al hacer clic sobre la cabecera para alternar expansión."""
        if event.button() == Qt.LeftButton:
            self.toggle()
            event.accept()
        else:
            super(QFrame, self.header).mousePressEvent(event)

    def _on_reset_clicked(self) -> None:
        """Dispara el callback de reinicio si está configurado."""
        if self._on_reset_callback:
            self._on_reset_callback()

    # =========================================================================
    # MÉTODOS PÚBLICOS
    # =========================================================================

    def set_content_widget(self, widget: QWidget) -> None:
        """Asigna el widget principal de contenido dentro del área colapsable."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self.content_layout.addWidget(widget)

    def add_widget(self, widget: QWidget) -> None:
        """Añade un widget directamente al layout del contenido."""
        self.content_layout.addWidget(widget)

    def set_expanded(self, expanded: bool) -> None:
        """Fuerza la expansión o el colapso del contenido."""
        self._is_expanded = expanded
        self._update_arrow_pixmap()
        self.content_area.setVisible(expanded)
        self._update_styles()
        self.toggled.emit(expanded)

    def expand(self) -> None:
        """Expande la sección."""
        if not self._is_expanded:
            self.set_expanded(True)

    def collapse(self) -> None:
        """Colapsa la sección."""
        if self._is_expanded:
            self.set_expanded(False)

    def toggle(self) -> None:
        """Alterna el estado de expansión de la sección."""
        self.set_expanded(not self._is_expanded)

    def is_expanded(self) -> bool:
        """Devuelve True si la sección está desplegada."""
        return self._is_expanded

    def set_title(self, title: str) -> None:
        """Actualiza el texto del título en la cabecera, limpiando emojis y actualizando icono."""
        clean_title, detected_icon = self._clean_title_and_detect_icon(title, self._explicit_icon)
        self._title_text = clean_title
        self.lbl_title.setText(clean_title)

        if detected_icon:
            self._icon_name = detected_icon
            self.lbl_icon.setPixmap(get_svg_pixmap(detected_icon, color="#38bdf8", size=15))
            self.lbl_icon.setVisible(True)

    def set_icon(self, icon_name: str) -> None:
        """Asigna explícitamente un icono SVG vectorial a la cabecera."""
        self._explicit_icon = icon_name
        self._icon_name = icon_name
        self.lbl_icon.setPixmap(get_svg_pixmap(icon_name, color="#38bdf8", size=15))
        self.lbl_icon.setVisible(True)

    def set_status_badge(self, text: str, color: str = "#80d8ff") -> None:
        """Muestra u oculta una insignia de estado en la cabecera (ej: '● ON', '● Separadas')."""
        if text:
            fm = self.lbl_badge.fontMetrics()
            elided = fm.elidedText(text, Qt.ElideRight, 125)
            self.lbl_badge.setText(elided)
            self.lbl_badge.setToolTip(text)
            self.lbl_badge.setStyleSheet(
                f"background-color: #121622; border: 1px solid #2b3648; border-radius: 4px; "
                f"padding: 2px 6px; font-size: 10px; font-weight: bold; color: {color}; margin-right: 4px;"
            )
            self.lbl_badge.setVisible(True)
        else:
            self.lbl_badge.setText("")
            self.lbl_badge.setToolTip("")
            self.lbl_badge.setVisible(False)

    def set_reset_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Asigna o actualiza la acción del botón '↺' de reinicio."""
        self._on_reset_callback = callback
        self.btn_reset.setVisible(callback is not None)

    # Alias para compatibilidad con código existente que use setTitle() de QGroupBox
    setTitle = set_title

    def setContentLayout(self, layout) -> None:
        self.content_layout.addLayout(layout)

    def addWidget(self, widget) -> None:
        self.content_layout.addWidget(widget)

    def addLayout(self, layout) -> None:
        self.content_layout.addLayout(layout)

    def enable_switch(self, checked: bool = True, on_toggled: Optional[Callable[[bool], None]] = None) -> QPushButton:
        """Habilita y muestra el switch On/Off en la cabecera del acordeón."""
        self._switch_enabled = True
        self.btn_switch.blockSignals(True)
        self.btn_switch.setChecked(checked)
        self.btn_switch.blockSignals(False)
        self.btn_switch.setVisible(True)
        if on_toggled:
            self.switchToggled.connect(on_toggled)
        return self.btn_switch

    def is_switch_checked(self) -> bool:
        """Retorna True si el switch de la cabecera está activo o si no existe."""
        if getattr(self, "_switch_enabled", False):
            return self.btn_switch.isChecked()
        return True

    def set_switch_checked(self, checked: bool) -> None:
        """Actualiza el estado del switch de la cabecera."""
        self.btn_switch.setChecked(checked)

    def _on_switch_toggled(self, checked: bool) -> None:
        """Emite la señal al alternar el switch."""
        self.switchToggled.emit(checked)

    def minimumSizeHint(self) -> QSize:
        """Permite que la sección se adapte con total fluidez a columnas estrechas sin desbordamiento."""
        return QSize(250, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        """Tamaño sugerido por defecto para la sección."""
        return QSize(350, super().sizeHint().height())
