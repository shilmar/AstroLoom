# gui/widgets/collapsible_section.py
"""
gui/widgets/collapsible_section.py - Sección Plegable (Acordeón) para AstroLoom.

Proporciona un contenedor colapsable reutilizable con cabecera interactiva,
flecha de estado, título en negrita, indicador/badge de estado (ej: '● ON'),
botón de restablecimiento '↺' integrado y estilo adaptado a Cosmic Dark Pro.
"""

from typing import Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFont


class CollapsibleSection(QWidget):
    """
    Contenedor de sección plegable estilo acordeón.
    Permite expandir y contraer su contenido al hacer clic en la cabecera.
    """
    toggled = Signal(bool)  # Emite True si se expande, False si se contrae

    def __init__(
        self,
        title: str,
        is_expanded: bool = True,
        on_reset: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._is_expanded: bool = is_expanded
        self._on_reset_callback: Optional[Callable[[], None]] = on_reset
        self._title_text: str = title

        self._setup_ui()
        self.set_expanded(self._is_expanded)

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
        header_layout.setSpacing(6)

        # Flecha indicadora (▾ o ▸)
        self.lbl_arrow = QLabel("▾" if self._is_expanded else "▸")
        self.lbl_arrow.setStyleSheet("color: #80d8ff; font-weight: bold; font-size: 13px;")
        header_layout.addWidget(self.lbl_arrow)

        # Título de la sección
        self.lbl_title = QLabel(self._title_text)
        self.lbl_title.setStyleSheet("font-weight: bold; color: #eceff4; font-size: 12px;")
        header_layout.addWidget(self.lbl_title)

        header_layout.addStretch(1)

        # Badge / Indicador de estado (opcional, ej: '● ON', '● Separadas')
        self.lbl_badge = QLabel("")
        self.lbl_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #80d8ff; margin-right: 4px;")
        self.lbl_badge.setVisible(False)
        header_layout.addWidget(self.lbl_badge)

        # Botón de reinicio rápido en cabecera
        self.btn_reset = QPushButton("↺")
        self.btn_reset.setFixedSize(22, 22)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setToolTip("Restablecer controles de esta sección")
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #78909c;
                border: none;
                font-weight: bold;
                font-size: 13px;
                border-radius: 11px;
                padding: 0px;
            }
            QPushButton:hover {
                background: #2a3148;
                color: #80d8ff;
            }
            QPushButton:pressed {
                background: #37474f;
                color: #ffffff;
            }
        """)
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        if not self._on_reset_callback:
            self.btn_reset.setVisible(False)
        header_layout.addWidget(self.btn_reset)

        main_layout.addWidget(self.header)

        # 2. Contenedor del contenido (cuerpo)
        self.content_area = QWidget()
        self.content_area.setObjectName("CollapsibleContentArea")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(4, 6, 4, 4)
        self.content_layout.setSpacing(6)
        main_layout.addWidget(self.content_area)

        # Evento de clic en toda la cabecera (excepto si se hace clic directo en el botón reset)
        self.header.mousePressEvent = self._on_header_clicked
        self._update_styles()

    def _update_styles(self) -> None:
        """Aplica estilos acordes al tema Cosmic Dark Pro."""
        bg_col = "#1e2230" if self._is_expanded else "#161924"
        border_col = "#2f364d" if self._is_expanded else "#222738"
        border_radius = "4px 4px 0px 0px" if self._is_expanded else "4px"

        self.header.setStyleSheet(f"""
            QFrame#CollapsibleHeader {{
                background-color: {bg_col};
                border: 1px solid {border_col};
                border-radius: {border_radius};
            }}
            QFrame#CollapsibleHeader:hover {{
                background-color: #262c3e;
                border-color: #3b4563;
            }}
        """)

        self.content_area.setStyleSheet("""
            QWidget#CollapsibleContentArea {
                background-color: #12141e;
                border-left: 1px solid #232738;
                border-right: 1px solid #232738;
                border-bottom: 1px solid #232738;
                border-radius: 0px 0px 4px 4px;
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
        # Limpiar widgets previos si existieran
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
        self.lbl_arrow.setText("▾" if expanded else "▸")
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
        """Actualiza el texto del título en la cabecera."""
        self._title_text = title
        self.lbl_title.setText(title)

    def set_status_badge(self, text: str, color: str = "#80d8ff") -> None:
        """Muestra u oculta una insignia de estado en la cabecera (ej: '● ON', '● Separadas')."""
        if text:
            self.lbl_badge.setText(text)
            self.lbl_badge.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {color}; margin-right: 4px;")
            self.lbl_badge.setVisible(True)
        else:
            self.lbl_badge.setText("")
            self.lbl_badge.setVisible(False)

    def set_reset_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Asigna o actualiza la acción del botón '↺' de reinicio."""
        self._on_reset_callback = callback
        self.btn_reset.setVisible(callback is not None)
