# gui/widgets/export_button.py
"""
gui/widgets/export_button.py - Botón dividido (Split Button) reutilizable para exportación de imágenes.

Proporciona un control unificado en AstroLoom para:
- Clic directo (1-click): Ejecutar el flujo maestro habitual (guardado nativo TIFF 16/32-bit, PSD, etc.).
- Flecha desplegable (▼): Menú contextual con opciones avanzadas, principalmente "Exportar para Web / Redes...".
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QToolButton, QWidget

from gui.icons import get_icon
from gui.theme import STYLE_SPLIT_RUN_BTN, STYLE_SPLIT_SECONDARY_BTN


class ExportSplitButton(QToolButton):
    """
    Botón dividido (Split Button) reutilizable para exportación de imágenes en AstroLoom.
    Combina acción de guardado primario y menú emergente para exportación optimizada web.
    """

    def __init__(
        self,
        text: str = "Exportar...",
        on_master_export: Optional[Callable[[], None]] = None,
        on_web_export: Optional[Callable[[], None]] = None,
        style: str = "primary",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setText(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setIcon(get_icon("save", color="#ffffff", size=16))
        self.set_button_style(style)

        # Conectar acción primaria (clic en la zona del botón principal)
        if on_master_export:
            self.clicked.connect(on_master_export)

        # Menú desplegable lateral
        self._export_menu = QMenu(self)
        self.setMenu(self._export_menu)

        self.action_web: Optional[QAction] = None
        if on_web_export:
            self.action_web = QAction("Exportar para Web / Redes...", self)
            self.action_web.setIcon(get_icon("globe", color="#ffffff", size=14))
            self.action_web.setShortcut("Ctrl+Alt+W")
            self.action_web.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self.action_web.triggered.connect(on_web_export)
            self._export_menu.addAction(self.action_web)
            self.addAction(self.action_web)

    def set_button_style(self, style: str = "primary") -> None:
        """Aplica la variante visual deseada: 'primary' (azul cobalto) o 'secondary' (gris neutro)."""
        if style == "secondary":
            self.setStyleSheet(STYLE_SPLIT_SECONDARY_BTN)
        else:
            self.setStyleSheet(STYLE_SPLIT_RUN_BTN)

    def set_on_master_export(self, callback: Callable[[], None]) -> None:
        """Conecta o reconecta la acción principal del botón."""
        try:
            self.clicked.disconnect()
        except RuntimeError:
            pass
        self.clicked.connect(callback)

    def set_on_web_export(self, callback: Callable[[], None]) -> None:
        """Configura o actualiza la acción del menú de exportación web."""
        if self.action_web is None:
            self.action_web = QAction("Exportar para Web / Redes...", self)
            self.action_web.setIcon(get_icon("globe", color="#ffffff", size=14))
            self.action_web.setShortcut("Ctrl+Alt+W")
            self.action_web.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self._export_menu.addAction(self.action_web)
            self.addAction(self.action_web)
        else:
            try:
                self.action_web.triggered.disconnect()
            except RuntimeError:
                pass
        self.action_web.triggered.connect(callback)

    def add_separator(self) -> None:
        """Añade una línea divisoria al menú desplegable."""
        self._export_menu.addSeparator()

    def setup_standard_menu(
        self,
        on_format_export: Callable[[str], None],
        psd_label: str = "Guardar Photoshop (PSD)...",
    ) -> None:
        """
        Configura las opciones estándar de formato en el menú desplegable:
        TIFF 16-bit, TIFF 32-bit Float, Photoshop PSD, PNG, JPEG y Web / Redes al final.
        """
        self._export_menu.clear()

        self.add_menu_action("Guardar TIFF 16-bit...", lambda: on_format_export("tiff16"), icon_name="save")
        self.add_menu_action("Guardar TIFF 32-bit Float...", lambda: on_format_export("tiff32"), icon_name="save")
        self.add_menu_action(psd_label, lambda: on_format_export("psd"), icon_name="layers")
        self.add_menu_action("Guardar PNG...", lambda: on_format_export("png"), icon_name="image")
        self.add_menu_action("Guardar JPEG...", lambda: on_format_export("jpg"), icon_name="image")
        self._export_menu.addSeparator()

        if self.action_web:
            self._export_menu.addAction(self.action_web)

    def add_menu_action(
        self,
        text: str,
        callback: Callable[[], None],
        icon_name: Optional[str] = None,
        shortcut: Optional[str] = None,
    ) -> QAction:
        """Permite añadir opciones suplementarias al menú desplegable del módulo."""
        act = QAction(text, self)
        if icon_name:
            act.setIcon(get_icon(icon_name, color="#ffffff", size=14))
        if shortcut:
            act.setShortcut(shortcut)
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self.addAction(act)
        act.triggered.connect(callback)
        self._export_menu.addAction(act)
        return act
