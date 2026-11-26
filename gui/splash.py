# gui/splash.py
"""
gui/splash.py - Pantalla de carga (Splash Screen) moderna y minimalista para AstroLoom.
Muestra el estado de inicialización durante el arranque en frío sin retrasar la apertura.
"""
from typing import Optional
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QPen, QBrush
from PySide6.QtWidgets import QApplication, QSplashScreen

from core import APP_VERSION
from gui.icons import get_svg_pixmap


class AstroSplashScreen(QSplashScreen):
    """Pantalla de inicio frameless con estética Slate espacial y reporte dinámico de progreso."""

    def __init__(self, width: int = 480, height: int = 270):
        self.w = width
        self.h = height
        pix = QPixmap(self.w, self.h)
        pix.fill(Qt.transparent)
        super().__init__(pix, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._status = "Iniciando AstroLoom..."
        self._progress = 0
        self._render()

    def set_status(self, text: str, progress: int = 0) -> None:
        """Actualiza el mensaje y porcentaje de carga refrescando de inmediato el bucle de eventos."""
        self._status = text
        self._progress = max(0, min(100, progress))
        self._render()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _render(self) -> None:
        pix = QPixmap(self.w, self.h)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Fondo contenedor oscuro con borde fino y esquinas redondeadas
        rect = QRectF(1, 1, self.w - 2, self.h - 2)
        painter.setBrush(QBrush(QColor("#0b0e17")))
        painter.setPen(QPen(QColor("#222938"), 1.5))
        painter.drawRoundedRect(rect, 14.0, 14.0)

        # 2. Icono SVG astronómico centrado
        ico = get_svg_pixmap("sparkles", color="#38bdf8", size=36)
        if not ico.isNull():
            painter.drawPixmap((self.w - 36) // 2, 34, ico)

        # 3. Título principal AstroLoom
        f_title = QFont("Segoe UI", 20, QFont.Bold)
        f_title.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
        painter.setFont(f_title)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QRectF(0, 78, self.w, 36), Qt.AlignCenter, "AstroLoom")

        # 4. Subtítulo descriptivo
        f_sub = QFont("Segoe UI", 9)
        painter.setFont(f_sub)
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QRectF(0, 116, self.w, 20), Qt.AlignCenter, "Suite de Astrofotografía y Paisaje Nocturno")

        # 5. Badge de versión y motor
        f_ver = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(f_ver)
        painter.setPen(QColor("#64748b"))
        painter.drawText(QRectF(0, 138, self.w, 18), Qt.AlignCenter, f"v{APP_VERSION} • Rust Rayon Engine")

        # 6. Barra de progreso moderna
        bar_w = 340
        bar_h = 4
        bar_x = (self.w - bar_w) // 2
        bar_y = 185
        painter.setBrush(QBrush(QColor("#171b26")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2, 2)

        if self._progress > 0:
            fill_w = int(bar_w * (self._progress / 100.0))
            painter.setBrush(QBrush(QColor("#38bdf8")))
            painter.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 2, 2)

        # 7. Mensaje de estado en vivo
        f_msg = QFont("Segoe UI", 9)
        painter.setFont(f_msg)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QRectF(0, 205, self.w, 22), Qt.AlignCenter, self._status)

        painter.end()
        self.setPixmap(pix)
