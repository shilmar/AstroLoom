# gui/widgets/live_histogram.py
"""
gui/widgets/live_histogram.py - Histograma RGB Multicanal en Tiempo Real para AstroLoom.

Proporciona un visualizador científico de 256 niveles con compresión logarítmica,
curvas translúcidas superpuestas de los canales Rojo, Verde, Azul y Luminancia,
indicadores interactivos de recorte de sombras y altas luces (clipping)
y renderizado acelerado con QPainter.
"""
from typing import Optional, Tuple
import numpy as np
import cv2

from PySide6.QtCore import Qt, QRect, QRectF, Signal
from PySide6.QtGui import (
    QPainter, QPainterPath, QColor, QPen, QBrush, QFont,
    QLinearGradient, QCursor
)
from PySide6.QtWidgets import QWidget, QSizePolicy


class LiveRGBHistogramWidget(QWidget):
    """
    Widget de Histograma RGB en vivo para la cabecera del panel de flujo de trabajo.
    Permite visualizar la distribución tonal de la imagen a 60 FPS con compresión logarítmica
    y alertas interactivas de recorte en las esquinas.
    """
    clippingChanged = Signal(bool, bool)  # (shadows_active, highlights_active)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(220, 115)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(125)
        self.setMouseTracking(True)

        # Datos normalizados (256 elementos cada uno, 0.0 a 1.0)
        self._hist_r: Optional[np.ndarray] = None
        self._hist_g: Optional[np.ndarray] = None
        self._hist_b: Optional[np.ndarray] = None
        self._hist_lum: Optional[np.ndarray] = None

        # Estado de recorte
        self._has_shadow_clip: bool = False
        self._has_highlight_clip: bool = False
        self._clip_shadows_active: bool = False
        self._clip_highlights_active: bool = False

        # Modo de visualización: 'rgb' (todos los canales), 'r', 'g', 'b', 'lum'
        self._active_channel_mode: str = "rgb"

    def update_from_image(self, img_rgb: Optional[np.ndarray]) -> None:
        """
        Calcula de forma ultrarrápida (< 2 ms) los histogramas logarítmicos
        de los canales R, G, B y Luminancia a partir de un submuestreo de la imagen.
        """
        if img_rgb is None or img_rgb.size == 0:
            self._hist_r = None
            self._hist_g = None
            self._hist_b = None
            self._hist_lum = None
            self._has_shadow_clip = False
            self._has_highlight_clip = False
            self.update()
            return

        # Submuestreo eficiente para alta fluidez interactiva
        step = max(1, min(img_rgb.shape[0], img_rgb.shape[1]) // 256)
        sample = img_rgb[::step, ::step]

        if sample.dtype != np.float32:
            sample = sample.astype(np.float32)

        # Recorte seguro a rango 0.0 - 1.0
        clipped = np.clip(sample, 0.0, 1.0)
        u8 = (clipped * 255.0).astype(np.uint8)

        # 1. Histogramas por canal
        hist_r = cv2.calcHist([u8], [0], None, [256], [0, 256]).ravel()
        hist_g = cv2.calcHist([u8], [1], None, [256], [0, 256]).ravel()
        hist_b = cv2.calcHist([u8], [2], None, [256], [0, 256]).ravel()

        gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
        hist_lum = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()

        # Detección de píxeles en extremos
        self._has_shadow_clip = bool(hist_lum[0] > 0)
        self._has_highlight_clip = bool(hist_lum[255] > 0)

        # Compresión logarítmica para visualización astronómica
        r_log = np.log1p(hist_r)
        g_log = np.log1p(hist_g)
        b_log = np.log1p(hist_b)
        l_log = np.log1p(hist_lum)

        # Normalización compartida para mantener equilibrio relativo
        max_val = max(float(r_log.max()), float(g_log.max()), float(b_log.max()), float(l_log.max()), 1e-5)

        self._hist_r = (r_log / max_val).astype(np.float32)
        self._hist_g = (g_log / max_val).astype(np.float32)
        self._hist_b = (b_log / max_val).astype(np.float32)
        self._hist_lum = (l_log / max_val).astype(np.float32)

        self.update()

    def set_clipping_states(self, shadows: bool, highlights: bool) -> None:
        """Sincroniza el estado de las alertas de recorte con el visor general."""
        self._clip_shadows_active = shadows
        self._clip_highlights_active = highlights
        self.update()

    def toggle_shadow_clipping(self) -> None:
        """Alterna la visualización de sombras empastadas en el lienzo."""
        self._clip_shadows_active = not self._clip_shadows_active
        self.clippingChanged.emit(self._clip_shadows_active, self._clip_highlights_active)
        self.update()

    def toggle_highlight_clipping(self) -> None:
        """Alterna la visualización de luces quemadas en el lienzo."""
        self._clip_highlights_active = not self._clip_highlights_active
        self.clippingChanged.emit(self._clip_shadows_active, self._clip_highlights_active)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        margin_x = 8
        margin_top = 8
        margin_bottom = 8

        inner_w = max(1, w - 2 * margin_x)
        inner_h = max(1, h - margin_top - margin_bottom)

        # 1. Fondo de tarjeta oscura en relieve
        card_rect = QRectF(margin_x, margin_top, inner_w, inner_h)
        painter.setBrush(QBrush(QColor("#131622")))
        painter.setPen(QPen(QColor("#232738"), 1.0))
        painter.drawRoundedRect(card_rect, 6.0, 6.0)

        # 2. Rejilla y guías tonales sutiles (25%, 50%, 75%)
        pen_grid = QPen(QColor("#1c2234"), 1.0, Qt.DotLine)
        painter.setPen(pen_grid)
        for pct in (0.25, 0.50, 0.75):
            gx = margin_x + inner_w * pct
            painter.drawLine(int(gx), margin_top, int(gx), margin_top + inner_h)

        # 3. Dibujo de curvas de histograma con mezcla aditiva
        if self._hist_r is not None:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

            def make_hist_path(data: np.ndarray) -> QPainterPath:
                path = QPainterPath()
                base_y = margin_top + inner_h - 1
                path.moveTo(margin_x, base_y)
                for i in range(256):
                    px = margin_x + (i / 255.0) * inner_w
                    py = base_y - (data[i] * (inner_h - 4))
                    path.lineTo(px, py)
                path.lineTo(margin_x + inner_w, base_y)
                path.closeSubpath()
                return path

            # Canal Azul
            if self._active_channel_mode in ("rgb", "b"):
                path_b = make_hist_path(self._hist_b)
                painter.setBrush(QBrush(QColor(59, 130, 246, 50)))
                painter.setPen(QPen(QColor(59, 130, 246, 180), 1.2))
                painter.drawPath(path_b)

            # Canal Verde
            if self._active_channel_mode in ("rgb", "g"):
                path_g = make_hist_path(self._hist_g)
                painter.setBrush(QBrush(QColor(34, 197, 94, 50)))
                painter.setPen(QPen(QColor(34, 197, 94, 180), 1.2))
                painter.drawPath(path_g)

            # Canal Rojo
            if self._active_channel_mode in ("rgb", "r"):
                path_r = make_hist_path(self._hist_r)
                painter.setBrush(QBrush(QColor(239, 68, 68, 50)))
                painter.setPen(QPen(QColor(239, 68, 68, 180), 1.2))
                painter.drawPath(path_r)

            # Curva de Luminancia (trazo blanco superior)
            if self._active_channel_mode in ("rgb", "lum"):
                path_l = QPainterPath()
                base_y = margin_top + inner_h - 1
                path_l.moveTo(margin_x, base_y - (self._hist_lum[0] * (inner_h - 4)))
                for i in range(1, 256):
                    px = margin_x + (i / 255.0) * inner_w
                    py = base_y - (self._hist_lum[i] * (inner_h - 4))
                    path_l.lineTo(px, py)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(241, 245, 249, 160), 1.2))
                painter.drawPath(path_l)
        else:
            painter.setPen(QColor("#64748b"))
            font = painter.font()
            font.setPixelSize(11)
            painter.setFont(font)
            painter.drawText(card_rect, Qt.AlignCenter, "Sin datos de imagen")

        # 4. Indicadores de recorte triangulares en las esquinas
        # Sombras (Esquina Superior Izquierda)
        tl_box = QRectF(margin_x + 3, margin_top + 3, 16, 16)
        color_shadow = QColor("#00e5ff") if self._clip_shadows_active else (
            QColor("#38bdf8") if self._has_shadow_clip else QColor("#334155")
        )
        painter.setBrush(QBrush(color_shadow))
        painter.setPen(Qt.NoPen)
        tri_sh = QPainterPath()
        tri_sh.moveTo(tl_box.left() + 8, tl_box.top() + 3)
        tri_sh.lineTo(tl_box.left() + 3, tl_box.top() + 12)
        tri_sh.lineTo(tl_box.left() + 13, tl_box.top() + 12)
        tri_sh.closeSubpath()
        painter.drawPath(tri_sh)

        # Luces (Esquina Superior Derecha)
        tr_box = QRectF(margin_x + inner_w - 19, margin_top + 3, 16, 16)
        color_hl = QColor("#f43f5e") if self._clip_highlights_active else (
            QColor("#fb7185") if self._has_highlight_clip else QColor("#334155")
        )
        painter.setBrush(QBrush(color_hl))
        painter.setPen(Qt.NoPen)
        tri_hl = QPainterPath()
        tri_hl.moveTo(tr_box.left() + 8, tr_box.top() + 3)
        tri_hl.lineTo(tr_box.left() + 3, tr_box.top() + 12)
        tri_hl.lineTo(tr_box.left() + 13, tr_box.top() + 12)
        tri_hl.closeSubpath()
        painter.drawPath(tri_hl)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            margin_x = 8
            margin_top = 8
            w = self.width()
            inner_w = max(1, w - 2 * margin_x)

            tl_rect = QRect(margin_x + 1, margin_top + 1, 20, 20)
            tr_rect = QRect(margin_x + inner_w - 21, margin_top + 1, 20, 20)

            pos = event.position().toPoint()
            if tl_rect.contains(pos):
                self.toggle_shadow_clipping()
                return
            elif tr_rect.contains(pos):
                self.toggle_highlight_clipping()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        margin_x = 8
        margin_top = 8
        w = self.width()
        inner_w = max(1, w - 2 * margin_x)

        tl_rect = QRect(margin_x + 1, margin_top + 1, 20, 20)
        tr_rect = QRect(margin_x + inner_w - 21, margin_top + 1, 20, 20)

        pos = event.position().toPoint()
        if tl_rect.contains(pos):
            status = " [Activo]" if self._clip_shadows_active else " [Inactivo]"
            detect = " (Hay sombras a 0.0)" if self._has_shadow_clip else ""
            self.setToolTip(f"Alerta de Recorte de Sombras (Azul){status}{detect}\nHaz clic para alternar (Atajo: J)")
            self.setCursor(Qt.PointingHandCursor)
        elif tr_rect.contains(pos):
            status = " [Activo]" if self._clip_highlights_active else " [Inactivo]"
            detect = " (Hay luces a 1.0)" if self._has_highlight_clip else ""
            self.setToolTip(f"Alerta de Recorte de Altas Luces (Rojo){status}{detect}\nHaz clic para alternar (Atajo: J)")
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setToolTip("Histograma RGB Multicanal en Tiempo Real")
            self.setCursor(Qt.ArrowCursor)

        super().mouseMoveEvent(event)
