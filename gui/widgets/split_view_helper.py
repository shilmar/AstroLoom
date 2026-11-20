"""
gui/widgets/split_view_helper.py - Componente reutilizable para comparador de cortina
interactiva (Split-View) Antes/Después con soporte vertical y horizontal.
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
)

try:
    import astroloom_rs
    HAS_RUST_RENDERER = hasattr(astroloom_rs, "render_canvas_frame_rs")
except ImportError:
    HAS_RUST_RENDERER = False


class SplitViewHelper:
    """
    Gestiona el estado, interacción de ratón y renderizado visual de una cortina divisoria
    interactiva (Split View) para comparar dos imágenes (Antes vs Después / Base vs Filtrada).
    """

    def __init__(self):
        self.enabled: bool = False
        self.orientation: str = "vertical"  # "vertical" | "horizontal"
        self.position: float = 0.5          # Coordenada normalizada [0.01, 0.99]
        self.before_pixmap: Optional[QPixmap] = None
        self.is_dragging: bool = False
        self._before_render_buf: Optional[bytearray] = None

    def set_enabled(self, enabled: bool) -> bool:
        if self.enabled != enabled:
            self.enabled = enabled
            if not enabled:
                self.is_dragging = False
            return True
        return False

    def set_orientation(self, orientation: str) -> bool:
        ori = orientation.lower().strip()
        if ori not in ("vertical", "horizontal"):
            ori = "vertical"
        if self.orientation != ori:
            self.orientation = ori
            return True
        return False

    def toggle_orientation(self) -> str:
        self.orientation = "horizontal" if self.orientation == "vertical" else "vertical"
        return self.orientation

    def set_position(self, pos: float) -> bool:
        clamped = max(0.01, min(0.99, float(pos)))
        if abs(self.position - clamped) > 1e-4:
            self.position = clamped
            return True
        return False

    def set_before_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self.before_pixmap = pixmap

    def set_before_image(self, img: Optional[np.ndarray]) -> None:
        if img is None:
            self.before_pixmap = None
            return

        h, w = img.shape[:2]
        if HAS_RUST_RENDERER:
            total_bytes = w * h * 4
            if self._before_render_buf is None or len(self._before_render_buf) != total_bytes:
                self._before_render_buf = bytearray(total_bytes)
            buf = astroloom_rs.render_canvas_frame_rs(
                np.ascontiguousarray(img),
                show_clipping=False,
                out=self._before_render_buf,
            )
            qimg = QImage(buf, w, h, 4 * w, QImage.Format_RGB32)
            self.before_pixmap = QPixmap.fromImage(qimg)
        else:
            if img.dtype != np.uint8:
                disp = cv2.convertScaleAbs(np.clip(img, 0.0, 1.0), alpha=255.0)
            else:
                disp = img
            if disp.ndim == 2:
                disp = np.stack([disp] * 3, axis=-1)
            disp_c = np.ascontiguousarray(disp, dtype=np.uint8)
            qimg = QImage(disp_c.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.before_pixmap = QPixmap.fromImage(qimg)

    def get_split_coord(self, target_rect: QRectF) -> Optional[float]:
        if not self.enabled or target_rect.isEmpty():
            return None
        if self.orientation == "horizontal":
            return target_rect.y() + target_rect.height() * self.position
        else:
            return target_rect.x() + target_rect.width() * self.position

    def handle_mouse_press(self, pos: QPoint, target_rect: QRectF, is_double_click: bool = False) -> bool:
        if not self.enabled or target_rect.isEmpty() or not target_rect.contains(pos):
            return False

        coord = self.get_split_coord(target_rect)
        if coord is None:
            return False

        cur_val = pos.y() if self.orientation == "horizontal" else pos.x()
        tolerance = 22 if is_double_click else 14

        if abs(cur_val - coord) <= tolerance:
            if is_double_click:
                self.set_position(0.5)
                self.is_dragging = False
            else:
                self.is_dragging = True
            return True

        return False

    def handle_mouse_move(self, pos: QPoint, target_rect: QRectF) -> Tuple[bool, Optional[Qt.CursorShape]]:
        if not self.enabled:
            return False, None

        if self.is_dragging:
            if not target_rect.isEmpty():
                if self.orientation == "horizontal":
                    rel = (pos.y() - target_rect.y()) / max(1.0, target_rect.height())
                else:
                    rel = (pos.x() - target_rect.x()) / max(1.0, target_rect.width())
                self.set_position(rel)
            cursor = Qt.SplitVCursor if self.orientation == "horizontal" else Qt.SplitHCursor
            return True, cursor

        if not target_rect.isEmpty() and target_rect.contains(pos):
            coord = self.get_split_coord(target_rect)
            if coord is not None:
                cur_val = pos.y() if self.orientation == "horizontal" else pos.x()
                if abs(cur_val - coord) <= 10:
                    cursor = Qt.SplitVCursor if self.orientation == "horizontal" else Qt.SplitHCursor
                    return False, cursor

        return False, None

    def handle_mouse_release(self) -> bool:
        if self.is_dragging:
            self.is_dragging = False
            return True
        return False

    def draw(
        self,
        painter: QPainter,
        target_rect: QRectF,
        before_pixmap: QPixmap,
        after_pixmap: QPixmap,
        source_rect: Optional[QRectF] = None,
        badge_before: str = "ANTES",
        badge_after: str = "DESPUÉS"
    ) -> None:
        if before_pixmap is None or before_pixmap.isNull():
            return
        if after_pixmap is None or after_pixmap.isNull():
            return

        is_horiz = (self.orientation == "horizontal")
        if is_horiz:
            split_y = target_rect.y() + target_rect.height() * self.position
            rect_antes = QRectF(target_rect.x(), target_rect.y(), target_rect.width(), max(0.0, split_y - target_rect.y()))
            rect_desp = QRectF(target_rect.x(), split_y, target_rect.width(), max(0.0, target_rect.bottom() - split_y))
            split_coord = split_y
        else:
            split_x = target_rect.x() + target_rect.width() * self.position
            rect_antes = QRectF(target_rect.x(), target_rect.y(), max(0.0, split_x - target_rect.x()), target_rect.height())
            rect_desp = QRectF(split_x, target_rect.y(), max(0.0, target_rect.right() - split_x), target_rect.height())
            split_coord = split_x

        # 1. Dibujar Zona "Antes" (Base / Original)
        painter.save()
        painter.setClipRect(rect_antes)
        if source_rect is not None:
            if before_pixmap.size() == after_pixmap.size():
                painter.drawPixmap(target_rect, before_pixmap, source_rect)
            else:
                painter.drawPixmap(target_rect, before_pixmap, QRectF(before_pixmap.rect()))
        else:
            painter.drawPixmap(target_rect.toRect(), before_pixmap)
        painter.restore()

        # 2. Dibujar Zona "Después" (Filtrada)
        painter.save()
        painter.setClipRect(rect_desp)
        if source_rect is not None:
            painter.drawPixmap(target_rect, after_pixmap, source_rect)
        else:
            painter.drawPixmap(target_rect.toRect(), after_pixmap)
        painter.restore()

        # 3. Línea Divisoria (Sombra oscura + Trazo luminoso)
        painter.setPen(QPen(QColor(0, 0, 0, 180), 4))
        if is_horiz:
            painter.drawLine(QPointF(target_rect.left(), split_coord), QPointF(target_rect.right(), split_coord))
        else:
            painter.drawLine(QPointF(split_coord, target_rect.top()), QPointF(split_coord, target_rect.bottom()))

        painter.setPen(QPen(QColor(245, 248, 255, 235), 2))
        if is_horiz:
            painter.drawLine(QPointF(target_rect.left(), split_coord), QPointF(target_rect.right(), split_coord))
        else:
            painter.drawLine(QPointF(split_coord, target_rect.top()), QPointF(split_coord, target_rect.bottom()))

        # 4. Tirador Central Circular con Flechas
        handle_r = 15.0
        if is_horiz:
            handle_cx = target_rect.center().x()
            handle_cy = split_coord
        else:
            handle_cx = split_coord
            handle_cy = target_rect.center().y()

        handle_rect = QRectF(handle_cx - handle_r, handle_cy - handle_r, handle_r * 2, handle_r * 2)

        # Sombra del tirador
        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(handle_rect.adjusted(1, 2, 1, 2))

        # Cuerpo del tirador
        painter.setBrush(QBrush(QColor("#1a1b24")))
        painter.setPen(QPen(QColor("#80d8ff"), 2))
        painter.drawEllipse(handle_rect)

        # Flechas direccionales dentro del tirador
        painter.setBrush(QBrush(QColor("#80d8ff")))
        painter.setPen(Qt.NoPen)

        if is_horiz:
            # Flechas arriba y abajo [ ▴=▾ ]
            tri_top = QPainterPath()
            tri_top.moveTo(handle_cx, handle_cy - 5.5)
            tri_top.lineTo(handle_cx - 4.5, handle_cy - 2.0)
            tri_top.lineTo(handle_cx + 4.5, handle_cy - 2.0)
            tri_top.closeSubpath()
            painter.drawPath(tri_top)

            tri_bot = QPainterPath()
            tri_bot.moveTo(handle_cx, handle_cy + 5.5)
            tri_bot.lineTo(handle_cx - 4.5, handle_cy + 2.0)
            tri_bot.lineTo(handle_cx + 4.5, handle_cy + 2.0)
            tri_bot.closeSubpath()
            painter.drawPath(tri_bot)
        else:
            # Flechas izquierda y derecha [ ◂||▸ ]
            tri_l = QPainterPath()
            tri_l.moveTo(split_coord - 5.5, handle_cy)
            tri_l.lineTo(split_coord - 2.0, handle_cy - 4.5)
            tri_l.lineTo(split_coord - 2.0, handle_cy + 4.5)
            tri_l.closeSubpath()
            painter.drawPath(tri_l)

            tri_r = QPainterPath()
            tri_r.moveTo(split_coord + 5.5, handle_cy)
            tri_r.lineTo(split_coord + 2.0, handle_cy - 4.5)
            tri_r.lineTo(split_coord + 2.0, handle_cy + 4.5)
            tri_r.closeSubpath()
            painter.drawPath(tri_r)

        # 5. Píldoras / Badges Flotantes
        font_badge = QFont()
        font_badge.setPointSize(9)
        font_badge.setBold(True)
        painter.setFont(font_badge)

        badge_w_antes = max(64.0, float(len(badge_before) * 8 + 16))
        badge_w_desp = max(64.0, float(len(badge_after) * 8 + 16))
        badge_h = 22.0

        if is_horiz:
            badge_x = target_rect.left() + 12.0
            b_antes_y = max(target_rect.top() + 8.0, split_coord - badge_h - 10.0)
            b_antes_y = min(b_antes_y, split_coord - badge_h - 6.0)
            b_antes_y = max(target_rect.top() + 8.0, b_antes_y)
            rect_b_antes = QRectF(badge_x, b_antes_y, badge_w_antes, badge_h)

            b_desp_y = min(target_rect.bottom() - badge_h - 8.0, split_coord + 10.0)
            b_desp_y = max(split_coord + 6.0, b_desp_y)
            b_desp_y = min(target_rect.bottom() - badge_h - 8.0, b_desp_y)
            rect_b_desp = QRectF(badge_x, b_desp_y, badge_w_desp, badge_h)
        else:
            badge_y = target_rect.top() + 12.0
            badge_antes_x = max(target_rect.left() + 10.0, split_coord - badge_w_antes - 12.0)
            badge_antes_x = min(badge_antes_x, split_coord - badge_w_antes - 6.0)
            badge_antes_x = max(target_rect.left() + 8.0, badge_antes_x)
            rect_b_antes = QRectF(badge_antes_x, badge_y, badge_w_antes, badge_h)

            badge_desp_x = min(target_rect.right() - badge_w_desp - 8.0, split_coord + 12.0)
            badge_desp_x = max(split_coord + 6.0, badge_desp_x)
            badge_desp_x = min(target_rect.right() - badge_w_desp - 8.0, badge_desp_x)
            rect_b_desp = QRectF(badge_desp_x, badge_y, badge_w_desp, badge_h)

        # Dibujar Badge "Antes"
        painter.setBrush(QBrush(QColor(16, 18, 24, 215)))
        painter.setPen(QPen(QColor(90, 100, 120), 1))
        painter.drawRoundedRect(rect_b_antes, 4.0, 4.0)
        painter.setPen(QColor("#eceff4"))
        painter.drawText(rect_b_antes, Qt.AlignCenter, badge_before)

        # Dibujar Badge "Después"
        painter.setBrush(QBrush(QColor(16, 24, 36, 215)))
        painter.setPen(QPen(QColor(0, 112, 255, 180), 1))
        painter.drawRoundedRect(rect_b_desp, 4.0, 4.0)
        painter.setPen(QColor("#80d8ff"))
        painter.drawText(rect_b_desp, Qt.AlignCenter, badge_after)
