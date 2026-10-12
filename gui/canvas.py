# gui/canvas.py
"""
gui/canvas.py - Lienzo interactivo acelerado para previsualización, zoom 100%
y dibujo interactivo de trazos para la segmentación de cielo y suelo.
"""
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class MaskCanvas(QWidget):
    """
    Widget de renderizado gráfico de alta precisión.
    Soporta visualización RGB normalizada [0.0, 1.0], trazos guiados multicanal,
    superposición continua de máscaras alfa, ampliación 1:1 nativa al píxel y
    modo comparador 'Antes / Después' con cortina divisoria interactiva.
    """
    brush_size_changed = Signal(int)
    zoom_toggled = Signal(bool, float, float)  # (is_zoomed, rel_x, rel_y)
    split_toggled = Signal(bool)
    split_position_changed = Signal(float)
    split_orientation_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, enable_masking: bool = True):
        super().__init__(parent)
        self.enable_masking = enable_masking
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.orig_rgb: Optional[np.ndarray] = None
        self.base_pixmap: Optional[QPixmap] = None
        self.scribble_pixmap: Optional[QPixmap] = None
        self.refined_overlay: Optional[QPixmap] = None
        self.last_mask_array: Optional[np.ndarray] = None

        self.show_mask_overlay: bool = True
        self.show_scribbles: bool = True
        self.brush_mode: int = 2            # 2: Cielo (Verde), 1: Suelo (Rojo)
        self.screen_brush_radius: int = 20
        self.last_img_pt: Optional[QPoint] = None
        self.drawing: bool = False

        # Parámetros del modo Zoom 1:1
        self.is_zoomed: bool = False
        self.zoom_center_img: Optional[QPoint] = None
        self.external_zoom_handler: bool = False  # True si el tab gestiona el recorte externo

        # Capa e indicadores de recorte de sombras (negros) y luces (blancos)
        self.show_clipping_shadows: bool = False
        self.show_clipping_highlights: bool = False
        self.clipping_overlay: Optional[QPixmap] = None
        self._cached_display_u8: Optional[np.ndarray] = None

        # Modo Comparador "Antes / Después" con Cortina Divisoria (Split View)
        self.split_view_enabled: bool = False
        self.split_position: float = 0.5   # Coordenada normalizada (0.0 a 1.0)
        self.split_orientation: str = "vertical"  # "vertical" u "horizontal"
        self.before_pixmap: Optional[QPixmap] = None
        self.is_dragging_split: bool = False

        if self.enable_masking:
            self.update_brush_cursor()

    def update_brush_cursor(self) -> None:
        """Genera dinámicamente un cursor circular concéntrico de alto contraste."""
        d = max(6, self.screen_brush_radius * 2)
        cursor_pix = QPixmap(d + 4, d + 4)
        cursor_pix.fill(Qt.transparent)

        p = QPainter(cursor_pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0, 230), 2))
        p.drawEllipse(2, 2, d, d)
        p.setPen(QPen(QColor(255, 255, 255, 230), 1))
        p.drawEllipse(2, 2, d, d)
        
        center = d // 2 + 2
        p.setPen(QPen(QColor(255, 255, 255, 240), 1))
        p.drawPoint(center, center)
        p.end()

        self.setCursor(QCursor(cursor_pix, center, center))

    def set_brush_radius(self, radius: int) -> None:
        """Actualiza el radio del pincel en coordenadas de pantalla."""
        self.screen_brush_radius = max(5, min(120, int(radius)))
        if self.enable_masking:
            self.update_brush_cursor()

    def set_mask_visible(self, visible: bool) -> None:
        """Activa o desactiva la visibilidad del overlay de la máscara refinada."""
        self.show_mask_overlay = visible
        self.update()

    def set_scribbles_visible(self, visible: bool) -> None:
        """Activa o desactiva la visualización de los trazos pintados."""
        self.show_scribbles = visible
        self.update()

    def wheelEvent(self, event) -> None:
        """Permite escalar el diámetro del pincel directamente con la rueda del ratón."""
        if not self.enable_masking:
            return
        delta = event.angleDelta().y()
        if delta != 0:
            step = 3 if delta > 0 else -3
            new_r = max(5, min(120, self.screen_brush_radius + step))
            if new_r != self.screen_brush_radius:
                self.set_brush_radius(new_r)
                self.brush_size_changed.emit(self.screen_brush_radius)
            event.accept()

    def load_image(self, rgb_float: np.ndarray, display_stretched: Optional[np.ndarray] = None) -> None:
        """
        Carga y transfiere un array float32 RGB [0.0, 1.0] a la superficie de renderizado.
        Optimizado mediante submuestreo rápido para evitar latencias en tomas de alta resolución.
        """
        self.orig_rgb = np.ascontiguousarray(rgb_float, dtype=np.float32)
        h, w = self.orig_rgb.shape[:2]

        if display_stretched is None:
            # Submuestreo rápido (stride 4) para cálculo instantáneo del percentil sin bloquear GUI
            fast_sample = self.orig_rgb[::4, ::4]
            p_hi = float(np.percentile(fast_sample, 99.5)) if np.max(fast_sample) > 0 else 1.0
            norm = np.clip(self.orig_rgb / max(p_hi, 1e-4), 0.0, 1.0)
            display = (np.power(norm, 0.4) * 255.0).astype(np.uint8)
        else:
            display = (np.clip(display_stretched, 0.0, 1.0) * 255.0).astype(np.uint8)

        self._cached_display_u8 = np.ascontiguousarray(display, dtype=np.uint8)
        qimg = QImage(self._cached_display_u8.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.base_pixmap = QPixmap.fromImage(qimg)

        # Si los indicadores de recorte están activos, actualizar su capa inmediatamente
        if self.show_clipping_shadows or self.show_clipping_highlights:
            self._update_clipping_overlay()
        else:
            self.clipping_overlay = None

        # Ajustar dimensiones de capas auxiliares si la resolución ha cambiado
        if self.enable_masking:
            if self.scribble_pixmap is None or self.scribble_pixmap.size() != self.base_pixmap.size():
                self.scribble_pixmap = QPixmap(w, h)
                self.scribble_pixmap.fill(Qt.transparent)

        # Si hay una máscara previa activa, regenerar el overlay ajustado a la nueva resolución
        if self.last_mask_array is not None and self.show_mask_overlay:
            self.set_refined_mask(self.last_mask_array)
        elif self.refined_overlay and self.refined_overlay.size() != self.base_pixmap.size():
            self.refined_overlay = None

        # Resetear centro de zoom si la nueva resolución es diferente
        if self.zoom_center_img:
            if self.zoom_center_img.x() >= w or self.zoom_center_img.y() >= h:
                self.zoom_center_img = QPoint(w // 2, h // 2)

        self.update()

    def _update_clipping_overlay(self) -> None:
        """
        Calcula la máscara de recorte de sombras y luces de forma vectorial acelerada.
        Azul vibrante [0, 112, 255] para sombras empastadas (negro absoluto = 0).
        Rojo vibrante [255, 23, 68] para luces quemadas (blanco absoluto = 255).
        """
        if not self.show_clipping_shadows and not self.show_clipping_highlights:
            self.clipping_overlay = None
            self.update()
            return

        if self._cached_display_u8 is None or self._cached_display_u8.size == 0:
            self.clipping_overlay = None
            self.update()
            return

        disp = self._cached_display_u8
        h, w = disp.shape[:2]

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        has_overlay = False

        if self.show_clipping_shadows:
            # Sombras empastadas: todos los canales RGB en 0 (pérdida total de detalle en negros)
            shadow_mask = (disp[..., 0] == 0) & (disp[..., 1] == 0) & (disp[..., 2] == 0)
            if np.any(shadow_mask):
                has_overlay = True
                rgba[shadow_mask] = [0, 112, 255, 220]  # Azul vibrante (Lightroom style)

        if self.show_clipping_highlights:
            # Luces quemadas: saturación en al menos un canal a 255 (pérdida de rango dinámico)
            highlight_mask = (disp[..., 0] >= 255) | (disp[..., 1] >= 255) | (disp[..., 2] >= 255)
            if np.any(highlight_mask):
                has_overlay = True
                rgba[highlight_mask] = [255, 23, 68, 220]  # Rojo vibrante (Lightroom style)

        if has_overlay:
            rgba_contiguous = np.ascontiguousarray(rgba)
            qimg = QImage(rgba_contiguous.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            self.clipping_overlay = QPixmap.fromImage(qimg)
        else:
            self.clipping_overlay = None

        self.update()

    def set_clipping_state(self, show_shadows: bool, show_highlights: bool) -> None:
        """Activa o desactiva la visualización de los recortes de sombras y luces."""
        if self.show_clipping_shadows != show_shadows or self.show_clipping_highlights != show_highlights:
            self.show_clipping_shadows = show_shadows
            self.show_clipping_highlights = show_highlights
            self._update_clipping_overlay()

    def set_split_view_enabled(self, enabled: bool) -> None:
        """Activa o desactiva la visualización dividida con cortina interactiva."""
        if self.split_view_enabled != enabled:
            self.split_view_enabled = enabled
            if enabled and (self.before_pixmap is None or self.before_pixmap.isNull()) and self.base_pixmap:
                self.before_pixmap = self.base_pixmap.copy()
            self.split_toggled.emit(enabled)
            self.update()

    def set_before_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        """Establece la imagen de referencia previa ('Antes') para el comparador."""
        self.before_pixmap = pixmap
        if self.split_view_enabled:
            self.update()

    def set_split_position(self, pos: float) -> None:
        """Ajusta la posición normalizada (0.01 a 0.99) de la cortina divisoria."""
        clamped = max(0.01, min(0.99, float(pos)))
        if abs(self.split_position - clamped) > 1e-4:
            self.split_position = clamped
            self.split_position_changed.emit(self.split_position)
            self.update()

    def set_split_orientation(self, orientation: str) -> None:
        """Establece la orientación de la cortina ('vertical' u 'horizontal')."""
        ori = orientation.lower().strip()
        if ori not in ("vertical", "horizontal"):
            ori = "vertical"
        if self.split_orientation != ori:
            self.split_orientation = ori
            self.split_orientation_changed.emit(ori)
            self.update()

    def toggle_split_orientation(self) -> str:
        """Alterna entre cortina vertical y horizontal."""
        new_ori = "horizontal" if self.split_orientation == "vertical" else "vertical"
        self.set_split_orientation(new_ori)
        return new_ori

    def _get_split_screen_coord(self) -> Optional[float]:
        """Calcula la coordenada en pantalla de la cortina (x para vertical, y para horizontal)."""
        if not self.split_view_enabled:
            return None
        target_rect, _, _ = self.get_render_geometry()
        if target_rect.isEmpty():
            return None
        if self.split_orientation == "horizontal":
            if target_rect.height() <= 0:
                return None
            return target_rect.y() + target_rect.height() * self.split_position
        else:
            if target_rect.width() <= 0:
                return None
            return target_rect.x() + target_rect.width() * self.split_position

    def _get_split_screen_x(self) -> Optional[float]:
        """Calcula la coordenada horizontal en pantalla de la cortina divisoria."""
        if not self.split_view_enabled or self.split_orientation == "horizontal":
            return None
        return self._get_split_screen_coord()

    def get_render_geometry(self) -> Tuple[QRectF, QRectF, float]:
        """Calcula los rectángulos de origen y destino para el escalado proporcional de la imagen."""
        if not self.base_pixmap or self.base_pixmap.isNull():
            return QRectF(), QRectF(), 1.0

        w_w, h_w = float(self.width()), float(self.height())
        w_i, h_i = float(self.base_pixmap.width()), float(self.base_pixmap.height())

        fit_scale = min(w_w / w_i, h_w / h_i)

        if not self.is_zoomed or self.external_zoom_handler:
            tw, th = w_i * fit_scale, h_i * fit_scale
            target_rect = QRectF((w_w - tw) / 2.0, (h_w - th) / 2.0, tw, th)
            source_rect = QRectF(0.0, 0.0, w_i, h_i)
            return target_rect, source_rect, fit_scale
        else:
            # Modo zoom 1:1 local centrado
            cx = float(self.zoom_center_img.x()) if self.zoom_center_img else (w_i / 2.0)
            cy = float(self.zoom_center_img.y()) if self.zoom_center_img else (h_i / 2.0)

            crop_w = min(w_i, w_w)
            crop_h = min(h_i, h_w)

            src_x = max(0.0, min(w_i - crop_w, cx - crop_w / 2.0))
            src_y = max(0.0, min(h_i - crop_h, cy - crop_h / 2.0))

            source_rect = QRectF(src_x, src_y, crop_w, crop_h)
            target_rect = QRectF((w_w - crop_w) / 2.0, (h_w - crop_h) / 2.0, crop_w, crop_h)
            return target_rect, source_rect, 1.0

    def widget_to_image_coords(self, pt: QPoint) -> Optional[QPoint]:
        """Transforma coordenadas de la ventana (widget) a píxeles de la imagen original."""
        target_rect, source_rect, scale = self.get_render_geometry()
        if not target_rect.contains(pt) or scale <= 0:
            return None

        rel_x = (pt.x() - target_rect.x()) / scale
        rel_y = (pt.y() - target_rect.y()) / scale

        img_x = int(source_rect.x() + rel_x)
        img_y = int(source_rect.y() + rel_y)

        if not self.base_pixmap:
            return None

        img_x = max(0, min(self.base_pixmap.width() - 1, img_x))
        img_y = max(0, min(self.base_pixmap.height() - 1, img_y))
        return QPoint(img_x, img_y)

    def mousePressEvent(self, event) -> None:
        if not self.base_pixmap or self.base_pixmap.isNull():
            return

        pos = event.position().toPoint()

        # 1. Comprobar interacción con la Cortina Divisoria (Split View)
        if self.split_view_enabled and event.button() == Qt.LeftButton:
            split_coord = self._get_split_screen_coord()
            target_rect, _, _ = self.get_render_geometry()
            if split_coord is not None and not target_rect.isEmpty() and target_rect.contains(pos):
                cur_val = pos.y() if self.split_orientation == "horizontal" else pos.x()
                if abs(cur_val - split_coord) <= 14:
                    self.is_dragging_split = True
                    cursor_shape = Qt.SplitVCursor if self.split_orientation == "horizontal" else Qt.SplitHCursor
                    self.setCursor(cursor_shape)
                    event.accept()
                    return

        # 2. Alternar Zoom con Clic Derecho
        if event.button() == Qt.RightButton:
            ipt = self.widget_to_image_coords(event.position().toPoint())
            w_i = float(self.base_pixmap.width())
            h_i = float(self.base_pixmap.height())

            if not self.is_zoomed:
                self.is_zoomed = True
                self.zoom_center_img = ipt if ipt else QPoint(int(w_i / 2), int(h_i / 2))
                rel_x = self.zoom_center_img.x() / max(1.0, w_i)
                rel_y = self.zoom_center_img.y() / max(1.0, h_i)
                self.zoom_toggled.emit(True, rel_x, rel_y)
            else:
                self.is_zoomed = False
                self.zoom_center_img = None
                self.zoom_toggled.emit(False, 0.5, 0.5)

            if not self.external_zoom_handler:
                self.update()
            event.accept()
            return

        if not self.enable_masking:
            return

        if event.button() == Qt.LeftButton:
            self.drawing = True
            ipt = self.widget_to_image_coords(event.position().toPoint())
            if ipt:
                self.last_img_pt = ipt
                self.paint_on_scribble(ipt, ipt)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()

        # 1. Arrastre y hover sobre la cortina divisoria
        if self.split_view_enabled:
            target_rect, _, _ = self.get_render_geometry()
            if self.is_dragging_split:
                if not target_rect.isEmpty():
                    if self.split_orientation == "horizontal":
                        if target_rect.height() > 0:
                            rel_pos = (pos.y() - target_rect.y()) / target_rect.height()
                            self.set_split_position(rel_pos)
                    else:
                        if target_rect.width() > 0:
                            rel_pos = (pos.x() - target_rect.x()) / target_rect.width()
                            self.set_split_position(rel_pos)
                event.accept()
                return

            split_coord = self._get_split_screen_coord()
            if split_coord is not None and not target_rect.isEmpty() and target_rect.contains(pos):
                cur_val = pos.y() if self.split_orientation == "horizontal" else pos.x()
                if abs(cur_val - split_coord) <= 10:
                    cursor_shape = Qt.SplitVCursor if self.split_orientation == "horizontal" else Qt.SplitHCursor
                    self.setCursor(cursor_shape)
                    event.accept()
                    return
                elif not self.enable_masking:
                    self.setCursor(Qt.ArrowCursor)

        if not self.enable_masking:
            return
        if self.drawing and self.base_pixmap and not self.base_pixmap.isNull():
            ipt = self.widget_to_image_coords(event.position().toPoint())
            if ipt and self.last_img_pt:
                self.paint_on_scribble(self.last_img_pt, ipt)
                self.last_img_pt = ipt

    def mouseDoubleClickEvent(self, event) -> None:
        """Doble clic sobre la cortina divisoria para autocentrarla al 50%."""
        if self.split_view_enabled and event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            split_coord = self._get_split_screen_coord()
            target_rect, _, _ = self.get_render_geometry()
            if split_coord is not None and not target_rect.isEmpty() and target_rect.contains(pos):
                cur_val = pos.y() if self.split_orientation == "horizontal" else pos.x()
                if abs(cur_val - split_coord) <= 22:
                    self.set_split_position(0.5)
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.is_dragging_split:
            self.is_dragging_split = False
            event.accept()
            return
        if not self.enable_masking:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_img_pt = None

    def paint_on_scribble(self, p1: QPoint, p2: QPoint) -> None:
        """Dibuja un trazo o punto continuo sobre la capa de semillas de segmentación."""
        _, _, scale = self.get_render_geometry()
        if scale <= 0 or not self.scribble_pixmap: 
            return

        native_d = max(2, int((self.screen_brush_radius * 2) / scale))
        painter = QPainter(self.scribble_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        color = QColor(40, 240, 60, 70) if self.brush_mode == 2 else QColor(240, 40, 40, 70)
        painter.setPen(QPen(color, native_d, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        
        if p1 == p2:
            painter.drawPoint(p1)
        else:
            painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))
        if not self.base_pixmap or self.base_pixmap.isNull():
            painter.setPen(QColor(130, 130, 130))
            painter.drawText(self.rect(), Qt.AlignCenter, "Carga una toma o imagen para previsualizar aquí.")
            return

        target_rect, source_rect, _ = self.get_render_geometry()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)

        # MODO COMPARADOR DIVIDIDO ("ANTES / DESPUÉS")
        if self.split_view_enabled and self.before_pixmap and not self.before_pixmap.isNull():
            is_horiz = (self.split_orientation == "horizontal")

            if is_horiz:
                split_y = target_rect.y() + target_rect.height() * self.split_position
                rect_antes = QRectF(target_rect.x(), target_rect.y(), target_rect.width(), max(0.0, split_y - target_rect.y()))
                rect_desp = QRectF(target_rect.x(), split_y, target_rect.width(), max(0.0, target_rect.bottom() - split_y))
            else:
                split_x = target_rect.x() + target_rect.width() * self.split_position
                rect_antes = QRectF(target_rect.x(), target_rect.y(), max(0.0, split_x - target_rect.x()), target_rect.height())
                rect_desp = QRectF(split_x, target_rect.y(), max(0.0, target_rect.right() - split_x), target_rect.height())

            # 1. Zona "Antes" (Arriba si horizontal, Izquierda si vertical)
            painter.save()
            painter.setClipRect(rect_antes)
            if self.before_pixmap.size() == self.base_pixmap.size():
                painter.drawPixmap(target_rect, self.before_pixmap, source_rect)
            else:
                painter.drawPixmap(target_rect, self.before_pixmap, QRectF(self.before_pixmap.rect()))
            painter.restore()

            # 2. Zona "Después" (Abajo si horizontal, Derecha si vertical)
            painter.save()
            painter.setClipRect(rect_desp)
            painter.drawPixmap(target_rect, self.base_pixmap, source_rect)
            if self.clipping_overlay and not self.clipping_overlay.isNull():
                painter.drawPixmap(target_rect, self.clipping_overlay, source_rect)
            if self.show_mask_overlay and self.refined_overlay and not self.refined_overlay.isNull():
                painter.drawPixmap(target_rect, self.refined_overlay, source_rect)
            if self.enable_masking and self.show_scribbles and self.scribble_pixmap and not self.scribble_pixmap.isNull():
                painter.drawPixmap(target_rect, self.scribble_pixmap, source_rect)
            painter.restore()

            # 3. Línea Divisoria (Horizontal o Vertical)
            painter.setPen(QPen(QColor(0, 0, 0, 180), 4))
            if is_horiz:
                painter.drawLine(QPointF(target_rect.left(), split_y), QPointF(target_rect.right(), split_y))
            else:
                painter.drawLine(QPointF(split_x, target_rect.top()), QPointF(split_x, target_rect.bottom()))

            painter.setPen(QPen(QColor(245, 248, 255, 235), 2))
            if is_horiz:
                painter.drawLine(QPointF(target_rect.left(), split_y), QPointF(target_rect.right(), split_y))
            else:
                painter.drawLine(QPointF(split_x, target_rect.top()), QPointF(split_x, target_rect.bottom()))

            # 4. Tirador Central Circular
            handle_r = 15.0
            if is_horiz:
                handle_cx = target_rect.center().x()
                handle_cy = split_y
            else:
                handle_cx = split_x
                handle_cy = target_rect.center().y()

            handle_rect = QRectF(handle_cx - handle_r, handle_cy - handle_r, handle_r * 2, handle_r * 2)

            painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(handle_rect.adjusted(1, 2, 1, 2))

            painter.setBrush(QBrush(QColor("#1a1b24")))
            painter.setPen(QPen(QColor("#80d8ff"), 2))
            painter.drawEllipse(handle_rect)

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
                tri_l.moveTo(split_x - 5.5, handle_cy)
                tri_l.lineTo(split_x - 2.0, handle_cy - 4.5)
                tri_l.lineTo(split_x - 2.0, handle_cy + 4.5)
                tri_l.closeSubpath()
                painter.drawPath(tri_l)

                tri_r = QPainterPath()
                tri_r.moveTo(split_x + 5.5, handle_cy)
                tri_r.lineTo(split_x + 2.0, handle_cy - 4.5)
                tri_r.lineTo(split_x + 2.0, handle_cy + 4.5)
                tri_r.closeSubpath()
                painter.drawPath(tri_r)

            # 5. Píldoras Flotantes Identificativas [ ANTES ] / [ DESPUÉS ]
            font_badge = painter.font()
            font_badge.setPointSize(9)
            font_badge.setBold(True)
            painter.setFont(font_badge)

            badge_antes_w, badge_desp_w, badge_h = 64.0, 76.0, 22.0

            if is_horiz:
                badge_x = target_rect.left() + 12.0
                b_antes_y = max(target_rect.top() + 8.0, split_y - badge_h - 10.0)
                b_antes_y = min(b_antes_y, split_y - badge_h - 6.0)
                b_antes_y = max(target_rect.top() + 8.0, b_antes_y)
                rect_b_antes = QRectF(badge_x, b_antes_y, badge_antes_w, badge_h)

                b_desp_y = min(target_rect.bottom() - badge_h - 8.0, split_y + 10.0)
                b_desp_y = max(split_y + 6.0, b_desp_y)
                b_desp_y = min(target_rect.bottom() - badge_h - 8.0, b_desp_y)
                rect_b_desp = QRectF(badge_x, b_desp_y, badge_desp_w, badge_h)
            else:
                badge_y = target_rect.top() + 12.0
                badge_antes_x = max(target_rect.left() + 10.0, split_x - badge_antes_w - 12.0)
                badge_antes_x = min(badge_antes_x, split_x - badge_antes_w - 6.0)
                badge_antes_x = max(target_rect.left() + 8.0, badge_antes_x)
                rect_b_antes = QRectF(badge_antes_x, badge_y, badge_antes_w, badge_h)

                badge_desp_x = min(target_rect.right() - badge_desp_w - 8.0, split_x + 12.0)
                badge_desp_x = max(split_x + 6.0, badge_desp_x)
                badge_desp_x = min(target_rect.right() - badge_desp_w - 8.0, badge_desp_x)
                rect_b_desp = QRectF(badge_desp_x, badge_y, badge_desp_w, badge_h)

            painter.setBrush(QBrush(QColor(16, 18, 24, 215)))
            painter.setPen(QPen(QColor(90, 100, 120), 1))
            painter.drawRoundedRect(rect_b_antes, 4.0, 4.0)
            painter.setPen(QColor("#eceff4"))
            painter.drawText(rect_b_antes, Qt.AlignCenter, "ANTES")

            painter.setBrush(QBrush(QColor(16, 24, 36, 215)))
            painter.setPen(QPen(QColor(0, 112, 255, 180), 1))
            painter.drawRoundedRect(rect_b_desp, 4.0, 4.0)
            painter.setPen(QColor("#80d8ff"))
            painter.drawText(rect_b_desp, Qt.AlignCenter, "DESPUÉS")

        else:
            # MODO COMPLETO ESTÁNDAR
            painter.drawPixmap(target_rect, self.base_pixmap, source_rect)

            # Overlay de recorte de sombras y luces
            if self.clipping_overlay and not self.clipping_overlay.isNull():
                painter.drawPixmap(target_rect, self.clipping_overlay, source_rect)

            # Overlay de máscara refinada
            if self.show_mask_overlay:
                if self.refined_overlay and not self.refined_overlay.isNull():
                    painter.drawPixmap(target_rect, self.refined_overlay, source_rect)
                
            if self.enable_masking and self.show_scribbles and self.scribble_pixmap and not self.scribble_pixmap.isNull():
                painter.drawPixmap(target_rect, self.scribble_pixmap, source_rect)

    def get_scribbles_matrix(self) -> Optional[np.ndarray]:
        """
        Extrae la matriz bidimensional de etiquetas de usuario.
        Retorna:
            np.ndarray uint8: 0 (sin marcar), 1 (suelo seguro), 2 (cielo seguro).
        """
        if not self.scribble_pixmap or self.scribble_pixmap.isNull(): 
            return None
        
        qimg = self.scribble_pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        arr = np.array(qimg.constBits(), dtype=np.uint8, copy=True).reshape((qimg.height(), qimg.width(), 4))
        scribbles = np.zeros((qimg.height(), qimg.width()), dtype=np.uint8)
        
        has_stroke = arr[..., 3] > 20
        scribbles[(arr[..., 0] > arr[..., 1]) & has_stroke] = 1
        scribbles[(arr[..., 1] > arr[..., 0]) & has_stroke] = 2
        return scribbles

    def set_refined_mask(self, mask_float: Optional[np.ndarray]) -> None:
        """
        Genera el overlay semitransparente coloreado para la máscara calculada.
        Rojo: Suelo (0.0). Verde: Cielo (1.0).
        """
        if mask_float is None:
            self.last_mask_array = None
            self.refined_overlay = None
            self.update()
            return

        self.last_mask_array = mask_float

        if mask_float.ndim == 3:
            mask_float = mask_float[..., 0]

        mask_f = mask_float.astype(np.float32)
        if mask_f.max() > 1.05:
            mask_f = mask_f / 255.0
        mask_f = np.clip(mask_f, 0.0, 1.0)

        h, w = mask_f.shape

        # Si hay una imagen base en el canvas, asegurar coincidencia exacta de resolución
        if self.base_pixmap and not self.base_pixmap.isNull():
            bw = self.base_pixmap.width()
            bh = self.base_pixmap.height()
            if (w != bw or h != bh) and bw > 0 and bh > 0:
                mask_f = cv2.resize(mask_f, (bw, bh), interpolation=cv2.INTER_LINEAR)
                h, w = bh, bw

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # Tinte rojo para suelo, verde para cielo con opacidad suave
        rgba[..., 0] = ((1.0 - mask_f) * 220).astype(np.uint8)
        rgba[..., 1] = (mask_f * 220).astype(np.uint8)
        rgba[..., 2] = 30
        rgba[..., 3] = 50  # Opacidad semitransparente óptima

        rgba_contiguous = np.ascontiguousarray(rgba)
        qimg = QImage(rgba_contiguous.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self.refined_overlay = QPixmap.fromImage(qimg)
        self.update()

    def _create_mask_pixmap(self, mask_float: np.ndarray) -> Optional[QPixmap]:
        """Método de compatibilidad para generar el QPixmap de la máscara."""
        self.set_refined_mask(mask_float)
        return self.refined_overlay