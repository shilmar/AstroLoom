# gui/canvas.py
"""
gui/canvas.py - Lienzo interactivo acelerado para previsualización, zoom 100%
y dibujo interactivo de trazos para la segmentación de cielo y suelo.
"""
from typing import Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class MaskCanvas(QWidget):
    """
    Widget de renderizado gráfico de alta precisión.
    Soporta visualización RGB normalizada [0.0, 1.0], trazos guiados multicanal,
    superposición continua de máscaras alfa y ampliación 1:1 nativa al píxel.
    """
    brush_size_changed = Signal(int)
    zoom_toggled = Signal(bool, float, float)  # (is_zoomed, rel_x, rel_y)

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

        # Alternar Zoom con Clic Derecho
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
        if not self.enable_masking:
            return
        if self.drawing and self.base_pixmap and not self.base_pixmap.isNull():
            ipt = self.widget_to_image_coords(event.position().toPoint())
            if ipt and self.last_img_pt:
                self.paint_on_scribble(self.last_img_pt, ipt)
                self.last_img_pt = ipt

    def mouseReleaseEvent(self, event) -> None:
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
        painter.drawPixmap(target_rect, self.base_pixmap, source_rect)

        # Overlay de recorte de sombras y luces (directamente sobre la imagen base)
        if self.clipping_overlay and not self.clipping_overlay.isNull():
            painter.drawPixmap(target_rect, self.clipping_overlay, source_rect)

        # Overlay de máscara (visible si show_mask_overlay es True, independientemente de enable_masking)
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