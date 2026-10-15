# gui/canvas.py
"""
gui/canvas.py - Lienzo interactivo acelerado para previsualización, zoom 100%
y dibujo interactivo de trazos para la segmentación de cielo y suelo.
"""
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget
from gui.widgets.drop_widgets import collect_image_files_from_mime


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
    files_dropped = Signal(list)
    color_sampled = Signal(float, float, float)  # (r, g, b) float32 [0.0, 1.0]
    eyedropper_toggled = Signal(bool)
    magnifier_toggled = Signal(bool)
    crop_rect_changed = Signal(int, int)         # (pixel_w, pixel_h)
    crop_toggled = Signal(bool)
    crop_applied = Signal()

    def __init__(self, parent: Optional[QWidget] = None, enable_masking: bool = True):
        super().__init__(parent)
        self.enable_masking = enable_masking
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._drag_hover: bool = False
        self._hover_pos: Optional[QPoint] = None
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

        # Pipeta de Balance de Blancos (Eyedropper)
        self.eyedropper_active: bool = False

        # Lupa de Inspección Flotante (Floating Magnifier)
        self.magnifier_active: bool = False
        self.magnifier_zoom: float = 2.5

        # Herramienta Interactiva de Recorte (Crop Tool)
        self.crop_mode: bool = False
        self.crop_rect_norm: QRectF = QRectF(0.0, 0.0, 1.0, 1.0)
        self.crop_aspect_ratio: Optional[float] = None
        self.crop_angle: float = 0.0
        self.crop_guides: str = "thirds"  # "thirds", "grid", "none"
        self._crop_drag_mode: Optional[str] = None
        self._crop_drag_start_pos: Optional[QPointF] = None
        self._crop_drag_start_rect: Optional[QRectF] = None
        self.native_image_size: Optional[Tuple[int, int]] = None  # (width, height) reales del sensor

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

    def set_eyedropper_active(self, active: bool) -> None:
        """Activa o desactiva la pipeta de balance de blancos interactiva."""
        if self.eyedropper_active != active:
            self.eyedropper_active = active
            if active:
                if self.magnifier_active:
                    self.set_magnifier_active(False)
                self.setCursor(Qt.CrossCursor)
            else:
                self._hover_pos = None
                if not self.magnifier_active:
                    if self.enable_masking:
                        self.update_brush_cursor()
                    else:
                        self.setCursor(Qt.ArrowCursor)
            self.eyedropper_toggled.emit(active)
            self.update()

    def set_magnifier_active(self, active: bool) -> None:
        """Activa o desactiva la lupa de inspección flotante."""
        if self.magnifier_active != active:
            self.magnifier_active = active
            if active:
                if self.eyedropper_active:
                    self.set_eyedropper_active(False)
                self.setCursor(Qt.CrossCursor)
            else:
                self._hover_pos = None
                if not self.eyedropper_active:
                    if self.enable_masking:
                        self.update_brush_cursor()
                    else:
                        self.setCursor(Qt.ArrowCursor)
            self.magnifier_toggled.emit(active)
            self.update()

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

        # -1. Manejo prioritario del Recorte Interactivo (Crop Tool)
        if self.crop_mode and event.button() == Qt.LeftButton:
            pt = event.position()
            target_rect, _, _ = self.get_render_geometry()
            hit = self._hit_test_crop(pt)
            if hit is not None:
                self._crop_drag_mode = hit
                self._crop_drag_start_pos = pt
                self._crop_drag_start_rect = QRectF(self.crop_rect_norm)
                event.accept()
                return
            elif target_rect.contains(pt):
                self._crop_drag_mode = "new"
                self._crop_drag_start_pos = pt
                norm_pt = self._widget_to_norm_img(pt)
                if norm_pt:
                    self.crop_rect_norm = QRectF(norm_pt.x(), norm_pt.y(), 0.001, 0.001)
                    self._crop_drag_start_rect = QRectF(self.crop_rect_norm)
                    self._emit_crop_size()
                    self.update()
                event.accept()
                return

        # 0. Manejo de la Pipeta de Balance de Blancos (Eyedropper)
        if self.eyedropper_active and event.button() == Qt.LeftButton:
            ipt = self.widget_to_image_coords(pos)
            if ipt and self.orig_rgb is not None:
                h, w = self.orig_rgb.shape[:2]
                ix = int(np.clip(ipt.x(), 0, w - 1))
                iy = int(np.clip(ipt.y(), 0, h - 1))
                x1 = max(0, ix - 2)
                x2 = min(w, ix + 3)
                y1 = max(0, iy - 2)
                y2 = min(h, iy + 3)
                patch = self.orig_rgb[y1:y2, x1:x2]
                r, g, b = np.mean(patch, axis=(0, 1))[:3]
                self.color_sampled.emit(float(r), float(g), float(b))
                self.set_eyedropper_active(False)
            event.accept()
            return

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

        # -1. Manejo de Recorte Interactivo (Crop Tool)
        if self.crop_mode:
            pt = event.position()
            if event.buttons() & Qt.LeftButton and self._crop_drag_mode is not None:
                self._handle_crop_drag(pt)
                event.accept()
                return
            else:
                hit = self._hit_test_crop(pt)
                self.setCursor(self._cursor_for_crop_handle(hit))
                event.accept()
                return

        # 0. Seguimiento de posición para Pipeta o Lupa Flotante
        if self.eyedropper_active or self.magnifier_active:
            self._hover_pos = pos
            self.update()
            if self.eyedropper_active:
                event.accept()
                return

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
        """Doble clic sobre la cortina divisoria para autocentrarla al 50%, o sobre el marco de recorte para aplicar."""
        if self.crop_mode and event.button() == Qt.LeftButton:
            s = self._get_crop_screen_rect()
            if not s.isEmpty() and s.contains(event.position()):
                self.crop_applied.emit()
                event.accept()
                return

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

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.crop_mode:
            if self._crop_drag_mode is not None:
                self._crop_drag_mode = None
                self._crop_drag_start_pos = None
                self._crop_drag_start_rect = None
                self._emit_crop_size()
                self.update()
                event.accept()
                return
        if event.button() == Qt.LeftButton and self.is_dragging_split:
            self.is_dragging_split = False
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_img_pt = None

    def leaveEvent(self, event) -> None:
        if self._hover_pos is not None:
            self._hover_pos = None
            self.update()
        if self.crop_mode:
            self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        """Permite ajustar dinámicamente el factor de aumento de la lupa flotante con la rueda del ratón."""
        if self.magnifier_active:
            delta = event.angleDelta().y()
            if delta > 0:
                self.magnifier_zoom = min(8.0, round(self.magnifier_zoom + 0.5, 1))
            elif delta < 0:
                self.magnifier_zoom = max(1.5, round(self.magnifier_zoom - 0.5, 1))
            self.update()
            event.accept()
            return
        super().wheelEvent(event)

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
            if self._drag_hover:
                self._draw_drag_feedback(painter)
            return

        target_rect, source_rect, _ = self.get_render_geometry()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)

        # MODO RECORTE CON ROTACIÓN ACTIVA (Nivelación de horizonte)
        if self.crop_mode and abs(self.crop_angle) > 1e-3:
            painter.save()
            center_pt = target_rect.center()
            painter.translate(center_pt)
            painter.rotate(self.crop_angle)
            painter.translate(-center_pt)
            painter.drawPixmap(target_rect, self.base_pixmap, source_rect)
            painter.restore()
        # MODO COMPARADOR DIVIDIDO ("ANTES / DESPUÉS")
        elif self.split_view_enabled and self.before_pixmap and not self.before_pixmap.isNull():
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

        # Realimentación visual al arrastrar archivos sobre el visor
        if self._drag_hover:
            self._draw_drag_feedback(painter)

        # Lupa de Inspección Flotante (Floating Magnifier)
        if self.magnifier_active:
            self._draw_magnifier(painter)

        # Insignia interactiva de la Pipeta de Balance de Blancos
        if self.eyedropper_active:
            self._draw_eyedropper_badge(painter)

        # Superposición de Recorte Interactivo (Crop Overlay)
        if self.crop_mode:
            self._draw_crop_overlay(painter)

    def _draw_magnifier(self, painter: QPainter) -> None:
        """Renderiza una lupa circular flotante de alta precisión centrada en la posición del cursor."""
        if not self._hover_pos or self.base_pixmap is None or self.base_pixmap.isNull():
            return
        target_rect, source_rect, fit_scale = self.get_render_geometry()
        if not target_rect.contains(self._hover_pos):
            return

        ipt = self.widget_to_image_coords(self._hover_pos)
        if not ipt:
            return

        pos = self._hover_pos
        cx, cy = float(pos.x()), float(pos.y())
        lens_r = 95.0
        lens_diam = lens_r * 2.0
        lens_rect = QRectF(cx - lens_r, cy - lens_r, lens_diam, lens_diam)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # 1. Sombra exterior difusa para dar profundidad flotante
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 110)))
        painter.drawEllipse(lens_rect.adjusted(3, 4, 3, 4))

        # 2. Recorte circular para el contenido magnificado
        clip_path = QPainterPath()
        clip_path.addEllipse(lens_rect)
        painter.setClipPath(clip_path)

        # 3. Dibujar contenido ampliado centrado
        zoom = self.magnifier_zoom
        eff_scale = fit_scale * zoom
        if eff_scale > 0:
            src_span = lens_diam / eff_scale
            src_x = float(ipt.x()) - (src_span / 2.0)
            src_y = float(ipt.y()) - (src_span / 2.0)
            src_rect = QRectF(src_x, src_y, src_span, src_span)

            painter.fillRect(lens_rect, QColor(14, 14, 18))
            painter.drawPixmap(lens_rect, self.base_pixmap, src_rect)

        # 4. Retícula / Punto de mira central de alta precisión
        gap = 5.0
        arm = 14.0
        painter.setPen(QPen(QColor(0, 0, 0, 200), 2.5))
        painter.drawLine(QPointF(cx - arm, cy), QPointF(cx - gap, cy))
        painter.drawLine(QPointF(cx + gap, cy), QPointF(cx + arm, cy))
        painter.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy - gap))
        painter.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + arm))

        painter.setPen(QPen(QColor(128, 216, 255, 240), 1.2))
        painter.drawLine(QPointF(cx - arm, cy), QPointF(cx - gap, cy))
        painter.drawLine(QPointF(cx + gap, cy), QPointF(cx + arm, cy))
        painter.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy - gap))
        painter.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + arm))

        painter.restore()

        # 5. Marco exterior metálico oscuro con aro cian de enfoque
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(24, 30, 42, 240), 5))
        painter.drawEllipse(lens_rect)
        painter.setPen(QPen(QColor("#80d8ff"), 2))
        painter.drawEllipse(lens_rect.adjusted(2.5, 2.5, -2.5, -2.5))

        # 6. Insignia de aumento en la parte inferior de la lente
        badge_w, badge_h = 72.0, 18.0
        badge_rect = QRectF(cx - badge_w / 2.0, cy + lens_r - badge_h - 4, badge_w, badge_h)
        painter.setBrush(QBrush(QColor(16, 22, 32, 230)))
        painter.setPen(QPen(QColor("#80d8ff"), 1))
        painter.drawRoundedRect(badge_rect, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        bf = painter.font()
        bf.setBold(True)
        bf.setPointSize(8)
        painter.setFont(bf)
        painter.drawText(badge_rect, Qt.AlignCenter, f"{zoom:.1f}x LUPA")
        painter.restore()

    def _draw_eyedropper_badge(self, painter: QPainter) -> None:
        """Dibuja la mirilla y la insignia de muestra interactiva bajo la pipeta."""
        if not self._hover_pos or self.base_pixmap is None or self.base_pixmap.isNull():
            return
        target_rect, _, _ = self.get_render_geometry()
        if not target_rect.contains(self._hover_pos):
            return

        ipt = self.widget_to_image_coords(self._hover_pos)
        if not ipt:
            return

        pos = self._hover_pos
        cx, cy = pos.x(), pos.y()

        # Obtener color muestreado (u8 de display o float de orig)
        r_u8, g_u8, b_u8 = 128, 128, 128
        if self._cached_display_u8 is not None:
            h, w = self._cached_display_u8.shape[:2]
            ix = int(np.clip(ipt.x(), 0, w - 1))
            iy = int(np.clip(ipt.y(), 0, h - 1))
            pixel = self._cached_display_u8[iy, ix]
            r_u8, g_u8, b_u8 = int(pixel[0]), int(pixel[1]), int(pixel[2])

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Mirilla de precisión en cruz con centro abierto
        cr = 12
        painter.setPen(QPen(QColor(0, 0, 0, 200), 2))
        painter.drawLine(cx - cr, cy, cx + cr, cy)
        painter.drawLine(cx, cy - cr, cx, cy + cr)
        painter.drawEllipse(QPointF(cx, cy), 4, 4)

        painter.setPen(QPen(QColor("#80d8ff"), 1))
        painter.drawLine(cx - cr, cy, cx + cr, cy)
        painter.drawLine(cx, cy - cr, cx, cy + cr)
        painter.drawEllipse(QPointF(cx, cy), 4, 4)

        # 2. Insignia flotante al lado del cursor
        bx = cx + 18
        by = cy - 44
        bw, bh = 155, 46
        if bx + bw > self.width() - 8:
            bx = cx - bw - 18
        if by < 8:
            by = cy + 18

        b_rect = QRectF(bx, by, bw, bh)
        painter.setPen(QPen(QColor("#80d8ff"), 1.2))
        painter.setBrush(QBrush(QColor(16, 22, 32, 235)))
        painter.drawRoundedRect(b_rect, 6.0, 6.0)

        # Cuadro de color de muestra
        swatch_rect = QRectF(bx + 8, by + 9, 28, 28)
        painter.setBrush(QBrush(QColor(r_u8, g_u8, b_u8)))
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        painter.drawRoundedRect(swatch_rect, 3.0, 3.0)

        # Textos de valores RGB y ayuda
        painter.setPen(QColor("#ffffff"))
        f = painter.font()
        f.setBold(True)
        f.setPointSize(9)
        painter.setFont(f)
        painter.drawText(int(bx + 42), int(by + 21), f"R:{r_u8} G:{g_u8} B:{b_u8}")

        painter.setPen(QColor("#80cbc4"))
        f.setBold(False)
        f.setPointSize(8)
        painter.setFont(f)
        painter.drawText(int(bx + 42), int(by + 36), "Clic: balancear WB")

        painter.restore()

    def _draw_drag_feedback(self, painter: QPainter) -> None:
        """Dibuja un marco delimitador punteado y una insignia informativa durante el arrastre."""
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#80d8ff"), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(20, 35, 55, 130)))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(6, 6, -6, -6), 8.0, 8.0)

        cw, ch = 270, 42
        cx = (self.width() - cw) / 2
        cy = (self.height() - ch) / 2
        pill_rect = QRectF(cx, cy, cw, ch)
        painter.setPen(QPen(QColor("#80d8ff"), 1.5))
        painter.setBrush(QBrush(QColor(16, 26, 38, 235)))
        painter.drawRoundedRect(pill_rect, 6.0, 6.0)
        painter.setPen(QColor("#ffffff"))
        f = painter.font()
        f.setBold(True)
        f.setPointSize(10)
        painter.setFont(f)
        painter.drawText(pill_rect, Qt.AlignCenter, "📥 Soltar archivo(s) para cargar")
        painter.restore()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag_hover = True
            self.update()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._drag_hover = False
        self.update()
        event.accept()

    def dropEvent(self, event) -> None:
        self._drag_hover = False
        self.update()
        paths = collect_image_files_from_mime(event.mimeData())
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

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

    # =========================================================================
    # HERRAMIENTA INTERACTIVA DE RECORTE (CROP TOOL)
    # =========================================================================

    def set_crop_mode(self, enabled: bool) -> None:
        """Activa o desactiva la herramienta interactiva de recorte."""
        if self.crop_mode != enabled:
            self.crop_mode = enabled
            if enabled:
                if self.eyedropper_active:
                    self.set_eyedropper_active(False)
                if self.magnifier_active:
                    self.set_magnifier_active(False)
                if self.split_view_enabled:
                    self.split_view_enabled = False
                    self.split_toggled.emit(False)
                if self.is_zoomed:
                    self.is_zoomed = False
                    self.zoom_center_img = None
                    self.zoom_toggled.emit(False, 0.5, 0.5)
                if self.crop_rect_norm.isEmpty() or self.crop_rect_norm.width() <= 0:
                    self.crop_rect_norm = QRectF(0.0, 0.0, 1.0, 1.0)
                if self.crop_aspect_ratio is not None and self.base_pixmap:
                    self._enforce_aspect_ratio()
                self._emit_crop_size()
            else:
                self._crop_drag_mode = None
                self._crop_drag_start_pos = None
                self._crop_drag_start_rect = None
                if not self.enable_masking:
                    self.setCursor(Qt.ArrowCursor)
            self.crop_toggled.emit(enabled)
            self.update()

    def set_native_image_size(self, size: Optional[Tuple[int, int]]) -> None:
        """Configura las dimensiones nativas reales (ancho, alto) de la imagen para el cálculo de píxeles."""
        self.native_image_size = size
        if self.crop_mode:
            self._emit_crop_size()
            self.update()

    def set_crop_aspect_ratio(self, ratio: Optional[float]) -> None:
        """Establece la relación de aspecto forzada (ancho / alto, ej. 16/9, 1.0) o None para libre."""
        self.crop_aspect_ratio = ratio
        if self.crop_mode and ratio is not None and self.base_pixmap and not self.base_pixmap.isNull():
            self._enforce_aspect_ratio()
            self._emit_crop_size()
            self.update()

    def set_crop_angle(self, angle: float) -> None:
        """Establece el ángulo de rotación / nivelación en grados (-45.0 a +45.0)."""
        self.crop_angle = float(angle)
        self.update()

    def set_crop_guides(self, guides: str) -> None:
        """Configura el tipo de guías de composición ('thirds', 'grid', 'none')."""
        if guides in ("thirds", "grid", "none"):
            self.crop_guides = guides
            self.update()

    def reset_crop_rect(self) -> None:
        """Restablece el marco de recorte al 100% de la imagen y el ángulo a 0°."""
        self.crop_rect_norm = QRectF(0.0, 0.0, 1.0, 1.0)
        self.crop_angle = 0.0
        if self.crop_aspect_ratio is not None and self.base_pixmap and not self.base_pixmap.isNull():
            self._enforce_aspect_ratio()
        self._emit_crop_size()
        self.update()

    def get_crop_pixel_rect(self, use_native: bool = True) -> Tuple[int, int, int, int]:
        """Calcula y retorna las coordenadas de recorte (x, y, ancho, alto) en píxeles de la imagen."""
        if use_native and self.native_image_size:
            iw, ih = self.native_image_size
        elif self.base_pixmap and not self.base_pixmap.isNull():
            iw = self.base_pixmap.width()
            ih = self.base_pixmap.height()
        else:
            return 0, 0, 0, 0
        r = self.crop_rect_norm
        px_x = int(round(r.x() * iw))
        px_y = int(round(r.y() * ih))
        px_w = int(round(r.width() * iw))
        px_h = int(round(r.height() * ih))

        px_x = max(0, min(iw - 1, px_x))
        px_y = max(0, min(ih - 1, px_y))
        px_w = max(10, min(iw - px_x, px_w))
        px_h = max(10, min(ih - px_y, px_h))
        return px_x, px_y, px_w, px_h

    def _get_crop_screen_rect(self) -> QRectF:
        """Calcula el rectángulo del marco de recorte proyectado en coordenadas de pantalla del widget."""
        if not self.base_pixmap or self.base_pixmap.isNull():
            return QRectF()
        target_rect, source_rect, scale = self.get_render_geometry()
        if scale <= 0:
            return QRectF()
        iw = float(self.base_pixmap.width())
        ih = float(self.base_pixmap.height())
        px_x = self.crop_rect_norm.x() * iw
        px_y = self.crop_rect_norm.y() * ih
        px_w = self.crop_rect_norm.width() * iw
        px_h = self.crop_rect_norm.height() * ih

        sx = target_rect.x() + (px_x - source_rect.x()) * scale
        sy = target_rect.y() + (px_y - source_rect.y()) * scale
        sw = px_w * scale
        sh = px_h * scale
        return QRectF(sx, sy, sw, sh)

    def _get_crop_handles(self, s: QRectF) -> Dict[str, QRectF]:
        """Calcula las cajas envolventes de los 8 tiradores de recorte en pantalla."""
        hr = 7.0
        d = hr * 2.0
        return {
            "tl": QRectF(s.left() - hr, s.top() - hr, d, d),
            "tc": QRectF(s.center().x() - hr, s.top() - hr, d, d),
            "tr": QRectF(s.right() - hr, s.top() - hr, d, d),
            "mr": QRectF(s.right() - hr, s.center().y() - hr, d, d),
            "br": QRectF(s.right() - hr, s.bottom() - hr, d, d),
            "bc": QRectF(s.center().x() - hr, s.bottom() - hr, d, d),
            "bl": QRectF(s.left() - hr, s.bottom() - hr, d, d),
            "ml": QRectF(s.left() - hr, s.center().y() - hr, d, d),
        }

    def _hit_test_crop(self, pt: QPointF) -> Optional[str]:
        """Determina sobre qué parte del marco de recorte está situado el cursor."""
        s = self._get_crop_screen_rect()
        if s.isEmpty():
            return None
        handles = self._get_crop_handles(s)
        for name, rect in handles.items():
            if rect.adjusted(-4, -4, 4, 4).contains(pt):
                return name
        if s.contains(pt):
            return "move"
        target_rect, _, _ = self.get_render_geometry()
        if target_rect.contains(pt):
            return "new"
        return None

    def _cursor_for_crop_handle(self, handle: Optional[str]) -> QCursor:
        """Retorna el cursor contextual adecuado según el tirador activo."""
        if handle in ("tl", "br"):
            return QCursor(Qt.SizeFDiagCursor)
        elif handle in ("tr", "bl"):
            return QCursor(Qt.SizeBDiagCursor)
        elif handle in ("tc", "bc"):
            return QCursor(Qt.SizeVerCursor)
        elif handle in ("ml", "mr"):
            return QCursor(Qt.SizeHorCursor)
        elif handle == "move":
            return QCursor(Qt.SizeAllCursor)
        elif handle == "new":
            return QCursor(Qt.CrossCursor)
        return QCursor(Qt.ArrowCursor)

    def _widget_to_norm_img(self, pt: QPointF) -> Optional[QPointF]:
        """Convierte una coordenada del widget en coordenada normalizada [0.0, 1.0] de la imagen."""
        target_rect, source_rect, scale = self.get_render_geometry()
        if not self.base_pixmap or self.base_pixmap.isNull() or scale <= 0:
            return None
        iw = float(self.base_pixmap.width())
        ih = float(self.base_pixmap.height())
        rel_x = (pt.x() - target_rect.x()) / scale
        rel_y = (pt.y() - target_rect.y()) / scale
        px_x = source_rect.x() + rel_x
        px_y = source_rect.y() + rel_y
        nx = float(np.clip(px_x / iw, 0.0, 1.0))
        ny = float(np.clip(px_y / ih, 0.0, 1.0))
        return QPointF(nx, ny)

    def _enforce_aspect_ratio(self, anchor: str = "center") -> None:
        """Ajusta crop_rect_norm para satisfacer exactamente crop_aspect_ratio."""
        if not self.crop_aspect_ratio or not self.base_pixmap or self.base_pixmap.isNull():
            return
        iw = float(self.base_pixmap.width())
        ih = float(self.base_pixmap.height())
        r_norm = self.crop_aspect_ratio * (ih / iw)
        r = self.crop_rect_norm
        cur_w = max(0.01, r.width())

        new_w = cur_w
        new_h = cur_w / r_norm
        if new_h > 1.0:
            new_h = 1.0
            new_w = new_h * r_norm
        if new_w > 1.0:
            new_w = 1.0
            new_h = new_w / r_norm

        if anchor == "tl":
            new_x = r.left()
            new_y = r.top()
            if new_x + new_w > 1.0:
                new_x = 1.0 - new_w
            if new_y + new_h > 1.0:
                new_y = 1.0 - new_h
        else:
            cx = r.center().x()
            cy = r.center().y()
            new_x = max(0.0, min(1.0 - new_w, cx - new_w / 2.0))
            new_y = max(0.0, min(1.0 - new_h, cy - new_h / 2.0))

        self.crop_rect_norm = QRectF(new_x, new_y, new_w, new_h)

    def _emit_crop_size(self) -> None:
        """Emite la señal de cambio de resolución del recorte actual."""
        if self.native_image_size:
            iw, ih = self.native_image_size
        elif self.base_pixmap and not self.base_pixmap.isNull():
            iw = self.base_pixmap.width()
            ih = self.base_pixmap.height()
        else:
            return
        px_w = int(round(self.crop_rect_norm.width() * iw))
        px_h = int(round(self.crop_rect_norm.height() * ih))
        self.crop_rect_changed.emit(px_w, px_h)

    def _handle_crop_drag(self, pt: QPointF) -> None:
        """Calcula la nueva geometría del marco de recorte durante el arrastre con el ratón."""
        if self._crop_drag_start_pos is None or self._crop_drag_start_rect is None or not self.base_pixmap:
            return
        target_rect, source_rect, scale = self.get_render_geometry()
        if scale <= 0:
            return

        iw = float(self.base_pixmap.width())
        ih = float(self.base_pixmap.height())
        dx_norm = (pt.x() - self._crop_drag_start_pos.x()) / (scale * iw)
        dy_norm = (pt.y() - self._crop_drag_start_pos.y()) / (scale * ih)

        orig = self._crop_drag_start_rect

        if self._crop_drag_mode == "move":
            new_x = float(np.clip(orig.x() + dx_norm, 0.0, 1.0 - orig.width()))
            new_y = float(np.clip(orig.y() + dy_norm, 0.0, 1.0 - orig.height()))
            self.crop_rect_norm = QRectF(new_x, new_y, orig.width(), orig.height())
            self._emit_crop_size()
            self.update()
            return

        if self._crop_drag_mode == "new":
            p1 = self._widget_to_norm_img(self._crop_drag_start_pos)
            p2 = self._widget_to_norm_img(pt)
            if p1 and p2:
                x1, x2 = min(p1.x(), p2.x()), max(p1.x(), p2.x())
                y1, y2 = min(p1.y(), p2.y()), max(p1.y(), p2.y())
                w = max(0.02, x2 - x1)
                h = max(0.02, y2 - y1)
                self.crop_rect_norm = QRectF(x1, y1, w, h)
                if self.crop_aspect_ratio is not None:
                    self._enforce_aspect_ratio(anchor="tl")
                self._emit_crop_size()
                self.update()
            return

        # Ajuste de bordes y esquinas
        left = orig.left()
        right = orig.right()
        top = orig.top()
        bottom = orig.bottom()

        if "l" in self._crop_drag_mode:
            left = min(right - 0.02, max(0.0, orig.left() + dx_norm))
        if "r" in self._crop_drag_mode:
            right = max(left + 0.02, min(1.0, orig.right() + dx_norm))
        if "t" in self._crop_drag_mode:
            top = min(bottom - 0.02, max(0.0, orig.top() + dy_norm))
        if "b" in self._crop_drag_mode:
            bottom = max(top + 0.02, min(1.0, orig.bottom() + dy_norm))

        new_w = right - left
        new_h = bottom - top

        # Imponer relación de aspecto fija si está configurada
        if self.crop_aspect_ratio is not None:
            r_norm = self.crop_aspect_ratio * (ih / iw)
            if self._crop_drag_mode in ("ml", "mr"):
                new_h = new_w / r_norm
                cy = (top + bottom) / 2.0
                top = max(0.0, cy - new_h / 2.0)
                bottom = min(1.0, top + new_h)
                if bottom >= 1.0:
                    top = max(0.0, 1.0 - new_h)
            elif self._crop_drag_mode in ("tc", "bc"):
                new_w = new_h * r_norm
                cx = (left + right) / 2.0
                left = max(0.0, cx - new_w / 2.0)
                right = min(1.0, left + new_w)
                if right >= 1.0:
                    left = max(0.0, 1.0 - new_w)
            elif self._crop_drag_mode == "br":
                new_h = new_w / r_norm
                bottom = top + new_h
                if bottom > 1.0:
                    bottom = 1.0
                    new_h = bottom - top
                    new_w = new_h * r_norm
                    right = left + new_w
            elif self._crop_drag_mode == "bl":
                new_h = new_w / r_norm
                bottom = top + new_h
                if bottom > 1.0:
                    bottom = 1.0
                    new_h = bottom - top
                    new_w = new_h * r_norm
                    left = right - new_w
            elif self._crop_drag_mode == "tr":
                new_h = new_w / r_norm
                top = bottom - new_h
                if top < 0.0:
                    top = 0.0
                    new_h = bottom - top
                    new_w = new_h * r_norm
                    right = left + new_w
            elif self._crop_drag_mode == "tl":
                new_h = new_w / r_norm
                top = bottom - new_h
                if top < 0.0:
                    top = 0.0
                    new_h = bottom - top
                    new_w = new_h * r_norm
                    left = right - new_w

        self.crop_rect_norm = QRectF(left, top, right - left, bottom - top)
        self._emit_crop_size()
        self.update()

    def _draw_crop_overlay(self, painter: QPainter) -> None:
        """Renderiza las zonas oscurecidas, el marco de recorte, tiradores y guías de composición."""
        if not self.crop_mode or not self.base_pixmap or self.base_pixmap.isNull():
            return
        target_rect, source_rect, scale = self.get_render_geometry()
        if scale <= 0:
            return
        s = self._get_crop_screen_rect()
        if s.isEmpty() or s.width() <= 2 or s.height() <= 2:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Zona descartada oscurecida (4 rectángulos alrededor del área de recorte)
        mask_color = QColor(0, 0, 0, 160)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(mask_color))

        # Superior
        if s.top() > target_rect.top():
            painter.drawRect(QRectF(target_rect.left(), target_rect.top(), target_rect.width(), s.top() - target_rect.top()))
        # Inferior
        if s.bottom() < target_rect.bottom():
            painter.drawRect(QRectF(target_rect.left(), s.bottom(), target_rect.width(), target_rect.bottom() - s.bottom()))
        # Izquierda
        if s.left() > target_rect.left():
            painter.drawRect(QRectF(target_rect.left(), s.top(), s.left() - target_rect.left(), s.height()))
        # Derecha
        if s.right() < target_rect.right():
            painter.drawRect(QRectF(s.right(), s.top(), target_rect.right() - s.right(), s.height()))

        # 2. Guías de composición internas
        if self.crop_guides == "thirds":
            painter.setPen(QPen(QColor(255, 255, 255, 115), 1, Qt.SolidLine))
            x1 = s.left() + s.width() / 3.0
            x2 = s.left() + 2.0 * s.width() / 3.0
            y1 = s.top() + s.height() / 3.0
            y2 = s.top() + 2.0 * s.height() / 3.0
            painter.drawLine(QPointF(x1, s.top()), QPointF(x1, s.bottom()))
            painter.drawLine(QPointF(x2, s.top()), QPointF(x2, s.bottom()))
            painter.drawLine(QPointF(s.left(), y1), QPointF(s.right(), y1))
            painter.drawLine(QPointF(s.left(), y2), QPointF(s.right(), y2))
        elif self.crop_guides == "grid":
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.SolidLine))
            for i in range(1, 6):
                gx = s.left() + (i / 6.0) * s.width()
                gy = s.top() + (i / 6.0) * s.height()
                painter.drawLine(QPointF(gx, s.top()), QPointF(gx, s.bottom()))
                painter.drawLine(QPointF(s.left(), gy), QPointF(s.right(), gy))

        # 3. Borde del marco de recorte (alta visibilidad con sombra exterior)
        painter.setPen(QPen(QColor(0, 0, 0, 180), 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(s)
        painter.setPen(QPen(QColor(255, 255, 255, 240), 1.5))
        painter.drawRect(s)

        # 4. Tiradores y esquineros tipo Lightroom / Camera Raw
        c_len = min(18.0, min(s.width(), s.height()) * 0.25)
        c_thick = 3.0
        pen_handle = QPen(QColor(255, 255, 255, 255), c_thick, Qt.SolidLine, Qt.SquareCap)
        pen_shadow = QPen(QColor(0, 0, 0, 200), c_thick + 2, Qt.SolidLine, Qt.SquareCap)

        def draw_corner(corner_pt: QPointF, dx: float, dy: float):
            painter.setPen(pen_shadow)
            painter.drawLine(corner_pt, QPointF(corner_pt.x() + dx, corner_pt.y()))
            painter.drawLine(corner_pt, QPointF(corner_pt.x(), corner_pt.y() + dy))
            painter.setPen(pen_handle)
            painter.drawLine(corner_pt, QPointF(corner_pt.x() + dx, corner_pt.y()))
            painter.drawLine(corner_pt, QPointF(corner_pt.x(), corner_pt.y() + dy))

        draw_corner(QPointF(s.left(), s.top()), c_len, c_len)
        draw_corner(QPointF(s.right(), s.top()), -c_len, c_len)
        draw_corner(QPointF(s.left(), s.bottom()), c_len, -c_len)
        draw_corner(QPointF(s.right(), s.bottom()), -c_len, -c_len)

        def draw_edge_bar(p1: QPointF, p2: QPointF):
            painter.setPen(pen_shadow)
            painter.drawLine(p1, p2)
            painter.setPen(pen_handle)
            painter.drawLine(p1, p2)

        half_len = c_len * 0.7
        draw_edge_bar(QPointF(s.center().x() - half_len, s.top()), QPointF(s.center().x() + half_len, s.top()))
        draw_edge_bar(QPointF(s.center().x() - half_len, s.bottom()), QPointF(s.center().x() + half_len, s.bottom()))
        draw_edge_bar(QPointF(s.left(), s.center().y() - half_len), QPointF(s.left(), s.center().y() + half_len))
        draw_edge_bar(QPointF(s.right(), s.center().y() - half_len), QPointF(s.right(), s.center().y() + half_len))

        # 5. Píldora de resolución flotante informativa
        if self.native_image_size:
            iw, ih = self.native_image_size
        else:
            iw = self.base_pixmap.width()
            ih = self.base_pixmap.height()
        px_w = int(round(self.crop_rect_norm.width() * iw))
        px_h = int(round(self.crop_rect_norm.height() * ih))
        badge_txt = f"{px_w} × {px_h}"
        if abs(self.crop_angle) > 1e-2:
            badge_txt += f"  ({self.crop_angle:+.1f}°)"

        font_b = painter.font()
        font_b.setPointSize(9)
        font_b.setBold(True)
        painter.setFont(font_b)

        badge_w = 125.0 if abs(self.crop_angle) > 1e-2 else 95.0
        badge_h = 22.0
        badge_x = s.center().x() - badge_w / 2.0
        badge_y = s.bottom() - badge_h - 10.0
        if badge_y < s.top() + 10.0:
            badge_y = s.top() + 10.0
        badge_rect = QRectF(badge_x, badge_y, badge_w, badge_h)

        painter.setBrush(QBrush(QColor(16, 20, 28, 220)))
        painter.setPen(QPen(QColor(80, 160, 255, 180), 1))
        painter.drawRoundedRect(badge_rect, 4.0, 4.0)
        painter.setPen(QColor("#80d8ff"))
        painter.drawText(badge_rect, Qt.AlignCenter, badge_txt)

        painter.restore()