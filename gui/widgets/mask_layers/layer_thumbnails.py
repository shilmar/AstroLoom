# gui/widgets/mask_layers/layer_thumbnails.py
"""
Módulo de generación y gestión de miniaturas y tirador interactivo de lista de capas.
Contiene:
- THUMB_SPECS y cálculo de relación de aspecto / dimensiones adaptativas.
- LayerListResizeHandle: Tirador de arrastre libre para ajustar la altura de la lista de capas.
- Generación y caché perezosa de miniaturas RGB y B/N para capas normales y ráster.
"""
from typing import Optional, Tuple, Callable, Any
import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QColor, QCursor, QImage, QPixmap, QPainter, QPen
from PySide6.QtWidgets import QFrame, QWidget

from core.mask_layers import (
    MaskLayer,
    compute_effective_layer_mask,
    LAYER_KIND_IMAGE,
    MASK_TYPE_IMAGE_RASTER,
    MASK_TYPE_IMAGE_STARS,
    MASK_TYPE_IMAGE_STARLESS,
    MASK_TYPE_IMAGE_GROUND,
    MASK_TYPE_BRUSH,
)
from gui.icons import get_svg_pixmap

# Calibración de dimensiones para los 3 modos de miniaturas de capa (con soporte para horizontal y vertical)
THUMB_SPECS = {
    "S": {"width": 42, "height": 28, "max_w": 42, "max_h": 34, "row_height": 42, "icon_size": 12, "title_pt": 9.5},
    "M": {"width": 56, "height": 38, "max_w": 56, "max_h": 48, "row_height": 56, "icon_size": 14, "title_pt": 10},
    "L": {"width": 75, "height": 50, "max_w": 75, "max_h": 66, "row_height": 74, "icon_size": 16, "title_pt": 11},
}


class LayerListResizeHandle(QFrame):
    """
    Tirador interactivo y elegante para ajustar la altura de la lista de capas.
    Permite arrastrar hacia arriba/abajo sin restricciones de divisores rígidos,
    y realizar doble clic para volver al modo auto-ajuste adaptativo.
    """
    height_adjusted = Signal(int)
    reset_auto_fit = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setCursor(QCursor(Qt.SizeVerCursor))
        self.setToolTip("Arrastra para redimensionar la lista | Doble clic para auto-ajustar a todas las capas")
        self._dragging = False
        self._drag_start_y = 0
        self._hover = False

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        bar_w = 40 if self._hover else 28
        bar_h = 2
        bar_x = (w - bar_w) // 2
        bar_y = (h - bar_h) // 2

        color = QColor("#00e5ff") if (self._hover or self._dragging) else QColor("#475569")
        painter.fillRect(bar_x, bar_y, bar_w, bar_h, color)

        if self._hover:
            pen = QPen(QColor("#1e293b"), 1)
            painter.setPen(pen)
            painter.drawLine(0, 0, w, 0)
        painter.end()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_y = event.globalPosition().toPoint().y()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            curr_y = event.globalPosition().toPoint().y()
            dy = curr_y - self._drag_start_y
            self._drag_start_y = curr_y
            self.height_adjusted.emit(dy)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.update()
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.reset_auto_fit.emit()
            event.accept()


def get_image_aspect_ratio(
    image_reference_getter: Optional[Callable[[], Any]] = None,
    starless_getter: Optional[Callable[[], Any]] = None,
    stars_getter: Optional[Callable[[], Any]] = None,
    master_mask_getter: Optional[Callable[[], Any]] = None,
) -> float:
    """
    Determina la relación de aspecto real (ancho / alto) de la imagen de trabajo.
    Retorna 1.5 (3:2 horizontal) por defecto si aún no hay imagen cargada.
    """
    raw = None
    if image_reference_getter:
        try:
            raw = image_reference_getter()
        except Exception:
            raw = None
    if raw is None and starless_getter:
        try:
            raw = starless_getter()
        except Exception:
            raw = None
    if raw is None and stars_getter:
        try:
            raw = stars_getter()
        except Exception:
            raw = None
    if raw is None and master_mask_getter:
        try:
            raw = master_mask_getter()
        except Exception:
            raw = None
    if raw is not None and hasattr(raw, "shape") and len(raw.shape) >= 2:
        h, w = raw.shape[:2]
        if h > 0 and w > 0:
            return float(w) / float(h)
    return 1.5


def get_thumbnail_dimensions(thumb_mode: str, aspect_ratio: float) -> Tuple[int, int]:
    """
    Calcula las dimensiones (ancho, alto) de la miniatura respetando la orientación
    y relación de aspecto real de la imagen.
    """
    spec = THUMB_SPECS.get(thumb_mode, THUMB_SPECS["M"])
    max_w = spec.get("max_w", spec.get("width", 42))
    max_h = spec.get("max_h", spec.get("height", 34))

    box_ar = max_w / float(max_h)
    if aspect_ratio >= box_ar:
        tw = max_w
        th = max(12, int(round(tw / aspect_ratio)))
    else:
        th = max_h
        tw = max(12, int(round(th * aspect_ratio)))

    return tw, th


def get_layer_thumbnail_pixmap(
    layer: MaskLayer,
    thumb_mode: str,
    tw: int,
    th: int,
    image_reference_getter: Optional[Callable[[], Any]] = None,
    starless_getter: Optional[Callable[[], Any]] = None,
    stars_getter: Optional[Callable[[], Any]] = None,
    master_mask_getter: Optional[Callable[[], Any]] = None,
    layer_image_provider: Optional[Callable[..., Any]] = None,
) -> QPixmap:
    """
    Genera o recupera de la caché perezosa la miniatura de la capa según el modo actual (S/M/L).
    Capas de imagen: RGB real a escala.
    Capas de ajuste/máscaras: Máscara en escala de grises [0..255] (B/N).
    """
    cache = getattr(layer, "_thumb_pixmap_cache", None)
    if cache is None:
        layer._thumb_pixmap_cache = cache = {}

    cache_key = (
        thumb_mode, tw, th, layer.enabled, layer.inverted,
        getattr(layer, 'refine_edges', False), getattr(layer, 'refine_strength', 0.0)
    )
    if cache_key in cache:
        return cache[cache_key]

    pix = None
    # --- CASO 1: CAPAS DE IMAGEN (RGB) ---
    if layer.layer_kind == LAYER_KIND_IMAGE:
        raw_img = None
        mask_img = None
        if layer_image_provider is not None:
            try:
                try:
                    img_prov, m_prov, _ = layer_image_provider(layer, target_size=(tw, th))
                except TypeError:
                    img_prov, m_prov, _ = layer_image_provider(layer)
                if img_prov is not None:
                    raw_img = img_prov
                    mask_img = m_prov
            except Exception:
                raw_img = None

        if raw_img is None:
            if layer.mask_type == MASK_TYPE_IMAGE_RASTER and getattr(layer, 'image_data', None) is not None:
                raw_img = layer.image_data
            elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
                raw_img = stars_getter() if stars_getter else None
            elif layer.mask_type == MASK_TYPE_IMAGE_STARLESS:
                raw_img = starless_getter() if starless_getter else None
                if raw_img is None and image_reference_getter:
                    raw_img = image_reference_getter()
            else:
                raw_img = image_reference_getter() if image_reference_getter else None

        if raw_img is not None:
            if raw_img.ndim == 2:
                raw_img = np.repeat(raw_img[..., np.newaxis], 3, axis=-1)
            elif raw_img.ndim == 3 and raw_img.shape[2] == 1:
                raw_img = np.repeat(raw_img, 3, axis=-1)
            try:
                if mask_img is not None and layer.mask_type == MASK_TYPE_IMAGE_GROUND:
                    m_3d = mask_img if mask_img.ndim == 3 else mask_img[..., np.newaxis]
                    if m_3d.shape[:2] != raw_img.shape[:2]:
                        m_3d = cv2.resize(m_3d, (raw_img.shape[1], raw_img.shape[0]))
                        if m_3d.ndim == 2:
                            m_3d = m_3d[..., np.newaxis]
                    raw_img = raw_img * m_3d

                if raw_img.shape[:2] == (th, tw):
                    small = raw_img
                else:
                    small = cv2.resize(raw_img, (tw, th), interpolation=cv2.INTER_AREA)

                if layer.mask_type == MASK_TYPE_IMAGE_RASTER and not getattr(layer, 'is_transform_identity', lambda: True)():
                    sx_val = max(0.01, float(getattr(layer, 'scale_x', getattr(layer, 'scale', 1.0))))
                    sy_val = max(0.01, float(getattr(layer, 'scale_y', getattr(layer, 'scale', 1.0))))
                    a_val = float(getattr(layer, 'angle', 0.0))
                    theta = np.deg2rad(a_val)
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    a, b = sx_val * cos_t, -sy_val * sin_t
                    c, d = sx_val * sin_t, sy_val * cos_t
                    cx_th, cy_th = tw / 2.0, th / 2.0
                    tx_th = getattr(layer, 'offset_x', 0.0) * tw
                    ty_th = getattr(layer, 'offset_y', 0.0) * th
                    M_th = np.array([
                        [a, b, cx_th - a * cx_th - b * cy_th + tx_th],
                        [c, d, cy_th - c * cx_th - d * cy_th + ty_th]
                    ], dtype=np.float32)
                    small = cv2.warpAffine(small, M_th, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

                if layer.mask_type not in (MASK_TYPE_IMAGE_STARS, MASK_TYPE_IMAGE_GROUND) and small.ndim == 3 and np.percentile(small, 98) < 0.12:
                    from core.stretch import manual_stretch
                    small = manual_stretch(small, black_point=0.0, midtone=0.15)

                u8 = (np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)
                u8_contig = np.ascontiguousarray(u8)
                qimg = QImage(u8_contig.data, tw, th, tw * 3, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
            except Exception:
                pix = None

    # --- CASO 2: CAPAS DE AJUSTE / MÁSCARAS (B/N) ---
    else:
        m = None
        if layer.mask_type == MASK_TYPE_BRUSH:
            m = layer.brush_mask
        else:
            base_img = image_reference_getter() if image_reference_getter else None
            if base_img is not None:
                try:
                    m = compute_effective_layer_mask(
                        layer,
                        base_img,
                        master_sky_mask=master_mask_getter() if master_mask_getter else None,
                        starless_image=starless_getter() if starless_getter else None,
                        stars_image=stars_getter() if stars_getter else None
                    )
                except Exception:
                    m = None

        if m is not None:
            try:
                m_eval = 1.0 - m if layer.mask_type == MASK_TYPE_BRUSH and layer.inverted else m
                small = cv2.resize(m_eval, (tw, th), interpolation=cv2.INTER_AREA)
                u8 = (np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)
                u8_contig = np.ascontiguousarray(u8)
                qimg = QImage(u8_contig.data, tw, th, tw, QImage.Format_Grayscale8)
                pix = QPixmap.fromImage(qimg)
            except Exception:
                pix = None
        else:
            pix = QPixmap(tw, th)
            pix.fill(QColor("#0d1017"))
            painter = QPainter(pix)
            painter.setPen(QColor("#222a3d"))
            painter.drawRect(0, 0, tw - 1, th - 1)
            b_ico = get_svg_pixmap("brush", color="#3b4866", size=max(10, min(tw, th) - 8))
            if not b_ico.isNull():
                painter.drawPixmap((tw - b_ico.width()) // 2, (th - b_ico.height()) // 2, b_ico)
            painter.end()

    if pix is None or pix.isNull():
        pix = QPixmap(tw, th)
        pix.fill(QColor("#0a0c10"))

    if not layer.enabled:
        dimmed = QPixmap(pix.size())
        dimmed.fill(Qt.transparent)
        p = QPainter(dimmed)
        p.setOpacity(0.35)
        p.drawPixmap(0, 0, pix)
        p.end()
        pix = dimmed

    cache[cache_key] = pix
    return pix


def get_raster_mask_thumbnail_pixmap(
    layer: MaskLayer,
    thumb_mode: str,
    tw: int,
    th: int,
) -> QPixmap:
    """
    Genera o recupera de la caché la miniatura B/N de la máscara de capa de una capa ráster.
    Si la capa no tiene máscara de pincel aún, muestra un recuadro punteado de placeholder (+ M).
    """
    cache = getattr(layer, "_thumb_mask_cache", None)
    if cache is None:
        layer._thumb_mask_cache = cache = {}

    m = layer.brush_mask
    cache_key = (thumb_mode, tw, th, id(m), layer.inverted, layer.enabled)
    if cache_key in cache:
        return cache[cache_key]

    pix = None
    if m is not None:
        try:
            m_eval = 1.0 - m if layer.inverted else m
            if m_eval.ndim == 3:
                m_eval = m_eval[..., 0]
            small = cv2.resize(m_eval, (tw, th), interpolation=cv2.INTER_AREA)
            u8 = (np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)
            u8_contig = np.ascontiguousarray(u8)
            qimg = QImage(u8_contig.data, tw, th, tw, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
        except Exception:
            pix = None
    else:
        pix = QPixmap(tw, th)
        pix.fill(QColor("#080b11"))
        painter = QPainter(pix)
        pen = QPen(QColor("#334155"), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(0, 0, tw - 1, th - 1)
        painter.setPen(QColor("#64748b"))
        f = painter.font()
        f.setPointSize(max(7, min(tw, th) // 4))
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(QRect(0, 0, tw, th), Qt.AlignCenter, "+ M")
        painter.end()

    if pix is None or pix.isNull():
        pix = QPixmap(tw, th)
        pix.fill(QColor("#080b11"))

    if not layer.enabled:
        dimmed = QPixmap(pix.size())
        dimmed.fill(Qt.transparent)
        p = QPainter(dimmed)
        p.setOpacity(0.35)
        p.drawPixmap(0, 0, pix)
        p.end()
        pix = dimmed

    cache[cache_key] = pix
    return pix


def invalidate_layer_thumbnail_cache(layer: MaskLayer) -> None:
    """Limpia las cachés internas de miniaturas de una capa."""
    if hasattr(layer, "_thumb_pixmap_cache"):
        layer._thumb_pixmap_cache.clear()
    if hasattr(layer, "_thumb_mask_cache"):
        layer._thumb_mask_cache.clear()
