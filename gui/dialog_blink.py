# gui/dialog_blink.py
"""
gui/dialog_blink.py - Visor Rápido de Subframes (Blink & Culling Interactivo).

Permite inspeccionar secuencias de tomas RAW/FITS/TIFF a alta velocidad (1 a 15 fps),
con estirado adaptativo en tiempo real (Display Stretch LUT) para revelar detalles de cielo profundo
en tomas oscuras, análisis de calidad óptica integrado y atajos de teclado rápidos ('X' para descartar,
'P' para conservar, flechas o 'A'/'D' para navegar y barra espaciadora para reproducir/pausar).
"""

from __future__ import annotations

import os
from typing import List, Set, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QComboBox, QFrame, QSizePolicy, QApplication,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize, QPointF, QRectF, QRect, QModelIndex
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QColor, QPainter, QWheelEvent, QMouseEvent, QFont, QPalette, QBrush

from gui.theme import apply_theme, BG_CARD, BORDER_BASE, TEXT_MUTED, BG_CANVAS, BG_PANEL
from gui.icons import get_icon


def _safe_cv2_read(filepath: str) -> Optional[np.ndarray]:
    """Lee una imagen con OpenCV soportando rutas con caracteres especiales en Windows."""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(filepath, cv2.IMREAD_COLOR)


def _fast_load_and_stretch(filepath: str, auto_stretch: bool = True, max_dim: int = 1280) -> Optional[np.ndarray]:
    """Carga y procesa un subframe a resolución proxy con estirado de previsualización."""
    if not os.path.exists(filepath):
        return None

    ext = os.path.splitext(filepath)[1].lower()
    img_rgb = None

    try:
        # Carga acelerada según tipo de archivo
        if ext in ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw']:
            import rawpy
            with rawpy.imread(filepath) as raw:
                # half_size=True hace el debayer 4x más rápido para previsualización blink
                rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                img_rgb = rgb.astype(np.float32) / 255.0
        elif ext in ['.fits', '.fit', '.fts']:
            data = None
            try:
                from core.fits_io import read_fits
                data = read_fits(filepath)
            except Exception:
                try:
                    from astropy.io import fits
                    with fits.open(filepath) as hdul:
                        data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                except Exception:
                    data = None

            if data is not None:
                data = np.nan_to_num(data.astype(np.float32))
                if data.ndim == 2:
                    d_min, d_max = float(np.min(data)), float(np.max(data))
                    if d_max > d_min:
                        data = (data - d_min) / (d_max - d_min)
                    img_rgb = np.repeat(data[..., np.newaxis], 3, axis=2)
                elif data.ndim == 3 and data.shape[0] in (3, 4):
                    data = np.transpose(data[:3], (1, 2, 0))
                    d_min, d_max = float(np.min(data)), float(np.max(data))
                    if d_max > d_min:
                        data = (data - d_min) / (d_max - d_min)
                    img_rgb = data
                elif data.ndim == 3:
                    d_min, d_max = float(np.min(data)), float(np.max(data))
                    if d_max > d_min:
                        data = (data - d_min) / (d_max - d_min)
                    img_rgb = data
        elif ext in ['.tif', '.tiff']:
            try:
                import tifffile
                tdata = tifffile.imread(filepath).astype(np.float32)
                if tdata.ndim == 2:
                    tdata = np.repeat(tdata[..., np.newaxis], 3, axis=2)
                d_max = float(tdata.max())
                if d_max > 1.05:
                    tdata /= (65535.0 if d_max > 255.0 else 255.0)
                img_rgb = np.clip(tdata, 0.0, 1.0)
            except Exception:
                bgr = _safe_cv2_read(filepath)
                if bgr is not None:
                    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            bgr = _safe_cv2_read(filepath)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception:
        # Fallback genérico
        try:
            bgr = _safe_cv2_read(filepath)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        except Exception:
            return None

    if img_rgb is None:
        return None

    # Redimensionar al tamaño proxy si supera max_dim
    h, w = img_rgb.shape[:2]
    max_side = max(h, w)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    # Estirado automático de histograma / MTF para tomas lineales oscuras
    if auto_stretch:
        sample = img_rgb[::4, ::4]
        bp = float(np.percentile(sample, 1.0))
        wp = float(np.percentile(sample, 99.8))
        if wp > bp + 1e-4:
            norm = np.clip((img_rgb - bp) / (wp - bp), 0.0, 1.0)
            # Curva gamma de realce para medios tonos astronómicos
            stretched = np.power(norm, 0.45)
            out_u8 = (stretched * 255.0).astype(np.uint8)
        else:
            out_u8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        out_u8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)

    return out_u8


class BlinkCanvas(QWidget):
    """Lienzo interactivo optimizado para el visor Blink con escalado, zoom 1:1, pan y navegación."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #07090e; border: 1px solid #1a2233; border-radius: 6px;")
        self.setMouseTracking(True)

        self._pixmap: Optional[QPixmap] = None
        self._fit_mode: bool = True
        self._zoom_scale: float = 1.0
        self._pan_offset: QPointF = QPointF(0.0, 0.0)
        self._is_panning: bool = False
        self._last_mouse_pos: QPointF = QPointF(0.0, 0.0)

    def set_image(self, img_u8: Optional[np.ndarray]):
        if img_u8 is None:
            self._pixmap = None
        else:
            h, w = img_u8.shape[:2]
            img_contig = np.ascontiguousarray(img_u8)
            qimg = QImage(img_contig.data, w, h, w * 3, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def reset_view(self):
        """Restaura la vista completa ajustada a la ventana (Fit)."""
        self._fit_mode = True
        self._zoom_scale = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return

        # Si Ctrl está presionado -> Zoom sobre la imagen
        if event.modifiers() & Qt.ControlModifier:
            if not self._pixmap or self._pixmap.isNull():
                return
            pw, ph = float(self._pixmap.width()), float(self._pixmap.height())
            w, h = float(self.width()), float(self.height())
            fit_scale = min(w / max(1.0, pw), h / max(1.0, ph))

            if self._fit_mode:
                self._fit_mode = False
                self._zoom_scale = fit_scale
                self._pan_offset = QPointF(0.0, 0.0)

            factor = 1.25 if delta > 0 else 0.8
            new_scale = max(fit_scale * 0.5, min(8.0, self._zoom_scale * factor))

            if new_scale <= fit_scale * 1.02:
                self._fit_mode = True
                self._zoom_scale = 1.0
                self._pan_offset = QPointF(0.0, 0.0)
                self.setCursor(Qt.ArrowCursor)
            else:
                cursor_pos = event.position()
                cx = (w / 2.0) + self._pan_offset.x()
                cy = (h / 2.0) + self._pan_offset.y()
                vx = cursor_pos.x() - cx
                vy = cursor_pos.y() - cy
                ratio = new_scale / max(1e-4, self._zoom_scale)
                self._pan_offset = QPointF(
                    self._pan_offset.x() - vx * (ratio - 1.0),
                    self._pan_offset.y() - vy * (ratio - 1.0)
                )
                self._zoom_scale = new_scale
                self.setCursor(Qt.OpenHandCursor)
            self.update()
        else:
            # Rueda directa -> Pasar fotos (rueda arriba: anterior, rueda abajo: siguiente)
            parent = self.parent()
            while parent is not None and not hasattr(parent, 'next_frame'):
                parent = parent.parent()
            if parent is not None:
                if delta > 0:
                    parent.prev_frame()
                else:
                    parent.next_frame()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if not self._pixmap or self._pixmap.isNull():
                return
            pw, ph = float(self._pixmap.width()), float(self._pixmap.height())
            w, h = float(self.width()), float(self.height())

            if self._fit_mode:
                # Pasar a 1:1 (100% tamaño real) centrado en la posición de clic
                self._fit_mode = False
                self._zoom_scale = 1.0
                fit_scale = min(w / max(1.0, pw), h / max(1.0, ph))
                click_pos = event.position()
                vx = click_pos.x() - (w / 2.0)
                vy = click_pos.y() - (h / 2.0)
                ratio = (1.0 / max(1e-4, fit_scale)) - 1.0
                self._pan_offset = QPointF(-vx * max(0.0, ratio), -vy * max(0.0, ratio))
                self.setCursor(Qt.OpenHandCursor)
            else:
                # Volver a Fit completo
                self._fit_mode = True
                self._zoom_scale = 1.0
                self._pan_offset = QPointF(0.0, 0.0)
                self.setCursor(Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if not self._fit_mode and event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_panning and not self._fit_mode:
            curr_pos = event.position()
            delta = curr_pos - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = curr_pos
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor if self._fit_mode else Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07090e"))
        if self._pixmap and not self._pixmap.isNull():
            w, h = float(self.width()), float(self.height())
            pw, ph = float(self._pixmap.width()), float(self._pixmap.height())

            if self._fit_mode:
                scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                painter.drawPixmap(x, y, scaled)
            else:
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                draw_w = pw * self._zoom_scale
                draw_h = ph * self._zoom_scale
                cx = (w / 2.0) + self._pan_offset.x()
                cy = (h / 2.0) + self._pan_offset.y()
                target_rect = QRectF(cx - draw_w / 2.0, cy - draw_h / 2.0, draw_w, draw_h)
                painter.drawPixmap(target_rect, self._pixmap, QRectF(0.0, 0.0, pw, ph))

                # Indicador discreto de zoom en la esquina superior derecha
                zoom_pct = int(round(self._zoom_scale * 100))
                tag_str = f"Zoom {zoom_pct}% (1:1)" if abs(self._zoom_scale - 1.0) < 0.05 else f"Zoom {zoom_pct}%"
                painter.setFont(QFont("Segoe UI", 9))
                badge_rect = QRectF(w - 120.0, 10.0, 110.0, 22.0)
                painter.fillRect(badge_rect, QColor(15, 23, 42, 200))
                painter.setPen(QPen(QColor(BORDER_BASE), 1))
                painter.drawRoundedRect(badge_rect, 4, 4)
                painter.setPen(QColor("#38bdf8"))
                painter.drawText(badge_rect, Qt.AlignCenter, tag_str)
        else:
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, "Cargando fotograma...")


def _format_schematic_phase(phase_label: Optional[str]) -> Tuple[str, QColor, bool]:
    """
    Retorna (texto_esquemático, color, es_bold) para la fase del eclipse:
    - Totalidad
    - Anillo C2 / Anillo C3 (o Anillo)
    - Baily C2 / Baily C3 (o Baily)
    - C1-C2
    - C3-C4
    """
    if not phase_label or phase_label in ("—", "Pendiente"):
        return ("—", QColor("#64748b"), False)
    pl_upper = phase_label.upper()
    if "TOTALIDAD" in pl_upper:
        return ("Totalidad", QColor("#c084fc"), True)
    elif "ANILLO" in pl_upper or "DIAMANTE" in pl_upper:
        if "C3" in pl_upper or "SALIDA" in pl_upper or "EGRESO" in pl_upper:
            txt = "Anillo C3"
        elif "C2" in pl_upper or "ENTRADA" in pl_upper or "INGRESO" in pl_upper:
            txt = "Anillo C2"
        else:
            txt = "Anillo"
        return (txt, QColor("#fbbf24"), True)
    elif "BAILY" in pl_upper or "PERLAS" in pl_upper:
        if "C3" in pl_upper or "SALIDA" in pl_upper or "EGRESO" in pl_upper:
            txt = "Baily C3"
        elif "C2" in pl_upper or "ENTRADA" in pl_upper or "INGRESO" in pl_upper:
            txt = "Baily C2"
        else:
            txt = "Baily"
        return (txt, QColor("#f59e0b"), True)
    elif "C1-C2" in pl_upper or "ENTRADA" in pl_upper:
        return ("C1-C2", QColor("#38bdf8"), False)
    elif "C3-C4" in pl_upper or "SALIDA" in pl_upper:
        return ("C3-C4", QColor("#fb923c"), False)
    clean = phase_label.split("(")[0].strip()
    return (clean[:12], QColor("#94a3b8"), False)


def _format_obscuration(obsc: Optional[float]) -> str:
    if obsc is None or obsc < 0:
        return "—"
    return f"{obsc:.1f}%"


def _format_sharpness(score: Optional[float], is_trepidated: bool = False) -> Tuple[str, QColor]:
    if score is None or score <= 0:
        return ("—", QColor("#64748b"))
    if is_trepidated:
        return (f"● {score:.0f} (Mov)", QColor("#ef4444"))
    elif score >= 80:
        return (f"● {score:.0f}", QColor("#38bdf8"))
    elif score >= 50:
        return (f"● {score:.0f}", QColor("#94a3b8"))
    else:
        return (f"● {score:.0f}", QColor("#cbd5e1"))


class BlinkTableDelegate(QStyledItemDelegate):
    """
    Delegado de renderizado de alto contraste y silencio visual para la tabla lateral Blink.
    - Filas seleccionadas (incluidas): tinte completo (#122336/#0F291E), barra de acento 3px en col 0,
      etiqueta destacada en col 1 (★ COL / ✓ INCL), nombre de archivo en blanco puro negrita (#F8FAFC).
    - Filas no seleccionadas (descartadas): fondo oscuro neutro (#14171D), silencio visual en col 1 (·),
      tipografía atenuada en gris (#7C8798).
    - Fila activa en el visor: fondo reforzado (#1E3A5F/#163E2E o #1C222C).
    """

    def __init__(self, dialog: "BlinkDialog"):
        super().__init__(dialog)
        self.dialog = dialog

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        row = index.row()
        col = index.column()
        if row < 0 or row >= len(self.dialog.file_list):
            super().paint(painter, option, index)
            return

        path = self.dialog.file_list[row]
        is_included = (path not in self.dialog.excluded_files)
        is_eclipse = (self.dialog.mode == "eclipse")
        is_current = bool(opt.state & QStyle.State_Selected) or (row == self.dialog.current_idx)

        # 1. Tinte de fondo completo
        if is_included:
            if is_current:
                bg_color = QColor("#1E3A5F") if is_eclipse else QColor("#163E2E")
            else:
                bg_color = QColor("#122336") if is_eclipse else QColor("#0F291E")
        else:
            if is_current:
                bg_color = QColor("#1C222C")
            else:
                bg_color = QColor("#14171D")

        painter.save()
        painter.fillRect(opt.rect, bg_color)

        # 2. Barra lateral de acento (3px) en la primera columna (#)
        if col == 0 and is_included:
            accent_color = QColor("#38BDF8") if is_eclipse else QColor("#10B981")
            accent_rect = QRect(opt.rect.left(), opt.rect.top(), 3, opt.rect.height())
            painter.fillRect(accent_rect, accent_color)

        # 3. Tipografía y color de texto según columna y estado
        opt.state &= ~QStyle.State_Selected
        opt.backgroundBrush = QBrush(Qt.NoBrush)

        if not is_included:
            # Fila no seleccionada: atenuada
            opt.palette.setColor(QPalette.Text, QColor("#7C8798"))
            opt.palette.setColor(QPalette.WindowText, QColor("#7C8798"))
            opt.palette.setColor(QPalette.HighlightedText, QColor("#7C8798"))
            if col == 1:
                # Silencio visual en columna de estado: punto central minimalista
                opt.text = "·"
                opt.displayAlignment = Qt.AlignCenter
                opt.palette.setColor(QPalette.Text, QColor("#334155"))
                opt.palette.setColor(QPalette.WindowText, QColor("#334155"))
                opt.palette.setColor(QPalette.HighlightedText, QColor("#334155"))
                f = opt.font
                f.setBold(True)
                opt.font = f
        else:
            # Fila seleccionada (incluida)
            if col == 1:
                # Etiqueta compacta en negrita
                col_text = "★ COL" if is_eclipse else "✓ INCL"
                color = QColor("#38BDF8") if is_eclipse else QColor("#10B981")
                opt.text = col_text
                opt.displayAlignment = Qt.AlignCenter
                opt.palette.setColor(QPalette.Text, color)
                opt.palette.setColor(QPalette.WindowText, color)
                opt.palette.setColor(QPalette.HighlightedText, color)
                f = opt.font
                f.setBold(True)
                opt.font = f
            elif col == 2:
                # Nombre de archivo en blanco puro y negrita
                opt.palette.setColor(QPalette.Text, QColor("#F8FAFC"))
                opt.palette.setColor(QPalette.WindowText, QColor("#F8FAFC"))
                opt.palette.setColor(QPalette.HighlightedText, QColor("#F8FAFC"))
                f = opt.font
                f.setBold(True)
                opt.font = f
            else:
                # Resto de columnas (#, HORA, EXP, ISO, Fase, Ocult, Nitidez)
                table_widget = opt.widget
                item = table_widget.item(row, col) if isinstance(table_widget, QTableWidget) else None
                if item and item.foreground().color().isValid() and col in (3, 5) and is_eclipse:
                    c = item.foreground().color()
                    opt.palette.setColor(QPalette.Text, c)
                    opt.palette.setColor(QPalette.WindowText, c)
                    opt.palette.setColor(QPalette.HighlightedText, c)
                else:
                    opt.palette.setColor(QPalette.Text, QColor("#E2E8F0"))
                    opt.palette.setColor(QPalette.WindowText, QColor("#E2E8F0"))
                    opt.palette.setColor(QPalette.HighlightedText, QColor("#E2E8F0"))

        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
        painter.restore()


class BlinkDialog(QDialog):
    """
    Diálogo flotante interactivo para inspección ultrarrápida (Blink) y cribado de tomas.
    """

    exclusion_changed = Signal(str, bool)     # (filepath, is_included)
    selection_applied = Signal(set)           # set de archivos excluidos
    order_changed = Signal(list)              # (reordered_file_list)
    _meta_scanned = Signal(int, str, dict)    # (row, filepath, metadata_dict)

    def __init__(
        self,
        file_list: List[str],
        excluded_files: Optional[Set[str]] = None,
        culling_results: Optional[Dict[str, Any]] = None,
        current_index: int = 0,
        metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None,
        mode: str = "general",
        contacts: Optional[Dict[str, float]] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent, Qt.Window)
        self.mode: str = (mode or "general").lower()
        if self.mode == "eclipse":
            self.setWindowTitle("AstroLoom - Visor Rápido (Blink & Selección Eclipse)")
            self.resize(1280, 780)
        else:
            self.setWindowTitle("AstroLoom - Visor Rápido de Subframes (Blink & Culling)")
            self.resize(1150, 750)
        apply_theme(self)

        self.file_list: List[str] = list(file_list)
        self.excluded_files: Set[str] = set(excluded_files) if excluded_files else set()
        self.culling_results: Dict[str, Any] = culling_results if culling_results else {}
        self.current_idx: int = max(0, min(current_index, len(self.file_list) - 1)) if self.file_list else 0
        self._sort_col: Optional[int] = None
        self._sort_asc: bool = True
        self._metadata_cache: Dict[str, Dict[str, Any]] = metadata_cache if metadata_cache is not None else {}
        self._eclipse_contacts: Dict[str, float] = dict(contacts) if contacts else {}
        self._is_shutting_down: bool = False

        self._meta_scanned.connect(self._on_meta_scanned)

        # Caché de memoria en búfer circular
        self._cache: Dict[str, np.ndarray] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Temporizador para reproducción continua
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_play_step)

        self._init_ui()
        self._update_current_frame()
        self._preload_neighbors()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # =====================================================================
        # BARRA SUPERIOR: Información de Toma, Estado y Métricas Ópticas
        # =====================================================================
        header = QFrame()
        header.setStyleSheet(f"background-color: {BG_CARD}; border: 1px solid {BORDER_BASE}; border-radius: 6px;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 6, 10, 6)
        h_layout.setSpacing(10)

        self.lbl_title = QLabel("Fotograma 0 / 0")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #f1f5f9;")
        h_layout.addWidget(self.lbl_title)

        # Botón píldora interactivo para alternar estado de inclusión
        self.btn_status_badge = QPushButton("INCLUIDA")
        self.btn_status_badge.setCursor(Qt.PointingHandCursor)
        self.btn_status_badge.setToolTip("Alternar inclusión de la toma (Atajo: P o X)")
        self.btn_status_badge.clicked.connect(self._toggle_current_frame_status)
        self.lbl_status_badge = self.btn_status_badge  # Alias para retrocompatibilidad
        h_layout.addWidget(self.btn_status_badge)

        # Parámetros clave de disparo (Velocidad · ISO)
        self.lbl_exif_badge = QLabel("")
        self.lbl_exif_badge.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px; background-color: #0c1929; border: 1px solid #1e3a5f; border-radius: 4px; padding: 2px 8px;")
        self.lbl_exif_badge.setVisible(False)
        h_layout.addWidget(self.lbl_exif_badge)

        # Badge de calidad óptica si ya fue analizada
        self.lbl_quality_badge = QLabel("")
        self.lbl_quality_badge.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 6px;")
        h_layout.addWidget(self.lbl_quality_badge)

        h_layout.addStretch()

        self.btn_toggle_side = QPushButton("Ocultar Lista")
        self.btn_toggle_side.setIcon(get_icon("sidebar", size=13))
        self.btn_toggle_side.setToolTip("Muestra u oculta la lista lateral de tomas de la serie (Atajo: L)")
        self.btn_toggle_side.clicked.connect(self._toggle_side_panel)
        h_layout.addWidget(self.btn_toggle_side)

        self.btn_toggle_timeline = QPushButton("Ocultar Timeline")
        self.btn_toggle_timeline.setIcon(get_icon("activity", size=13))
        self.btn_toggle_timeline.setToolTip("Muestra u oculta la barra gráfica de línea de tiempo interactiva (Atajo: T)")
        self.btn_toggle_timeline.clicked.connect(self._toggle_timeline)
        h_layout.addWidget(self.btn_toggle_timeline)

        self.chk_auto_stretch = QCheckBox("Auto-Estirado (LUT)")
        self.chk_auto_stretch.setChecked(True)
        self.chk_auto_stretch.setToolTip("Aplica un estirado temporal para ver tomas lineales oscuras con brillo óptimo")
        self.chk_auto_stretch.toggled.connect(self._on_stretch_toggled)
        h_layout.addWidget(self.chk_auto_stretch)

        root_layout.addWidget(header)

        # =====================================================================
        # ZONA CENTRAL: SPLITTER CON TABLA DE TOMAS A LA IZQUIERDA Y CANVAS A LA DERECHA
        # =====================================================================
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # Panel lateral con tabla de tomas
        self.side_panel = QFrame()
        is_eclipse = (self.mode == "eclipse")
        if is_eclipse:
            self.side_panel.setMinimumWidth(380)
            self.side_panel.setMaximumWidth(680)
        else:
            self.side_panel.setMinimumWidth(280)
            self.side_panel.setMaximumWidth(480)
        self.side_panel.setStyleSheet(f"background-color: {BG_CARD}; border: 1px solid {BORDER_BASE}; border-radius: 6px;")
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(4)

        side_hdr = QHBoxLayout()
        lbl_list_title = QLabel("Tomas del Eclipse" if is_eclipse else "Tomas de la Serie")
        lbl_list_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #cbd5e1;")
        side_hdr.addWidget(lbl_list_title)
        side_hdr.addStretch()

        total = len(self.file_list)
        inc = total - len(self.excluded_files)
        term = "en collage" if is_eclipse else "activas"
        self.lbl_blink_summary = QLabel(f"{inc}/{total} {term}")
        self.lbl_blink_summary.setStyleSheet("color: #94a3b8; font-size: 10px;")
        side_hdr.addWidget(self.lbl_blink_summary)
        side_layout.addLayout(side_hdr)

        col1_name = "Collage" if is_eclipse else "Uso"
        if is_eclipse:
            self.table_blink = QTableWidget(len(self.file_list), 8)
            self.table_blink.setHorizontalHeaderLabels([
                "#", col1_name, "Archivo", "Fase", "% Ocult.", "Nitidez", "Exp.", "ISO"
            ])
            hdr_blink = self.table_blink.horizontalHeader()
            hdr_blink.setMinimumSectionSize(28)
            hdr_blink.setSectionResizeMode(0, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(1, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(2, QHeaderView.Stretch)
            hdr_blink.setSectionResizeMode(3, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(4, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(5, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(6, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(7, QHeaderView.Interactive)
            self.table_blink.setColumnWidth(0, 36)
            self.table_blink.setColumnWidth(1, 64)
            self.table_blink.setColumnWidth(3, 76)
            self.table_blink.setColumnWidth(4, 58)
            self.table_blink.setColumnWidth(5, 74)
            self.table_blink.setColumnWidth(6, 54)
            self.table_blink.setColumnWidth(7, 48)
            tooltips = {
                0: "Ordenar por orden de lista",
                1: "Ordenar por inclusión en collage",
                2: "Ordenar alfabéticamente por nombre de archivo",
                3: "Ordenar por fase del eclipse",
                4: "Ordenar por % de ocultación solar/lunar",
                5: "Ordenar por nitidez óptica",
                6: "Ordenar por tiempo de exposición",
                7: "Ordenar por sensibilidad ISO",
            }
        else:
            self.table_blink = QTableWidget(len(self.file_list), 5)
            self.table_blink.setHorizontalHeaderLabels(["#", col1_name, "Archivo", "Exp.", "ISO"])
            hdr_blink = self.table_blink.horizontalHeader()
            hdr_blink.setMinimumSectionSize(28)
            hdr_blink.setSectionResizeMode(0, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(1, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(2, QHeaderView.Stretch)
            hdr_blink.setSectionResizeMode(3, QHeaderView.Interactive)
            hdr_blink.setSectionResizeMode(4, QHeaderView.Interactive)
            self.table_blink.setColumnWidth(0, 38)
            self.table_blink.setColumnWidth(1, 68)
            self.table_blink.setColumnWidth(3, 58)
            self.table_blink.setColumnWidth(4, 54)
            tooltips = {
                0: "Ordenar por orden de lista",
                1: "Ordenar por estado de uso",
                2: "Ordenar alfabéticamente por nombre de archivo",
                3: "Ordenar por tiempo de exposición",
                4: "Ordenar por sensibilidad ISO",
            }
        hdr_blink.setSectionsClickable(True)
        hdr_blink.setSortIndicatorShown(True)
        hdr_blink.sectionClicked.connect(self._on_header_section_clicked)

        for c, tip in tooltips.items():
            h_item = self.table_blink.horizontalHeaderItem(c)
            if h_item:
                h_item.setToolTip(tip)
        self.table_blink.verticalHeader().setVisible(False)
        self.table_blink.verticalHeader().setDefaultSectionSize(22)
        self.table_blink.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_blink.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_blink.setStyleSheet(f"""
            QTableWidget {{
                background-color: {BG_CANVAS};
                border: 1px solid {BORDER_BASE};
                border-radius: 4px;
                gridline-color: #1a1a1e;
                font-size: 11px;
            }}
            QTableWidget::item, QTableWidget::item:selected, QTableWidget::item:hover {{
                background-color: transparent;
                border: none;
            }}
            QHeaderView::section {{
                background-color: {BG_PANEL};
                color: #94a3b8;
                padding: 2px 4px;
                font-size: 10px;
                font-weight: 600;
                border: 1px solid {BORDER_BASE};
            }}
        """)
        self.table_delegate = BlinkTableDelegate(self)
        self.table_blink.setItemDelegate(self.table_delegate)
        self.table_blink.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_blink.cellClicked.connect(self._on_table_cell_clicked)
        self.table_blink.cellDoubleClicked.connect(self._on_table_cell_double_clicked)
        side_layout.addWidget(self.table_blink)

        self.canvas = BlinkCanvas(self)
        self.splitter.addWidget(self.side_panel)
        self.splitter.addWidget(self.canvas)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        if is_eclipse:
            self.splitter.setSizes([480, 770])
        else:
            self.splitter.setSizes([340, 810])

        root_layout.addWidget(self.splitter, stretch=1)
        self._populate_table()

        # =====================================================================
        # BARRA INFERIOR: Controles de Reproducción y Cribado
        # =====================================================================
        footer = QFrame()
        footer.setStyleSheet(f"background-color: {BG_CARD}; border: 1px solid {BORDER_BASE}; border-radius: 6px;")
        f_layout = QVBoxLayout(footer)
        f_layout.setContentsMargins(8, 6, 8, 6)
        f_layout.setSpacing(6)

        # Contenedor de la línea de tiempo interactiva y leyenda de fases (ocultable con botón/atajo T)
        self.timeline_container = QWidget()
        tl_layout = QVBoxLayout(self.timeline_container)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        tl_layout.setSpacing(4)

        # Fila 1: Línea de tiempo astronómica interactiva (Timeline con hitos y gaps)
        from gui.widgets.session_timeline_widget import SessionTimelineWidget
        self.timeline = SessionTimelineWidget(mode=self.mode, parent=self)
        if self._eclipse_contacts:
            self.timeline.set_eclipse_contacts(self._eclipse_contacts)
        self.timeline.set_data(
            files=self.file_list,
            metadata_cache=self._metadata_cache,
            excluded_files=self.excluded_files,
            current_idx=self.current_idx
        )
        self.timeline.frame_selected.connect(self.set_current_index)
        self.timeline.gap_selected.connect(self._on_timeline_gap_selected)
        if self.mode == "eclipse":
            self.timeline.auto_select_requested.connect(self._on_timeline_auto_select)
        tl_layout.addWidget(self.timeline)

        # Barra de Leyenda de Fases (Modo Eclipse)
        if self.mode == "eclipse":
            legend_frame = QFrame()
            legend_frame.setStyleSheet("background: transparent; border: none;")
            legend_layout = QHBoxLayout(legend_frame)
            legend_layout.setContentsMargins(4, 0, 4, 0)
            legend_layout.setSpacing(10)

            lbl_leg_title = QLabel("Fases:")
            lbl_leg_title.setStyleSheet("font-size: 10px; font-weight: bold; color: #94a3b8;")
            legend_layout.addWidget(lbl_leg_title)

            legend_items = [
                ("C1-C2 Entrada", "#0284c7"),
                ("C2/C3 Contacto", "#fbbf24"),
                ("Totalidad", "#9333ea"),
                ("C3-C4 Salida", "#ea580c"),
                ("En Collage", "#38bdf8"),
                ("Disponible", "#64748b"),
            ]
            for label, col in legend_items:
                dot = QLabel(f"<span style='color: {col}; font-size: 11px;'>■</span> <span style='color: #cbd5e1; font-size: 10px;'>{label}</span>")
                legend_layout.addWidget(dot)

            legend_layout.addStretch()
            tl_layout.addWidget(legend_frame)

        f_layout.addWidget(self.timeline_container)

        # Fila 2: Deslizador de posición compacto
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, len(self.file_list) - 1))
        self.slider.setValue(self.current_idx)
        self.slider.valueChanged.connect(self._on_slider_changed)
        f_layout.addWidget(self.slider)

        # Fila 3: Botones de transporte, hitos de gaps y cribado
        row_ctrl = QHBoxLayout()
        row_ctrl.setSpacing(8)

        # Transporte
        self.btn_prev = QPushButton("◀ Anterior (A)")
        self.btn_prev.setIcon(get_icon("chevron-left", size=13))
        self.btn_prev.clicked.connect(self.prev_frame)
        row_ctrl.addWidget(self.btn_prev)

        self.btn_play = QPushButton("▶ Reproducir (Espacio)")
        self.btn_play.setIcon(get_icon("play", size=13))
        self.btn_play.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.btn_play.clicked.connect(self.toggle_play)
        row_ctrl.addWidget(self.btn_play)

        self.btn_next = QPushButton("Siguiente (D) ▶")
        self.btn_next.setIcon(get_icon("chevron-right", size=13))
        self.btn_next.clicked.connect(self.next_frame)
        row_ctrl.addWidget(self.btn_next)

        # Velocidad FPS: 1 a 5 fps, 1 por defecto
        row_ctrl.addWidget(QLabel("Velocidad:"))
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["1 fps", "2 fps", "3 fps", "4 fps", "5 fps"])
        self.combo_fps.setCurrentIndex(0)  # 1 fps default
        self.combo_fps.currentIndexChanged.connect(self._on_fps_changed)
        row_ctrl.addWidget(self.combo_fps)

        # Filtro de reproducción y navegación por tomas incluidas
        filter_label = "Blink solo collage" if self.mode == "eclipse" else "Blink solo activas"
        self.chk_play_filtered = QCheckBox(filter_label)
        self.chk_play_filtered.setStyleSheet("color: #94a3b8; font-size: 11px; margin-left: 6px;")
        self.chk_play_filtered.setToolTip("Al reproducir o pasar fotos, salta únicamente entre tomas seleccionadas / en collage")
        row_ctrl.addWidget(self.chk_play_filtered)

        # Navegación rápida de Meteoros (Modo meteors)
        if self.mode == "meteors":
            self.btn_prev_meteor = QPushButton("◄ Meteoro")
            self.btn_prev_meteor.setIcon(get_icon("chevron-left", size=11))
            self.btn_prev_meteor.setToolTip("Saltar al fotograma anterior con meteoro conservado")
            self.btn_prev_meteor.setStyleSheet("font-weight: bold; color: #10b981; border: 1px solid #059669; border-radius: 4px; padding: 4px 8px;")
            self.btn_prev_meteor.clicked.connect(self._on_jump_prev_meteor)
            row_ctrl.addWidget(self.btn_prev_meteor)

            self.btn_next_meteor = QPushButton("Meteoro ▶")
            self.btn_next_meteor.setIcon(get_icon("chevron-right", size=11))
            self.btn_next_meteor.setToolTip("Saltar al siguiente fotograma con meteoro conservado")
            self.btn_next_meteor.setStyleSheet("font-weight: bold; color: #10b981; border: 1px solid #059669; border-radius: 4px; padding: 4px 8px;")
            self.btn_next_meteor.clicked.connect(self._on_jump_next_meteor)
            row_ctrl.addWidget(self.btn_next_meteor)

        # Navegación entre discontinuidades (Gaps)
        self.btn_prev_gap = QPushButton("◀ Gap")
        self.btn_prev_gap.setIcon(get_icon("chevron-left", size=11))
        self.btn_prev_gap.setToolTip("Saltar a la discontinuidad temporal anterior")
        self.btn_prev_gap.clicked.connect(self._on_jump_prev_gap)
        row_ctrl.addWidget(self.btn_prev_gap)

        self.btn_next_gap = QPushButton("Gap ▶")
        self.btn_next_gap.setIcon(get_icon("chevron-right", size=11))
        self.btn_next_gap.setToolTip("Saltar a la siguiente discontinuidad temporal (falta de fotos)")
        self.btn_next_gap.clicked.connect(self._on_jump_next_gap)
        row_ctrl.addWidget(self.btn_next_gap)

        gaps = self.timeline.get_gaps()
        self.lbl_gaps_summary = QLabel(f"⚠️ {len(gaps)} gaps" if gaps else "")
        self.lbl_gaps_summary.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
        self.lbl_gaps_summary.setVisible(len(gaps) > 0)
        row_ctrl.addWidget(self.lbl_gaps_summary)

        # Indicador de continuidad para Startrails
        if self.mode == "startrails":
            self.lbl_startrails_continuity = QLabel()
            self.lbl_startrails_continuity.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold; margin-left: 4px;")
            row_ctrl.addWidget(self.lbl_startrails_continuity)
            self._update_startrails_continuity()

        # Atajo Zoom Totalidad (Modo eclipse)
        if self.mode == "eclipse":
            self.btn_zoom_totality = QPushButton("🔍 Zoom Totalidad")
            self.btn_zoom_totality.setToolTip("Encuadra la ventana temporal en la fase de totalidad (C2 a C3)")
            self.btn_zoom_totality.setStyleSheet("font-weight: bold; color: #c084fc; border: 1px solid #9333ea; border-radius: 4px; padding: 4px 8px;")
            self.btn_zoom_totality.clicked.connect(self.timeline.zoom_to_totality)
            row_ctrl.addWidget(self.btn_zoom_totality)

        # Botón Vista Completa (se muestra cuando hay zoom activo en el timeline)
        self.btn_full_view = QPushButton("Vista Completa")
        self.btn_full_view.setToolTip("Restaura la escala temporal completa del timeline (o clic derecho en la barra)")
        self.btn_full_view.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.btn_full_view.clicked.connect(self.timeline.reset_zoom)
        self.btn_full_view.setVisible(False)
        self.timeline.zoom_changed.connect(self.btn_full_view.setVisible)
        row_ctrl.addWidget(self.btn_full_view)

        row_ctrl.addStretch()

        # Acciones de Cribado
        if self.mode == "eclipse":
            self.btn_keep = QPushButton("✓ Al Collage (P)")
            self.btn_keep.setStyleSheet("""
                QPushButton {
                    background-color: #082f49;
                    color: #38bdf8;
                    font-weight: bold;
                    border: 1px solid #0284c7;
                    border-radius: 4px;
                    padding: 5px 12px;
                }
                QPushButton:hover { background-color: #0284c7; color: #ffffff; }
            """)
            self.btn_keep.setToolTip("Incluye la toma actual en el collage (Atajo: P)")
            self.btn_keep.clicked.connect(self.keep_current_frame)
            row_ctrl.addWidget(self.btn_keep)

            self.btn_discard = QPushButton("✕ Quitar (X)")
            self.btn_discard.setStyleSheet("""
                QPushButton {
                    background-color: #1e293b;
                    color: #94a3b8;
                    font-weight: bold;
                    border: 1px solid #475569;
                    border-radius: 4px;
                    padding: 5px 12px;
                }
                QPushButton:hover { background-color: #334155; color: #f1f5f9; }
            """)
            self.btn_discard.setToolTip("Quita la toma actual del collage manteniéndola en la sesión (Atajo: X)")
            self.btn_discard.clicked.connect(self.discard_current_frame)
            row_ctrl.addWidget(self.btn_discard)
        else:
            self.btn_keep = QPushButton("✓ Conservar (P)")
            self.btn_keep.setStyleSheet("""
                QPushButton {
                    background-color: #132b1e;
                    color: #4ade80;
                    font-weight: bold;
                    border: 1px solid #22c55e;
                    border-radius: 4px;
                    padding: 5px 12px;
                }
                QPushButton:hover { background-color: #22c55e; color: #ffffff; }
            """)
            self.btn_keep.setToolTip("Incluye la toma actual en el apilado y avanza al siguiente fotograma (Atajo: P)")
            self.btn_keep.clicked.connect(self.keep_current_frame)
            row_ctrl.addWidget(self.btn_keep)

            self.btn_discard = QPushButton("✗ Descartar (X)")
            self.btn_discard.setStyleSheet("""
                QPushButton {
                    background-color: #2a1520;
                    color: #f87171;
                    font-weight: bold;
                    border: 1px solid #ef4444;
                    border-radius: 4px;
                    padding: 5px 12px;
                }
                QPushButton:hover { background-color: #ef4444; color: #ffffff; }
            """)
            self.btn_discard.setToolTip("Excluye la toma actual del apilado y avanza al siguiente fotograma (Atajo: X)")
            self.btn_discard.clicked.connect(self.discard_current_frame)
            row_ctrl.addWidget(self.btn_discard)

        self.btn_close = QPushButton("Aplicar y Cerrar")
        self.btn_close.clicked.connect(self.accept)
        row_ctrl.addWidget(self.btn_close)

        f_layout.addLayout(row_ctrl)
        root_layout.addWidget(footer)

    def _on_fps_changed(self):
        fps_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        fps = fps_map.get(self.combo_fps.currentIndex(), 1)
        interval_ms = int(1000.0 / fps)
        if self._timer.isActive():
            self._timer.setInterval(interval_ms)

    def toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self.btn_play.setText("▶ Reproducir (Espacio)")
            self.btn_play.setIcon(get_icon("play", size=13))
            self.btn_play.setStyleSheet("font-weight: bold; color: #38bdf8;")
        else:
            fps_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
            fps = fps_map.get(self.combo_fps.currentIndex(), 1)
            self._timer.start(int(1000.0 / fps))
            self.btn_play.setText("⏸ Pausar (Espacio)")
            self.btn_play.setIcon(get_icon("pause", size=13))
            self.btn_play.setStyleSheet("font-weight: bold; color: #fbbf24;")

    def _on_jump_next_gap(self):
        if hasattr(self, 'timeline'):
            self.timeline.jump_to_next_gap()

    def _on_jump_prev_gap(self):
        if hasattr(self, 'timeline'):
            self.timeline.jump_to_prev_gap()

    def _on_jump_next_meteor(self):
        if hasattr(self, 'timeline'):
            self.timeline.jump_to_next_kept()

    def _on_jump_prev_meteor(self):
        if hasattr(self, 'timeline'):
            self.timeline.jump_to_prev_kept()

    def _update_startrails_continuity(self):
        if not hasattr(self, 'lbl_startrails_continuity') or not hasattr(self, 'timeline'):
            return
        stats = self.timeline.get_continuity_stats()
        dt_lost = stats["lost_seconds"]
        lost_str = f"{dt_lost / 60.0:.1f}m" if dt_lost >= 60.0 else f"{dt_lost:.0f}s"
        txt = f"Continuidad: {stats['continuity_pct']:.1f}% · {stats['capture_gaps_count']} huecos ({lost_str})"
        self.lbl_startrails_continuity.setText(txt)

    def _on_timeline_auto_select(self, interval_seconds: float):
        """Auto-selecciona tomas cada N minutos en tramos libres de parcialidad (modo Eclipse)."""
        if self.mode != "eclipse" or not self.file_list:
            return
        items = getattr(self.timeline, "_items", [])
        if not items:
            return

        contacts = getattr(self.timeline, "_eclipse_contacts", {}) or {}
        t_c2 = contacts.get("C2")
        t_c3 = contacts.get("C3")

        new_excluded = set()
        last_t_selected = -float("inf")

        for it in sorted(items, key=lambda x: x["t"]):
            p = it["path"]
            t = it["t"]
            p_label = it.get("phase_label", "").upper()

            # Preservar siempre las tomas de totalidad o contactos singulares (Baily / Diamante)
            is_totality = False
            if t_c2 and t_c3 and t_c2 <= t <= t_c3:
                is_totality = True
            elif "TOTALIDAD" in p_label or "DIAMANTE" in p_label or "BAILY" in p_label:
                is_totality = True

            if is_totality:
                continue

            # Parcialidad: espaciar cada interval_seconds
            if (t - last_t_selected) >= interval_seconds:
                last_t_selected = t
            else:
                new_excluded.add(p)

        self.excluded_files = new_excluded
        self.timeline.set_excluded_files(self.excluded_files)
        self.selection_applied.emit(self.excluded_files)
        self._update_badges()
        self._populate_table()

    def _on_timeline_gap_selected(self, from_idx: int, to_idx: int):
        self.set_current_index(from_idx)

    def _on_header_section_clicked(self, col: int) -> None:
        if len(self.file_list) < 2:
            return

        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        selected_path = self.file_list[self.current_idx] if 0 <= self.current_idx < len(self.file_list) else None

        if self.mode == "eclipse":
            if col == 0:  # "#"
                def _key(p):
                    return os.path.basename(p).lower()
            elif col == 1:  # "Collage"
                def _key(p):
                    return (0 if p not in self.excluded_files else 1, os.path.basename(p).lower())
            elif col == 2:  # "Archivo"
                def _key(p):
                    return os.path.basename(p).lower()
            elif col == 3:  # "Fase"
                def _key(p):
                    meta = self._metadata_cache.get(p, {})
                    ts = meta.get("timestamp")
                    if ts is not None:
                        try:
                            return (0, float(ts))
                        except (ValueError, TypeError):
                            pass
                    return (1, meta.get("phase_label") or "")
            elif col == 4:  # "% Ocult."
                def _key(p):
                    meta = self._metadata_cache.get(p, {})
                    obsc = meta.get("obscuration_pct")
                    if obsc is not None:
                        try:
                            return (0, float(obsc))
                        except (ValueError, TypeError):
                            pass
                    return (1, -1.0)
            elif col == 5:  # "Nitidez"
                def _key(p):
                    meta = self._metadata_cache.get(p, {})
                    score = meta.get("sharpness_score")
                    trep = meta.get("is_trepidated", False)
                    if score is not None:
                        try:
                            val = float(score) - (100.0 if trep else 0.0)
                            return (0, val)
                        except (ValueError, TypeError):
                            pass
                    return (1, -1000.0)
            elif col == 6:  # "Exp."
                def _key(p):
                    meta = self._metadata_cache.get(p, {})
                    exp = meta.get("exp")
                    if exp is not None:
                        try:
                            return (0, float(exp))
                        except (ValueError, TypeError):
                            pass
                    return (1, 0.0)
            elif col == 7:  # "ISO"
                def _key(p):
                    meta = self._metadata_cache.get(p, {})
                    iso = meta.get("iso")
                    if iso is not None:
                        try:
                            return (0, int(iso))
                        except (ValueError, TypeError):
                            pass
                    return (1, 0)
            else:
                return
        else:
            if col == 0:  # "#"
                def _key(p):
                    return os.path.basename(p).lower()
            elif col == 1:  # "Uso"
                def _key(p):
                    return (0 if p not in self.excluded_files else 1, os.path.basename(p).lower())
            elif col == 2:  # "Archivo"
                def _key(p):
                    return os.path.basename(p).lower()
            elif col == 3:  # "Exp."
                def _key(p):
                    meta = self._metadata_cache.get(p, {})
                    exp = meta.get("exp")
                    if exp is not None:
                        try:
                            return (0, float(exp))
                        except (ValueError, TypeError):
                            pass
                    return (1, 0.0)
            elif col == 4:  # "ISO"
                def _key(p):
                    meta = self._metadata_cache.get(p, {})
                    iso = meta.get("iso")
                    if iso is not None:
                        try:
                            return (0, int(iso))
                        except (ValueError, TypeError):
                            pass
                    return (1, 0)
            else:
                return

        self.file_list.sort(key=_key, reverse=not self._sort_asc)

        if selected_path and selected_path in self.file_list:
            self.current_idx = self.file_list.index(selected_path)
        else:
            self.current_idx = 0

        self.table_blink.horizontalHeader().setSortIndicator(
            col, Qt.AscendingOrder if self._sort_asc else Qt.DescendingOrder
        )

        self._populate_table()
        if hasattr(self, 'timeline'):
            self.timeline.set_data(
                files=self.file_list,
                metadata_cache=self._metadata_cache,
                excluded_files=self.excluded_files,
                current_idx=self.current_idx
            )
        self.order_changed.emit(self.file_list)

    def _on_play_step(self):
        if not self.file_list:
            return
        if getattr(self, 'chk_play_filtered', None) and self.chk_play_filtered.isChecked():
            included = [i for i, p in enumerate(self.file_list) if p not in self.excluded_files]
            if not included:
                return
            next_included = [i for i in included if i > self.current_idx]
            next_idx = next_included[0] if next_included else included[0]
            self.set_current_index(next_idx)
        else:
            next_idx = (self.current_idx + 1) % len(self.file_list)
            self.set_current_index(next_idx)

    def prev_frame(self):
        if not self.file_list:
            return
        if getattr(self, 'chk_play_filtered', None) and self.chk_play_filtered.isChecked():
            included = [i for i, p in enumerate(self.file_list) if p not in self.excluded_files]
            if not included:
                return
            prev_included = [i for i in included if i < self.current_idx]
            new_idx = prev_included[-1] if prev_included else included[-1]
            self.set_current_index(new_idx)
        else:
            new_idx = max(0, self.current_idx - 1)
            self.set_current_index(new_idx)

    def next_frame(self):
        if not self.file_list:
            return
        if getattr(self, 'chk_play_filtered', None) and self.chk_play_filtered.isChecked():
            included = [i for i, p in enumerate(self.file_list) if p not in self.excluded_files]
            if not included:
                return
            next_included = [i for i in included if i > self.current_idx]
            new_idx = next_included[0] if next_included else included[0]
            self.set_current_index(new_idx)
        else:
            new_idx = min(len(self.file_list) - 1, self.current_idx + 1)
            self.set_current_index(new_idx)

    def set_current_index(self, idx: int):
        if idx == self.current_idx or idx < 0 or idx >= len(self.file_list):
            return
        self.current_idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        if hasattr(self, 'timeline'):
            self.timeline.set_current_index(idx)
        self._update_current_frame()
        self._preload_neighbors()

    def _on_slider_changed(self, val: int):
        self.set_current_index(val)

    def _on_stretch_toggled(self, checked: bool):
        self._cache.clear()
        self._update_current_frame()
        self._preload_neighbors()

    def _toggle_frame_status_at(self, row: int):
        """Alterna el estado de inclusión/exclusión del fotograma en la fila especificada."""
        if row < 0 or row >= len(self.file_list):
            return
        path = self.file_list[row]
        is_now_included = False
        if path in self.excluded_files:
            self.excluded_files.remove(path)
            is_now_included = True
            self.exclusion_changed.emit(path, True)
        else:
            self.excluded_files.add(path)
            is_now_included = False
            self.exclusion_changed.emit(path, False)

        if hasattr(self, 'timeline'):
            self.timeline.set_excluded_files(self.excluded_files)

        is_eclipse = (self.mode == "eclipse")
        status_item = self.table_blink.item(row, 1)
        if status_item:
            if is_now_included:
                status_item.setText("★ COL" if is_eclipse else "✓ INCL")
                status_item.setForeground(QColor("#38bdf8") if is_eclipse else QColor("#10b981"))
                status_item.setToolTip("En Collage (Clic para alternar)" if is_eclipse else "Incluida (Clic para alternar)")
            else:
                status_item.setText("·")
                status_item.setForeground(QColor("#334155"))
                status_item.setToolTip("Disponible (Clic para alternar)" if is_eclipse else "Descartada (Clic para alternar)")

        if hasattr(self, 'lbl_blink_summary'):
            total = len(self.file_list)
            inc = total - len(self.excluded_files)
            term = "en collage" if is_eclipse else "activas"
            self.lbl_blink_summary.setText(f"{inc}/{total} {term}")

        if row == self.current_idx:
            self._update_badges()

        self.table_blink.viewport().update()

    def _toggle_current_frame_status(self):
        """Alterna el estado de inclusión/exclusión de la toma activa sin avanzar de fotograma."""
        self._toggle_frame_status_at(self.current_idx)

    def keep_current_frame(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        if path in self.excluded_files:
            self.excluded_files.remove(path)
            self.exclusion_changed.emit(path, True)
            if hasattr(self, 'timeline'):
                self.timeline.set_excluded_files(self.excluded_files)
            status_item = self.table_blink.item(self.current_idx, 1)
            if status_item:
                is_eclipse = (self.mode == "eclipse")
                status_item.setText("★ COL" if is_eclipse else "✓ INCL")
                status_item.setForeground(QColor("#38bdf8") if is_eclipse else QColor("#10b981"))
                status_item.setToolTip("En Collage (Clic para alternar)" if is_eclipse else "Incluida (Clic para alternar)")
            self.table_blink.viewport().update()
        self._update_badges()
        self.next_frame()

    def discard_current_frame(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        if path not in self.excluded_files:
            self.excluded_files.add(path)
            self.exclusion_changed.emit(path, False)
            if hasattr(self, 'timeline'):
                self.timeline.set_excluded_files(self.excluded_files)
            status_item = self.table_blink.item(self.current_idx, 1)
            if status_item:
                status_item.setText("·")
                status_item.setForeground(QColor("#334155"))
                is_eclipse = (self.mode == "eclipse")
                status_item.setToolTip("Disponible (Clic para alternar)" if is_eclipse else "Descartada (Clic para alternar)")
            self.table_blink.viewport().update()
        self._update_badges()
        self.next_frame()

    def _on_table_cell_clicked(self, row: int, col: int):
        """Un clic directo en la columna de estado (col 1) alterna inmediatamente el estado de esa toma."""
        if col == 1:
            self._toggle_frame_status_at(row)

    def _on_table_cell_double_clicked(self, row: int, col: int):
        """Doble clic sobre cualquier celda de la fila alterna el estado de esa toma."""
        if col != 1:  # En col 1 el primer clic ya realizó el toggle
            self._toggle_frame_status_at(row)

    def _update_badges(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        is_included = path not in self.excluded_files
        is_eclipse = (self.mode == "eclipse")

        if is_eclipse:
            if is_included:
                self.btn_status_badge.setText("● EN COLLAGE [P]")
                self.btn_status_badge.setStyleSheet("color: #38bdf8; background-color: #082f49; border: 1px solid #0284c7; border-radius: 11px; padding: 2px 10px; font-weight: bold; font-size: 11px;")
            else:
                self.btn_status_badge.setText("○ DISPONIBLE [P]")
                self.btn_status_badge.setStyleSheet("color: #94a3b8; background-color: #1e293b; border: 1px solid #475569; border-radius: 11px; padding: 2px 10px; font-weight: bold; font-size: 11px;")
        else:
            if is_included:
                self.btn_status_badge.setText("✓ INCLUIDA [P]")
                self.btn_status_badge.setStyleSheet("color: #4ade80; background-color: #132b1e; border: 1px solid #22c55e; border-radius: 11px; padding: 2px 10px; font-weight: bold; font-size: 11px;")
            else:
                self.btn_status_badge.setText("✗ DESCARTADA [X]")
                self.btn_status_badge.setStyleSheet("color: #f87171; background-color: #2a1520; border: 1px solid #ef4444; border-radius: 11px; padding: 2px 10px; font-weight: bold; font-size: 11px;")

        # Actualizar indicador en la tabla lateral
        if hasattr(self, 'table_blink') and self.table_blink.rowCount() > self.current_idx:
            status_item = self.table_blink.item(self.current_idx, 1)
            if status_item:
                if is_eclipse:
                    status_item.setText("★ COL" if is_included else "·")
                    status_item.setForeground(QColor("#38bdf8") if is_included else QColor("#334155"))
                    status_item.setToolTip("En Collage (Clic para alternar)" if is_included else "Disponible (Clic para alternar)")
                else:
                    status_item.setText("✓ INCL" if is_included else "·")
                    status_item.setForeground(QColor("#10b981") if is_included else QColor("#334155"))
                    status_item.setToolTip("Incluida (Clic para alternar)" if is_included else "Descartada (Clic para alternar)")
        if hasattr(self, 'lbl_blink_summary'):
            total = len(self.file_list)
            inc = total - len(self.excluded_files)
            term = "en collage" if is_eclipse else "activas"
            self.lbl_blink_summary.setText(f"{inc}/{total} {term}")

        if self.mode == "startrails":
            self._update_startrails_continuity()

        # Métricas ópticas si están disponibles
        if self.mode == "eclipse":
            meta = self._metadata_cache.get(path, {})
            score = meta.get("sharpness_score")
            if score is not None and score > 0:
                is_trep = meta.get("is_trepidated", False)
                trep_str = " · ⚠️ Trepidada" if is_trep else " · Estable"
                obsc = meta.get("obscuration_pct")
                obsc_str = f" · Ocultación: {obsc:.1f}%" if (obsc is not None and obsc >= 0) else ""
                self.lbl_quality_badge.setText(f"Nitidez: {score:.0f}/100{trep_str}{obsc_str}")
            else:
                self.lbl_quality_badge.setText("")
        elif path in self.culling_results:
            r = self.culling_results[path]
            fwhm_str = f"{r.fwhm:.1f} px" if (r.fwhm is not None and not np.isnan(r.fwhm) and r.fwhm > 0) else "N/D"
            ecc_str = f"{r.eccentricity:.2f}" if (r.eccentricity is not None and not np.isnan(r.eccentricity)) else "N/D"
            self.lbl_quality_badge.setText(f"★ {r.score:.0f}% | FWHM: {fwhm_str} | Redondez: {ecc_str} | Estrellas: {r.star_count}")
        else:
            self.lbl_quality_badge.setText("")

    def _update_current_frame(self):
        if not self.file_list:
            self.lbl_title.setText("Sin fotogramas cargados")
            self.canvas.set_image(None)
            return

        path = self.file_list[self.current_idx]
        filename = os.path.basename(path)
        total = len(self.file_list)
        included_count = total - len(self.excluded_files)

        meta = self._metadata_cache.get(path, {})
        dt_label = meta.get("datetime_str") or meta.get("time_str")
        dt_suffix = f" | {dt_label}" if dt_label and dt_label != "—" else ""
        phase_lbl = meta.get("phase_label")
        phase_suffix = f" | Fase: {phase_lbl}" if phase_lbl and phase_lbl != "Pendiente" else ""
        term = "en collage" if self.mode == "eclipse" else "activas"
        self.lbl_title.setText(f"Fotograma {self.current_idx + 1} de {total} : {filename}{dt_suffix}{phase_suffix} ({included_count}/{total} {term})")

        # Actualizar datos EXIF destacados (Velocidad · ISO)
        exp_s = meta.get("exp_str")
        iso_s = meta.get("iso_str")
        exif_parts = []
        if exp_s and exp_s != "—":
            exif_parts.append(exp_s)
        if iso_s and iso_s != "—":
            exif_parts.append(f"ISO {iso_s}")
        if exif_parts:
            self.lbl_exif_badge.setText(" · ".join(exif_parts))
            self.lbl_exif_badge.setVisible(True)
        else:
            self.lbl_exif_badge.setVisible(False)

        self._update_badges()

        # Sincronizar selección y scroll en la tabla lateral
        if hasattr(self, 'table_blink') and self.table_blink.rowCount() > self.current_idx:
            self.table_blink.blockSignals(True)
            self.table_blink.selectRow(self.current_idx)
            item = self.table_blink.item(self.current_idx, 0)
            if item:
                self.table_blink.scrollToItem(item)
            self.table_blink.blockSignals(False)
            self.table_blink.viewport().update()

        # Obtener imagen de caché o cargar
        if path in self._cache:
            self.canvas.set_image(self._cache[path])
        else:
            auto_stretch = self.chk_auto_stretch.isChecked()
            img_u8 = _fast_load_and_stretch(path, auto_stretch=auto_stretch)
            if img_u8 is not None:
                self._cache[path] = img_u8
            self.canvas.set_image(img_u8)

    def _toggle_side_panel(self):
        """Muestra u oculta el panel lateral de tomas para dar más espacio al lienzo."""
        self._side_panel_visible = not getattr(self, '_side_panel_visible', True)
        self.side_panel.setVisible(self._side_panel_visible)
        self.btn_toggle_side.setText("Ocultar Lista" if self._side_panel_visible else "Mostrar Lista")

    def _toggle_timeline(self):
        """Muestra u oculta la barra gráfica de timeline para dar más espacio vertical al lienzo."""
        self._timeline_visible = not getattr(self, '_timeline_visible', True)
        if hasattr(self, 'timeline_container'):
            self.timeline_container.setVisible(self._timeline_visible)
        elif hasattr(self, 'timeline'):
            self.timeline.setVisible(self._timeline_visible)
        self.btn_toggle_timeline.setText("Ocultar Timeline" if self._timeline_visible else "Mostrar Timeline")

    def _populate_table(self):
        """Rellena la tabla lateral con la lista de tomas e inicia escaneo asíncrono si falta metadatos."""
        self.table_blink.blockSignals(True)
        missing_paths = []
        is_eclipse = (self.mode == "eclipse")

        for row, path in enumerate(self.file_list):
            meta = self._metadata_cache.get(path)
            if meta is None:
                missing_paths.append((row, path))
                meta = {}

            is_inc = (path not in self.excluded_files)

            idx_item = QTableWidgetItem(f"{row + 1}")
            idx_item.setTextAlignment(Qt.AlignCenter)
            idx_item.setForeground(QColor("#94a3b8"))

            if is_eclipse:
                status_text = "★ COL" if is_inc else "·"
                status_color = QColor("#38bdf8") if is_inc else QColor("#334155")
                status_tip = "En Collage (Clic para alternar)" if is_inc else "Disponible (Clic para alternar)"
            else:
                status_text = "✓ INCL" if is_inc else "·"
                status_color = QColor("#10b981") if is_inc else QColor("#334155")
                status_tip = "Incluida (Clic para alternar)" if is_inc else "Descartada (Clic para alternar)"

            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(status_color)
            status_item.setToolTip(status_tip)

            name_item = QTableWidgetItem(os.path.basename(path))
            dt_tip = meta.get("datetime_str") or meta.get("time_str")
            phase_tip = meta.get("phase_label")
            tip_parts = [path]
            if dt_tip and dt_tip != "—":
                tip_parts.append(dt_tip)
            if phase_tip and phase_tip != "Pendiente":
                tip_parts.append(f"Fase: {phase_tip}")
            name_item.setToolTip("\n".join(tip_parts))
            ts = meta.get("timestamp")
            if ts is not None:
                try:
                    name_item.setData(Qt.UserRole, float(ts))
                except (ValueError, TypeError):
                    pass

            if is_eclipse:
                phase_raw = meta.get("phase_label")
                phase_txt, phase_color, is_bold = _format_schematic_phase(phase_raw)
                phase_item = QTableWidgetItem(phase_txt)
                phase_item.setTextAlignment(Qt.AlignCenter)
                phase_item.setForeground(phase_color)
                if is_bold:
                    fnt = phase_item.font()
                    fnt.setBold(True)
                    phase_item.setFont(fnt)
                if phase_raw and phase_raw != "—":
                    phase_item.setToolTip(f"Fase detectada: {phase_raw}")

                obsc_val = meta.get("obscuration_pct")
                obsc_txt = _format_obscuration(obsc_val)
                obsc_item = QTableWidgetItem(obsc_txt)
                obsc_item.setTextAlignment(Qt.AlignCenter)
                obsc_item.setForeground(QColor("#e2e8f0"))

                sharp_val = meta.get("sharpness_score")
                is_trep = meta.get("is_trepidated", False)
                sharp_txt, sharp_col = _format_sharpness(sharp_val, is_trep)
                sharp_item = QTableWidgetItem(sharp_txt)
                sharp_item.setTextAlignment(Qt.AlignCenter)
                sharp_item.setForeground(sharp_col)
                if sharp_val is not None and sharp_val > 0:
                    trep_str = " (Trepidación detectada)" if is_trep else " (Estable)"
                    sharp_item.setToolTip(f"Puntuación de nitidez: {sharp_val:.1f}/100{trep_str}")

                exp_item = QTableWidgetItem(meta.get("exp_str", "..." if path not in self._metadata_cache else "—"))
                exp_item.setTextAlignment(Qt.AlignCenter)

                iso_item = QTableWidgetItem(meta.get("iso_str", "..." if path not in self._metadata_cache else "—"))
                iso_item.setTextAlignment(Qt.AlignCenter)

                self.table_blink.setItem(row, 0, idx_item)
                self.table_blink.setItem(row, 1, status_item)
                self.table_blink.setItem(row, 2, name_item)
                self.table_blink.setItem(row, 3, phase_item)
                self.table_blink.setItem(row, 4, obsc_item)
                self.table_blink.setItem(row, 5, sharp_item)
                self.table_blink.setItem(row, 6, exp_item)
                self.table_blink.setItem(row, 7, iso_item)
            else:
                exp_item = QTableWidgetItem(meta.get("exp_str", "..." if path not in self._metadata_cache else "—"))
                exp_item.setTextAlignment(Qt.AlignCenter)

                iso_item = QTableWidgetItem(meta.get("iso_str", "..." if path not in self._metadata_cache else "—"))
                iso_item.setTextAlignment(Qt.AlignCenter)

                self.table_blink.setItem(row, 0, idx_item)
                self.table_blink.setItem(row, 1, status_item)
                self.table_blink.setItem(row, 2, name_item)
                self.table_blink.setItem(row, 3, exp_item)
                self.table_blink.setItem(row, 4, iso_item)

        if 0 <= self.current_idx < len(self.file_list):
            self.table_blink.selectRow(self.current_idx)
        self.table_blink.blockSignals(False)

        if hasattr(self, 'timeline'):
            self.timeline.set_data(
                files=self.file_list,
                metadata_cache=self._metadata_cache,
                excluded_files=self.excluded_files,
                current_idx=self.current_idx
            )
            gaps = self.timeline.get_gaps()
            if hasattr(self, 'lbl_gaps_summary'):
                gap_count = len(gaps)
                self.lbl_gaps_summary.setText(f"⚠️ {gap_count} gaps" if gap_count > 0 else "")
                self.lbl_gaps_summary.setVisible(gap_count > 0)

        if missing_paths:
            self._start_background_metadata_scan(missing_paths)

    def _start_background_metadata_scan(self, missing_paths: List[tuple[int, str]]):
        def _scan_worker():
            from core.startrails import extract_shot_exif
            for row, path in missing_paths:
                if self._is_shutting_down:
                    break
                try:
                    meta = extract_shot_exif(path)
                except Exception:
                    meta = {}
                if self._is_shutting_down:
                    break
                try:
                    self._meta_scanned.emit(row, path, meta)
                except (RuntimeError, Exception):
                    break

        self._executor.submit(_scan_worker)

    def _on_meta_scanned(self, row: int, path: str, meta: Dict[str, Any]):
        if self._is_shutting_down or not hasattr(self, 'table_blink'):
            return
        self._metadata_cache[path] = meta
        if row < self.table_blink.rowCount():
            self.table_blink.blockSignals(True)
            if self.mode == "eclipse":
                phase_raw = meta.get("phase_label")
                phase_item = self.table_blink.item(row, 3)
                if phase_item:
                    txt, col, bold = _format_schematic_phase(phase_raw)
                    phase_item.setText(txt)
                    phase_item.setForeground(col)
                    if bold:
                        fnt = phase_item.font()
                        fnt.setBold(True)
                        phase_item.setFont(fnt)
                    if phase_raw and phase_raw != "—":
                        phase_item.setToolTip(f"Fase detectada: {phase_raw}")

                obsc_item = self.table_blink.item(row, 4)
                if obsc_item:
                    obsc_item.setText(_format_obscuration(meta.get("obscuration_pct")))

                sharp_item = self.table_blink.item(row, 5)
                if sharp_item:
                    txt, col = _format_sharpness(meta.get("sharpness_score"), meta.get("is_trepidated", False))
                    sharp_item.setText(txt)
                    sharp_item.setForeground(col)
                    sharp_val = meta.get("sharpness_score")
                    if sharp_val is not None and sharp_val > 0:
                        trep_str = " (Trepidación detectada)" if meta.get("is_trepidated", False) else " (Estable)"
                        sharp_item.setToolTip(f"Puntuación de nitidez: {sharp_val:.1f}/100{trep_str}")

                exp_item = self.table_blink.item(row, 6)
                if exp_item:
                    exp_item.setText(meta.get("exp_str", "—"))
                iso_item = self.table_blink.item(row, 7)
                if iso_item:
                    iso_item.setText(meta.get("iso_str", "—"))
            else:
                exp_item = self.table_blink.item(row, 3)
                if exp_item:
                    exp_item.setText(meta.get("exp_str", "—"))
                iso_item = self.table_blink.item(row, 4)
                if iso_item:
                    iso_item.setText(meta.get("iso_str", "—"))
            name_item = self.table_blink.item(row, 2)
            if name_item:
                dt_tip = meta.get("datetime_str") or meta.get("time_str")
                phase_tip = meta.get("phase_label")
                tip_parts = [path]
                if dt_tip and dt_tip != "—":
                    tip_parts.append(dt_tip)
                if phase_tip and phase_tip != "Pendiente":
                    tip_parts.append(f"Fase: {phase_tip}")
                name_item.setToolTip("\n".join(tip_parts))
                ts = meta.get("timestamp")
                if ts is not None:
                    try:
                        name_item.setData(Qt.UserRole, float(ts))
                    except (ValueError, TypeError):
                        pass
            self.table_blink.blockSignals(False)

        if hasattr(self, 'timeline'):
            self.timeline.update_item_metadata(path, meta)
            gaps = self.timeline.get_gaps()
            if hasattr(self, 'lbl_gaps_summary'):
                gap_count = len(gaps)
                self.lbl_gaps_summary.setText(f"⚠️ {gap_count} gaps" if gap_count > 0 else "")
                self.lbl_gaps_summary.setVisible(gap_count > 0)

        if row == self.current_idx:
            dt_label = meta.get("datetime_str") or meta.get("time_str")
            if dt_label and dt_label != "—":
                filename = os.path.basename(path)
                total = len(self.file_list)
                included_count = total - len(self.excluded_files)
                term = "en collage" if self.mode == "eclipse" else "activas"
                self.lbl_title.setText(f"Fotograma {self.current_idx + 1} de {total} : {filename} | {dt_label} ({included_count}/{total} {term})")
            exp_s = meta.get("exp_str")
            iso_s = meta.get("iso_str")
            exif_parts = []
            if exp_s and exp_s != "—":
                exif_parts.append(exp_s)
            if iso_s and iso_s != "—":
                exif_parts.append(f"ISO {iso_s}")
            if exif_parts and hasattr(self, 'lbl_exif_badge'):
                self.lbl_exif_badge.setText(" · ".join(exif_parts))
                self.lbl_exif_badge.setVisible(True)

    def _on_table_selection_changed(self):
        """Cambia el fotograma activo al hacer clic o navegar con teclado en la tabla lateral."""
        selected = self.table_blink.selectedIndexes()
        if selected:
            row = selected[0].row()
            if 0 <= row < len(self.file_list) and row != self.current_idx:
                self.set_current_index(row)

    def _preload_neighbors(self):
        """Precarga fotogramas adyacentes en segundo plano para transición instantánea."""
        if not self.file_list:
            return
        total = len(self.file_list)
        auto_stretch = self.chk_auto_stretch.isChecked()

        indices_to_preload = [
            (self.current_idx + 1) % total,
            (self.current_idx + 2) % total,
            (self.current_idx - 1 + total) % total
        ]

        for idx in indices_to_preload:
            p = self.file_list[idx]
            if p not in self._cache:
                self._executor.submit(self._worker_load, p, auto_stretch)

        # Poda de caché para evitar saturación de RAM (máx 30 imágenes en caché)
        if len(self._cache) > 30:
            keep_paths = {self.file_list[(self.current_idx + i) % total] for i in range(-5, 6)}
            for k in list(self._cache.keys()):
                if k not in keep_paths:
                    del self._cache[k]

    def _worker_load(self, path: str, auto_stretch: bool):
        try:
            img = _fast_load_and_stretch(path, auto_stretch=auto_stretch)
            if img is not None:
                self._cache[path] = img
        except Exception:
            pass

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_A):
            self.prev_frame()
        elif key in (Qt.Key_Right, Qt.Key_D):
            self.next_frame()
        elif key == Qt.Key_Space:
            self.toggle_play()
        elif key in (Qt.Key_X, Qt.Key_Delete):
            self.discard_current_frame()
        elif key in (Qt.Key_P, Qt.Key_Insert):
            self.keep_current_frame()
        elif key == Qt.Key_T:
            self._toggle_timeline()
        elif key == Qt.Key_L:
            self._toggle_side_panel()
        elif key == Qt.Key_Up:
            if hasattr(self, 'table_blink') and self.table_blink.hasFocus():
                super().keyPressEvent(event)
            else:
                self.prev_frame()
        elif key == Qt.Key_Down:
            if hasattr(self, 'table_blink') and self.table_blink.hasFocus():
                super().keyPressEvent(event)
            else:
                self.next_frame()
        elif key == Qt.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)

    def accept(self):
        self._is_shutting_down = True
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        self.selection_applied.emit(self.excluded_files)
        super().accept()

    def reject(self):
        self._is_shutting_down = True
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        super().reject()

    def closeEvent(self, event):
        self._is_shutting_down = True
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        self.selection_applied.emit(self.excluded_files)
        super().closeEvent(event)
