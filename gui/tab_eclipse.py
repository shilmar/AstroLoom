# gui/tab_eclipse.py
"""
gui/tab_eclipse.py - Pestaña dedicada al procesado avanzado de eclipses solares y lunares.

Permite:
1. Selección de modalidad: Solar (corona y protuberancias) vs. Lunar (umbra y penumbra).
2. Carga y gestión de series de bracketing de exposición (RAW, TIFF, FIT).
3. Detección automática del limbo solar/lunar POR CADA IMAGEN individual con escáner radial
   subpíxel inmune a protuberancias y perlas de Baily.
4. Botón de refinado subpíxel y control fino de centro/radio (paso de 0.5 px).
5. Cálculo de derivas (ΔX, ΔY) respecto a la toma de referencia.
6. Alineación afín Lanczos4 y Fusión HDR con estirado Asinh para levantar la corona externa.
7. Filtro NRGF con compuerta radial (Radial Gate) y alcance configurable para suprimir al 100%
   el ruido de fondo en el cielo exterior.
8. Realce de filamentos finos mediante filtro tangencial / bilateral enmascarado.
9. Transferencia directa de la imagen 32-bit al Revelador / Editor.
"""
import os
import tempfile
from typing import Tuple, List, Optional
import numpy as np
import cv2
import tifffile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox, QGroupBox, QRadioButton, QSlider,
    QDoubleSpinBox, QSpinBox, QSplitter, QScrollArea, QCheckBox,
    QSizePolicy, QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog,
    QLineEdit, QTextEdit, QPlainTextEdit
)
from PySide6.QtCore import Qt, Signal, QThread, QPoint, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QCursor, QFont, QBrush, QKeySequence, QShortcut

from core.log_hub import log_hub

from gui.dialog_coronal_stack import CoronalStackDialog
from gui.widgets.viewer_toolbar import ViewerToolbar, CanvasFooterBar
from gui.widgets.split_view_helper import SplitViewHelper

from core.eclipse import (
    extract_exposure_info,
    detect_eclipse_disk,
    refine_disk_subpixel,
    register_frame_to_center,
    crop_frame_roi,
    align_eclipse_bracketing,
    fuse_hdr_bracketing,
    apply_asinh_stretch,
    apply_nrgf,
    apply_mgn,
    apply_coronal_highpass,
    display_detail_rings,
    apply_steerable_g2,
    apply_larson_sekanina,
    apply_swap_filter,
    compute_red_excess_mask,
    apply_nafe,
    apply_wow_edge_aware,
    apply_fnrgf,
    apply_dark_structure_enhancement,
    export_coronal_layers_psd
)
from core.stacking import load_image_as_float32
from core.icc_manager import save_tiff_with_icc
from core.config_manager import get_config_val
from concurrent.futures import ThreadPoolExecutor, as_completed
from gui.widgets.drop_widgets import DropTableWidget, collect_image_files_from_mime
from gui.theme import STYLE_RUN_IDLE, STYLE_TRANSFER_BTN, STYLE_SECONDARY_BTN, STYLE_SPECIAL_ACTION_BTN
from gui.icons import get_icon, get_svg_pixmap, ICON_COLOR_DEFAULT, ICON_COLOR_ACTIVE, ICON_COLOR_ACCENT
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.widgets.export_button import ExportSplitButton


class EclipseModeToggle(QWidget):
    """
    Selector de modalidad Sol / Luna en formato sub-pestañas integradas.
    Diseño armónico con las pestañas de navegación principales de AstroLoom.
    """
    mode_changed = Signal(str)  # "solar" o "lunar"

    def __init__(self, parent=None, is_solar: bool = True):
        super().__init__(parent)
        self._is_solar = is_solar

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 4)
        layout.setSpacing(2)

        self.btn_solar = QPushButton(" Eclipse Solar")
        self.btn_solar.setCheckable(True)
        self.btn_solar.setCursor(Qt.PointingHandCursor)

        self.btn_lunar = QPushButton(" Eclipse Lunar")
        self.btn_lunar.setCheckable(True)
        self.btn_lunar.setCursor(Qt.PointingHandCursor)

        tab_style = """
            QPushButton {
                font-size: 11px;
                font-weight: bold;
                padding: 7px 12px;
                background-color: #1a1a1e;
                color: #a1a1aa;
                border: 1px solid #2e2e35;
                border-bottom: 2px solid transparent;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QPushButton:hover:!checked {
                background-color: #222227;
                color: #e4e4e7;
            }
            QPushButton:checked {
                background-color: #222227;
                color: #ffffff;
                border-color: #2e2e35;
                border-bottom: 2px solid #2f81f7;
            }
        """
        self.btn_solar.setStyleSheet(tab_style)
        self.btn_lunar.setStyleSheet(tab_style)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.addButton(self.btn_solar)
        self.group.addButton(self.btn_lunar)

        self.btn_solar.clicked.connect(lambda: self.set_solar(True))
        self.btn_lunar.clicked.connect(lambda: self.set_solar(False))

        layout.addWidget(self.btn_solar, 1)
        layout.addWidget(self.btn_lunar, 1)

        self._update_state(is_solar)

    def _update_state(self, solar: bool):
        self._is_solar = solar
        if solar:
            self.btn_solar.setChecked(True)
            self.btn_solar.setIcon(get_icon("sun", color="#2f81f7"))
            self.btn_lunar.setIcon(get_icon("moon", color="#71717a"))
        else:
            self.btn_lunar.setChecked(True)
            self.btn_solar.setIcon(get_icon("sun", color="#71717a"))
            self.btn_lunar.setIcon(get_icon("moon", color="#2f81f7"))

    def is_solar(self) -> bool:
        return self._is_solar

    def isChecked(self) -> bool:
        return self._is_solar

    def set_solar(self, solar: bool):
        if self._is_solar != solar:
            self._update_state(solar)
            self.mode_changed.emit("solar" if solar else "lunar")



class EclipseCanvas(QWidget):
    """
    Lienzo interactivo optimizado para eclipses.
    Permite zoom con rueda, paneo con arrastre, visualización del disco/limbo
    y selección interactiva del centro haciendo clic.
    """
    center_clicked = Signal(float, float)
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._drag_hover: bool = False
        self.orig_f32: np.ndarray = None
        self.base_pixmap: QPixmap = None

        # Transformación de vista
        self.zoom_level: float = 1.0
        self.pan_offset = QPoint(0, 0)
        self.last_mouse_pos = QPoint(0, 0)
        self.is_panning: bool = False

        # Guía geométrica del eclipse
        self.show_guide: bool = True
        self.center_x: float = 0.0
        self.center_y: float = 0.0
        self.radius: float = 0.0
        self.mask_offset: float = -2.5
        self.pick_center_mode: bool = False

        # Comparador interactivo de cortina (Split-View)
        self.split_helper = SplitViewHelper()

    @property
    def image(self) -> Optional[np.ndarray]:
        return self.orig_f32

    def set_image(self, img_f32: np.ndarray):
        """Asigna y renderiza una imagen float32 en rango [0, 1]."""
        self.orig_f32 = img_f32
        h, w = img_f32.shape[:2]

        if hasattr(self.parent(), 'canvas_footer') and img_f32 is not None:
            self.parent().canvas_footer.set_image_info(w, h, "32-bit Float", "sRGB")

        img_u8 = np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)
        if img_u8.ndim == 2:
            img_u8 = np.stack([img_u8] * 3, axis=-1)

        bytes_per_line = 3 * w
        qimg = QImage(img_u8.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.base_pixmap = QPixmap.fromImage(qimg)

        self.reset_view()
        self.update()

    def reset_view(self):
        """Ajusta el zoom para que la imagen quepa completa en el lienzo."""
        if self.base_pixmap is None:
            return
        vw, vh = self.width(), self.height()
        iw, ih = self.base_pixmap.width(), self.base_pixmap.height()
        if iw > 0 and ih > 0:
            scale_w = vw / iw
            scale_h = vh / ih
            self.zoom_level = min(scale_w, scale_h) * 0.95
            self.pan_offset = QPoint(
                int((vw - iw * self.zoom_level) / 2),
                int((vh - ih * self.zoom_level) / 2)
            )

    def set_zoom_100(self, active: bool = True):
        """Ajusta el nivel de aumento al 100% nativo centrado en el visor."""
        if self.base_pixmap is None:
            return
        if active:
            self.zoom_level = 1.0
            vw, vh = self.width(), self.height()
            iw, ih = self.base_pixmap.width(), self.base_pixmap.height()
            self.pan_offset = QPoint(
                int((vw - iw) / 2),
                int((vh - ih) / 2)
            )
        else:
            self.reset_view()
        self.update()

    def fit_view(self):
        """Ajusta la imagen completa a la ventana."""
        self.reset_view()
        self.update()

    def set_guide_params(self, cx: float, cy: float, r: float, show: bool = True, mask_offset: float = -2.5):
        """Actualiza las coordenadas de la guía circular de limbo y máscara interna."""
        self.center_x = cx
        self.center_y = cy
        self.radius = r
        self.show_guide = show
        self.mask_offset = mask_offset
        self.update()

    def set_split_view_enabled(self, enabled: bool) -> None:
        """Activa o desactiva la cortina comparadora interactiva."""
        if self.split_helper.set_enabled(enabled):
            self.update()

    def set_split_orientation(self, orientation: str) -> None:
        """Establece la orientación de la cortina ('vertical' u 'horizontal')."""
        if self.split_helper.set_orientation(orientation):
            self.update()

    def toggle_split_orientation(self) -> str:
        """Alterna entre orientación vertical y horizontal."""
        new_ori = self.split_helper.toggle_orientation()
        self.update()
        return new_ori

    def set_before_image(self, img_f32: Optional[np.ndarray]) -> None:
        """Asigna la imagen de referencia previa ('Antes') a partir de un array float32/uint8."""
        self.split_helper.set_before_image(img_f32)
        if self.split_helper.enabled:
            self.update()

    def set_before_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        """Asigna la imagen de referencia previa ('Antes') directamente como QPixmap."""
        self.split_helper.set_before_pixmap(pixmap)
        if self.split_helper.enabled:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(18, 18, 22))

        if self.base_pixmap is None:
            painter.setPen(QColor(150, 150, 160))
            painter.drawText(self.rect(), Qt.AlignCenter, "Carga una serie de bracketing para comenzar.")
            if self._drag_hover:
                self._draw_drag_feedback(painter)
            return

        # Dibujar imagen con zoom y pan (o cortina interactiva Antes / Después)
        iw = self.base_pixmap.width() * self.zoom_level
        ih = self.base_pixmap.height() * self.zoom_level
        target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), iw, ih)

        if self.split_helper.enabled and self.split_helper.before_pixmap and not self.split_helper.before_pixmap.isNull():
            self.split_helper.draw(
                painter, target_rect,
                self.split_helper.before_pixmap,
                self.base_pixmap,
                badge_before="BASE HDR",
                badge_after="FILTRADA"
            )
        else:
            painter.drawPixmap(target_rect.toRect(), self.base_pixmap)

        # Realimentación visual al arrastrar archivos sobre el visor
        if self._drag_hover:
            self._draw_drag_feedback(painter)

        # Dibujar guía geométrica del limbo y máscara interna si está habilitada
        if self.show_guide and self.radius > 5:
            scx = self.pan_offset.x() + self.center_x * self.zoom_level
            scy = self.pan_offset.y() + self.center_y * self.zoom_level
            sr = self.radius * self.zoom_level
            mask_r = max(2.0, self.radius + self.mask_offset)
            smr = mask_r * self.zoom_level

            # 1. Guía de limbo físico (Cian discontinuo)
            pen_bg = QPen(QColor(0, 0, 0, 220), 3, Qt.DashLine)
            pen_fg = QPen(QColor(0, 240, 255, 240), 1.5, Qt.DashLine)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(pen_bg)
            painter.drawEllipse(QPoint(int(scx), int(scy)), int(sr), int(sr))
            painter.setPen(pen_fg)
            painter.drawEllipse(QPoint(int(scx), int(scy)), int(sr), int(sr))

            # 2. Guía de máscara lunar interior (Ámbar punteado)
            if smr > 3:
                pen_mask_bg = QPen(QColor(0, 0, 0, 200), 2.5, Qt.DotLine)
                pen_mask_fg = QPen(QColor(255, 183, 77, 230), 1.3, Qt.DotLine)
                painter.setPen(pen_mask_bg)
                painter.drawEllipse(QPoint(int(scx), int(scy)), int(smr), int(smr))
                painter.setPen(pen_mask_fg)
                painter.drawEllipse(QPoint(int(scx), int(scy)), int(smr), int(smr))

            # Cruz central
            cl = 10
            painter.setPen(pen_bg)
            painter.drawLine(int(scx - cl), int(scy), int(scx + cl), int(scy))
            painter.drawLine(int(scx), int(scy - cl), int(scx), int(scy + cl))
            painter.setPen(pen_fg)
            painter.drawLine(int(scx - cl), int(scy), int(scx + cl), int(scy))
            painter.drawLine(int(scx), int(scy - cl), int(scx), int(scy + cl))

            painter.setPen(QColor(0, 240, 255))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            info_txt = f"Limbo: R={self.radius:.1f}px | Máscara: R={mask_r:.1f}px ({self.mask_offset:+.1f}px)"
            painter.drawText(int(scx + sr + 8), int(scy), info_txt)

    def mousePressEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self.base_pixmap and self.split_helper.enabled:
                iw = self.base_pixmap.width() * self.zoom_level
                ih = self.base_pixmap.height() * self.zoom_level
                target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), iw, ih)
                if self.split_helper.handle_mouse_press(event.pos(), target_rect, is_double_click=False):
                    self.setCursor(Qt.SplitVCursor if self.split_helper.orientation == "horizontal" else Qt.SplitHCursor)
                    self.update()
                    return

            if self.pick_center_mode and self.base_pixmap is not None:
                img_x = (event.position().x() - self.pan_offset.x()) / self.zoom_level
                img_y = (event.position().y() - self.pan_offset.y()) / self.zoom_level
                iw, ih = self.base_pixmap.width(), self.base_pixmap.height()
                if 0 <= img_x < iw and 0 <= img_y < ih:
                    self.center_clicked.emit(img_x, img_y)
                    self.pick_center_mode = False
                    self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.base_pixmap and self.split_helper.enabled:
            iw = self.base_pixmap.width() * self.zoom_level
            ih = self.base_pixmap.height() * self.zoom_level
            target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), iw, ih)
            if self.split_helper.handle_mouse_press(event.pos(), target_rect, is_double_click=True):
                self.update()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()
            return

        if self.base_pixmap and self.split_helper.enabled:
            iw = self.base_pixmap.width() * self.zoom_level
            ih = self.base_pixmap.height() * self.zoom_level
            target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), iw, ih)
            handled, cursor = self.split_helper.handle_mouse_move(event.pos(), target_rect)
            if handled:
                self.update()
                return
            if not self.pick_center_mode:
                if cursor is not None:
                    self.setCursor(cursor)
                elif not self.is_panning:
                    self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.is_panning = False
            self.setCursor(Qt.CrossCursor if self.pick_center_mode else Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self.split_helper.is_dragging:
            self.split_helper.handle_mouse_release()
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

    def wheelEvent(self, event):
        if self.base_pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 0.87
        new_zoom = np.clip(self.zoom_level * factor, 0.05, 30.0)

        mouse_pos = event.position()
        self.pan_offset = QPoint(
            int(mouse_pos.x() - (mouse_pos.x() - self.pan_offset.x()) * (new_zoom / self.zoom_level)),
            int(mouse_pos.y() - (mouse_pos.y() - self.pan_offset.y()) * (new_zoom / self.zoom_level))
        )
        self.zoom_level = new_zoom
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.base_pixmap is not None and self.zoom_level == 1.0:
            self.reset_view()

    def _draw_drag_feedback(self, painter: QPainter) -> None:
        """Dibuja un marco delimitador punteado y una insignia informativa durante el arrastre."""
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#2f81f7"), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(18, 18, 20, 180)))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(6, 6, -6, -6), 8.0, 8.0)

        cw, ch = 270, 42
        cx = (self.width() - cw) / 2
        cy = (self.height() - ch) / 2
        pill_rect = QRectF(cx, cy, cw, ch)
        painter.setPen(QPen(QColor("#2f81f7"), 1.5))
        painter.setBrush(QBrush(QColor(26, 26, 30, 240)))
        painter.drawRoundedRect(pill_rect, 6.0, 6.0)
        painter.setPen(QColor("#ffffff"))
        f = painter.font()
        f.setBold(True)
        f.setPointSize(10)
        painter.setFont(f)
        painter.drawText(pill_rect, Qt.AlignCenter, "Soltar archivo(s) para cargar")
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


class EclipseWorker(QThread):
    """Worker en segundo plano para detección de limbo, alineación y cálculo intensivo."""
    progress = Signal(int, str)
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(self, task_type: str, params: dict):
        super().__init__()
        self.task_type = task_type
        self.params = params

    def run(self):
        try:
            if self.task_type == "batch_detect":
                files = self.params["files"]
                mode = self.params.get("mode", "solar")
                ref_idx = self.params.get("ref_idx", 0)
                ref_r_param = self.params.get("ref_r")
                total = len(files)
                detections = [None] * total

                self.progress.emit(5, "Iniciando detección subpíxel de limbo (1:1 nativo)...")

                # 1. Detectar primero la toma de referencia para anclar el radio y centro físicos esperados
                ref_img = load_image_as_float32(files[ref_idx])
                ref_cx, ref_cy, ref_r = detect_eclipse_disk(ref_img, mode=mode, expected_radius=None)
                detections[ref_idx] = (ref_cx, ref_cy, ref_r)
                expected_r = ref_r if ref_r > 10 else None
                expected_c = (ref_cx, ref_cy) if expected_r is not None else None

                # 2. Detectar el resto de tomas usando expected_r y expected_c en paralelo a resolución 1:1 nativa
                items_to_detect = [(idx, path) for idx, path in enumerate(files) if idx != ref_idx]

                if items_to_detect:
                    try:
                        cpu_workers = int(get_config_val("cpu_workers", max(2, (os.cpu_count() or 4) - 2)))
                    except Exception:
                        cpu_workers = max(2, (os.cpu_count() or 4) - 2)
                    eff_workers = max(1, min(cpu_workers, 4, len(items_to_detect)))

                    def _detect_task(item):
                        idx, path = item
                        img = load_image_as_float32(path)
                        cx, cy, r = detect_eclipse_disk(
                            img, mode=mode, expected_radius=expected_r, expected_center=expected_c
                        )
                        return idx, (cx, cy, r), os.path.basename(path)

                    completed = 1  # La referencia ya está procesada
                    if eff_workers == 1 or len(items_to_detect) == 1:
                        for item in items_to_detect:
                            if self.isInterruptionRequested():
                                break
                            idx, det, bname = _detect_task(item)
                            detections[idx] = det
                            completed += 1
                            pct = int(10 + (completed / total) * 85)
                            self.progress.emit(pct, f"Escaneando limbo ({completed}/{total}): {bname}")
                    else:
                        with ThreadPoolExecutor(max_workers=eff_workers) as executor:
                            future_to_item = {executor.submit(_detect_task, item): item for item in items_to_detect}
                            for future in as_completed(future_to_item):
                                if self.isInterruptionRequested():
                                    try:
                                        executor.shutdown(wait=False, cancel_futures=True)
                                    except Exception:
                                        executor.shutdown(wait=False)
                                    break
                                idx, det, bname = future.result()
                                detections[idx] = det
                                completed += 1
                                pct = int(10 + (completed / total) * 85)
                                self.progress.emit(pct, f"Escaneando limbo ({completed}/{total}): {bname}")

                if self.isInterruptionRequested():
                    return

                self.progress.emit(100, "Detección subpíxel completada para todas las tomas.")
                self.finished.emit(detections, "batch_detect")

            elif self.task_type == "hdr_fusion":
                files = self.params["files"]
                exposures = self.params["exposures"]
                centers = self.params.get("centers", None)
                ref_idx = self.params.get("ref_idx", 0)
                auto_align = self.params.get("auto_align", True)
                sat_thresh = self.params.get("sat_thresh", 0.90)
                asinh_factor = self.params.get("asinh_factor", 0.0)
                mode = self.params.get("mode", "solar")

                cached_aligned = self.params.get("cached_aligned_images")
                if cached_aligned is not None:
                    self.progress.emit(20, "Reutilizando tomas alineadas en memoria caché (omitiendo lectura de disco y alineación)...")
                    aligned_images = cached_aligned
                    final_centers = self.params["cached_final_centers"]
                    raw_centers = self.params.get("cached_raw_centers", final_centers)
                    shifts = self.params["cached_shifts"]
                    was_cached = True
                else:
                    was_cached = False
                    total = len(files)
                    loaded_images = [None] * total
                    final_centers = [None] * total
                    raw_centers = [None] * total
                    self.progress.emit(5, "Iniciando carga paralela de tomas de bracketing (1:1 nativo)...")

                    expected_r = None
                    expected_c = None
                    if centers is not None:
                        valid_rs = [c[2] for c in centers if c is not None and len(c) > 2 and c[2] > 10]
                        if valid_rs:
                            expected_r = float(np.median(valid_rs))
                        if ref_idx < len(centers) and centers[ref_idx] is not None and len(centers[ref_idx]) > 2:
                            expected_c = (float(centers[ref_idx][0]), float(centers[ref_idx][1]))

                    try:
                        cpu_workers = int(get_config_val("cpu_workers", max(2, (os.cpu_count() or 4) - 2)))
                    except Exception:
                        cpu_workers = max(2, (os.cpu_count() or 4) - 2)
                    eff_workers = max(1, min(cpu_workers, 4, total))

                    crop_roi = self.params.get("crop_roi", False)
                    crop_size = int(self.params.get("crop_size", 3000))

                    def _load_and_detect_one(idx_path):
                        idx, path = idx_path
                        img = load_image_as_float32(path)
                        if centers is not None and idx < len(centers) and centers[idx] is not None and centers[idx][2] > 5:
                            raw_c = centers[idx]
                        else:
                            cx, cy, r = detect_eclipse_disk(
                                img, mode=mode, expected_radius=expected_r, expected_center=expected_c
                            )
                            raw_c = (cx, cy, r)

                        if crop_roi and raw_c is not None and len(raw_c) > 2 and raw_c[2] > 5:
                            cropped_img, cropped_c = crop_frame_roi(img, raw_c[0], raw_c[1], raw_c[2], crop_size=crop_size)
                        else:
                            cropped_img, cropped_c = img, raw_c

                        return idx, cropped_img, raw_c, cropped_c, os.path.basename(path)

                    completed = 0
                    items = list(enumerate(files))
                    if eff_workers == 1 or total == 1:
                        for item in items:
                            if self.isInterruptionRequested():
                                break
                            idx, img, raw_c, cropped_c, bname = _load_and_detect_one(item)
                            loaded_images[idx] = img
                            raw_centers[idx] = raw_c
                            final_centers[idx] = cropped_c
                            completed += 1
                            self.progress.emit(
                                int(5 + (completed / total) * 40),
                                f"Cargando ({completed}/{total}): {bname}"
                            )
                    else:
                        with ThreadPoolExecutor(max_workers=eff_workers) as executor:
                            future_to_item = {executor.submit(_load_and_detect_one, item): item for item in items}
                            for future in as_completed(future_to_item):
                                if self.isInterruptionRequested():
                                    try:
                                        executor.shutdown(wait=False, cancel_futures=True)
                                    except Exception:
                                        executor.shutdown(wait=False)
                                    break
                                idx, img, raw_c, cropped_c, bname = future.result()
                                loaded_images[idx] = img
                                raw_centers[idx] = raw_c
                                final_centers[idx] = cropped_c
                                completed += 1
                                self.progress.emit(
                                    int(5 + (completed / total) * 40),
                                    f"Cargando ({completed}/{total}): {bname}"
                                )

                    if self.isInterruptionRequested():
                        return

                    if auto_align and len(final_centers) == len(loaded_images) and all(img is not None for img in loaded_images):
                        self.progress.emit(50, "Alineando tomas por centro de limbo (interpolación Lanczos4)...")
                        center_pts = [(c[0], c[1]) for c in final_centers]
                        aligned_images, shifts = align_eclipse_bracketing(
                            loaded_images,
                            center_pts,
                            ref_idx=ref_idx,
                            consume_input=True
                        )
                    else:
                        aligned_images = [img for img in loaded_images if img is not None]
                        shifts = [(0.0, 0.0)] * len(aligned_images)

                    # shortcut: liberamos tomas originales antes de fusionar para reducir el pico de memoria a la mitad
                    del loaded_images

                ref_center_info = final_centers[ref_idx] if (ref_idx < len(final_centers) and final_centers[ref_idx] is not None) else None
                lunar_mask = (ref_center_info[0], ref_center_info[1], ref_center_info[2]) if (ref_center_info is not None and len(ref_center_info) > 2 and ref_center_info[2] > 5) else None
                subtract_sky = self.params.get("subtract_sky", True)

                progress_msg = "Calculando ponderaciones y fusionando flujo fotométrico HDR..." if mode == "solar" else "Fusionando tomas con exposición multiescala (Mertens HDR)..."
                self.progress.emit(75, progress_msg)
                auto_wb = self.params.get("auto_wb_corona", True)
                gradual_handover = self.params.get("gradual_handover", True)
                corona_warmth = self.params.get("corona_warmth", 0.0)
                sky_temp = self.params.get("sky_temp", 0.0)
                equalize_sky = self.params.get("equalize_sky_gradient", False)
                hdr_raw = fuse_hdr_bracketing(
                    aligned_images,
                    exposures,
                    saturation_threshold=sat_thresh,
                    asinh_stretch=0.0,
                    mode=mode,
                    lunar_mask_params=lunar_mask,
                    subtract_sky=subtract_sky,
                    auto_wb_corona=auto_wb,
                    gradual_handover=gradual_handover,
                    corona_warmth=corona_warmth,
                    sky_temp=sky_temp,
                    equalize_sky_gradient=equalize_sky,
                    clip_highlights=False
                )

                prom_protect = float(self.params.get("prom_protect", 0.45))
                white_pt = 1.0 + prom_protect * 5.0
                color_pres = 0.90 if prom_protect > 0.001 else 0.0
                hdr_stretched = apply_asinh_stretch(
                    hdr_raw,
                    stretch_factor=asinh_factor,
                    shoulder_threshold=0.78,
                    white_point=white_pt,
                    color_preserve=color_pres
                )

                self.progress.emit(100, "Fusión HDR de 32 bits completada.")
                self.finished.emit({
                    "hdr_raw": hdr_raw,
                    "hdr_image": hdr_stretched,
                    "centers": final_centers,
                    "raw_centers": raw_centers,
                    "shifts": shifts,
                    "ref_idx": ref_idx,
                    "aligned_images": aligned_images,
                    "fused_geom": ref_center_info,
                    "was_cached": was_cached
                }, "hdr_fusion")

            elif self.task_type == "apply_filters":
                base_img = self.params["image"]
                cx = self.params["cx"]
                cy = self.params["cy"]
                r = self.params["radius"]
                mask_r = self.params.get("mask_radius", r)
                filter_type = self.params.get("filter_type", "mgn")
                use_highpass = self.params.get("use_highpass", False)
                hp_strength = self.params.get("hp_strength", 0.5)
                reach = self.params.get("reach", 3.5)

                scale = float(self.params.get("scale", 1.0))
                blend = self.params.get("blend", 0.50)
                use_detail_rings = self.params.get("use_detail_rings", True)
                if filter_type == "steerable_g2":
                    self.progress.emit(25, "Calculando filtro Steerable G2 (Freeman & Adelson / tensor de estructura)...")
                    blend = self.params.get("blend", 0.50)
                    sigmas = tuple(max(0.75, s * scale) for s in (1.5, 3.0, 6.0, 12.0))
                    filtered = apply_steerable_g2(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        radius_min=mask_r,
                        sigmas=sigmas,
                        blend_ratio=blend,
                        radius_max_factor=reach,
                        use_detail_rings=use_detail_rings
                    )
                elif filter_type == "larson_sekanina":
                    self.progress.emit(25, "Calculando gradiente rotacional Larson-Sekanina (Keys bicubic)...")
                    ls_dtheta = self.params.get("ls_delta_theta", 1.0)
                    ls_protect = self.params.get("ls_protect_prominences", True)
                    blend = self.params.get("blend", 0.50)
                    filtered = apply_larson_sekanina(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        delta_theta_deg=ls_dtheta,
                        blend_ratio=blend,
                        protect_prominences=ls_protect,
                        radius_min=mask_r,
                        radius_max_factor=reach,
                        use_detail_rings=use_detail_rings
                    )
                elif filter_type == "swap":
                    self.progress.emit(25, "Calculando filtro SWAP (compresión polar angular tipo PROBA2)...")
                    swap_win = self.params.get("swap_window_deg", 15.0)
                    swap_c0 = self.params.get("swap_c0", 0.75)
                    blend = self.params.get("blend", 1.0)
                    filtered = apply_swap_filter(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        radius_min=mask_r,
                        window_deg=swap_win,
                        c0=swap_c0,
                        blend_ratio=blend,
                        radius_max_factor=reach
                    )
                elif filter_type == "mgn":
                    self.progress.emit(25, "Calculando filtro multiescala MGN (Morgan & Druckmüller)...")
                    mgn_k = self.params.get("mgn_k", 0.70)
                    mgn_gamma = self.params.get("mgn_gamma", 2.8)
                    blend = self.params.get("blend", 0.50)
                    noise_floor = self.params.get("mgn_noise_floor", 0.003)
                    chroma_denoise = self.params.get("mgn_chroma_denoise", 16.0) * scale
                    lunar_mask = (cx, cy, mask_r) if mask_r > 5 else None
                    sigmas = tuple(max(0.75, s * scale) for s in (1.5, 3.0, 6.0, 12.0, 24.0, 48.0))

                    filtered = apply_mgn(
                        base_img,
                        sigmas=sigmas,
                        k=mgn_k,
                        gamma=mgn_gamma,
                        blend_ratio=blend,
                        noise_floor=noise_floor,
                        chroma_denoise_sigma=chroma_denoise,
                        lunar_mask_params=lunar_mask
                    )
                elif filter_type == "nafe":
                    self.progress.emit(25, "Calculando ecualización difusa NAFE (Druckmüller 2013 / FFT)...")
                    sigma = self.params.get("nafe_sigma", 0.035)
                    use_ov = self.params.get("nafe_use_overlay", True)
                    blend = self.params.get("blend", 0.79)
                    filtered = apply_nafe(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        radius_min=mask_r,
                        sigma=sigma,
                        blend_ratio=blend,
                        use_overlay=use_ov
                    )
                elif filter_type == "wow":
                    self.progress.emit(25, "Calculando WOW guiado bilateral (starlet B3 / AEL 1.3.5)...")
                    scales = self.params.get("wow_scales", 7)
                    guided = self.params.get("wow_guided", True)
                    blend = self.params.get("blend", 0.50)
                    filtered = apply_wow_edge_aware(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        radius_min=mask_r,
                        n_scales=scales,
                        guided=guided,
                        blend_ratio=blend,
                        use_detail_rings=use_detail_rings
                    )
                elif filter_type == "fnrgf":
                    self.progress.emit(25, "Calculando FNRGF azimutal de Fourier (Morgan & Druckmüller)...")
                    order = self.params.get("fnrgf_order", 4)
                    blend = self.params.get("blend", 0.55)
                    bin_px = max(0.5, self.params.get("bin_px", 1.0) * scale)
                    filtered = apply_fnrgf(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        fourier_order=order,
                        bin_px=bin_px,
                        blend_ratio=blend,
                        radius_min=mask_r,
                        radius_max_factor=reach,
                        use_detail_rings=use_detail_rings
                    )
                else:
                    bin_px = max(0.5, self.params.get("bin_px", 2.0) * scale)
                    robust = self.params.get("robust", True)
                    blend = self.params.get("blend", 0.55)
                    part_annuli = self.params.get("nrgf_partial_annuli", "observed")

                    self.progress.emit(20, "Calculando filtro NRGF (gradiente radial polar)...")
                    filtered = apply_nrgf(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        bin_px=bin_px,
                        robust=robust,
                        blend_ratio=blend,
                        radius_min=mask_r,
                        radius_max_factor=reach,
                        use_detail_rings=use_detail_rings,
                        partial_annuli=part_annuli
                    )

                if use_highpass and hp_strength > 0.01:
                    self.progress.emit(75, "Realzando filamentos finos mediante filtro tangencial...")
                    filtered = apply_coronal_highpass(
                        filtered,
                        strength=hp_strength,
                        radius_min=mask_r,
                        center_x=cx,
                        center_y=cy,
                        radius_max_factor=reach,
                        use_detail_rings=use_detail_rings
                    )

                use_dark = self.params.get("use_dark_structure", False)
                dark_amount = self.params.get("dark_amount", 0.0)
                dark_layers = self.params.get("dark_layers", 5)
                dark_protect_prom = self.params.get("dark_protect_prominences", True)

                if use_dark and dark_amount > 0.001:
                    self.progress.emit(85, "Realzando estructuras oscuras y cavidades coronales...")
                    prom_m = None
                    if dark_protect_prom and filtered.ndim == 3:
                        prom_m = compute_red_excess_mask(filtered, cx, cy, radius_min=mask_r)
                    filtered = apply_dark_structure_enhancement(
                        filtered,
                        amount=dark_amount,
                        center_x=cx,
                        center_y=cy,
                        radius_mask=mask_r,
                        radius_max_factor=reach,
                        layers=dark_layers,
                        prominence_mask=prom_m
                    )

                self.progress.emit(100, "Filtros de corona aplicados con éxito.")
                self.finished.emit(filtered, "apply_filters")

            elif self.task_type == "export_coronal_psd":
                filepath = self.params["filepath"]
                base_img = self.params["base_image"]
                cx = self.params["cx"]
                cy = self.params["cy"]
                r = self.params["radius"]
                depth = self.params.get("depth", 16)

                self.progress.emit(20, "Generando capas de filtros coronales (NRGF, WOW, MGN, NAFE, Steerable G2, Larson-Sekanina, SWAP)...")
                ok, desc, num_layers = export_coronal_layers_psd(
                    filepath,
                    base_img,
                    cx,
                    cy,
                    r,
                    depth=depth,
                    log_fn=lambda m: self.progress.emit(50, m)
                )
                if not ok:
                    raise RuntimeError(desc)
                self.progress.emit(100, f"PSD exportado: {desc}")
                self.finished.emit((filepath, desc), "export_coronal_psd")

            elif self.task_type == "compute_coronal_stack":
                base_img = self.params["base_image"]
                cx = self.params["cx"]
                cy = self.params["cy"]
                r = self.params["radius"]
                selected_configs = self.params["selected_configs"]
                out_base_path = self.params["out_base_path"]

                # 1. Guardar la imagen base como TIFF 32-bit float con ICC
                self.progress.emit(10, "Guardando imagen base HDR en 32-bit...")
                photometric = 'minisblack' if base_img.ndim == 2 else 'rgb'
                out_img = np.ascontiguousarray(np.clip(base_img, 0.0, 1.0), dtype=np.float32)
                save_tiff_with_icc(out_base_path, out_img, compression='zlib', photometric=photometric)

                base_rgb = out_img
                if base_rgb.ndim == 2:
                    base_rgb = np.repeat(base_rgb[..., None], 3, axis=2)

                computed_layers = []
                total = len(selected_configs)
                for i, cfg in enumerate(selected_configs):
                    flt_id = cfg["id"]
                    name = cfg["name"]
                    pct = 15 + int(((i + 1) / max(1, total)) * 80)
                    self.progress.emit(pct, f"Calculando {name}...")

                    filtered = None
                    mask_r = self.params.get("mask_radius", r)
                    fp = self.params.get("filter_params", {})
                    use_detail_rings = fp.get("use_detail_rings", True)

                    if flt_id == "mgn":
                        lunar_mask = (cx, cy, mask_r) if mask_r > 5 else None
                        filtered = apply_mgn(
                            base_rgb,
                            sigmas=(1.5, 3.0, 6.0, 12.0, 24.0, 48.0),
                            k=fp.get("mgn_k", 0.70),
                            gamma=fp.get("mgn_gamma", 2.8),
                            blend_ratio=fp.get("mgn_blend", 0.50),
                            noise_floor=fp.get("mgn_noise_floor", 0.003),
                            chroma_denoise_sigma=fp.get("mgn_chroma_denoise", 16.0),
                            lunar_mask_params=lunar_mask
                        )
                    elif flt_id == "wow":
                        filtered = apply_wow_edge_aware(
                            base_rgb, cx, cy, radius_min=mask_r,
                            n_scales=fp.get("wow_scales", 7),
                            guided=fp.get("wow_guided", True),
                            blend_ratio=fp.get("wow_blend", 0.50),
                            use_detail_rings=use_detail_rings
                        )
                    elif flt_id == "nafe":
                        filtered = apply_nafe(
                            base_rgb, cx, cy, radius_min=mask_r,
                            sigma=fp.get("nafe_sigma", 0.035),
                            blend_ratio=fp.get("nafe_blend", 0.79),
                            use_overlay=fp.get("nafe_use_overlay", True)
                        )
                    elif flt_id == "nrgf":
                        filtered = apply_nrgf(
                            base_rgb, cx, cy, radius_min=mask_r,
                            bin_px=fp.get("bin_px", 2.0),
                            robust=fp.get("robust", True),
                            blend_ratio=fp.get("nrgf_blend", 0.55),
                            radius_max_factor=fp.get("nrgf_reach", 3.5),
                            use_detail_rings=use_detail_rings,
                            partial_annuli=fp.get("nrgf_partial_annuli", "observed")
                        )
                    elif flt_id == "fnrgf":
                        filtered = apply_fnrgf(
                            base_rgb, cx, cy, radius_min=mask_r,
                            fourier_order=fp.get("fnrgf_order", 4),
                            bin_px=fp.get("bin_px", 2.0),
                            blend_ratio=fp.get("fnrgf_blend", 0.55),
                            radius_max_factor=fp.get("fnrgf_reach", 3.5),
                            use_detail_rings=use_detail_rings
                        )
                    elif flt_id == "steerable_g2":
                        filtered = apply_steerable_g2(
                            base_rgb, cx, cy, radius_min=mask_r,
                            blend_ratio=fp.get("g2_blend", 0.50),
                            radius_max_factor=fp.get("g2_reach", 3.5),
                            use_detail_rings=use_detail_rings
                        )
                    elif flt_id == "larson_sekanina":
                        filtered = apply_larson_sekanina(
                            base_rgb, cx, cy, radius_min=mask_r,
                            delta_theta_deg=fp.get("ls_delta_theta", 1.0),
                            blend_ratio=fp.get("ls_blend", 0.20),
                            protect_prominences=fp.get("ls_protect_prominences", True),
                            radius_max_factor=fp.get("ls_reach", 3.5),
                            use_detail_rings=use_detail_rings
                        )
                    elif flt_id == "swap":
                        filtered = apply_swap_filter(
                            base_rgb, cx, cy, radius_min=mask_r,
                            window_deg=fp.get("swap_window_deg", 15.0),
                            c0=fp.get("swap_c0", 0.75),
                            blend_ratio=fp.get("swap_blend", 1.0)
                        )
                    elif flt_id == "dark_structure":
                        prom_m = None
                        if fp.get("dark_protect_prominences", True) and base_rgb.ndim == 3:
                            prom_m = compute_red_excess_mask(base_rgb, cx, cy, radius_min=mask_r)
                        filtered = apply_dark_structure_enhancement(
                            base_rgb,
                            center_x=cx,
                            center_y=cy,
                            radius_mask=mask_r,
                            amount=fp.get("dark_amount", 0.40),
                            prominence_mask=prom_m
                        )

                    if filtered is not None:
                        computed_layers.append({
                            "name": name,
                            "image_data": filtered,
                            "blend_mode": cfg["blend_mode"],
                            "opacity": cfg["opacity"]
                        })

                h_b, w_b = base_rgb.shape[:2]
                if w_b > 0 and h_b > 0 and r > 5:
                    computed_layers.append({
                        "layer_type": "gradient_radial",
                        "name": "Fondo Exterior (2R)",
                        "radial_center": (float(cx / w_b), float(cy / h_b)),
                        "radial_radius_x": float((2.0 * r) / w_b),
                        "radial_radius_y": float((2.0 * r) / h_b),
                        "radial_feather": 0.50,
                        "inverted": True,
                    })

                self.progress.emit(100, f"Multicapa coronal calculada ({len(computed_layers)} capas).")
                self.finished.emit((out_base_path, computed_layers), "compute_coronal_stack")

        except Exception as e:
            self.error.emit(f"Error durante el procesado: {str(e)}")


class EclipseTab(QWidget):
    """Pestaña interactiva de Eclipses (Solar y Lunar)."""
    export_to_developer = Signal(str)
    export_coronal_stack_to_developer = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bracketing_files = []      # Lista de dicts con metadatos y centros
        self.ref_file_idx = 0           # Índice de la toma de referencia para alineación
        self.current_preview_img = None # Imagen float32 visible
        self.raw_fused_hdr = None       # Flujo puro HDR sin estirar
        self.fused_hdr_img = None       # Imagen HDR con estirado
        self.proxy_hdr_img = None       # Imagen proxy al 50% para previsualización rápida
        self._current_is_proxy: bool = False # Flag que indica si la imagen en el visor es un proxy
        self.fused_hdr_geom = None      # Geometría local (cx, cy, r) de la imagen fusionada/recortada
        self._viewing_fused: bool = False
        self.final_filtered_img = None   # Imagen tras aplicar filtros de corona
        self.worker = None
        self.cached_aligned_data = None   # Caché intermedia de tomas alineadas en memoria
        self._current_hdr_cache_key = None

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -------------------------------------------------------------
        # 1. PANEL DE CONTROL IZQUIERDO (Scrollable con Log Fijo Abajo)
        # -------------------------------------------------------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left_panel.setMaximumWidth(520)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(4, 4, 4, 4)
        left_panel_layout.setSpacing(6)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        panel_content = QWidget()
        left_layout = QVBoxLayout(panel_content)
        left_layout.setContentsMargins(4, 4, 10, 4)
        left_layout.setSpacing(12)

        # MODALIDAD DE ECLIPSE: Selector Compacto Boolean Toggle (Sol / Luna)
        self.mode_toggle = EclipseModeToggle(self, is_solar=True)
        self.rb_solar = self.mode_toggle  # Compatibilidad directa con llamadas a rb_solar.isChecked()
        self.mode_toggle.mode_changed.connect(self._on_mode_changed)
        left_layout.addWidget(self.mode_toggle)

        # SECCIÓN 1: Serie de Bracketing y Alineación
        self.sec_brack = CollapsibleSection("1. Serie de Bracketing", is_expanded=True, icon_name="camera")
        brack_layout = self.sec_brack.content_layout

        btn_row_files = QHBoxLayout()
        self.btn_load_brack = QPushButton("Cargar Bracketing...")
        self.btn_load_brack.setIcon(get_icon("folder"))
        self.btn_load_brack.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_load_brack.clicked.connect(self._on_load_bracketing_clicked)
        self.btn_clear_brack = QPushButton("Limpiar")
        self.btn_clear_brack.setIcon(get_icon("trash"))
        self.btn_clear_brack.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_clear_brack.clicked.connect(self._on_clear_bracketing_clicked)
        self.btn_blink = QPushButton("Visor Blink")
        self.btn_blink.setIcon(get_icon("eye"))
        self.btn_blink.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_blink.setToolTip("Abre el visor interactivo Blink para revisar tomas de bracketing y cribar con teclado (X/P)")
        self.btn_blink.clicked.connect(self.open_blink_viewer)
        btn_row_files.addWidget(self.btn_load_brack)
        btn_row_files.addWidget(self.btn_clear_brack)
        btn_row_files.addWidget(self.btn_blink)
        brack_layout.addLayout(btn_row_files)

        self.table_brack = DropTableWidget(0, 7)
        self.table_brack.setHorizontalHeaderLabels([
            "REF", "Archivo", "Exp", "ISO", "Centro (X,Y)", "ΔX,ΔY (px)", "Usar"
        ])
        self.table_brack.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_brack.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_brack.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_brack.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_brack.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_brack.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_brack.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table_brack.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_brack.setMinimumHeight(160)
        self.table_brack.itemChanged.connect(self._on_table_brack_item_changed)
        self.table_brack.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_brack.cellDoubleClicked.connect(self._on_table_cell_double_clicked)
        self.table_brack.files_dropped.connect(self.add_bracketing_paths)
        brack_layout.addWidget(self.table_brack)

        btn_align_row = QHBoxLayout()
        self.btn_detect_all = QPushButton("Detectar Limbo en Lote")
        self.btn_detect_all.setIcon(get_icon("target"))
        self.btn_detect_all.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_detect_all.setToolTip("Analiza el limbo con escáner radial subpíxel en cada imagen")
        self.btn_detect_all.clicked.connect(self._on_detect_all_clicked)

        self.btn_set_ref = QPushButton("Marcar REF")
        self.btn_set_ref.setIcon(get_icon("pin"))
        self.btn_set_ref.setStyleSheet(STYLE_SECONDARY_BTN)
        self.btn_set_ref.setToolTip("Establece la toma seleccionada como centro de referencia de alineación")
        self.btn_set_ref.clicked.connect(self._on_set_ref_clicked)

        btn_align_row.addWidget(self.btn_detect_all)
        btn_align_row.addWidget(self.btn_set_ref)
        brack_layout.addLayout(btn_align_row)

        self.chk_limb_autostretch = QCheckBox("Auto-estirado vista limbo (solo visual)")
        self.chk_limb_autostretch.setChecked(False)
        self.chk_limb_autostretch.setToolTip(
            "Aplica un estirado temporal en el visor para inspeccionar el limbo en tomas muy oscuras (< 1/250s) sin alterar los datos lineales."
        )
        self.chk_limb_autostretch.toggled.connect(self._on_limb_autostretch_toggled)
        brack_layout.addWidget(self.chk_limb_autostretch)

        left_layout.addWidget(self.sec_brack)

        # SECCIÓN 2: Geometría del Disco (Limbo de la Toma Seleccionada)
        self.sec_geom = CollapsibleSection("2. Geometría del Disco (Limbo)", is_expanded=True, icon_name="crosshair")
        geom_layout = self.sec_geom.content_layout

        btn_auto_row = QHBoxLayout()
        self.btn_detect_disk = QPushButton("Auto-detectar")
        self.btn_detect_disk.setIcon(get_icon("target"))
        self.btn_detect_disk.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_detect_disk.setToolTip("Detección global del disco lunar/solar")
        self.btn_detect_disk.clicked.connect(self._on_auto_detect_clicked)

        self.btn_refine_subpixel = QPushButton("Ajuste Fino Subpíxel")
        self.btn_refine_subpixel.setIcon(get_icon("search"))
        self.btn_refine_subpixel.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_refine_subpixel.setToolTip("Ajusta el círculo exactamente al borde del limbo ignorando protuberancias")
        self.btn_refine_subpixel.clicked.connect(self._on_refine_subpixel_clicked)

        self.btn_pick_center = QPushButton("Clic Canvas")
        self.btn_pick_center.setIcon(get_icon("crosshair"))
        self.btn_pick_center.setCheckable(True)
        self.btn_pick_center.setStyleSheet("""
            QPushButton {
                background-color: #2c2c32;
                color: #e4e4e7;
                border: 1px solid #2e2e35;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #35353d;
                border-color: #3f3f46;
            }
            QPushButton:checked {
                background-color: rgba(47, 129, 247, 0.15);
                border-color: #2f81f7;
                color: #2f81f7;
            }
        """)
        self.btn_pick_center.setToolTip("Haz clic en el canvas para situar el centro")
        self.btn_pick_center.toggled.connect(self._on_pick_center_toggled)

        btn_auto_row.addWidget(self.btn_detect_disk)
        btn_auto_row.addWidget(self.btn_refine_subpixel)
        btn_auto_row.addWidget(self.btn_pick_center)
        geom_layout.addLayout(btn_auto_row)

        row_coords = QHBoxLayout()
        row_coords.addWidget(QLabel("Centro X:"))
        self.spin_cx = QDoubleSpinBox()
        self.spin_cx.setRange(0, 50000)
        self.spin_cx.setDecimals(1)
        self.spin_cx.setSingleStep(0.5)
        self.spin_cx.valueChanged.connect(self._on_geom_spin_changed)
        row_coords.addWidget(self.spin_cx)

        row_coords.addWidget(QLabel("Centro Y:"))
        self.spin_cy = QDoubleSpinBox()
        self.spin_cy.setRange(0, 50000)
        self.spin_cy.setDecimals(1)
        self.spin_cy.setSingleStep(0.5)
        self.spin_cy.valueChanged.connect(self._on_geom_spin_changed)
        row_coords.addWidget(self.spin_cy)
        geom_layout.addLayout(row_coords)

        row_radius = QHBoxLayout()
        row_radius.addWidget(QLabel("Radio (px):"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(1, 50000)
        self.spin_radius.setDecimals(1)
        self.spin_radius.setSingleStep(0.5)
        self.spin_radius.setValue(100.0)
        self.spin_radius.valueChanged.connect(self._on_geom_spin_changed)
        row_radius.addWidget(self.spin_radius)

        self.chk_show_guide = QCheckBox("Mostrar Guía")
        self.chk_show_guide.setChecked(True)
        self.chk_show_guide.toggled.connect(self._update_canvas_guide)
        row_radius.addWidget(self.chk_show_guide)
        geom_layout.addLayout(row_radius)

        # Control de Máscara Lunar Interna (Doble Radio)
        row_mask = QHBoxLayout()
        row_mask.addWidget(QLabel("Offset Máscara:"))
        self.spin_mask_offset = QDoubleSpinBox()
        self.spin_mask_offset.setRange(-100.0, 50.0)
        self.spin_mask_offset.setDecimals(1)
        self.spin_mask_offset.setSingleStep(0.5)
        self.spin_mask_offset.setValue(-2.5)
        self.spin_mask_offset.setSuffix(" px")
        self.spin_mask_offset.setToolTip("Desplazamiento radial de la máscara negra interna respecto al limbo lunar físico (por defecto -2.5 px).")
        self.spin_mask_offset.valueChanged.connect(self._on_mask_offset_changed)
        row_mask.addWidget(self.spin_mask_offset)

        self.lbl_mask_r_disp = QLabel("R_mask: 97.5 px")
        self.lbl_mask_r_disp.setStyleSheet("color: #a1a1aa; font-size: 11px; font-weight: bold;")
        row_mask.addWidget(self.lbl_mask_r_disp)
        geom_layout.addLayout(row_mask)

        left_layout.addWidget(self.sec_geom)

        # SECCIÓN 3: Fusión HDR (Bracketing y Estirado Asinh)
        self.grp_hdr = CollapsibleSection("3. Fusión HDR (Lineal 32-bit)", is_expanded=True, icon_name="zap")
        hdr_layout = self.grp_hdr.content_layout

        self.chk_auto_align = QCheckBox("Alinear tomas por limbo antes de fusionar")
        self.chk_auto_align.setChecked(True)
        hdr_layout.addWidget(self.chk_auto_align)

        self.chk_sub_sky = QCheckBox("Sustracción de pedestal / cielo por toma")
        self.chk_sub_sky.setChecked(True)
        self.chk_sub_sky.setToolTip("Anivela el fondo de cielo y luz difusa en cada fotograma antes de fusionar para erradicar anillos o escalones concéntricos en las transiciones de saturación.")
        hdr_layout.addWidget(self.chk_sub_sky)

        self.chk_twilight_gradient = QCheckBox("Ecualizar gradiente de cielo crepuscular (R >= 6 R)")
        self.chk_twilight_gradient.setChecked(False)
        self.chk_twilight_gradient.setToolTip(
            "Modela y sustrae en espacio lineal la inclinación lumínica del horizonte o crepúsculo "
            "en la corona lejana (R >= 6 R), eliminando fondos descompensados sin tocar la corona interior."
        )
        hdr_layout.addWidget(self.chk_twilight_gradient)

        self.chk_auto_wb = QCheckBox("Balance de blancos coronal neutro (anillo medio)")
        self.chk_auto_wb.setChecked(True)
        self.chk_auto_wb.setToolTip("Neutraliza la dominante cálida/crepuscular en el anillo medio de la corona (1.2 - 1.8 R) según AEL, manteniendo intactas las protuberancias H-alfa.")
        hdr_layout.addWidget(self.chk_auto_wb)

        # Controles térmicos duales (Corona vs Fondo de Cielo)
        self.slider_corona_warmth = PrecisionSliderRow(
            title="Calidez Corona",
            min_val=-100,
            max_val=100,
            default_val=0,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_corona_warmth.setToolTip("Modula la calidez cromática de la corona interior/media (dispersión de Thomson). + = tono dorado/nacarado; - = tono frío.")
        hdr_layout.addWidget(self.slider_corona_warmth)

        self.slider_sky_temp = PrecisionSliderRow(
            title="Tono Cielo Ext.",
            min_val=-100,
            max_val=100,
            default_val=0,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_sky_temp.setToolTip("Modula la temperatura cromática del cielo periférico. - = azul marino frío crepuscular; + = tono cálido.")
        hdr_layout.addWidget(self.slider_sky_temp)

        row_roi = QHBoxLayout()
        self.chk_crop_roi = QCheckBox("Recorte ROI centrado en limbo")
        self.chk_crop_roi.setChecked(False)
        self.chk_crop_roi.setToolTip("Recorta automáticamente un recuadro optimizado alrededor del limbo detectado en cada toma (reduciendo de 24MP a ~9MP). Ahorra 65% de RAM y triplica la velocidad de procesado.")
        row_roi.addWidget(self.chk_crop_roi)

        self.combo_crop_size = QComboBox()
        self.combo_crop_size.addItem("3000 x 3000 px", 3000)
        self.combo_crop_size.addItem("2400 x 2400 px", 2400)
        self.combo_crop_size.addItem("3600 x 3600 px", 3600)
        row_roi.addWidget(self.combo_crop_size)
        hdr_layout.addLayout(row_roi)

        self.slider_sat = PrecisionSliderRow(
            title="Corte Saturación",
            min_val=70,
            max_val=98,
            default_val=92,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.lbl_sat_val = self.slider_sat.spin  # compatibilidad
        hdr_layout.addWidget(self.slider_sat)

        self.slider_asinh = PrecisionSliderRow(
            title="Compresión Asinh",
            min_val=0,
            max_val=100,
            default_val=50,
            unit="x",
            decimals=0,
            label_width=135
        )
        self.lbl_asinh_val = self.slider_asinh.spin  # compatibilidad
        self.slider_asinh.valueChanged.connect(self._on_asinh_slider_changed)
        hdr_layout.addWidget(self.slider_asinh)

        self.slider_highlight_shoulder = PrecisionSliderRow(
            title="Protección Llamaradas",
            min_val=0,
            max_val=100,
            default_val=45,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_highlight_shoulder.setToolTip(
            "Comprime suavemente los picos extremos (> 1.0) de protuberancias y llamaradas sin empastar los filamentos coronales."
        )
        self.slider_highlight_shoulder.valueChanged.connect(self._on_asinh_slider_changed)
        hdr_layout.addWidget(self.slider_highlight_shoulder)

        self.chk_gradual_handover = QCheckBox("Traspaso gradual de exposición (anti-anillos HDR)")
        self.chk_gradual_handover.setChecked(True)
        self.chk_gradual_handover.setToolTip(
            "Atenúa la caída abrupta de varianza distribuyendo el peso con la exposición más corta (AEL 1.4.0), "
            "eliminando discontinuidades y falsos anillos en las isófotas de transición."
        )
        self.chk_gradual_handover.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        hdr_layout.addWidget(self.chk_gradual_handover)

        self.btn_fuse_hdr = QPushButton("FUSIONAR BRACKETING HDR")
        self.btn_fuse_hdr.setIcon(get_icon("zap", color="#ffffff"))
        self.btn_fuse_hdr.setFixedHeight(40)
        self.btn_fuse_hdr.setCursor(Qt.PointingHandCursor)
        self.btn_fuse_hdr.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_fuse_hdr.clicked.connect(self._on_fuse_hdr_clicked)
        hdr_layout.addWidget(self.btn_fuse_hdr)
        left_layout.addWidget(self.grp_hdr)

        # SECCIÓN 4: Filtros de Corona Solar
        self.grp_nrgf = CollapsibleSection("4. Filtros de Corona Solar", is_expanded=True, icon_name="sparkles")
        nrgf_layout = self.grp_nrgf.content_layout

        lbl_algo = QLabel("Algoritmo de Corona Solar:")
        lbl_algo.setStyleSheet("font-weight: bold; color: #e4e4e7; font-size: 11px;")
        nrgf_layout.addWidget(lbl_algo)

        self.combo_filter_algo = QComboBox()
        self.combo_filter_algo.addItems([
            "Steerable G2 (Freeman & Adelson) — Recomendado 2026",
            "MGN Multiescala (Morgan & Druckmüller)",
            "Gradiente Rotacional (Larson-Sekanina)",
            "Filtro SWAP (Misión PROBA2 / Seaton et al.)",
            "Polar Radial NRGF",
            "NAFE Difuso (Miloslav Druckmüller 2013)",
            "WOW Guiado Bilateral (Edge-Aware / Auchère et al.)",
            "FNRGF Azimutal de Fourier (Morgan & Druckmüller)"
        ])
        nrgf_layout.addWidget(self.combo_filter_algo)

        self.chk_detail_rings = QCheckBox("Normalización de detalle por anillos (anti-quemado en limbo)")
        self.chk_detail_rings.setChecked(True)
        self.chk_detail_rings.setToolTip(
            "Normaliza el detalle coronal anillo por anillo con mediana y MAD radiales continuas (AEL 1.3.4 / Issue #9), "
            "evitando la saturación a blanco/negro puro cerca del limbo solar."
        )
        self.chk_detail_rings.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        nrgf_layout.addWidget(self.chk_detail_rings)

        # 1. Contenedor Steerable G2
        self.container_steerable = QWidget()
        g2_box_layout = QVBoxLayout(self.container_steerable)
        g2_box_layout.setContentsMargins(0, 2, 0, 2)
        g2_box_layout.setSpacing(6)

        self.slider_blend_g2 = PrecisionSliderRow(
            title="Fuerza Steerable G2",
            min_val=0,
            max_val=100,
            default_val=50,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_blend_g2.setToolTip("Ponderación de la segunda derivada orientada analítica a lo largo del tensor de estructura coronal.")
        g2_box_layout.addWidget(self.slider_blend_g2)

        self.slider_reach_g2 = PrecisionSliderRow(
            title="Alcance Corona",
            min_val=12,
            max_val=50,
            default_val=35,
            scale=10.0,
            unit="x",
            decimals=1,
            label_width=135
        )
        g2_box_layout.addWidget(self.slider_reach_g2)
        nrgf_layout.addWidget(self.container_steerable)

        # 2. Contenedor MGN
        self.container_mgn = QWidget()
        mgn_box_layout = QVBoxLayout(self.container_mgn)
        mgn_box_layout.setContentsMargins(0, 2, 0, 2)
        mgn_box_layout.setSpacing(6)

        self.slider_blend = PrecisionSliderRow(
            title="Fuerza Detalle",
            min_val=0,
            max_val=100,
            default_val=50,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.lbl_blend_val = self.slider_blend.spin
        mgn_box_layout.addWidget(self.slider_blend)

        self.slider_mgn_k = PrecisionSliderRow(
            title="Contraste Local k",
            min_val=10,
            max_val=150,
            default_val=70,
            scale=100.0,
            unit="",
            decimals=2,
            label_width=135
        )
        mgn_box_layout.addWidget(self.slider_mgn_k)

        self.slider_mgn_gamma = PrecisionSliderRow(
            title="Compresión Gamma",
            min_val=15,
            max_val=40,
            default_val=28,
            scale=10.0,
            unit="",
            decimals=1,
            label_width=135
        )
        mgn_box_layout.addWidget(self.slider_mgn_gamma)

        self.slider_mgn_noise_floor = PrecisionSliderRow(
            title="Piso de Ruido (Fondo)",
            min_val=1,
            max_val=60,
            default_val=3,
            scale=1000.0,
            unit="",
            decimals=3,
            label_width=135
        )
        self.slider_mgn_noise_floor.setToolTip("Piso mínimo de dispersión en MGN. Un valor bajo preserva filamentos lejanos; un valor alto atenúa el grano del sensor.")
        mgn_box_layout.addWidget(self.slider_mgn_noise_floor)

        self.slider_mgn_chroma_denoise = PrecisionSliderRow(
            title="Denoise Croma (Color)",
            min_val=0,
            max_val=40,
            default_val=16,
            unit="px",
            decimals=0,
            label_width=135
        )
        self.slider_mgn_chroma_denoise.setToolTip("Suavizado espacial de crominancia (AEL local_chromaticity). Elimina el grano de ruido coloreado del fondo sin afectar la nitidez de los filamentos.")
        mgn_box_layout.addWidget(self.slider_mgn_chroma_denoise)
        nrgf_layout.addWidget(self.container_mgn)
        self.container_mgn.setVisible(False)

        # 3. Contenedor Larson-Sekanina
        self.container_ls = QWidget()
        ls_box_layout = QVBoxLayout(self.container_ls)
        ls_box_layout.setContentsMargins(0, 2, 0, 2)
        ls_box_layout.setSpacing(6)

        row_ls_dtheta = QHBoxLayout()
        row_ls_dtheta.addWidget(QLabel("Ángulo Δθ:"))
        self.spin_ls_dtheta = QDoubleSpinBox()
        self.spin_ls_dtheta.setRange(0.2, 10.0)
        self.spin_ls_dtheta.setValue(1.0)
        self.spin_ls_dtheta.setSingleStep(0.2)
        self.spin_ls_dtheta.setSuffix("°")
        self.spin_ls_dtheta.setToolTip("Paso angular para el gradiente rotacional con interpolador de Keys (a = -0.5).")
        row_ls_dtheta.addWidget(self.spin_ls_dtheta)
        ls_box_layout.addLayout(row_ls_dtheta)

        self.chk_ls_prominences = QCheckBox("Proteger protuberancias H-alfa (anti-fantasmas)")
        self.chk_ls_prominences.setChecked(True)
        self.chk_ls_prominences.setToolTip("Enmascara el exceso cromático R/G en el limbo para evitar que protuberancias intensas creen réplicas oscuras (AEL 1.4.2).")
        self.chk_ls_prominences.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        ls_box_layout.addWidget(self.chk_ls_prominences)

        self.slider_blend_ls = PrecisionSliderRow(
            title="Fuerza Larson-Sek.",
            min_val=0,
            max_val=100,
            default_val=50,
            unit="%",
            decimals=0,
            label_width=135
        )
        ls_box_layout.addWidget(self.slider_blend_ls)

        self.slider_reach_ls = PrecisionSliderRow(
            title="Alcance Corona",
            min_val=12,
            max_val=50,
            default_val=35,
            scale=10.0,
            unit="x",
            decimals=1,
            label_width=135
        )
        ls_box_layout.addWidget(self.slider_reach_ls)
        nrgf_layout.addWidget(self.container_ls)
        self.container_ls.setVisible(False)

        # 4. Contenedor SWAP
        self.container_swap = QWidget()
        swap_box_layout = QVBoxLayout(self.container_swap)
        swap_box_layout.setContentsMargins(0, 2, 0, 2)
        swap_box_layout.setSpacing(6)

        row_swap_win = QHBoxLayout()
        row_swap_win.addWidget(QLabel("Ventana angular ±:"))
        self.spin_swap_window = QDoubleSpinBox()
        self.spin_swap_window.setRange(3.0, 60.0)
        self.spin_swap_window.setValue(15.0)
        self.spin_swap_window.setSingleStep(1.0)
        self.spin_swap_window.setSuffix("°")
        self.spin_swap_window.setToolTip("Ventana angular para la mediana móvil a lo largo de cada círculo concéntrico polar (Seaton et al. 2023).")
        row_swap_win.addWidget(self.spin_swap_window)
        swap_box_layout.addLayout(row_swap_win)

        row_swap_c0 = QHBoxLayout()
        row_swap_c0.addWidget(QLabel("Compresión radial c0:"))
        self.spin_swap_c0 = QDoubleSpinBox()
        self.spin_swap_c0.setRange(0.1, 1.5)
        self.spin_swap_c0.setValue(0.75)
        self.spin_swap_c0.setSingleStep(0.05)
        self.spin_swap_c0.setToolTip("Factor de potencia radial reducida c0 (estándar PROBA2 = 0.75).")
        row_swap_c0.addWidget(self.spin_swap_c0)
        swap_box_layout.addLayout(row_swap_c0)

        self.slider_blend_swap = PrecisionSliderRow(
            title="Mezcla con Original",
            min_val=0,
            max_val=100,
            default_val=100,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_blend_swap.setToolTip("100% = Representación completa autónoma SWAP (visión satelital); valores inferiores fusionan con la imagen base.")
        swap_box_layout.addWidget(self.slider_blend_swap)
        nrgf_layout.addWidget(self.container_swap)
        self.container_swap.setVisible(False)

        # 5. Contenedor NRGF
        self.container_nrgf = QWidget()
        nrgf_box_layout = QVBoxLayout(self.container_nrgf)
        nrgf_box_layout.setContentsMargins(0, 2, 0, 2)
        nrgf_box_layout.setSpacing(6)

        row_bin = QHBoxLayout()
        row_bin.addWidget(QLabel("Ancho anillo radial:"))
        self.spin_bin_px = QDoubleSpinBox()
        self.spin_bin_px.setRange(0.5, 20.0)
        self.spin_bin_px.setValue(2.0)
        self.spin_bin_px.setSingleStep(0.5)
        self.spin_bin_px.setSuffix(" px")
        row_bin.addWidget(self.spin_bin_px)
        nrgf_box_layout.addLayout(row_bin)

        row_nrgf_mode = QHBoxLayout()
        self.rb_robust = QRadioButton("Mediana (MAD)")
        self.rb_robust.setChecked(True)
        self.rb_mean = QRadioButton("Media Gauss")
        row_nrgf_mode.addWidget(self.rb_robust)
        row_nrgf_mode.addWidget(self.rb_mean)
        nrgf_box_layout.addLayout(row_nrgf_mode)

        self.slider_reach = PrecisionSliderRow(
            title="Alcance Corona",
            min_val=12,
            max_val=50,
            default_val=35,
            scale=10.0,
            unit="x",
            decimals=1,
            label_width=135
        )
        self.lbl_reach_val = self.slider_reach.spin
        nrgf_box_layout.addWidget(self.slider_reach)

        self.slider_blend_nrgf = PrecisionSliderRow(
            title="Fuerza NRGF",
            min_val=0,
            max_val=100,
            default_val=50,
            unit="%",
            decimals=0,
            label_width=135
        )
        nrgf_box_layout.addWidget(self.slider_blend_nrgf)

        self.chk_nrgf_chain = QCheckBox("Encadenamiento angular (anti-halos en bordes)")
        self.chk_nrgf_chain.setChecked(True)
        self.chk_nrgf_chain.setToolTip("Encadena perfiles radiales sobre sectores angulares comunes (AEL 1.4.1), eliminando saltos cuando el sensor o la Luna recortan el campo.")
        nrgf_box_layout.addWidget(self.chk_nrgf_chain)

        nrgf_layout.addWidget(self.container_nrgf)
        self.container_nrgf.setVisible(False)

        # 6. Contenedor NAFE (Druckmüller 2013)
        self.container_nafe = QWidget()
        nafe_box_layout = QVBoxLayout(self.container_nafe)
        nafe_box_layout.setContentsMargins(0, 2, 0, 2)
        nafe_box_layout.setSpacing(6)

        row_nafe_sigma = QHBoxLayout()
        row_nafe_sigma.addWidget(QLabel("Compuerta de Ruido σ:"))
        self.spin_nafe_sigma = QDoubleSpinBox()
        self.spin_nafe_sigma.setRange(0.001, 0.5)
        self.spin_nafe_sigma.setValue(0.035)
        self.spin_nafe_sigma.setSingleStep(0.005)
        self.spin_nafe_sigma.setDecimals(4)
        self.spin_nafe_sigma.setToolTip("Dispersión gaussiana σ sobre el eje de intensidad (12x ruido medido en corona exterior).")
        row_nafe_sigma.addWidget(self.spin_nafe_sigma)
        nafe_box_layout.addLayout(row_nafe_sigma)

        self.chk_nafe_overlay = QCheckBox("Modo de fusión Overlay (capa estándar 09 NAFE)")
        self.chk_nafe_overlay.setChecked(True)
        self.chk_nafe_overlay.setToolTip("Aplica la ecualización difusa E en modo de fusión Overlay fotométrico.")
        nafe_box_layout.addWidget(self.chk_nafe_overlay)

        self.slider_blend_nafe = PrecisionSliderRow(
            title="Opacidad NAFE",
            min_val=0,
            max_val=100,
            default_val=79,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_blend_nafe.setToolTip("Opacidad de capa (estándar AEL = 79% en modo Overlay).")
        nafe_box_layout.addWidget(self.slider_blend_nafe)
        nrgf_layout.addWidget(self.container_nafe)
        self.container_nafe.setVisible(False)

        # 7. Contenedor WOW Bilateral Guiado
        self.container_wow = QWidget()
        wow_box_layout = QVBoxLayout(self.container_wow)
        wow_box_layout.setContentsMargins(0, 2, 0, 2)
        wow_box_layout.setSpacing(6)

        row_wow_scales = QHBoxLayout()
        row_wow_scales.addWidget(QLabel("Escalas à-trous B3:"))
        self.spin_wow_scales = QSpinBox()
        self.spin_wow_scales.setRange(3, 8)
        self.spin_wow_scales.setValue(7)
        self.spin_wow_scales.setToolTip("Número de escalas de descomposición wavelet B3 (estándar = 7).")
        row_wow_scales.addWidget(self.spin_wow_scales)
        wow_box_layout.addLayout(row_wow_scales)

        self.chk_wow_guided = QCheckBox("Ponderación guiada (previene inversión de signo, Issue #4)")
        self.chk_wow_guided.setChecked(True)
        self.chk_wow_guided.setToolTip("Calcula los pesos bilaterales a partir de la guía desacoplada, preservando la polaridad de las ondas finas.")
        wow_box_layout.addWidget(self.chk_wow_guided)

        self.slider_blend_wow = PrecisionSliderRow(
            title="Mezcla WOW",
            min_val=0,
            max_val=100,
            default_val=50,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_blend_wow.setToolTip("Porcentaje de mezcla en modo Overlay sobre la corona base.")
        wow_box_layout.addWidget(self.slider_blend_wow)
        nrgf_layout.addWidget(self.container_wow)
        self.container_wow.setVisible(False)

        # 8. Contenedor FNRGF (Morgan, Habbal & Woo 2006)
        self.container_fnrgf = QWidget()
        fnrgf_box_layout = QVBoxLayout(self.container_fnrgf)
        fnrgf_box_layout.setContentsMargins(0, 2, 0, 2)
        fnrgf_box_layout.setSpacing(6)

        row_fnrgf_order = QHBoxLayout()
        row_fnrgf_order.addWidget(QLabel("Orden armónicos Fourier (N):"))
        self.spin_fnrgf_order = QSpinBox()
        self.spin_fnrgf_order.setRange(1, 8)
        self.spin_fnrgf_order.setValue(4)
        self.spin_fnrgf_order.setToolTip("Número de armónicos azimutales N (default 4: modela componente dipolar, cuadrupolar y sextupolar de streamers).")
        row_fnrgf_order.addWidget(self.spin_fnrgf_order)
        fnrgf_box_layout.addLayout(row_fnrgf_order)

        self.slider_blend_fnrgf = PrecisionSliderRow(
            title="Fuerza FNRGF",
            min_val=0,
            max_val=100,
            default_val=55,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_blend_fnrgf.setToolTip("Mezcla con la imagen base (55% = estándar de microcontraste armónico).")
        fnrgf_box_layout.addWidget(self.slider_blend_fnrgf)

        self.slider_reach_fnrgf = PrecisionSliderRow(
            title="Alcance Corona",
            min_val=12,
            max_val=50,
            default_val=35,
            scale=10.0,
            unit="x",
            decimals=1,
            label_width=135
        )
        fnrgf_box_layout.addWidget(self.slider_reach_fnrgf)
        nrgf_layout.addWidget(self.container_fnrgf)
        self.container_fnrgf.setVisible(False)

        self.combo_filter_algo.currentIndexChanged.connect(self._on_filter_algo_changed)

        self.chk_highpass = QCheckBox("Realce de hilos finos (filtro bilateral tangencial)")
        self.chk_highpass.setChecked(False)
        self.chk_highpass.setToolTip("Realce tangencial complementario (recomendado principalmente para NRGF; en MGN es opcional ya que MGN extrae filamentos nativamente).")
        nrgf_layout.addWidget(self.chk_highpass)

        self.slider_hp = PrecisionSliderRow(
            title="Nitidez Hilos",
            min_val=0,
            max_val=100,
            default_val=50,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.lbl_hp_val = self.slider_hp.spin  # compatibilidad
        nrgf_layout.addWidget(self.slider_hp)

        self.chk_dark_structure = QCheckBox("Realce de estructuras oscuras (cavidades y canales)")
        self.chk_dark_structure.setChecked(False)
        self.chk_dark_structure.setToolTip("Aísla y profundiza cavidades coronales, canales de filamentos y valles de absorción inter-streamer mediante descomposición multiescala B3 y transferencia racional MTF.")
        nrgf_layout.addWidget(self.chk_dark_structure)

        self.slider_dark_amount = PrecisionSliderRow(
            title="Fuerza Cavidades",
            min_val=0,
            max_val=100,
            default_val=40,
            unit="%",
            decimals=0,
            label_width=135
        )
        self.slider_dark_amount.setToolTip("Intensidad de realce y oscurecimiento selectivo de cavidades y valles coronales.")
        nrgf_layout.addWidget(self.slider_dark_amount)

        self.chk_dark_protect_prom = QCheckBox("Proteger protuberancias en cavidades")
        self.chk_dark_protect_prom.setChecked(True)
        self.chk_dark_protect_prom.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        self.chk_dark_protect_prom.setToolTip("Evita que las protuberancias de hidrógeno alfa sean atenuadas por el realce de estructuras oscuras.")
        nrgf_layout.addWidget(self.chk_dark_protect_prom)

        self.chk_proxy_preview = QCheckBox("Previsualización rápida Proxy 50% (4x más veloz)")
        self.chk_proxy_preview.setChecked(True)
        self.chk_proxy_preview.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        self.chk_proxy_preview.setToolTip(
            "Ejecuta los filtros coronales sobre una matriz proxy al 50% para previsualizar y calibrar "
            "deslizadores con máxima fluidez. Pulsa 'RENDER FINAL 1:1' para procesar la resolución nativa completa."
        )
        self.chk_proxy_preview.toggled.connect(self._on_proxy_preview_toggled)
        nrgf_layout.addWidget(self.chk_proxy_preview)

        self.btn_apply_filters = QPushButton("APLICAR FILTROS (VISTA PREVIA 50%)")
        self.btn_apply_filters.setIcon(get_icon("sparkles"))
        self.btn_apply_filters.setFixedHeight(40)
        self.btn_apply_filters.setCursor(Qt.PointingHandCursor)
        self.btn_apply_filters.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_apply_filters.clicked.connect(self._on_apply_filters_btn_clicked)
        nrgf_layout.addWidget(self.btn_apply_filters)

        self.btn_render_full = QPushButton("RENDER FINAL 1:1 (RESOLUCIÓN COMPLETA)")
        self.btn_render_full.setIcon(get_icon("zap"))
        self.btn_render_full.setFixedHeight(34)
        self.btn_render_full.setCursor(Qt.PointingHandCursor)
        self.btn_render_full.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_render_full.setToolTip("Aplica los filtros coronales a resolución completa 1:1 sobre la matriz nativa de 32 bits.")
        self.btn_render_full.clicked.connect(self._on_render_full_clicked)
        nrgf_layout.addWidget(self.btn_render_full)

        lbl_credits = QLabel(
            "✦ Algoritmos de corona inspirados en las investigaciones de M. Druckmüller y "
            "la librería Agentic-Eclipse-Library de Pere Guerra (PeterWar)."
        )
        lbl_credits.setStyleSheet("color: #71717a; font-size: 10px; font-style: italic; padding: 4px 2px;")
        lbl_credits.setWordWrap(True)
        lbl_credits.setToolTip(
            "Multi-scale Gaussian Normalization (MGN), balance de blancos coronal y "
            "denoise cromático basados en Agentic-Eclipse-Library (https://github.com/PeterWar/Agentic-Eclipse-Library)."
        )
        nrgf_layout.addWidget(lbl_credits)

        left_layout.addWidget(self.grp_nrgf)

        # SECCIÓN 5: Exportación y Acabado
        self.sec_export = CollapsibleSection("5. Exportación y Acabado", is_expanded=True, icon_name="save")
        export_layout = self.sec_export.content_layout

        btn_row_exp = QHBoxLayout()
        btn_row_exp.setSpacing(6)

        self.btn_save_image = ExportSplitButton(
            text="Guardar...",
            on_master_export=self._on_save_image_clicked,
            on_web_export=self._on_export_web_clicked,
            style="secondary",
            parent=self
        )
        self.btn_save_image.setup_standard_menu(
            on_format_export=lambda fmt: self._on_save_image_clicked(force_format=fmt),
            psd_label="Guardar Photoshop Multicapa (PSD)..."
        )
        self.btn_save_image.setFixedHeight(36)
        self.btn_save_image.setToolTip(
            "Guarda la imagen procesada en disco.\n"
            "• Clic principal: Formatos maestros (TIFF 16/32-bit, PSD multicapa coronal, PNG, JPEG directo).\n"
            "• Desplegable (▼) o Ctrl+Alt+W: Acceso directo a formatos y Exportar para Web / Redes Sociales (sRGB 8-bit, Lanczos, WebP/JPEG/PNG)."
        )

        # Aliases para compatibilidad interna
        self.btn_save_tiff = self.btn_save_image
        self.btn_save_tiff_16 = self.btn_save_image
        self.btn_save_tiff_32 = self.btn_save_image
        self.btn_export_psd = self.btn_save_image

        # Enlace directo al Revelador (Multicapa Coronal / Base) en línea junto a Guardar
        self.btn_send_coronal_stack_to_dev = QPushButton("Enviar al Revelador")
        self.btn_send_coronal_stack_to_dev.setIcon(get_icon("sliders", size=13, color="#2f81f7"))
        self.btn_send_coronal_stack_to_dev.setFixedHeight(36)
        self.btn_send_coronal_stack_to_dev.setCursor(Qt.PointingHandCursor)
        self.btn_send_coronal_stack_to_dev.setStyleSheet(STYLE_SPECIAL_ACTION_BTN)
        self.btn_send_coronal_stack_to_dev.setToolTip(
            "Transfiere la imagen base HDR junto con los filtros coronales seleccionados "
            "(MGN, WOW, NAFE, NRGF, etc.) como capas ráster independientes con modos de fusión al Revelador "
            "(o la imagen base directamente si no se eligen capas adicionales)."
        )
        self.btn_send_coronal_stack_to_dev.clicked.connect(self._on_send_coronal_stack_clicked)

        # Alias para compatibilidad hacia atrás
        self.btn_send_to_dev = self.btn_send_coronal_stack_to_dev

        btn_row_exp.addWidget(self.btn_save_image, 1)
        btn_row_exp.addWidget(self.btn_send_coronal_stack_to_dev, 1)
        export_layout.addLayout(btn_row_exp)

        left_layout.addWidget(self.sec_export)

        scroll_area.setWidget(panel_content)
        left_panel_layout.addWidget(scroll_area, stretch=1)

        self.splitter = splitter
        self.left_panel = left_panel
        splitter.addWidget(left_panel)

        # -------------------------------------------------------------
        # 2. PANEL DERECHO: Toolbar Unificada + Canvas Interactivo
        # -------------------------------------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.toolbar = ViewerToolbar(
            self,
            show_left_toggle=True,
            show_right_toggle=False,
            show_view_mode=False,
            show_composite_btn=False,
            show_clipping=False,
            show_zoom=True,
            show_magnifier=False,
        )
        self.toolbar.left_panel_toggled.connect(self.toggle_left_panel)

        # Botón Centrar Eclipse (Clic Canvas) en la barra
        self.btn_pick_center_tb = QPushButton("Clic Canvas")
        self.btn_pick_center_tb.setIcon(get_icon("crosshair", color="#ffd54f"))
        self.btn_pick_center_tb.setCheckable(True)
        self.btn_pick_center_tb.setCursor(Qt.PointingHandCursor)
        self.btn_pick_center_tb.setToolTip("Haz clic en el canvas para situar el centro geométrico del eclipse")
        self.btn_pick_center_tb.toggled.connect(self._on_pick_center_tb_toggled)
        self.toolbar.insert_custom_widget_before_stretch(self.btn_pick_center_tb)

        # Botón Guía Disco en la barra
        self.btn_toggle_guide_tb = QPushButton("Guía Disco")
        self.btn_toggle_guide_tb.setIcon(get_icon("target", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_toggle_guide_tb.setCheckable(True)
        self.btn_toggle_guide_tb.setChecked(True)
        self.btn_toggle_guide_tb.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_guide_tb.setToolTip("Muestra u oculta la circunferencia de referencia del limbo lunar/solar")
        self.btn_toggle_guide_tb.toggled.connect(self._on_toggle_guide_tb)
        self.toolbar.insert_custom_widget_before_stretch(self.btn_toggle_guide_tb)

        # Botón Comparar A/B (Bypass Filtros Coronales)
        self.btn_bypass_tb = QPushButton("Comparar A/B")
        self.btn_bypass_tb.setIcon(get_icon("eye", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_bypass_tb.setCheckable(True)
        self.btn_bypass_tb.setChecked(False)
        self.btn_bypass_tb.setEnabled(False)
        self.btn_bypass_tb.setCursor(Qt.PointingHandCursor)
        self.btn_bypass_tb.setToolTip("Alterna instantáneamente entre el HDR Base (A) y la imagen con Filtros Coronales (B).\nAtajo: [Espacio] o [\\]")
        self.btn_bypass_tb.toggled.connect(self._on_bypass_tb_toggled)
        self.toolbar.insert_custom_widget_before_stretch(self.btn_bypass_tb)

        # Botón Cortina Divisoria (Split-View Antes/Después)
        self.btn_split_view = QPushButton("Cortina")
        self.btn_split_view.setIcon(get_icon("split", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_split_view.setCheckable(True)
        self.btn_split_view.setChecked(False)
        self.btn_split_view.setEnabled(False)
        self.btn_split_view.setCursor(Qt.PointingHandCursor)
        self.btn_split_view.setToolTip(
            "Activa la cortina divisoria interactiva para comparar la Base HDR con los Filtros Coronales.\n"
            "Arrastra la línea divisoria con el ratón o haz doble clic para centrarla.\n"
            "Atajo: Y"
        )
        self.btn_split_view.toggled.connect(self._on_split_view_toggled)
        self.toolbar.insert_custom_widget_before_stretch(self.btn_split_view)

        # Botón Orientación Cortina (Vertical / Horizontal)
        self.btn_split_orientation = QPushButton("Vertical")
        self.btn_split_orientation.setIcon(get_icon("sliders", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_split_orientation.setCursor(Qt.PointingHandCursor)
        self.btn_split_orientation.setEnabled(False)
        self.btn_split_orientation.setToolTip(
            "Alterna la orientación de la cortina divisoria:\n"
            "• Vertical: Izquierda ('Base HDR') / Derecha ('Filtros')\n"
            "• Horizontal: Arriba ('Base HDR') / Abajo ('Filtros')\n"
            "Atajo: Shift+Y"
        )
        self.btn_split_orientation.clicked.connect(self._on_toggle_split_orientation)
        self.toolbar.insert_custom_widget_before_stretch(self.btn_split_orientation)

        self.toolbar.add_separator_before_stretch()

        # Badge de modo y estado a la derecha
        self.lbl_mode_badge = QLabel("● Modo: Solar (HDR)")
        self.lbl_mode_badge.setStyleSheet(
            "background-color: #222227; border: 1px solid #2e2e35; border-radius: 4px; padding: 2px 8px; color: #2f81f7; font-weight: bold; font-size: 11px;"
        )
        self.toolbar.add_widget(self.lbl_mode_badge)

        self.lbl_canvas_status = QLabel("Listo.")
        self.lbl_canvas_status.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        self.toolbar.add_widget(self.lbl_canvas_status)

        right_layout.addWidget(self.toolbar)

        self.canvas = EclipseCanvas(self)
        self.canvas.center_clicked.connect(self._on_center_picked)
        self.canvas.files_dropped.connect(self.add_bracketing_paths)
        self.toolbar.connect_canvas(self.canvas)
        right_layout.addWidget(self.canvas, 1)

        self.canvas_footer = CanvasFooterBar(self)
        self.progress_bar = self.canvas_footer.progress_bar
        right_layout.addWidget(self.canvas_footer)

        # Sincronizar botones de barra con los del panel
        self.btn_pick_center.toggled.connect(self._sync_pick_center_to_tb)
        self.chk_show_guide.toggled.connect(self._sync_guide_to_tb)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        splitter.setSizes([520, 1400])

        main_layout.addWidget(splitter)

        # Atajos de teclado locales para el visor
        self.shortcut_f7 = QShortcut(QKeySequence(Qt.Key_F7), self)
        self.shortcut_f7.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_f7.activated.connect(lambda: self.toggle_left_panel())

        self.shortcut_bracket = QShortcut(QKeySequence(Qt.Key_BracketLeft), self)
        self.shortcut_bracket.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_bracket.activated.connect(lambda: self.toggle_left_panel())

        # Atajos de comparación A/B (Bypass rápido)
        self.shortcut_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_space.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_space.activated.connect(self._toggle_bypass_shortcut)

        self.shortcut_backslash = QShortcut(QKeySequence(Qt.Key_Backslash), self)
        self.shortcut_backslash.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_backslash.activated.connect(self._toggle_bypass_shortcut)

        # Atajos de Cortina Split-View
        self.shortcut_split = QShortcut(QKeySequence(Qt.Key_Y), self)
        self.shortcut_split.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_split.activated.connect(self._toggle_split_shortcut)

        self.shortcut_split_orientation = QShortcut(QKeySequence("Shift+Y"), self)
        self.shortcut_split_orientation.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_split_orientation.activated.connect(self._toggle_split_orientation_shortcut)

    def _toggle_bypass_shortcut(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_bypass_tb') and self.btn_bypass_tb.isEnabled():
            self.btn_bypass_tb.setChecked(not self.btn_bypass_tb.isChecked())

    def _toggle_split_shortcut(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_split_view') and self.btn_split_view.isEnabled():
            self.btn_split_view.setChecked(not self.btn_split_view.isChecked())

    def _toggle_split_orientation_shortcut(self) -> None:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_split_orientation') and self.btn_split_orientation.isEnabled():
            self._on_toggle_split_orientation()

    def _on_bypass_tb_toggled(self, checked: bool) -> None:
        """Conmuta instantáneamente entre el HDR Base (A) y la imagen con filtros aplicados (B)."""
        if self.fused_hdr_img is None or self.final_filtered_img is None:
            return

        # Si la cortina interactiva está activa, desactivarla para evitar interferencia visual
        if checked and hasattr(self, 'btn_split_view') and self.btn_split_view.isChecked():
            self.btn_split_view.blockSignals(True)
            self.btn_split_view.setChecked(False)
            self.canvas.set_split_view_enabled(False)
            if hasattr(self, 'btn_split_orientation'):
                self.btn_split_orientation.setEnabled(False)
            self.btn_split_view.blockSignals(False)

        is_proxy = getattr(self, "_current_is_proxy", False)
        base_to_show = self.proxy_hdr_img if (is_proxy and getattr(self, "proxy_hdr_img", None) is not None) else self.fused_hdr_img

        if checked:
            self.canvas.set_image(base_to_show)
            self.btn_bypass_tb.setText("A: Base HDR" + (" [Proxy]" if is_proxy else ""))
            self.lbl_canvas_status.setText(f"Mostrando: Base HDR ({'Proxy 50%' if is_proxy else 'Nativo 1:1'}, Bypass activo)")
        else:
            self.canvas.set_image(self.final_filtered_img)
            self.btn_bypass_tb.setText("B: Filtrada" + (" [Proxy]" if is_proxy else " [1:1]"))
            self.lbl_canvas_status.setText(f"Mostrando: Filtros Coronales ({'Proxy 50%' if is_proxy else 'Nativo 1:1'})")

        self._update_canvas_guide()

    def _on_split_view_toggled(self, checked: bool) -> None:
        """Activa o desactiva la cortina divisoria interactiva Antes / Después."""
        if checked:
            # Desactivar bypass A/B si estaba activo
            if hasattr(self, 'btn_bypass_tb') and self.btn_bypass_tb.isChecked():
                self.btn_bypass_tb.blockSignals(True)
                self.btn_bypass_tb.setChecked(False)
                self.btn_bypass_tb.blockSignals(False)

            if self.final_filtered_img is not None:
                self.canvas.set_image(self.final_filtered_img)

            self._update_before_reference()
            self.canvas.set_split_view_enabled(True)
            self.btn_split_orientation.setEnabled(True)
            ori_txt = "Vertical" if self.canvas.split_helper.orientation == "vertical" else "Horizontal"
            self.lbl_canvas_status.setText(f"Cortina interactiva ({ori_txt}): Base HDR vs Filtros")
            self.log(f"Cortina comparadora activada ({ori_txt}). Arrastra la línea central o doble clic para centrar.")
        else:
            self.canvas.set_split_view_enabled(False)
            self.btn_split_orientation.setEnabled(False)
            self.lbl_canvas_status.setText("Listo.")
            self.canvas.update()
        self._update_canvas_guide()

    def _on_toggle_split_orientation(self) -> None:
        """Alterna entre orientación vertical y horizontal de la cortina."""
        new_ori = self.canvas.toggle_split_orientation()
        is_vert = (new_ori == "vertical")
        self.btn_split_orientation.setText("Vertical" if is_vert else "Horizontal")
        ori_txt = "Vertical (Izq/Der)" if is_vert else "Horizontal (Arr/Abj)"
        self.lbl_canvas_status.setText(f"Cortina interactiva: {ori_txt}")
        self.log(f"Orientación de cortina cambiada a: {ori_txt}")

    def _update_before_reference(self) -> None:
        """Sincroniza la imagen 'Antes' (Base HDR) en la escala correspondiente (Proxy 50% o Nativo 1:1)."""
        if self.fused_hdr_img is None:
            return
        is_proxy = getattr(self, "_current_is_proxy", False)
        base_ref = self.proxy_hdr_img if (is_proxy and getattr(self, "proxy_hdr_img", None) is not None) else self.fused_hdr_img
        self.canvas.set_before_image(base_ref)

    def _on_pick_center_tb_toggled(self, checked: bool) -> None:
        if hasattr(self, 'btn_pick_center') and self.btn_pick_center.isChecked() != checked:
            self.btn_pick_center.setChecked(checked)

    def _sync_pick_center_to_tb(self, checked: bool) -> None:
        if hasattr(self, 'btn_pick_center_tb') and self.btn_pick_center_tb.isChecked() != checked:
            self.btn_pick_center_tb.blockSignals(True)
            self.btn_pick_center_tb.setChecked(checked)
            self.btn_pick_center_tb.blockSignals(False)

    def _on_toggle_guide_tb(self, checked: bool) -> None:
        if hasattr(self, 'chk_show_guide') and self.chk_show_guide.isChecked() != checked:
            self.chk_show_guide.setChecked(checked)

    def _sync_guide_to_tb(self, checked: bool) -> None:
        if hasattr(self, 'btn_toggle_guide_tb') and self.btn_toggle_guide_tb.isChecked() != checked:
            self.btn_toggle_guide_tb.blockSignals(True)
            self.btn_toggle_guide_tb.setChecked(checked)
            self.btn_toggle_guide_tb.blockSignals(False)

    def toggle_left_panel(self, visible: Optional[bool] = None) -> None:
        """Muestra u oculta el panel lateral izquierdo de controles de eclipses (Atajo: F7 o [)."""
        if not hasattr(self, 'left_panel'):
            return
        if visible is None:
            visible = not self.left_panel.isVisible()
        if not visible and self.left_panel.isVisible() and hasattr(self, 'splitter'):
            sizes = self.splitter.sizes()
            if len(sizes) >= 2 and sizes[0] > 50:
                self._last_left_panel_width = sizes[0]
        self.left_panel.setVisible(visible)
        if hasattr(self, 'toolbar'):
            self.toolbar.set_left_panel_checked(visible)
        if visible and hasattr(self, 'splitter'):
            sizes = self.splitter.sizes()
            if len(sizes) >= 2 and sizes[0] < 50:
                sizes[0] = getattr(self, '_last_left_panel_width', 520)
                self.splitter.setSizes(sizes)

    def keyPressEvent(self, event):
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key_Space, Qt.Key_Backslash):
            if hasattr(self, 'btn_bypass_tb') and self.btn_bypass_tb.isEnabled():
                self.btn_bypass_tb.setChecked(not self.btn_bypass_tb.isChecked())
                event.accept()
                return
        if event.key() in (Qt.Key_F7, Qt.Key_BracketLeft):
            self.toggle_left_panel()
            event.accept()
            return
        super().keyPressEvent(event)
        self.log("Módulo de Eclipses inicializado con escáner subpíxel y compuerta radial anti-ruido.")

    def _apply_styles(self) -> None:
        """Los estilos base provienen de gui.theme (Cosmic Dark Pro)."""
        pass

    def log(self, message: str):
        log_hub.log("Eclipses", message)
        self.lbl_canvas_status.setText(message)

    # -------------------------------------------------------------
    # Métodos Auxiliares Seguros
    # -------------------------------------------------------------
    def _set_geom_spin_values(self, cx: float, cy: float, r: float):
        """Asigna valores a los spinboxes bloqueando señales para no disparar eventos en bucle."""
        self.spin_cx.blockSignals(True)
        self.spin_cy.blockSignals(True)
        self.spin_radius.blockSignals(True)

        self.spin_cx.setValue(cx)
        self.spin_cy.setValue(cy)
        self.spin_radius.setValue(r)

        self.spin_cx.blockSignals(False)
        self.spin_cy.blockSignals(False)
        self.spin_radius.blockSignals(False)

        self._update_canvas_guide()

    def _update_canvas_guide(self):
        cx = self.spin_cx.value()
        cy = self.spin_cy.value()
        r = self.spin_radius.value()
        offset = self.spin_mask_offset.value() if hasattr(self, 'spin_mask_offset') else -2.5
        show = self.chk_show_guide.isChecked()

        # Si el lienzo tiene asignada una imagen proxy al 50%, escalamos la guía visual
        if getattr(self.canvas, "orig_f32", None) is not None and self.fused_hdr_img is not None:
            cw = self.canvas.orig_f32.shape[1]
            fw = self.fused_hdr_img.shape[1]
            if cw > 0 and fw > 0 and abs(cw - fw) > 5:
                scale = cw / fw
                cx *= scale
                cy *= scale
                r *= scale
                offset *= scale

        self.canvas.set_guide_params(cx, cy, r, show=show, mask_offset=offset)
        if hasattr(self, 'lbl_mask_r_disp'):
            self.lbl_mask_r_disp.setText(f"R_mask: {max(0.0, self.spin_radius.value() + (self.spin_mask_offset.value() if hasattr(self, 'spin_mask_offset') else -2.5)):.1f} px")

    def _on_mask_offset_changed(self):
        self._update_canvas_guide()

    def _get_active_geometry_for_image(self, base_img: np.ndarray) -> Tuple[float, float, float]:
        """Devuelve (cx, cy, r) de la imagen objetivo (fusión HDR recortada o toma individual)."""
        is_fused = (self.fused_hdr_img is not None and 
                    (base_img is self.fused_hdr_img or 
                     base_img is getattr(self, "proxy_hdr_img", None) or
                     (self.final_filtered_img is not None and base_img is self.final_filtered_img)))
        if is_fused:
            if getattr(self, "fused_hdr_geom", None) is not None:
                return float(self.fused_hdr_geom[0]), float(self.fused_hdr_geom[1]), float(self.fused_hdr_geom[2])
            return float(self.spin_cx.value()), float(self.spin_cy.value()), float(self.spin_radius.value())

        ref_f = self.bracketing_files[self.ref_file_idx] if (self.bracketing_files and 0 <= self.ref_file_idx < len(self.bracketing_files)) else None
        if ref_f and ref_f.get("cx") is not None:
            return float(ref_f["cx"]), float(ref_f["cy"]), float(ref_f["radius"])
        return float(self.spin_cx.value()), float(self.spin_cy.value()), float(self.spin_radius.value())

    # -------------------------------------------------------------
    # Lógica de Bracketing y Limbo Individual
    # -------------------------------------------------------------
    def _on_mode_changed(self, mode_str: str = None):
        is_solar = self.mode_toggle.is_solar()
        self.grp_nrgf.setVisible(is_solar)
        mode_label = "Solar" if is_solar else "Lunar"
        self.log(f"Modo cambiado a: Eclipse {mode_label}")

        if is_solar:
            self.grp_hdr.set_title("3. Fusión HDR (Lineal 32-bit)")
            self.slider_asinh.setValue(35)
            if hasattr(self, "lbl_mode_badge"):
                self.lbl_mode_badge.setText("● Modo: Solar (HDR)")
                self.lbl_mode_badge.setStyleSheet(
                    "background-color: #222227; border: 1px solid #2e2e35; border-radius: 4px; padding: 2px 8px; color: #2f81f7; font-weight: bold; font-size: 11px;"
                )
        else:
            self.grp_hdr.set_title("3. Fusión HDR (Mertens Multiescala)")
            self.slider_asinh.setValue(0)
            if hasattr(self, "lbl_mode_badge"):
                self.lbl_mode_badge.setText("● Modo: Lunar (Mertens)")
                self.lbl_mode_badge.setStyleSheet(
                    "background-color: #222227; border: 1px solid #2e2e35; border-radius: 4px; padding: 2px 8px; color: #2f81f7; font-weight: bold; font-size: 11px;"
                )

    def add_bracketing_paths(self, paths: list):
        """Añade tomas a la serie de bracketing (Drag & Drop o diálogo) y detecta limbo de referencia."""
        if not paths:
            return

        existing_paths = {f["filepath"] for f in self.bracketing_files}
        new_paths = [p for p in paths if p not in existing_paths]
        if not new_paths:
            return

        total_new = len(new_paths)
        self.cached_aligned_data = None
        self._current_hdr_cache_key = None
        self.log(f"Cargando {total_new} ficheros de eclipse...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total_new)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Cargando ficheros... (%v/{total_new})")
        QApplication.processEvents()

        for idx, p in enumerate(new_paths):
            info = extract_exposure_info(p)
            info["cx"] = None
            info["cy"] = None
            info["radius"] = None
            info["shift_x"] = 0.0
            info["shift_y"] = 0.0
            info["is_ref"] = False
            self.bracketing_files.append(info)
            if (idx + 1) % 15 == 0 or idx == total_new - 1:
                self.progress_bar.setValue(idx + 1)
                if hasattr(self, 'canvas_footer'):
                    self.canvas_footer.set_progress(int((idx + 1) / total_new * 100), f"Cargando {idx + 1}/{total_new}")
                QApplication.processEvents()

        self.progress_bar.setFormat("Detectando limbo en toma de referencia...")
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.set_progress(95, "Detectando limbo en referencia...")
        QApplication.processEvents()

        self.bracketing_files.sort(key=lambda x: x["exposure_s"])

        if self.bracketing_files:
            self.ref_file_idx = len(self.bracketing_files) // 2
            for i, f in enumerate(self.bracketing_files):
                f["is_ref"] = (i == self.ref_file_idx)

        # Auto-detectar de inmediato con escáner subpíxel en la toma de referencia
        ref_f = self.bracketing_files[self.ref_file_idx]
        try:
            ref_img = load_image_as_float32(ref_f["filepath"])
            mode = "solar" if self.rb_solar.isChecked() else "lunar"
            cx, cy, r = detect_eclipse_disk(ref_img, mode=mode)
            ref_f["cx"] = cx
            ref_f["cy"] = cy
            ref_f["radius"] = r
            self.current_preview_img = ref_img
            self.canvas.set_image(ref_img)
            self._set_geom_spin_values(cx, cy, r)
            self.log(f"Limbo inicial detectado en referencia: ({cx:.1f}, {cy:.1f}), R={r:.1f}px")
        except Exception as e:
            self.log(f"Aviso en detección inicial: {e}")

        self._refresh_table()
        self.table_brack.selectRow(self.ref_file_idx)
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.hide_progress()
        self.progress_bar.setVisible(False)
        self.log(f"Carga finalizada: se cargaron {total_new} tomas de bracketing (Total: {len(self.bracketing_files)}).")

    def _on_load_bracketing_clicked(self):
        filters = "Imágenes Astrofotográficas (*.arw *.cr2 *.cr3 *.nef *.dng *.raw *.tif *.tiff *.fits *.fit *.jpg *.jpeg);;Todos (*.*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Cargar Serie de Bracketing de Eclipse", "", filters)
        if not paths:
            return
        self.add_bracketing_paths(paths)

    def _on_clear_bracketing_clicked(self):
        self.bracketing_files.clear()
        self.ref_file_idx = 0
        self.cached_aligned_data = None
        self._current_hdr_cache_key = None
        self._refresh_table()
        self.current_preview_img = None
        self.raw_fused_hdr = None
        self.fused_hdr_img = None
        self.proxy_hdr_img = None
        self._current_is_proxy = False
        self.fused_hdr_geom = None
        self._viewing_fused = False
        self.final_filtered_img = None
        if hasattr(self, 'btn_bypass_tb'):
            self.btn_bypass_tb.setEnabled(False)
            self.btn_bypass_tb.blockSignals(True)
            self.btn_bypass_tb.setChecked(False)
            self.btn_bypass_tb.setText("Comparar A/B")
            self.btn_bypass_tb.blockSignals(False)
        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.setEnabled(False)
            self.btn_split_view.blockSignals(True)
            self.btn_split_view.setChecked(False)
            self.btn_split_view.blockSignals(False)
            self.canvas.set_split_view_enabled(False)
        if hasattr(self, 'btn_split_orientation'):
            self.btn_split_orientation.setEnabled(False)
        self.canvas.set_before_image(None)
        self.canvas.set_image(np.zeros((100, 100, 3), dtype=np.float32))
        self.log("Serie de bracketing vaciada.")

    def _refresh_table(self):
        sel_rows = self.table_brack.selectionModel().selectedRows()
        prev_idx = sel_rows[0].row() if sel_rows else self.ref_file_idx

        self.table_brack.blockSignals(True)
        self.table_brack.setRowCount(len(self.bracketing_files))

        for row, item in enumerate(self.bracketing_files):
            ref_txt = "● REF" if item.get("is_ref", False) else ""
            item_ref = QTableWidgetItem(ref_txt)
            item_ref.setTextAlignment(Qt.AlignCenter)
            if item.get("is_ref", False):
                item_ref.setForeground(QColor("#2f81f7"))
                font = item_ref.font()
                font.setBold(True)
                item_ref.setFont(font)
            self.table_brack.setItem(row, 0, item_ref)

            self.table_brack.setItem(row, 1, QTableWidgetItem(item["filename"]))
            self.table_brack.setItem(row, 2, QTableWidgetItem(item["shutter_str"]))
            self.table_brack.setItem(row, 3, QTableWidgetItem(str(item["iso"])))

            if item.get("cx") is not None:
                center_str = f"{item['cx']:.1f}, {item['cy']:.1f}"
            else:
                center_str = "—"
            self.table_brack.setItem(row, 4, QTableWidgetItem(center_str))

            if item.get("cx") is not None and not item.get("is_ref", False):
                shift_str = f"{item['shift_x']:+.1f}, {item['shift_y']:+.1f}"
            elif item.get("is_ref", False):
                shift_str = "0.0, 0.0"
            else:
                shift_str = "—"
            item_shift = QTableWidgetItem(shift_str)
            item_shift.setTextAlignment(Qt.AlignCenter)
            self.table_brack.setItem(row, 5, item_shift)

            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if item.get("is_included", True) else Qt.Unchecked)
            self.table_brack.setItem(row, 6, chk_item)

        self.table_brack.blockSignals(False)

        if 0 <= prev_idx < len(self.bracketing_files):
            self.table_brack.blockSignals(True)
            self.table_brack.selectRow(prev_idx)
            self.table_brack.blockSignals(False)

    def _on_table_brack_item_changed(self, item: QTableWidgetItem):
        if item.column() == 6:
            row = item.row()
            if 0 <= row < len(self.bracketing_files):
                self.bracketing_files[row]["is_included"] = (item.checkState() == Qt.Checked)

    def open_blink_viewer(self):
        """Abre el visor interactivo Blink para revisar tomas del bracketing de eclipse."""
        if not self.bracketing_files:
            QMessageBox.information(
                self,
                "Visor Blink",
                "No hay tomas cargadas para inspeccionar.\n\n"
                "Carga primero tomas de bracketing antes de abrir el visor Blink."
            )
            return

        from gui.dialog_blink import BlinkDialog

        file_list = [f["filepath"] for f in self.bracketing_files]
        excluded = {f["filepath"] for f in self.bracketing_files if not f.get("is_included", True)}

        curr_idx = 0
        rows = self.table_brack.selectionModel().selectedRows()
        if rows:
            curr_idx = rows[0].row()
        elif 0 <= self.ref_file_idx < len(self.bracketing_files):
            curr_idx = self.ref_file_idx

        limb_guides = {}
        if self.current_preview_img is not None:
            full_h, full_w = self.current_preview_img.shape[:2]
            if full_w > 0 and full_h > 0:
                for f in self.bracketing_files:
                    if f.get("cx") is not None and f.get("radius", 0) > 5:
                        limb_guides[f["filepath"]] = (
                            float(f["cx"]) / full_w,
                            float(f["cy"]) / full_h,
                            float(f["radius"]) / full_w,
                            float(f["radius"]) / full_h,
                        )

        dlg = BlinkDialog(
            file_list=file_list,
            excluded_files=excluded,
            current_index=curr_idx,
            limb_guides=limb_guides or None,
            parent=self
        )
        dlg.exclusion_changed.connect(self._on_blink_exclusion_changed)
        dlg.selection_applied.connect(self._on_blink_selection_applied)
        dlg.exec()

    def _on_blink_exclusion_changed(self, filepath: str, is_included: bool):
        for row, f in enumerate(self.bracketing_files):
            if f["filepath"] == filepath:
                f["is_included"] = is_included
                item = self.table_brack.item(row, 6)
                if item:
                    self.table_brack.blockSignals(True)
                    item.setCheckState(Qt.Checked if is_included else Qt.Unchecked)
                    self.table_brack.blockSignals(False)
                break

    def _on_blink_selection_applied(self, excluded_set: set):
        self.table_brack.blockSignals(True)
        for row, f in enumerate(self.bracketing_files):
            f["is_included"] = (f["filepath"] not in excluded_set)
            item = self.table_brack.item(row, 6)
            if item:
                item.setCheckState(Qt.Checked if f["is_included"] else Qt.Unchecked)
        self.table_brack.blockSignals(False)

    def _recalculate_shifts(self):
        """Calcula los desplazamientos de cada toma respecto al centro de la referencia."""
        if not self.bracketing_files or self.ref_file_idx >= len(self.bracketing_files):
            return

        ref_item = self.bracketing_files[self.ref_file_idx]
        ref_cx = ref_item.get("cx")
        ref_cy = ref_item.get("cy")

        for i, f in enumerate(self.bracketing_files):
            if f.get("cx") is not None and ref_cx is not None:
                f["shift_x"] = float(ref_cx - f["cx"])
                f["shift_y"] = float(ref_cy - f["cy"])
            else:
                f["shift_x"] = 0.0
                f["shift_y"] = 0.0

    def _on_table_cell_double_clicked(self, row: int, col: int):
        self._set_reference_index(row)

    def _on_set_ref_clicked(self):
        rows = self.table_brack.selectionModel().selectedRows()
        if rows:
            self._set_reference_index(rows[0].row())

    def _set_reference_index(self, idx: int):
        if 0 <= idx < len(self.bracketing_files):
            self.ref_file_idx = idx
            for i, f in enumerate(self.bracketing_files):
                f["is_ref"] = (i == idx)
            self.cached_aligned_data = None
            self._current_hdr_cache_key = None
            self._recalculate_shifts()
            self._refresh_table()
            self.table_brack.selectRow(idx)
            self.log(f"Toma de referencia establecida: {self.bracketing_files[idx]['filename']}")

    def _display_single_preview(self, img: Optional[np.ndarray]):
        if img is None:
            return
        if hasattr(self, "chk_limb_autostretch") and self.chk_limb_autostretch.isChecked():
            p_lo, p_hi = np.percentile(img, [75.0, 99.8])
            norm = np.clip((img - float(p_lo)) / max(float(p_hi - p_lo), 1e-6), 0.0, 1.0).astype(np.float32)
            vis = apply_asinh_stretch(norm, stretch_factor=60.0, shoulder_threshold=0.80)
            self.canvas.set_image(vis)
        else:
            self.canvas.set_image(img)

    def _on_limb_autostretch_toggled(self, _checked: bool):
        if not getattr(self, "_viewing_fused", False) and self.current_preview_img is not None:
            self._display_single_preview(self.current_preview_img)

    def _on_table_selection_changed(self):
        rows = self.table_brack.selectionModel().selectedRows()
        if not rows:
            return
        self._viewing_fused = False
        idx = rows[0].row()
        if 0 <= idx < len(self.bracketing_files):
            item = self.bracketing_files[idx]
            filepath = item["filepath"]
            try:
                img = load_image_as_float32(filepath)
                self.current_preview_img = img
                self._display_single_preview(img)

                if item.get("cx") is not None:
                    cx, cy, r = item["cx"], item["cy"], item["radius"]
                    self._set_geom_spin_values(cx, cy, r)
                else:
                    ref_f = self.bracketing_files[self.ref_file_idx]
                    if ref_f.get("cx") is not None:
                        cx, cy, r = ref_f["cx"], ref_f["cy"], ref_f["radius"]
                        self._set_geom_spin_values(cx, cy, r)

                self.log(f"Previsualizando ({idx+1}/{len(self.bracketing_files)}): {item['filename']} [{item['shutter_str']}]")

            except Exception as e:
                self.log(f"Error cargando previsualización: {e}")

    def _on_auto_detect_clicked(self):
        """Auto-detecta el limbo únicamente en la toma activa seleccionada."""
        rows = self.table_brack.selectionModel().selectedRows()
        if not rows or self.current_preview_img is None:
            QMessageBox.warning(self, "Atención", "Selecciona primero una toma para detectar su limbo.")
            return

        idx = rows[0].row()
        mode = "solar" if self.rb_solar.isChecked() else "lunar"
        self.log(f"Detectando limbo en toma activa ({mode})...")

        # Si la toma activa es la de referencia, calculamos el radio de forma independiente.
        # En tomas secundarias, aprovechamos el radio validado de la referencia si existe (> 10 px).
        if idx == self.ref_file_idx:
            expected_r = None
        else:
            ref_r = self.bracketing_files[self.ref_file_idx].get("radius")
            expected_r = ref_r if (ref_r is not None and ref_r > 10) else None

        cx, cy, r = detect_eclipse_disk(self.current_preview_img, mode=mode, expected_radius=expected_r)

        self.bracketing_files[idx]["cx"] = cx
        self.bracketing_files[idx]["cy"] = cy
        self.bracketing_files[idx]["radius"] = r
        self.cached_aligned_data = None
        self._current_hdr_cache_key = None

        self._set_geom_spin_values(cx, cy, r)
        self._recalculate_shifts()
        self._refresh_table()
        self.log(f"Limbo detectado para {self.bracketing_files[idx]['filename']}: ({cx:.1f}, {cy:.1f}), R={r:.1f}px")

    def _on_refine_subpixel_clicked(self):
        """Ajuste fino de precisión milimétrica usando escaneo radial en el centro actual."""
        rows = self.table_brack.selectionModel().selectedRows()
        if not rows or self.current_preview_img is None:
            QMessageBox.warning(self, "Atención", "Selecciona primero una toma para refinar su limbo.")
            return

        idx = rows[0].row()
        mode = "solar" if self.rb_solar.isChecked() else "lunar"
        cx = self.spin_cx.value()
        cy = self.spin_cy.value()
        r = self.spin_radius.value()

        self.log("Refinando centro y radio del limbo con escáner radial subpíxel...")
        fcx, fcy, fr = refine_disk_subpixel(self.current_preview_img, cx, cy, r, mode=mode)

        self.bracketing_files[idx]["cx"] = fcx
        self.bracketing_files[idx]["cy"] = fcy
        self.bracketing_files[idx]["radius"] = fr
        self.cached_aligned_data = None
        self._current_hdr_cache_key = None

        self._set_geom_spin_values(fcx, fcy, fr)
        self._recalculate_shifts()
        self._refresh_table()
        self.log(f"Limbo refinado a subpíxel: ({fcx:.1f}, {fcy:.1f}), R={fr:.1f}px")

    def _on_detect_all_clicked(self):
        """Lanza la detección automática del limbo en todas las tomas del bracketing."""
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.bracketing_files:
            QMessageBox.warning(self, "Atención", "Carga tomas de bracketing antes de detectar el limbo.")
            return

        mode = "solar" if self.rb_solar.isChecked() else "lunar"
        files = [f["filepath"] for f in self.bracketing_files]

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_detect_all.setEnabled(False)
        self.btn_fuse_hdr.setEnabled(False)
        self.btn_apply_filters.setEnabled(False)

        ref_r = self.bracketing_files[self.ref_file_idx].get("radius") if self.bracketing_files else None
        params = {
            "files": files,
            "mode": mode,
            "ref_idx": self.ref_file_idx,
            "ref_r": ref_r
        }
        self.worker = EclipseWorker("batch_detect", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_pick_center_toggled(self, checked: bool):
        self.canvas.pick_center_mode = checked
        if checked:
            self.canvas.setCursor(Qt.CrossCursor)
            self.log("Haz clic en el canvas para posicionar el centro del eclipse en esta toma.")
        else:
            self.canvas.setCursor(Qt.ArrowCursor)

    def _on_center_picked(self, cx: float, cy: float):
        if getattr(self, "_viewing_fused", False):
            r = self.spin_radius.value()
            self.fused_hdr_geom = (float(cx), float(cy), float(r))
            self._set_geom_spin_values(cx, cy, r)
            self.btn_pick_center.setChecked(False)
            self.log(f"Centro de imagen fusionada HDR establecido por clic en: ({cx:.1f}, {cy:.1f})")
            return

        rows = self.table_brack.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            r = self.spin_radius.value()
            self.bracketing_files[idx]["cx"] = cx
            self.bracketing_files[idx]["cy"] = cy
            self.cached_aligned_data = None
            self._current_hdr_cache_key = None
            self._set_geom_spin_values(cx, cy, r)
            self._recalculate_shifts()
            self._refresh_table()

        self.btn_pick_center.setChecked(False)
        self.log(f"Centro establecido por clic en: ({cx:.1f}, {cy:.1f})")

    def _on_geom_spin_changed(self):
        self.cached_aligned_data = None
        self._current_hdr_cache_key = None

        if getattr(self, "_viewing_fused", False):
            self.fused_hdr_geom = (float(self.spin_cx.value()), float(self.spin_cy.value()), float(self.spin_radius.value()))
        else:
            rows = self.table_brack.selectionModel().selectedRows()
            if rows:
                idx = rows[0].row()
                self.bracketing_files[idx]["cx"] = self.spin_cx.value()
                self.bracketing_files[idx]["cy"] = self.spin_cy.value()
                self.bracketing_files[idx]["radius"] = self.spin_radius.value()
                self._recalculate_shifts()

                cx_str = f"{self.spin_cx.value():.1f}, {self.spin_cy.value():.1f}"
                if self.table_brack.item(idx, 4):
                    self.table_brack.item(idx, 4).setText(cx_str)

                for r in range(self.table_brack.rowCount()):
                    f = self.bracketing_files[r]
                    shift_str = "0.0, 0.0" if f.get("is_ref") else f"{f['shift_x']:+.1f}, {f['shift_y']:+.1f}"
                    if self.table_brack.item(r, 5):
                        self.table_brack.item(r, 5).setText(shift_str)

        self._update_canvas_guide()

    def _get_prom_protect(self) -> float:
        if hasattr(self, "slider_highlight_shoulder"):
            return float(self.slider_highlight_shoulder.value()) / 100.0
        return 0.45

    def _on_asinh_slider_changed(self, _val: int = 0):
        factor = float(self.slider_asinh.value())
        self.lbl_asinh_val.setText(f"{int(factor)}x" if factor > 0 else "Lineal (1x)")

        if self.raw_fused_hdr is not None:
            self._viewing_fused = True
            prom_protect = self._get_prom_protect()
            white_pt = 1.0 + prom_protect * 5.0
            color_pres = 0.90 if prom_protect > 0.001 else 0.0
            self.fused_hdr_img = apply_asinh_stretch(
                self.raw_fused_hdr,
                stretch_factor=factor,
                shoulder_threshold=0.78,
                white_point=white_pt,
                color_preserve=color_pres
            )
            if self.fused_hdr_img.shape[0] >= 500 and self.fused_hdr_img.shape[1] >= 500:
                self.proxy_hdr_img = cv2.resize(self.fused_hdr_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            else:
                self.proxy_hdr_img = None
            self._current_is_proxy = False
            self.canvas.set_image(self.fused_hdr_img)
            if getattr(self, "fused_hdr_geom", None) is not None:
                self._set_geom_spin_values(*self.fused_hdr_geom)
            else:
                self._update_canvas_guide()

    def _on_fuse_hdr_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            return
        active_files = []
        active_exposures = []
        active_centers = []
        active_ref_idx = 0

        for row in range(self.table_brack.rowCount()):
            chk = self.table_brack.item(row, 6)
            if chk and chk.checkState() == Qt.Checked:
                f_data = self.bracketing_files[row]
                if f_data.get("is_ref", False):
                    active_ref_idx = len(active_files)
                active_files.append(f_data["filepath"])
                active_exposures.append(f_data["exposure_s"])
                if f_data.get("cx") is not None:
                    active_centers.append((f_data["cx"], f_data["cy"], f_data["radius"]))
                else:
                    active_centers.append(None)

        if len(active_files) < 2:
            QMessageBox.warning(self, "Atención", "Selecciona al menos 2 tomas activas para la fusión HDR.")
            return

        sat_thresh = self.slider_sat.value() / 100.0
        asinh_factor = float(self.slider_asinh.value())
        auto_align = self.chk_auto_align.isChecked()
        sub_sky = self.chk_sub_sky.isChecked()
        crop_roi = self.chk_crop_roi.isChecked()
        crop_size = int(self.combo_crop_size.currentData() or 3000)
        mode = "solar" if self.rb_solar.isChecked() else "lunar"

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_fuse_hdr.setEnabled(False)
        self.btn_apply_filters.setEnabled(False)
        self.btn_detect_all.setEnabled(False)

        cache_centers_key = tuple(
            (round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 2))
            if c is not None and len(c) >= 3 else None
            for c in active_centers
        )
        current_cache_key = (
            tuple(active_files),
            cache_centers_key,
            active_ref_idx,
            auto_align,
            crop_roi,
            crop_size
        )
        self._current_hdr_cache_key = current_cache_key

        params = {
            "files": active_files,
            "exposures": active_exposures,
            "centers": active_centers,
            "ref_idx": active_ref_idx,
            "auto_align": auto_align,
            "crop_roi": crop_roi,
            "crop_size": crop_size,
            "subtract_sky": sub_sky,
            "equalize_sky_gradient": self.chk_twilight_gradient.isChecked() if hasattr(self, 'chk_twilight_gradient') else False,
            "auto_wb_corona": self.chk_auto_wb.isChecked(),
            "gradual_handover": self.chk_gradual_handover.isChecked(),
            "corona_warmth": float(self.slider_corona_warmth.value() / 100.0) if hasattr(self, 'slider_corona_warmth') else 0.0,
            "sky_temp": float(self.slider_sky_temp.value() / 100.0) if hasattr(self, 'slider_sky_temp') else 0.0,
            "sat_thresh": sat_thresh,
            "asinh_factor": asinh_factor,
            "prom_protect": self._get_prom_protect(),
            "mode": mode
        }

        if (self.cached_aligned_data is not None and 
            self.cached_aligned_data.get("key") == current_cache_key):
            params["cached_aligned_images"] = self.cached_aligned_data["aligned_images"]
            params["cached_final_centers"] = self.cached_aligned_data["final_centers"]
            params["cached_raw_centers"] = self.cached_aligned_data.get("raw_centers", self.cached_aligned_data["final_centers"])
            params["cached_shifts"] = self.cached_aligned_data["shifts"]
        elif self.cached_aligned_data is not None:
            self.cached_aligned_data = None
            import gc
            gc.collect()

        self.worker = EclipseWorker("hdr_fusion", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_filter_algo_changed(self):
        idx = self.combo_filter_algo.currentIndex()
        self.container_steerable.setVisible(idx == 0)
        self.container_mgn.setVisible(idx == 1)
        self.container_ls.setVisible(idx == 2)
        self.container_swap.setVisible(idx == 3)
        self.container_nrgf.setVisible(idx == 4)
        self.container_nafe.setVisible(idx == 5)
        self.container_wow.setVisible(idx == 6)
        self.container_fnrgf.setVisible(idx == 7)
        self.chk_detail_rings.setVisible(idx in (0, 2, 4, 6, 7))

    def _on_proxy_preview_toggled(self, checked: bool):
        if checked:
            self.btn_apply_filters.setText("APLICAR FILTROS (VISTA PREVIA 50%)")
        else:
            self.btn_apply_filters.setText("APLICAR FILTROS DE CORONA (1:1)")

    def _on_apply_filters_btn_clicked(self):
        use_proxy = self.chk_proxy_preview.isChecked() if hasattr(self, 'chk_proxy_preview') else False
        self._on_apply_filters_clicked(use_proxy=use_proxy)

    def _on_render_full_clicked(self):
        self._on_apply_filters_clicked(use_proxy=False)

    def _on_apply_filters_clicked(self, use_proxy: bool = True):
        if self.worker is not None and self.worker.isRunning():
            return
        base_img = self.fused_hdr_img if self.fused_hdr_img is not None else self.current_preview_img
        if base_img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen cargada o fusionada sobre la cual aplicar filtros.")
            return

        cx, cy, r = self._get_active_geometry_for_image(base_img)

        if r <= 5:
            QMessageBox.warning(self, "Atención", "Define o auto-detecta el radio del limbo antes de aplicar los filtros.")
            return

        h_orig, w_orig = base_img.shape[:2]
        is_proxy = use_proxy and (h_orig >= 500 and w_orig >= 500)
        scale = 0.5 if is_proxy else 1.0

        if is_proxy:
            expected_shape = (int(round(h_orig * 0.5)), int(round(w_orig * 0.5)))
            if (self.proxy_hdr_img is None or self.proxy_hdr_img.shape[:2] != expected_shape):
                self.proxy_hdr_img = cv2.resize(base_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            proc_img = self.proxy_hdr_img
            proc_cx = cx * scale
            proc_cy = cy * scale
            proc_r = r * scale
        else:
            proc_img = base_img
            proc_cx = cx
            proc_cy = cy
            proc_r = r

        idx = self.combo_filter_algo.currentIndex()
        reach = 3.5
        if idx == 0:
            filter_type = "steerable_g2"
            blend = self.slider_blend_g2.value() / 100.0
            reach = self.slider_reach_g2.value() / 10.0
        elif idx == 1:
            filter_type = "mgn"
            blend = self.slider_blend.value() / 100.0
            reach = 3.5
        elif idx == 2:
            filter_type = "larson_sekanina"
            blend = self.slider_blend_ls.value() / 100.0
            reach = self.slider_reach_ls.value() / 10.0
        elif idx == 3:
            filter_type = "swap"
            blend = self.slider_blend_swap.value() / 100.0
            reach = 3.5
        elif idx == 4:
            filter_type = "nrgf"
            blend = self.slider_blend_nrgf.value() / 100.0
            reach = self.slider_reach.value() / 10.0
        elif idx == 5:
            filter_type = "nafe"
            blend = self.slider_blend_nafe.value() / 100.0
            reach = 3.5
        elif idx == 6:
            filter_type = "wow"
            blend = self.slider_blend_wow.value() / 100.0
            reach = 3.5
        elif idx == 7:
            filter_type = "fnrgf"
            blend = self.slider_blend_fnrgf.value() / 100.0
            reach = self.slider_reach_fnrgf.value() / 10.0
        else:
            filter_type = "wow"
            blend = self.slider_blend_wow.value() / 100.0
            reach = 3.5

        bin_px = self.spin_bin_px.value()
        robust = self.rb_robust.isChecked()
        mgn_k = self.slider_mgn_k.value() / 100.0
        mgn_gamma = self.slider_mgn_gamma.value() / 10.0
        mgn_noise_floor = self.slider_mgn_noise_floor.value() / 1000.0
        use_hp = self.chk_highpass.isChecked()
        hp_strength = (self.slider_hp.value() / 100.0) if use_hp else 0.0

        use_dark = self.chk_dark_structure.isChecked() if hasattr(self, 'chk_dark_structure') else False
        dark_amount = (self.slider_dark_amount.value() / 100.0) if use_dark else 0.0
        dark_protect_prom = self.chk_dark_protect_prom.isChecked() if hasattr(self, 'chk_dark_protect_prom') else True

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_apply_filters.setEnabled(False)
        if hasattr(self, 'btn_render_full'):
            self.btn_render_full.setEnabled(False)
        self.btn_fuse_hdr.setEnabled(False)
        self.btn_detect_all.setEnabled(False)
        if hasattr(self, 'btn_bypass_tb'):
            self.btn_bypass_tb.setEnabled(False)
        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.setEnabled(False)
            self.btn_split_view.blockSignals(True)
            self.btn_split_view.setChecked(False)
            self.btn_split_view.blockSignals(False)
            self.canvas.set_split_view_enabled(False)
        if hasattr(self, 'btn_split_orientation'):
            self.btn_split_orientation.setEnabled(False)

        offset = (self.spin_mask_offset.value() if hasattr(self, 'spin_mask_offset') else -2.5) * scale
        mask_radius = max(1.0, proc_r + offset)

        params = {
            "image": proc_img,
            "cx": proc_cx,
            "cy": proc_cy,
            "radius": proc_r,
            "mask_radius": mask_radius,
            "mask_offset": offset,
            "scale": scale,
            "is_proxy": is_proxy,
            "filter_type": filter_type,
            "mgn_k": mgn_k,
            "mgn_gamma": mgn_gamma,
            "mgn_noise_floor": mgn_noise_floor,
            "mgn_chroma_denoise": float(self.slider_mgn_chroma_denoise.value()),
            "bin_px": bin_px,
            "robust": robust,
            "blend": blend,
            "reach": reach,
            "use_highpass": use_hp,
            "hp_strength": hp_strength,
            "use_dark_structure": use_dark,
            "dark_amount": dark_amount,
            "dark_protect_prominences": dark_protect_prom,
            "use_detail_rings": self.chk_detail_rings.isChecked(),
            "ls_delta_theta": self.spin_ls_dtheta.value(),
            "ls_protect_prominences": self.chk_ls_prominences.isChecked(),
            "swap_window_deg": self.spin_swap_window.value(),
            "swap_c0": self.spin_swap_c0.value(),
            "nrgf_partial_annuli": "chain" if self.chk_nrgf_chain.isChecked() else "observed",
            "nafe_sigma": self.spin_nafe_sigma.value(),
            "nafe_use_overlay": self.chk_nafe_overlay.isChecked(),
            "wow_scales": self.spin_wow_scales.value(),
            "wow_guided": self.chk_wow_guided.isChecked(),
            "fnrgf_order": self.spin_fnrgf_order.value() if hasattr(self, 'spin_fnrgf_order') else 4,
        }

        self.worker = EclipseWorker("apply_filters", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_export_psd_clicked(self, out_path: Optional[str] = None):
        base_img = self.fused_hdr_img if self.fused_hdr_img is not None else self.current_preview_img
        if base_img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen cargada o fusionada para exportar como PSD.")
            return

        cx, cy, r = self._get_active_geometry_for_image(base_img)

        if r <= 5:
            QMessageBox.warning(self, "Atención", "Define o auto-detecta el radio del limbo antes de exportar el PSD multicapa.")
            return

        if not out_path:
            default_dir = get_config_val("export_dir", os.getcwd())
            os.makedirs(default_dir, exist_ok=True)
            default_path = os.path.join(default_dir, "eclipse_coronal_layers.psd")
            out_path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar PSD Multicapa de Corona Solar",
                default_path,
                "Archivos PSD de Photoshop (*.psd)"
            )
            if not out_path:
                return

        out_path = os.path.abspath(os.path.normpath(str(out_path).strip()))
        if not out_path.lower().endswith(".psd"):
            out_path += ".psd"

        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError as err:
                QMessageBox.critical(
                    self, "Error de Exportación",
                    f"No se puede sobreescribir el archivo destino porque está abierto en otro programa:\n{out_path}\n\nDetalle: {err}"
                )
                return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.log(f"Iniciando exportación de PSD multicapa coronal a {os.path.basename(out_path)}...")

        params = {
            "filepath": out_path,
            "base_image": base_img,
            "cx": cx,
            "cy": cy,
            "radius": r,
            "depth": 16
        }

        self.worker = EclipseWorker("export_coronal_psd", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, val: int, msg: str):
        self.progress_bar.setValue(val)
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.set_progress(val, msg)
        self.log(msg)

    def _on_worker_finished(self, result: object, task_type: str):
        self.progress_bar.setVisible(False)
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.hide_progress()
        self.btn_detect_all.setEnabled(True)
        self.btn_fuse_hdr.setEnabled(True)
        self.btn_apply_filters.setEnabled(True)
        if hasattr(self, 'btn_render_full'):
            self.btn_render_full.setEnabled(True)

        if task_type == "batch_detect":
            self.cached_aligned_data = None
            self._current_hdr_cache_key = None
            ref_det = result[self.ref_file_idx] if (0 <= self.ref_file_idx < len(result)) else None
            ref_cx = ref_det[0] if ref_det is not None else None
            ref_cy = ref_det[1] if ref_det is not None else None
            ref_r = ref_det[2] if (ref_det is not None and ref_det[2] > 10) else None
            outliers = []

            for i, (cx, cy, r) in enumerate(result):
                if i < len(self.bracketing_files):
                    self.bracketing_files[i]["cx"] = cx
                    self.bracketing_files[i]["cy"] = cy
                    self.bracketing_files[i]["radius"] = r
                    if i != self.ref_file_idx and ref_r is not None:
                        r_err = abs(r - ref_r) / ref_r
                        shift_d = np.hypot(cx - ref_cx, cy - ref_cy)
                        if r_err > 0.025 or shift_d > max(0.35 * ref_r, 80.0):
                            self.bracketing_files[i]["is_included"] = False
                            outliers.append(self.bracketing_files[i]["filename"])
            self._recalculate_shifts()

            rows = self.table_brack.selectionModel().selectedRows()
            sel_idx = rows[0].row() if rows else self.ref_file_idx
            if sel_idx < 0 or sel_idx >= len(self.bracketing_files):
                sel_idx = 0

            self._refresh_table()
            self.table_brack.selectRow(sel_idx)

            f_data = self.bracketing_files[sel_idx]
            if f_data.get("cx") is not None:
                self._set_geom_spin_values(f_data["cx"], f_data["cy"], f_data["radius"])
                try:
                    img = load_image_as_float32(f_data["filepath"])
                    self.current_preview_img = img
                    self._display_single_preview(img)
                except Exception:
                    pass
                self._update_canvas_guide()

            if outliers:
                self.log(f"Aviso: {len(outliers)} toma(s) desmarcadas (USAR=False) por desviación respecto a REF: {', '.join(outliers)}")
            self.log("Detección de limbo en lote completada. Desplazamientos calculados.")

        elif task_type == "hdr_fusion":
            self._viewing_fused = True
            self.raw_fused_hdr = result["hdr_raw"]
            self.fused_hdr_img = result["hdr_image"]
            if self.fused_hdr_img.shape[0] >= 500 and self.fused_hdr_img.shape[1] >= 500:
                self.proxy_hdr_img = cv2.resize(self.fused_hdr_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            else:
                self.proxy_hdr_img = None
            self._current_is_proxy = False
            self.canvas.set_image(self.fused_hdr_img)

            raw_centers = result.get("raw_centers", result["centers"])
            final_centers = result["centers"]
            shifts = result["shifts"]
            ref_idx = result.get("ref_idx", self.ref_file_idx)

            for i, rc in enumerate(raw_centers):
                if i < len(self.bracketing_files) and rc is not None and len(rc) >= 3:
                    self.bracketing_files[i]["cx"] = rc[0]
                    self.bracketing_files[i]["cy"] = rc[1]
                    self.bracketing_files[i]["radius"] = rc[2]
                    self.bracketing_files[i]["shift_x"] = shifts[i][0]
                    self.bracketing_files[i]["shift_y"] = shifts[i][1]

            self._refresh_table()

            fused_geom = result.get("fused_geom")
            if fused_geom is None and ref_idx < len(final_centers):
                fused_geom = final_centers[ref_idx]

            if fused_geom is not None and len(fused_geom) >= 3:
                self.fused_hdr_geom = (float(fused_geom[0]), float(fused_geom[1]), float(fused_geom[2]))
                self._set_geom_spin_values(fused_geom[0], fused_geom[1], fused_geom[2])
            else:
                ref_f = self.bracketing_files[self.ref_file_idx] if (self.bracketing_files and 0 <= self.ref_file_idx < len(self.bracketing_files)) else None
                if ref_f and ref_f.get("cx") is not None:
                    self.fused_hdr_geom = (float(ref_f["cx"]), float(ref_f["cy"]), float(ref_f["radius"]))
                    self._set_geom_spin_values(ref_f["cx"], ref_f["cy"], ref_f["radius"])

            if "aligned_images" in result:
                aligned_list = result["aligned_images"]
                total_bytes = sum(getattr(im, "nbytes", 0) for im in aligned_list if im is not None)
                if total_bytes <= 3 * 1024 ** 3:
                    cache_raw_key = tuple(
                        (round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 2))
                        if c is not None and len(c) >= 3 else None
                        for c in raw_centers
                    )
                    worker_params = getattr(self.worker, "params", {}) if self.worker else {}
                    saved_cache_key = (
                        tuple(worker_params.get("files", [])),
                        cache_raw_key,
                        worker_params.get("ref_idx", 0),
                        worker_params.get("auto_align", True),
                        worker_params.get("crop_roi", False),
                        int(worker_params.get("crop_size", 3000))
                    )
                    self.cached_aligned_data = {
                        "key": saved_cache_key,
                        "aligned_images": aligned_list,
                        "final_centers": final_centers,
                        "raw_centers": raw_centers,
                        "fused_geom": self.fused_hdr_geom,
                        "shifts": shifts
                    }
                    self._current_hdr_cache_key = saved_cache_key
                else:
                    self.cached_aligned_data = None
                    self._current_hdr_cache_key = None
                    self.log(f"Caché de alineación omitida ({total_bytes / (1024**3):.1f} GB > 3.0 GB) para liberar RAM. Activa 'Recorte ROI' si deseas cachear tomas alineadas.")

            if result.get("was_cached", False):
                self.log("Fusión HDR finalizada instantáneamente (utilizando tomas alineadas en memoria caché).")
            else:
                self.log("Fusión HDR finalizada con alineación subpíxel y estirado Asinh.")

            if hasattr(self, 'btn_bypass_tb'):
                self.btn_bypass_tb.setEnabled(False)
                self.btn_bypass_tb.blockSignals(True)
                self.btn_bypass_tb.setChecked(False)
                self.btn_bypass_tb.setText("Comparar A/B")
                self.btn_bypass_tb.blockSignals(False)
            if hasattr(self, 'btn_split_view'):
                self.btn_split_view.setEnabled(False)
                self.btn_split_view.blockSignals(True)
                self.btn_split_view.setChecked(False)
                self.btn_split_view.blockSignals(False)
                self.canvas.set_split_view_enabled(False)
            if hasattr(self, 'btn_split_orientation'):
                self.btn_split_orientation.setEnabled(False)

        elif task_type == "apply_filters":
            self._viewing_fused = True
            self.final_filtered_img = result
            is_proxy = self.worker.params.get("is_proxy", False) if self.worker else False
            self._current_is_proxy = is_proxy
            self.canvas.set_image(result)
            self._update_canvas_guide()
            if hasattr(self, 'btn_bypass_tb'):
                self.btn_bypass_tb.setEnabled(True)
                self.btn_bypass_tb.blockSignals(True)
                self.btn_bypass_tb.setChecked(False)
                self.btn_bypass_tb.setText("B: Filtrada" + (" [Proxy]" if is_proxy else " [1:1]"))
                self.btn_bypass_tb.blockSignals(False)
            if hasattr(self, 'btn_split_view'):
                self.btn_split_view.setEnabled(True)
                self.btn_split_view.blockSignals(True)
                self.btn_split_view.setChecked(False)
                self.btn_split_view.blockSignals(False)
            if hasattr(self, 'btn_split_orientation'):
                self.btn_split_orientation.setEnabled(False)
            self._update_before_reference()
            status_tag = "[Proxy 50%]" if is_proxy else "[Nativo 1:1]"
            self.lbl_canvas_status.setText(f"Mostrando: Filtros Coronales {status_tag}")
            self.log(f"Filtros de corona aplicados {status_tag} con compuerta radial.")

        elif task_type == "export_coronal_psd":
            filepath, desc = result
            self.log(f"✓ {desc}")
            QMessageBox.information(
                self,
                "Exportación PSD Exitosa",
                f"Archivo PSD multicapa guardado con éxito:\n{filepath}\n\n{desc}"
            )

        elif task_type == "compute_coronal_stack":
            base_path, computed_layers = result
            self.log(f"✓ Multicapa coronal procesada: transfiriendo {len(computed_layers)} capas al Revelador...")
            self.export_coronal_stack_to_developer.emit(base_path, computed_layers)

        if hasattr(self, 'btn_send_coronal_stack_to_dev'):
            self.btn_send_coronal_stack_to_dev.setEnabled(True)

    def _on_worker_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        if hasattr(self, 'canvas_footer'):
            self.canvas_footer.hide_progress()
        self.btn_detect_all.setEnabled(True)
        self.btn_fuse_hdr.setEnabled(True)
        self.btn_apply_filters.setEnabled(True)
        if hasattr(self, 'btn_render_full'):
            self.btn_render_full.setEnabled(True)
        if hasattr(self, 'btn_send_coronal_stack_to_dev'):
            self.btn_send_coronal_stack_to_dev.setEnabled(True)
        QMessageBox.critical(self, "Error de Procesado", err_msg)
        self.log(f"ERROR: {err_msg}")

    def _get_best_current_image(self) -> np.ndarray:
        if self.final_filtered_img is not None:
            return self.final_filtered_img
        if self.fused_hdr_img is not None:
            return self.fused_hdr_img
        return self.current_preview_img

    def _on_save_image_clicked(self, force_format: Optional[str] = None):
        """Abre el diálogo unificado de guardado permitiendo elegir formato por filtro o extensión."""
        img = self._get_best_current_image()
        if img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen disponible para guardar.")
            return

        filtros = (
            "TIFF 16-bit (*.tif *.tiff);;"
            "TIFF 32-bit Float (*.tif *.tiff);;"
            "Photoshop Multicapa Coronal (*.psd);;"
            "PNG (*.png);;"
            "JPEG (*.jpg *.jpeg)"
        )
        if force_format == "psd":
            filtro_por_defecto = "Photoshop Multicapa Coronal (*.psd)"
            default_filename = "eclipse_coronal.psd"
        elif force_format == "tiff32":
            filtro_por_defecto = "TIFF 32-bit Float (*.tif *.tiff)"
            default_filename = "eclipse_32bit.tif"
        elif force_format == "png":
            filtro_por_defecto = "PNG (*.png)"
            default_filename = "eclipse_procesado.png"
        elif force_format in ("jpg", "jpeg"):
            filtro_por_defecto = "JPEG (*.jpg *.jpeg)"
            default_filename = "eclipse_procesado.jpg"
        else:
            filtro_por_defecto = "TIFF 16-bit (*.tif *.tiff)"
            default_filename = "eclipse_procesado.tif"

        default_dir = get_config_val("export_dir", os.getcwd())
        os.makedirs(default_dir, exist_ok=True)
        default_path = os.path.join(default_dir, default_filename)

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Guardar Imagen de Eclipse",
            default_path,
            filtros,
            selectedFilter=filtro_por_defecto
        )
        if not path:
            return

        path = os.path.abspath(os.path.normpath(str(path).strip()))
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            if "Photoshop" in selected_filter or "*.psd" in selected_filter:
                ext = ".psd"
            elif "PNG" in selected_filter or "*.png" in selected_filter:
                ext = ".png"
            elif "JPEG" in selected_filter or "*.jpg" in selected_filter:
                ext = ".jpg"
            else:
                ext = ".tif"
            path += ext

        # PSD Multicapa coronal
        if ext == ".psd" or ("Photoshop" in selected_filter and ext not in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']):
            self._on_export_psd_clicked(out_path=path)
            return

        # Si la imagen a guardar proviene de un proxy 50%, advertir al usuario para máxima calidad
        if getattr(self, "_current_is_proxy", False):
            resp = QMessageBox.question(
                self,
                "Aviso de Previsualización Proxy (50%)",
                "La imagen actual en el visor es un borrador Proxy al 50%.\n\n"
                "Para obtener la máxima resolución nativa (1:1), se recomienda generar primero el 'RENDER FINAL 1:1'.\n\n"
                "¿Deseas cancelar el guardado para pulsar 'RENDER FINAL 1:1' primero?\n"
                "(Pulsa 'Sí' para volver y hacer Render 1:1, o 'No' para continuar guardando el proxy al 50%).",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if resp == QMessageBox.Yes:
                return

        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as err:
                QMessageBox.critical(
                    self, "Error de Exportación",
                    f"No se puede sobreescribir el archivo destino porque está abierto en otro programa:\n{path}\n\nDetalle: {err}"
                )
                return

        photometric = 'minisblack' if img.ndim == 2 else 'rgb'

        try:
            if ext in ['.jpg', '.jpeg'] or "JPEG" in selected_filter:
                if img.ndim == 2:
                    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                    bgr8 = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
                else:
                    rgb_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                    bgr8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, bgr8, [cv2.IMWRITE_JPEG_QUALITY, 96])
                desc = "JPEG 8-bit (Calidad 96)"
            elif ext == '.png' or "PNG" in selected_filter:
                u16 = np.ascontiguousarray(np.clip(img * 65535.0 + 0.5, 0.0, 65535.0), dtype=np.uint16)
                bgr16 = cv2.cvtColor(u16, cv2.COLOR_RGB2BGR) if img.ndim == 3 else u16
                cv2.imwrite(path, bgr16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                desc = "PNG 16-bit"
            elif "32-bit" in selected_filter:
                out_img = np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)
                desc = "TIFF 32-bit float (HDR)"
                save_tiff_with_icc(path, out_img, compression='zlib', photometric=photometric)
            else:
                out_img = np.ascontiguousarray(np.clip(img * 65535.0 + 0.5, 0.0, 65535.0), dtype=np.uint16)
                desc = "TIFF 16-bit entero (Photoshop / Lightroom)"
                save_tiff_with_icc(path, out_img, compression='zlib', photometric=photometric)

            self.log(f"Imagen guardada exitosamente ({desc} con perfil ICC): {path}")
            QMessageBox.information(
                self, "Exportación Completada",
                f"Imagen guardada con éxito en:\n{path}\n\nFormato: {desc} ({img.shape[1]}x{img.shape[0]} px)"
            )
        except Exception as e:
            self.log(f"ERROR al guardar imagen: {e}")
            QMessageBox.critical(self, "Error de Exportación", f"No se pudo guardar la imagen:\n{e}")

    def _on_save_tiff_clicked(self, bit_depth: int = 16):
        """Compatibilidad hacia atrás con llamadas directas."""
        self._on_save_image_clicked()

    def _on_export_web_clicked(self):
        """Abre el diálogo modal de exportación optimizada para Web y Redes Sociales."""
        img = self._get_best_current_image()
        if img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen disponible para exportar.")
            return

        from gui.dialog_web_export import WebExportDialog, extract_exif_bytes
        ref_path = None
        if hasattr(self, 'bracketing_files') and self.bracketing_files:
            idx = getattr(self, 'ref_file_idx', 0)
            if 0 <= idx < len(self.bracketing_files):
                ref_path = self.bracketing_files[idx].get("filepath")
            elif self.bracketing_files:
                ref_path = self.bracketing_files[0].get("filepath")

        exif_bytes = extract_exif_bytes(ref_path) if ref_path else None
        base_name = os.path.splitext(os.path.basename(ref_path))[0] if ref_path else "eclipse_procesado"

        dlg = WebExportDialog(
            image_buffer=img,
            default_filename=base_name,
            exif_bytes=exif_bytes,
            source_colorspace="srgb",
            parent=self,
        )
        dlg.exec()

    def _on_send_to_developer_clicked(self):
        img = self._get_best_current_image()
        if img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen disponible para transferir.")
            return

        out_dir = os.path.join(os.getcwd(), "exportaciones")
        os.makedirs(out_dir, exist_ok=True)
        temp_path = os.path.join(out_dir, "eclipse_processed_32bit.tif")

        try:
            photometric = 'minisblack' if img.ndim == 2 else 'rgb'
            out_img = np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)
            save_tiff_with_icc(temp_path, out_img, compression='zlib', photometric=photometric)
            self.log("Transfiriendo imagen 32-bit al Revelador...")
            self.export_to_developer.emit(temp_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error transfiriendo al Revelador: {e}")

    def _on_send_coronal_stack_clicked(self):
        base_img = self.fused_hdr_img if self.fused_hdr_img is not None else self.current_preview_img
        if base_img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen cargada o fusionada para transferir.")
            return

        cx, cy, r = self._get_active_geometry_for_image(base_img)

        if r <= 5:
            QMessageBox.warning(self, "Atención", "Define o auto-detecta el centro y radio del limbo antes de generar la multicapa coronal.")
            return

        dlg = CoronalStackDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return

        selected_configs = dlg.get_selected_layers()
        if not selected_configs:
            # Si no seleccionó ninguna capa ráster adicional, transferir sólo la base
            self._on_send_to_developer_clicked()
            return

        out_dir = os.path.join(os.getcwd(), "exportaciones")
        os.makedirs(out_dir, exist_ok=True)
        out_base_path = os.path.join(out_dir, "eclipse_hdr_base_32bit.tif")

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(5)
        self.btn_detect_all.setEnabled(False)
        self.btn_fuse_hdr.setEnabled(False)
        self.btn_apply_filters.setEnabled(False)
        if hasattr(self, 'btn_send_coronal_stack_to_dev'):
            self.btn_send_coronal_stack_to_dev.setEnabled(False)

        self.log(f"Iniciando cálculo de {len(selected_configs)} capas coronales para el Revelador...")

        offset = self.spin_mask_offset.value() if hasattr(self, 'spin_mask_offset') else -2.5
        mask_radius = max(1.0, r + offset)

        filter_params = {
            "use_detail_rings": self.chk_detail_rings.isChecked(),
            "bin_px": self.spin_bin_px.value(),
            "robust": self.rb_robust.isChecked(),
            "mgn_k": self.slider_mgn_k.value() / 100.0,
            "mgn_gamma": self.slider_mgn_gamma.value() / 10.0,
            "mgn_noise_floor": self.slider_mgn_noise_floor.value() / 1000.0,
            "mgn_chroma_denoise": float(self.slider_mgn_chroma_denoise.value()),
            "mgn_blend": self.slider_blend.value() / 100.0,
            "g2_blend": self.slider_blend_g2.value() / 100.0,
            "g2_reach": self.slider_reach_g2.value() / 10.0,
            "ls_blend": self.slider_blend_ls.value() / 100.0,
            "ls_reach": self.slider_reach_ls.value() / 10.0,
            "ls_delta_theta": self.spin_ls_dtheta.value(),
            "ls_protect_prominences": self.chk_ls_prominences.isChecked(),
            "swap_blend": self.slider_blend_swap.value() / 100.0,
            "swap_window_deg": self.spin_swap_window.value(),
            "swap_c0": self.spin_swap_c0.value(),
            "nrgf_blend": self.slider_blend_nrgf.value() / 100.0,
            "nrgf_reach": self.slider_reach.value() / 10.0,
            "nrgf_partial_annuli": "chain" if self.chk_nrgf_chain.isChecked() else "observed",
            "fnrgf_blend": self.slider_blend_fnrgf.value() / 100.0,
            "fnrgf_reach": self.slider_reach_fnrgf.value() / 10.0,
            "fnrgf_order": self.spin_fnrgf_order.value() if hasattr(self, 'spin_fnrgf_order') else 4,
            "nafe_blend": self.slider_blend_nafe.value() / 100.0,
            "nafe_sigma": self.spin_nafe_sigma.value(),
            "nafe_use_overlay": self.chk_nafe_overlay.isChecked(),
            "wow_blend": self.slider_blend_wow.value() / 100.0,
            "wow_scales": self.spin_wow_scales.value(),
            "wow_guided": self.chk_wow_guided.isChecked(),
            "dark_amount": self.slider_dark_amount.value() / 100.0 if hasattr(self, 'slider_dark_amount') else 0.40,
            "dark_protect_prominences": self.chk_dark_protect_prom.isChecked() if hasattr(self, 'chk_dark_protect_prom') else True,
        }

        params = {
            "base_image": base_img,
            "cx": cx,
            "cy": cy,
            "radius": r,
            "mask_radius": mask_radius,
            "mask_offset": offset,
            "selected_configs": selected_configs,
            "out_base_path": out_base_path,
            "filter_params": filter_params,
        }

        self.worker = EclipseWorker("compute_coronal_stack", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    @property
    def mode(self) -> str:
        if hasattr(self, 'rb_solar') and self.rb_solar is not None:
            return "solar" if self.rb_solar.isChecked() else "lunar"
        return "solar"



