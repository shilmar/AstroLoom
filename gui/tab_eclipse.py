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
import numpy as np
import cv2
import tifffile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox, QGroupBox, QRadioButton, QSlider,
    QDoubleSpinBox, QSplitter, QTextEdit, QScrollArea, QCheckBox,
    QSizePolicy, QAbstractItemView, QApplication, QButtonGroup
)
from PySide6.QtCore import Qt, Signal, QThread, QPoint, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QCursor, QFont, QTextCursor, QBrush

from core.eclipse import (
    extract_exposure_info,
    detect_eclipse_disk,
    refine_disk_subpixel,
    register_frame_to_center,
    align_eclipse_bracketing,
    fuse_hdr_bracketing,
    apply_asinh_stretch,
    apply_nrgf,
    apply_mgn,
    apply_coronal_highpass
)
from core.stacking import load_image_as_float32
from core.icc_manager import save_tiff_with_icc
from core.config_manager import get_config_val
from concurrent.futures import ThreadPoolExecutor, as_completed
from gui.widgets.drop_widgets import DropTableWidget, collect_image_files_from_mime
from gui.theme import STYLE_RUN_IDLE, STYLE_TRANSFER_BTN
from gui.icons import get_icon, get_svg_pixmap, ICON_COLOR_DEFAULT, ICON_COLOR_ACTIVE, ICON_COLOR_ACCENT
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow


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
                background-color: #10121a;
                color: #78909c;
                border: 1px solid #1e2230;
                border-bottom: 2px solid transparent;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QPushButton:hover:!checked {
                background-color: #151822;
                color: #eceff4;
            }
            QPushButton:checked {
                background-color: #171a26;
                color: #00e5ff;
                border-color: #232738;
                border-bottom: 2px solid #00e5ff;
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
            self.btn_solar.setIcon(get_icon("sun", color="#ffd54f"))
            self.btn_lunar.setIcon(get_icon("moon", color="#64748b"))
        else:
            self.btn_lunar.setChecked(True)
            self.btn_solar.setIcon(get_icon("sun", color="#64748b"))
            self.btn_lunar.setIcon(get_icon("moon", color="#00e5ff"))

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
        self.pick_center_mode: bool = False

    def set_image(self, img_f32: np.ndarray):
        """Asigna y renderiza una imagen float32 en rango [0, 1]."""
        self.orig_f32 = img_f32
        h, w = img_f32.shape[:2]

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

    def set_guide_params(self, cx: float, cy: float, r: float, show: bool = True):
        """Actualiza las coordenadas de la guía circular de limbo."""
        self.center_x = cx
        self.center_y = cy
        self.radius = r
        self.show_guide = show
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

        # Dibujar imagen con zoom y pan
        iw = self.base_pixmap.width() * self.zoom_level
        ih = self.base_pixmap.height() * self.zoom_level
        target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), iw, ih)
        painter.drawPixmap(target_rect.toRect(), self.base_pixmap)

        # Realimentación visual al arrastrar archivos sobre el visor
        if self._drag_hover:
            self._draw_drag_feedback(painter)

        # Dibujar guía geométrica del limbo si está habilitada y es válida
        if self.show_guide and self.radius > 5:
            scx = self.pan_offset.x() + self.center_x * self.zoom_level
            scy = self.pan_offset.y() + self.center_y * self.zoom_level
            sr = self.radius * self.zoom_level

            pen_bg = QPen(QColor(0, 0, 0, 220), 3, Qt.DashLine)
            pen_fg = QPen(QColor(0, 240, 255, 240), 1.5, Qt.DashLine)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(pen_bg)
            painter.drawEllipse(QPoint(int(scx), int(scy)), int(sr), int(sr))
            painter.setPen(pen_fg)
            painter.drawEllipse(QPoint(int(scx), int(scy)), int(sr), int(sr))

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
            info_txt = f"Limbo (X: {self.center_x:.1f}, Y: {self.center_y:.1f}, R: {self.radius:.1f}px)"
            painter.drawText(int(scx + sr + 8), int(scy), info_txt)

    def mousePressEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self.pick_center_mode and self.base_pixmap is not None:
                img_x = (event.position().x() - self.pan_offset.x()) / self.zoom_level
                img_y = (event.position().y() - self.pan_offset.y()) / self.zoom_level
                iw, ih = self.base_pixmap.width(), self.base_pixmap.height()
                if 0 <= img_x < iw and 0 <= img_y < ih:
                    self.center_clicked.emit(img_x, img_y)
                    self.pick_center_mode = False
                    self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.is_panning = False
            self.setCursor(Qt.CrossCursor if self.pick_center_mode else Qt.ArrowCursor)

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

                # 1. Detectar primero la toma de referencia para anclar el radio físico esperado
                ref_img = load_image_as_float32(files[ref_idx])
                ref_cx, ref_cy, ref_r = detect_eclipse_disk(ref_img, mode=mode, expected_radius=None)
                detections[ref_idx] = (ref_cx, ref_cy, ref_r)
                expected_r = ref_r if ref_r > 10 else None

                # 2. Detectar el resto de tomas usando expected_r en paralelo a resolución 1:1 nativa
                items_to_detect = [(idx, path) for idx, path in enumerate(files) if idx != ref_idx]

                if items_to_detect:
                    try:
                        cpu_workers = int(get_config_val("cpu_workers", max(2, (os.cpu_count() or 4) - 2)))
                    except Exception:
                        cpu_workers = max(2, (os.cpu_count() or 4) - 2)
                    eff_workers = max(1, min(cpu_workers, os.cpu_count() or 16, len(items_to_detect)))

                    def _detect_task(item):
                        idx, path = item
                        img = load_image_as_float32(path)
                        cx, cy, r = detect_eclipse_disk(img, mode=mode, expected_radius=expected_r)
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

                total = len(files)
                loaded_images = [None] * total
                final_centers = [None] * total
                self.progress.emit(5, "Iniciando carga paralela de tomas de bracketing (1:1 nativo)...")

                expected_r = None
                if centers is not None:
                    valid_rs = [c[2] for c in centers if c is not None and len(c) > 2 and c[2] > 10]
                    if valid_rs:
                        expected_r = float(np.median(valid_rs))

                try:
                    cpu_workers = int(get_config_val("cpu_workers", max(2, (os.cpu_count() or 4) - 2)))
                except Exception:
                    cpu_workers = max(2, (os.cpu_count() or 4) - 2)
                eff_workers = max(1, min(cpu_workers, os.cpu_count() or 16, total))

                def _load_and_detect_one(idx_path):
                    idx, path = idx_path
                    img = load_image_as_float32(path)
                    if centers is not None and idx < len(centers) and centers[idx] is not None and centers[idx][2] > 5:
                        c = centers[idx]
                    else:
                        cx, cy, r = detect_eclipse_disk(img, mode=mode, expected_radius=expected_r)
                        c = (cx, cy, r)
                    return idx, img, c, os.path.basename(path)

                completed = 0
                items = list(enumerate(files))
                if eff_workers == 1 or total == 1:
                    for item in items:
                        if self.isInterruptionRequested():
                            break
                        idx, img, c, bname = _load_and_detect_one(item)
                        loaded_images[idx] = img
                        final_centers[idx] = c
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
                            idx, img, c, bname = future.result()
                            loaded_images[idx] = img
                            final_centers[idx] = c
                            completed += 1
                            self.progress.emit(
                                int(5 + (completed / total) * 40),
                                f"Cargando ({completed}/{total}): {bname}"
                            )

                if auto_align and len(final_centers) == len(loaded_images) and all(img is not None for img in loaded_images):
                    self.progress.emit(50, "Alineando tomas por centro de limbo (interpolación Lanczos4)...")
                    center_pts = [(c[0], c[1]) for c in final_centers]
                    aligned_images, shifts = align_eclipse_bracketing(
                        loaded_images,
                        center_pts,
                        ref_idx=ref_idx
                    )
                else:
                    aligned_images = [img for img in loaded_images if img is not None]
                    shifts = [(0.0, 0.0)] * len(aligned_images)

                ref_center_info = final_centers[ref_idx] if (ref_idx < len(final_centers) and final_centers[ref_idx] is not None) else None
                lunar_mask = (ref_center_info[0], ref_center_info[1], ref_center_info[2]) if (ref_center_info is not None and len(ref_center_info) > 2 and ref_center_info[2] > 5) else None
                subtract_sky = self.params.get("subtract_sky", True)

                progress_msg = "Calculando ponderaciones y fusionando flujo fotométrico HDR..." if mode == "solar" else "Fusionando tomas con exposición multiescala (Mertens HDR)..."
                self.progress.emit(75, progress_msg)
                hdr_raw = fuse_hdr_bracketing(
                    aligned_images,
                    exposures,
                    saturation_threshold=sat_thresh,
                    asinh_stretch=0.0,
                    mode=mode,
                    lunar_mask_params=lunar_mask,
                    subtract_sky=subtract_sky
                )

                hdr_stretched = apply_asinh_stretch(hdr_raw, stretch_factor=asinh_factor) if asinh_factor > 1.0 else hdr_raw.copy()

                self.progress.emit(100, "Fusión HDR de 32 bits completada.")
                self.finished.emit({
                    "hdr_raw": hdr_raw,
                    "hdr_image": hdr_stretched,
                    "centers": final_centers,
                    "shifts": shifts,
                    "ref_idx": ref_idx
                }, "hdr_fusion")

            elif self.task_type == "apply_filters":
                base_img = self.params["image"]
                cx = self.params["cx"]
                cy = self.params["cy"]
                r = self.params["radius"]
                filter_type = self.params.get("filter_type", "mgn")
                use_highpass = self.params.get("use_highpass", False)
                hp_strength = self.params.get("hp_strength", 0.5)
                reach = self.params.get("reach", 3.5)

                if filter_type == "mgn":
                    self.progress.emit(25, "Calculando filtro multiescala MGN (Morgan & Druckmüller)...")
                    mgn_k = self.params.get("mgn_k", 0.7)
                    mgn_gamma = self.params.get("mgn_gamma", 2.8)
                    blend = self.params.get("blend", 0.50)
                    lunar_mask = (cx, cy, r) if r > 5 else None

                    filtered = apply_mgn(
                        base_img,
                        sigmas=(1.5, 3.0, 6.0, 12.0, 24.0, 48.0),
                        k=mgn_k,
                        gamma=mgn_gamma,
                        blend_ratio=blend,
                        lunar_mask_params=lunar_mask
                    )
                else:
                    bin_px = self.params.get("bin_px", 2.0)
                    robust = self.params.get("robust", True)
                    blend = self.params.get("blend", 0.55)

                    self.progress.emit(20, "Calculando filtro NRGF (gradiente radial polar)...")
                    filtered = apply_nrgf(
                        base_img,
                        center_x=cx,
                        center_y=cy,
                        bin_px=bin_px,
                        robust=robust,
                        blend_ratio=blend,
                        radius_min=r,
                        radius_max_factor=reach
                    )

                if use_highpass and hp_strength > 0.01:
                    self.progress.emit(75, "Realzando filamentos finos mediante filtro tangencial...")
                    filtered = apply_coronal_highpass(
                        filtered,
                        strength=hp_strength,
                        radius_min=r,
                        center_x=cx,
                        center_y=cy,
                        radius_max_factor=reach
                    )

                self.progress.emit(100, "Filtros de corona aplicados con éxito.")
                self.finished.emit(filtered, "apply_filters")

        except Exception as e:
            self.error.emit(f"Error durante el procesado: {str(e)}")


class EclipseTab(QWidget):
    """Pestaña interactiva de Eclipses (Solar y Lunar)."""
    export_to_developer = Signal(str)
    open_collage_requested = Signal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bracketing_files = []      # Lista de dicts con metadatos y centros
        self.ref_file_idx = 0           # Índice de la toma de referencia para alineación
        self.current_preview_img = None # Imagen float32 visible
        self.raw_fused_hdr = None       # Flujo puro HDR sin estirar
        self.fused_hdr_img = None       # Imagen HDR con estirado
        self.final_filtered_img = None   # Imagen tras aplicar filtros de corona
        self.worker = None

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
        self.btn_load_brack.setIcon(get_icon("folder", color="#80d8ff"))
        self.btn_load_brack.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_load_brack.clicked.connect(self._on_load_bracketing_clicked)
        self.btn_clear_brack = QPushButton("Limpiar")
        self.btn_clear_brack.setIcon(get_icon("trash", color="#ff8a80"))
        self.btn_clear_brack.setStyleSheet("font-weight: bold; color: #ff8a80;")
        self.btn_clear_brack.clicked.connect(self._on_clear_bracketing_clicked)
        btn_row_files.addWidget(self.btn_load_brack)
        btn_row_files.addWidget(self.btn_clear_brack)
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
        self.table_brack.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_brack.cellDoubleClicked.connect(self._on_table_cell_double_clicked)
        self.table_brack.files_dropped.connect(self.add_bracketing_paths)
        brack_layout.addWidget(self.table_brack)

        btn_align_row = QHBoxLayout()
        self.btn_detect_all = QPushButton("Detectar Limbo en Lote")
        self.btn_detect_all.setIcon(get_icon("target", color="#80d8ff"))
        self.btn_detect_all.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_detect_all.setToolTip("Analiza el limbo con escáner radial subpíxel en cada imagen")
        self.btn_detect_all.clicked.connect(self._on_detect_all_clicked)

        self.btn_set_ref = QPushButton("Marcar REF")
        self.btn_set_ref.setIcon(get_icon("pin", color="#a5d6a7"))
        self.btn_set_ref.setStyleSheet("font-weight: bold; color: #a5d6a7;")
        self.btn_set_ref.setToolTip("Establece la toma seleccionada como centro de referencia de alineación")
        self.btn_set_ref.clicked.connect(self._on_set_ref_clicked)

        btn_align_row.addWidget(self.btn_detect_all)
        btn_align_row.addWidget(self.btn_set_ref)
        brack_layout.addLayout(btn_align_row)

        left_layout.addWidget(self.sec_brack)

        # SECCIÓN 2: Geometría del Disco (Limbo de la Toma Seleccionada)
        self.sec_geom = CollapsibleSection("2. Geometría del Disco (Limbo)", is_expanded=True, icon_name="crosshair")
        geom_layout = self.sec_geom.content_layout

        btn_auto_row = QHBoxLayout()
        self.btn_detect_disk = QPushButton("Auto-detectar")
        self.btn_detect_disk.setIcon(get_icon("target", color="#80d8ff"))
        self.btn_detect_disk.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_detect_disk.setToolTip("Detección global del disco lunar/solar")
        self.btn_detect_disk.clicked.connect(self._on_auto_detect_clicked)

        self.btn_refine_subpixel = QPushButton("Ajuste Fino Subpíxel")
        self.btn_refine_subpixel.setIcon(get_icon("search", color="#b388ff"))
        self.btn_refine_subpixel.setStyleSheet("color: #b388ff; font-weight: bold;")
        self.btn_refine_subpixel.setToolTip("Ajusta el círculo exactamente al borde del limbo ignorando protuberancias")
        self.btn_refine_subpixel.clicked.connect(self._on_refine_subpixel_clicked)

        self.btn_pick_center = QPushButton("Clic Canvas")
        self.btn_pick_center.setIcon(get_icon("crosshair", color="#ffd54f"))
        self.btn_pick_center.setCheckable(True)
        self.btn_pick_center.setStyleSheet("font-weight: bold; color: #ffd54f;")
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

        left_layout.addWidget(self.sec_geom)

        # SECCIÓN 3: Fusión HDR (Bracketing y Estirado Asinh)
        self.grp_hdr = CollapsibleSection("3. Fusión HDR (Lineal 32-bit)", is_expanded=True, icon_name="zap")
        hdr_layout = self.grp_hdr.content_layout

        self.chk_auto_align = QCheckBox("Alinear tomas por limbo antes de fusionar")
        self.chk_auto_align.setChecked(True)
        self.chk_auto_align.setStyleSheet("font-weight: bold; color: #b9f6ca;")
        hdr_layout.addWidget(self.chk_auto_align)

        self.chk_sub_sky = QCheckBox("Sustracción de pedestal / cielo por toma")
        self.chk_sub_sky.setChecked(True)
        self.chk_sub_sky.setToolTip("Anivela el fondo de cielo y luz difusa en cada fotograma antes de fusionar para erradicar anillos o escalones concéntricos en las transiciones de saturación.")
        hdr_layout.addWidget(self.chk_sub_sky)

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
            default_val=35,
            unit="x",
            decimals=0,
            label_width=135
        )
        self.lbl_asinh_val = self.slider_asinh.spin  # compatibilidad
        self.slider_asinh.valueChanged.connect(self._on_asinh_slider_changed)
        hdr_layout.addWidget(self.slider_asinh)

        self.btn_fuse_hdr = QPushButton("FUSIONAR BRACKETING HDR")
        self.btn_fuse_hdr.setIcon(get_icon("zap", color="#ffffff"))
        self.btn_fuse_hdr.setFixedHeight(40)
        self.btn_fuse_hdr.setCursor(Qt.PointingHandCursor)
        self.btn_fuse_hdr.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_fuse_hdr.clicked.connect(self._on_fuse_hdr_clicked)
        hdr_layout.addWidget(self.btn_fuse_hdr)
        left_layout.addWidget(self.grp_hdr)

        # SECCIÓN 4: Filtros de Corona Solar (MGN / NRGF)
        self.grp_nrgf = CollapsibleSection("4. Filtros de Corona Solar (MGN / NRGF)", is_expanded=True, icon_name="sparkles")
        nrgf_layout = self.grp_nrgf.content_layout

        lbl_algo = QLabel("Algoritmo de Detalle Coronal:")
        lbl_algo.setStyleSheet("font-weight: bold; color: #eceff4; font-size: 11px;")
        nrgf_layout.addWidget(lbl_algo)

        row_algo = QHBoxLayout()
        self.rb_filter_mgn = QRadioButton("MGN Multiescala (Druckmüller)")
        self.rb_filter_mgn.setToolTip("Multi-scale Gaussian Normalization (Morgan & Druckmüller 2014). Normaliza el contraste simultáneamente en múltiples escalas espaciales sin generar anillos concéntricos ni recortar el cielo exterior.")
        self.rb_filter_mgn.setChecked(True)
        self.rb_filter_nrgf = QRadioButton("NRGF Clásico")
        self.rb_filter_nrgf.setToolTip("Normalized Radial Gradient Filter estándar por bines polares radiales con compuerta.")
        row_algo.addWidget(self.rb_filter_mgn)
        row_algo.addWidget(self.rb_filter_nrgf)
        nrgf_layout.addLayout(row_algo)

        # Contenedor MGN
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
            min_val=20,
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
        nrgf_layout.addWidget(self.container_mgn)

        # Contenedor NRGF
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
        nrgf_layout.addWidget(self.container_nrgf)
        self.container_nrgf.setVisible(False)

        self.rb_filter_mgn.toggled.connect(self._on_filter_algo_changed)

        self.chk_highpass = QCheckBox("Realce de hilos finos (filtro bilateral tangencial)")
        self.chk_highpass.setChecked(True)
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

        self.btn_apply_filters = QPushButton("APLICAR FILTROS DE CORONA")
        self.btn_apply_filters.setIcon(get_icon("sparkles", color="#ffffff"))
        self.btn_apply_filters.setFixedHeight(40)
        self.btn_apply_filters.setCursor(Qt.PointingHandCursor)
        self.btn_apply_filters.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 12px;
                background-color: #1565c0;
                color: white;
                border-radius: 6px;
                border: 1px solid #1976d2;
            }
            QPushButton:hover {
                background-color: #1976d2;
                border-color: #42a5f5;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        self.btn_apply_filters.clicked.connect(self._on_apply_filters_clicked)
        nrgf_layout.addWidget(self.btn_apply_filters)

        left_layout.addWidget(self.grp_nrgf)

        # SECCIÓN 5: Exportación y Acabado
        self.sec_export = CollapsibleSection("5. Exportación y Acabado", is_expanded=True, icon_name="save")
        export_layout = self.sec_export.content_layout

        btn_tiff_row = QHBoxLayout()
        self.btn_save_tiff_16 = QPushButton("Guardar TIFF 16-bit...")
        self.btn_save_tiff_16.setIcon(get_icon("save", color="#80cbc4"))
        self.btn_save_tiff_16.setStyleSheet("font-weight: bold; color: #80cbc4;")
        self.btn_save_tiff_16.setToolTip("Exporta en TIFF 16-bit entero compatible directamente con Photoshop, Lightroom y editores estándar sin conversiones extrañas.")
        self.btn_save_tiff_16.clicked.connect(lambda: self._on_save_tiff_clicked(bit_depth=16))

        self.btn_save_tiff_32 = QPushButton("Guardar TIFF 32-bit...")
        self.btn_save_tiff_32.setIcon(get_icon("save", color="#90caf9"))
        self.btn_save_tiff_32.setStyleSheet("font-weight: bold; color: #90caf9;")
        self.btn_save_tiff_32.setToolTip("Exporta en TIFF 32-bit float con el rango dinámico HDR lineal completo.")
        self.btn_save_tiff_32.clicked.connect(lambda: self._on_save_tiff_clicked(bit_depth=32))

        self.btn_save_tiff = self.btn_save_tiff_32  # Compatibilidad

        btn_tiff_row.addWidget(self.btn_save_tiff_16)
        btn_tiff_row.addWidget(self.btn_save_tiff_32)
        export_layout.addLayout(btn_tiff_row)

        self.btn_send_to_dev = QPushButton("Enviar al Revelador / Editor")
        self.btn_send_to_dev.setIcon(get_icon("arrow_right", color="#ffffff"))
        self.btn_send_to_dev.setFixedHeight(38)
        self.btn_send_to_dev.setCursor(Qt.PointingHandCursor)
        self.btn_send_to_dev.setStyleSheet(STYLE_TRANSFER_BTN)
        self.btn_send_to_dev.clicked.connect(self._on_send_to_developer_clicked)
        export_layout.addWidget(self.btn_send_to_dev)

        self.btn_open_collage = QPushButton("Diseñar Secuencia / Collage")
        self.btn_open_collage.setIcon(get_icon("grid", color="#e1bee7"))
        self.btn_open_collage.setFixedHeight(38)
        self.btn_open_collage.setCursor(Qt.PointingHandCursor)
        self.btn_open_collage.setStyleSheet("font-weight: bold; background: #2e1c4e; color: #e1bee7; border: 1px solid #ba68c8; padding: 6px; border-radius: 4px;")
        self.btn_open_collage.setToolTip("Cambia a la pestaña dedicada de Collage de Eclipses transfiriendo la serie de fotos")
        self.btn_open_collage.clicked.connect(self._on_open_collage_clicked)
        export_layout.addWidget(self.btn_open_collage)

        left_layout.addWidget(self.sec_export)

        scroll_area.setWidget(panel_content)
        left_panel_layout.addWidget(scroll_area, stretch=1)

        # Consola de estado y Progreso fija abajo (siempre visible, sin scroll)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        left_panel_layout.addWidget(self.progress_bar)

        lbl_log = QLabel("Registro de Actividad:")
        lbl_log.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold;")
        left_panel_layout.addWidget(lbl_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(100)
        left_panel_layout.addWidget(self.txt_log)

        splitter.addWidget(left_panel)

        # -------------------------------------------------------------
        # 2. PANEL DERECHO: Canvas Interactivo
        # -------------------------------------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)

        canvas_bar = QHBoxLayout()
        self.btn_reset_zoom = QPushButton("Ajustar Vista")
        self.btn_reset_zoom.setIcon(get_icon("maximize", color="#80d8ff"))
        self.btn_reset_zoom.clicked.connect(self._on_reset_zoom_clicked)

        self.lbl_mode_badge = QLabel("● Modo: Solar (HDR)")
        self.lbl_mode_badge.setStyleSheet(
            "background-color: #141724; border: 1px solid #ff4081; border-radius: 4px; padding: 2px 8px; color: #ff4081; font-weight: bold; font-size: 11px;"
        )

        self.lbl_canvas_status = QLabel("Listo.")
        self.lbl_canvas_status.setStyleSheet("color: #90a4ae; font-size: 11px;")
        canvas_bar.addWidget(self.btn_reset_zoom)
        canvas_bar.addWidget(self.lbl_mode_badge)
        canvas_bar.addStretch()
        canvas_bar.addWidget(self.lbl_canvas_status)
        right_layout.addLayout(canvas_bar)

        self.canvas = EclipseCanvas(self)
        self.canvas.center_clicked.connect(self._on_center_picked)
        self.canvas.files_dropped.connect(self.add_bracketing_paths)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        splitter.setSizes([520, 1400])

        main_layout.addWidget(splitter)
        self.log("Módulo de Eclipses inicializado con escáner subpíxel y compuerta radial anti-ruido.")

    def _apply_styles(self) -> None:
        """Los estilos base provienen de gui.theme (Cosmic Dark Pro)."""
        pass

    def log(self, message: str):
        self.txt_log.append(f"• {message}")
        self.txt_log.moveCursor(QTextCursor.End)
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
        show = self.chk_show_guide.isChecked()
        self.canvas.set_guide_params(cx, cy, r, show=show)

    # -------------------------------------------------------------
    # Lógica de Bracketing y Limbo Individual
    # -------------------------------------------------------------
    def _on_mode_changed(self, mode_str: str = None):
        is_solar = self.mode_toggle.is_solar()
        self.grp_nrgf.setVisible(is_solar)
        mode_label = "Solar" if is_solar else "Lunar"
        self.log(f"Modo cambiado a: Eclipse {mode_label}")

        if is_solar:
            self.grp_hdr.setTitle("3. Fusión HDR (Lineal 32-bit)")
            self.slider_asinh.setValue(35)
            if hasattr(self, "lbl_mode_badge"):
                self.lbl_mode_badge.setText("● Modo: Solar (HDR)")
                self.lbl_mode_badge.setStyleSheet(
                    "background-color: #141724; border: 1px solid #ff4081; border-radius: 4px; padding: 2px 8px; color: #ff4081; font-weight: bold; font-size: 11px;"
                )
        else:
            self.grp_hdr.setTitle("3. Fusión HDR (Mertens Multiescala)")
            self.slider_asinh.setValue(0)
            if hasattr(self, "lbl_mode_badge"):
                self.lbl_mode_badge.setText("● Modo: Lunar (Mertens)")
                self.lbl_mode_badge.setStyleSheet(
                    "background-color: #141724; border: 1px solid #00e5ff; border-radius: 4px; padding: 2px 8px; color: #00e5ff; font-weight: bold; font-size: 11px;"
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
                QApplication.processEvents()

        self.progress_bar.setFormat("Detectando limbo en toma de referencia...")
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
        self._refresh_table()
        self.current_preview_img = None
        self.raw_fused_hdr = None
        self.fused_hdr_img = None
        self.final_filtered_img = None
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
                item_ref.setForeground(QColor("#4CAF50"))
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
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Checked)
            self.table_brack.setItem(row, 6, chk_item)

        self.table_brack.blockSignals(False)

        if 0 <= prev_idx < len(self.bracketing_files):
            self.table_brack.blockSignals(True)
            self.table_brack.selectRow(prev_idx)
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
            self._recalculate_shifts()
            self._refresh_table()
            self.table_brack.selectRow(idx)
            self.log(f"Toma de referencia establecida: {self.bracketing_files[idx]['filename']}")

    def _on_table_selection_changed(self):
        rows = self.table_brack.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self.bracketing_files):
            item = self.bracketing_files[idx]
            filepath = item["filepath"]
            try:
                img = load_image_as_float32(filepath)
                self.current_preview_img = img
                self.canvas.set_image(img)

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

        self._set_geom_spin_values(fcx, fcy, fr)
        self._recalculate_shifts()
        self._refresh_table()
        self.log(f"Limbo refinado a subpíxel: ({fcx:.1f}, {fcy:.1f}), R={fr:.1f}px")

    def _on_detect_all_clicked(self):
        """Lanza la detección automática del limbo en todas las tomas del bracketing."""
        if not self.bracketing_files:
            QMessageBox.warning(self, "Atención", "Carga tomas de bracketing antes de detectar el limbo.")
            return

        mode = "solar" if self.rb_solar.isChecked() else "lunar"
        files = [f["filepath"] for f in self.bracketing_files]

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_detect_all.setEnabled(False)

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
        rows = self.table_brack.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            r = self.spin_radius.value()
            self.bracketing_files[idx]["cx"] = cx
            self.bracketing_files[idx]["cy"] = cy
            self._set_geom_spin_values(cx, cy, r)
            self._recalculate_shifts()
            self._refresh_table()

        self.btn_pick_center.setChecked(False)
        self.log(f"Centro establecido por clic en: ({cx:.1f}, {cy:.1f})")

    def _on_geom_spin_changed(self):
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

    def _on_reset_zoom_clicked(self):
        self.canvas.reset_view()
        self.canvas.update()

    def _on_asinh_slider_changed(self, val: int):
        factor = float(val)
        self.lbl_asinh_val.setText(f"{int(factor)}x" if factor > 0 else "Lineal (1x)")

        if self.raw_fused_hdr is not None:
            self.fused_hdr_img = apply_asinh_stretch(self.raw_fused_hdr, stretch_factor=factor) if factor > 1.0 else self.raw_fused_hdr.copy()
            self.canvas.set_image(self.fused_hdr_img)
            self._update_canvas_guide()

    def _on_fuse_hdr_clicked(self):
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
        mode = "solar" if self.rb_solar.isChecked() else "lunar"

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_fuse_hdr.setEnabled(False)

        params = {
            "files": active_files,
            "exposures": active_exposures,
            "centers": active_centers,
            "ref_idx": active_ref_idx,
            "auto_align": auto_align,
            "subtract_sky": sub_sky,
            "sat_thresh": sat_thresh,
            "asinh_factor": asinh_factor,
            "mode": mode
        }

        self.worker = EclipseWorker("hdr_fusion", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_filter_algo_changed(self):
        is_mgn = self.rb_filter_mgn.isChecked()
        self.container_mgn.setVisible(is_mgn)
        self.container_nrgf.setVisible(not is_mgn)

    def _on_apply_filters_clicked(self):
        base_img = self.fused_hdr_img if self.fused_hdr_img is not None else self.current_preview_img
        if base_img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen cargada o fusionada sobre la cual aplicar filtros.")
            return

        ref_f = self.bracketing_files[self.ref_file_idx] if self.bracketing_files else None
        if ref_f and ref_f.get("cx") is not None:
            cx, cy, r = ref_f["cx"], ref_f["cy"], ref_f["radius"]
        else:
            cx = self.spin_cx.value()
            cy = self.spin_cy.value()
            r = self.spin_radius.value()

        if r <= 5:
            QMessageBox.warning(self, "Atención", "Define o auto-detecta el radio del limbo antes de aplicar los filtros.")
            return

        is_mgn = self.rb_filter_mgn.isChecked()
        filter_type = "mgn" if is_mgn else "nrgf"
        bin_px = self.spin_bin_px.value()
        robust = self.rb_robust.isChecked()
        blend = (self.slider_blend.value() if is_mgn else self.slider_blend_nrgf.value()) / 100.0
        reach = self.slider_reach.value() / 10.0
        mgn_k = self.slider_mgn_k.value() / 100.0
        mgn_gamma = self.slider_mgn_gamma.value() / 10.0
        use_hp = self.chk_highpass.isChecked()
        hp_strength = (self.slider_hp.value() / 100.0) if use_hp else 0.0

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_apply_filters.setEnabled(False)

        params = {
            "image": base_img,
            "cx": cx,
            "cy": cy,
            "radius": r,
            "filter_type": filter_type,
            "mgn_k": mgn_k,
            "mgn_gamma": mgn_gamma,
            "bin_px": bin_px,
            "robust": robust,
            "blend": blend,
            "reach": reach,
            "use_highpass": use_hp,
            "hp_strength": hp_strength
        }

        self.worker = EclipseWorker("apply_filters", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, val: int, msg: str):
        self.progress_bar.setValue(val)
        self.log(msg)

    def _on_worker_finished(self, result: object, task_type: str):
        self.progress_bar.setVisible(False)
        self.btn_detect_all.setEnabled(True)
        self.btn_fuse_hdr.setEnabled(True)
        self.btn_apply_filters.setEnabled(True)

        if task_type == "batch_detect":
            for i, (cx, cy, r) in enumerate(result):
                if i < len(self.bracketing_files):
                    self.bracketing_files[i]["cx"] = cx
                    self.bracketing_files[i]["cy"] = cy
                    self.bracketing_files[i]["radius"] = r
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
                    self.canvas.set_image(img)
                except Exception:
                    pass
                self._update_canvas_guide()

            self.log("Detección de limbo en lote completada. Desplazamientos calculados.")

        elif task_type == "hdr_fusion":
            self.raw_fused_hdr = result["hdr_raw"]
            self.fused_hdr_img = result["hdr_image"]
            self.canvas.set_image(self.fused_hdr_img)

            centers = result["centers"]
            shifts = result["shifts"]
            for i, c in enumerate(centers):
                if i < len(self.bracketing_files):
                    self.bracketing_files[i]["cx"] = c[0]
                    self.bracketing_files[i]["cy"] = c[1]
                    self.bracketing_files[i]["radius"] = c[2]
                    self.bracketing_files[i]["shift_x"] = shifts[i][0]
                    self.bracketing_files[i]["shift_y"] = shifts[i][1]

            self._refresh_table()

            ref_f = self.bracketing_files[self.ref_file_idx]
            if ref_f.get("cx") is not None:
                self._set_geom_spin_values(ref_f["cx"], ref_f["cy"], ref_f["radius"])

            self.log("Fusión HDR finalizada con alineación subpíxel y estirado Asinh.")

        elif task_type == "apply_filters":
            self.final_filtered_img = result
            self.canvas.set_image(result)
            self._update_canvas_guide()
            self.log("Filtros de corona aplicados con compuerta radial (fondo limpio).")

    def _on_worker_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.btn_detect_all.setEnabled(True)
        self.btn_fuse_hdr.setEnabled(True)
        self.btn_apply_filters.setEnabled(True)
        QMessageBox.critical(self, "Error de Procesado", err_msg)
        self.log(f"ERROR: {err_msg}")

    def _get_best_current_image(self) -> np.ndarray:
        if self.final_filtered_img is not None:
            return self.final_filtered_img
        if self.fused_hdr_img is not None:
            return self.fused_hdr_img
        return self.current_preview_img

    def _on_save_tiff_clicked(self, bit_depth: int = 16):
        img = self._get_best_current_image()
        if img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen disponible para guardar.")
            return

        default_name = f"eclipse_procesado_{bit_depth}bit.tif"
        if bit_depth == 16:
            filters = "TIFF 16-bit (*.tif *.tiff);;TIFF 32-bit (*.tif *.tiff);;Todos (*.*)"
            title = "Guardar Imagen de Eclipse (TIFF 16-bit)"
        else:
            filters = "TIFF 32-bit (*.tif *.tiff);;TIFF 16-bit (*.tif *.tiff);;Todos (*.*)"
            title = "Guardar Imagen de Eclipse (TIFF 32-bit Float)"

        path, selected_filter = QFileDialog.getSaveFileName(self, title, default_name, filters)
        if not path:
            return

        # Si el usuario seleccionó un filtro diferente en el diálogo, respetar su elección
        is_32 = ("32-bit" in selected_filter) if selected_filter else (bit_depth == 32)
        photometric = 'minisblack' if img.ndim == 2 else 'rgb'

        try:
            if is_32:
                out_img = np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)
                desc = "TIFF 32-bit float (HDR)"
            else:
                out_img = np.ascontiguousarray(np.clip(img * 65535.0 + 0.5, 0.0, 65535.0), dtype=np.uint16)
                desc = "TIFF 16-bit entero (Photoshop / Lightroom)"

            save_tiff_with_icc(path, out_img, compression='zlib', photometric=photometric)
            self.log(f"Imagen guardada exitosamente ({desc} con perfil ICC): {path}")
            QMessageBox.information(
                self, "Exportación Completada",
                f"Imagen guardada con éxito en:\n{path}\n\nFormato: {desc} ({out_img.shape[1]}x{out_img.shape[0]} px)"
            )
        except Exception as e:
            self.log(f"ERROR al guardar imagen: {e}")
            QMessageBox.critical(self, "Error de Exportación", f"No se pudo guardar la imagen:\n{e}")

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

    @property
    def mode(self) -> str:
        if hasattr(self, 'rb_solar') and self.rb_solar is not None:
            return "solar" if self.rb_solar.isChecked() else "lunar"
        return "solar"

    def _on_open_collage_clicked(self):
        """Transfiere los archivos a la pestaña dedicada de Collage y cambia a ella."""
        initial_fps = [item["filepath"] for item in self.bracketing_files] if hasattr(self, 'bracketing_files') and self.bracketing_files else []
        self.open_collage_requested.emit(initial_fps, self.mode)



