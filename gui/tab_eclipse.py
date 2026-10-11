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
    QSizePolicy, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QThread, QPoint, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QCursor, QFont, QTextCursor

from core.eclipse import (
    extract_exposure_info,
    detect_eclipse_disk,
    refine_disk_subpixel,
    register_frame_to_center,
    align_eclipse_bracketing,
    fuse_hdr_bracketing,
    apply_asinh_stretch,
    apply_nrgf,
    apply_coronal_highpass
)
from core.stacking import load_image_as_float32
from core.icc_manager import save_tiff_with_icc


class ToggleSwitch(QWidget):
    """
    Interruptor deslizante moderno tipo píldora (Pill Toggle Switch).
    checked = False -> Posición Izquierda (Solar: rosa/magenta)
    checked = True  -> Posición Derecha (Lunar: cyan/azul)
    """
    toggled = Signal(bool)

    def __init__(self, parent=None, checked: bool = False):
        super().__init__(parent)
        self._checked = checked
        self.setFixedSize(52, 26)
        self.setCursor(Qt.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self.update()
            self.toggled.emit(self._checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setChecked(not self._checked)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        radius = h / 2.0
        knob_radius = radius - 3.2

        if not self._checked:
            # Solar: Rosa/Magenta estilizado como en el diseño de referencia
            bg_brush = QColor(255, 64, 129, 35)
            border_pen = QPen(QColor(255, 64, 129, 230), 2.2)
            knob_brush = QColor(255, 64, 129)
            knob_x = radius
        else:
            # Lunar: Cyan brillante estilizado como en el diseño de referencia
            bg_brush = QColor(0, 229, 255, 35)
            border_pen = QPen(QColor(0, 229, 255, 230), 2.2)
            knob_brush = QColor(0, 229, 255)
            knob_x = w - radius

        # Track exterior redondeado
        painter.setPen(border_pen)
        painter.setBrush(bg_brush)
        painter.drawRoundedRect(QRectF(1.5, 1.5, w - 3, h - 3), radius, radius)

        # Círculo / Knob interior deslizante
        painter.setPen(Qt.NoPen)
        painter.setBrush(knob_brush)
        painter.drawEllipse(QPointF(knob_x, h / 2.0), knob_radius, knob_radius)


class EclipseModeToggle(QWidget):
    """
    Selector compacto de modalidad Sol/Luna en formato botón boolean (Pill Toggle).
    Ocupa mínimo espacio vertical (~34px) ahorrando espacio en la interfaz.
    """
    mode_changed = Signal(str)  # "solar" o "lunar"

    def __init__(self, parent=None, is_solar: bool = True):
        super().__init__(parent)
        self._is_solar = is_solar

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        self.lbl_solar = QLabel("☀️ Eclipse Solar")
        self.lbl_solar.setCursor(Qt.PointingHandCursor)
        self.lbl_solar.mousePressEvent = lambda e: self.set_solar(True)

        self.toggle_btn = ToggleSwitch(checked=not is_solar)
        self.toggle_btn.toggled.connect(self._on_switch_toggled)

        self.lbl_lunar = QLabel("🌕 Eclipse Lunar")
        self.lbl_lunar.setCursor(Qt.PointingHandCursor)
        self.lbl_lunar.mousePressEvent = lambda e: self.set_solar(False)

        layout.addStretch()
        layout.addWidget(self.lbl_solar)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.lbl_lunar)
        layout.addStretch()

        self.setStyleSheet("""
            EclipseModeToggle {
                background-color: #1a1a22;
                border: 1px solid #2d2d3a;
                border-radius: 6px;
            }
        """)

        self._update_styles()

    def is_solar(self) -> bool:
        return self._is_solar

    def isChecked(self) -> bool:
        return self._is_solar

    def set_solar(self, solar: bool):
        if self._is_solar != solar:
            self._is_solar = solar
            self.toggle_btn.setChecked(not solar)
            self._update_styles()
            self.mode_changed.emit("solar" if solar else "lunar")

    def _on_switch_toggled(self, checked: bool):
        solar = not checked
        if self._is_solar != solar:
            self._is_solar = solar
            self._update_styles()
            self.mode_changed.emit("solar" if solar else "lunar")

    def _update_styles(self):
        if self._is_solar:
            self.lbl_solar.setStyleSheet("font-weight: bold; font-size: 12px; color: #ff4081;")
            self.lbl_lunar.setStyleSheet("font-weight: normal; font-size: 12px; color: #78909c;")
        else:
            self.lbl_solar.setStyleSheet("font-weight: normal; font-size: 12px; color: #78909c;")
            self.lbl_lunar.setStyleSheet("font-weight: bold; font-size: 12px; color: #00e5ff;")


class EclipseCanvas(QWidget):
    """
    Lienzo interactivo optimizado para eclipses.
    Permite zoom con rueda, paneo con arrastre, visualización del disco/limbo
    y selección interactiva del centro haciendo clic.
    """
    center_clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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
            return

        # Dibujar imagen con zoom y pan
        iw = self.base_pixmap.width() * self.zoom_level
        ih = self.base_pixmap.height() * self.zoom_level
        target_rect = QRectF(self.pan_offset.x(), self.pan_offset.y(), iw, ih)
        painter.drawPixmap(target_rect.toRect(), self.base_pixmap)

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

                self.progress.emit(5, "Iniciando detección subpíxel de limbo en lote...")

                # 1. Detectar primero la toma de referencia para anclar el radio físico esperado
                ref_img = load_image_as_float32(files[ref_idx])
                ref_cx, ref_cy, ref_r = detect_eclipse_disk(ref_img, mode=mode, expected_radius=None)
                detections[ref_idx] = (ref_cx, ref_cy, ref_r)
                expected_r = ref_r if ref_r > 10 else None

                # 2. Detectar el resto de tomas usando expected_r
                for idx, path in enumerate(files):
                    if idx == ref_idx:
                        continue
                    self.progress.emit(
                        int(10 + (idx / total) * 85),
                        f"Escaneando limbo ({idx+1}/{total}): {os.path.basename(path)}"
                    )
                    img = load_image_as_float32(path)
                    cx, cy, r = detect_eclipse_disk(img, mode=mode, expected_radius=expected_r)
                    detections[idx] = (cx, cy, r)

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
                loaded_images = []
                self.progress.emit(5, "Iniciando carga de tomas de bracketing...")

                final_centers = []
                expected_r = None

                for idx, path in enumerate(files):
                    self.progress.emit(
                        int(5 + (idx / total) * 40),
                        f"Cargando ({idx+1}/{total}): {os.path.basename(path)}"
                    )
                    img = load_image_as_float32(path)
                    loaded_images.append(img)

                    if centers is not None and idx < len(centers) and centers[idx] is not None and centers[idx][2] > 5:
                        final_centers.append(centers[idx])
                        if expected_r is None and centers[idx][2] > 10:
                            expected_r = centers[idx][2]
                    else:
                        cx, cy, r = detect_eclipse_disk(img, mode=mode, expected_radius=expected_r)
                        if expected_r is None and r > 10:
                            expected_r = r
                        final_centers.append((cx, cy, r))

                if auto_align and len(final_centers) == len(loaded_images):
                    self.progress.emit(50, "Alineando tomas por centro de limbo (interpolación Lanczos4)...")
                    center_pts = [(c[0], c[1]) for c in final_centers]
                    aligned_images, shifts = align_eclipse_bracketing(
                        loaded_images,
                        center_pts,
                        ref_idx=ref_idx
                    )
                else:
                    aligned_images = loaded_images
                    shifts = [(0.0, 0.0)] * len(loaded_images)

                progress_msg = "Calculando ponderaciones y fusionando flujo fotométrico HDR..." if mode == "solar" else "Fusionando tomas con exposición multiescala (Mertens HDR)..."
                self.progress.emit(75, progress_msg)
                hdr_raw = fuse_hdr_bracketing(
                    aligned_images,
                    exposures,
                    saturation_threshold=sat_thresh,
                    asinh_stretch=0.0,
                    mode=mode
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
                bin_px = self.params.get("bin_px", 2.0)
                robust = self.params.get("robust", True)
                blend = self.params.get("blend", 0.55)
                reach = self.params.get("reach", 2.6)
                use_highpass = self.params.get("use_highpass", False)
                hp_strength = self.params.get("hp_strength", 0.5)

                self.progress.emit(20, "Calculando filtro NRGF limpio (con compuerta radial anti-ruido)...")
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

                self.progress.emit(100, "Filtros de corona aplicados con fondo 100% limpio.")
                self.finished.emit(filtered, "apply_filters")

        except Exception as e:
            self.error.emit(f"Error durante el procesado: {str(e)}")


class EclipseTab(QWidget):
    """Pestaña interactiva de Eclipses (Solar y Lunar)."""
    export_to_developer = Signal(str)

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

        # GRUPO 2: Serie de Bracketing y Alineación
        grp_brack = QGroupBox("2. Serie de Bracketing")
        brack_layout = QVBoxLayout(grp_brack)
        brack_layout.setContentsMargins(10, 14, 10, 10)
        brack_layout.setSpacing(8)

        btn_row_files = QHBoxLayout()
        self.btn_load_brack = QPushButton("📁 Cargar Bracketing...")
        self.btn_load_brack.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_load_brack.clicked.connect(self._on_load_bracketing_clicked)
        self.btn_clear_brack = QPushButton("🗑️ Limpiar")
        self.btn_clear_brack.setStyleSheet("font-weight: bold; color: #ff8a80;")
        self.btn_clear_brack.clicked.connect(self._on_clear_bracketing_clicked)
        btn_row_files.addWidget(self.btn_load_brack)
        btn_row_files.addWidget(self.btn_clear_brack)
        brack_layout.addLayout(btn_row_files)

        self.table_brack = QTableWidget(0, 7)
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
        brack_layout.addWidget(self.table_brack)

        btn_align_row = QHBoxLayout()
        self.btn_detect_all = QPushButton("⚙️ Detectar Limbo en Todo el Lote")
        self.btn_detect_all.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_detect_all.setToolTip("Analiza el limbo con escáner radial subpíxel en cada imagen")
        self.btn_detect_all.clicked.connect(self._on_detect_all_clicked)

        self.btn_set_ref = QPushButton("📌 Marcar REF")
        self.btn_set_ref.setStyleSheet("font-weight: bold; color: #a5d6a7;")
        self.btn_set_ref.setToolTip("Establece la toma seleccionada como centro de referencia de alineación")
        self.btn_set_ref.clicked.connect(self._on_set_ref_clicked)

        btn_align_row.addWidget(self.btn_detect_all)
        btn_align_row.addWidget(self.btn_set_ref)
        brack_layout.addLayout(btn_align_row)

        left_layout.addWidget(grp_brack)

        # GRUPO 3: Geometría del Disco (Limbo de la Toma Seleccionada)
        grp_geom = QGroupBox("3. Geometría del Disco (Toma Seleccionada)")
        geom_layout = QVBoxLayout(grp_geom)
        geom_layout.setContentsMargins(10, 14, 10, 10)
        geom_layout.setSpacing(8)

        btn_auto_row = QHBoxLayout()
        self.btn_detect_disk = QPushButton("🎯 Auto-detectar")
        self.btn_detect_disk.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_detect_disk.setToolTip("Detección global del disco lunar/solar")
        self.btn_detect_disk.clicked.connect(self._on_auto_detect_clicked)

        self.btn_refine_subpixel = QPushButton("🔍 Ajuste Fino Subpíxel")
        self.btn_refine_subpixel.setStyleSheet("color: #b388ff; font-weight: bold;")
        self.btn_refine_subpixel.setToolTip("Ajusta el círculo exactamente al borde del limbo ignorando protuberancias")
        self.btn_refine_subpixel.clicked.connect(self._on_refine_subpixel_clicked)

        self.btn_pick_center = QPushButton("👆 Clic Canvas")
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

        left_layout.addWidget(grp_geom)

        # GRUPO 4: Fusión HDR (Bracketing y Estirado Asinh)
        self.grp_hdr = QGroupBox("4. Fusión HDR (Lineal 32-bit)")
        hdr_layout = QVBoxLayout(self.grp_hdr)
        hdr_layout.setContentsMargins(10, 14, 10, 10)
        hdr_layout.setSpacing(8)

        self.chk_auto_align = QCheckBox("Alinear tomas por limbo antes de fusionar")
        self.chk_auto_align.setChecked(True)
        self.chk_auto_align.setStyleSheet("font-weight: bold; color: #b9f6ca;")
        hdr_layout.addWidget(self.chk_auto_align)

        row_sat = QHBoxLayout()
        row_sat.addWidget(QLabel("Corte de Saturación:"))
        self.lbl_sat_val = QLabel("92%")
        self.lbl_sat_val.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_sat.addWidget(self.lbl_sat_val)
        hdr_layout.addLayout(row_sat)

        self.slider_sat = QSlider(Qt.Horizontal)
        self.slider_sat.setRange(70, 98)
        self.slider_sat.setValue(92)
        self.slider_sat.valueChanged.connect(lambda v: self.lbl_sat_val.setText(f"{v}%"))
        hdr_layout.addWidget(self.slider_sat)

        row_asinh = QHBoxLayout()
        row_asinh.addWidget(QLabel("Compresión Tonal (Asinh):"))
        self.lbl_asinh_val = QLabel("35x")
        self.lbl_asinh_val.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_asinh.addWidget(self.lbl_asinh_val)
        hdr_layout.addLayout(row_asinh)

        self.slider_asinh = QSlider(Qt.Horizontal)
        self.slider_asinh.setRange(0, 100)
        self.slider_asinh.setValue(35)
        self.slider_asinh.setToolTip("Levanta suavemente la corona exterior tenue sin quemar el centro ni las protuberancias")
        self.slider_asinh.valueChanged.connect(self._on_asinh_slider_changed)
        hdr_layout.addWidget(self.slider_asinh)

        self.btn_fuse_hdr = QPushButton("⚡ FUSIONAR BRACKETING HDR")
        self.btn_fuse_hdr.setFixedHeight(40)
        self.btn_fuse_hdr.setCursor(Qt.PointingHandCursor)
        self.btn_fuse_hdr.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 12px;
                background-color: #2e7d32;
                color: white;
                border-radius: 6px;
                border: 1px solid #388e3c;
            }
            QPushButton:hover {
                background-color: #388e3c;
                border-color: #4caf50;
            }
            QPushButton:pressed {
                background-color: #1b5e20;
            }
        """)
        self.btn_fuse_hdr.clicked.connect(self._on_fuse_hdr_clicked)
        hdr_layout.addWidget(self.btn_fuse_hdr)
        left_layout.addWidget(self.grp_hdr)

        # GRUPO 5: Filtros de Corona Solar (NRGF Limpio)
        self.grp_nrgf = QGroupBox("5. Filtros de Corona Solar (NRGF Limpio)")
        nrgf_layout = QVBoxLayout(self.grp_nrgf)
        nrgf_layout.setContentsMargins(10, 14, 10, 10)
        nrgf_layout.setSpacing(8)

        row_bin = QHBoxLayout()
        row_bin.addWidget(QLabel("Ancho de anillo radial (px):"))
        self.spin_bin_px = QDoubleSpinBox()
        self.spin_bin_px.setRange(0.5, 20.0)
        self.spin_bin_px.setValue(2.0)
        self.spin_bin_px.setSingleStep(0.5)
        row_bin.addWidget(self.spin_bin_px)
        nrgf_layout.addLayout(row_bin)

        self.rb_robust = QRadioButton("Mediana robusta (1.4826·MAD)")
        self.rb_robust.setChecked(True)
        self.rb_mean = QRadioButton("Media estándar (Gauss)")
        nrgf_layout.addWidget(self.rb_robust)
        nrgf_layout.addWidget(self.rb_mean)

        row_reach = QHBoxLayout()
        row_reach.addWidget(QLabel("Alcance Radial de Corona:"))
        self.lbl_reach_val = QLabel("3.5x")
        self.lbl_reach_val.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_reach.addWidget(self.lbl_reach_val)
        nrgf_layout.addLayout(row_reach)

        self.slider_reach = QSlider(Qt.Horizontal)
        self.slider_reach.setRange(12, 50)
        self.slider_reach.setValue(35)
        self.slider_reach.setToolTip("Extiende la acción del filtro a lo largo de toda la corona media y externa, suprimiendo al 100% el ruido del cielo negro")
        self.slider_reach.valueChanged.connect(lambda v: self.lbl_reach_val.setText(f"{v/10.0:.1f}x"))
        nrgf_layout.addWidget(self.slider_reach)

        row_blend = QHBoxLayout()
        row_blend.addWidget(QLabel("Fuerza de Filamentos (Mezcla):"))
        self.lbl_blend_val = QLabel("50%")
        self.lbl_blend_val.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_blend.addWidget(self.lbl_blend_val)
        nrgf_layout.addLayout(row_blend)

        self.slider_blend = QSlider(Qt.Horizontal)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(50)
        self.slider_blend.setToolTip("Aumenta el estiramiento y contraste de los filamentos magnéticos ('hilos') de la corona")
        self.slider_blend.valueChanged.connect(lambda v: self.lbl_blend_val.setText(f"{v}%"))
        nrgf_layout.addWidget(self.slider_blend)

        self.chk_highpass = QCheckBox("Realce de hilos finos (Filtro bilateral tangencial)")
        self.chk_highpass.setChecked(True)
        nrgf_layout.addWidget(self.chk_highpass)

        row_hp = QHBoxLayout()
        row_hp.addWidget(QLabel("Nitidez de hilos:"))
        self.lbl_hp_val = QLabel("50%")
        self.lbl_hp_val.setStyleSheet("color: #b388ff; font-weight: bold;")
        row_hp.addWidget(self.lbl_hp_val)
        nrgf_layout.addLayout(row_hp)

        self.slider_hp = QSlider(Qt.Horizontal)
        self.slider_hp.setRange(0, 100)
        self.slider_hp.setValue(50)
        self.slider_hp.valueChanged.connect(lambda v: self.lbl_hp_val.setText(f"{v}%"))
        nrgf_layout.addWidget(self.slider_hp)

        self.btn_apply_filters = QPushButton("✨ APLICAR FILTROS DE CORONA (NRGF)")
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

        # GRUPO 6: Exportación y Conexión
        grp_export = QGroupBox("6. Exportación")
        export_layout = QVBoxLayout(grp_export)
        export_layout.setContentsMargins(10, 14, 10, 10)
        export_layout.setSpacing(8)

        btn_tiff_row = QHBoxLayout()
        self.btn_save_tiff_16 = QPushButton("💾 Guardar TIFF 16-bit...")
        self.btn_save_tiff_16.setStyleSheet("font-weight: bold; color: #80cbc4;")
        self.btn_save_tiff_16.setToolTip("Exporta en TIFF 16-bit entero compatible directamente con Photoshop, Lightroom y editores estándar sin conversiones extrañas.")
        self.btn_save_tiff_16.clicked.connect(lambda: self._on_save_tiff_clicked(bit_depth=16))

        self.btn_save_tiff_32 = QPushButton("💾 Guardar TIFF 32-bit...")
        self.btn_save_tiff_32.setStyleSheet("font-weight: bold; color: #90caf9;")
        self.btn_save_tiff_32.setToolTip("Exporta en TIFF 32-bit float con el rango dinámico HDR lineal completo.")
        self.btn_save_tiff_32.clicked.connect(lambda: self._on_save_tiff_clicked(bit_depth=32))

        self.btn_save_tiff = self.btn_save_tiff_32  # Compatibilidad

        btn_tiff_row.addWidget(self.btn_save_tiff_16)
        btn_tiff_row.addWidget(self.btn_save_tiff_32)
        export_layout.addLayout(btn_tiff_row)

        self.btn_send_to_dev = QPushButton("➡️ Enviar al Revelador / Editor")
        self.btn_send_to_dev.setFixedHeight(38)
        self.btn_send_to_dev.setCursor(Qt.PointingHandCursor)
        self.btn_send_to_dev.setStyleSheet("font-weight: bold; color: #ffcc80; padding: 6px;")
        self.btn_send_to_dev.clicked.connect(self._on_send_to_developer_clicked)
        export_layout.addWidget(self.btn_send_to_dev)

        left_layout.addWidget(grp_export)

        scroll_area.setWidget(panel_content)
        left_panel_layout.addWidget(scroll_area, stretch=1)

        # Consola de estado y Progreso fija abajo (siempre visible, sin scroll)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2d2d3a;
                border-radius: 3px;
                text-align: center;
                font-size: 10px;
                background-color: #141418;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0288d1;
            }
        """)
        self.progress_bar.setVisible(False)
        left_panel_layout.addWidget(self.progress_bar)

        lbl_log = QLabel("Registro de Actividad:")
        lbl_log.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold;")
        left_panel_layout.addWidget(lbl_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(100)
        self.txt_log.setStyleSheet(
            "background-color: #141418; color: #a5d6a7; "
            "font-family: Consolas, monospace; font-size: 11px; "
            "border: 1px solid #2d2d3a; border-radius: 4px; padding: 4px;"
        )
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
        self.btn_reset_zoom.clicked.connect(self._on_reset_zoom_clicked)
        self.lbl_canvas_status = QLabel("Listo.")
        self.lbl_canvas_status.setStyleSheet("color: #90a4ae; font-size: 11px;")
        canvas_bar.addWidget(self.btn_reset_zoom)
        canvas_bar.addStretch()
        canvas_bar.addWidget(self.lbl_canvas_status)
        right_layout.addLayout(canvas_bar)

        self.canvas = EclipseCanvas(self)
        self.canvas.center_clicked.connect(self._on_center_picked)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        splitter.setSizes([520, 1400])

        main_layout.addWidget(splitter)
        self.log("Módulo de Eclipses inicializado con escáner subpíxel y compuerta radial anti-ruido.")

    def _apply_styles(self) -> None:
        """Aplica la hoja de estilos unificada consistente con el resto de módulos."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #2d2d3a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 14px;
                background-color: #1a1a22;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 4px;
                color: #90caf9;
            }
            QPushButton {
                background-color: #252530;
                border: 1px solid #3a3a4c;
                border-radius: 4px;
                padding: 5px 10px;
                color: #e0e0e0;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2f2f3d;
                border-color: #55556a;
            }
            QPushButton:pressed {
                background-color: #1c1c24;
            }
            QTableWidget {
                background-color: #14141a;
                border: 1px solid #262633;
                border-radius: 4px;
                color: #ffffff;
                gridline-color: #22222e;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 3px;
            }
            QTableWidget::item:selected {
                background-color: #1b384d;
                color: #80d8ff;
            }
            QHeaderView::section {
                background-color: #1a1a24;
                color: #90caf9;
                padding: 4px;
                border: 1px solid #2d2d3a;
                font-weight: bold;
                font-size: 11px;
            }
            QComboBox {
                background-color: #121216;
                border: 1px solid #33333f;
                border-radius: 4px;
                padding: 4px 6px;
                color: #f0f0f0;
            }
            QComboBox:focus {
                border: 1px solid #0288d1;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a22;
                selection-background-color: #1976d2;
                color: #ffffff;
                border: 1px solid #3a3a4c;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #121216;
                border: 1px solid #33333f;
                border-radius: 4px;
                padding: 3px 6px;
                color: #f0f0f0;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #0288d1;
            }
            QSlider::groove:horizontal {
                border: 1px solid #2d2d3a;
                height: 4px;
                background: #14141a;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #0288d1;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #80d8ff;
                border: 1px solid #0288d1;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QCheckBox, QRadioButton {
                color: #e0e0e0;
                font-size: 11px;
                spacing: 6px;
            }
            QScrollBar:vertical {
                background: #121216;
                width: 8px;
                margin: 0;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2f2f3d;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3d3d52;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

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
            self.grp_hdr.setTitle("4. Fusión HDR (Lineal 32-bit)")
            self.slider_asinh.setValue(35)
        else:
            self.grp_hdr.setTitle("4. Fusión HDR (Mertens Multiescala)")
            self.slider_asinh.setValue(0)

    def _on_load_bracketing_clicked(self):
        filters = "Imágenes Astrofotográficas (*.arw *.cr2 *.cr3 *.nef *.dng *.raw *.tif *.tiff *.fits *.fit *.jpg *.jpeg);;Todos (*.*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Cargar Serie de Bracketing de Eclipse", "", filters)
        if not paths:
            return

        for p in paths:
            info = extract_exposure_info(p)
            info["cx"] = None
            info["cy"] = None
            info["radius"] = None
            info["shift_x"] = 0.0
            info["shift_y"] = 0.0
            info["is_ref"] = False
            self.bracketing_files.append(info)

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
        self.log(f"Se cargaron {len(paths)} tomas de bracketing ordenadas por exposición.")

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
            ref_txt = "✓ REF" if item.get("is_ref", False) else ""
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
            "sat_thresh": sat_thresh,
            "asinh_factor": asinh_factor,
            "mode": mode
        }

        self.worker = EclipseWorker("hdr_fusion", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

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
            QMessageBox.warning(self, "Atención", "Define o auto-detecta el radio del limbo antes de aplicar el NRGF.")
            return

        bin_px = self.spin_bin_px.value()
        robust = self.rb_robust.isChecked()
        blend = self.slider_blend.value() / 100.0
        reach = self.slider_reach.value() / 10.0
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

