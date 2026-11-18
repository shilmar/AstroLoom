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
    QSlider, QCheckBox, QComboBox, QFrame, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QColor, QPainter

from gui.theme import apply_theme, BG_CARD, BORDER_BASE, TEXT_MUTED
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
    """Lienzo optimizado para el visor Blink con escalado manteniendo proporción."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #07090e; border: 1px solid #1a2233; border-radius: 6px;")
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, img_u8: Optional[np.ndarray]):
        if img_u8 is None:
            self._pixmap = None
        else:
            h, w = img_u8.shape[:2]
            img_contig = np.ascontiguousarray(img_u8)
            qimg = QImage(img_contig.data, w, h, w * 3, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07090e"))
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, "Cargando fotograma...")


class BlinkDialog(QDialog):
    """
    Diálogo flotante interactivo para inspección ultrarrápida (Blink) y cribado de tomas.
    """

    exclusion_changed = Signal(str, bool)     # (filepath, is_included)
    selection_applied = Signal(set)           # set de archivos excluidos

    def __init__(
        self,
        file_list: List[str],
        excluded_files: Optional[Set[str]] = None,
        culling_results: Optional[Dict[str, Any]] = None,
        current_index: int = 0,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("AstroLoom - Visor Rápido de Subframes (Blink & Culling)")
        self.resize(1100, 750)
        apply_theme(self)

        self.file_list: List[str] = list(file_list)
        self.excluded_files: Set[str] = set(excluded_files) if excluded_files else set()
        self.culling_results: Dict[str, Any] = culling_results if culling_results else {}
        self.current_idx: int = max(0, min(current_index, len(self.file_list) - 1)) if self.file_list else 0

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

        # Badge de inclusión
        self.lbl_status_badge = QLabel("INCLUIDA")
        self.lbl_status_badge.setStyleSheet("padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(self.lbl_status_badge)

        # Badge de calidad óptica si ya fue analizada
        self.lbl_quality_badge = QLabel("")
        self.lbl_quality_badge.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 6px;")
        h_layout.addWidget(self.lbl_quality_badge)

        h_layout.addStretch()

        self.chk_auto_stretch = QCheckBox("Auto-Estirado (LUT)")
        self.chk_auto_stretch.setChecked(True)
        self.chk_auto_stretch.setToolTip("Aplica un estirado temporal para ver tomas lineales oscuras con brillo óptimo")
        self.chk_auto_stretch.toggled.connect(self._on_stretch_toggled)
        h_layout.addWidget(self.chk_auto_stretch)

        root_layout.addWidget(header)

        # =====================================================================
        # LIENZO CENTRAL
        # =====================================================================
        self.canvas = BlinkCanvas(self)
        root_layout.addWidget(self.canvas, stretch=1)

        # =====================================================================
        # BARRA INFERIOR: Controles de Reproducción y Cribado
        # =====================================================================
        footer = QFrame()
        footer.setStyleSheet(f"background-color: {BG_CARD}; border: 1px solid {BORDER_BASE}; border-radius: 6px;")
        f_layout = QVBoxLayout(footer)
        f_layout.setContentsMargins(8, 6, 8, 6)
        f_layout.setSpacing(6)

        # Fila 1: Deslizador de posición
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, len(self.file_list) - 1))
        self.slider.setValue(self.current_idx)
        self.slider.valueChanged.connect(self._on_slider_changed)
        f_layout.addWidget(self.slider)

        # Fila 2: Botones de transporte y cribado
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

        # Velocidad FPS
        row_ctrl.addWidget(QLabel("Velocidad:"))
        self.combo_fps = QComboBox()
        self.combo_fps.addItems(["1 fps", "2 fps", "5 fps", "10 fps", "15 fps"])
        self.combo_fps.setCurrentIndex(2)  # 5 fps default
        self.combo_fps.currentIndexChanged.connect(self._on_fps_changed)
        row_ctrl.addWidget(self.combo_fps)

        row_ctrl.addStretch()

        # Acciones de Cribado
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
        fps_map = {0: 1, 1: 2, 2: 5, 3: 10, 4: 15}
        fps = fps_map.get(self.combo_fps.currentIndex(), 5)
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
            fps_map = {0: 1, 1: 2, 2: 5, 3: 10, 4: 15}
            fps = fps_map.get(self.combo_fps.currentIndex(), 5)
            self._timer.start(int(1000.0 / fps))
            self.btn_play.setText("⏸ Pausar (Espacio)")
            self.btn_play.setIcon(get_icon("pause", size=13))
            self.btn_play.setStyleSheet("font-weight: bold; color: #fbbf24;")

    def _on_play_step(self):
        if not self.file_list:
            return
        next_idx = (self.current_idx + 1) % len(self.file_list)
        self.set_current_index(next_idx)

    def prev_frame(self):
        if not self.file_list:
            return
        new_idx = max(0, self.current_idx - 1)
        self.set_current_index(new_idx)

    def next_frame(self):
        if not self.file_list:
            return
        new_idx = min(len(self.file_list) - 1, self.current_idx + 1)
        self.set_current_index(new_idx)

    def set_current_index(self, idx: int):
        if idx == self.current_idx or idx < 0 or idx >= len(self.file_list):
            return
        self.current_idx = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self._update_current_frame()
        self._preload_neighbors()

    def _on_slider_changed(self, val: int):
        self.set_current_index(val)

    def _on_stretch_toggled(self, checked: bool):
        self._cache.clear()
        self._update_current_frame()
        self._preload_neighbors()

    def keep_current_frame(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        if path in self.excluded_files:
            self.excluded_files.remove(path)
            self.exclusion_changed.emit(path, True)
        self._update_badges()
        self.next_frame()

    def discard_current_frame(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        if path not in self.excluded_files:
            self.excluded_files.add(path)
            self.exclusion_changed.emit(path, False)
        self._update_badges()
        self.next_frame()

    def _update_badges(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        is_included = path not in self.excluded_files
        if is_included:
            self.lbl_status_badge.setText("✓ INCLUIDA")
            self.lbl_status_badge.setStyleSheet("color: #4ade80; background-color: #132b1e; border: 1px solid #22c55e; border-radius: 4px; padding: 2px 8px; font-weight: bold;")
        else:
            self.lbl_status_badge.setText("✗ DESCARTADA")
            self.lbl_status_badge.setStyleSheet("color: #f87171; background-color: #2a1520; border: 1px solid #ef4444; border-radius: 4px; padding: 2px 8px; font-weight: bold;")

        # Métricas ópticas si están disponibles
        if path in self.culling_results:
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

        self.lbl_title.setText(f"Fotograma {self.current_idx + 1} de {total} : {filename} ({included_count}/{total} activas)")
        self._update_badges()

        # Obtener imagen de caché o cargar
        if path in self._cache:
            self.canvas.set_image(self._cache[path])
        else:
            auto_stretch = self.chk_auto_stretch.isChecked()
            img_u8 = _fast_load_and_stretch(path, auto_stretch=auto_stretch)
            if img_u8 is not None:
                self._cache[path] = img_u8
            self.canvas.set_image(img_u8)

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
        elif key in (Qt.Key_X, Qt.Key_Down, Qt.Key_Delete):
            self.discard_current_frame()
        elif key in (Qt.Key_P, Qt.Key_Up, Qt.Key_Insert):
            self.keep_current_frame()
        elif key == Qt.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)

    def accept(self):
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.selection_applied.emit(self.excluded_files)
        super().accept()

    def reject(self):
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().reject()

    def closeEvent(self, event):
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.selection_applied.emit(self.excluded_files)
        super().closeEvent(event)
