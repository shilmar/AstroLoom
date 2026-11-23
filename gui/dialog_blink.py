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
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QColor, QPainter

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
        else:
            self.setWindowTitle("AstroLoom - Visor Rápido de Subframes (Blink & Culling)")
        self.resize(1150, 750)
        apply_theme(self)

        self.file_list: List[str] = list(file_list)
        self.excluded_files: Set[str] = set(excluded_files) if excluded_files else set()
        self.culling_results: Dict[str, Any] = culling_results if culling_results else {}
        self.current_idx: int = max(0, min(current_index, len(self.file_list) - 1)) if self.file_list else 0
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

        # Badge de inclusión
        self.lbl_status_badge = QLabel("INCLUIDA")
        self.lbl_status_badge.setStyleSheet("padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(self.lbl_status_badge)

        # Badge de calidad óptica si ya fue analizada
        self.lbl_quality_badge = QLabel("")
        self.lbl_quality_badge.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 6px;")
        h_layout.addWidget(self.lbl_quality_badge)

        h_layout.addStretch()

        self.btn_toggle_side = QPushButton("Ocultar Lista")
        self.btn_toggle_side.setIcon(get_icon("sidebar", size=13))
        self.btn_toggle_side.setToolTip("Muestra u oculta la lista lateral de tomas de la serie")
        self.btn_toggle_side.clicked.connect(self._toggle_side_panel)
        h_layout.addWidget(self.btn_toggle_side)

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
        self.side_panel.setMinimumWidth(280)
        self.side_panel.setMaximumWidth(480)
        self.side_panel.setStyleSheet(f"background-color: {BG_CARD}; border: 1px solid {BORDER_BASE}; border-radius: 6px;")
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(4)

        side_hdr = QHBoxLayout()
        lbl_list_title = QLabel("Tomas del Eclipse" if self.mode == "eclipse" else "Tomas de la Serie")
        lbl_list_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #cbd5e1;")
        side_hdr.addWidget(lbl_list_title)
        side_hdr.addStretch()

        total = len(self.file_list)
        inc = total - len(self.excluded_files)
        term = "en collage" if self.mode == "eclipse" else "activas"
        self.lbl_blink_summary = QLabel(f"{inc}/{total} {term}")
        self.lbl_blink_summary.setStyleSheet("color: #94a3b8; font-size: 10px;")
        side_hdr.addWidget(self.lbl_blink_summary)
        side_layout.addLayout(side_hdr)

        col1_name = "Collage" if self.mode == "eclipse" else "Uso"
        self.table_blink = QTableWidget(len(self.file_list), 5)
        self.table_blink.setHorizontalHeaderLabels(["#", col1_name, "Archivo", "Exp.", "ISO"])
        self.table_blink.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_blink.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_blink.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_blink.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_blink.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
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
            QTableWidget::item {{
                padding: 1px 4px;
                color: #e2e8f0;
            }}
            QTableWidget::item:selected {{
                background-color: #1e293b;
                color: #ffffff;
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
        self.table_blink.itemSelectionChanged.connect(self._on_table_selection_changed)
        side_layout.addWidget(self.table_blink)

        self.canvas = BlinkCanvas(self)
        self.splitter.addWidget(self.side_panel)
        self.splitter.addWidget(self.canvas)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
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
        f_layout.addWidget(self.timeline)

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
            f_layout.addWidget(legend_frame)

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

    def _on_timeline_gap_selected(self, from_idx: int, to_idx: int):
        self.set_current_index(from_idx)

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

    def keep_current_frame(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        if path in self.excluded_files:
            self.excluded_files.remove(path)
            self.exclusion_changed.emit(path, True)
            if hasattr(self, 'timeline'):
                self.timeline.set_excluded_files(self.excluded_files)
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
        self._update_badges()
        self.next_frame()

    def _update_badges(self):
        if not self.file_list:
            return
        path = self.file_list[self.current_idx]
        is_included = path not in self.excluded_files
        is_eclipse = (self.mode == "eclipse")

        if is_eclipse:
            if is_included:
                self.lbl_status_badge.setText("● EN COLLAGE")
                self.lbl_status_badge.setStyleSheet("color: #38bdf8; background-color: #082f49; border: 1px solid #0284c7; border-radius: 4px; padding: 2px 8px; font-weight: bold;")
            else:
                self.lbl_status_badge.setText("○ DISPONIBLE")
                self.lbl_status_badge.setStyleSheet("color: #94a3b8; background-color: #1e293b; border: 1px solid #475569; border-radius: 4px; padding: 2px 8px; font-weight: bold;")
        else:
            if is_included:
                self.lbl_status_badge.setText("✓ INCLUIDA")
                self.lbl_status_badge.setStyleSheet("color: #4ade80; background-color: #132b1e; border: 1px solid #22c55e; border-radius: 4px; padding: 2px 8px; font-weight: bold;")
            else:
                self.lbl_status_badge.setText("✗ DESCARTADA")
                self.lbl_status_badge.setStyleSheet("color: #f87171; background-color: #2a1520; border: 1px solid #ef4444; border-radius: 4px; padding: 2px 8px; font-weight: bold;")

        # Actualizar indicador en la tabla lateral
        if hasattr(self, 'table_blink') and self.table_blink.rowCount() > self.current_idx:
            status_item = self.table_blink.item(self.current_idx, 1)
            if status_item:
                if is_eclipse:
                    status_item.setText("● COL" if is_included else "○ DISP")
                    status_item.setForeground(QColor("#38bdf8") if is_included else QColor("#94a3b8"))
                else:
                    status_item.setText("● INCL" if is_included else "✕ DESC")
                    status_item.setForeground(QColor("#4ade80") if is_included else QColor("#f87171"))
        if hasattr(self, 'lbl_blink_summary'):
            total = len(self.file_list)
            inc = total - len(self.excluded_files)
            term = "en collage" if is_eclipse else "activas"
            self.lbl_blink_summary.setText(f"{inc}/{total} {term}")

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

        meta = self._metadata_cache.get(path, {})
        dt_label = meta.get("datetime_str") or meta.get("time_str")
        dt_suffix = f" | {dt_label}" if dt_label and dt_label != "—" else ""
        phase_lbl = meta.get("phase_label")
        phase_suffix = f" | Fase: {phase_lbl}" if phase_lbl and phase_lbl != "Pendiente" else ""
        term = "en collage" if self.mode == "eclipse" else "activas"
        self.lbl_title.setText(f"Fotograma {self.current_idx + 1} de {total} : {filename}{dt_suffix}{phase_suffix} ({included_count}/{total} {term})")
        self._update_badges()

        # Sincronizar selección y scroll en la tabla lateral
        if hasattr(self, 'table_blink') and self.table_blink.rowCount() > self.current_idx:
            self.table_blink.blockSignals(True)
            self.table_blink.selectRow(self.current_idx)
            item = self.table_blink.item(self.current_idx, 0)
            if item:
                self.table_blink.scrollToItem(item)
            self.table_blink.blockSignals(False)

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
                status_text = "● COL" if is_inc else "○ DISP"
                status_color = QColor("#38bdf8") if is_inc else QColor("#94a3b8")
            else:
                status_text = "● INCL" if is_inc else "✕ DESC"
                status_color = QColor("#4ade80") if is_inc else QColor("#f87171")

            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(status_color)

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
