"""
gui/dialog_eclipse_collage.py - Diseñador interactivo de secuencias y collages de eclipses.
Permite:
  - Carga y cribado masivo de cientos de fotos con detección de trepidación y % de ocultación.
  - Recomendación automática de las mejores tomas simétricas (C1 a C4).
  - Maquetación en Cinta Horizontal, Arco Parabólico o Cuadrícula.
  - Vista previa en tiempo real.
  - Exportación directa a PSD nativo por capas de 16 bits para Photoshop y TIFF/JPG.
"""
import os
from typing import List, Optional
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
    QSlider,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QSplitter,
    QScrollArea,
    QAbstractItemView,
    QApplication
)

from core.eclipse_collage import (
    EclipseFrameMetadata,
    batch_analyze_eclipse_series,
    recommend_key_frames,
    compute_collage_layout,
    render_flat_collage,
    export_layered_psd,
    LAYOUT_HORIZONTAL,
    LAYOUT_ARC,
    LAYOUT_GRID,
    HAS_PYTOSHOP
)
from core.icc_manager import save_tiff_with_icc
from gui.widgets.drop_widgets import DropTableWidget, collect_image_files_from_mime


class EclipseAnalysisWorker(QThread):
    """Hilo secundario para el análisis de calidad y porcentaje de eclipse en lote."""
    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, filepaths: List[str], mode: str = "solar"):
        super().__init__()
        self.filepaths = filepaths
        self.mode = mode
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            def cb(curr, total, msg):
                if not self._is_cancelled:
                    self.progress.emit(curr, total, msg)

            results = batch_analyze_eclipse_series(self.filepaths, mode=self.mode, callback_progress=cb)
            if not self._is_cancelled:
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class EclipseCollageDialog(QDialog):
    """Ventana interactiva de diseño y exportación de collages de eclipses."""

    def __init__(self, parent=None, initial_filepaths: Optional[List[str]] = None, mode: str = "solar"):
        super().__init__(parent)
        self.setWindowTitle("Diseñador de Secuencia y Collage de Eclipses - Apilator")
        self.setMinimumSize(1240, 780)
        self.resize(1380, 840)

        self.eclipse_mode = mode  # "solar" o "lunar"
        self.frames_meta: List[EclipseFrameMetadata] = []
        self.preview_image: Optional[np.ndarray] = None
        self.current_placements = []
        self.current_canvas_w = 1920
        self.current_canvas_h = 1080

        self.worker: Optional[EclipseAnalysisWorker] = None

        self._setup_ui()

        if initial_filepaths:
            self._load_filepaths(initial_filepaths)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # -----------------------------------------------------------------
        # Panel Izquierdo: Controles, Tabla y Configuración
        # -----------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # --- SECCIÓN 1: Ingesta y Análisis ---
        grp_files = QGroupBox("1. Ingesta y Cribado de Calidad")
        grp_files.setStyleSheet("QGroupBox { font-weight: bold; color: #80d8ff; }")
        box_files = QVBoxLayout(grp_files)
        box_files.setSpacing(6)

        btn_row_f = QHBoxLayout()
        self.btn_load_files = QPushButton("📁 Cargar Fotos...")
        self.btn_load_files.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_load_files.clicked.connect(self._on_load_files_clicked)
        self.btn_clear_files = QPushButton("🗑️ Limpiar")
        self.btn_clear_files.clicked.connect(self._on_clear_clicked)
        btn_row_f.addWidget(self.btn_load_files)
        btn_row_f.addWidget(self.btn_clear_files)
        box_files.addLayout(btn_row_f)

        # Tabla interactiva con Drag & Drop
        self.table_files = DropTableWidget()
        self.table_files.setColumnCount(5)
        self.table_files.setHorizontalHeaderLabels(["[✓]", "Archivo", "Hora", "% Ocultación", "Nitidez / Estado"])
        self.table_files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_files.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_files.itemChanged.connect(self._on_table_item_changed)
        self.table_files.files_dropped.connect(self._on_files_dropped)
        box_files.addWidget(self.table_files)

        self.btn_analyze = QPushButton("🔍 Analizar Calidad y % Ocultación en Lote")
        self.btn_analyze.setFixedHeight(34)
        self.btn_analyze.setStyleSheet("font-weight: bold; background: #263238; color: #80d8ff; border: 1px solid #80d8ff;")
        self.btn_analyze.clicked.connect(self._start_analysis)
        box_files.addWidget(self.btn_analyze)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        box_files.addWidget(self.progress_bar)

        left_layout.addWidget(grp_files, stretch=3)

        # --- SECCIÓN 2: Recomendador Inteligente ---
        grp_rec = QGroupBox("2. Recomendador Inteligente de Fases")
        grp_rec.setStyleSheet("QGroupBox { font-weight: bold; color: #ffd54f; }")
        box_rec = QVBoxLayout(grp_rec)
        box_rec.setSpacing(6)

        row_rec_ctrl = QHBoxLayout()
        row_rec_ctrl.addWidget(QLabel("Nº Fases:"))
        self.combo_target_count = QComboBox()
        self.combo_target_count.addItems(["5 fases", "7 fases", "9 fases", "11 fases", "13 fases", "15 fases", "Todas"])
        self.combo_target_count.setCurrentIndex(2)  # 9 fases por defecto
        row_rec_ctrl.addWidget(self.combo_target_count)

        self.btn_auto_select = QPushButton("⚡ Auto-Seleccionar Mejores Tomas")
        self.btn_auto_select.setStyleSheet("font-weight: bold; color: #ffd54f;")
        self.btn_auto_select.setToolTip("Selecciona simétricamente las fotos más nítidas descartando trepidaciones")
        self.btn_auto_select.clicked.connect(self._auto_select_recommended)
        row_rec_ctrl.addWidget(self.btn_auto_select)
        box_rec.addLayout(row_rec_ctrl)

        left_layout.addWidget(grp_rec, stretch=1)

        # --- SECCIÓN 3: Maquetación y Espaciado ---
        grp_layout = QGroupBox("3. Maquetación y Composición Espacial")
        grp_layout.setStyleSheet("QGroupBox { font-weight: bold; color: #00e676; }")
        box_layout = QVBoxLayout(grp_layout)
        box_layout.setSpacing(6)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Modalidad:"))
        self.combo_layout_mode = QComboBox()
        self.combo_layout_mode.addItems(["⇋ Cinta Horizontal (Timeline)", "⌒ Arco Celeste (Parábola)", "▦ Cuadrícula (Grid)"])
        self.combo_layout_mode.currentIndexChanged.connect(self._update_preview)
        row_mode.addWidget(self.combo_layout_mode, stretch=1)
        box_layout.addLayout(row_mode)

        # Controles finos
        row_sp = QHBoxLayout()
        row_sp.addWidget(QLabel("Espaciado:"))
        self.slider_gap = QSlider(Qt.Horizontal)
        self.slider_gap.setRange(10, 150)
        self.slider_gap.setValue(40)
        self.slider_gap.valueChanged.connect(self._update_preview)
        row_sp.addWidget(self.slider_gap)
        box_layout.addLayout(row_sp)

        row_sc = QHBoxLayout()
        row_sc.addWidget(QLabel("Escala Totalidad:"))
        self.slider_tot_scale = QSlider(Qt.Horizontal)
        self.slider_tot_scale.setRange(100, 200)
        self.slider_tot_scale.setValue(130)
        self.slider_tot_scale.valueChanged.connect(self._update_preview)
        row_sc.addWidget(self.slider_tot_scale)
        box_layout.addLayout(row_sc)

        self.chk_captions = QCheckBox("Mostrar marcas horarias EXIF bajo cada fase")
        self.chk_captions.setChecked(True)
        self.chk_captions.toggled.connect(self._update_preview)
        box_layout.addWidget(self.chk_captions)

        left_layout.addWidget(grp_layout, stretch=2)

        # --- SECCIÓN 4: Exportación ---
        grp_exp = QGroupBox("4. Exportación")
        grp_exp.setStyleSheet("QGroupBox { font-weight: bold; color: #ff8a80; }")
        box_exp = QVBoxLayout(grp_exp)
        box_exp.setSpacing(6)

        btn_row_exp = QHBoxLayout()
        self.btn_export_psd = QPushButton("💾 Exportar PSD por Capas (16-bit Photoshop)")
        self.btn_export_psd.setFixedHeight(36)
        self.btn_export_psd.setStyleSheet("font-weight: bold; background: #0d47a1; color: #ffffff;")
        self.btn_export_psd.setToolTip("Exporta un archivo de Photoshop nativo con cada fase en su propia capa nombrada")
        self.btn_export_psd.clicked.connect(self._export_psd_clicked)
        btn_row_exp.addWidget(self.btn_export_psd)

        self.btn_export_flat = QPushButton("🖼️ Guardar TIFF / JPG")
        self.btn_export_flat.setFixedHeight(36)
        self.btn_export_flat.setStyleSheet("font-weight: bold; color: #80cbc4;")
        self.btn_export_flat.clicked.connect(self._export_flat_clicked)
        btn_row_exp.addWidget(self.btn_export_flat)

        box_exp.addLayout(btn_row_exp)
        left_layout.addWidget(grp_exp, stretch=1)

        splitter.addWidget(left_widget)

        # -----------------------------------------------------------------
        # Panel Derecho: Vista Previa Interactiva del Collage
        # -----------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        header_row = QHBoxLayout()
        self.lbl_info = QLabel("Vista Previa del Collage (0 fases seleccionadas)")
        self.lbl_info.setStyleSheet("font-weight: bold; color: #cfd8dc; font-size: 12px;")
        header_row.addWidget(self.lbl_info)
        header_row.addStretch()

        self.btn_refresh = QPushButton("🔄 Actualizar")
        self.btn_refresh.clicked.connect(self._update_preview)
        header_row.addWidget(self.btn_refresh)
        right_layout.addLayout(header_row)

        self.scroll_preview = QScrollArea()
        self.scroll_preview.setStyleSheet("background: #121216; border: 1px solid #2a2a35;")
        self.scroll_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview = QLabel("Carga tomas del eclipse y pulsa 'Analizar' para diseñar el collage.")
        self.lbl_preview.setStyleSheet("color: #78909c; font-size: 13px;")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.scroll_preview.setWidget(self.lbl_preview)
        self.scroll_preview.setWidgetResizable(True)
        right_layout.addWidget(self.scroll_preview, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        main_layout.addWidget(splitter)

    # -----------------------------------------------------------------
    # Gestión de Archivos y Tabla
    # -----------------------------------------------------------------
    def _on_load_files_clicked(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar fotos del eclipse",
            "",
            "Imágenes (*.tif *.tiff *.jpg *.jpeg *.png *.cr2 *.cr3 *.nef *.arw *.dng *.fits *.fit)"
        )
        if files:
            self._load_filepaths(files)

    def _on_files_dropped(self, files: List[str]):
        if files:
            self._load_filepaths(files)

    def _load_filepaths(self, filepaths: List[str]):
        if not filepaths:
            return
        total = len(filepaths)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Cargando ficheros... (%v/{total})")
        self.lbl_info.setText(f"Cargando {total} ficheros...")
        QApplication.processEvents()

        new_metas = []
        for idx, fp in enumerate(filepaths):
            fn = os.path.basename(fp)
            new_metas.append(EclipseFrameMetadata(
                filepath=fp,
                filename=fn,
                timestamp=0.0,
                time_str="--:--:--",
                exp_time=0.001,
                iso=100,
                center_x=0.0,
                center_y=0.0,
                radius=100.0,
                sharpness_score=0.0,
                is_trepidated=False,
                obscuration_pct=0.0,
                is_ingress=True,
                phase_label="Pendiente",
                is_recommended=True
            ))
            if (idx + 1) % 25 == 0 or idx == total - 1:
                self.progress_bar.setValue(idx + 1)
                QApplication.processEvents()

        self.frames_meta.extend(new_metas)
        self._populate_table()
        self.progress_bar.setVisible(False)
        self.lbl_info.setText(f"Carga finalizada: {len(self.frames_meta)} tomas añadidas (pulsa 'Analizar').")

    def _on_clear_clicked(self):
        self.frames_meta.clear()
        self.table_files.setRowCount(0)
        self.lbl_preview.setPixmap(QPixmap())
        self.lbl_preview.setText("Carga tomas del eclipse y pulsa 'Analizar' para diseñar el collage.")
        self.lbl_info.setText("Vista Previa del Collage (0 fases seleccionadas)")

    def _populate_table(self):
        self.table_files.blockSignals(True)
        self.table_files.setRowCount(len(self.frames_meta))

        for row, f in enumerate(self.frames_meta):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if f.is_recommended else Qt.Unchecked)
            self.table_files.setItem(row, 0, chk_item)

            name_item = QTableWidgetItem(f.filename)
            name_item.setToolTip(f.filepath)
            self.table_files.setItem(row, 1, name_item)

            time_item = QTableWidgetItem(f.time_str)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.table_files.setItem(row, 2, time_item)

            obsc_str = f"{f.obscuration_pct:.1f}%" if f.sharpness_score > 0 else "--"
            obsc_item = QTableWidgetItem(obsc_str)
            obsc_item.setTextAlignment(Qt.AlignCenter)
            self.table_files.setItem(row, 3, obsc_item)

            if f.sharpness_score > 0:
                if f.is_trepidated:
                    status_text = f"⚠️ {f.sharpness_score:.0f} (Movida)"
                    color = QColor("#ff5252")
                elif f.sharpness_score >= 80:
                    status_text = f"⭐ {f.sharpness_score:.0f} (Nítida)"
                    color = QColor("#00e676")
                else:
                    status_text = f"✓ {f.sharpness_score:.0f}"
                    color = QColor("#ffd54f")
            else:
                status_text = "Pendiente"
                color = QColor("#90a4ae")

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(color)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.table_files.setItem(row, 4, status_item)

        self.table_files.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            row = item.row()
            if 0 <= row < len(self.frames_meta):
                self.frames_meta[row].is_recommended = (item.checkState() == Qt.Checked)
                self._update_preview()

    # -----------------------------------------------------------------
    # Análisis y Clasificación Automática
    # -----------------------------------------------------------------
    def _start_analysis(self):
        if not self.frames_meta:
            QMessageBox.information(self, "Aviso", "Añade fotos primero.")
            return

        fps = [f.filepath for f in self.frames_meta]
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_analyze.setEnabled(False)

        self.worker = EclipseAnalysisWorker(fps, mode=self.eclipse_mode)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, curr, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)
        self.progress_bar.setFormat(f"{curr}/{total} - {msg}")

    def _on_worker_finished(self, results: List[EclipseFrameMetadata]):
        self.progress_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)
        self.frames_meta = results
        self._populate_table()
        self._auto_select_recommended()

    def _on_worker_error(self, err: str):
        self.progress_bar.setVisible(False)
        self.btn_analyze.setEnabled(True)
        QMessageBox.critical(self, "Error de Análisis", f"Ocurrió un error analizando el lote:\n{err}")

    def _auto_select_recommended(self):
        count_str = self.combo_target_count.currentText()
        if count_str == "Todas":
            num = len(self.frames_meta)
        else:
            num = int(count_str.split()[0])

        recommend_key_frames(self.frames_meta, num_targets=num)
        self._populate_table()
        self._update_preview()

    # -----------------------------------------------------------------
    # Generación y Vista Previa del Collage
    # -----------------------------------------------------------------
    def _get_selected_frames(self) -> List[EclipseFrameMetadata]:
        return [f for f in self.frames_meta if f.is_recommended]

    def _update_preview(self):
        selected = self._get_selected_frames()
        self.lbl_info.setText(f"Vista Previa del Collage ({len(selected)} fases seleccionadas)")
        if not selected:
            self.lbl_preview.setPixmap(QPixmap())
            self.lbl_preview.setText("Selecciona al menos una fase para previsualizar.")
            return

        mode_idx = self.combo_layout_mode.currentIndex()
        if mode_idx == 0:
            layout_mode = LAYOUT_HORIZONTAL
        elif mode_idx == 1:
            layout_mode = LAYOUT_ARC
        else:
            layout_mode = LAYOUT_GRID

        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        draw_caps = self.chk_captions.isChecked()

        try:
            # Para previsualización rápida usamos radio base de 90px
            cw, ch, placements = compute_collage_layout(
                selected,
                layout_type=layout_mode,
                canvas_width=1800,
                target_disk_radius=90,
                gap_spacing=gap,
                totality_scale=tot_scale
            )
            self.current_placements = placements
            self.current_canvas_w = cw
            self.current_canvas_h = ch

            flat = render_flat_collage(cw, ch, placements, bg_color=(0.0, 0.0, 0.0), draw_captions=draw_caps)
            self.preview_image = flat

            # Convertir a QPixmap para renderizado
            u8 = (flat * 255.0).astype(np.uint8)
            h, w, c = u8.shape
            bytes_per_line = 3 * w
            qimg = QImage(u8.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            # Escalar a la ventana del visor si excede el tamaño
            viewport_w = max(400, self.scroll_preview.width() - 30)
            viewport_h = max(300, self.scroll_preview.height() - 30)
            scaled_pix = pix.scaled(viewport_w, viewport_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.lbl_preview.setPixmap(scaled_pix)
            self.lbl_preview.setText("")
        except Exception as e:
            self.lbl_preview.setText(f"Error generando vista previa: {e}")

    # -----------------------------------------------------------------
    # Exportaciones
    # -----------------------------------------------------------------
    def _export_psd_clicked(self):
        selected = self._get_selected_frames()
        if not selected:
            QMessageBox.warning(self, "Aviso", "No hay fases seleccionadas para exportar.")
            return

        if not HAS_PYTOSHOP:
            QMessageBox.warning(self, "Aviso", "El módulo 'pytoshop' no está disponible en este entorno.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Collage de Eclipse como PSD por Capas",
            "eclipse_collage_capas.psd",
            "Photoshop PSD (*.psd)"
        )
        if not filepath:
            return

        mode_idx = self.combo_layout_mode.currentIndex()
        layout_mode = LAYOUT_HORIZONTAL if mode_idx == 0 else (LAYOUT_ARC if mode_idx == 1 else LAYOUT_GRID)
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0

        # Para exportación a Photoshop usamos resolución nativa alta (radio 150px)
        cw, ch, placements = compute_collage_layout(
            selected,
            layout_type=layout_mode,
            canvas_width=3840,
            target_disk_radius=150,
            gap_spacing=gap,
            totality_scale=tot_scale
        )

        ok = export_layered_psd(filepath, cw, ch, placements, depth=16)
        if ok:
            QMessageBox.information(
                self, 
                "Exportación Exitosa", 
                f"Collage exportado en PSD por capas de 16-bit:\n{filepath}\n\n"
                f"Lienzo: {cw} × {ch} px ({len(placements)} capas individuales)."
            )
        else:
            QMessageBox.critical(self, "Error", "No se pudo generar el archivo PSD.")

    def _export_flat_clicked(self):
        selected = self._get_selected_frames()
        if not selected:
            QMessageBox.warning(self, "Aviso", "No hay fases seleccionadas para exportar.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Imagen del Collage",
            "eclipse_collage_final.tif",
            "TIFF 16-bit (*.tif *.tiff);;JPEG Alta Calidad (*.jpg *.jpeg)"
        )
        if not filepath:
            return

        mode_idx = self.combo_layout_mode.currentIndex()
        layout_mode = LAYOUT_HORIZONTAL if mode_idx == 0 else (LAYOUT_ARC if mode_idx == 1 else LAYOUT_GRID)
        gap = self.slider_gap.value()
        tot_scale = self.slider_tot_scale.value() / 100.0
        draw_caps = self.chk_captions.isChecked()

        cw, ch, placements = compute_collage_layout(
            selected,
            layout_type=layout_mode,
            canvas_width=3840,
            target_disk_radius=150,
            gap_spacing=gap,
            totality_scale=tot_scale
        )

        flat = render_flat_collage(cw, ch, placements, bg_color=(0.0, 0.0, 0.0), draw_captions=draw_caps)

        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".tif", ".tiff"):
            save_tiff_with_icc(filepath, flat, bit_depth=16, profile_name="sRGB")
        else:
            u8 = (np.clip(flat, 0.0, 1.0) * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])

        QMessageBox.information(
            self,
            "Guardado Exitoso",
            f"Imagen de collage guardada correctamente:\n{filepath}\n\nDimensiones: {cw} × {ch} px"
        )
