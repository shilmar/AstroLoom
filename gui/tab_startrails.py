# gui/tab_startrails.py
"""
gui/tab_startrails.py - Pestaña dedicada para la generación de Trazas de Estrellas (Startrails / Circumpolares).

Implementa:
1. Ingesta de serie temporal de tomas (Lights) y tomas oscuras (Darks).
2. Algoritmo de Máximo Estándar (Lighten Clásico).
3. Algoritmo de Efecto Cometa / Estela Progresiva (Comet / Meteor Fade).
4. Fusión de Suelo Limpio (Separación Cielo/Suelo mediante máscara y reducción drástica de ruido en sombras).
5. Visualización interactiva en Canvas con zoom 1:1.
6. Exportación en TIFF 16-bit / 32-bit y transferencia directa al Revelador / Editor.
"""
import os
from typing import Optional, List, Dict, Any, Callable
import tifffile
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, QCoreApplication
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QMessageBox, QSlider, QRadioButton,
    QButtonGroup, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QProgressBar, QTextEdit, QScrollArea,
    QSplitter, QFrame, QSizePolicy, QProgressDialog
)

from core.stacking import load_image_as_float32
from core.startrails import generate_startrail, enhance_star_trails_color
from core.icc_manager import save_tiff_with_icc
from gui.canvas import MaskCanvas
from gui.widgets.drop_widgets import DropTableWidget
from gui.theme import (
    STYLE_RUN_IDLE, STYLE_RUN_ACTIVE, STYLE_RUN_CANCELLING, STYLE_TRANSFER_BTN
)


class StartrailWorker(QThread):
    """Hilo de trabajo en segundo plano para el procesamiento de trazas sin bloquear la interfaz."""
    progress = Signal(int, str)
    finished = Signal(object, str)
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            result, video_path = generate_startrail(
                files=self.params["files"],
                mode=self.params.get("mode", "lighten"),
                comet_params=self.params.get("comet_params"),
                dark_files=self.params.get("dark_files"),
                use_clean_ground=self.params.get("use_clean_ground", False),
                mask=self.params.get("mask"),
                ground_mode=self.params.get("ground_mode", "average"),
                ref_idx=self.params.get("ref_idx", 0),
                feather_radius=self.params.get("feather_radius", 5),
                suppress_streaks=self.params.get("suppress_streaks", False),
                streak_sensitivity=self.params.get("streak_sensitivity", 0.5),
                fill_gaps=self.params.get("fill_gaps", False),
                gap_fill_radius=self.params.get("gap_fill_radius", 3),
                gap_fill_threshold_factor=self.params.get("gap_fill_threshold_factor", 3.0),
                timelapse_params=self.params.get("timelapse_params"),
                color_boost_params=self.params.get("color_boost_params"),
                progress_callback=self.progress.emit,
                abort_flag=lambda: self._is_cancelled
            )
            if self._is_cancelled:
                self.cancelled.emit()
                return
            self.finished.emit((result, video_path), "startrail_generation")
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(str(e))



class StartrailsTab(QWidget):
    """Pestaña maestra para la creación y posprocesado de circumpolares y trazas de estrellas."""
    export_to_developer = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.files_list: List[str] = []
        self.dark_files: List[str] = []
        self.ref_idx: int = 0
        self.current_mask: Optional[np.ndarray] = None
        self.current_preview_img: Optional[np.ndarray] = None
        self.final_startrail_img: Optional[np.ndarray] = None
        self.raw_startrail_img: Optional[np.ndarray] = None
        self.worker: Optional[StartrailWorker] = None
        self._mask_provider: Optional[Callable[[], Optional[np.ndarray]]] = None
        self.timelapse_custom_path: Optional[str] = None
        self.last_timelapse_path: Optional[str] = None

        self._setup_ui()
        self._apply_styles()

    def set_stacker_mask_provider(self, provider: Callable[[], Optional[np.ndarray]]) -> None:
        """Establece una función proveedora para importar directamente la máscara calculada en el Apilador."""
        self._mask_provider = provider

    def _setup_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -------------------------------------------------------------
        # PANEL IZQUIERDO: Controles en ScrollArea
        # -------------------------------------------------------------
        left_container = QWidget()
        left_container.setMinimumWidth(420)
        left_container.setMaximumWidth(520)
        left_container_layout = QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(4, 4, 4, 4)
        left_container_layout.setSpacing(6)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        panel_content = QWidget()
        left_layout = QVBoxLayout(panel_content)
        left_layout.setContentsMargins(4, 4, 10, 4)
        left_layout.setSpacing(12)

        # 1. GRUPO: SERIE TEMPORAL DE TOMAS
        grp_files = QGroupBox("1. Serie Temporal de Tomas (Lights)")
        files_layout = QVBoxLayout(grp_files)
        files_layout.setContentsMargins(10, 14, 10, 10)
        files_layout.setSpacing(8)

        btn_row_files = QHBoxLayout()
        self.btn_load_series = QPushButton("📁 Cargar Serie...")
        self.btn_load_series.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.btn_load_series.clicked.connect(self._on_load_series_clicked)

        self.btn_clear_series = QPushButton("🗑️ Limpiar")
        self.btn_clear_series.clicked.connect(self._on_clear_series_clicked)

        self.btn_set_ref = QPushButton("📌 Marcar REF")
        self.btn_set_ref.setToolTip("Establece la toma seleccionada como referencia para el suelo y previsualización")
        self.btn_set_ref.clicked.connect(self._on_set_ref_clicked)

        btn_row_files.addWidget(self.btn_load_series)
        btn_row_files.addWidget(self.btn_clear_series)
        btn_row_files.addWidget(self.btn_set_ref)
        files_layout.addLayout(btn_row_files)

        self.table_files = DropTableWidget(0, 3)
        self.table_files.setHorizontalHeaderLabels(["REF", "Archivo", "Ubicación"])
        self.table_files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_files.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_files.setMinimumHeight(150)
        self.table_files.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_files.files_dropped.connect(self.add_series_files)
        files_layout.addWidget(self.table_files)

        self.lbl_files_summary = QLabel("0 tomas cargadas")
        self.lbl_files_summary.setStyleSheet("color: #ffd54f; font-weight: bold; font-size: 11px;")
        files_layout.addWidget(self.lbl_files_summary)

        # Fila compacta de Darks
        row_dark = QHBoxLayout()
        row_dark.addWidget(QLabel("🌑 Darks (opcional):"))
        self.lbl_darks_count = QLabel("0 darks")
        self.lbl_darks_count.setStyleSheet("color: #90a4ae; font-size: 11px;")
        row_dark.addWidget(self.lbl_darks_count)
        row_dark.addStretch()

        self.btn_load_darks = QPushButton("Añadir Darks...")
        self.btn_load_darks.setStyleSheet("font-size: 11px; padding: 3px 8px;")
        self.btn_load_darks.clicked.connect(self._on_load_darks_clicked)
        row_dark.addWidget(self.btn_load_darks)

        self.btn_clear_darks = QPushButton("Quitar")
        self.btn_clear_darks.setStyleSheet("font-size: 11px; padding: 3px 8px;")
        self.btn_clear_darks.clicked.connect(self._on_clear_darks_clicked)
        row_dark.addWidget(self.btn_clear_darks)
        files_layout.addLayout(row_dark)

        left_layout.addWidget(grp_files)

        # 2. GRUPO: ALGORITMO DE TRAZAS (CIELO)
        grp_modes = QGroupBox("2. Algoritmo de Trazas (Cielo)")
        modes_layout = QVBoxLayout(grp_modes)
        modes_layout.setContentsMargins(10, 14, 10, 10)
        modes_layout.setSpacing(8)

        self.rb_lighten = QRadioButton("⭐ Máximo Estándar (Lighten Clásico)")
        self.rb_lighten.setChecked(True)
        self.rb_lighten.toggled.connect(self._on_mode_toggled)
        modes_layout.addWidget(self.rb_lighten)

        self.rb_comet = QRadioButton("☄️ Efecto Cometa / Estela Progresiva (Fade)")
        self.rb_comet.toggled.connect(self._on_mode_toggled)
        modes_layout.addWidget(self.rb_comet)

        # Contenedor de parámetros del Efecto Cometa
        self.container_comet = QWidget()
        comet_layout = QVBoxLayout(self.container_comet)
        comet_layout.setContentsMargins(12, 4, 4, 4)
        comet_layout.setSpacing(8)

        # Longitud de estela
        row_tail = QHBoxLayout()
        row_tail.addWidget(QLabel("Longitud de Estela:"))
        self.lbl_tail_val = QLabel("35%")
        self.lbl_tail_val.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_tail.addWidget(self.lbl_tail_val)
        row_tail.addStretch()
        comet_layout.addLayout(row_tail)

        self.slider_tail = QSlider(Qt.Horizontal)
        self.slider_tail.setRange(5, 100)
        self.slider_tail.setValue(35)
        self.slider_tail.valueChanged.connect(lambda v: self.lbl_tail_val.setText(f"{v}%"))
        comet_layout.addWidget(self.slider_tail)

        # Perfil de decaimiento
        row_decay = QHBoxLayout()
        row_decay.addWidget(QLabel("Curva de Decaimiento:"))
        self.combo_decay = QComboBox()
        self.combo_decay.addItem("Lineal uniforme", "linear")
        self.combo_decay.addItem("Suave (Cosenoidal)", "smooth")
        self.combo_decay.addItem("Exponencial rápida", "exponential")
        row_decay.addWidget(self.combo_decay)
        comet_layout.addLayout(row_decay)

        # Dirección
        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Dirección:"))
        self.combo_dir = QComboBox()
        self.combo_dir.addItem("Hacia atrás (Cola en tomas pasadas)", "backward")
        self.combo_dir.addItem("Hacia adelante (Cola en tomas futuras)", "forward")
        self.combo_dir.addItem("Doble / Ambos sentidos (Simétrico: brillo central y colas afiladas)", "bidirectional")
        row_dir.addWidget(self.combo_dir)
        comet_layout.addLayout(row_dir)

        # Cota mínima de luminancia de fondo
        row_floor = QHBoxLayout()
        row_floor.addWidget(QLabel("Luminancia mínima de fondo:"))
        self.lbl_floor_val = QLabel("0%")
        self.lbl_floor_val.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_floor.addWidget(self.lbl_floor_val)
        row_floor.addStretch()
        comet_layout.addLayout(row_floor)

        self.slider_floor = QSlider(Qt.Horizontal)
        self.slider_floor.setRange(0, 25)
        self.slider_floor.setValue(0)
        self.slider_floor.valueChanged.connect(lambda v: self.lbl_floor_val.setText(f"{v}%"))
        comet_layout.addWidget(self.slider_floor)

        modes_layout.addWidget(self.container_comet)
        self.container_comet.setVisible(False)  # Inicialmente oculto en modo Lighten

        # Supresión de satélites y aviones
        self.chk_suppress_streaks = QCheckBox("🛸 Suprimir satélites y aviones automáticamente")
        self.chk_suppress_streaks.setStyleSheet("font-weight: bold; color: #80cbc4;")
        self.chk_suppress_streaks.setToolTip(
            "Detecta trazas lineales transitorias (satélites artificiales y aviones con luces estroboscópicas)\n"
            "mediante comparación temporal diferencial y las elimina sin afectar a las estrellas."
        )
        self.chk_suppress_streaks.toggled.connect(self._on_suppress_streaks_toggled)
        modes_layout.addWidget(self.chk_suppress_streaks)

        self.container_streaks = QWidget()
        streaks_layout = QVBoxLayout(self.container_streaks)
        streaks_layout.setContentsMargins(16, 2, 4, 4)
        streaks_layout.setSpacing(6)

        row_sens = QHBoxLayout()
        row_sens.addWidget(QLabel("Sensibilidad:"))
        self.combo_streak_sens = QComboBox()
        self.combo_streak_sens.addItem("Baja (Solo trazas muy brillantes o gruesas)", 0.25)
        self.combo_streak_sens.addItem("Media (Recomendada / Equilibrada)", 0.50)
        self.combo_streak_sens.addItem("Alta (Satélites débiles y luces tenues)", 0.75)
        self.combo_streak_sens.setCurrentIndex(1)
        row_sens.addWidget(self.combo_streak_sens)
        streaks_layout.addLayout(row_sens)

        self.container_streaks.setVisible(False)
        modes_layout.addWidget(self.container_streaks)

        # Relleno de pausas entre tomas (Gap Filling)
        self.chk_fill_gaps = QCheckBox("✨ Rellenar pausas entre tomas (Gap Filling)")
        self.chk_fill_gaps.setStyleSheet("font-weight: bold; color: #ffb74d;")
        self.chk_fill_gaps.setToolTip(
            "Elimina la apariencia de línea punteada o saltos conectando de forma continua\n"
            "las estrellas entre fotogramas consecutivos."
        )
        self.chk_fill_gaps.toggled.connect(self._on_fill_gaps_toggled)
        modes_layout.addWidget(self.chk_fill_gaps)

        self.container_gaps = QWidget()
        gaps_layout = QVBoxLayout(self.container_gaps)
        gaps_layout.setContentsMargins(16, 2, 4, 4)
        gaps_layout.setSpacing(6)

        row_gap_radius = QHBoxLayout()
        row_gap_radius.addWidget(QLabel("Radio del salto (px):"))
        self.lbl_gap_radius_val = QLabel("2 px")
        self.lbl_gap_radius_val.setStyleSheet("color: #ffb74d; font-weight: bold;")
        row_gap_radius.addWidget(self.lbl_gap_radius_val)
        row_gap_radius.addStretch()
        gaps_layout.addLayout(row_gap_radius)

        self.slider_gap_radius = QSlider(Qt.Horizontal)
        self.slider_gap_radius.setRange(1, 4)
        self.slider_gap_radius.setValue(2)
        self.slider_gap_radius.setToolTip("1–2 px para pausas de intervalómetro de 1–2s; 3–4 px para pausas mayores.")
        self.slider_gap_radius.valueChanged.connect(lambda v: self.lbl_gap_radius_val.setText(f"{v} px"))
        gaps_layout.addWidget(self.slider_gap_radius)

        lbl_gap_hint = QLabel("💡 1–2 px cierra huecos sin engrosar trazas; valores mayores ensanchan las estrellas.")
        lbl_gap_hint.setStyleSheet("color: #90a4ae; font-size: 10px; font-style: italic;")
        lbl_gap_hint.setWordWrap(True)
        gaps_layout.addWidget(lbl_gap_hint)

        row_gap_sens = QHBoxLayout()
        row_gap_sens.addWidget(QLabel("Umbral estelar:"))
        self.combo_gap_sens = QComboBox()
        self.combo_gap_sens.addItem("Equilibrado / Automático (Recomendado)", 3.0)
        self.combo_gap_sens.addItem("Permisivo (Incluir estrellas tenues)", 2.0)
        self.combo_gap_sens.addItem("Conservador (Solo estrellas brillantes)", 4.5)
        self.combo_gap_sens.setCurrentIndex(0)
        row_gap_sens.addWidget(self.combo_gap_sens)
        gaps_layout.addLayout(row_gap_sens)

        self.container_gaps.setVisible(False)
        modes_layout.addWidget(self.container_gaps)

        # Realce y preservación de color estelar (Star Color Boost)
        self.chk_star_color = QCheckBox("🌈 Realzar y preservar color estelar (Star Color Boost)")
        self.chk_star_color.setStyleSheet("font-weight: bold; color: #ce93d8;")
        self.chk_star_color.setToolTip(
            "Preserva los colores espectrales naturales de las estrellas (azules, dorados, anaranjados y rojos)\n"
            "evitando que las trazas queden blancas y recuperando el color en los núcleos sobreexpuestos."
        )
        self.chk_star_color.toggled.connect(self._on_star_color_toggled)
        modes_layout.addWidget(self.chk_star_color)

        self.container_star_color = QWidget()
        color_layout = QVBoxLayout(self.container_star_color)
        color_layout.setContentsMargins(16, 2, 4, 4)
        color_layout.setSpacing(6)

        # Saturación estelar
        row_sat = QHBoxLayout()
        row_sat.addWidget(QLabel("Saturación Estelar:"))
        self.lbl_star_sat_val = QLabel("1.5x")
        self.lbl_star_sat_val.setStyleSheet("color: #ce93d8; font-weight: bold;")
        row_sat.addWidget(self.lbl_star_sat_val)
        row_sat.addStretch()
        color_layout.addLayout(row_sat)

        self.slider_star_sat = QSlider(Qt.Horizontal)
        self.slider_star_sat.setRange(100, 300)  # 1.0x a 3.0x
        self.slider_star_sat.setValue(150)
        self.slider_star_sat.valueChanged.connect(lambda v: self.lbl_star_sat_val.setText(f"{v/100.0:.1f}x"))
        color_layout.addWidget(self.slider_star_sat)

        # Recuperación de núcleos (Core Recovery)
        row_core = QHBoxLayout()
        row_core.addWidget(QLabel("Recuperación de Núcleos Quemados:"))
        self.lbl_core_rec_val = QLabel("50%")
        self.lbl_core_rec_val.setStyleSheet("color: #ffd54f; font-weight: bold;")
        row_core.addWidget(self.lbl_core_rec_val)
        row_core.addStretch()
        color_layout.addLayout(row_core)

        self.slider_core_rec = QSlider(Qt.Horizontal)
        self.slider_core_rec.setRange(0, 100)  # 0% a 100%
        self.slider_core_rec.setValue(50)
        self.slider_core_rec.valueChanged.connect(lambda v: self.lbl_core_rec_val.setText(f"{v}%"))
        color_layout.addWidget(self.slider_core_rec)

        # Botón para reajuste rápido sin recalcular todo
        self.btn_apply_color_live = QPushButton("✨ Aplicar a la Imagen Actual (Retoque Rápido)")
        self.btn_apply_color_live.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                font-weight: bold;
                background-color: #4a148c;
                color: #e1bee7;
                border: 1px solid #7b1fa2;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #6a1b9a;
            }
            QPushButton:disabled {
                background-color: #212121;
                color: #616161;
                border-color: #333333;
            }
        """)
        self.btn_apply_color_live.setToolTip(
            "Aplica los ajustes de saturación y recuperación de núcleos a la circumpolar ya generada\n"
            "en menos de un segundo, sin tener que volver a integrar todas las tomas."
        )
        self.btn_apply_color_live.setEnabled(False)
        self.btn_apply_color_live.clicked.connect(self._on_apply_star_color_live_clicked)
        color_layout.addWidget(self.btn_apply_color_live)

        self.container_star_color.setVisible(False)
        modes_layout.addWidget(self.container_star_color)

        left_layout.addWidget(grp_modes)

        # 3. GRUPO: FUSIÓN DE SUELO LIMPIO (ANTI-RUIDO)
        grp_ground = QGroupBox("3. Fusión de Suelo Limpio (Separación Cielo / Suelo)")
        ground_layout = QVBoxLayout(grp_ground)
        ground_layout.setContentsMargins(10, 14, 10, 10)
        ground_layout.setSpacing(8)

        self.chk_clean_ground = QCheckBox("Activar Suelo Limpio (Eliminación de Ruido)")
        self.chk_clean_ground.setStyleSheet("font-weight: bold; color: #a5d6a7;")
        self.chk_clean_ground.toggled.connect(self._on_clean_ground_toggled)
        ground_layout.addWidget(self.chk_clean_ground)

        self.container_ground = QWidget()
        gnd_sub_layout = QVBoxLayout(self.container_ground)
        gnd_sub_layout.setContentsMargins(12, 4, 4, 4)
        gnd_sub_layout.setSpacing(8)

        row_gnd_mode = QHBoxLayout()
        row_gnd_mode.addWidget(QLabel("Integración del Suelo:"))
        self.combo_gnd_mode = QComboBox()
        self.combo_gnd_mode.addItem("Promedio temporal (Ruido mínimo)", "average")
        self.combo_gnd_mode.addItem("Toma de referencia única", "ref")
        row_gnd_mode.addWidget(self.combo_gnd_mode)
        gnd_sub_layout.addLayout(row_gnd_mode)

        # Máscara de horizonte
        self.btn_ai_mask = QPushButton("🧠 Auto-Detectar Suelo (IA)")
        self.btn_ai_mask.setCursor(Qt.PointingHandCursor)
        self.btn_ai_mask.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a1b9a, stop:1 #8e24aa);
                color: #ffffff;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                border: 1px solid #ba68c8;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b1fa2, stop:1 #9c27b0);
                border-color: #ce93d8;
            }
            QPushButton:pressed {
                background-color: #4a148c;
            }
        """)
        self.btn_ai_mask.setToolTip("Segmenta automáticamente el horizonte con IA ligera U²-Net y Guided Filter sobre la toma de referencia")
        self.btn_ai_mask.clicked.connect(self._on_ai_mask_clicked)
        gnd_sub_layout.addWidget(self.btn_ai_mask)

        row_mask_btns = QHBoxLayout()
        self.btn_load_mask = QPushButton("📁 Cargar...")
        self.btn_load_mask.setToolTip("Carga un archivo PNG, TIFF o FITS con la máscara de cielo/suelo")
        self.btn_load_mask.clicked.connect(self._on_load_mask_clicked)

        self.btn_import_stacker_mask = QPushButton("🖌️ Del Apilador")
        self.btn_import_stacker_mask.setStyleSheet("color: #ffcc80;")
        self.btn_import_stacker_mask.setToolTip("Importa con un solo clic la máscara de horizonte calculada en la Pestaña 1")
        self.btn_import_stacker_mask.clicked.connect(self._on_import_stacker_mask_clicked)

        row_mask_btns.addWidget(self.btn_load_mask)
        row_mask_btns.addWidget(self.btn_import_stacker_mask)
        gnd_sub_layout.addLayout(row_mask_btns)

        self.lbl_mask_status = QLabel("⚠️ Sin máscara de horizonte asignada")
        self.lbl_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
        gnd_sub_layout.addWidget(self.lbl_mask_status)

        # Suavizado de bordes (Feathering)
        row_feather = QHBoxLayout()
        row_feather.addWidget(QLabel("Suavizado de Borde (Feather):"))
        self.lbl_feather_val = QLabel("5 px")
        self.lbl_feather_val.setStyleSheet("color: #80d8ff; font-weight: bold;")
        row_feather.addWidget(self.lbl_feather_val)
        row_feather.addStretch()
        gnd_sub_layout.addLayout(row_feather)

        self.slider_feather = QSlider(Qt.Horizontal)
        self.slider_feather.setRange(0, 30)
        self.slider_feather.setValue(5)
        self.slider_feather.valueChanged.connect(lambda v: self.lbl_feather_val.setText(f"{v} px"))
        gnd_sub_layout.addWidget(self.slider_feather)

        self.chk_show_mask = QCheckBox("Superponer máscara en el visor")
        self.chk_show_mask.setChecked(False)
        self.chk_show_mask.toggled.connect(self._on_show_mask_toggled)
        gnd_sub_layout.addWidget(self.chk_show_mask)

        ground_layout.addWidget(self.container_ground)
        self.container_ground.setEnabled(False)

        left_layout.addWidget(grp_ground)

        # 4. GRUPO: VÍDEO TIME-LAPSE
        grp_timelapse = QGroupBox("4. Vídeo Time-Lapse de la Circumpolar (Opcional)")
        tl_layout = QVBoxLayout(grp_timelapse)
        tl_layout.setContentsMargins(10, 14, 10, 10)
        tl_layout.setSpacing(8)

        self.chk_enable_timelapse = QCheckBox("🎬 Generar Vídeo Time-Lapse (.mp4)")
        self.chk_enable_timelapse.setStyleSheet("font-weight: bold; color: #80d8ff;")
        self.chk_enable_timelapse.setToolTip(
            "Crea un vídeo .mp4 mostrando el nacimiento y giro progresivo de las trazas de estrellas,\n"
            "fotograma a fotograma a lo largo de toda la sesión."
        )
        self.chk_enable_timelapse.toggled.connect(self._on_timelapse_toggled)
        tl_layout.addWidget(self.chk_enable_timelapse)

        self.container_timelapse = QWidget()
        tl_sub_layout = QVBoxLayout(self.container_timelapse)
        tl_sub_layout.setContentsMargins(12, 4, 4, 4)
        tl_sub_layout.setSpacing(8)

        # FPS
        row_fps = QHBoxLayout()
        row_fps.addWidget(QLabel("Fotogramas por seg (FPS):"))
        self.combo_tl_fps = QComboBox()
        self.combo_tl_fps.addItem("24 FPS (Cinemático estándar)", 24)
        self.combo_tl_fps.addItem("30 FPS (Fluido)", 30)
        self.combo_tl_fps.addItem("60 FPS (Ultra fluido)", 60)
        self.combo_tl_fps.addItem("12 FPS (Paso a paso lento)", 12)
        self.combo_tl_fps.setCurrentIndex(0)
        row_fps.addWidget(self.combo_tl_fps)
        tl_sub_layout.addLayout(row_fps)

        # Resolución
        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Resolución del Vídeo:"))
        self.combo_tl_res = QComboBox()
        self.combo_tl_res.addItem("Full HD 1080p (1920x1080 recomendado)", "1080p")
        self.combo_tl_res.addItem("4K Ultra HD (3840x2160)", "4k")
        self.combo_tl_res.addItem("Resolución Original del Sensor", "original")
        self.combo_tl_res.setCurrentIndex(0)
        row_res.addWidget(self.combo_tl_res)
        tl_sub_layout.addLayout(row_res)

        # Exportar fotogramas individuales
        self.chk_tl_save_frames = QCheckBox("Exportar fotogramas secuenciales (.jpg)")
        self.chk_tl_save_frames.setStyleSheet("color: #b0bec5; font-size: 11px;")
        self.chk_tl_save_frames.setToolTip(
            "Guarda cada paso de la acumulación como una imagen JPG numerada en una subcarpeta,\n"
            "útil para edición manual en Premiere, DaVinci Resolve o After Effects."
        )
        tl_sub_layout.addWidget(self.chk_tl_save_frames)

        # Destino del vídeo
        row_dest = QHBoxLayout()
        self.lbl_tl_path = QLabel("Destino: Automático (junto a las tomas)")
        self.lbl_tl_path.setStyleSheet("color: #90a4ae; font-size: 10px;")
        self.lbl_tl_path.setWordWrap(True)
        self.btn_tl_dest = QPushButton("📁 Cambiar Destino...")
        self.btn_tl_dest.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        self.btn_tl_dest.clicked.connect(self._on_select_timelapse_dest_clicked)
        row_dest.addWidget(self.lbl_tl_path, 1)
        row_dest.addWidget(self.btn_tl_dest)
        tl_sub_layout.addLayout(row_dest)

        # Botón de reproducción de timelapse generado
        self.btn_play_timelapse = QPushButton("▶️ Reproducir Vídeo Time-Lapse")
        self.btn_play_timelapse.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 12px;
                background-color: #00838f;
                color: white;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #0097a7;
            }
            QPushButton:disabled {
                background-color: #37474f;
                color: #78909c;
            }
        """)
        self.btn_play_timelapse.setEnabled(False)
        self.btn_play_timelapse.clicked.connect(self._on_play_timelapse_clicked)
        tl_sub_layout.addWidget(self.btn_play_timelapse)

        tl_layout.addWidget(self.container_timelapse)
        self.container_timelapse.setVisible(False)

        left_layout.addWidget(grp_timelapse)

        # 5. GRUPO: EXPORTACIÓN
        grp_export = QGroupBox("5. Exportación y Flujo de Trabajo")
        export_layout = QVBoxLayout(grp_export)
        export_layout.setContentsMargins(10, 14, 10, 10)
        export_layout.setSpacing(8)

        btn_tiff_row = QHBoxLayout()
        self.btn_save_tiff_16 = QPushButton("💾 Guardar TIFF 16-bit...")
        self.btn_save_tiff_16.setStyleSheet("font-weight: bold; color: #80cbc4;")
        self.btn_save_tiff_16.clicked.connect(lambda: self._on_save_tiff_clicked(bit_depth=16))

        self.btn_save_tiff_32 = QPushButton("💾 Guardar TIFF 32-bit...")
        self.btn_save_tiff_32.setStyleSheet("font-weight: bold; color: #90caf9;")
        self.btn_save_tiff_32.clicked.connect(lambda: self._on_save_tiff_clicked(bit_depth=32))

        btn_tiff_row.addWidget(self.btn_save_tiff_16)
        btn_tiff_row.addWidget(self.btn_save_tiff_32)
        export_layout.addLayout(btn_tiff_row)

        self.btn_send_to_dev = QPushButton("➡️ Enviar al Revelador / Editor")
        self.btn_send_to_dev.setStyleSheet(STYLE_TRANSFER_BTN)
        self.btn_send_to_dev.clicked.connect(self._on_send_to_developer_clicked)
        export_layout.addWidget(self.btn_send_to_dev)

        left_layout.addWidget(grp_export)
        left_layout.addStretch()

        scroll_area.setWidget(panel_content)
        left_container_layout.addWidget(scroll_area, 1)

        # Consola de estado, barra de progreso y botón de ejecución FIJOS abajo
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        left_container_layout.addWidget(self.progress_bar)

        lbl_log = QLabel("Registro de Actividad:")
        lbl_log.setStyleSheet("color: #a0a0a0; font-size: 11px; font-weight: bold;")
        left_container_layout.addWidget(lbl_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(100)
        left_container_layout.addWidget(self.txt_log)

        self.btn_run = QPushButton()
        self.btn_run.setFixedHeight(42)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self._set_run_button_state("idle")
        self.btn_run.clicked.connect(self._on_run_clicked)
        left_container_layout.addWidget(self.btn_run)

        # -------------------------------------------------------------
        # PANEL DERECHO: Canvas Interactivo con Zoom 1:1 y Drag & Drop
        # -------------------------------------------------------------
        self.canvas = MaskCanvas(self, enable_masking=False)
        self.canvas.files_dropped.connect(self.add_series_files)

        splitter.addWidget(left_container)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 10)
        splitter.setSizes([520, 1400])

        main_layout.addWidget(splitter)

    def _apply_styles(self) -> None:
        """Los estilos base provienen de gui.theme (Cosmic Dark Pro)."""
        pass

    def log(self, message: str) -> None:
        """Registra un mensaje en la consola inferior con autoscroll."""
        self.txt_log.append(message)
        self.txt_log.moveCursor(QTextCursor.End)

    # -------------------------------------------------------------
    # GESTIÓN DE EVENTOS Y ACCIONES
    # -------------------------------------------------------------
    def _on_mode_toggled(self) -> None:
        """Muestra u oculta los controles específicos del efecto cometa."""
        is_comet = self.rb_comet.isChecked()
        self.container_comet.setVisible(is_comet)

    def _on_suppress_streaks_toggled(self, checked: bool) -> None:
        """Muestra u oculta los controles de sensibilidad para la supresión de satélites y aviones."""
        self.container_streaks.setVisible(checked)

    def _on_fill_gaps_toggled(self, checked: bool) -> None:
        """Muestra u oculta los controles de ajuste de Gap Filling."""
        self.container_gaps.setVisible(checked)

    def _on_clean_ground_toggled(self, checked: bool) -> None:
        """Habilita o deshabilita los controles de integración de suelo limpio."""
        self.container_ground.setEnabled(checked)
        if checked:
            self._update_mask_status_label()

    def add_series_files(self, files: List[str]) -> None:
        """Añade tomas a la serie temporal de circumpolares (Drag & Drop o diálogo)."""
        if not files:
            return
        new_files = [f for f in files if f not in self.files_list]
        if not new_files:
            return
        was_empty = (len(self.files_list) == 0)
        self.files_list.extend(new_files)
        self.files_list.sort()
        if was_empty:
            self.ref_idx = 0
        self._refresh_table()
        self.lbl_files_summary.setText(f"{len(self.files_list)} tomas cargadas")
        self.log(f"Añadidas {len(new_files)} tomas para la serie de trazas (Total: {len(self.files_list)}).")

        # Cargar la primera toma en el visor para previsualización inmediata si estaba vacío
        if was_empty and self.files_list:
            try:
                img = load_image_as_float32(self.files_list[0])
                self.current_preview_img = img
                self.canvas.load_image(img)
            except Exception as e:
                self.log(f"Error al previsualizar primera toma: {e}")

    def _on_load_series_clicked(self) -> None:
        """Abre el diálogo de archivos para cargar la serie temporal de fotografías."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar Serie de Tomas para Startrails",
            os.getcwd(),
            "Imágenes (*.nef *.cr2 *.cr3 *.arw *.dng *.raw *.tif *.tiff *.fits *.jpg *.png);;Todos (*.*)"
        )
        if not files:
            return
        self.add_series_files(files)

    def _on_clear_series_clicked(self) -> None:
        """Limpia la lista de archivos."""
        self.files_list.clear()
        self.table_files.setRowCount(0)
        self.lbl_files_summary.setText("0 tomas cargadas")
        self.current_preview_img = None
        self.final_startrail_img = None
        self.raw_startrail_img = None
        self.current_mask = None
        self._update_mask_status_label()
        self.canvas.clear_image()
        self.timelapse_custom_path = None
        self.last_timelapse_path = None
        self.btn_play_timelapse.setEnabled(False)
        self.btn_apply_color_live.setEnabled(False)
        self.lbl_tl_path.setText("Destino: Automático (junto a las tomas)")
        self.lbl_tl_path.setStyleSheet("color: #90a4ae; font-size: 10px;")
        self.log("Lista de tomas vaciada y máscara reiniciada.")

    def _refresh_table(self) -> None:
        """Actualiza la tabla con los archivos cargados."""
        self.table_files.setRowCount(len(self.files_list))
        for row, path in enumerate(self.files_list):
            is_ref = (row == self.ref_idx)
            ref_item = QTableWidgetItem("📌 REF" if is_ref else "")
            ref_item.setTextAlignment(Qt.AlignCenter)
            if is_ref:
                ref_item.setForeground(Qt.yellow)

            name_item = QTableWidgetItem(os.path.basename(path))
            dir_item = QTableWidgetItem(os.path.dirname(path))

            self.table_files.setItem(row, 0, ref_item)
            self.table_files.setItem(row, 1, name_item)
            self.table_files.setItem(row, 2, dir_item)

    def _on_set_ref_clicked(self) -> None:
        """Establece la fila actualmente seleccionada como referencia."""
        selected_rows = self.table_files.selectionModel().selectedRows()
        if not selected_rows:
            return
        self.ref_idx = selected_rows[0].row()
        self._refresh_table()
        ref_path = self.files_list[self.ref_idx]
        self.log(f"Toma de referencia establecida: {os.path.basename(ref_path)}")
        try:
            img = load_image_as_float32(ref_path)
            self.current_preview_img = img
            self.canvas.load_image(img)
        except Exception as e:
            self.log(f"Error al cargar referencia: {e}")

    def _on_table_selection_changed(self) -> None:
        """Carga en el canvas la imagen que el usuario cliquee en la lista."""
        selected_rows = self.table_files.selectionModel().selectedRows()
        if not selected_rows or self.final_startrail_img is not None:
            return
        idx = selected_rows[0].row()
        if 0 <= idx < len(self.files_list):
            try:
                img = load_image_as_float32(self.files_list[idx])
                self.current_preview_img = img
                self.canvas.load_image(img)
                self._update_canvas_mask_overlay()
            except Exception:
                pass

    def _on_load_darks_clicked(self) -> None:
        """Carga tomas oscuras para la sustracción de píxeles calientes."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar Tomas Dark",
            os.getcwd(),
            "Imágenes (*.nef *.cr2 *.cr3 *.arw *.dng *.raw *.tif *.tiff *.fits);;Todos (*.*)"
        )
        if files:
            self.dark_files = files
            self.lbl_darks_count.setText(f"{len(files)} darks")
            self.lbl_darks_count.setStyleSheet("color: #a5d6a7; font-weight: bold; font-size: 11px;")
            self.log(f"Añadidos {len(files)} tomas dark para sustracción térmica.")

    def _on_clear_darks_clicked(self) -> None:
        """Descarta las tomas oscuras."""
        self.dark_files.clear()
        self.lbl_darks_count.setText("0 darks")
        self.lbl_darks_count.setStyleSheet("color: #90a4ae; font-size: 11px;")
        self.log("Darks descartados.")

    def _on_ai_mask_clicked(self) -> None:
        """Detección automática de cielo/suelo mediante IA Ligera U²-Net para Startrails."""
        if not self.files_list:
            QMessageBox.information(self, "Aviso", "Añade primero tomas a la serie para detectar el suelo.")
            return

        ref_img = self.canvas.orig_rgb
        if ref_img is None:
            ref_idx = min(self.ref_idx, len(self.files_list) - 1)
            ref_path = self.files_list[ref_idx]
            try:
                ref_img = load_image_as_float32(ref_path)
                self.canvas.load_image(ref_img)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cargar la toma de referencia:\n{e}")
                return

        from core.ai_horizon import is_ai_model_available
        if not is_ai_model_available():
            msg = QMessageBox(self)
            msg.setWindowTitle("Modelo de IA no encontrado")
            msg.setText(
                "El modelo de IA para detección semántica precisa de suelo (skyseg.onnx, ~167 MB) no se encuentra en la carpeta 'models/'.\n\n"
                "¿Deseas descargarlo ahora automáticamente o prefieres usar el método algorítmico rápido?"
            )
            btn_download = msg.addButton("📥 Descargar IA (~167 MB)", QMessageBox.AcceptRole)
            btn_algo = msg.addButton("⚡ Usar Algorítmico", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_download)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_download:
                self._start_model_download_startrails()
                return
            elif clicked == btn_algo:
                self._run_ai_mask_startrails(force_algorithmic=True)
                return
            else:
                return

        self._run_ai_mask_startrails(force_algorithmic=False)

    def _start_model_download_startrails(self):
        """Descarga asistida del modelo ONNX con diálogo de progreso para Startrails."""
        progress = QProgressDialog("Conectando con el servidor de modelos...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Descarga de Modelo IA (AstroLoom)")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        from gui.worker import ModelDownloadWorker
        self.download_worker = ModelDownloadWorker(parent=self)

        def on_prog(pct, d_mb, t_mb):
            progress.setValue(pct)
            progress.setLabelText(f"Descargando modelo de IA (skyseg.onnx)...\n{d_mb:.1f} MB de {t_mb:.1f} MB ({pct}%)")

        def on_finished(path):
            progress.close()
            self.log(f"✓ [IA HORIZONTE] Modelo descargado e instalado en: {os.path.basename(path)}")
            self._run_ai_mask_startrails(force_algorithmic=False)

        def on_error(err):
            progress.close()
            self.log(f"[ERROR DESCARGA] {err}")
            QMessageBox.warning(self, "Error en Descarga", f"No se pudo descargar el modelo:\n{err}\n\nSe usará el método algorítmico.")
            self._run_ai_mask_startrails(force_algorithmic=True)

        def on_cancel():
            self.download_worker.cancel()
            self.log("[IA HORIZONTE] Descarga cancelada por el usuario.")

        self.download_worker.progress_changed.connect(on_prog)
        self.download_worker.finished_success.connect(on_finished)
        self.download_worker.error_occurred.connect(on_error)
        progress.canceled.connect(on_cancel)

        self.download_worker.start()

    def _run_ai_mask_startrails(self, force_algorithmic: bool = False):
        self.setCursor(Qt.WaitCursor)
        self.log("[IA HORIZONTE] Ejecutando detección de cielo/suelo...")
        QCoreApplication.processEvents()
        try:
            from core.ai_horizon import auto_detect_sky_mask
            ref_img = self.canvas.orig_rgb
            mask, method = auto_detect_sky_mask(ref_img, force_algorithmic=force_algorithmic)
            self.current_mask = (mask * 255.0).astype(np.uint8)
            self.chk_show_mask.setChecked(True)
            self._update_mask_status_label()
            self._update_canvas_mask_overlay()
            self.log(f"✓ Máscara de horizonte calculada con éxito ({method}).")
        except Exception as e:
            self.log(f"[ERROR IA HORIZONTE] Falló la detección: {e}")
            QMessageBox.critical(self, "Error Detección IA", f"No se pudo completar la segmentación automática:\n{e}")
        finally:
            self.unsetCursor()

    def _on_load_mask_clicked(self) -> None:
        """Permite al usuario seleccionar un archivo de imagen con la máscara de horizonte."""
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Máscara de Horizonte",
            os.getcwd(),
            "Imágenes (*.png *.tif *.tiff *.fits *.jpg);;Todos (*.*)"
        )
        if f:
            try:
                mask_data = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if mask_data is None:
                    # Intento con tifffile si es float o 16-bit
                    mask_data = tifffile.imread(f)
                    if mask_data.ndim == 3:
                        mask_data = mask_data[..., 0]
                    if mask_data.max() <= 1.05:
                        mask_data = (mask_data * 255.0).astype(np.uint8)

                self.current_mask = mask_data
                self.chk_show_mask.setChecked(True)
                self._update_mask_status_label()
                self._update_canvas_mask_overlay()
                self.log(f"Máscara de horizonte cargada desde: {os.path.basename(f)}")
            except Exception as e:
                QMessageBox.critical(self, "Error de Máscara", f"No se pudo leer la máscara:\n{e}")

    def _on_import_stacker_mask_clicked(self) -> None:
        """Importa la máscara generada en la Pestaña 1 (Apilador)."""
        if self._mask_provider is not None:
            m = self._mask_provider()
            if m is not None:
                self.current_mask = m.copy()
                self.chk_show_mask.setChecked(True)
                self._update_mask_status_label()
                self._update_canvas_mask_overlay()
                self.log("Máscara del Apilador importada correctamente.")
                QMessageBox.information(self, "Máscara Importada", "Se ha importado con éxito la máscara de horizonte desde el módulo Apilador.")
                return

        QMessageBox.warning(
            self, "Sin Máscara Disponible",
            "No hay ninguna máscara calculada en el módulo de Apilado.\n"
            "Calcula primero una segmentación en el Apilador o carga una máscara externa con 'Cargar Máscara...'."
        )

    def _update_mask_status_label(self) -> None:
        """Actualiza el texto descriptivo del estado de la máscara."""
        if self.current_mask is not None:
            h, w = self.current_mask.shape[:2]
            self.lbl_mask_status.setText(f"✓ Máscara activa ({w}x{h} px)")
            self.lbl_mask_status.setStyleSheet("color: #81c784; font-weight: bold; font-size: 11px;")
        else:
            self.lbl_mask_status.setText("⚠️ Sin máscara de horizonte asignada")
            self.lbl_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")

    def _on_show_mask_toggled(self, checked: bool) -> None:
        """Muestra u oculta la máscara superpuesta en el visor."""
        self._update_canvas_mask_overlay()

    def _update_canvas_mask_overlay(self) -> None:
        """Transfiere la máscara al visor para representación visual en color."""
        if self.chk_show_mask.isChecked() and self.current_mask is not None:
            self.canvas.set_refined_mask(self.current_mask)
            self.canvas.set_mask_visible(True)
        else:
            self.canvas.set_mask_visible(False)

    def _set_run_button_state(self, state: str) -> None:
        """Configura el texto, estilo y disponibilidad del botón principal de ejecución."""
        if state == "running":
            self.btn_run.setText("🛑 CANCELAR PROCESAMIENTO")
            self.btn_run.setStyleSheet(STYLE_RUN_ACTIVE)
            self.btn_run.setEnabled(True)
        elif state == "cancelling":
            self.btn_run.setText("⏳ Cancelando procesamiento...")
            self.btn_run.setStyleSheet(STYLE_RUN_CANCELLING)
            self.btn_run.setEnabled(False)
        else: # idle
            self.btn_run.setText("⚡ Generar Trazas de Estrellas (Startrails)")
            self.btn_run.setStyleSheet(STYLE_RUN_IDLE)
            self.btn_run.setEnabled(True)

    def _cancel_processing(self) -> None:
        """Solicita la detención inmediata del procesamiento en segundo plano."""
        if self.worker is not None and self.worker.isRunning():
            self._set_run_button_state("cancelling")
            self.log("Cancelación solicitada por el usuario. Deteniendo procesos...")
            self.worker.cancel()

    def _on_worker_cancelled(self) -> None:
        """Maneja la confirmación de cancelación del worker."""
        self._set_run_button_state("idle")
        self.progress_bar.setVisible(False)
        self.log("Procesamiento de trazas cancelado por el usuario.")

    def _on_run_clicked(self) -> None:
        """Dispara la generación del Startrail en segundo plano o cancela si está en curso."""
        if self.worker is not None and self.worker.isRunning():
            self._cancel_processing()
            return

        if not self.files_list:
            QMessageBox.warning(self, "Atención", "Debes cargar al menos dos tomas para generar trazas.")
            return

        mode = "comet" if self.rb_comet.isChecked() else "lighten"
        comet_params = None
        if mode == "comet":
            comet_params = {
                "tail_pct": self.slider_tail.value(),
                "decay": self.combo_decay.currentData(),
                "direction": self.combo_dir.currentData(),
                "min_floor": self.slider_floor.value() / 100.0
            }

        use_clean_ground = self.chk_clean_ground.isChecked()
        if use_clean_ground and self.current_mask is None:
            ans = QMessageBox.question(
                self, "Máscara No Asignada",
                "Has activado 'Suelo Limpio' pero no has asignado ninguna máscara.\n\n"
                "¿Deseas continuar calculando el Startrail estándar sobre toda la imagen?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if ans != QMessageBox.Yes:
                return
            use_clean_ground = False

        timelapse_params = None
        if self.chk_enable_timelapse.isChecked():
            timelapse_params = {
                "enabled": True,
                "output_video_path": self.timelapse_custom_path,
                "fps": float(self.combo_tl_fps.currentData() or 24),
                "resolution": self.combo_tl_res.currentData() or "1080p",
                "save_frames": self.chk_tl_save_frames.isChecked()
            }

        color_boost_params = None
        if self.chk_star_color.isChecked():
            color_boost_params = {
                "enabled": True,
                "saturation": self.slider_star_sat.value() / 100.0,
                "core_recovery": self.slider_core_rec.value() / 100.0
            }

        params = {
            "files": self.files_list,
            "mode": mode,
            "comet_params": comet_params,
            "dark_files": self.dark_files if self.dark_files else None,
            "use_clean_ground": use_clean_ground,
            "mask": self.current_mask if use_clean_ground else None,
            "ground_mode": self.combo_gnd_mode.currentData(),
            "ref_idx": self.ref_idx,
            "feather_radius": self.slider_feather.value(),
            "suppress_streaks": self.chk_suppress_streaks.isChecked(),
            "streak_sensitivity": float(self.combo_streak_sens.currentData() or 0.5),
            "fill_gaps": self.chk_fill_gaps.isChecked(),
            "gap_fill_radius": self.slider_gap_radius.value(),
            "gap_fill_threshold_factor": float(self.combo_gap_sens.currentData() or 3.0),
            "timelapse_params": timelapse_params,
            "color_boost_params": color_boost_params
        }

        self.btn_play_timelapse.setEnabled(False)
        self._set_run_button_state("running")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log(f"Iniciando cálculo de trazas en modo: {mode.upper()}...")

        self.worker = StartrailWorker(params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.cancelled.connect(self._on_worker_cancelled)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, pct: int, msg: str) -> None:
        self.progress_bar.setValue(pct)
        if pct % 10 == 0 or pct >= 90:
            self.log(msg)

    def _on_worker_finished(self, data: Any, task_type: str) -> None:
        self._set_run_button_state("idle")
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

        if isinstance(data, (tuple, list)):
            result, video_path = data[0], data[1]
        else:
            result, video_path = data, None

        self.final_startrail_img = result
        self.raw_startrail_img = result.copy() if result is not None else None
        self.btn_apply_color_live.setEnabled(self.raw_startrail_img is not None)
        self.canvas.load_image(result)
        self._update_canvas_mask_overlay()
        self.log("Trazas de estrellas compuestas y renderizadas en el visor.")

        if video_path and os.path.exists(video_path):
            self.last_timelapse_path = video_path
            self.btn_play_timelapse.setEnabled(True)
            size_mb = os.path.getsize(video_path) / (1024.0 * 1024.0)
            self.log(f"🎬 Vídeo Time-Lapse generado con éxito ({size_mb:.2f} MB): {video_path}")

    def _on_worker_error(self, err_msg: str) -> None:
        self._set_run_button_state("idle")
        self.progress_bar.setVisible(False)
        self.log(f"ERROR: {err_msg}")
        QMessageBox.critical(self, "Error de Procesado", f"Ocurrió un error al generar las trazas:\n{err_msg}")

    def _on_star_color_toggled(self, checked: bool) -> None:
        """Muestra u oculta los controles de saturación y preservación cromática estelar."""
        self.container_star_color.setVisible(checked)

    def _on_apply_star_color_live_clicked(self) -> None:
        """Aplica el realce cromático de forma ultrarrápida (0.4s) sobre la circumpolar ya generada."""
        if self.raw_startrail_img is None:
            return

        sat = self.slider_star_sat.value() / 100.0
        core_rec = self.slider_core_rec.value() / 100.0

        self.log(f"Aplicando realce de color estelar (Sat: {sat:.1f}x, Recuperación: {int(core_rec*100)}%)...")
        try:
            enhanced = enhance_star_trails_color(
                self.raw_startrail_img,
                saturation_boost=sat if self.chk_star_color.isChecked() else 1.0,
                core_recovery=core_rec if self.chk_star_color.isChecked() else 0.0,
                sky_mask=self.current_mask if self.chk_clean_ground.isChecked() else None,
                bg_protect=True
            )
            self.final_startrail_img = enhanced
            self.canvas.load_image(enhanced)
            self._update_canvas_mask_overlay()
            self.log("Color estelar actualizado con éxito en el visor.")
        except Exception as e:
            self.log(f"Error al actualizar color estelar: {e}")
            QMessageBox.critical(self, "Error de Color", f"No se pudo aplicar el realce de color:\n{e}")

    def _on_timelapse_toggled(self, checked: bool) -> None:
        """Muestra u oculta los ajustes del generador de timelapse."""
        self.container_timelapse.setVisible(checked)

    def _on_select_timelapse_dest_clicked(self) -> None:
        """Permite al usuario elegir dónde guardar el vídeo Time-Lapse."""
        default_dir = os.path.dirname(self.files_list[0]) if self.files_list else os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self, "Seleccionar Destino del Vídeo Time-Lapse",
            os.path.join(default_dir, "timelapse_startrails.mp4"),
            "Vídeo MP4 (*.mp4);;Todos (*.*)"
        )
        if path:
            self.timelapse_custom_path = path
            self.lbl_tl_path.setText(f"Destino: {os.path.basename(path)}")
            self.lbl_tl_path.setStyleSheet("color: #81c784; font-size: 10px; font-weight: bold;")
            self.lbl_tl_path.setToolTip(path)

    def _on_play_timelapse_clicked(self) -> None:
        """Abre el vídeo Time-Lapse con el reproductor predeterminado del sistema operativo."""
        if self.last_timelapse_path and os.path.exists(self.last_timelapse_path):
            try:
                os.startfile(self.last_timelapse_path)
            except Exception as e:
                self.log(f"No se pudo reproducir el vídeo automáticamente: {e}")
                QMessageBox.information(
                    self, "Vídeo Generado",
                    f"El vídeo está guardado en:\n{self.last_timelapse_path}"
                )

    # -------------------------------------------------------------
    # EXPORTACIÓN
    # -------------------------------------------------------------
    def _get_best_image(self) -> Optional[np.ndarray]:
        if self.final_startrail_img is not None:
            return self.final_startrail_img
        return self.current_preview_img

    def _on_save_tiff_clicked(self, bit_depth: int = 16) -> None:
        img = self._get_best_image()
        if img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen disponible para guardar.")
            return

        default_name = f"startrail_{bit_depth}bit.tif"
        if bit_depth == 16:
            filters = "TIFF 16-bit (*.tif *.tiff);;TIFF 32-bit (*.tif *.tiff);;Todos (*.*)"
            title = "Guardar Trazas de Estrellas (TIFF 16-bit)"
        else:
            filters = "TIFF 32-bit (*.tif *.tiff);;TIFF 16-bit (*.tif *.tiff);;Todos (*.*)"
            title = "Guardar Trazas de Estrellas (TIFF 32-bit Float)"

        path, selected_filter = QFileDialog.getSaveFileName(self, title, default_name, filters)
        if not path:
            return

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
                f"Imagen de trazas guardada con éxito en:\n{path}\n\nFormato: {desc} ({out_img.shape[1]}x{out_img.shape[0]} px)"
            )
        except Exception as e:
            self.log(f"ERROR al guardar imagen: {e}")
            QMessageBox.critical(self, "Error de Exportación", f"No se pudo guardar la imagen:\n{e}")

    def _on_send_to_developer_clicked(self) -> None:
        """Transfiere la imagen generada directamente al Revelador / Editor."""
        img = self._get_best_image()
        if img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen disponible para transferir.")
            return

        out_dir = os.path.join(os.getcwd(), "exportaciones")
        os.makedirs(out_dir, exist_ok=True)
        temp_path = os.path.join(out_dir, "startrail_processed_32bit.tif")

        try:
            out_f32 = np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)
            save_tiff_with_icc(temp_path, out_f32, compression='zlib', photometric='rgb')
            self.log(f"Imagen transferida al Revelador: {temp_path}")
            self.export_to_developer.emit(temp_path)
        except Exception as e:
            self.log(f"ERROR al transferir al Revelador: {e}")
            QMessageBox.critical(self, "Error de Transferencia", f"No se pudo transferir al Revelador:\n{e}")
