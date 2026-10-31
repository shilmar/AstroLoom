# gui/tab_meteors.py
"""
gui/tab_meteors.py - Pestaña dedicada para la Detección, Clasificación y Composición de Lluvias de Meteoros.

Implementa:
1. Ingesta de serie temporal de tomas de lluvia de estrellas (RAW, TIFF, FITS, JPEG).
2. Detección automática de trazas transitorias y descarte de balizas de aviación / satélites.
3. Reproyección astronómica homográfica de cada meteoro para compensar la rotación terrestre.
4. Estimación y visualización del radiante celeste con presets astronómicos (Perseidas, Gemínidas, etc.).
5. Tira de inspección y culling rápido para validación visual de meteoros reales vs. artefactos.
6. Fusión fotométrica de flujo neto sin degradación de SNR sobre fondo limpio apilado.
7. Exportación en TIFF 16-bit / 32-bit, PSD multicapa nativo y transferencia al Revelador.
"""

import os
import time
from typing import Optional, List, Dict, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tifffile
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QMessageBox, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar, QTextEdit, QScrollArea, QSplitter, QFrame, QSizePolicy
)

from core.stacking import load_image_as_float32
from core.meteors import (
    MeteorCandidate,
    MeteorSessionConfig,
    METEOR_SHOWER_PRESETS,
    detect_streaks_in_frame,
    compute_frame_homography_to_ref,
    reproject_candidate_to_ref,
    estimate_radiant_intersection,
    classify_sporadics,
    refine_satellite_classifications,
    compose_meteor_shower_image,
    export_meteor_layered_psd,
    save_meteor_session,
    load_meteor_session,
    load_fast_meteor_frame,
    ThreadSafeFrameCache,
    extract_frame_timestamp
)
from core.panorama import extract_bright_stars, build_star_asterisms
from core.config_manager import get_config_val
from core.icc_manager import save_tiff_with_icc
from gui.canvas import MaskCanvas
from gui.worker import AIHorizonWorker
from gui.widgets.drop_widgets import DropTableWidget
from gui.theme import (
    STYLE_RUN_IDLE, STYLE_RUN_ACTIVE, STYLE_RUN_CANCELLING, STYLE_TRANSFER_BTN
)
from gui.icons import get_icon
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow


class MeteorWorker(QThread):
    """Worker en segundo plano para escaneo y composición de lluvias de meteoros."""
    progress = Signal(int, str)
    finished = Signal(object, str)
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, task_type: str, params: Dict[str, Any]):
        super().__init__()
        self.task_type = task_type
        self.params = params
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self.task_type == "scan_meteors":
                self._run_scan()
            elif self.task_type == "fuse_meteors":
                self._run_fuse()
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(str(e))

    def _run_scan(self):
        files: List[str] = self.params["files"]
        ref_idx: int = self.params.get("ref_idx", 0)
        sens_sigma: float = self.params.get("sensitivity_sigma", 4.2)
        min_len: float = self.params.get("min_length_px", 24.0)
        max_len: float = self.params.get("max_length_px", 2500.0)
        filter_airplanes: bool = self.params.get("filter_airplanes", True)
        filter_satellites: bool = self.params.get("filter_satellites", True)
        tol_deg: float = self.params.get("radiant_tolerance_deg", 7.0)
        ground_mask: Optional[np.ndarray] = self.params.get("ground_mask")

        total = len(files)
        if total < 2:
            raise ValueError("Se requieren al menos 2 tomas para escanear trazas diferenciales.")

        self.progress.emit(2, f"Cargando toma de referencia maestra ({os.path.basename(files[ref_idx])})...")
        ref_img_scan, scale_ref = load_fast_meteor_frame(files[ref_idx])
        h_ref_scan, w_ref_scan = ref_img_scan.shape[:2]
        h_ref_full = int(round(h_ref_scan * scale_ref))
        w_ref_full = int(round(w_ref_scan * scale_ref))

        self.progress.emit(5, "Extrayendo asterismos estelares de referencia...")
        pts_ref = extract_bright_stars(ref_img_scan, max_stars=90)
        feats_ref, quads_ref = (None, None)
        if len(pts_ref) >= 6:
            feats_ref, quads_ref = build_star_asterisms(pts_ref)
        ref_asterisms = (feats_ref, quads_ref, pts_ref) if (feats_ref is not None) else None

        # Configuración de hilos concurrentes optimizada
        cfg_threads = get_config_val("performance.max_cpu_threads", None)
        if cfg_threads and isinstance(cfg_threads, int) and cfg_threads > 0:
            eff_workers = min(cfg_threads, total)
        else:
            cpu_cnt = os.cpu_count() or 4
            eff_workers = min(max(2, cpu_cnt - 1), 8, total)

        frame_cache = ThreadSafeFrameCache(max_items=max(eff_workers * 2, 8))

        def _process_single_frame(idx: int) -> Tuple[int, List[MeteorCandidate], Optional[np.ndarray]]:
            if self._is_cancelled:
                return idx, [], None

            target_fp = files[idx]
            neighbor_idx = idx - 1 if idx > 0 else min(1, total - 1)
            neighbor_fp = files[neighbor_idx]

            curr_img, scale_factor = frame_cache.get_or_load(target_fp)
            bg_img, _ = frame_cache.get_or_load(neighbor_fp)

            cands = detect_streaks_in_frame(
                img_curr=curr_img,
                img_bg=bg_img,
                filepath=target_fp,
                frame_idx=idx,
                start_id=1,
                sensitivity_sigma=sens_sigma,
                min_length_px=min_len,
                max_length_px=max_len,
                filter_airplanes=filter_airplanes,
                filter_satellites=filter_satellites,
                scale_factor=scale_factor,
                ground_mask=ground_mask
            )

            H_res = None
            if cands:
                if idx == ref_idx:
                    H_res = np.eye(3, dtype=np.float32)
                else:
                    H_res = compute_frame_homography_to_ref(
                        img_target=curr_img,
                        img_ref=ref_img_scan,
                        ref_asterisms=ref_asterisms,
                        scale_factor=scale_factor
                    )

                for c in cands:
                    reproject_candidate_to_ref(c, H_res)

            return idx, cands, H_res

        self.progress.emit(10, f"Iniciando escaneo paralelo ({eff_workers} hilos CPU con Numba)...")
        completed_tasks: List[Tuple[int, List[MeteorCandidate], Optional[np.ndarray]]] = []

        with ThreadPoolExecutor(max_workers=eff_workers) as executor:
            future_to_idx = {executor.submit(_process_single_frame, i): i for i in range(total)}
            done_count = 0
            for fut in as_completed(future_to_idx):
                if self._is_cancelled:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise InterruptedError("Operación cancelada por el usuario.")

                idx_res, cands_res, H_res = fut.result()
                completed_tasks.append((idx_res, cands_res, H_res))
                done_count += 1
                prog = int(10 + 75 * (done_count / total))
                self.progress.emit(prog, f"Escaneadas ({done_count}/{total}) tomas: {os.path.basename(files[idx_res])}...")

        # Ordenar por índice de fotograma y consolidar
        completed_tasks.sort(key=lambda t: t[0])
        all_candidates: List[MeteorCandidate] = []
        homographies_map: Dict[int, Optional[np.ndarray]] = {ref_idx: np.eye(3, dtype=np.float32)}
        next_cand_id = 1

        for idx_res, cands_res, H_res in completed_tasks:
            if H_res is not None:
                homographies_map[idx_res] = H_res
            for c in cands_res:
                c.id = next_cand_id
                next_cand_id += 1
                all_candidates.append(c)

        # Refinar satélites multicuadro y cruces orbitales (Propuestas 2 y 3A)
        self.progress.emit(86, "Analizando persistencia orbital y cruces de satélites...")
        refine_satellite_classifications(all_candidates, filter_satellites=filter_satellites)

        # Estimar punto radiante óptimo por intersección
        self.progress.emit(88, "Calculando punto radiante celeste óptimo...")
        radiant_pt = estimate_radiant_intersection(all_candidates, (h_ref_full, w_ref_full))

        if radiant_pt is not None:
            self.progress.emit(94, f"Radiante estimado en ({radiant_pt[0]:.1f}, {radiant_pt[1]:.1f}) px. Clasificando esporádicos...")
            classify_sporadics(all_candidates, radiant_pt, tolerance_deg=tol_deg)

        self.progress.emit(100, f"Escaneo finalizado: {len(all_candidates)} trazas detectadas.")
        result = {
            "candidates": all_candidates,
            "radiant_pt": radiant_pt,
            "homographies": homographies_map,
            "ref_img": ref_img_scan
        }
        self.finished.emit(result, "scan_meteors")

    def _run_fuse(self):
        base_img: np.ndarray = self.params["base_img"]
        candidates: List[MeteorCandidate] = self.params["candidates"]
        homographies_map: Dict[int, Optional[np.ndarray]] = self.params["homographies"]
        boost: float = self.params.get("boost_flux", 1.0)
        include_sporadics: bool = self.params.get("include_sporadics", False)
        ground_mask: Optional[np.ndarray] = self.params.get("ground_mask")
        feather: int = self.params.get("feather_px", 10)

        fused = compose_meteor_shower_image(
            base_image=base_img,
            candidates=candidates,
            homographies_map=homographies_map,
            boost_flux=boost,
            include_sporadics=include_sporadics,
            ground_mask=ground_mask,
            feather_px=feather,
            progress_callback=self.progress.emit
        )
        self.finished.emit(fused, "fuse_meteors")


class MeteorTab(QWidget):
    """Pestaña maestra para la detección y composición de lluvias de meteoros."""
    export_to_developer = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.files_list: List[str] = []
        self.ref_idx: int = 0
        self.candidates: List[MeteorCandidate] = []
        self.homographies_map: Dict[int, Optional[np.ndarray]] = {}
        self.radiant_point: Optional[Tuple[float, float]] = None
        self.base_reference_img: Optional[np.ndarray] = None
        self.custom_stacked_sky: Optional[np.ndarray] = None
        self.composite_result: Optional[np.ndarray] = None
        self.current_preview_img: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self._mask_provider: Optional[Callable[[], Optional[np.ndarray]]] = None
        self.worker: Optional[MeteorWorker] = None
        self.ai_worker: Optional[AIHorizonWorker] = None

        self._setup_ui()

    def set_stacker_mask_provider(self, provider: Callable[[], Optional[np.ndarray]]) -> None:
        """Permite importar la máscara de horizonte calculada en el Apilador."""
        self._mask_provider = provider

    def _setup_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter_h = QSplitter(Qt.Horizontal)
        splitter_h.setChildrenCollapsible(False)

        # -------------------------------------------------------------
        # PANEL IZQUIERDO: Acordeón de Configuración y Control
        # -------------------------------------------------------------
        left_container = QWidget()
        left_container.setMinimumWidth(410)
        left_container.setMaximumWidth(490)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        panel_layout = QVBoxLayout(scroll_content)
        panel_layout.setContentsMargins(2, 2, 6, 2)
        panel_layout.setSpacing(8)

        # SECCIÓN 1: Ingesta de Serie y Toma Maestra
        sec_ingest = CollapsibleSection("1. Ingesta de Serie Temporal", is_expanded=True, icon_name="camera")
        ingest_l = sec_ingest.content_layout

        btn_row_load = QHBoxLayout()
        self.btn_load_series = QPushButton("Cargar Serie...")
        self.btn_load_series.setIcon(get_icon("folder", color="#80d8ff"))
        self.btn_load_series.clicked.connect(self._on_load_series_clicked)

        self.btn_clear_series = QPushButton("Limpiar")
        self.btn_clear_series.setIcon(get_icon("trash", color="#ff8a80"))
        self.btn_clear_series.clicked.connect(self._on_clear_series_clicked)

        self.btn_set_ref = QPushButton("Marcar REF")
        self.btn_set_ref.setIcon(get_icon("target", color="#00e5ff"))
        self.btn_set_ref.setToolTip("Establece la toma seleccionada como marco de referencia astronómica y composición")
        self.btn_set_ref.clicked.connect(self._on_set_ref_clicked)

        btn_row_load.addWidget(self.btn_load_series)
        btn_row_load.addWidget(self.btn_clear_series)
        btn_row_load.addWidget(self.btn_set_ref)
        ingest_l.addLayout(btn_row_load)

        self.table_files = DropTableWidget(0, 3)
        self.table_files.setHorizontalHeaderLabels(["REF", "Archivo", "Timestamp / Fecha"])
        self.table_files.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_files.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_files.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_files.setFixedHeight(140)
        self.table_files.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table_files.files_dropped.connect(self.add_series_files)
        ingest_l.addWidget(self.table_files)

        panel_layout.addWidget(sec_ingest)

        # SECCIÓN 2: Máscara de Horizonte y Suelo (Prioridad antes del Escaneo)
        sec_mask = CollapsibleSection("2. Máscara de Horizonte y Suelo", is_expanded=True, icon_name="brush")
        mask_l = sec_mask.content_layout

        row_mask_btns = QHBoxLayout()
        self.btn_detect_ai_mask = QPushButton("Detectar por IA")
        self.btn_detect_ai_mask.setIcon(get_icon("sparkles", color="#80d8ff"))
        self.btn_detect_ai_mask.setToolTip("Segmentación semántica neuronal local (U²-Net) para delimitar el cielo y suelo automáticamente en 1 segundo.")
        self.btn_detect_ai_mask.clicked.connect(self._on_detect_ai_mask_clicked)

        self.btn_load_mask_file = QPushButton("Cargar Archivo...")
        self.btn_load_mask_file.setIcon(get_icon("folder", color="#ffd54f"))
        self.btn_load_mask_file.setToolTip("Carga una imagen de máscara desde archivo (PNG, TIFF, JPG).")
        self.btn_load_mask_file.clicked.connect(self._on_load_mask_file_clicked)

        self.btn_import_mask = QPushButton("Del Apilador")
        self.btn_import_mask.setIcon(get_icon("layers", color="#b39ddb"))
        self.btn_import_mask.setToolTip("Importa la máscara de horizonte calculada en el Apilador (Pestaña 1).")
        self.btn_import_mask.clicked.connect(self._on_import_mask_clicked)

        row_mask_btns.addWidget(self.btn_detect_ai_mask)
        row_mask_btns.addWidget(self.btn_load_mask_file)
        row_mask_btns.addWidget(self.btn_import_mask)
        mask_l.addLayout(row_mask_btns)

        row_mask_opts = QHBoxLayout()
        self.chk_show_mask = QCheckBox("Mostrar Máscara en Visor")
        self.chk_show_mask.setChecked(False)
        self.chk_show_mask.setToolTip("Superpone la máscara en tono rubí sobre la toma en el visor para inspección visual directa.")
        self.chk_show_mask.toggled.connect(self._on_toggle_show_mask)

        self.btn_clear_mask = QPushButton("Limpiar")
        self.btn_clear_mask.setIcon(get_icon("trash", color="#ef5350"))
        self.btn_clear_mask.setToolTip("Elimina la máscara actual y permite el escaneo en todo el encuadre.")
        self.btn_clear_mask.clicked.connect(self._on_clear_mask_clicked)

        row_mask_opts.addWidget(self.chk_show_mask)
        row_mask_opts.addStretch()
        row_mask_opts.addWidget(self.btn_clear_mask)
        mask_l.addLayout(row_mask_opts)

        self.lbl_mask_status = QLabel("Máscara: Ninguna (suelo no protegido)")
        self.lbl_mask_status.setStyleSheet("color: #888888; font-size: 11px;")
        mask_l.addWidget(self.lbl_mask_status)

        panel_layout.addWidget(sec_mask)

        # SECCIÓN 3: Detector Automático de Trazas y Bólidos
        sec_detect = CollapsibleSection("3. Detector de Meteoros y Trazas", is_expanded=True, icon_name="zap")
        det_l = sec_detect.content_layout

        self.slider_sens = PrecisionSliderRow(
            title="Sensibilidad Detección",
            min_val=25,
            max_val=70,
            default_val=42,
            scale=10.0,
            unit=" σ",
            decimals=1,
            label_width=135
        )
        self.slider_sens.setToolTip("Umbral estadístico de corte en sigma. Un valor menor detecta meteoros más débiles; un valor mayor suprime ruido térmico.")
        det_l.addWidget(self.slider_sens)

        self.slider_min_len = PrecisionSliderRow(
            title="Longitud Mínima",
            min_val=10,
            max_val=150,
            default_val=24,
            scale=1.0,
            unit=" px",
            decimals=0,
            label_width=135
        )
        det_l.addWidget(self.slider_min_len)

        self.chk_filter_planes = QCheckBox("Descartar balizas de aviones (Rojo/Verde)")
        self.chk_filter_planes.setChecked(True)
        self.chk_filter_planes.setToolTip("Identifica automáticamente luces estroboscópicas rojas y verdes puras de navegación aérea.")
        det_l.addWidget(self.chk_filter_planes)

        self.chk_filter_sat = QCheckBox("Descartar trazas de satélites (Starlink)")
        self.chk_filter_sat.setChecked(True)
        self.chk_filter_sat.setToolTip("Descarta trazas ultralargas uniformes de borde a borde sin perfil asimétrico meteórico.")
        det_l.addWidget(self.chk_filter_sat)

        self.lbl_scan_mask_info = QLabel("Máscara: Ninguna (escaneo en todo el encuadre)")
        self.lbl_scan_mask_info.setStyleSheet("color: #888888; font-size: 11px;")
        det_l.addWidget(self.lbl_scan_mask_info)

        self.btn_scan = QPushButton("ESCANEAR SERIE DE METEOROS")
        self.btn_scan.setIcon(get_icon("search", color="#ffffff"))
        self.btn_scan.setFixedHeight(36)
        self.btn_scan.setCursor(Qt.PointingHandCursor)
        self.btn_scan.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_scan.clicked.connect(self._on_scan_clicked)
        det_l.addWidget(self.btn_scan)

        panel_layout.addWidget(sec_detect)

        # SECCIÓN 4: Radiante Celeste y Clasificación
        sec_radiant = CollapsibleSection("4. Radiante Celeste y Reproyección", is_expanded=True, icon_name="sparkles")
        rad_l = sec_radiant.content_layout

        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("Lluvia:"))
        self.combo_shower = QComboBox()
        for k, v in METEOR_SHOWER_PRESETS.items():
            self.combo_shower.addItem(f"{k} ({v['name']})", k)
        row_preset.addWidget(self.combo_shower)
        rad_l.addLayout(row_preset)

        self.slider_tol = PrecisionSliderRow(
            title="Tolerancia Radiante",
            min_val=2,
            max_val=20,
            default_val=7,
            unit="°",
            decimals=0,
            label_width=135
        )
        self.slider_tol.setToolTip("Divergencia máxima angular permitida respecto al radiante para clasificar como miembro de la lluvia.")
        rad_l.addWidget(self.slider_tol)

        self.chk_include_sporadics = QCheckBox("Incluir bólidos esporádicos en compuesto")
        self.chk_include_sporadics.setChecked(True)
        self.chk_include_sporadics.setToolTip("Si se desmarca, solo se fundirán los meteoros que convergen físicamente hacia el radiante.")
        rad_l.addWidget(self.chk_include_sporadics)

        self.lbl_radiant_info = QLabel("Radiante: No calculado (ejecuta el escaneo)")
        self.lbl_radiant_info.setStyleSheet("color: #80d8ff; font-size: 11px; font-weight: bold;")
        rad_l.addWidget(self.lbl_radiant_info)

        panel_layout.addWidget(sec_radiant)

        # SECCIÓN 5: Fusión Fotométrica y Acabado
        sec_fuse = CollapsibleSection("5. Fusión Fotométrica y Acabado", is_expanded=True, icon_name="sliders")
        fuse_l = sec_fuse.content_layout

        row_sky = QHBoxLayout()
        self.btn_load_sky_stack = QPushButton("Cargar Cielo Apilado...")
        self.btn_load_sky_stack.setIcon(get_icon("layers", color="#b39ddb"))
        self.btn_load_sky_stack.setToolTip("Opcional: Si tienes una imagen del cielo previamente apilada sin ruido (Pestaña 1), úsala como fondo maestro.")
        self.btn_load_sky_stack.clicked.connect(self._on_load_sky_stack_clicked)
        row_sky.addWidget(self.btn_load_sky_stack)
        fuse_l.addLayout(row_sky)

        self.slider_boost = PrecisionSliderRow(
            title="Realce de Flujo",
            min_val=10,
            max_val=300,
            default_val=100,
            scale=100.0,
            unit="x",
            decimals=2,
            label_width=135
        )
        fuse_l.addWidget(self.slider_boost)

        self.btn_fuse = QPushButton("FUSIONAR LLUVIA DE METEOROS")
        self.btn_fuse.setIcon(get_icon("sparkles", color="#ffffff"))
        self.btn_fuse.setFixedHeight(40)
        self.btn_fuse.setCursor(Qt.PointingHandCursor)
        self.btn_fuse.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_fuse.clicked.connect(self._on_fuse_clicked)
        fuse_l.addWidget(self.btn_fuse)

        panel_layout.addWidget(sec_fuse)

        # SECCIÓN 6: Exportación y Entrega
        sec_export = CollapsibleSection("6. Exportación y Entrega", is_expanded=True, icon_name="save")
        exp_l = sec_export.content_layout

        btn_row_exp = QHBoxLayout()
        self.btn_save_tiff = QPushButton("Guardar TIFF 32-bit...")
        self.btn_save_tiff.setIcon(get_icon("save", color="#90caf9"))
        self.btn_save_tiff.clicked.connect(self._on_save_tiff_clicked)

        self.btn_save_psd = QPushButton("Exportar PSD Capas...")
        self.btn_save_psd.setIcon(get_icon("layers", color="#ce93d8"))
        self.btn_save_psd.setToolTip("Exporta un archivo Photoshop con cada meteoro en su propia capa Screen para control individual de opacidad.")
        self.btn_save_psd.clicked.connect(self._on_save_psd_clicked)

        btn_row_exp.addWidget(self.btn_save_tiff)
        btn_row_exp.addWidget(self.btn_save_psd)
        exp_l.addLayout(btn_row_exp)

        self.btn_send_dev = QPushButton("Enviar al Revelador / Editor")
        self.btn_send_dev.setIcon(get_icon("sliders", color="#ffffff"))
        self.btn_send_dev.setStyleSheet(STYLE_TRANSFER_BTN)
        self.btn_send_dev.clicked.connect(self._on_send_to_developer_clicked)
        exp_l.addWidget(self.btn_send_dev)

        panel_layout.addWidget(sec_export)
        panel_layout.addStretch()

        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)

        # Barra de progreso y consola inferior izquierda
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(85)
        left_layout.addWidget(self.txt_log)

        # -------------------------------------------------------------
        # PANEL DERECHO: Canvas Interactivo y Galería de Culling
        # -------------------------------------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(4)

        # Barra superior del visor con selector de visualización calibrada
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(6, 2, 6, 2)
        top_bar.setSpacing(8)

        lbl_view = QLabel("Visualización:")
        lbl_view.setStyleSheet("font-weight: bold; color: #80d8ff;")
        top_bar.addWidget(lbl_view)

        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems([
            "Natural / sRGB (Fiel al Revelador)",
            "Auto-MTF (Cielo Profundo)",
            "Realce de Trazas (Alto Contraste)"
        ])
        self.combo_view_mode.setToolTip("Ajusta la representación visual en pantalla sin alterar los datos científicos de 32 bits ni la exportación.")
        self.combo_view_mode.currentIndexChanged.connect(self._on_view_mode_changed)
        top_bar.addWidget(self.combo_view_mode)

        self.btn_view_composite = QPushButton("Ver Compuesto")
        self.btn_view_composite.setIcon(get_icon("layers", color="#ffffff"))
        self.btn_view_composite.setCheckable(True)
        self.btn_view_composite.setChecked(False)
        self.btn_view_composite.setEnabled(False)
        self.btn_view_composite.setToolTip("Alterna la visualización entre el compuesto final con todos los meteoros y los fotogramas individuales de la serie.")
        self.btn_view_composite.toggled.connect(self._on_toggle_view_composite)
        self.btn_view_composite.setStyleSheet("""
            QPushButton {
                background-color: #1e2638;
                color: #eceff4;
                border: 1px solid #3a415a;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #28324a;
                border-color: #00bcd4;
            }
            QPushButton:checked {
                background-color: #00838f;
                color: #ffffff;
                border: 1px solid #00e5ff;
            }
            QPushButton:disabled {
                background-color: #12141e;
                color: #546e7a;
                border-color: #232738;
            }
        """)
        top_bar.addWidget(self.btn_view_composite)

        top_bar.addStretch()

        self.lbl_view_status = QLabel("")
        self.lbl_view_status.setStyleSheet("color: #b0bec5; font-size: 11px;")
        top_bar.addWidget(self.lbl_view_status)

        right_layout.addLayout(top_bar)

        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.setChildrenCollapsible(False)

        # 1. Visor Canvas
        self.canvas = MaskCanvas(self, enable_masking=False)
        self.canvas.files_dropped.connect(self.add_series_files)
        splitter_v.addWidget(self.canvas)

        # 2. Tira Inferior: Galería de Culling e Inspección de Trazas
        culling_box = QGroupBox("Galería de Trazas Detectadas (Culling Interactivo)")
        culling_box.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #80d8ff; margin-top: 4px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        cull_l = QVBoxLayout(culling_box)
        cull_l.setContentsMargins(6, 6, 6, 6)
        cull_l.setSpacing(6)

        # Barra de herramientas de culling rápido
        bar_cull = QHBoxLayout()
        self.btn_select_all = QPushButton("Incluir Todos")
        self.btn_select_all.clicked.connect(lambda: self._set_cull_filter_state("all", True))
        self.btn_deselect_planes = QPushButton("Descartar Aviones/Satélites")
        self.btn_deselect_planes.clicked.connect(lambda: self._set_cull_filter_state("planes", False))
        self.btn_deselect_sporadics = QPushButton("Descartar Esporádicos")
        self.btn_deselect_sporadics.clicked.connect(lambda: self._set_cull_filter_state("sporadics", False))

        bar_cull.addWidget(self.btn_select_all)
        bar_cull.addWidget(self.btn_deselect_planes)
        bar_cull.addWidget(self.btn_deselect_sporadics)
        bar_cull.addSpacing(10)
        self.chk_highlight_streak = QCheckBox("🎯 Resaltar Traza")
        self.chk_highlight_streak.setChecked(True)
        self.chk_highlight_streak.setToolTip("Dibuja un óvalo / cápsula luminosa alrededor de la traza seleccionada")
        self.chk_highlight_streak.toggled.connect(self._on_toggle_highlight)
        bar_cull.addWidget(self.chk_highlight_streak)

        self.chk_highlight_all = QCheckBox("👁️ Resaltar Todas")
        self.chk_highlight_all.setChecked(False)
        self.chk_highlight_all.setToolTip("Muestra cápsulas y etiquetas luminosas sobre todas las trazas incluidas simultáneamente en el visor")
        self.chk_highlight_all.toggled.connect(self._on_toggle_highlight_all)
        bar_cull.addWidget(self.chk_highlight_all)

        self.btn_zoom_streak = QPushButton("🔍 Centrar Traza (1:1)")
        self.btn_zoom_streak.setToolTip("Centra y amplía la vista al 100% exactamente sobre la traza seleccionada")
        self.btn_zoom_streak.clicked.connect(self._on_center_streak_clicked)
        bar_cull.addWidget(self.btn_zoom_streak)

        bar_cull.addStretch()
        self.lbl_cand_summary = QLabel("0 trazas detectadas")
        self.lbl_cand_summary.setStyleSheet("color: #ffd54f; font-weight: bold;")
        bar_cull.addWidget(self.lbl_cand_summary)
        cull_l.addLayout(bar_cull)

        # Tabla de candidatos
        self.table_cull = QTableWidget(0, 8)
        self.table_cull.setHorizontalHeaderLabels([
            "Incluir", "ID", "Fotograma", "Clasificación",
            "Longitud", "Ángulo", "Brillo Pico", "Divergencia Radiante"
        ])
        self.table_cull.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_cull.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_cull.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_cull.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_cull.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_cull.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_cull.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table_cull.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.table_cull.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_cull.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_cull.itemChanged.connect(self._on_cull_item_changed)
        self.table_cull.itemSelectionChanged.connect(self._on_cull_selection_changed)
        self.table_cull.cellDoubleClicked.connect(lambda r, c: self._on_center_streak_clicked())
        cull_l.addWidget(self.table_cull)

        splitter_v.addWidget(culling_box)
        splitter_v.setSizes([600, 220])

        right_layout.addWidget(splitter_v)

        splitter_h.addWidget(left_container)
        splitter_h.addWidget(right_container)
        splitter_h.setStretchFactor(0, 0)
        splitter_h.setStretchFactor(1, 10)
        splitter_h.setSizes([450, 1350])

        main_layout.addWidget(splitter_h)

    def log(self, msg: str) -> None:
        self.txt_log.append(msg)
        self.txt_log.moveCursor(QTextCursor.End)

    # -------------------------------------------------------------
    # EVENTOS DE CARGA Y TABLA DE ARCHIVOS
    # -------------------------------------------------------------
    def add_series_files(self, paths: List[str]) -> None:
        """Añade tomas de la serie temporal mediante diálogo o arrastrar y soltar (Drag and Drop)."""
        valid_exts = {
            ".tif", ".tiff", ".fit", ".fits", ".fts",
            ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raw", ".raf", ".orf", ".rw2",
            ".jpg", ".jpeg", ".png"
        }
        clean_paths = [os.path.abspath(p) for p in paths if os.path.splitext(p)[1].lower() in valid_exts]
        if not clean_paths:
            return

        all_set = set(self.files_list)
        all_set.update(clean_paths)
        self.files_list = sorted(list(all_set))
        self.ref_idx = min(self.ref_idx, max(0, len(self.files_list) - 1))

        self._refresh_files_table()
        self.log(f"Serie de meteoros: {len(self.files_list)} tomas registradas.")

        if self.files_list:
            self._preview_frame(self.ref_idx)

    def _on_load_series_clicked(self) -> None:
        filters = "Imágenes Astrofotográficas (*.tif *.tiff *.fit *.fits *.fts *.arw *.cr2 *.cr3 *.nef *.dng *.raw *.raf *.orf *.rw2 *.jpg *.jpeg *.png);;Todos (*.*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Cargar Serie de Lluvia de Meteoros", "", filters)
        if paths:
            self.add_series_files(paths)

    def _on_clear_series_clicked(self) -> None:
        self.files_list.clear()
        self.candidates.clear()
        self.ref_idx = 0
        self._refresh_files_table()
        self._refresh_cull_table()
        self.composite_result = None
        self.base_reference_img = None
        self._base_ref_is_fast_preview = False
        self.current_preview_img = None
        if hasattr(self, 'btn_view_composite'):
            self.btn_view_composite.setChecked(False)
            self.btn_view_composite.setEnabled(False)
        self.canvas.clear_image()
        self.canvas.clear_multi_highlight_reticles()
        self.lbl_view_status.setText("")
        self.log("Serie de meteoros vaciada.")

    def _on_set_ref_clicked(self) -> None:
        rows = self.table_files.selectionModel().selectedRows()
        if rows:
            self.ref_idx = rows[0].row()
            self._refresh_files_table()
            self._preview_frame(self.ref_idx)
            self.log(f"Toma de referencia maestra establecida: {os.path.basename(self.files_list[self.ref_idx])}")

    def _refresh_files_table(self) -> None:
        self.table_files.blockSignals(True)
        self.table_files.setRowCount(len(self.files_list))
        for row, fp in enumerate(self.files_list):
            is_ref = (row == self.ref_idx)
            ref_item = QTableWidgetItem("★ REF" if is_ref else "")
            ref_item.setTextAlignment(Qt.AlignCenter)
            if is_ref:
                ref_item.setForeground(QColor("#00e5ff"))

            fname_item = QTableWidgetItem(os.path.basename(fp))
            time_str = "--:--:--"
            try:
                ts = extract_frame_timestamp(fp)
                if ts > 0:
                    time_str = time.strftime("%H:%M:%S", time.localtime(ts))
            except Exception:
                time_str = "--:--:--"
            time_item = QTableWidgetItem(time_str)

            self.table_files.setItem(row, 0, ref_item)
            self.table_files.setItem(row, 1, fname_item)
            self.table_files.setItem(row, 2, time_item)

        self.table_files.blockSignals(False)

    def _on_table_selection_changed(self) -> None:
        rows = self.table_files.selectionModel().selectedRows()
        if rows and rows[0].row() < len(self.files_list):
            idx = rows[0].row()
            self._preview_frame(idx)

    def _on_view_mode_changed(self) -> None:
        if self.current_preview_img is not None:
            self._display_image_in_canvas(self.current_preview_img)

    def _compute_display_image(self, img: np.ndarray) -> np.ndarray:
        """Calcula la representación en pantalla en uint8 según el modo de visualización elegido."""
        idx = self.combo_view_mode.currentIndex()
        try:
            if idx == 0:
                # 1. Modo Natural / sRGB (Fiel al Revelador y a la cámara, sin doble gamma)
                return cv2.convertScaleAbs(np.clip(img, 0.0, 1.0), alpha=255.0)

            elif idx == 1:
                # 2. Modo Auto-MTF (Cielo Profundo / FITS lineal)
                from core.stretch import calculate_mtf_params, manual_stretch
                bp, m = calculate_mtf_params(img, target_background=0.18)
                stretched = manual_stretch(img, black_point=bp, midtone=m)
                return cv2.convertScaleAbs(np.clip(stretched, 0.0, 1.0), alpha=255.0)

            elif idx == 2:
                # 3. Modo Realce de Trazas (Alto Contraste y realce adaptativo de trazas tenues)
                sample = img[::4, ::4]
                med = float(np.median(sample))
                mad = float(np.median(np.abs(sample - med)))
                bp = float(np.clip(med - 0.5 * mad, 0.0, 0.9))
                clipped = np.clip((img - bp) / max(1e-4, 1.0 - bp), 0.0, 1.0)
                if clipped.ndim == 3:
                    hsv = cv2.cvtColor((clipped * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    hsv[..., 2] = clahe.apply(hsv[..., 2])
                    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                else:
                    gray = (clipped * 255.0).astype(np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    return clahe.apply(gray)
        except Exception:
            return cv2.convertScaleAbs(np.clip(img, 0.0, 1.0), alpha=255.0)
        return cv2.convertScaleAbs(np.clip(img, 0.0, 1.0), alpha=255.0)

    def _display_image_in_canvas(self, img: np.ndarray) -> None:
        """Transfiere un array float32 al canvas utilizando la visualización calibrada."""
        self.current_preview_img = img
        display_u8 = self._compute_display_image(img)
        self.canvas.load_image(img, display_stretched=display_u8)

    def _preview_frame(self, idx: int) -> None:
        try:
            img = load_image_as_float32(self.files_list[idx])
            if idx == self.ref_idx:
                self.base_reference_img = img.copy()
                self._base_ref_is_fast_preview = False
            self._display_image_in_canvas(img)
            self.lbl_view_status.setText(f"Toma ({idx + 1}/{len(self.files_list)}): {os.path.basename(self.files_list[idx])}")
            self._update_multi_highlight_reticles()
        except Exception as e:
            self.log(f"Error cargando previsualización: {e}")

    # -------------------------------------------------------------
    # ESCANEO AUTOMÁTICO Y WORKER
    # -------------------------------------------------------------
    def _on_scan_clicked(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.btn_scan.setText("Cancelando...")
            return

        if len(self.files_list) < 2:
            QMessageBox.warning(self, "Atención", "Carga al menos 2 tomas de la serie para detectar trazas de meteoros.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_scan.setText("DETENER ESCANEO")
        self.btn_scan.setStyleSheet(STYLE_RUN_ACTIVE)
        self.btn_fuse.setEnabled(False)

        params = {
            "files": self.files_list,
            "ref_idx": self.ref_idx,
            "sensitivity_sigma": self.slider_sens.float_value(),
            "min_length_px": self.slider_min_len.float_value(),
            "filter_airplanes": self.chk_filter_planes.isChecked(),
            "filter_satellites": self.chk_filter_sat.isChecked(),
            "radiant_tolerance_deg": self.slider_tol.float_value(),
            "ground_mask": self.current_mask
        }

        self.worker = MeteorWorker("scan_meteors", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.cancelled.connect(self._on_worker_cancelled)
        self.worker.start()

    def _on_fuse_clicked(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        active_cands = [c for c in self.candidates if c.is_meteor]
        if not active_cands:
            QMessageBox.warning(self, "Atención", "No hay meteoros seleccionados en la galería para componer.")
            return

        if self.custom_stacked_sky is not None:
            base_img = self.custom_stacked_sky
        elif self.files_list:
            if self.base_reference_img is None or getattr(self, '_base_ref_is_fast_preview', False):
                self.log("Cargando fotograma maestro a resolución nativa completa para composición...")
                base_img = load_image_as_float32(self.files_list[self.ref_idx])
                self.base_reference_img = base_img
                self._base_ref_is_fast_preview = False
            else:
                base_img = self.base_reference_img
        else:
            base_img = None

        if base_img is None:
            QMessageBox.warning(self, "Atención", "No hay imagen base de referencia cargada.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_fuse.setEnabled(False)
        self.btn_scan.setEnabled(False)

        params = {
            "base_img": base_img,
            "candidates": self.candidates,
            "homographies": self.homographies_map,
            "boost_flux": self.slider_boost.float_value(),
            "include_sporadics": self.chk_include_sporadics.isChecked(),
            "ground_mask": self.current_mask
        }

        self.worker = MeteorWorker("fuse_meteors", params)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, val: int, msg: str) -> None:
        self.progress_bar.setValue(val)
        self.log(msg)

    def _on_worker_finished(self, result: object, task_type: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_scan.setText("ESCANEAR SERIE DE METEOROS")
        self.btn_scan.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_scan.setEnabled(True)
        self.btn_fuse.setEnabled(True)

        if task_type == "scan_meteors":
            data = result
            self.candidates = data["candidates"]
            self.radiant_point = data["radiant_pt"]
            self.homographies_map = data["homographies"]
            if self.base_reference_img is None:
                self.base_reference_img = data["ref_img"]
                self._base_ref_is_fast_preview = True

            if self.radiant_point is not None:
                self.lbl_radiant_info.setText(
                    f"Radiante detectado: ({self.radiant_point[0]:.1f}, {self.radiant_point[1]:.1f}) px"
                )
            else:
                self.lbl_radiant_info.setText("Radiante: No convergente (pocos meteoros o dispersión alta)")

            self._refresh_cull_table()
            if self.candidates:
                self.table_cull.selectRow(0)
            self._update_multi_highlight_reticles()
            self.log(f"Escaneo completado: {len(self.candidates)} trazas detectadas y registradas.")

        elif task_type == "fuse_meteors":
            self.composite_result = result
            self._base_ref_is_fast_preview = False
            if hasattr(self, 'btn_view_composite'):
                self.btn_view_composite.setEnabled(True)
                self.btn_view_composite.setChecked(True)
            self._display_image_in_canvas(result)
            self.lbl_view_status.setText("Compuesto final de meteoros")
            self.log("Composición final de lluvia de meteoros completada con flujo neto.")
            rows = self.table_cull.selectionModel().selectedRows()
            if rows and rows[0].row() < len(self.candidates):
                self._update_highlight_reticle(self.candidates[rows[0].row()])
            self._update_multi_highlight_reticles()

    def _on_worker_error(self, err_msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_scan.setText("ESCANEAR SERIE DE METEOROS")
        self.btn_scan.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_scan.setEnabled(True)
        self.btn_fuse.setEnabled(True)
        QMessageBox.critical(self, "Error de Procesado", err_msg)
        self.log(f"ERROR: {err_msg}")

    def _on_worker_cancelled(self) -> None:
        self.progress_bar.setVisible(False)
        self.btn_scan.setText("ESCANEAR SERIE DE METEOROS")
        self.btn_scan.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_scan.setEnabled(True)
        self.btn_fuse.setEnabled(True)
        self.log("Operación cancelada por el usuario.")

    # -------------------------------------------------------------
    # GALERÍA DE CULLING E INSPECCIÓN INTERACTIVA
    # -------------------------------------------------------------
    def _refresh_cull_table(self) -> None:
        self.table_cull.blockSignals(True)
        self.table_cull.setRowCount(len(self.candidates))

        meteors_active = 0
        for row, c in enumerate(self.candidates):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.Checked if c.is_meteor else Qt.Unchecked)
            if c.is_meteor:
                meteors_active += 1

            id_item = QTableWidgetItem(f"#{c.id:02d}")
            fn_item = QTableWidgetItem(c.filename)

            if c.is_airplane:
                status_str = "Avión / Baliza"
                color = "#ef5350"
            elif c.is_satellite:
                status_str = "Satélite"
                color = "#42a5f5"
            elif c.is_sporadic:
                status_str = "Meteoro Esporádico"
                color = "#ffd54f"
            else:
                status_str = "Meteoro (Radiante)"
                color = "#66bb6a"

            status_item = QTableWidgetItem(status_str)
            status_item.setForeground(QColor(color))
            status_item.setToolTip(f"Continuidad de haz luminoso: {c.line_continuity * 100:.0f}%")

            len_item = QTableWidgetItem(f"{c.length_px:.1f} px")
            ang_item = QTableWidgetItem(f"{c.angle_deg:.1f}°")
            flux_item = QTableWidgetItem(f"{c.peak_flux:.2f}")
            div_str = f"{c.divergence_angle_deg:.1f}°" if c.divergence_angle_deg is not None else "-"
            div_item = QTableWidgetItem(div_str)

            self.table_cull.setItem(row, 0, chk)
            self.table_cull.setItem(row, 1, id_item)
            self.table_cull.setItem(row, 2, fn_item)
            self.table_cull.setItem(row, 3, status_item)
            self.table_cull.setItem(row, 4, len_item)
            self.table_cull.setItem(row, 5, ang_item)
            self.table_cull.setItem(row, 6, flux_item)
            self.table_cull.setItem(row, 7, div_item)

        self.lbl_cand_summary.setText(f"{meteors_active}/{len(self.candidates)} meteoros seleccionados")
        self.table_cull.blockSignals(False)

    def _on_cull_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            row = item.row()
            if row < len(self.candidates):
                self.candidates[row].is_meteor = (item.checkState() == Qt.Checked)
                active_count = sum(1 for c in self.candidates if c.is_meteor)
                self.lbl_cand_summary.setText(f"{active_count}/{len(self.candidates)} meteoros seleccionados")
                rows = self.table_cull.selectionModel().selectedRows()
                if rows and rows[0].row() == row:
                    self._update_highlight_reticle(self.candidates[row])
                self._update_multi_highlight_reticles()

    def _on_toggle_view_composite(self, checked: bool) -> None:
        if checked:
            if self.composite_result is not None:
                self._display_image_in_canvas(self.composite_result)
                self.lbl_view_status.setText("Compuesto final de meteoros")
                rows = self.table_cull.selectionModel().selectedRows()
                if rows and rows[0].row() < len(self.candidates):
                    self._update_highlight_reticle(self.candidates[rows[0].row()])
            else:
                self.btn_view_composite.setChecked(False)
        else:
            rows = self.table_cull.selectionModel().selectedRows()
            if rows and rows[0].row() < len(self.candidates):
                c = self.candidates[rows[0].row()]
                self._preview_frame(c.frame_index)
                self._update_highlight_reticle(c)
            elif self.files_list:
                self._preview_frame(self.ref_idx)
        self._update_multi_highlight_reticles()

    def _on_cull_selection_changed(self) -> None:
        rows = self.table_cull.selectionModel().selectedRows()
        if not rows:
            self.canvas.clear_highlight_reticle()
            return
        row = rows[0].row()
        if row < len(self.candidates):
            c = self.candidates[row]
            if hasattr(self, 'btn_view_composite') and self.btn_view_composite.isChecked() and self.composite_result is not None:
                # Mantener la vista del compuesto y mover el resalte a las coordenadas reproyectadas
                self._update_highlight_reticle(c)
            else:
                # Previsualizar el fotograma individual que contiene la traza original
                self._preview_frame(c.frame_index)
                self._update_highlight_reticle(c)
            self._update_multi_highlight_reticles()

    def _update_highlight_reticle(self, c: MeteorCandidate) -> None:
        """Actualiza el óvalo luminoso en el visor para la traza indicada."""
        if not hasattr(self, 'chk_highlight_streak') or not self.chk_highlight_streak.isChecked():
            self.canvas.clear_highlight_reticle()
            return
        # Color: amarillo eléctrico si está incluido como meteoro, rojo brillante si es avión o descartado
        color = "#ffea00" if (c.is_meteor and not c.is_airplane) else "#ff5252"
        label = f"#{c.id:02d} ({c.length_px:.0f}px)"

        is_comp = hasattr(self, 'btn_view_composite') and self.btn_view_composite.isChecked() and self.composite_result is not None
        start = c.reprojected_start if (is_comp and c.reprojected_start is not None) else c.start_pt
        end = c.reprojected_end if (is_comp and c.reprojected_end is not None) else c.end_pt
        if is_comp and c.reprojected_start is not None and c.reprojected_end is not None:
            pad = 20
            bbox = (
                int(min(start[0], end[0])) - pad,
                int(min(start[1], end[1])) - pad,
                int(max(start[0], end[0])) + pad,
                int(max(start[1], end[1])) + pad,
            )
        else:
            bbox = c.bbox

        self.canvas.set_highlight_reticle(
            start_pt=start,
            end_pt=end,
            bbox=bbox,
            color=color,
            label=label,
            padding=18.0
        )

    def _on_toggle_highlight(self, enabled: bool) -> None:
        if not enabled:
            self.canvas.clear_highlight_reticle()
        else:
            rows = self.table_cull.selectionModel().selectedRows()
            if rows and rows[0].row() < len(self.candidates):
                self._update_highlight_reticle(self.candidates[rows[0].row()])

    def _on_center_streak_clicked(self) -> None:
        """Centra y amplía la vista 1:1 sobre la traza seleccionada."""
        rows = self.table_cull.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.candidates):
            return
        c = self.candidates[rows[0].row()]
        is_comp = hasattr(self, 'btn_view_composite') and self.btn_view_composite.isChecked() and self.composite_result is not None
        start = c.reprojected_start if (is_comp and c.reprojected_start is not None) else c.start_pt
        end = c.reprojected_end if (is_comp and c.reprojected_end is not None) else c.end_pt
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        self.canvas.center_on_image_point((mid_x, mid_y), zoom_in=True)

    def _set_cull_filter_state(self, filter_type: str, state: bool) -> None:
        for c in self.candidates:
            if filter_type == "all":
                c.is_meteor = state
            elif filter_type == "planes" and (c.is_airplane or c.is_satellite):
                c.is_meteor = state
            elif filter_type == "sporadics" and c.is_sporadic:
                c.is_meteor = state
        self._refresh_cull_table()
        rows = self.table_cull.selectionModel().selectedRows()
        if rows and rows[0].row() < len(self.candidates):
            self._update_highlight_reticle(self.candidates[rows[0].row()])
        self._update_multi_highlight_reticles()

    def _update_multi_highlight_reticles(self) -> None:
        """Actualiza el resalte simultáneo de todas las trazas activas en el visor."""
        if not hasattr(self, 'chk_highlight_all') or not self.chk_highlight_all.isChecked():
            self.canvas.clear_multi_highlight_reticles()
            return

        is_comp = hasattr(self, 'btn_view_composite') and self.btn_view_composite.isChecked() and self.composite_result is not None
        active_reticles = []

        rows = self.table_cull.selectionModel().selectedRows()
        curr_frame_idx = self.candidates[rows[0].row()].frame_index if (rows and rows[0].row() < len(self.candidates)) else self.ref_idx

        for c in self.candidates:
            if not c.is_meteor:
                continue

            if is_comp:
                start = c.reprojected_start if c.reprojected_start is not None else c.start_pt
                end = c.reprojected_end if c.reprojected_end is not None else c.end_pt
            else:
                if c.frame_index != curr_frame_idx:
                    continue
                start = c.start_pt
                end = c.end_pt

            color = "#00e5ff" if not c.is_sporadic else "#ffd54f"
            label = f"#{c.id:02d}"

            active_reticles.append({
                "start_pt": (float(start[0]), float(start[1])),
                "end_pt": (float(end[0]), float(end[1])),
                "color": color,
                "label": label,
                "padding": 15.0,
                "id": c.id
            })

        self.canvas.set_multi_highlight_reticles(active_reticles)

    def _on_toggle_highlight_all(self, enabled: bool) -> None:
        self._update_multi_highlight_reticles()

    # -------------------------------------------------------------
    # IMPORTACIÓN DE SUELO Y CIELO APILADO
    # -------------------------------------------------------------
    def _on_load_sky_stack_clicked(self) -> None:
        filters = "Imágenes TIFF / FITS (*.tif *.tiff *.fit *.fits);;Todos (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen de Cielo Apilado Base", "", filters)
        if not path:
            return
        try:
            self.custom_stacked_sky = load_image_as_float32(path)
            self._display_image_in_canvas(self.custom_stacked_sky)
            self.lbl_view_status.setText(f"Cielo apilado: {os.path.basename(path)}")
            self.log(f"Cielo apilado maestro cargado desde: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar la imagen: {e}")

    def _update_canvas_mask_overlay(self) -> None:
        """Transfiere la máscara al visor para representación visual en color (Rojo = Suelo, Verde = Cielo)."""
        if hasattr(self, 'chk_show_mask') and self.chk_show_mask.isChecked() and self.current_mask is not None:
            self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
            self.canvas.set_mask_visible(True)
        else:
            self.canvas.set_mask_visible(False)

    def _set_active_mask(self, mask: Optional[np.ndarray], source_name: str) -> None:
        self.current_mask = mask
        if mask is not None:
            txt = f"Máscara activa: {source_name} (Suelo protegido ✓)"
            color = "#00e676"
            info_txt = "Máscara: Suelo protegido ✓"
        else:
            txt = "Máscara: Ninguna (suelo no protegido)"
            color = "#888888"
            info_txt = "Máscara: Ninguna (escaneo en todo el encuadre)"

        if hasattr(self, 'lbl_mask_status'):
            self.lbl_mask_status.setText(txt)
            self.lbl_mask_status.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        if hasattr(self, 'lbl_scan_mask_info'):
            self.lbl_scan_mask_info.setText(info_txt)
            self.lbl_scan_mask_info.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

        self._update_canvas_mask_overlay()
        if source_name:
            self.log(f"Máscara de horizonte establecida: {source_name}")

    def _on_toggle_show_mask(self, checked: bool) -> None:
        self._update_canvas_mask_overlay()

    def _on_clear_mask_clicked(self) -> None:
        self._set_active_mask(None, "")
        self.chk_show_mask.setChecked(False)
        self.canvas.set_mask_visible(False)
        self.log("Máscara de horizonte eliminada.")

    def _on_detect_ai_mask_clicked(self) -> None:
        """Segmentación semántica neuronal del cielo y suelo usando el modelo IA local en segundo plano."""
        if self.ai_worker is not None and self.ai_worker.isRunning():
            return

        target: Any = None
        if self.canvas.orig_rgb is not None:
            target = self.canvas.orig_rgb
        elif self.base_reference_img is not None:
            target = self.base_reference_img
        elif self.files_list:
            target = self.files_list[self.ref_idx]

        if target is None:
            QMessageBox.warning(self, "Atención", "Carga primero la serie temporal para detectar el horizonte con IA.")
            return

        self.btn_detect_ai_mask.setEnabled(False)
        self.btn_detect_ai_mask.setText("Detectando IA...")
        self.btn_detect_ai_mask.setIcon(get_icon("refresh", color="#ffffff"))
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.setCursor(Qt.WaitCursor)
        self.log("[IA HORIZONTE] Ejecutando segmentación neuronal de cielo y suelo en segundo plano...")

        self.ai_worker = AIHorizonWorker(
            image_rgb=target,
            feather_radius=7,
            threshold=0.5,
            parent=self
        )
        self.ai_worker.finished_success.connect(self._on_ai_mask_worker_finished)
        self.ai_worker.error_occurred.connect(self._on_ai_mask_worker_error)
        self.ai_worker.start()

    def _on_ai_mask_worker_finished(self, mask: np.ndarray, method_name: str) -> None:
        self.btn_detect_ai_mask.setEnabled(True)
        self.btn_detect_ai_mask.setText("Detectar por IA")
        self.btn_detect_ai_mask.setIcon(get_icon("sparkles", color="#80d8ff"))
        self.unsetCursor()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        self._set_active_mask(mask.astype(np.float32), f"IA ({method_name})")
        if hasattr(self, 'chk_show_mask'):
            self.chk_show_mask.setChecked(True)
        self.log(f"✓ [IA HORIZONTE] Suelo y cielo detectados con éxito usando {method_name}.")

    def _on_ai_mask_worker_error(self, err_msg: str) -> None:
        self.btn_detect_ai_mask.setEnabled(True)
        self.btn_detect_ai_mask.setText("Detectar por IA")
        self.btn_detect_ai_mask.setIcon(get_icon("sparkles", color="#80d8ff"))
        self.unsetCursor()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.log(f"[ERROR IA HORIZONTE] Falló la detección: {err_msg}")
        QMessageBox.warning(self, "Error Detección IA", f"No se pudo completar la segmentación por IA:\n{err_msg}")

    def _on_import_mask_clicked(self) -> None:
        if self._mask_provider is not None:
            mask = self._mask_provider()
            if mask is not None:
                self._set_active_mask(mask, "Importada del Apilador")
                QMessageBox.information(self, "Máscara Importada", "Máscara de horizonte vinculada con éxito desde el Apilador.\nEl suelo y follaje quedarán automáticamente excluidos del escaneo.")
                return
        QMessageBox.warning(self, "Aviso", "No hay máscara de horizonte generada en el Apilador. Créala primero en la Pestaña 1 o usa 'Cargar Máscara...' para abrir una desde disco.")

    def _on_load_mask_file_clicked(self) -> None:
        filters = "Imágenes de Máscara (*.png *.tif *.tiff *.jpg *.jpeg);;Todos (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Cargar Archivo de Máscara de Suelo/Horizonte", "", filters)
        if not path:
            return
        try:
            m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise ValueError("No se pudo decodificar el archivo de imagen.")
            if m.ndim == 3:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            m_f32 = m.astype(np.float32) / (65535.0 if m.dtype == np.uint16 else 255.0)
            self._set_active_mask(m_f32, os.path.basename(path))
            QMessageBox.information(self, "Máscara Cargada", f"Máscara de horizonte cargada exitosamente:\n{os.path.basename(path)}\nEl suelo quedará protegido contra falsos positivos.")
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar máscara", f"Fallo al leer la máscara:\n{e}")

    # -------------------------------------------------------------
    # EXPORTACIÓN Y REVELADOR
    # -------------------------------------------------------------
    def _on_save_tiff_clicked(self) -> None:
        img = self.composite_result if self.composite_result is not None else self.base_reference_img
        if img is None:
            QMessageBox.warning(self, "Aviso", "No hay imagen compuesta disponible para guardar.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Guardar Compuesto de Meteoros TIFF", "lluvia_meteoros_32bit.tif", "TIFF (*.tif *.tiff)")
        if not path:
            return

        try:
            save_tiff_with_icc(path, img.astype(np.float32), bit_depth=32, target_icc="sRGB")
            self.log(f"Compuesto de meteoros guardado en: {path}")
            QMessageBox.information(self, "Guardado Exitoso", f"Archivo TIFF guardado:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar: {e}")

    def _on_save_psd_clicked(self) -> None:
        base = self.custom_stacked_sky if self.custom_stacked_sky is not None else self.base_reference_img
        if base is None:
            QMessageBox.warning(self, "Aviso", "No hay imagen base para exportar en capas PSD.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Exportar PSD Multicapa Photoshop", "lluvia_meteoros_capas.psd", "Photoshop PSD (*.psd)")
        if not path:
            return

        ok = export_meteor_layered_psd(
            output_path=path,
            base_image=base,
            candidates=self.candidates,
            homographies_map=self.homographies_map,
            include_sporadics=self.chk_include_sporadics.isChecked(),
            boost_flux=self.slider_boost.float_value()
        )

        if ok:
            self.log(f"Composición PSD multicapa exportada con éxito en: {path}")
            QMessageBox.information(self, "PSD Exportado", f"Archivo PSD multicapa generado con éxito:\n{path}")
        else:
            QMessageBox.warning(self, "Aviso", "La librería pytoshop no está disponible o falló la exportación.")

    def _on_send_to_developer_clicked(self) -> None:
        img = self.composite_result if self.composite_result is not None else self.base_reference_img
        if img is None:
            QMessageBox.warning(self, "Aviso", "Genera o carga primero un compuesto de meteoros para transferir.")
            return

        export_dir = os.path.abspath("exportaciones")
        os.makedirs(export_dir, exist_ok=True)
        transfer_path = os.path.join(export_dir, "meteoros_transferencia_dev.tif")

        try:
            save_tiff_with_icc(transfer_path, img.astype(np.float32), bit_depth=32, target_icc="sRGB")
            self.export_to_developer.emit(transfer_path)
            self.log(f"Compuesto de meteoros transferido al Revelador / Editor ({transfer_path}).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo transferir la imagen: {e}")
