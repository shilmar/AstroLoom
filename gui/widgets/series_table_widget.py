# gui/widgets/series_table_widget.py
"""
gui/widgets/series_table_widget.py - Tabla genérica y reactiva para series temporales de fotos.

Proporciona:
1. Carga inmediata de tomas (UI fluida a 60 FPS sin congelamientos para series de 300-1200+ fotos).
2. Extracción asíncrona de metadatos EXIF/FITS (Hora, Exp, ISO, f/) en segundo plano con QThread.
3. Barra de progreso compacta integrada con auto-ocultación.
4. Soporte para Drag & Drop directo desde el sistema de archivos.
5. Modos adaptables:
   - "lights" / "meteors": [Usar, REF, Archivo, Hora, Exp., ISO, f/]
   - "darks": [Archivo, Exp., ISO, ✕]
   - "flats": [Archivo, Exp., ISO, f/, ✕]
"""
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Dict, Any, Optional

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PySide6.QtGui import QColor

from gui.widgets.drop_widgets import DropTableWidget
from core.startrails import extract_shot_exif
from gui.theme import (
    BG_CANVAS, BORDER_BASE, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY
)


class TimeTableWidgetItem(QTableWidgetItem):
    """
    QTableWidgetItem para columnas de tiempo con ordenación cronológica continua.
    Compara mediante Qt.UserRole (Unix epoch float) para evitar discontinuidades
    por cruce de medianoche en sesiones de astrofotografía nocturna.
    """
    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            t1 = self.data(Qt.UserRole)
            t2 = other.data(Qt.UserRole)
            if t1 is not None and t2 is not None:
                try:
                    return float(t1) < float(t2)
                except (ValueError, TypeError):
                    pass
        return super().__lt__(other)


class ScoreTableWidgetItem(QTableWidgetItem):
    """
    QTableWidgetItem para columnas de calidad/score numérico con ordenación continua.
    Compara mediante Qt.UserRole (float) para ordenar ascendentemente o descendentemente.
    """
    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            s1 = self.data(Qt.UserRole)
            s2 = other.data(Qt.UserRole)
            if s1 is not None and s2 is not None:
                try:
                    return float(s1) < float(s2)
                except (ValueError, TypeError):
                    pass
        return super().__lt__(other)


class MetadataScanWorker(QThread):
    """
    Worker en segundo plano para escanear metadatos fotográficos sin bloquear la interfaz.
    Solapa I/O y CPU mediante un pool dinámico de hilos (por defecto 2) para no saturar HDDs mecánicos.
    """
    progress = Signal(int, int, str, dict)  # (current_1_based, total, path, meta_dict)
    finished = Signal()

    def __init__(
        self,
        paths: List[str],
        meta_batch: Optional[Dict[str, Any]] = None,
        max_workers: int = 2,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.paths = list(paths)
        self.meta_batch = meta_batch or {}
        # shortcut: 2 workers fijos por defecto (óptimo HDD/SSD básico), ampliar con detección HDD vs NVMe si se requiere mayor throughput
        self.max_workers = max(1, int(max_workers))
        self._is_cancelled = False
        self._executor: Optional[ThreadPoolExecutor] = None

    def cancel(self) -> None:
        self._is_cancelled = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _process_item(self, path: str) -> tuple[str, Dict[str, Any]]:
        if self._is_cancelled:
            return path, {}
        meta_item = self.meta_batch.get(path)
        meta = extract_shot_exif(path, meta_item)
        return path, meta

    def run(self) -> None:
        total = len(self.paths)
        if total == 0:
            self.finished.emit()
            return

        if total == 1:
            meta_item = self.meta_batch.get(self.paths[0])
            meta = extract_shot_exif(self.paths[0], meta_item)
            if not self._is_cancelled:
                self.progress.emit(1, 1, self.paths[0], meta)
            self.finished.emit()
            return

        workers = min(self.max_workers, total)
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            self._executor = executor
            futures = {executor.submit(self._process_item, p): p for p in self.paths}
            try:
                for future in as_completed(futures):
                    if self._is_cancelled:
                        break
                    try:
                        path, meta = future.result()
                    except (concurrent.futures.CancelledError, Exception):
                        continue
                    if self._is_cancelled:
                        break
                    completed += 1
                    self.progress.emit(completed, total, path, meta)
            finally:
                self._executor = None

        self.finished.emit()


class SeriesFilesTableWidget(QWidget):
    """
    Widget de tabla reactiva y desacoplada para listas de fotografías (Lights, Darks, Flats, Bias, Suelo, Meteoros).
    """
    files_changed = Signal(list)
    files_dropped = Signal(list)
    selection_changed = Signal(set)
    reference_changed = Signal(int)
    file_selected = Signal(int, str)
    file_double_clicked = Signal(int, str)
    file_removed = Signal(str)
    scan_progress = Signal(int, int)
    scan_finished = Signal()

    def __init__(
        self,
        mode: str = "lights",
        max_workers: int = 2,
        show_quality: bool = False,
        allow_delete: Optional[bool] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.mode = mode.lower()
        self.max_workers = max(1, int(max_workers))
        self.show_quality = show_quality
        self.allow_delete = allow_delete if allow_delete is not None else (self.mode not in ("lights", "meteors"))
        self.files_list: List[str] = []
        self.excluded_files: Set[str] = set()
        self.ref_idx: int = 0
        self._sort_col: Optional[int] = None
        self._sort_asc: bool = True
        self._user_sorted: bool = False
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._quality_results: Dict[str, Any] = {}
        self._worker: Optional[MetadataScanWorker] = None

        self._setup_columns()
        self._setup_ui()

    def _setup_columns(self) -> None:
        self.headers = []
        self.col_use = -1
        self.col_quality = -1
        self.col_ref = -1
        self.col_name = -1
        self.col_time = -1
        self.col_exp = -1
        self.col_iso = -1
        self.col_ap = -1
        self.col_del = -1

        if self.mode in ("lights", "meteors"):
            self.headers.append("Usar")
            self.col_use = len(self.headers) - 1

            if self.show_quality:
                self.headers.append("Calidad")
                self.col_quality = len(self.headers) - 1

            self.headers.append("REF")
            self.col_ref = len(self.headers) - 1

            self.headers.append("Archivo")
            self.col_name = len(self.headers) - 1

            self.headers.append("Hora")
            self.col_time = len(self.headers) - 1

            self.headers.append("Exp.")
            self.col_exp = len(self.headers) - 1

            self.headers.append("ISO")
            self.col_iso = len(self.headers) - 1

            self.headers.append("f/")
            self.col_ap = len(self.headers) - 1

            if self.allow_delete:
                self.headers.append("")
                self.col_del = len(self.headers) - 1

        elif self.mode in ("darks", "bias"):
            self.headers = ["Archivo", "Exp.", "ISO"]
            self.col_name = 0
            self.col_exp = 1
            self.col_iso = 2
            if self.allow_delete:
                self.headers.append("")
                self.col_del = 3

        elif self.mode in ("flats", "ground", "suelo"):
            self.headers = ["Archivo", "Exp.", "ISO", "f/"]
            self.col_name = 0
            self.col_exp = 1
            self.col_iso = 2
            self.col_ap = 3
            if self.allow_delete:
                self.headers.append("")
                self.col_del = 4

        else:
            self.headers = ["Usar", "REF", "Archivo", "Hora", "Exp.", "ISO", "f/"]
            self.col_use = 0
            self.col_ref = 1
            self.col_name = 2
            self.col_time = 3
            self.col_exp = 4
            self.col_iso = 5
            self.col_ap = 6

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.table = DropTableWidget(0, len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setShowGrid(True)

        header = self.table.horizontalHeader()
        header.setHighlightSections(False)
        header.setMinimumSectionSize(28)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.sectionClicked.connect(self._on_header_section_clicked)

        tooltips = {
            self.col_use: "Ordenar por estado (incluida / descartada)",
            self.col_quality: "Ordenar por puntuación de calidad óptica",
            self.col_ref: "Toma de referencia para alineación",
            self.col_name: "Ordenar alfabéticamente por nombre de archivo",
            self.col_time: "Ordenar cronológicamente por fecha/hora",
            self.col_exp: "Ordenar por tiempo de exposición",
            self.col_iso: "Ordenar por sensibilidad ISO",
            self.col_ap: "Ordenar por apertura de diafragma",
        }
        for c, tip in tooltips.items():
            if c >= 0 and c < len(self.headers):
                h_item = self.table.horizontalHeaderItem(c)
                if h_item:
                    h_item.setToolTip(tip)

        if self.col_use >= 0:
            header.setSectionResizeMode(self.col_use, QHeaderView.Interactive)
            self.table.setColumnWidth(self.col_use, 44)
        if self.col_quality >= 0:
            header.setSectionResizeMode(self.col_quality, QHeaderView.Interactive)
            self.table.setColumnWidth(self.col_quality, 76)
        if self.col_ref >= 0:
            header.setSectionResizeMode(self.col_ref, QHeaderView.Interactive)
            self.table.setColumnWidth(self.col_ref, 36)
        if self.col_name >= 0:
            header.setSectionResizeMode(self.col_name, QHeaderView.Stretch)
        if self.col_time >= 0:
            header.setSectionResizeMode(self.col_time, QHeaderView.ResizeToContents)
        if self.col_exp >= 0:
            header.setSectionResizeMode(self.col_exp, QHeaderView.Interactive)
            self.table.setColumnWidth(self.col_exp, 50)
        if self.col_iso >= 0:
            header.setSectionResizeMode(self.col_iso, QHeaderView.Interactive)
            self.table.setColumnWidth(self.col_iso, 52)
        if self.col_ap >= 0:
            header.setSectionResizeMode(self.col_ap, QHeaderView.Interactive)
            self.table.setColumnWidth(self.col_ap, 44)
        if self.col_del >= 0:
            header.setSectionResizeMode(self.col_del, QHeaderView.Interactive)
            self.table.setColumnWidth(self.col_del, 28)

        table_compact_style = f"""
            QTableWidget {{
                background-color: {BG_CANVAS};
                border: 1px solid {BORDER_BASE};
                border-radius: 4px;
                gridline-color: #1a1a1e;
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 1px 4px;
                color: {TEXT_PRIMARY};
            }}
            QTableWidget::item:selected {{
                background-color: #1e293b;
                color: #ffffff;
            }}
            QHeaderView::section {{
                background-color: {BG_PANEL};
                color: {TEXT_SECONDARY};
                padding: 2px 4px;
                font-size: 10px;
                font-weight: 600;
                border: 1px solid {BORDER_BASE};
            }}
        """
        self.table.setStyleSheet(table_compact_style)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.cellClicked.connect(self._on_table_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_table_cell_double_clicked)
        self.table.files_dropped.connect(self.add_files)
        self.table.files_dropped.connect(self.files_dropped.emit)
        layout.addWidget(self.table)

        # Barra de progreso compacta para escaneo asíncrono
        self.status_container = QWidget()
        status_l = QHBoxLayout(self.status_container)
        status_l.setContentsMargins(2, 1, 2, 1)
        status_l.setSpacing(6)

        self.lbl_scan_status = QLabel("Extrayendo metadatos...")
        self.lbl_scan_status.setStyleSheet("color: #90a4ae; font-size: 10px;")
        status_l.addWidget(self.lbl_scan_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #2f81f7;
                border-radius: 2px;
            }
        """)
        status_l.addWidget(self.progress_bar, 1)

        btn_cancel = QPushButton("✕")
        btn_cancel.setFixedSize(16, 16)
        btn_cancel.setToolTip("Detener escaneo de metadatos")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #a1a1aa;
                border: none;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ef4444;
            }
        """)
        btn_cancel.clicked.connect(self.cancel_scan)
        status_l.addWidget(btn_cancel)

        self.status_container.setVisible(False)
        layout.addWidget(self.status_container)

    def rowCount(self) -> int:
        return self.table.rowCount()

    def get_metadata(self, path: str) -> Dict[str, Any]:
        return self._metadata_cache.get(path, {})

    def set_metadata_cache(self, cache: Dict[str, Dict[str, Any]]) -> None:
        self._metadata_cache.update(cache)

    def get_timestamp(self, path: str) -> Optional[float]:
        """Retorna el timestamp Unix (segundos float) de la toma si está disponible."""
        meta = self._metadata_cache.get(path)
        if meta and meta.get("timestamp") is not None:
            try:
                return float(meta["timestamp"])
            except (ValueError, TypeError):
                return None
        return None

    def get_datetime_str(self, path: str) -> Optional[str]:
        """Retorna la fecha y hora completa en formato YYYY-MM-DD HH:MM:SS."""
        meta = self._metadata_cache.get(path)
        if meta and meta.get("datetime_str"):
            return str(meta["datetime_str"])
        return None

    def get_timeline_data(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """
        Retorna la secuencia de tomas con su información temporal continua y estado:
        [{"path": str, "timestamp": Optional[float], "time_str": str, "datetime_str": str, "is_active": bool}, ...]
        """
        files = self.get_active_files() if active_only else self.files_list
        res = []
        for p in files:
            meta = self._metadata_cache.get(p, {})
            res.append({
                "path": p,
                "timestamp": meta.get("timestamp"),
                "time_str": meta.get("time_str", "—"),
                "datetime_str": meta.get("datetime_str", ""),
                "is_active": (p not in self.excluded_files),
                "exp": meta.get("exp"),
                "iso": meta.get("iso")
            })
        return res

    def detect_time_gaps(
        self,
        threshold_seconds: Optional[float] = None,
        gap_multiplier: float = 2.5,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detecta huecos temporales anómalos (gaps) entre tomas consecutivas.
        Identifica interrupciones por paso de nubes, agotamiento de batería o pausas en la sesión.
        """
        timeline = [t for t in self.get_timeline_data(active_only=active_only) if t.get("timestamp") is not None]
        if len(timeline) < 2:
            return []

        deltas = []
        for i in range(len(timeline) - 1):
            dt = float(timeline[i + 1]["timestamp"]) - float(timeline[i]["timestamp"])
            if dt > 0:
                deltas.append(dt)

        if not deltas:
            return []

        deltas_sorted = sorted(deltas)
        median_delta = deltas_sorted[len(deltas_sorted) // 2]
        effective_thresh = threshold_seconds if threshold_seconds is not None else (median_delta * gap_multiplier)

        gaps = []
        for i in range(len(timeline) - 1):
            dt = float(timeline[i + 1]["timestamp"]) - float(timeline[i]["timestamp"])
            if dt > effective_thresh:
                gaps.append({
                    "from_idx": i,
                    "to_idx": i + 1,
                    "from_path": timeline[i]["path"],
                    "to_path": timeline[i + 1]["path"],
                    "from_time": timeline[i]["datetime_str"] or timeline[i]["time_str"],
                    "to_time": timeline[i + 1]["datetime_str"] or timeline[i + 1]["time_str"],
                    "gap_seconds": dt,
                    "expected_interval": median_delta
                })
        return gaps

    def _on_header_section_clicked(self, col: int) -> None:
        if col in (self.col_del, self.col_ref) or col < 0:
            return
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            # Calidad defaults to descending (highest quality first)
            self._sort_asc = False if col == self.col_quality else True

        self._user_sorted = True
        self._apply_column_sort(col, self._sort_asc)

    def _apply_column_sort(self, col: int, ascending: bool) -> None:
        if len(self.files_list) < 2:
            return

        ref_file = self.get_reference_file()

        if col == self.col_time:
            def _key(p):
                meta = self._metadata_cache.get(p, {})
                ts = meta.get("timestamp")
                if ts is not None:
                    try:
                        return (0, float(ts))
                    except (ValueError, TypeError):
                        pass
                return (1, os.path.basename(p).lower())
        elif col == self.col_name:
            def _key(p):
                return os.path.basename(p).lower()
        elif col == self.col_quality:
            def _key(p):
                res = self._quality_results.get(p)
                score = getattr(res, "score", None) if res else None
                if score is not None and score == score:
                    return (0, float(score))
                return (1, -1.0)
        elif col == self.col_exp:
            def _key(p):
                meta = self._metadata_cache.get(p, {})
                exp = meta.get("exp")
                if exp is not None:
                    try:
                        return (0, float(exp))
                    except (ValueError, TypeError):
                        pass
                return (1, 0.0)
        elif col == self.col_iso:
            def _key(p):
                meta = self._metadata_cache.get(p, {})
                iso = meta.get("iso")
                if iso is not None:
                    try:
                        return (0, int(iso))
                    except (ValueError, TypeError):
                        pass
                return (1, 0)
        elif col == self.col_ap:
            def _key(p):
                meta = self._metadata_cache.get(p, {})
                ap = meta.get("aperture")
                if ap is not None:
                    try:
                        return (0, float(ap))
                    except (ValueError, TypeError):
                        pass
                return (1, 0.0)
        elif col == self.col_use:
            def _key(p):
                return (0 if p not in self.excluded_files else 1, os.path.basename(p).lower())
        else:
            return

        self._sort_col = col
        self._sort_asc = ascending
        self.files_list.sort(key=_key, reverse=not ascending)

        if ref_file and ref_file in self.files_list:
            self.ref_idx = self.files_list.index(ref_file)
        else:
            self.ref_idx = 0

        self.table.horizontalHeader().setSortIndicator(
            col, Qt.AscendingOrder if ascending else Qt.DescendingOrder
        )
        self._refresh_all_rows()
        self.files_changed.emit(self.files_list)

    def sort_by_time(self, ascending: bool = True) -> None:
        """
        Reordena la lista de archivos por timestamp cronológico continuo (resuelve medianoche).
        Las tomas sin timestamp quedan al final.
        """
        if self.col_time >= 0:
            self._apply_column_sort(self.col_time, ascending)
        else:
            if len(self.files_list) < 2:
                return
            ref_file = self.get_reference_file()
            def _sort_key(path: str):
                meta = self._metadata_cache.get(path, {})
                ts = meta.get("timestamp")
                if ts is not None:
                    return (0, float(ts))
                return (1, os.path.basename(path).lower())
            self.files_list.sort(key=_sort_key, reverse=not ascending)
            if ref_file and ref_file in self.files_list:
                self.ref_idx = self.files_list.index(ref_file)
            else:
                self.ref_idx = 0
            self._refresh_all_rows()
            self.files_changed.emit(self.files_list)

    def set_files(self, files: List[str], meta_batch: Optional[Dict[str, Any]] = None) -> None:
        """Carga y reemplaza la lista completa de archivos."""
        self.cancel_scan()
        self.files_list.clear()
        self.excluded_files.clear()
        self.ref_idx = 0
        self.add_files(files, meta_batch=meta_batch)

    def add_files(self, files: List[str], meta_batch: Optional[Dict[str, Any]] = None) -> None:
        """Añade tomas deduplicando y poblando la tabla al instante."""
        if not files:
            return
        new_files = [f for f in files if f not in self.files_list]
        if not new_files:
            return

        self.files_list.extend(new_files)
        if self.mode in ("lights", "meteors"):
            # Ordenar inicialmente por nombre base si no se ha ordenado manualmente
            if not self._user_sorted:
                self.files_list.sort(key=lambda p: os.path.basename(p).lower())

        self._refresh_all_rows()
        self.files_changed.emit(self.files_list)

        # Detectar qué archivos requieren escaneo de metadatos en segundo plano
        missing = [p for p in self.files_list if p not in self._metadata_cache]
        if missing:
            self._start_metadata_scan(missing, meta_batch)

    def clear_files(self) -> None:
        """Limpia la lista de archivos y detiene cualquier escaneo activo."""
        self.cancel_scan()
        self.files_list.clear()
        self.excluded_files.clear()
        self._quality_results.clear()
        self.ref_idx = 0
        self._sort_col = None
        self._sort_asc = True
        self._user_sorted = False
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        self.files_changed.emit(self.files_list)
        self.selection_changed.emit(self.excluded_files)

    def remove_file(self, path: str) -> None:
        """Elimina un archivo de la lista."""
        if path in self.files_list:
            row = self.files_list.index(path)
            self.files_list.remove(path)
            self.excluded_files.discard(path)
            self._quality_results.pop(path, None)
            if self.ref_idx >= len(self.files_list):
                self.ref_idx = max(0, len(self.files_list) - 1)
            self.table.blockSignals(True)
            self.table.removeRow(row)
            self.table.blockSignals(False)
            self._update_ref_badges()
            self.files_changed.emit(self.files_list)
            self.file_removed.emit(path)

    def set_reference_index(self, idx: int) -> None:
        """Establece el índice de la toma de referencia."""
        if 0 <= idx < len(self.files_list):
            self.ref_idx = idx
            self._update_ref_badges()
            self.reference_changed.emit(self.ref_idx)

    def set_reference_file(self, path: str) -> None:
        """Fija la toma de referencia por su ruta de archivo."""
        if path in self.files_list:
            self.set_reference_index(self.files_list.index(path))

    def set_quality_results(self, results: Dict[str, Any]) -> None:
        """Asigna los resultados del análisis de calidad óptica (culling) y actualiza la tabla."""
        self._quality_results.update(results)
        if self.col_quality >= 0:
            self.table.blockSignals(True)
            for row, p in enumerate(self.files_list):
                if p in self._quality_results:
                    self._populate_quality_cell(row, p)
            self.table.blockSignals(False)

    def get_reference_index(self) -> int:
        return self.ref_idx

    def get_reference_file(self) -> Optional[str]:
        if 0 <= self.ref_idx < len(self.files_list):
            return self.files_list[self.ref_idx]
        return None

    def get_active_files(self) -> List[str]:
        return [f for f in self.files_list if f not in self.excluded_files]

    def set_excluded_files(self, excluded: Set[str]) -> None:
        new_set = set(excluded)
        self.excluded_files.clear()
        self.excluded_files.update(new_set)
        self.table.blockSignals(True)
        if self.col_use >= 0:
            for row, p in enumerate(self.files_list):
                chk_item = self.table.item(row, self.col_use)
                if chk_item:
                    chk_item.setCheckState(Qt.Unchecked if p in self.excluded_files else Qt.Checked)
        self.table.blockSignals(False)
        self.selection_changed.emit(self.excluded_files)

    def select_row(self, row: int) -> None:
        if 0 <= row < self.table.rowCount():
            self.table.selectRow(row)

    def selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def selected_file(self) -> Optional[str]:
        r = self.selected_row()
        if 0 <= r < len(self.files_list):
            return self.files_list[r]
        return None

    def cancel_scan(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(500)
        self.status_container.setVisible(False)

    def closeEvent(self, event) -> None:
        self.cancel_scan()
        super().closeEvent(event)

    # --- Métodos internos de renderizado y sincronización ---

    def _refresh_all_rows(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.files_list))
        for row, path in enumerate(self.files_list):
            self._populate_row(row, path)
        self.table.blockSignals(False)

    def _populate_row(self, row: int, path: str) -> None:
        meta = self._metadata_cache.get(path, {})

        # Columna Usar (Checkbox)
        if self.col_use >= 0:
            is_included = (path not in self.excluded_files)
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk_item.setCheckState(Qt.Checked if is_included else Qt.Unchecked)
            self.table.setItem(row, self.col_use, chk_item)

        # Columna Calidad
        if self.col_quality >= 0:
            self._populate_quality_cell(row, path)

        # Columna REF
        if self.col_ref >= 0:
            is_ref = (row == self.ref_idx)
            ref_item = QTableWidgetItem("● REF" if is_ref else "")
            ref_item.setTextAlignment(Qt.AlignCenter)
            if is_ref:
                ref_item.setForeground(QColor("#2f81f7"))
            self.table.setItem(row, self.col_ref, ref_item)

        # Columna Archivo (Nombre)
        if self.col_name >= 0:
            name_item = QTableWidgetItem(os.path.basename(path))
            name_item.setToolTip(path)
            self.table.setItem(row, self.col_name, name_item)

        # Columna Hora
        if self.col_time >= 0:
            time_item = TimeTableWidgetItem(meta.get("time_str", "—"))
            time_item.setTextAlignment(Qt.AlignCenter)
            ts = meta.get("timestamp")
            if ts is not None:
                try:
                    time_item.setData(Qt.UserRole, float(ts))
                except (ValueError, TypeError):
                    pass
            dt_str = meta.get("datetime_str")
            if dt_str:
                time_item.setData(Qt.UserRole + 1, str(dt_str))
                time_item.setToolTip(str(dt_str))
            else:
                time_item.setToolTip(path)
            self.table.setItem(row, self.col_time, time_item)

        # Columna Exp
        if self.col_exp >= 0:
            exp_item = QTableWidgetItem(meta.get("exp_str", "—"))
            exp_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.col_exp, exp_item)

        # Columna ISO
        if self.col_iso >= 0:
            iso_item = QTableWidgetItem(meta.get("iso_str", "—"))
            iso_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.col_iso, iso_item)

        # Columna Apertura (f/)
        if self.col_ap >= 0:
            ap_item = QTableWidgetItem(meta.get("aperture_str", "—"))
            ap_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.col_ap, ap_item)

        # Columna Eliminar (✕)
        if self.col_del >= 0:
            btn_del = QPushButton("×")
            btn_del.setFixedSize(18, 18)
            btn_del.setToolTip("Eliminar esta toma")
            btn_del.setStyleSheet("""
                QPushButton {
                    color: #a1a1aa;
                    background: transparent;
                    border: 1px solid #2e2e35;
                    border-radius: 3px;
                    font-weight: bold;
                    font-size: 13px;
                    line-height: 14px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #b91c1c;
                    color: #ffffff;
                    border-color: #ef4444;
                }
            """)
            p_cap = path
            btn_del.clicked.connect(lambda checked=False, p=p_cap: self.remove_file(p))
            self.table.setCellWidget(row, self.col_del, btn_del)

    def _update_row_metadata(self, row: int, meta: Dict[str, Any]) -> None:
        self.table.blockSignals(True)
        if self.col_time >= 0:
            item = self.table.item(row, self.col_time)
            if item:
                item.setText(meta.get("time_str", "—"))
                ts = meta.get("timestamp")
                if ts is not None:
                    try:
                        item.setData(Qt.UserRole, float(ts))
                    except (ValueError, TypeError):
                        pass
                dt_str = meta.get("datetime_str")
                if dt_str:
                    item.setData(Qt.UserRole + 1, str(dt_str))
                    item.setToolTip(str(dt_str))
                else:
                    item.setToolTip("")
        if self.col_exp >= 0:
            item = self.table.item(row, self.col_exp)
            if item:
                item.setText(meta.get("exp_str", "—"))
        if self.col_iso >= 0:
            item = self.table.item(row, self.col_iso)
            if item:
                item.setText(meta.get("iso_str", "—"))
        if self.col_ap >= 0:
            item = self.table.item(row, self.col_ap)
            if item:
                item.setText(meta.get("aperture_str", "—"))
        self.table.blockSignals(False)

    def _update_ref_badges(self) -> None:
        if self.col_ref < 0:
            return
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            item = self.table.item(r, self.col_ref)
            if not item:
                continue
            if r == self.ref_idx:
                item.setText("● REF")
                item.setForeground(QColor("#2f81f7"))
            else:
                item.setText("")
        self.table.blockSignals(False)

    def _start_metadata_scan(self, missing_paths: List[str], meta_batch: Optional[Dict[str, Any]] = None) -> None:
        self.cancel_scan()
        if not missing_paths:
            self.status_container.setVisible(False)
            return

        self.progress_bar.setValue(0)
        self.lbl_scan_status.setText(f"Extrayendo metadatos: 0 / {len(missing_paths)} (0%)")
        self.status_container.setVisible(True)

        self._worker = MetadataScanWorker(
            missing_paths,
            meta_batch=meta_batch,
            max_workers=self.max_workers,
            parent=self
        )
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    def _on_scan_progress(self, curr: int, total: int, path: str, meta: Dict[str, Any]) -> None:
        self._metadata_cache[path] = meta
        pct = int((curr / total) * 100) if total > 0 else 100
        self.progress_bar.setValue(pct)
        self.lbl_scan_status.setText(f"Extrayendo metadatos: {curr} / {total} tomas ({pct}%)")

        if path in self.files_list:
            row = self.files_list.index(path)
            self._update_row_metadata(row, meta)

        self.scan_progress.emit(curr, total)

    def _on_scan_finished(self) -> None:
        self.status_container.setVisible(False)
        # Auto-ordenar cronológicamente si no hubo orden manual del usuario y hay timestamps
        if not self._user_sorted and self.mode in ("lights", "meteors") and self.col_time >= 0:
            has_timestamps = any(
                self._metadata_cache.get(p, {}).get("timestamp") is not None
                for p in self.files_list
            )
            if has_timestamps:
                self.sort_by_time(ascending=True)
        self.scan_finished.emit()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self.col_use >= 0 and item.column() == self.col_use:
            row = item.row()
            if 0 <= row < len(self.files_list):
                path = self.files_list[row]
                if item.checkState() == Qt.Checked:
                    self.excluded_files.discard(path)
                else:
                    self.excluded_files.add(path)
                self.selection_changed.emit(self.excluded_files)

    def _on_table_selection_changed(self) -> None:
        row = self.selected_row()
        if 0 <= row < len(self.files_list):
            self.file_selected.emit(row, self.files_list[row])

    def _on_table_cell_clicked(self, row: int, col: int) -> None:
        if self.col_ref >= 0 and col == self.col_ref:
            self.set_reference_index(row)

    def _on_table_cell_double_clicked(self, row: int, col: int) -> None:
        if 0 <= row < len(self.files_list):
            self.file_double_clicked.emit(row, self.files_list[row])

    def _populate_quality_cell(self, row: int, path: str) -> None:
        if self.col_quality < 0:
            return
        res = self._quality_results.get(path)
        if res is None:
            item = ScoreTableWidgetItem("—")
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor("#71717a"))
            item.setData(Qt.UserRole, -1.0)
            self.table.setItem(row, self.col_quality, item)
            return

        score_val = getattr(res, "score", None)
        score_int = int(round(score_val)) if (score_val is not None and score_val == score_val) else 0
        status = getattr(res, "status", "ACCEPTABLE")
        status_label = getattr(res, "status_label", "")

        defect_type = ""
        if "Movida" in status_label:
            defect_type = " Mov"
        elif "Nubes" in status_label:
            defect_type = " Nub"
        elif "Desenfoque" in status_label:
            defect_type = " Foc"

        txt = f"● {score_int}%{defect_type}"
        item = ScoreTableWidgetItem(txt)
        item.setTextAlignment(Qt.AlignCenter)
        item.setData(Qt.UserRole, float(score_val) if score_val is not None else -1.0)

        if status == "EXCELLENT":
            item.setForeground(QColor("#4ade80"))
        elif status == "ACCEPTABLE":
            item.setForeground(QColor("#facc15"))
        else:
            item.setForeground(QColor("#f87171"))

        fwhm = getattr(res, "fwhm", None)
        fwhm_str = f"{fwhm:.1f} px" if (fwhm is not None and fwhm > 0 and fwhm == fwhm) else "N/D"
        c_ecc = getattr(res, "center_ecc", None)
        c_ecc_str = f"{c_ecc:.2f}" if (c_ecc is not None and c_ecc == c_ecc) else "N/D"
        cor_ecc = getattr(res, "corner_ecc", None)
        cor_ecc_str = f"{cor_ecc:.2f}" if (cor_ecc is not None and cor_ecc == cor_ecc) else "N/D"

        score_disp = f"{score_val:.1f}%" if (score_val is not None and score_val == score_val) else "N/D"
        tip = (
            f"Puntuación: {score_disp}\n"
            f"Nitidez (FWHM): {fwhm_str}\n"
            f"Redondez centro (eje óptico): {c_ecc_str}\n"
            f"Coma esquinas: {cor_ecc_str}\n"
            f"Estrellas detectadas: {getattr(res, 'star_count', 0)}"
        )
        diag = getattr(res, "optical_diagnosis_msg", "")
        if diag:
            tip += f"\nÓptica: {diag}"
        defect = getattr(res, "defect_reason", "")
        if defect:
            tip += f"\nMotivo: {defect}"
        item.setToolTip(tip)

        self.table.setItem(row, self.col_quality, item)
