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


class MetadataScanWorker(QThread):
    """Worker en segundo plano para escanear metadatos fotográficos sin bloquear la interfaz."""
    progress = Signal(int, int, str, dict)  # (current_1_based, total, path, meta_dict)
    finished = Signal()

    def __init__(self, paths: List[str], meta_batch: Optional[Dict[str, Any]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.paths = list(paths)
        self.meta_batch = meta_batch or {}
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        total = len(self.paths)
        for i, path in enumerate(self.paths, 1):
            if self._is_cancelled:
                break
            meta_item = self.meta_batch.get(path)
            meta = extract_shot_exif(path, meta_item)
            if self._is_cancelled:
                break
            self.progress.emit(i, total, path, meta)
        self.finished.emit()


class SeriesFilesTableWidget(QWidget):
    """
    Widget de tabla reactiva y desacoplada para listas de fotografías (Lights, Darks, Flats, Meteoros).
    """
    files_changed = Signal(list)
    files_dropped = Signal(list)
    selection_changed = Signal(set)
    reference_changed = Signal(int)
    file_selected = Signal(int, str)
    file_removed = Signal(str)
    scan_progress = Signal(int, int)
    scan_finished = Signal()

    def __init__(self, mode: str = "lights", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mode = mode.lower()
        self.files_list: List[str] = []
        self.excluded_files: Set[str] = set()
        self.ref_idx: int = 0
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._worker: Optional[MetadataScanWorker] = None

        self._setup_columns()
        self._setup_ui()

    def _setup_columns(self) -> None:
        if self.mode in ("lights", "meteors"):
            self.headers = ["Usar", "REF", "Archivo", "Hora", "Exp.", "ISO", "f/"]
            self.col_use = 0
            self.col_ref = 1
            self.col_name = 2
            self.col_time = 3
            self.col_exp = 4
            self.col_iso = 5
            self.col_ap = 6
            self.col_del = -1
        elif self.mode == "darks":
            self.headers = ["Archivo", "Exp.", "ISO", ""]
            self.col_use = -1
            self.col_ref = -1
            self.col_name = 0
            self.col_time = -1
            self.col_exp = 1
            self.col_iso = 2
            self.col_ap = -1
            self.col_del = 3
        elif self.mode == "flats":
            self.headers = ["Archivo", "Exp.", "ISO", "f/", ""]
            self.col_use = -1
            self.col_ref = -1
            self.col_name = 0
            self.col_time = -1
            self.col_exp = 1
            self.col_iso = 2
            self.col_ap = 3
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
            self.col_del = -1

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

        if self.col_use >= 0:
            header.setSectionResizeMode(self.col_use, QHeaderView.ResizeToContents)
        if self.col_ref >= 0:
            header.setSectionResizeMode(self.col_ref, QHeaderView.ResizeToContents)
        if self.col_name >= 0:
            header.setSectionResizeMode(self.col_name, QHeaderView.Stretch)
        if self.col_time >= 0:
            header.setSectionResizeMode(self.col_time, QHeaderView.ResizeToContents)
        if self.col_exp >= 0:
            header.setSectionResizeMode(self.col_exp, QHeaderView.ResizeToContents)
        if self.col_iso >= 0:
            header.setSectionResizeMode(self.col_iso, QHeaderView.ResizeToContents)
        if self.col_ap >= 0:
            header.setSectionResizeMode(self.col_ap, QHeaderView.ResizeToContents)
        if self.col_del >= 0:
            header.setSectionResizeMode(self.col_del, QHeaderView.ResizeToContents)

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

    def sort_by_time(self, ascending: bool = True) -> None:
        """
        Reordena la lista de archivos por timestamp cronológico continuo (resuelve medianoche).
        Las tomas sin timestamp quedan al final.
        """
        if len(self.files_list) < 2:
            return

        ref_file = self.get_reference_file()

        def _sort_key(path: str):
            meta = self._metadata_cache.get(path, {})
            ts = meta.get("timestamp")
            if ts is not None:
                return (0, float(ts))
            return (1, path)

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
            self.files_list.sort()

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
        self.ref_idx = 0
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

        self._worker = MetadataScanWorker(missing_paths, meta_batch=meta_batch, parent=self)
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
