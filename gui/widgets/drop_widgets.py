# gui/widgets/drop_widgets.py
"""
gui/widgets/drop_widgets.py - Componentes UI con soporte integral para Arrastrar y Soltar (Drag & Drop).

Permite arrastrar archivos de imagen (RAW, TIFF, FITS, etc.) o carpetas completas
directamente desde el explorador de archivos del sistema operativo sobre las tablas y listas.
"""
import os
from typing import List, Set
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QTableWidget
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDragLeaveEvent, QDropEvent


SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raw", ".raf", ".orf", ".rw2",
    ".tif", ".tiff", ".fits", ".fit", ".fts", ".jpg", ".jpeg", ".png",
    ".devloom", ".loom"
}


def collect_image_files_from_mime(mime_data) -> List[str]:
    """
    Extrae recursivamente todas las rutas de archivo válidas de un QMimeData de arrastre.
    Procesa tanto archivos seleccionados individualmente como carpetas completas.
    """
    if not mime_data.hasUrls():
        return []

    collected: List[str] = []
    for url in mime_data.urls():
        local_path = url.toLocalFile()
        if not local_path:
            continue
        local_path = os.path.normpath(local_path)
        if os.path.isdir(local_path):
            for root, _, files in os.walk(local_path):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_IMAGE_EXTENSIONS:
                        collected.append(os.path.join(root, f))
        elif os.path.isfile(local_path):
            ext = os.path.splitext(local_path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                collected.append(local_path)

    # Eliminar duplicados preservando orden original
    seen = set()
    result = []
    for p in collected:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def classify_astro_files(paths: List[str]) -> dict:
    """
    Clasifica automáticamente una lista de rutas de archivo en categorías de astrofotografía:
    'Darks', 'Flats', 'Bias', 'Suelo' y 'Lights' basándose en subcarpetas y nombres de archivo.
    """
    categories = {
        "Darks": [],
        "Flats": [],
        "Bias": [],
        "Suelo": [],
        "Lights": []
    }
    for p in paths:
        norm = p.replace("\\", "/").lower()
        fname = os.path.basename(norm)
        parts = norm.split("/")

        # 1. Darks
        if any(d in parts[:-1] for d in ("dark", "darks", "tomas_oscuras")) or fname.startswith("dark") or "_dark" in fname:
            categories["Darks"].append(p)
        # 2. Flats
        elif any(f in parts[:-1] for f in ("flat", "flats", "tomas_planas")) or fname.startswith("flat") or "_flat" in fname:
            categories["Flats"].append(p)
        # 3. Bias / Offset
        elif any(b in parts[:-1] for b in ("bias", "biases", "offset", "offsets")) or fname.startswith("bias") or fname.startswith("offset") or "_bias" in fname or "_offset" in fname:
            categories["Bias"].append(p)
        # 4. Suelo / Paisaje
        elif any(s in parts[:-1] for s in ("suelo", "ground", "foreground", "paisaje")) or fname.startswith("suelo") or fname.startswith("ground") or "_suelo" in fname:
            categories["Suelo"].append(p)
        # 5. Lights (tomas de luz por defecto)
        else:
            categories["Lights"].append(p)

    return categories


class DropListWidget(QListWidget):
    """QListWidget enriquecido con eventos de arrastrar y soltar (Drag & Drop)."""
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._orig_style = ""

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._orig_style = self.styleSheet()
            self.setStyleSheet(self._orig_style + "\nQListWidget { border: 2px dashed #80d8ff; background-color: #141c24; }")
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.setStyleSheet(self._orig_style)
        event.accept()

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._orig_style)
        paths = collect_image_files_from_mime(event.mimeData())
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class DropTableWidget(QTableWidget):
    """QTableWidget enriquecido con eventos de arrastrar y soltar (Drag & Drop)."""
    files_dropped = Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self._orig_style = ""

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._orig_style = self.styleSheet()
            self.setStyleSheet(self._orig_style + "\nQTableWidget { border: 2px dashed #80d8ff; background-color: #141c24; }")
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.setStyleSheet(self._orig_style)
        event.accept()

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._orig_style)
        paths = collect_image_files_from_mime(event.mimeData())
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()
