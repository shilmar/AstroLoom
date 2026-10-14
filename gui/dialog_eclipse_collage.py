# gui/dialog_eclipse_collage.py
"""
gui/dialog_eclipse_collage.py - Envoltorio de compatibilidad hacia atrás para EclipseCollageDialog.
Utiliza internamente la implementación de EclipseCollageTab (gui/tab_eclipse_collage.py).
"""
from typing import List, Optional
from PySide6.QtWidgets import QDialog, QVBoxLayout
from gui.tab_eclipse_collage import (
    EclipseCollageTab,
    CollageInteractiveCanvas,
    EclipseAnalysisWorker
)

__all__ = ["EclipseCollageDialog", "CollageInteractiveCanvas", "EclipseAnalysisWorker"]


class EclipseCollageDialog(QDialog):
    """Ventana interactiva modal de diseño y exportación de collages de eclipses."""

    def __init__(self, parent=None, initial_filepaths: Optional[List[str]] = None, mode: str = "solar"):
        super().__init__(parent)
        self.setWindowTitle("Diseñador de Secuencia y Collage de Eclipses - Apilator")
        self.setMinimumSize(1240, 780)
        self.resize(1380, 840)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab = EclipseCollageTab(parent=self, initial_filepaths=initial_filepaths, mode=mode)
        layout.addWidget(self.tab)

    def closeEvent(self, event):
        self.tab.closeEvent(event)
        super().closeEvent(event)
