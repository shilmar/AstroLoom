# run_app.py
"""
run_app.py - Punto de entrada principal para el inicio de AstroLoom.
"""
import multiprocessing
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Soporte para pantallas HiDPI (monitores 2K, 4K y escalado de Windows)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from gui.theme import apply_theme
    apply_theme(app)

    # Pantalla de carga visual inmediata (< 80 ms)
    from gui.splash import AstroSplashScreen
    splash = AstroSplashScreen()
    splash.show()
    splash.set_status("Iniciando módulos y motor de procesado...", 10)

    from gui.main_window import MainWindow
    win = MainWindow(splash=splash)
    splash.set_status("Listo", 100)
    win.show()
    splash.finish(win)
    sys.exit(app.exec())
