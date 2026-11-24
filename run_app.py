# run_app.py
"""
run_app.py - Punto de entrada principal para el inicio de AstroLoom.
"""
import multiprocessing
import os
import sys
import traceback
from datetime import datetime
import faulthandler

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Inicialización de faulthandler para capturar caídas nativas C/C++/Rust
_logs_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(_logs_dir, exist_ok=True)
_crash_log_path = os.path.join(_logs_dir, "crash.log")
try:
    _crash_file = open(_crash_log_path, "a", encoding="utf-8")
    _crash_file.write(f"\n--- Sesion iniciada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (PID: {os.getpid()}) ---\n")
    _crash_file.flush()
    faulthandler.enable(file=_crash_file, all_threads=True)
except Exception:
    _crash_file = None


def _handle_uncaught_exception(exc_type, exc_value, exc_tb):
    """Captura excepciones no controladas de Python y las persiste en crash.log y log_hub."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    msg = f"[{ts}] [CRITICAL_UNCAUGHT_EXCEPTION]\n{tb}\n"
    try:
        with open(_crash_log_path, "a", encoding="utf-8") as f:
            f.write(msg)
            f.flush()
    except Exception:
        pass

    try:
        from core.log_hub import log_hub
        log_hub.log("CRASH", f"Excepcion no controlada: {exc_value}")
    except Exception:
        pass

    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _handle_uncaught_exception

from PySide6.QtCore import Qt, qInstallMessageHandler, QtMsgType
from PySide6.QtWidgets import QApplication


def _qt_message_handler(mode, context, message):
    """Captura avisos internos y errores fatales de Qt en el registro de caídas."""
    if mode == QtMsgType.QtDebugMsg:
        return
    levels = {
        QtMsgType.QtInfoMsg: "QtInfo",
        QtMsgType.QtWarningMsg: "QtWarning",
        QtMsgType.QtCriticalMsg: "QtCritical",
        QtMsgType.QtFatalMsg: "QtFatal",
    }
    lvl = levels.get(mode, "Qt")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{lvl}] {message}\n"
    try:
        with open(_crash_log_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except Exception:
        pass


qInstallMessageHandler(_qt_message_handler)

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
