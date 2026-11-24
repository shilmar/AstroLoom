# core/log_hub.py
"""
core/log_hub.py - Hub centralizado de eventos y registro de actividad para AstroLoom.

Canaliza mensajes de estado, eventos de telemetría, avisos y errores desde
todos los módulos y procesos hacia la interfaz gráfica de forma segura para hilos.
"""
import os
import threading
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Tuple
import logging
from PySide6.QtCore import QObject, Signal

_LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
_LOG_FILE = os.path.join(_LOGS_DIR, "astroloom.log")
_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB


class _LogHub(QObject):
    """
    Despachador central de eventos de registro.
    Emite una señal Qt segura para subprocesos (QueuedConnection cruzando hilos)
    y persiste el flujo de eventos en disco con volcado inmediato.
    """
    message_logged = Signal(str, str, str)  # (timestamp_str, module, message)

    def __init__(self, maxlen: int = 3000):
        super().__init__()
        self._history: Deque[Tuple[str, str, str]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._file_handle: Optional[object] = None
        self._init_file()

    def _init_file(self) -> None:
        try:
            os.makedirs(_LOGS_DIR, exist_ok=True)
            if os.path.exists(_LOG_FILE) and os.path.getsize(_LOG_FILE) > _MAX_LOG_SIZE:
                old_file = os.path.join(_LOGS_DIR, "astroloom.old.log")
                if os.path.exists(old_file):
                    os.remove(old_file)
                os.rename(_LOG_FILE, old_file)
            self._file_handle = open(_LOG_FILE, "a", encoding="utf-8")
        except Exception:
            self._file_handle = None

    def log(self, module: str, message: str) -> None:
        """Registra un mensaje de forma thread-safe en el búfer circular, emite la señal y escribe a disco."""
        now = datetime.now()
        ts = now.strftime("%H:%M:%S")
        self._history.append((ts, module, message))
        self.message_logged.emit(ts, module, message)

        if self._file_handle is not None:
            with self._lock:
                try:
                    full_ts = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self._file_handle.write(f"[{full_ts}] [{module}] {message}\n")
                    self._file_handle.flush()
                except Exception:
                    pass

    def get_history(self) -> List[Tuple[str, str, str]]:
        """Devuelve una instantánea de todas las entradas almacenadas."""
        return list(self._history)

    def clear(self) -> None:
        """Limpia el historial en memoria."""
        self._history.clear()


# Instancia única singleton del sistema
log_hub = _LogHub()


class _QtLogHandler(logging.Handler):
    """Redirige logs de la librería estándar 'logging' bajo el espacio 'AstroLoom' hacia log_hub."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            mod_name = record.name.replace("AstroLoom.", "").replace("AstroLoom", "Sistema")
            log_hub.log(mod_name, msg)
        except Exception:
            pass


_handler = _QtLogHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger = logging.getLogger("AstroLoom")
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
