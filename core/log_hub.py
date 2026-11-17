# core/log_hub.py
"""
core/log_hub.py - Hub centralizado de eventos y registro de actividad para AstroLoom.

Canaliza mensajes de estado, eventos de telemetría, avisos y errores desde
todos los módulos y procesos hacia la interfaz gráfica de forma segura para hilos.
"""
from collections import deque
from datetime import datetime
from typing import Deque, List, Tuple
import logging
from PySide6.QtCore import QObject, Signal


class _LogHub(QObject):
    """
    Despachador central de eventos de registro.
    Emite una señal Qt segura para subprocesos (QueuedConnection cruzando hilos).
    """
    message_logged = Signal(str, str, str)  # (timestamp_str, module, message)

    def __init__(self, maxlen: int = 3000):
        super().__init__()
        self._history: Deque[Tuple[str, str, str]] = deque(maxlen=maxlen)

    def log(self, module: str, message: str) -> None:
        """Registra un mensaje de forma thread-safe en el búfer circular y emite la señal."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._history.append((ts, module, message))
        self.message_logged.emit(ts, module, message)

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
