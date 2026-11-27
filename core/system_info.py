# core/system_info.py
"""
core/system_info.py - Monitor ligero y nativo de recursos de memoria RAM física del sistema.
Impulsado por el motor nativo en Rust (astroloom_rs) para Windows y Linux.
"""
from typing import NamedTuple
import astroloom_rs


class MemoryInfo(NamedTuple):
    total: int
    available: int
    used: int
    percent: float


def get_system_memory() -> MemoryInfo:
    """
    Obtiene el estado de la memoria RAM física en bytes y porcentaje de uso.
    Devuelve MemoryInfo(total, available, used, percent).
    """
    total, avail, used, pct = astroloom_rs.get_system_memory()
    if total > 0:
        return MemoryInfo(total, avail, used, pct)
    default_total = 16 * (1024 ** 3)
    default_avail = 8 * (1024 ** 3)
    return MemoryInfo(default_total, default_avail, default_total - default_avail, 50.0)
