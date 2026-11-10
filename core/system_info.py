# core/system_info.py
"""
core/system_info.py - Monitor ligero y nativo de recursos de memoria RAM física del sistema.
Elimina la dependencia externa de `psutil` utilizando el motor en Rust (astroloom_rs)
con respaldo sin dependencias vía ctypes (Windows API GlobalMemoryStatusEx) y /proc/meminfo (Linux).
"""
import sys
from typing import NamedTuple

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
    # 1. Intentar vía motor nativo Rust
    try:
        import astroloom_rs
        if hasattr(astroloom_rs, "get_system_memory"):
            total, avail, used, pct = astroloom_rs.get_system_memory()
            if total > 0:
                return MemoryInfo(total, avail, used, pct)
    except Exception:
        pass

    # 2. Respaldo transparente en Windows mediante Win32 API (kernel32.dll vía ctypes)
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total = int(status.ullTotalPhys)
                avail = int(status.ullAvailPhys)
                used = max(0, total - avail)
                pct = (used / total * 100.0) if total > 0 else 0.0
                return MemoryInfo(total, avail, used, pct)
        except Exception:
            pass

    # 3. Respaldo transparente en Linux leyendo /proc/meminfo
    elif sys.platform.startswith("linux"):
        try:
            total_kb = 0
            avail_kb = 0
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
            if total_kb > 0:
                total = total_kb * 1024
                avail = avail_kb * 1024
                used = max(0, total - avail)
                pct = (used / total * 100.0)
                return MemoryInfo(total, avail, used, pct)
        except Exception:
            pass

    # Valor seguro por defecto si no se puede determinar (16 GB ficticios)
    default_total = 16 * (1024 ** 3)
    default_avail = 8 * (1024 ** 3)
    return MemoryInfo(default_total, default_avail, default_total - default_avail, 50.0)
