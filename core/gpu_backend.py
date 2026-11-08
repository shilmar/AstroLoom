# core/gpu_backend.py
"""
core/gpu_backend.py - Detección e inicialización de aceleración GPU (CuPy/CUDA, PyTorch CUDA y OpenCV OpenCL/UMat).
Proporciona aceleración opcional con tolerancia total a fallos, respaldo automático en CPU
y protección contra cuelgues de driver mediante el patrón Crash Sentinel (LightCraft).
"""
import logging
import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger("AstroLoom.GPU")

_CUPY_AVAILABLE = False
_TORCH_CUDA_AVAILABLE = False
_OPENCL_AVAILABLE = False
_GPU_DEVICE_NAME = ""
_cp = None
_torch = None
_USE_GPU = False
_CRASH_SENTINEL_TRIGGERED = False


def get_gpu_marker_path() -> str:
    """Retorna la ruta al archivo centinela de inicialización GPU (Crash Sentinel)."""
    appdata = os.environ.get("APPDATA")
    if appdata and os.path.isdir(appdata):
        marker_dir = os.path.join(appdata, "AstroLoom")
    else:
        marker_dir = os.path.join(os.path.expanduser("~"), ".astroloom")
    try:
        os.makedirs(marker_dir, exist_ok=True)
        return os.path.join(marker_dir, "gpu-init.marker")
    except Exception:
        fallback_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(fallback_root, "gpu-init.marker")


def _init_gpu_backends(force: bool = False) -> None:
    """
    Inicializa la detección de aceleración GPU con protección por Crash Sentinel (LightCraft pattern).
    Si ocurre un cuelgue irrecuperable a nivel de driver en un arranque previo, el centinela permanece
    y este método desactiva la GPU automáticamente, arrancando de forma segura en modo CPU.
    """
    global _CUPY_AVAILABLE, _TORCH_CUDA_AVAILABLE, _OPENCL_AVAILABLE
    global _GPU_DEVICE_NAME, _cp, _torch, _USE_GPU, _CRASH_SENTINEL_TRIGGERED

    marker_path = get_gpu_marker_path()

    # Si no se fuerza el reintento y el marcador existe, hubo cuelgue previo de driver
    if not force and os.path.isfile(marker_path):
        _CRASH_SENTINEL_TRIGGERED = True
        logger.warning(
            f"[Crash Sentinel] Se detectó un fallo crítico previo en driver GPU (marcador {marker_path}). "
            "Aceleración GPU desactivada por seguridad para prevenir cuelgues del proceso. Fallback seguro a CPU."
        )
        _CUPY_AVAILABLE = False
        _TORCH_CUDA_AVAILABLE = False
        _OPENCL_AVAILABLE = False
        _USE_GPU = False
        _GPU_DEVICE_NAME = "CPU (Fallback seguro tras cuelgue de driver GPU)"
        return

    # Escribir archivo marcador antes de sondear backends nativos
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write("gpu-init-in-progress\n")
    except Exception as e:
        logger.debug(f"[Crash Sentinel] No se pudo escribir marcador: {e}")

    try:
        _CRASH_SENTINEL_TRIGGERED = False

        # 1. Detección de CuPy (CUDA nativo)
        try:
            import cupy as cp_mod
            num_devices = cp_mod.cuda.runtime.getDeviceCount()
            if num_devices > 0:
                props = cp_mod.cuda.runtime.getDeviceProperties(0)
                gpu_raw_name = props.get("name", b"GPU NVIDIA")
                if isinstance(gpu_raw_name, bytes):
                    _GPU_DEVICE_NAME = gpu_raw_name.decode("utf-8", errors="ignore")
                else:
                    _GPU_DEVICE_NAME = str(gpu_raw_name)
                _CUPY_AVAILABLE = True
                _cp = cp_mod
        except Exception:
            _CUPY_AVAILABLE = False
            _cp = None

        # 2. Detección de PyTorch (CUDA / ROCm)
        try:
            import torch as torch_mod
            if torch_mod.cuda.is_available():
                _TORCH_CUDA_AVAILABLE = True
                _torch = torch_mod
                if not _GPU_DEVICE_NAME:
                    _GPU_DEVICE_NAME = torch_mod.cuda.get_device_name(0)
        except Exception:
            _TORCH_CUDA_AVAILABLE = False
            _torch = None

        # 3. Detección de OpenCV OpenCL (Transparent API / UMat)
        try:
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    _OPENCL_AVAILABLE = True
                    if not _GPU_DEVICE_NAME:
                        dev = cv2.ocl.Device.getDefault()
                        _GPU_DEVICE_NAME = dev.name() if dev else "OpenCL GPU"
        except Exception:
            _OPENCL_AVAILABLE = False

        _USE_GPU = _CUPY_AVAILABLE or _TORCH_CUDA_AVAILABLE or _OPENCL_AVAILABLE

    finally:
        # Si la inicialización terminó sin que el proceso colapsara a nivel nativo C/C++, eliminar centinela
        try:
            if os.path.isfile(marker_path):
                os.remove(marker_path)
        except Exception:
            pass


def is_crash_sentinel_active() -> bool:
    """Indica si el centinela de cuelgue GPU está activo debido a un fallo previo."""
    return _CRASH_SENTINEL_TRIGGERED


def clear_gpu_crash_sentinel() -> bool:
    """Elimina el marcador del centinela de cuelgue y reintenta la detección GPU."""
    marker_path = get_gpu_marker_path()
    if os.path.isfile(marker_path):
        try:
            os.remove(marker_path)
        except Exception as e:
            logger.error(f"[Crash Sentinel] No se pudo eliminar el marcador GPU: {e}")
            return False
    _init_gpu_backends(force=True)
    return True


def is_cupy_installed() -> bool:
    """Indica si CuPy está instalado y detecta hardware CUDA funcional."""
    return _CUPY_AVAILABLE


def is_torch_available() -> bool:
    """Indica si PyTorch con soporte CUDA está disponible."""
    return _TORCH_CUDA_AVAILABLE


def is_opencl_available() -> bool:
    """Indica si OpenCV OpenCL (Transparent API) está disponible y activo."""
    return _OPENCL_AVAILABLE and cv2.ocl.useOpenCL()


def is_gpu_available() -> bool:
    """Indica si hay al menos un backend de aceleración GPU activo (CuPy, PyTorch o OpenCL)."""
    return _CUPY_AVAILABLE or _TORCH_CUDA_AVAILABLE or _OPENCL_AVAILABLE


def get_gpu_name() -> str:
    """Retorna el nombre comercial del dispositivo acelerador detectado."""
    return _GPU_DEVICE_NAME if _GPU_DEVICE_NAME else "CPU (Sin aceleración GPU detectada)"


def set_gpu_enabled(enabled: bool) -> None:
    """Habilita o deshabilita el uso de la GPU (si está disponible en hardware)."""
    global _USE_GPU
    _USE_GPU = bool(enabled and is_gpu_available())
    if _OPENCL_AVAILABLE:
        try:
            cv2.ocl.setUseOpenCL(bool(enabled and _OPENCL_AVAILABLE))
        except Exception:
            pass


def is_gpu_enabled() -> bool:
    """Comprueba si el pipeline debe utilizar aceleración GPU."""
    return _USE_GPU


def get_array_module(arr=None):
    """
    Retorna el módulo matemático correspondiente (cupy si la GPU está activa,
    o numpy en caso contrario). Si arr se suministra, puede inspeccionarse su tipo.
    """
    if _USE_GPU and _cp is not None:
        if arr is not None:
            return _cp.get_array_module(arr)
        return _cp
    return np


def get_gpu_backend_info() -> Dict[str, Any]:
    """Retorna un diccionario detallado con el estado de todos los backends GPU."""
    return {
        "gpu_available": is_gpu_available(),
        "gpu_enabled": is_gpu_enabled(),
        "device_name": get_gpu_name(),
        "cupy_available": _CUPY_AVAILABLE,
        "torch_cuda_available": _TORCH_CUDA_AVAILABLE,
        "opencl_available": _OPENCL_AVAILABLE,
        "opencl_active": cv2.ocl.useOpenCL() if _OPENCL_AVAILABLE else False,
        "crash_sentinel_triggered": _CRASH_SENTINEL_TRIGGERED,
    }


def accelerated_filter2d(
    img: np.ndarray,
    kernel: np.ndarray,
    border_type: int = cv2.BORDER_REFLECT
) -> np.ndarray:
    """
    Aplica convolución 2D acelerada por GPU (OpenCL UMat) con tolerancia total a fallos.
    Si la GPU está inactiva o falla, recurre de forma transparente a cv2.filter2D en CPU.
    """
    if is_gpu_enabled() and is_opencl_available():
        try:
            u_img = cv2.UMat(img)
            u_k = cv2.UMat(kernel)
            u_out = cv2.filter2D(u_img, -1, u_k, borderType=border_type)
            return u_out.get()
        except Exception:
            pass
    return cv2.filter2D(img, -1, kernel, borderType=border_type)


def accelerated_gaussian_blur(
    img: np.ndarray,
    ksize: Tuple[int, int] = (0, 0),
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    border_type: int = cv2.BORDER_REFLECT
) -> np.ndarray:
    """
    Aplica desenfoque gaussiano acelerado por GPU (OpenCL UMat) con tolerancia total a fallos.
    Si la GPU está inactiva o falla, recurre de forma transparente a cv2.GaussianBlur en CPU.
    """
    if is_gpu_enabled() and is_opencl_available():
        try:
            u_img = cv2.UMat(img)
            u_out = cv2.GaussianBlur(u_img, ksize, sigmaX=sigma_x, sigmaY=sigma_y, borderType=border_type)
            return u_out.get()
        except Exception:
            pass
    return cv2.GaussianBlur(img, ksize, sigmaX=sigma_x, sigmaY=sigma_y, borderType=border_type)


# Ejecutar inicialización protegida al importar el módulo
_init_gpu_backends()
