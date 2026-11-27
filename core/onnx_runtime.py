# core/onnx_runtime.py
"""
core/onnx_runtime.py - Gestión unificada de sesiones ONNX Runtime para los modelos de IA de AstroLoom
(StarNet v2, NAFNet Denoise y SkySeg Horizon).

Centraliza:
- Selección de Execution Providers en Windows (DirectML/CUDA) y Linux (CUDA/ROCm/CPU).
- Configuración de sesión óptima (incluyendo enable_mem_pattern=False en DirectML para evitar OOM).
- Caché en disco de grafos optimizados (models/.cache/) y caché en memoria con expulsión/limpieza.
- Notificación unificada de degradación GPU -> CPU a la interfaz.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNX = False

logger = logging.getLogger("AstroLoom.ONNX")

_GPU_FALLBACK_LISTENERS: List[Callable[[str], None]] = []
_SESSION_CACHE: dict[str, "ort.InferenceSession"] = {}


def add_gpu_fallback_listener(cb: Callable[[str], None]) -> None:
    """Registra un callback invocado cuando una inferencia degrada de GPU a CPU."""
    if cb not in _GPU_FALLBACK_LISTENERS:
        _GPU_FALLBACK_LISTENERS.append(cb)


def notify_gpu_fallback(reason: str) -> None:
    """Notifica a todos los escuchas registrados de una degradación GPU -> CPU."""
    for cb in _GPU_FALLBACK_LISTENERS:
        try:
            cb(reason)
        except Exception:
            pass


def clear_session_cache() -> None:
    """Libera todas las sesiones ONNX cacheadas para devolver VRAM/RAM al sistema."""
    _SESSION_CACHE.clear()


def get_cached_session(model_path: str, force_cpu: bool = False) -> Optional["ort.InferenceSession"]:
    abs_path = os.path.abspath(model_path)
    key = f"{abs_path}_cpu" if force_cpu else abs_path
    return _SESSION_CACHE.get(key)


def set_cached_session(model_path: str, session: "ort.InferenceSession", force_cpu: bool = False) -> None:
    abs_path = os.path.abspath(model_path)
    key = f"{abs_path}_cpu" if force_cpu else abs_path
    _SESSION_CACHE[key] = session


def evict_session(model_path: str, force_cpu: bool = False) -> None:
    abs_path = os.path.abspath(model_path)
    key = f"{abs_path}_cpu" if force_cpu else abs_path
    _SESSION_CACHE.pop(key, None)


def _get_cached_graph_path(model_path: str, provider_tag: str) -> str:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(cache_dir, f"{stem}_{provider_tag}_opt.onnx")


def _is_cache_valid(cache_path: str, original_path: str) -> bool:
    try:
        if not os.path.isfile(cache_path) or os.path.getsize(cache_path) == 0:
            return False
        return os.path.getmtime(cache_path) >= os.path.getmtime(original_path)
    except Exception:
        return False


def create_inference_session(
    model_path: str,
    use_cache: bool = False,
    force_cpu: bool = False,
) -> "ort.InferenceSession":
    """
    Inicializa o recupera de caché una sesión ONNX Runtime con el proveedor más rápido
    disponible en Windows/Linux y caché de grafo optimizado en models/.cache/.
    """
    if not HAS_ONNX:
        raise RuntimeError("onnxruntime no está instalado en el entorno.")

    abs_path = os.path.abspath(model_path)
    cache_key = f"{abs_path}_cpu" if force_cpu else abs_path
    if use_cache and cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    available_providers = ort.get_available_providers()
    if force_cpu:
        providers = ["CPUExecutionProvider"]
        provider_tag = "cpu"
    else:
        providers = []
        for ep in ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider"):
            if ep in available_providers:
                providers.append(ep)
        providers.append("CPUExecutionProvider")
        provider_tag = "gpu" if len(providers) > 1 else "cpu"

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_cores = os.cpu_count() or 4
    opts.intra_op_num_threads = min(16, max(1, cpu_cores))
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # DirectML recomienda desactivar memory pattern para evitar fragmentación/OOM en VRAM
    if "DmlExecutionProvider" in providers:
        opts.enable_mem_pattern = False

    cache_file = _get_cached_graph_path(abs_path, provider_tag)

    if _is_cache_valid(cache_file, abs_path):
        try:
            sess = ort.InferenceSession(cache_file, sess_options=opts, providers=providers)
            if use_cache:
                _SESSION_CACHE[cache_key] = sess
            return sess
        except Exception as e:
            logger.warning(f"Error cargando grafo optimizado en caché ({cache_file}): {e}. Recompilando...")
            try:
                os.remove(cache_file)
            except OSError:
                pass

    try:
        opts.optimized_model_filepath = cache_file
    except Exception:
        pass

    try:
        sess = ort.InferenceSession(abs_path, sess_options=opts, providers=providers)
    except Exception:
        opts.optimized_model_filepath = ""
        sess = ort.InferenceSession(abs_path, sess_options=opts, providers=providers)

    if use_cache:
        _SESSION_CACHE[cache_key] = sess
    return sess
