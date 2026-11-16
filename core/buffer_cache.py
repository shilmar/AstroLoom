# core/buffer_cache.py
"""
core/buffer_cache.py - Caché LRU de búferes intermedios con presupuesto estricto de memoria RAM.

Inspirado en el pixelpipe_cache de Darktable pero con arquitectura ligera y desacoplada:
almacena los hitos intermedios del pipeline de revelado asociados a una clave determinista y forma
dimensional, permitiendo retener simultáneamente vistas de pantalla completa (Fit) y
recortes de inspección al 100% (Zoom 1:1) sin recalcular ni desbordar la memoria.
"""

from collections import OrderedDict
from typing import Optional, Tuple, Any
import numpy as np
import logging

from core.system_info import get_system_memory

logger = logging.getLogger("AstroLoom.BufferCache")


class PipelineBufferCache:
    """
    Caché LRU (Least Recently Used) acotada por presupuesto de memoria física.
    """

    def __init__(self, max_bytes: Optional[int] = None):
        self._cache: OrderedDict[Tuple[str, Any, Tuple[int, ...]], np.ndarray] = OrderedDict()
        self._current_bytes: int = 0
        if max_bytes is not None:
            self._max_bytes = max(1024, max_bytes)
        else:
            self._max_bytes = self._calc_default_budget()

    @staticmethod
    def _calc_default_budget() -> int:
        """Calcula el presupuesto óptimo como el 25% de la RAM disponible (entre 512 MB y 4 GB)."""
        try:
            mem = get_system_memory()
            budget = int(mem.available * 0.25)
            return max(512 * 1024 * 1024, min(4096 * 1024 * 1024, budget))
        except Exception:
            return 1024 * 1024 * 1024

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    def get(self, stage: str, key: Any, shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Recupera un búfer intermedio de la caché.
        Si existe y coincide en dimensiones, lo marca como recientemente usado y lo devuelve.
        """
        if key is None:
            return None
        cache_key = (stage, key, shape)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        return None

    def put(self, stage: str, key: Any, shape: Tuple[int, ...], img: Optional[np.ndarray]) -> None:
        """
        Almacena un búfer en la caché, desalojando elementos antiguos si excede el presupuesto.
        """
        if img is None or key is None:
            return

        cache_key = (stage, key, shape)
        item_bytes = img.nbytes

        # Si el elemento individual es mayor que el presupuesto total, no cachear
        if item_bytes > self._max_bytes:
            return

        # Si ya existe, restar sus bytes anteriores
        if cache_key in self._cache:
            self._current_bytes -= self._cache[cache_key].nbytes
            del self._cache[cache_key]

        # Desalojo LRU hasta hacer espacio
        while self._current_bytes + item_bytes > self._max_bytes and self._cache:
            _, evicted_img = self._cache.popitem(last=False)
            self._current_bytes -= evicted_img.nbytes

        self._cache[cache_key] = img
        self._current_bytes += item_bytes

    def clear(self) -> None:
        """Vacía completamente la caché y libera memoria."""
        self._cache.clear()
        self._current_bytes = 0

    def __len__(self) -> int:
        return len(self._cache)
