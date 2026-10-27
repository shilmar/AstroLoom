# gui/developer/async_worker.py
"""
gui/developer/async_worker.py - Gestor de renderizado asíncrono en segundo plano para DeveloperTab.
Permite ejecutar el pipeline de revelado en hilos del QThreadPool sin bloquear el hilo principal de la interfaz (UI).
Incluye cancelación de solicitudes obsoletas y entrega ordenada por request_id.
"""
from typing import Optional, Dict, Any, List
import logging
import numpy as np
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from gui.developer.developer_state import DeveloperState
from gui.developer.pipeline import apply_developer_pipeline

logger = logging.getLogger("AstroLoom.PipelineWorker")


class PipelineSignals(QObject):
    """Señales emitidas por el worker de procesamiento en segundo plano."""
    finished = Signal(int, object)  # (request_id, result_tuple)
    error = Signal(int, str)        # (request_id, error_msg)


class PipelineWorker(QRunnable):
    """
    Worker ejecutable en QThreadPool para procesar apply_developer_pipeline
    de forma completamente asíncrona sin bloquear la UI de AstroLoom.
    """

    def __init__(
        self,
        request_id: int,
        target_img: np.ndarray,
        state: DeveloperState,
        precomputed_mask: Optional[np.ndarray] = None,
        image_stars: Optional[np.ndarray] = None,
        image_starless: Optional[np.ndarray] = None,
        mask_layers: Optional[List[Any]] = None,
        curve_luts: Optional[Dict[str, Optional[np.ndarray]]] = None,
        psf_kernel: Optional[np.ndarray] = None,
        is_preview: bool = True,
        is_scrubbing: bool = False,
        crop_rect: Optional[tuple] = None,
        full_shape: Optional[tuple] = None,
        precomputed_linear_base: Optional[np.ndarray] = None,
        precomputed_stretched_base: Optional[np.ndarray] = None,
        precomputed_graded_base: Optional[np.ndarray] = None,
        precomputed_spatial_base: Optional[np.ndarray] = None,
        precomputed_stretched_stars: Optional[np.ndarray] = None,
        cached_ground_ai: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.request_id = request_id
        self.target_img = target_img
        self.state = state
        self.precomputed_mask = precomputed_mask
        self.image_stars = image_stars
        self.image_starless = image_starless
        self.mask_layers = mask_layers
        self.curve_luts = curve_luts
        self.psf_kernel = psf_kernel
        self.is_preview = is_preview
        self.is_scrubbing = is_scrubbing
        self.crop_rect = crop_rect
        self.full_shape = full_shape
        self.precomputed_linear_base = precomputed_linear_base
        self.precomputed_stretched_base = precomputed_stretched_base
        self.precomputed_graded_base = precomputed_graded_base
        self.precomputed_spatial_base = precomputed_spatial_base
        self.precomputed_stretched_stars = precomputed_stretched_stars
        self.cached_ground_ai = cached_ground_ai

        self.signals = PipelineSignals()
        self._is_aborted = False

    def abort(self):
        """Marca el worker como cancelado."""
        self._is_aborted = True

    def run(self):
        if self._is_aborted:
            return

        try:
            res = apply_developer_pipeline(
                target_img=self.target_img,
                state=self.state,
                precomputed_mask=self.precomputed_mask,
                image_stars=self.image_stars,
                image_starless=self.image_starless,
                mask_layers=self.mask_layers,
                curve_luts=self.curve_luts,
                psf_kernel=self.psf_kernel,
                is_preview=self.is_preview,
                is_scrubbing=self.is_scrubbing,
                crop_rect=self.crop_rect,
                full_shape=self.full_shape,
                precomputed_linear_base=self.precomputed_linear_base,
                precomputed_stretched_base=self.precomputed_stretched_base,
                precomputed_graded_base=self.precomputed_graded_base,
                precomputed_spatial_base=self.precomputed_spatial_base,
                precomputed_stretched_stars=self.precomputed_stretched_stars,
                cached_ground_ai=self.cached_ground_ai,
                return_milestones=True,
                abort_flag=lambda: self._is_aborted,
            )

            if not self._is_aborted and res is not None:
                self.signals.finished.emit(self.request_id, res)
        except Exception as e:
            if not self._is_aborted:
                logger.error(f"Error en PipelineWorker #{self.request_id}: {e}", exc_info=True)
                self.signals.error.emit(self.request_id, str(e))


class AsyncPipelineManager(QObject):
    """
    Coordinador de renderizado asíncrono. Asegura que sólo la solicitud más reciente
    actualice la interfaz gráfica y descarta automáticamente resultados obsoletos.
    """
    preview_ready = Signal(int, object)
    pipeline_failed = Signal(int, str)

    def __init__(self, parent: Optional[QObject] = None, max_threads: int = 2):
        super().__init__(parent)
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(max_threads)
        self._current_request_id: int = 0
        self._active_worker: Optional[PipelineWorker] = None

    def cancel_pending(self):
        """Cancela el worker activo si existe."""
        if self._active_worker is not None:
            self._active_worker.abort()
            self._active_worker = None

    def submit(
        self,
        target_img: np.ndarray,
        state: DeveloperState,
        precomputed_mask: Optional[np.ndarray] = None,
        image_stars: Optional[np.ndarray] = None,
        image_starless: Optional[np.ndarray] = None,
        mask_layers: Optional[List[Any]] = None,
        curve_luts: Optional[Dict[str, Optional[np.ndarray]]] = None,
        psf_kernel: Optional[np.ndarray] = None,
        is_preview: bool = True,
        is_scrubbing: bool = False,
        crop_rect: Optional[tuple] = None,
        full_shape: Optional[tuple] = None,
        precomputed_linear_base: Optional[np.ndarray] = None,
        precomputed_stretched_base: Optional[np.ndarray] = None,
        precomputed_graded_base: Optional[np.ndarray] = None,
        precomputed_spatial_base: Optional[np.ndarray] = None,
        precomputed_stretched_stars: Optional[np.ndarray] = None,
        cached_ground_ai: Optional[np.ndarray] = None,
    ) -> int:
        """Envía una nueva tarea de renderizado descartando las solicitudes anteriores."""
        self._current_request_id += 1
        req_id = self._current_request_id

        # Abortar worker anterior si aún sigue ejecutándose
        self.cancel_pending()

        worker = PipelineWorker(
            request_id=req_id,
            target_img=target_img,
            state=state,
            precomputed_mask=precomputed_mask,
            image_stars=image_stars,
            image_starless=image_starless,
            mask_layers=mask_layers,
            curve_luts=curve_luts,
            psf_kernel=psf_kernel,
            is_preview=is_preview,
            is_scrubbing=is_scrubbing,
            crop_rect=crop_rect,
            full_shape=full_shape,
            precomputed_linear_base=precomputed_linear_base,
            precomputed_stretched_base=precomputed_stretched_base,
            precomputed_graded_base=precomputed_graded_base,
            precomputed_spatial_base=precomputed_spatial_base,
            precomputed_stretched_stars=precomputed_stretched_stars,
            cached_ground_ai=cached_ground_ai,
        )
        self._active_worker = worker

        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)

        self.thread_pool.start(worker)
        return req_id

    def _on_worker_finished(self, req_id: int, result: Any):
        # Descartar si una solicitud más nueva ya fue enviada
        if req_id == self._current_request_id:
            self._active_worker = None
            self.preview_ready.emit(req_id, result)

    def _on_worker_error(self, req_id: int, error_msg: str):
        if req_id == self._current_request_id:
            self._active_worker = None
            self.pipeline_failed.emit(req_id, error_msg)
