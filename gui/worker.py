"""
gui/worker.py - Hilos secundarios de ejecución (QThread) para el pipeline de Apilator,
incluyendo apilado dual subpíxel, extracción de gradiente GraXpert y StarNet++.
"""

import os
import gc
import shutil
import tempfile
import time
import cv2
import numpy as np
import tifffile
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from PySide6.QtCore import QThread, Signal

from core.config_manager import load_config
from core.stacking import (
    load_image_as_float32, 
    get_image_dimensions,
    detect_sky_stars, 
    refine_star_centroids,
    stream_stack_auto,
    create_master_bias,
    create_master_dark,
    create_master_flat,
    calibrate_light,
    preprocess_subframe_lp,          
    align_single_light_task,
    save_frame_float32,
    evaluate_storage_mode,
    StorageModeDecision
)
from core.graxpert_bridge import run_graxpert_background_extraction
from core.starnet_bridge import run_starnet
from core.icc_manager import (
    extract_icc_profile, 
    get_icc_profile_bytes, 
    get_profile_description, 
    save_tiff_with_icc
)


def _terminate_executor_processes(executor: ProcessPoolExecutor):
    """Fuerza la terminación inmediata y segura de los procesos hijos."""
    if executor is None:
        return
    try:
        for pid, process in list(executor._processes.items()):
            try:
                process.terminate()
                process.join(timeout=0.05)
            except Exception:
                pass
    except Exception:
        pass


class StackingWorker(QThread):
    progress_changed = Signal(int)
    status_changed = Signal(str)
    finished_success = Signal(str)
    error_occurred = Signal(str)
    cancelled = Signal()

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._is_cancelled = False
        self._executor = None

    def cancel(self):
        """Solicita la parada inmediata del worker y de los subprocesos."""
        self._is_cancelled = True
        self.status_changed.emit("Deteniendo tareas y liberando recursos...")
        if self._executor is not None:
            try:
                _terminate_executor_processes(self._executor)
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        temp_dir = None
        ref_raw_ground = None
        t_global_start = time.perf_counter()
        sky_frames_collection = []
        ground_frames_collection = []

        try:
            # 1. Extracción de configuración y resolución de rutas de trabajo
            lights = self.config.get("lights", [])
            darks = self.config.get("darks", [])
            flats = self.config.get("flats", [])
            bias = self.config.get("bias", [])
            sky_mask = self.config.get("mask", None)
            mode = self.config.get("mode", "fixed_tripod")
            ground_mode = self.config.get("ground_mode", "dual")
            external_ground_path = self.config.get("external_ground_path", None)
            kappa = float(self.config.get("kappa", 2.2))

            app_cfg = load_config()
            default_stacked_dir = app_cfg.get("stacked_dir", os.getcwd())
            raw_output = self.config.get("output_path", "resultado_dual_32bit.tiff")

            if not os.path.isabs(raw_output):
                output_path = os.path.normpath(os.path.join(default_stacked_dir, raw_output))
            else:
                output_path = os.path.normpath(raw_output)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            lp_method = self.config.get("lp_method", "standard")
            lp_strength = float(self.config.get("lp_strength", 0.5))
            
            cpu_workers = min(4, max(1, int(self.config.get("cpu_workers", 4))))
            storage_strategy = self.config.get("storage_strategy", "auto")

            need_stack_ground = (sky_mask is not None and mode == "fixed_tripod" and ground_mode == "dual")

            if len(lights) < 2:
                self.error_occurred.emit("Se necesitan al menos 2 tomas de luz para apilar.")
                return

            self.status_changed.emit("Iniciando preparación del entorno temporal...")
            self.progress_changed.emit(1)

            temp_dir = tempfile.mkdtemp(prefix="astro_gui_")

            # ------------------------------------------------------------
            # VALIDACIÓN PREVENTIVA DE GEOMETRÍA Y ORIENTACIÓN
            # ------------------------------------------------------------
            ref_path = lights[0]
            ref_h, ref_w = get_image_dimensions(ref_path)

            checks = [
                ("Darks", darks),
                ("Flats", flats),
                ("Bias", bias),
            ]

            for group_name, file_list in checks:
                if not file_list:
                    continue
                sample_file = file_list[0]
                sample_h, sample_w = get_image_dimensions(sample_file)

                if (sample_h, sample_w) != (ref_h, ref_w):
                    # Comprobar si es un problema de orientación traspuesta
                    if (sample_h, sample_w) == (ref_w, ref_h):
                        err_msg = (
                            f"Discrepancia de orientación en {group_name}:\n\n"
                            f"• Las tomas Light están en {'Vertical' if ref_h > ref_w else 'Horizontal'} ({ref_w}x{ref_h} px).\n"
                            f"• Las tomas {group_name} ({os.path.basename(sample_file)}) están en "
                            f"{'Horizontal' if ref_h > ref_w else 'Vertical'} ({sample_w}x{sample_h} px).\n\n"
                            f"Carga tomas de {group_name} disparadas con la misma orientación de la cámara."
                        )
                    else:
                        err_msg = (
                            f"Incompatibilidad de resolución en {group_name}:\n\n"
                            f"• Referencia Light: {ref_w}x{ref_h} px\n"
                            f"• {group_name} ({os.path.basename(sample_file)}): {sample_w}x{sample_h} px\n\n"
                            f"Los archivos de calibración deben corresponder al mismo sensor y recorte."
                        )

                    self.status_changed.emit(f"[ERROR CONFIGURACIÓN] {err_msg}")
                    self.error_occurred.emit(err_msg)
                    return

            # ============================================================
            # FASE 1/4: CALIBRACIÓN (BIAS, DARKS, FLATS) (2% -> 10%)
            # ============================================================
            master_bias = None
            if bias:
                t0_bias = time.perf_counter()
                self.status_changed.emit(f"[Fase 1/4] Generando Master Bias a partir de {len(bias)} tomas...")
                self.progress_changed.emit(3)
                master_bias = create_master_bias(bias)
                if self._is_cancelled:
                    self.cancelled.emit()
                    return
                dt_bias = time.perf_counter() - t0_bias
                self.status_changed.emit(f"-> Master Bias completado en {dt_bias:.1f}s.")
            else:
                self.status_changed.emit("[Fase 1/4] Sin tomas Bias.")

            master_dark = None
            if darks:
                t0_darks = time.perf_counter()
                self.status_changed.emit(f"[Fase 1/4] Generando Master Dark a partir de {len(darks)} tomas...")
                self.progress_changed.emit(5)
                master_dark = create_master_dark(darks)
                if self._is_cancelled:
                    self.cancelled.emit()
                    return
                dt_darks = time.perf_counter() - t0_darks
                self.status_changed.emit(f"-> Master Dark completado en {dt_darks:.1f}s.")
            else:
                self.status_changed.emit("[Fase 1/4] Sin tomas Dark.")

            master_flat = None
            if flats:
                t0_flats = time.perf_counter()
                self.status_changed.emit(f"[Fase 1/4] Generando Master Flat a partir de {len(flats)} tomas...")
                self.progress_changed.emit(8)
                master_flat = create_master_flat(flats, master_dark=master_dark, master_bias=master_bias)
                if self._is_cancelled:
                    self.cancelled.emit()
                    return
                dt_flats = time.perf_counter() - t0_flats
                self.status_changed.emit(f"-> Master Flat completado en {dt_flats:.1f}s.")
            else:
                self.status_changed.emit("[Fase 1/4] Sin tomas Flat.")

            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.progress_changed.emit(10)

            # ============================================================
            # FASE 2/4: REFERENCIA Y ALINEACIÓN DE LIGHTS
            # ============================================================
            t0_ref = time.perf_counter()
            ref_path = lights[0]
            self.status_changed.emit(f"[Fase 2/4] Preparando toma de referencia: {os.path.basename(ref_path)}")
            
            ref_raw = load_image_as_float32(ref_path)
            h, w, c = ref_raw.shape

            use_ram_buffer, storage_msg = evaluate_storage_mode(
                storage_strategy, len(lights), h, w, c, need_stack_ground
            )
            self.status_changed.emit(f"[Fase 2/4] Estrategia de almacenamiento: {storage_msg}")

            icc_profile = self.config.get("icc_profile", "srgb")
            ref_icc = extract_icc_profile(ref_path)
            if ref_icc is not None:
                icc_bytes_to_embed = ref_icc
                icc_desc = get_profile_description(ref_icc)
                self.status_changed.emit(f"[Fase 2/4] Perfil de color detectado en referencia: {icc_desc}")
            else:
                icc_bytes_to_embed = get_icc_profile_bytes(icc_profile)
                icc_desc = get_profile_description(icc_bytes_to_embed)
                self.status_changed.emit(f"[Fase 2/4] Perfil de color ICC incrustado: {icc_desc}")

            if sky_mask is not None and sky_mask.shape != (h, w):
                sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Persistencia temporal de maestros en .bin (Cero sobrecarga IPC)
            p_mb = os.path.join(temp_dir, "master_bias.bin") if master_bias is not None else None
            if p_mb and master_bias is not None:
                master_bias.astype(np.float32).tofile(p_mb)

            p_md = os.path.join(temp_dir, "master_dark.bin") if master_dark is not None else None
            if p_md and master_dark is not None:
                master_dark.astype(np.float32).tofile(p_md)

            p_mf = os.path.join(temp_dir, "master_flat.bin") if master_flat is not None else None
            if p_mf and master_flat is not None:
                master_flat.astype(np.float32).tofile(p_mf)

            # Persistir máscara de cielo temporalmente
            p_mask = os.path.join(temp_dir, "sky_mask.bin") if sky_mask is not None else None
            if p_mask and sky_mask is not None:
                sky_mask.astype(np.float32).tofile(p_mask)

            # Calibrar referencia in-place sobre ref_raw
            ref_sky = calibrate_light(
                ref_raw,
                master_dark=master_dark,
                master_flat=master_flat,
                master_bias=master_bias
            )
            # Mediana rápida por submuestreo
            ref_stats = {"median": np.median(ref_sky[::8, ::8], axis=(0, 1))}

            # Liberar arrays maestros de la memoria principal
            del master_bias, master_dark, master_flat
            gc.collect()

            self.status_changed.emit("[Fase 2/4] Extrayendo estrellas de la referencia...")
            ref_kp, ref_desc, norm_type = detect_sky_stars(ref_sky, sky_mask=sky_mask)
            self.status_changed.emit(f"-> Estrellas base detectadas: {len(ref_kp)}")

            if ref_desc is None or len(ref_kp) < 15:
                raise RuntimeError("No se detectaron suficientes estrellas en la toma de referencia para alinear.")

            # Conversión ligera a escala de grises: de RGB float32 a GRAY float32 (24 MB) y luego uint8
            ref_gray_f32 = cv2.cvtColor(ref_sky, cv2.COLOR_RGB2GRAY)
            ref_gray = np.clip(ref_gray_f32 * 255.0, 0, 255).astype(np.uint8)
            del ref_gray_f32

            p_ref_gray = os.path.join(temp_dir, "ref_gray.bin")
            ref_gray.tofile(p_ref_gray)
            del ref_gray

            # Si necesitamos suelo (modo dual o solo referencia), preservamos una copia antes del procesado de cielo
            ref_raw_ground = ref_sky.copy() if (need_stack_ground or ground_mode == "reference") else None

            # Preprocesar fondo in-place sin duplicaciones
            ref_sky_processed = preprocess_subframe_lp(
                ref_sky,
                method=lp_method,
                strength=lp_strength,
                sky_mask=sky_mask,
                ref_stats=ref_stats
            )

            if use_ram_buffer:
                sky_frames_collection.append(ref_sky_processed)
                if need_stack_ground and ref_raw_ground is not None:
                    ground_frames_collection.append(ref_raw_ground)
                    if ground_mode != "reference":
                        ref_raw_ground = None
            else:
                p_sky_0 = os.path.join(temp_dir, "sky_0000.bin")
                save_frame_float32(p_sky_0, ref_sky_processed)
                sky_frames_collection.append(p_sky_0)

                if need_stack_ground and ref_raw_ground is not None:
                    p_gnd_0 = os.path.join(temp_dir, "gnd_0000.bin")
                    save_frame_float32(p_gnd_0, ref_raw_ground)
                    ground_frames_collection.append(p_gnd_0)
                    if ground_mode != "reference":
                        del ref_raw_ground
                        ref_raw_ground = None
                del ref_sky_processed

            del ref_sky
            if 'ref_raw' in locals():
                del ref_raw
            gc.collect()

            dt_ref = time.perf_counter() - t0_ref
            self.status_changed.emit(f"-> Referencia lista en {dt_ref:.1f}s.")
            self.progress_changed.emit(15)

            if self._is_cancelled:
                self.cancelled.emit()
                return

            # Sub-fase de alineación concurrente (15% -> 60%)
            t0_align = time.perf_counter()
            total_lights = len(lights)
            discarded_count = 0
            to_align_count = max(1, total_lights - 1)

            ref_kps_pts = [(kp.pt[0], kp.pt[1]) for kp in ref_kp]

            tasks = []
            for idx, path in enumerate(lights[1:], start=2):
                task_args = (
                    idx, path, total_lights, temp_dir, mode,
                    p_md, p_mf, p_mb, p_mask, p_ref_gray,
                    ref_kps_pts, ref_desc, norm_type,
                    w, h, lp_method, lp_strength, ref_stats,
                    use_ram_buffer
                )
                tasks.append(task_args)

            aligned_results = []
            completed_count = 0

            # ThreadPoolExecutor: multi-hilo nativo en C sin sobrecarga IPC ni caídas de procesos en Windows
            safe_workers = max(1, min(3, (os.cpu_count() or 4) - 1))

            self._executor = ThreadPoolExecutor(max_workers=safe_workers)
            try:
                futures = [self._executor.submit(align_single_light_task, t) for t in tasks]

                for future in as_completed(futures):
                    if self._is_cancelled:
                        self.status_changed.emit("Alineación cancelada por el usuario.")
                        self.cancelled.emit()
                        return

                    res = future.result()
                    completed_count += 1
                    pct_align = 15 + int((completed_count / max(1, to_align_count)) * 45)
                    self.progress_changed.emit(min(60, pct_align))

                    aligned_results.append(res)

                    if res["success"]:
                        self.status_changed.emit(
                            f"[Fase 2/4] [{res['idx']}/{total_lights}] {res['filename']} -> OK "
                            f"({res['inliers']} inliers | Subpíxel | dx: {res['dx']:.1f}px, dy: {res['dy']:.1f}px)"
                        )
                    else:
                        discarded_count += 1
                        self.status_changed.emit(
                            f"[DESCARTADA] [{res['idx']}/{total_lights}] {res['filename']} -> {res['error']}"
                        )
            finally:
                if self._executor is not None:
                    try:
                        self._executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    self._executor = None

            if self._is_cancelled:
                self.cancelled.emit()
                return

            aligned_results.sort(key=lambda r: r["idx"])
            for res in aligned_results:
                if res["success"]:
                    sky_frames_collection.append(res["sky_data"])
                    if res["gnd_data"] is not None:
                        ground_frames_collection.append(res["gnd_data"])

            del aligned_results
            gc.collect()

            dt_align = time.perf_counter() - t0_align
            self.status_changed.emit(
                f"-> Alineación completada ({len(sky_frames_collection)}/{total_lights} tomas integradas en {dt_align:.1f}s)."
            )

            if len(sky_frames_collection) < 2:
                raise RuntimeError(
                    f"Fallo crítico en alineación: Solo se registraron {len(sky_frames_collection)} tomas válidas. "
                    "Se requieren al menos 2 tomas para apilar."
                )

            self.progress_changed.emit(60)

            # ============================================================
            # FASE 3/4: INTEGRACIÓN MATEMÁTICA (Rango 60% -> 95%)
            # ============================================================
            if self._is_cancelled:
                self.cancelled.emit()
                return

            has_ground = (sky_mask is not None and mode == "fixed_tripod")
            sky_target_pct = 78 if has_ground else 95

            self.status_changed.emit(
                f"[Fase 3/4] Apilando Cielo ({len(sky_frames_collection)} tomas) con CPU Multi-Core (Kappa={kappa:.1f}, LP={lp_method})..."
            )
            self.progress_changed.emit(63)

            t0_stack_sky = time.perf_counter()
            sky_stacked = stream_stack_auto(
                sky_frames_collection, 
                (h, w, c), 
                chunk_rows=200, 
                kappa=kappa,
                lp_method=lp_method,
                lp_strength=lp_strength
            )

            # Liberación inmediata de la colección de cielo de la memoria RAM / disco
            sky_frames_collection.clear()
            gc.collect()

            if self._is_cancelled:
                del sky_stacked
                self.cancelled.emit()
                return

            dt_stack_sky = time.perf_counter() - t0_stack_sky
            self.status_changed.emit(f"-> Apilado de cielo completado en {dt_stack_sky:.1f}s.")
            self.progress_changed.emit(sky_target_pct)

            # ============================================================
            # OBTENCIÓN DE LA CAPA DE SUELO
            # ============================================================
            ground_layer = None

            if has_ground and sky_mask is not None:
                if self._is_cancelled:
                    del sky_stacked
                    self.cancelled.emit()
                    return

                # MODO 1: Apilar ráfaga de suelo completa (Dual)
                if ground_mode == "dual" and len(ground_frames_collection) >= 2:
                    self.status_changed.emit(
                        f"[Fase 3/4] Apilando Suelo ({len(ground_frames_collection)} tomas) con CPU Multi-Core (Kappa={kappa:.1f})..."
                    )
                    self.progress_changed.emit(80)

                    t0_stack_gnd = time.perf_counter()
                    ground_layer = stream_stack_auto(
                        ground_frames_collection, 
                        (h, w, c), 
                        chunk_rows=200, 
                        kappa=kappa
                    )
                    
                    # Liberación inmediata de la colección de suelo de la memoria RAM / disco
                    ground_frames_collection.clear()
                    gc.collect()

                    if self._is_cancelled:
                        del sky_stacked, ground_layer
                        self.cancelled.emit()
                        return

                    dt_stack_gnd = time.perf_counter() - t0_stack_gnd
                    self.status_changed.emit(f"-> Apilado de suelo completado en {dt_stack_gnd:.1f}s.")

                # MODO 2: Suelo de la imagen de referencia (sin promediar)
                elif ground_mode == "reference" and ref_raw_ground is not None:
                    self.status_changed.emit("[Fase 3/4] Usando suelo de la toma de referencia (sin promediar)...")
                    ground_layer = ref_raw_ground
                    ref_raw_ground = None

                # MODO 3: Suelo externo asignado desde la pestaña Suelo
                elif ground_mode == "external" and external_ground_path:
                    self.status_changed.emit(f"[Fase 3/4] Cargando suelo externo: {os.path.basename(external_ground_path)}...")
                    ext_img = load_image_as_float32(external_ground_path)
                    if ext_img.shape[:2] != (h, w):
                        ext_img = cv2.resize(ext_img, (w, h), interpolation=cv2.INTER_LINEAR)
                    ground_layer = ext_img

                self.progress_changed.emit(95)

            # ============================================================
            # FASE 4/4: COMPOSICIÓN Y GUARDADO
            # ============================================================
            if self._is_cancelled:
                del sky_stacked
                if ground_layer is not None:
                    del ground_layer
                self.cancelled.emit()
                return

            if has_ground and sky_mask is not None and ground_layer is not None:
                t0_comp = time.perf_counter()
                self.status_changed.emit("[Fase 4/4] Componiendo imagen final de 32 bits sin discontinuidades...")
                self.progress_changed.emit(96)

                raw_mask = np.clip(sky_mask.astype(np.float32), 0.0, 1.0)
                smooth_mask = cv2.GaussianBlur(raw_mask, (5, 5), sigmaX=0.8)
                mask_3d = smooth_mask[..., np.newaxis] if smooth_mask.ndim == 2 else smooth_mask

                base_composite = (sky_stacked * mask_3d) + (ground_layer * (1.0 - mask_3d))

                transition_zone = (mask_3d > 0.02) & (mask_3d < 0.90)
                final_composite = np.where(
                    transition_zone & (ground_layer < base_composite),
                    ground_layer * (1.0 - mask_3d * 0.5) + base_composite * (mask_3d * 0.5),
                    base_composite
                )

                del ground_layer, mask_3d, smooth_mask, base_composite, sky_stacked
                dt_comp = time.perf_counter() - t0_comp
                self.status_changed.emit(f"-> Composición completada en {dt_comp:.2f}s.")
            else:
                final_composite = sky_stacked
                self.status_changed.emit("[Fase 4/4] Sin capa de suelo adicional. Saltando composición de máscara.")

            if self._is_cancelled:
                del final_composite
                self.cancelled.emit()
                return

            # Guardado TIFF final con metadatos fotométricos RGB y perfil ICC incrustado
            t0_io = time.perf_counter()
            self.progress_changed.emit(98)
            self.status_changed.emit(f"[Fase 4/4] Guardando TIFF de 32 bits ({icc_desc}) en: {os.path.basename(output_path)}...")
            
            save_tiff_with_icc(
                output_path,
                final_composite,
                icc_bytes=icc_bytes_to_embed,
                compression='zlib',
                photometric='rgb',
                convert_color=False
            )
            del final_composite

            dt_io = time.perf_counter() - t0_io
            self.status_changed.emit(f"-> Guardado TIFF completado en {dt_io:.1f}s.")

            dt_total = time.perf_counter() - t_global_start
            m, s = divmod(dt_total, 60)
            self.progress_changed.emit(100)
            self.status_changed.emit(f"¡Apilado completado exitosamente en {int(m)}m {s:.1f}s!")
            self.finished_success.emit(output_path)

        except Exception as exc:
            if not self._is_cancelled:
                import traceback
                err_detail = traceback.format_exc()
                err_msg = str(exc).strip() or exc.__class__.__name__
                print(f"\n[TRACEBACK COMPLETO WORKER]:\n{err_detail}")
                self.error_occurred.emit(f"{err_msg}\n\nDetalle técnico:\n{err_detail[-300:]}")
            else:
                self.cancelled.emit()
                
        finally:
            sky_frames_collection.clear()
            ground_frames_collection.clear()
            gc.collect()
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                

class GraXpertWorker(QThread):
    finished_success = Signal(np.ndarray)
    error_occurred = Signal(str)
    status_changed = Signal(str)
    cancelled = Signal()

    def __init__(self, image_rgb: np.ndarray, sky_mask: np.ndarray = None, smoothing: float = 0.5, parent=None):
        super().__init__(parent)
        self.image_rgb = image_rgb
        self.sky_mask = sky_mask
        self.smoothing = smoothing
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.status_changed.emit("Invocando GraXpert AI para extracción de fondo...")
            if self._is_cancelled:
                self.cancelled.emit()
                return

            corrected = run_graxpert_background_extraction(
                self.image_rgb, 
                sky_mask=self.sky_mask, 
                smoothing=self.smoothing
            )
            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.status_changed.emit("Extracción de gradiente completada con éxito.")
            self.finished_success.emit(corrected)
        except Exception as e:
            if not self._is_cancelled:
                self.error_occurred.emit(str(e))
            else:
                self.cancelled.emit()


class StarNetWorker(QThread):
    status_changed = Signal(str)
    finished_success = Signal(object, object)
    error_occurred = Signal(str)
    cancelled = Signal()

    def __init__(self, img_rgb: np.ndarray, sky_mask: np.ndarray = None, stride: int = 256, custom_exe: str = None, parent=None):
        super().__init__(parent)
        self.img_rgb = img_rgb
        self.sky_mask = sky_mask
        self.stride = stride
        self.custom_exe = custom_exe
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self._is_cancelled:
                self.cancelled.emit()
                return

            starless, stars = run_starnet(
                self.img_rgb,
                sky_mask=self.sky_mask,
                stride=self.stride,
                starnet_exe=self.custom_exe,
                log_callback=self.status_changed.emit
            )
            if self._is_cancelled:
                self.cancelled.emit()
                return

            self.finished_success.emit(starless, stars)
        except Exception as exc:
            if not self._is_cancelled:
                self.error_occurred.emit(str(exc))
            else:
                self.cancelled.emit()