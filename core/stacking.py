# core/stacking.py
import os

# Configuración de hilos OpenMP para LibRaw y bibliotecas numéricas nativas
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import gc
import cv2
import numpy as np
import rawpy
import tifffile
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from astropy.io import fits

# Aceleración nativa de CPU con Numba (fallback transparente a NumPy si no está disponible)
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(fastmath=True)
    def _sort_inplace(arr):
        """Ordenamiento por inserción in-place ultrarrápido para vectores locales pequeños (N <= 100)."""
        n = len(arr)
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    @njit(parallel=True, fastmath=True, nogil=True)
    def _kappa_sigma_kernel_numba(data, kappa):
        """
        Kernel compilado en LLVM para integración Kappa-Sigma con MAD en una sola pasada.
        Opera en memoria caché L1/L2 con cero asignaciones en el heap de Python.
        data: shape (n_frames, total_items)
        """
        n_frames, total_items = data.shape
        out = np.empty(total_items, dtype=np.float32)

        for p in prange(total_items):
            # 1. Copiar valores locales a un buffer de pila
            v = np.empty(n_frames, dtype=np.float32)
            for i in range(n_frames):
                v[i] = data[i, p]

            # 2. Mediana local
            _sort_inplace(v)
            if n_frames % 2 == 1:
                med = v[n_frames // 2]
            else:
                med = 0.5 * (v[n_frames // 2 - 1] + v[n_frames // 2])

            # 3. Desviaciones absolutas respecto a la mediana
            d = np.empty(n_frames, dtype=np.float32)
            for i in range(n_frames):
                d[i] = abs(data[i, p] - med)

            # 4. MAD (Median Absolute Deviation)
            _sort_inplace(d)
            if n_frames % 2 == 1:
                mad = d[n_frames // 2]
            else:
                mad = 0.5 * (d[n_frames // 2 - 1] + d[n_frames // 2])

            sigma = 1.4826 * mad + 1e-6
            low = med - kappa * sigma
            high = med + kappa * sigma

            # 5. Promedio de valores válidos
            s = 0.0
            cnt = 0
            for i in range(n_frames):
                val = data[i, p]
                if low <= val <= high:
                    s += val
                    cnt += 1

            if cnt > 0:
                out[p] = s / cnt
            else:
                out[p] = med

        return out

    @njit(parallel=True, fastmath=True, nogil=True)
    def _percentile_kernel_numba(data, target_p):
        """Kernel compilado para cálculo exacto de percentil con interpolación lineal."""
        n_frames, total_items = data.shape
        out = np.empty(total_items, dtype=np.float32)
        rank = (target_p / 100.0) * (n_frames - 1)
        i0 = int(rank)
        i1 = min(i0 + 1, n_frames - 1)
        frac = float(rank - i0)

        for p in prange(total_items):
            v = np.empty(n_frames, dtype=np.float32)
            for i in range(n_frames):
                v[i] = data[i, p]
            _sort_inplace(v)
            out[p] = v[i0] + frac * (v[i1] - v[i0])

        return out

    @njit(parallel=True, fastmath=True, nogil=True)
    def _median_kernel_numba(data):
        """Kernel compilado para cálculo rápido de mediana en franjas."""
        n_frames, total_items = data.shape
        out = np.empty(total_items, dtype=np.float32)

        for p in prange(total_items):
            v = np.empty(n_frames, dtype=np.float32)
            for i in range(n_frames):
                v[i] = data[i, p]
            _sort_inplace(v)
            if n_frames % 2 == 1:
                out[p] = v[n_frames // 2]
            else:
                out[p] = 0.5 * (v[n_frames // 2 - 1] + v[n_frames // 2])

        return out

    # Calentamiento JIT anticipado con arrays mínimos de prueba
    try:
        _dummy = np.zeros((2, 2), dtype=np.float32)
        _kappa_sigma_kernel_numba(_dummy, 2.2)
        _percentile_kernel_numba(_dummy, 50.0)
        _median_kernel_numba(_dummy)
    except Exception:
        pass


def load_image_as_float32(filepath: str, color_space: str = None) -> np.ndarray:
    """Carga imágenes optimizando memoria con operaciones in-place y espacio de color configurado."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw']:
        from core.icc_manager import get_rawpy_colorspace
        if color_space is None:
            from core.config_manager import get_config_val
            color_space = get_config_val("icc_profile", "srgb")
        raw_cs = get_rawpy_colorspace(color_space)

        with rawpy.imread(filepath) as raw:
            rgb = raw.postprocess(
                gamma=(2.222, 4.5),
                no_auto_bright=True,
                output_bps=16,
                use_camera_wb=True,
                output_color=raw_cs,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                median_filter_passes=0,
                adjust_maximum_thr=0.0
            )
            img_f32 = rgb.astype(np.float32)
            del rgb
            img_f32 *= np.float32(1.0 / 65535.0)
            return img_f32

    elif ext in ['.fits', '.fit', '.fts']:
        with fits.open(filepath) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.ndim == 3 and data.shape[0] in [3, 4]:
                data = np.transpose(data[:3], (1, 2, 0))
            d_min, d_max = float(data.min()), float(data.max())
            denom = d_max - d_min + 1e-8
            data -= d_min
            data /= denom
            return data

    elif ext in ['.tif', '.tiff']:
        data = tifffile.imread(filepath).astype(np.float32)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        d_max = float(data.max())
        if d_max <= 1.05 and float(data.min()) >= 0.0:
            return np.clip(data, 0.0, 1.0)
        max_val = 65535.0 if d_max > 255.0 else 255.0
        data /= max_val
        return np.clip(data, 0.0, 1.0)

    else:
        bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"No se pudo leer el archivo: {filepath}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        del bgr
        rgb *= np.float32(1.0 / 255.0)
        return rgb


def get_image_dimensions(filepath: str) -> tuple[int, int]:
    """Obtiene (alto, ancho) de forma rápida sin procesar la matriz completa."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw']:
        with rawpy.imread(filepath) as raw:
            h, w = raw.sizes.height, raw.sizes.width
            if raw.sizes.flip in [5, 6, 7, 8]:
                return w, h
            return h, w
    else:
        info = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if info is not None:
            return info.shape[0], info.shape[1]
    
    img = load_image_as_float32(filepath)
    h, w = img.shape[:2]
    del img
    return h, w


# =========================================================================
# CALIBRACIÓN: STREAMING DIRECTO A DISCO Y MEDIANA POR FRANJAS
# =========================================================================

def _median_stack_chunked(file_paths: list, chunk_size: int = 500, temp_dir: str = None) -> np.ndarray:
    """
    Calcula la mediana de una lista de tomas en disco sin cargar todas a la vez en RAM.
    Vuelca cada toma a un binario temporal y lee por franjas horizontales.
    """
    if not file_paths:
        return None
    if len(file_paths) == 1:
        return load_image_as_float32(file_paths[0])

    h, w = get_image_dimensions(file_paths[0])
    c = 3

    work_dir = temp_dir if (temp_dir and os.path.isdir(temp_dir)) else tempfile.gettempdir()
    tmp_files = []
    pid = os.getpid()

    try:
        # 1. Volcar cada imagen a binario en disco en paralelo (multi-hilo en CPU)
        def _process_calib_frame(item):
            i, p = item
            img = load_image_as_float32(p)
            tmp_p = os.path.join(work_dir, f"calib_tmp_{pid}_{i:04d}.bin")
            img.tofile(tmp_p)
            del img
            return tmp_p

        max_workers = min(4, max(1, (os.cpu_count() or 4) - 1), len(file_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tmp_files = list(executor.map(_process_calib_frame, enumerate(file_paths)))

        # 2. Asignar matriz final
        master = np.empty((h, w, c), dtype=np.float32)
        n_frames = len(tmp_files)

        # 3. Mediana por bloques de filas con mapeo de memoria directo
        mms = [np.memmap(tf, dtype=np.float32, mode="r", shape=(h, w, c)) for tf in tmp_files]
        try:
            for y1 in range(0, h, chunk_size):
                y2 = min(h, y1 + chunk_size)
                chunk_rows = y2 - y1

                chunk_buf = np.empty((n_frames, chunk_rows, w, c), dtype=np.float32)
                for f_idx, mm in enumerate(mms):
                    chunk_buf[f_idx] = mm[y1:y2]

                if HAS_NUMBA:
                    flat_chunk = chunk_buf.reshape(n_frames, -1)
                    master[y1:y2] = _median_kernel_numba(flat_chunk).reshape((chunk_rows, w, c))
                else:
                    master[y1:y2] = np.median(chunk_buf, axis=0).astype(np.float32)
                del chunk_buf
        finally:
            for mm in mms:
                if hasattr(mm, "_mmap") and mm._mmap is not None:
                    try:
                        mm._mmap.close()
                    except Exception:
                        pass
            del mms

    finally:
        # Limpieza de temporales
        for tf in tmp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except Exception:
                    pass
        gc.collect()

    return master


def create_master_bias(bias_paths: list, temp_dir: str = None) -> np.ndarray:
    if not bias_paths:
        return None
    return _median_stack_chunked(bias_paths, chunk_size=500, temp_dir=temp_dir)


def create_master_dark(dark_paths: list, master_bias: np.ndarray = None, temp_dir: str = None) -> np.ndarray:
    if not dark_paths:
        return None
    master_dark = _median_stack_chunked(dark_paths, chunk_size=500, temp_dir=temp_dir)
    if master_bias is not None:
        np.subtract(master_dark, master_bias, out=master_dark)
        np.maximum(master_dark, 0.0, out=master_dark)
    gc.collect()
    return master_dark


def create_master_flat(flat_paths: list, master_dark: np.ndarray = None, master_bias: np.ndarray = None, temp_dir: str = None) -> np.ndarray:
    if not flat_paths:
        return None
    pedestal = master_bias if master_bias is not None else master_dark
    master_flat = _median_stack_chunked(flat_paths, chunk_size=500, temp_dir=temp_dir)

    if pedestal is not None:
        np.subtract(master_flat, pedestal, out=master_flat)
        np.maximum(master_flat, 0.0, out=master_flat)

    for ch in range(master_flat.shape[2]):
        channel_data = master_flat[:, :, ch]
        norm_val = float(np.mean(channel_data))
        if norm_val > 1e-5:
            channel_data /= norm_val
        else:
            channel_data.fill(1.0)

    gc.collect()
    return master_flat


def calibrate_light(
    light_img: np.ndarray, 
    master_dark: np.ndarray = None, 
    master_flat: np.ndarray = None,
    master_bias: np.ndarray = None
) -> np.ndarray:
    """Aplica calibración con operaciones in-place."""
    if master_dark is not None:
        np.subtract(light_img, master_dark, out=light_img)
        np.maximum(light_img, 0.0, out=light_img)
    elif master_bias is not None:
        np.subtract(light_img, master_bias, out=light_img)
        np.maximum(light_img, 0.0, out=light_img)

    if master_flat is not None:
        safe_flat = np.where(master_flat > 1e-4, master_flat, 1.0)
        np.divide(light_img, safe_flat, out=light_img)
        np.clip(light_img, 0.0, 1.0, out=light_img)
        del safe_flat

    return light_img


def estimate_frame_background_dome(img_rgb: np.ndarray, sky_mask: np.ndarray = None) -> np.ndarray:
    h, w, c = img_rgb.shape
    scale = max(1, min(h, w) // 120)
    sh, sw = max(16, h // scale), max(16, w // scale)

    small_img = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)

    if sky_mask is not None:
        sm_mask = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (sw, sh), interpolation=cv2.INTER_NEAREST)
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        valid_sky = cv2.erode((sm_mask > 0.5).astype(np.uint8), k_erode) > 0
    else:
        valid_sky = np.ones((sh, sw), dtype=bool)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bg_low = np.zeros_like(small_img)

    for ch in range(c):
        channel = small_img[..., ch]
        opened = cv2.morphologyEx(channel, cv2.MORPH_OPEN, k_open)
        sigma = max(5.0, min(sh, sw) / 10.0)
        bg_low[..., ch] = cv2.GaussianBlur(opened, (0, 0), sigmaX=sigma, sigmaY=sigma)

    bg_full = cv2.resize(bg_low, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg_full


def preprocess_subframe_lp(
    frame_rgb: np.ndarray,
    method: str = "standard",
    strength: float = 0.5,
    sky_mask: np.ndarray = None,
    ref_stats: dict = None
) -> np.ndarray:
    """
    Aplica sustracción de gradiente o normalización local in-place canal por canal.
    Pico de asignación en memoria < 95 MiB (cero arrays 3D duplicados).
    """
    if strength <= 1e-4 or method in ["standard", "min_rejection"]:
        return frame_rgb

    h, w, c = frame_rgb.shape

    if method == "sequator_subtraction":
        # 1. Estimar fondo a baja resolución (miniatura < 1 MB)
        scale = max(1, min(h, w) // 120)
        sh, sw = max(16, h // scale), max(16, w // scale)
        small_img = cv2.resize(frame_rgb, (sw, sh), interpolation=cv2.INTER_AREA)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bg_low = np.zeros_like(small_img)
        for ch in range(c):
            opened = cv2.morphologyEx(small_img[..., ch], cv2.MORPH_OPEN, k_open)
            sigma = max(5.0, min(sh, sw) / 10.0)
            bg_low[..., ch] = cv2.GaussianBlur(opened, (0, 0), sigmaX=sigma, sigmaY=sigma)
        del small_img

        # 2. Pedestal por submuestreo rápido (4.5 MB en lugar de ordenar 280 MB)
        sub_sample = frame_rgb[::8, ::8]
        pedestal = np.percentile(sub_sample, 5, axis=(0, 1))
        del sub_sample

        # 3. Máscara 2D de cielo
        m2d = None
        if sky_mask is not None:
            if sky_mask.shape[:2] != (h, w):
                m2d = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                m2d = np.squeeze(sky_mask).astype(np.float32)

        # 4. Sustracción in-place canal a canal (Máximo 93 MB asignados a la vez)
        for ch in range(c):
            bg_ch = cv2.resize(bg_low[..., ch], (w, h), interpolation=cv2.INTER_LINEAR)
            ped_val = float(pedestal[ch])

            # delta = (bg - ped) * strength
            delta = (bg_ch - ped_val) * strength
            del bg_ch

            if m2d is not None:
                delta *= m2d

            frame_rgb[..., ch] -= delta
            del delta
            np.clip(frame_rgb[..., ch], 0.0, 1.0, out=frame_rgb[..., ch])

        if m2d is not None:
            del m2d
        del bg_low
        return frame_rgb

    elif method == "local_norm" and ref_stats is not None:
        sub_sample = frame_rgb[::8, ::8]
        curr_median = np.median(sub_sample, axis=(0, 1))
        del sub_sample

        target_median = ref_stats.get('median', curr_median)
        diff = (target_median - curr_median) * strength

        m2d = None
        if sky_mask is not None:
            if sky_mask.shape[:2] != (h, w):
                m2d = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                m2d = np.squeeze(sky_mask).astype(np.float32)

        for ch in range(c):
            d_val = float(diff[ch])
            if abs(d_val) > 1e-6:
                if m2d is not None:
                    frame_rgb[..., ch] += (d_val * m2d)
                else:
                    frame_rgb[..., ch] += d_val
                np.clip(frame_rgb[..., ch], 0.0, 1.0, out=frame_rgb[..., ch])

        if m2d is not None:
            del m2d
        return frame_rgb

    return frame_rgb


def detect_sky_stars(image_rgb: np.ndarray, sky_mask: np.ndarray = None, max_stars: int = 3500):
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if sky_mask is not None:
        mask_u8 = ((sky_mask > 0.4) * 255).astype(np.uint8)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        mask_clean = cv2.erode(mask_u8, kernel_erode)
    else:
        mask_clean = np.full((h, w), 255, dtype=np.uint8)

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_clean)
    stretched = np.clip(np.power(masked_gray, 0.6) * 255.0, 0, 255).astype(np.uint8)

    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stars_isolated = cv2.morphologyEx(stretched, cv2.MORPH_TOPHAT, kernel_tophat)
    _, stars_thresholded = cv2.threshold(stars_isolated, 15, 255, cv2.THRESH_TOZERO)

    detector = cv2.ORB_create(
        nfeatures=max_stars,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=12,
        fastThreshold=6
    )
    kps, descs = detector.detectAndCompute(stars_thresholded, mask=mask_clean)
    return kps, descs, cv2.NORM_HAMMING


def refine_star_centroids(gray_img: np.ndarray, keypoints: list) -> np.ndarray:
    if not keypoints:
        return np.empty((0, 1, 2), dtype=np.float32)

    pts = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.005)
    refined_pts = cv2.cornerSubPix(gray_img, pts, (5, 5), (-1, -1), criteria)
    return refined_pts


def save_frame_float32(filepath: str, img_float32: np.ndarray):
    raw_data = np.ascontiguousarray(img_float32, dtype=np.float32)
    with open(filepath, "wb") as f:
        f.write(raw_data.tobytes())
    del raw_data


class StorageModeDecision(tuple):
    """
    Decisión de estrategia de almacenamiento intermedio.
    Compatible tanto con desempaquetado de tupla (use_ram, description),
    acceso a propiedades (.use_ram, .description), como evaluación booleana (if decision:).
    """
    def __new__(cls, use_ram: bool, description: str):
        return super().__new__(cls, (use_ram, description))

    @property
    def use_ram(self) -> bool:
        return self[0]

    @property
    def description(self) -> str:
        return self[1]

    def __bool__(self) -> bool:
        return self[0]


def evaluate_storage_mode(strategy: str, num_frames: int, h: int, w: int, c: int, has_ground: bool) -> StorageModeDecision:
    """
    Determina de forma dinámica y segura si usar memoria RAM (True) o Caché en Disco .bin (False).
    Calcula la huella de memoria en base a las dimensiones, número de fotogramas y modo de suelo.
    """
    bytes_per_frame = h * w * c * 4  # float32 = 4 bytes por elemento
    multiplier = 2 if has_ground else 1
    estimated_total_bytes = num_frames * bytes_per_frame * multiplier
    est_gb = estimated_total_bytes / (1024 ** 3)

    try:
        import psutil
        free_bytes = psutil.virtual_memory().available
        free_gb = free_bytes / (1024 ** 3)
    except Exception:
        free_bytes = None
        free_gb = 0.0

    if strategy == "disk":
        return StorageModeDecision(False, f"Caché en Disco SSD ({est_gb:.1f} GB estimados, forzado en Ajustes)")

    if strategy == "ram":
        if free_bytes is not None and estimated_total_bytes > (free_bytes * 0.85):
            return StorageModeDecision(False, f"Caché en Disco SSD (RAM insuficiente: se requieren {est_gb:.1f} GB y solo hay {free_gb:.1f} GB libres)")
        ram_txt = f"{free_gb:.1f} GB libres" if free_bytes else "RAM disponible"
        return StorageModeDecision(True, f"Memoria RAM Ultra-rápida ({est_gb:.1f} GB requeridos / {ram_txt})")

    # Modo "auto" (por defecto)
    if free_bytes is not None:
        if estimated_total_bytes < (free_bytes * 0.70):
            return StorageModeDecision(True, f"Memoria RAM Automática ({est_gb:.1f} GB requeridos < 70% de {free_gb:.1f} GB libres)")
        else:
            return StorageModeDecision(False, f"Caché en Disco SSD Automático ({est_gb:.1f} GB requeridos superan 70% de {free_gb:.1f} GB libres)")

    return StorageModeDecision(False, "Caché en Disco SSD (psutil no disponible para medir RAM libre)")


def align_single_light_task(args: tuple) -> dict:
    (
        idx, path, total_lights, temp_dir, mode,
        p_md, p_mf, p_mb, p_mask, p_ref_gray,
        ref_kps_pts, ref_desc, norm_type,
        w, h, lp_method, lp_strength, ref_stats
    ) = args[:18]
    use_ram_buffer = bool(args[18]) if len(args) > 18 else False

    filename = os.path.basename(path)
    res = {
        "idx": idx,
        "filename": filename,
        "success": False,
        "error": None,
        "sky_data": None,
        "gnd_data": None,
        "inliers": 0,
        "dx": 0.0,
        "dy": 0.0
    }

    try:
        # Carga instantánea por mapeo de memoria sin ocupar memoria en heap
        master_dark = np.memmap(p_md, dtype=np.float32, mode='r', shape=(h, w, 3)) if (p_md and os.path.exists(p_md)) else None
        master_bias = np.memmap(p_mb, dtype=np.float32, mode='r', shape=(h, w, 3)) if (p_mb and os.path.exists(p_mb) and master_dark is None) else None
        master_flat = np.memmap(p_mf, dtype=np.float32, mode='r', shape=(h, w, 3)) if (p_mf and os.path.exists(p_mf)) else None
        sky_mask = np.memmap(p_mask, dtype=np.float32, mode='r', shape=(h, w)) if (p_mask and os.path.exists(p_mask)) else None
        ref_gray = np.asarray(np.memmap(p_ref_gray, dtype=np.uint8, mode='r', shape=(h, w))) if (p_ref_gray and os.path.exists(p_ref_gray)) else None

        raw_frame = load_image_as_float32(path)
        calibrated_frame = calibrate_light(
            raw_frame, 
            master_dark=master_dark, 
            master_flat=master_flat, 
            master_bias=master_bias
        )
        del master_dark, master_flat, master_bias

        curr_kp, curr_desc, _ = detect_sky_stars(calibrated_frame, sky_mask=sky_mask)
        if curr_desc is None or len(curr_kp) < 15:
            raise RuntimeError(f"Solo se detectaron {len(curr_kp) if curr_kp else 0} estrellas.")

        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = bf.knnMatch(ref_desc, curr_desc, k=2)
        good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.78 * n.distance]

        if len(good) < 10:
            raise RuntimeError(f"Correspondencias insuficientes con la referencia ({len(good)} pares).")

        # Conversión directa y ligera a escala de grises (24 MB en lugar de 300 MB)
        curr_gray_f32 = cv2.cvtColor(calibrated_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = np.clip(curr_gray_f32 * 255.0, 0, 255).astype(np.uint8)
        del curr_gray_f32

        ref_matched_kps = [cv2.KeyPoint(ref_kps_pts[m.queryIdx][0], ref_kps_pts[m.queryIdx][1], 1.0) for m in good]
        curr_matched_kps = [curr_kp[m.trainIdx] for m in good]

        dst_pts = refine_star_centroids(ref_gray, ref_matched_kps)
        src_pts = refine_star_centroids(curr_gray, curr_matched_kps)
        del curr_gray, ref_gray, curr_kp, curr_desc, matches, good

        H_matrix, inliers = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=5000,
            confidence=0.995
        )

        if H_matrix is None:
            H_aff, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            if H_aff is None:
                raise RuntimeError("Fallo RANSAC en homografía.")
            H_matrix = np.vstack([H_aff, [0.0, 0.0, 1.0]])

        num_inliers = int(np.sum(inliers)) if inliers is not None else 0
        del dst_pts, src_pts, inliers

        if sky_mask is not None:
            sm_3d = sky_mask[..., np.newaxis]
            sky_median_val = np.median(calibrated_frame[sky_mask > 0.5]) if np.any(sky_mask > 0.5) else 0.05
            clean_sky_frame = (calibrated_frame * sm_3d) + (sky_median_val * (1.0 - sm_3d))
        else:
            clean_sky_frame = calibrated_frame

        warped = cv2.warpPerspective(
            clean_sky_frame, H_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        del clean_sky_frame, calibrated_frame

        warped_processed = preprocess_subframe_lp(
            warped,
            method=lp_method,
            strength=lp_strength,
            sky_mask=sky_mask,
            ref_stats=ref_stats
        )
        del warped, sky_mask

        if use_ram_buffer:
            res["sky_data"] = np.ascontiguousarray(warped_processed, dtype=np.float32)
            if mode == "fixed_tripod":
                res["gnd_data"] = np.ascontiguousarray(raw_frame, dtype=np.float32)
            else:
                del raw_frame
        else:
            p_sky = os.path.join(temp_dir, f"sky_{idx-1:04d}.bin")
            save_frame_float32(p_sky, warped_processed)
            res["sky_data"] = p_sky

            if mode == "fixed_tripod":
                p_gnd = os.path.join(temp_dir, f"gnd_{idx-1:04d}.bin")
                save_frame_float32(p_gnd, raw_frame)
                res["gnd_data"] = p_gnd

            del raw_frame

        del warped_processed

        res["success"] = True
        res["inliers"] = num_inliers
        res["dx"] = float(H_matrix[0, 2])
        res["dy"] = float(H_matrix[1, 2])

    except Exception as exc:
        res["error"] = str(exc)

    return res


# =========================================================================
# MOTOR DE INTEGRACIÓN ROBUSTO (KAPPA-SIGMA ESTRICTO)
# =========================================================================

def _process_single_chunk_cpu(args):
    source_arrays, y_start, y_end, w, c, kappa, lp_method, lp_strength = args
    actual_rows = y_end - y_start
    n_frames = len(source_arrays)

    # Slicing directo de memoria (idéntico para matrices en RAM o archivos np.memmap en disco)
    sub_stack = np.empty((n_frames, actual_rows, w, c), dtype=np.float32)
    for i, arr in enumerate(source_arrays):
        sub_stack[i] = arr[y_start:y_end]

    chunk_result = np.empty((actual_rows, w, c), dtype=np.float32)

    if lp_method == "min_rejection" and lp_strength > 1e-4:
        target_p = max(5.0, 50.0 - (lp_strength * 40.0))
        if HAS_NUMBA:
            flat_data = sub_stack.reshape(n_frames, -1)
            chunk_result = _percentile_kernel_numba(flat_data, target_p).reshape((actual_rows, w, c))
        else:
            for ch in range(c):
                channel_data = sub_stack[:, :, :, ch].reshape((n_frames, -1))
                res = np.percentile(channel_data, target_p, axis=0)
                chunk_result[:, :, ch] = res.reshape((actual_rows, w))
        del sub_stack
        return y_start, y_end, chunk_result

    # Rechazo estricto en CPU
    if HAS_NUMBA:
        flat_data = sub_stack.reshape(n_frames, -1)
        chunk_result = _kappa_sigma_kernel_numba(flat_data, kappa).reshape((actual_rows, w, c))
    else:
        for ch in range(c):
            channel_data = sub_stack[:, :, :, ch].reshape((n_frames, -1))
            
            med = np.median(channel_data, axis=0)
            abs_diff = np.abs(channel_data - med)
            mad = np.median(abs_diff, axis=0)
            sigma = 1.4826 * mad + 1e-6

            low = med - kappa * sigma
            high = med + kappa * sigma

            valid = (channel_data >= low) & (channel_data <= high)
            
            counts = np.sum(valid, axis=0)
            sums = np.sum(np.where(valid, channel_data, 0.0), axis=0)
            
            fallback = counts == 0
            counts[fallback] = 1
            res = sums / counts
            res[fallback] = med[fallback]

            chunk_result[:, :, ch] = res.reshape((actual_rows, w))

    del sub_stack
    return y_start, y_end, chunk_result


def stream_stack_auto(
    frames_source: list, 
    shape: tuple, 
    chunk_rows: int = 200, 
    kappa: float = 2.2, 
    max_workers: int = None,
    lp_method: str = "standard",
    lp_strength: float = 0.5
) -> np.ndarray:
    """
    Motor canónico de integración matemática mediante franjas (chunks) 
    y paralelización multi-hilo en CPU con acceso unificado a memoria.
    """
    h, w, c = shape
    stacked_out = np.zeros((h, w, c), dtype=np.float32)

    is_file_mode = len(frames_source) > 0 and isinstance(frames_source[0], str)
    mms = []

    try:
        if is_file_mode:
            # Mapeo de memoria por archivo una sola vez: cero syscalls de apertura/cierre en las franjas
            mms = [np.memmap(f, dtype=np.float32, mode="r", shape=(h, w, c)) for f in frames_source]
            source_arrays = mms
        else:
            source_arrays = frames_source

        tasks = []
        for y in range(0, h, chunk_rows):
            y_end = min(y + chunk_rows, h)
            tasks.append((source_arrays, y, y_end, w, c, kappa, lp_method, lp_strength))

        if max_workers is None:
            max_workers = max(1, min(4, (os.cpu_count() or 4) - 1))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for y_start, y_end, chunk_data in executor.map(_process_single_chunk_cpu, tasks):
                stacked_out[y_start:y_end, :, :] = chunk_data

    finally:
        # Cierre explícito de descriptores mapeados para evitar bloqueos WinError 32 en Windows
        for mm in mms:
            if hasattr(mm, "_mmap") and mm._mmap is not None:
                try:
                    mm._mmap.close()
                except Exception:
                    pass
        del mms

    gc.collect()
    return stacked_out