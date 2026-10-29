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
    def _kappa_sigma_kernel_numba(data, kappa, weights):
        """
        Kernel compilado en LLVM para integración Kappa-Sigma con MAD en una sola pasada.
        Excluye valores inválidos (< 0.0) de suelo desplazado o bordes geométricos (Sequator freeze ground).
        data: shape (n_frames, total_items)
        """
        n_frames, total_items = data.shape
        out = np.empty(total_items, dtype=np.float32)

        for p in prange(total_items):
            cnt_val = 0
            for i in range(n_frames):
                if data[i, p] >= 0.0:
                    cnt_val += 1

            if cnt_val == 0:
                out[p] = 0.0
                continue

            v = np.empty(cnt_val, dtype=np.float32)
            idx = 0
            for i in range(n_frames):
                val = data[i, p]
                if val >= 0.0:
                    v[idx] = val
                    idx += 1

            # 2. Mediana local
            _sort_inplace(v)
            if cnt_val % 2 == 1:
                med = v[cnt_val // 2]
            else:
                med = 0.5 * (v[cnt_val // 2 - 1] + v[cnt_val // 2])

            # 3. Desviaciones absolutas respecto a la mediana
            d = np.empty(cnt_val, dtype=np.float32)
            for i in range(cnt_val):
                d[i] = abs(v[i] - med)

            # 4. MAD (Median Absolute Deviation)
            _sort_inplace(d)
            if cnt_val % 2 == 1:
                mad = d[cnt_val // 2]
            else:
                mad = 0.5 * (d[cnt_val // 2 - 1] + d[cnt_val // 2])

            sigma = 1.4826 * mad + 1e-6
            low = med - kappa * sigma
            high = med + kappa * sigma

            # 5. Promedio ponderado de valores válidos
            s = 0.0
            w_sum = 0.0
            for i in range(n_frames):
                val = data[i, p]
                if val >= 0.0 and low <= val <= high:
                    w = weights[i]
                    s += val * w
                    w_sum += w

            if w_sum > 0.0:
                out[p] = s / w_sum
            else:
                out[p] = med

        return out

    @njit(parallel=True, fastmath=True, nogil=True)
    def _percentile_kernel_numba(data, target_p):
        """Kernel compilado para cálculo exacto de percentil con exclusión de píxeles inválidos (< 0.0)."""
        n_frames, total_items = data.shape
        out = np.empty(total_items, dtype=np.float32)

        for p in prange(total_items):
            cnt_val = 0
            for i in range(n_frames):
                if data[i, p] >= 0.0:
                    cnt_val += 1

            if cnt_val == 0:
                out[p] = 0.0
                continue

            v = np.empty(cnt_val, dtype=np.float32)
            idx = 0
            for i in range(n_frames):
                val = data[i, p]
                if val >= 0.0:
                    v[idx] = val
                    idx += 1

            _sort_inplace(v)
            rank = (target_p / 100.0) * (cnt_val - 1)
            i0 = int(rank)
            i1 = min(i0 + 1, cnt_val - 1)
            frac = float(rank - i0)
            out[p] = v[i0] + frac * (v[i1] - v[i0])

        return out

    @njit(parallel=True, fastmath=True, nogil=True)
    def _median_kernel_numba(data):
        """Kernel compilado para cálculo rápido de mediana con exclusión de píxeles inválidos (< 0.0)."""
        n_frames, total_items = data.shape
        out = np.empty(total_items, dtype=np.float32)

        for p in prange(total_items):
            cnt_val = 0
            for i in range(n_frames):
                if data[i, p] >= 0.0:
                    cnt_val += 1

            if cnt_val == 0:
                out[p] = 0.0
                continue

            v = np.empty(cnt_val, dtype=np.float32)
            idx = 0
            for i in range(n_frames):
                val = data[i, p]
                if val >= 0.0:
                    v[idx] = val
                    idx += 1

            _sort_inplace(v)
            if cnt_val % 2 == 1:
                out[p] = v[cnt_val // 2]
            else:
                out[p] = 0.5 * (v[cnt_val // 2 - 1] + v[cnt_val // 2])

        return out

    # Calentamiento JIT anticipado con arrays mínimos de prueba
    try:
        _dummy = np.zeros((2, 2), dtype=np.float32)
        _kappa_sigma_kernel_numba(_dummy, 2.2, np.ones(2, dtype=np.float32))
        _percentile_kernel_numba(_dummy, 50.0)
        _median_kernel_numba(_dummy)
    except Exception:
        pass


def load_image_as_float32(filepath: str, color_space: str = None) -> np.ndarray:
    """Carga imágenes optimizando memoria con operaciones in-place y espacio de color configurado."""
    if color_space is None:
        from core.config_manager import get_config_val
        color_space = get_config_val("icc_profile", "srgb")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw']:
        from core.icc_manager import get_rawpy_colorspace
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
            res = np.clip(data, 0.0, 1.0)
        else:
            max_val = 65535.0 if d_max > 255.0 else 255.0
            data /= max_val
            res = np.clip(data, 0.0, 1.0)

        # Si se solicita un espacio de color destino y el TIFF tiene perfil ICC incrustado, convertir
        if color_space is not None:
            from core.icc_manager import (
                extract_icc_profile,
                identify_colorspace_from_icc,
                convert_colorspace,
                normalize_colorspace_name
            )
            embedded_icc = extract_icc_profile(filepath)
            if embedded_icc:
                src_cs = identify_colorspace_from_icc(embedded_icc)
                target_cs = normalize_colorspace_name(color_space)
                if src_cs != target_cs:
                    res = convert_colorspace(res, src_space=src_cs, dst_space=target_cs)
        return res

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

        sm_mask = None
        if sky_mask is not None:
            sm_mask = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (sw, sh), interpolation=cv2.INTER_NEAREST)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bg_low = np.zeros_like(small_img)
        for ch in range(c):
            ch_data = small_img[..., ch].copy()
            # Si hay suelo, rellenar el suelo con la mediana del cielo para que no arrastre a la baja el fondo del cielo cerca del horizonte
            if sm_mask is not None and np.any(sm_mask <= 0.5) and np.any(sm_mask > 0.5):
                sky_val = float(np.median(ch_data[sm_mask > 0.5]))
                ch_data[sm_mask <= 0.5] = sky_val
            opened = cv2.morphologyEx(ch_data, cv2.MORPH_OPEN, k_open)
            del ch_data
            sigma = max(5.0, min(sh, sw) / 10.0)
            bg_low[..., ch] = cv2.GaussianBlur(opened, (0, 0), sigmaX=sigma, sigmaY=sigma)
        del small_img, sm_mask

        # 2. Pedestal por submuestreo rápido (4.5 MB en lugar de ordenar 280 MB)
        sub_sample = frame_rgb[::8, ::8]
        if sky_mask is not None:
            sm_sub = (sky_mask[::8, ::8] > 0.5)
            if np.any(sm_sub):
                pedestal = np.percentile(sub_sample[sm_sub], 5, axis=0)
            else:
                pedestal = np.percentile(sub_sample, 5, axis=(0, 1))
        else:
            pedestal = np.percentile(sub_sample, 5, axis=(0, 1))
        del sub_sample

        # 3. Máscara de cielo suavizada: el difuminado sólo atenúa en el cielo, NUNCA invade el suelo
        m2d_smooth = None
        if sky_mask is not None:
            if sky_mask.shape[:2] != (h, w):
                m2d_raw = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                m2d_raw = np.squeeze(sky_mask).astype(np.float32)
            blurred = cv2.GaussianBlur(m2d_raw, (31, 31), sigmaX=8.0)
            m2d_smooth = np.minimum(m2d_raw, blurred)
            del m2d_raw, blurred

        # 4. Sustracción in-place canal a canal
        for ch in range(c):
            bg_ch = cv2.resize(bg_low[..., ch], (w, h), interpolation=cv2.INTER_LINEAR)
            ped_val = float(pedestal[ch])

            # delta = (bg - ped) * strength
            delta = (bg_ch - ped_val) * strength
            del bg_ch

            if m2d_smooth is not None:
                delta *= m2d_smooth

            frame_rgb[..., ch] -= delta
            del delta
            np.clip(frame_rgb[..., ch], 0.0, 1.0, out=frame_rgb[..., ch])

        if m2d_smooth is not None:
            del m2d_smooth
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


def compute_sky_background_stats(frame: np.ndarray, sky_mask: np.ndarray = None) -> dict:
    """
    Calcula la mediana y la dispersión (MAD) robusta del cielo por canal (RGB).
    Utiliza submuestreo de paso 8 y filtra por la máscara de cielo si está presente.
    """
    sub = frame[::8, ::8]
    if sky_mask is not None:
        sm_sub = sky_mask[::8, ::8] > 0.5
        if np.count_nonzero(sm_sub) >= 100:
            pts = sub[sm_sub]
        else:
            pts = sub.reshape(-1, sub.shape[-1])
    else:
        pts = sub.reshape(-1, sub.shape[-1])

    med = np.median(pts, axis=0).astype(np.float32)
    mad = np.median(np.abs(pts - med), axis=0).astype(np.float32)
    return {"median": med, "mad": mad}


def apply_linear_fit_normalization(
    frame: np.ndarray,
    ref_stats: dict,
    sky_mask: np.ndarray = None,
    in_place: bool = True
) -> tuple:
    """
    Normalización Fotométrica Lineal (Linear Fit / Scale & Zero-Offset):
    I'_k = a_k * I_k + b_k
    donde a = MAD_ref / MAD_k y b = MED_ref - a * MED_k.
    Iguala el pedestal de fondo y la dispersión del cielo respecto a la referencia antes de Kappa-Sigma.
    """
    curr_stats = compute_sky_background_stats(frame, sky_mask)
    ref_med = ref_stats["median"]
    ref_mad = ref_stats["mad"]
    curr_med = curr_stats["median"]
    curr_mad = curr_stats["mad"]

    # Factor de escala acotado para estabilidad numérica [0.2, 5.0]
    scale = np.clip(ref_mad / np.maximum(curr_mad, 1e-6), 0.2, 5.0).astype(np.float32)
    offset = (ref_med - scale * curr_med).astype(np.float32)

    out = frame if in_place else frame.copy()
    for ch in range(frame.shape[-1]):
        a = float(scale[ch])
        b = float(offset[ch])
        if abs(a - 1.0) > 1e-4 or abs(b) > 1e-5:
            ch_view = out[..., ch]
            np.multiply(ch_view, a, out=ch_view)
            np.add(ch_view, b, out=ch_view)
            np.clip(ch_view, 0.0, 1.0, out=ch_view)

    return out, {"scale": scale, "offset": offset}


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
        "dy": 0.0,
        "fit_scale": None,
        "fit_offset": None
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
        del master_dark, master_flat, master_bias, raw_frame

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

        # 1. Preservar toma de suelo estática para trípode fijo 100% limpia de sustracciones
        if mode == "fixed_tripod":
            if use_ram_buffer:
                res["gnd_data"] = calibrated_frame.copy()
            else:
                p_gnd = os.path.join(temp_dir, f"gnd_{idx-1:04d}.bin")
                save_frame_float32(p_gnd, calibrated_frame)
                res["gnd_data"] = p_gnd

        # 2. Preprocesar contaminación lumínica en el fotograma para el CIELO
        calibrated_processed = preprocess_subframe_lp(
            calibrated_frame,
            method=lp_method,
            strength=lp_strength,
            sky_mask=sky_mask,
            ref_stats=ref_stats
        )

        # 2b. Normalización Fotométrica Lineal (Linear Fit) respecto a la referencia
        if ref_stats is not None and ref_stats.get("linear_fit", True) and "mad" in ref_stats:
            calibrated_processed, fit_info = apply_linear_fit_normalization(
                calibrated_processed,
                ref_stats,
                sky_mask=sky_mask,
                in_place=True
            )
            res["fit_scale"] = fit_info["scale"]
            res["fit_offset"] = fit_info["offset"]

        # 3. Warp óptico exacto del frame de cielo procesado
        warped = cv2.warpPerspective(
            calibrated_processed, H_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0
        )
        del calibrated_processed, calibrated_frame

        # 4. Excluir el suelo desplazado para que NUNCA contamine el cielo apilado (método Sequator)
        if sky_mask is not None:
            mask_bin = (sky_mask > 0.5).astype(np.uint8)
            kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            safe_sky = cv2.erode(mask_bin, kernel_safe)
            warped_mask = cv2.warpPerspective(
                safe_sky, H_matrix, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            warped[warped_mask == 0] = -1.0
            del warped_mask, safe_sky, mask_bin
        del sky_mask

        if use_ram_buffer:
            res["sky_data"] = np.ascontiguousarray(warped, dtype=np.float32)
        else:
            p_sky = os.path.join(temp_dir, f"sky_{idx-1:04d}.bin")
            save_frame_float32(p_sky, warped)
            res["sky_data"] = p_sky

        del warped

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
    source_arrays, y_start, y_end, w, c, kappa, lp_method, lp_strength = args[:8]
    weights = args[8] if len(args) > 8 else None
    actual_rows = y_end - y_start
    n_frames = len(source_arrays)
    if weights is None:
        weights = np.ones(n_frames, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)

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
        chunk_result = _kappa_sigma_kernel_numba(flat_data, kappa, weights).reshape((actual_rows, w, c))
    else:
        for ch in range(c):
            channel_data = sub_stack[:, :, :, ch].reshape((n_frames, -1))
            valid_mask = channel_data >= 0.0
            ch_masked = np.where(valid_mask, channel_data, np.nan)
            
            med = np.nanmedian(ch_masked, axis=0)
            abs_diff = np.abs(ch_masked - med)
            mad = np.nanmedian(abs_diff, axis=0)
            sigma = 1.4826 * mad + 1e-6

            low = med - kappa * sigma
            high = med + kappa * sigma

            in_range = valid_mask & (channel_data >= low) & (channel_data <= high)
            w_mat = weights.reshape((n_frames, 1))
            weighted_vals = np.where(in_range, channel_data * w_mat, 0.0)
            sums = np.sum(weighted_vals, axis=0)
            w_sums = np.sum(np.where(in_range, w_mat, 0.0), axis=0)


            fallback = w_sums == 0
            w_sums[fallback] = 1.0
            res = sums / w_sums
            
            med_fallback = np.nan_to_num(med, nan=0.0)
            res[fallback] = med_fallback[fallback]

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
    lp_strength: float = 0.5,
    weights: np.ndarray = None
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

        weights_arr = np.ascontiguousarray(weights, dtype=np.float32) if weights is not None else np.ones(len(source_arrays), dtype=np.float32)
        tasks = []
        for y in range(0, h, chunk_rows):
            y_end = min(y + chunk_rows, h)
            tasks.append((source_arrays, y, y_end, w, c, kappa, lp_method, lp_strength, weights_arr))

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


def compose_sky_ground_with_anti_ghosting(
    sky_stacked: np.ndarray,
    ground_layer: np.ndarray,
    sky_mask: np.ndarray,
    enable_anti_ghosting: bool = True,
    boundary_radius: int = 25,
    tree_sensitivity: float = 0.5
) -> np.ndarray:
    """
    Composición avanzada cielo/suelo con transición óptica y protección de follaje fino.
    Erradica ramas fantasma manteniendo la silueta del suelo nítida y sin halos oscuros.
    
    :param sky_stacked: Array float32 (H, W, 3) con el cielo integrado y alineado a estrellas.
    :param ground_layer: Array float32 (H, W, 3) con el suelo estático integrado o de referencia.
    :param sky_mask: Array float32 (H, W) donde 1.0 = cielo y 0.0 = suelo.
    :param enable_anti_ghosting: Si es False, aplica fusión por difuminado gausiano convencional.
    :param boundary_radius: Radio en píxeles de la franja de seguridad en el cielo (ej. 10 a 25 px).
    :param tree_sensitivity: Factor de modulación de silueta vegetal [0.1 a 1.0].
    """
    h, w = sky_stacked.shape[:2]
    raw_mask = np.clip(sky_mask.astype(np.float32), 0.0, 1.0)
    if raw_mask.shape[:2] != (h, w):
        raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    if not enable_anti_ghosting:
        smooth_mask = cv2.GaussianBlur(raw_mask, (5, 5), sigmaX=0.8)
        mask_3d = smooth_mask[..., np.newaxis]
        return (sky_stacked * mask_3d) + (ground_layer * (1.0 - mask_3d))

    bin_mask = (raw_mask > 0.5).astype(np.uint8)
    dist_to_gnd = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    dist_to_sky = cv2.distanceTransform(1 - bin_mask, cv2.DIST_L2, 5)
    sdf = dist_to_gnd - dist_to_sky  # >0 en cielo, <0 en suelo

    # 1. Transición óptica natural (radio de 2.5 px, ancho total 5 px sin translúcidos anchos)
    optical_feather = 2.5
    w_linear = np.clip((sdf / (2.0 * optical_feather)) + 0.5, 0.0, 1.0)
    w_smooth = 0.5 - 0.5 * np.cos(np.pi * w_linear)

    # 2. Protección de follaje fino en la franja de seguridad del cielo
    rad = max(4, int(boundary_radius))
    in_safety_band = (sdf > 0) & (sdf <= rad)

    if np.any(in_safety_band) and tree_sensitivity > 0.05:
        lum_sky = cv2.cvtColor(sky_stacked, cv2.COLOR_RGB2GRAY) if sky_stacked.ndim == 3 else sky_stacked
        lum_gnd = cv2.cvtColor(ground_layer, cv2.COLOR_RGB2GRAY) if ground_layer.ndim == 3 else ground_layer

        sky_ref_val = float(np.median(lum_sky[raw_mask > 0.6])) if np.any(raw_mask > 0.6) else 0.15
        gnd_ref_val = float(np.median(lum_gnd[raw_mask < 0.4])) if np.any(raw_mask < 0.4) else 0.02

        # Una rama fina en ground_layer es fotométricamente similar al suelo y mucho más oscura que el cielo
        sens = float(np.clip(tree_sensitivity, 0.1, 1.0))
        thresh = gnd_ref_val + (sky_ref_val - gnd_ref_val) * (0.20 + 0.30 * sens)
        dark_branch_mask = in_safety_band & (lum_gnd < thresh) & (lum_gnd < (lum_sky - 0.005))

        if np.any(dark_branch_mask):
            branch_f32 = dark_branch_mask.astype(np.float32)
            branch_smooth = cv2.GaussianBlur(branch_f32, (5, 5), sigmaX=1.0)
            # Imponer prioridad absoluta y opaca a la rama inmóvil del suelo sobre el cielo en movimiento
            w_smooth = np.minimum(w_smooth, 1.0 - branch_smooth)

    mask_3d = np.clip(w_smooth, 0.0, 1.0)[..., np.newaxis]
    composite = (sky_stacked * mask_3d) + (ground_layer * (1.0 - mask_3d))
    return composite
