# core/ai_denoise.py
"""
core/ai_denoise.py - Reducción de Ruido Profunda con IA (NAFNet ONNX) en AstroLoom.

Implementa inferencia acelerada mediante red neuronal NAFNet (Nonlinear Activation Free Network)
optimizada para restauración de imagen y desacoplada en bloques (tiling) con ventana 2D de Hann
para garantizar continuidad fotométrica sin costuras.
"""

from typing import Optional, Callable, List, Tuple
import os
import sys
import logging
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import astroloom_rs
    HAS_RUST_AI = hasattr(astroloom_rs, 'accumulate_tile_hann_inplace_rs')
except ImportError:
    HAS_RUST_AI = False

from core.config_manager import get_config_val

logger = logging.getLogger("AstroLoom.AIDenoise")

DEFAULT_NAFNET_URL = "https://huggingface.co/tomfl/obscura-models/resolve/main/nafnet-sidd-w32.onnx"
DEFAULT_NAFNET_FILENAME = "nafnet_denoise.onnx"


def get_nafnet_model_path() -> Optional[str]:
    """
    Localiza el archivo del modelo ONNX de NAFNet por orden de prioridad:
    1. Configuración global (`ai_denoise_model_path`).
    2. Directorio `models/nafnet_denoise.onnx` en la raíz del proyecto.
    3. Variantes en `models/` (ej. `nafnet-sidd-w32.onnx`).
    4. Directorio `assets/models/`.
    """
    custom = get_config_val("ai_denoise_model_path", "")
    if custom and os.path.isfile(custom):
        return os.path.abspath(custom)

    root = (
        os.path.dirname(sys.executable)
        if getattr(sys, "frozen", False)
        else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    candidates = [
        os.path.join(root, "models", DEFAULT_NAFNET_FILENAME),
        os.path.join(root, "models", "nafnet-sidd-w32.onnx"),
        os.path.join(root, "models", "NAFNet-SIDD-width32.onnx"),
        os.path.join(root, "models", "NAFNet-SIDD-width64.onnx"),
        os.path.join(root, "models", "nafnet-sidd-w64.onnx"),
        os.path.join(root, "assets", "models", DEFAULT_NAFNET_FILENAME),
        os.path.join(root, "assets", "models", "nafnet-sidd-w32.onnx"),
        os.path.join(root, "assets", "models", "NAFNet-SIDD-width64.onnx"),
        os.path.join(root, "assets", "models", "nafnet-sidd-w64.onnx"),
        os.path.join(root, DEFAULT_NAFNET_FILENAME)
    ]
    if getattr(sys, "frozen", False):
        dev_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates.append(os.path.join(dev_root, "models", DEFAULT_NAFNET_FILENAME))
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def is_nafnet_available() -> bool:
    """Indica si onnxruntime y el archivo de pesos de NAFNet están disponibles."""
    if not HAS_ONNX:
        return False
    path = get_nafnet_model_path()
    return path is not None and os.path.isfile(path)


def download_nafnet_model(
    target_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Descarga el modelo NAFNet ONNX desde Hugging Face con reporte de progreso en bloques.
    Escribe en archivo temporal (.tmp) y renombra atómicamente al terminar.
    """
    import urllib.request

    if not target_path:
        root = (
            os.path.dirname(sys.executable)
            if getattr(sys, "frozen", False)
            else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )
        models_dir = os.path.join(root, "models")
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, DEFAULT_NAFNET_FILENAME)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    temp_path = target_path + ".tmp"
    url = DEFAULT_NAFNET_URL

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AstroLoom/1.0 NAFNetDownloader"}
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 111.6
        chunk_size = 256 * 1024  # 256 KB
        downloaded = 0

        with open(temp_path, "wb") as f:
            while True:
                if cancel_check and cancel_check():
                    f.close()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise InterruptedError("Descarga cancelada por el usuario.")

                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                downloaded_mb = downloaded / (1024 * 1024)
                pct = int((downloaded / total_size) * 100) if total_size > 0 else 0

                if progress_callback:
                    progress_callback(pct, downloaded_mb, total_mb)

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    logger.info(f"Modelo NAFNet descargado e instalado en: {target_path}")
    return target_path


def preprocess_chroma_clean(img_rgb: np.ndarray, color_strength: float = 0.5) -> np.ndarray:
    """
    Limpia el moteado cromático en sombras convirtiendo a YCrCb y filtrando suavemente Croma.
    """
    if color_strength <= 1e-4:
        return img_rgb

    safe = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)
    ycrcb = cv2.cvtColor(safe, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]

    ksize = int(max(3, round(3.0 + 6.0 * color_strength)))
    if ksize % 2 == 0:
        ksize += 1
    sigma_c = float(1.0 + 3.0 * color_strength)

    cr_clean = cv2.GaussianBlur(cr, (ksize, ksize), sigma_c)
    cb_clean = cv2.GaussianBlur(cb, (ksize, ksize), sigma_c)

    blend_c = float(np.clip(color_strength, 0.0, 1.0))
    ycrcb[..., 1] = cr * (1.0 - blend_c) + cr_clean * blend_c
    ycrcb[..., 2] = cb * (1.0 - blend_c) + cb_clean * blend_c

    out_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return np.clip(out_rgb, 0.0, 1.0).astype(np.float32)


def compute_hann_2d_window(height: int, width: int) -> np.ndarray:
    """
    Calcula una ventana de ponderación 2D de Hann separable:
    w(y, x) = sin²(pi * (y+0.5) / H) * sin²(pi * (x+0.5) / W)
    Garantiza continuidad analítica en el solapamiento sin divisiones por cero.
    """
    if HAS_RUST_AI and hasattr(astroloom_rs, 'compute_hann_2d_window_rs'):
        try:
            return astroloom_rs.compute_hann_2d_window_rs(height, width)
        except Exception:
            pass
    y_w = np.sin(np.pi * (np.arange(height, dtype=np.float32) + 0.5) / height) ** 2
    x_w = np.sin(np.pi * (np.arange(width, dtype=np.float32) + 0.5) / width) ** 2
    return np.outer(y_w, x_w).astype(np.float32)[..., np.newaxis]


def _get_tile_coords(length: int, tile_size: int, overlap: int) -> List[int]:
    """Genera las coordenadas de inicio de los bloques cubriendo la longitud completa."""
    stride = tile_size - overlap
    if HAS_RUST_AI and hasattr(astroloom_rs, 'get_tile_coords_rs'):
        try:
            return astroloom_rs.get_tile_coords_rs(length, tile_size, stride)
        except Exception:
            pass
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    coords = []
    pos = 0
    while pos + tile_size < length:
        coords.append(pos)
        pos += step
    coords.append(length - tile_size)
    return sorted(list(set(coords)))


_GPU_FALLBACK_LISTENERS: List[Callable[[str], None]] = []


def add_gpu_fallback_listener(cb: Callable[[str], None]) -> None:
    """Registra un callback que se invoca cuando la inferencia degrada de GPU a CPU."""
    if cb not in _GPU_FALLBACK_LISTENERS:
        _GPU_FALLBACK_LISTENERS.append(cb)


def notify_gpu_fallback(reason: str) -> None:
    """Notifica a todos los escuchas registrados de una degradación a CPU."""
    for cb in _GPU_FALLBACK_LISTENERS:
        try:
            cb(reason)
        except Exception:
            pass


def _get_cache_dir(model_path: str) -> str:
    root = os.path.dirname(os.path.abspath(model_path))
    cache_dir = os.path.join(root, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_cached_graph_path(model_path: str, provider_tag: str) -> str:
    cache_dir = _get_cache_dir(model_path)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(cache_dir, f"{stem}_{provider_tag}_opt.onnx")


def _is_cache_valid(cache_path: str, original_path: str) -> bool:
    try:
        if not os.path.isfile(cache_path) or os.path.getsize(cache_path) == 0:
            return False
        return os.path.getmtime(cache_path) >= os.path.getmtime(original_path)
    except Exception:
        return False


_SESSION_CACHE: dict[str, "ort.InferenceSession"] = {}


def clear_nafnet_session_cache() -> None:
    """Libera las sesiones ONNX retenidas en memoria caché."""
    global _SESSION_CACHE
    _SESSION_CACHE.clear()


def _create_inference_session(model_path: str, use_cache: bool = True, force_cpu: bool = False) -> "ort.InferenceSession":
    """
    Inicializa o recupera de caché la sesión ONNX con el proveedor de hardware más rápido disponible.
    Utiliza caché de grafos optimizados serializados en models/.cache/.
    """
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
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available_providers:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        provider_tag = "gpu" if len(providers) > 1 else "cpu"

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_cores = os.cpu_count() or 4
    opts.intra_op_num_threads = min(16, max(1, cpu_cores))
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    cache_file = _get_cached_graph_path(abs_path, provider_tag)

    if _is_cache_valid(cache_file, abs_path):
        try:
            sess = ort.InferenceSession(cache_file, sess_options=opts, providers=providers)
            if use_cache:
                _SESSION_CACHE[cache_key] = sess
            return sess
        except Exception as e:
            logger.warning(f"Error cargando grafo NAFNet optimizado ({cache_file}): {e}. Recompilando...")
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


def _compute_image_stats(img: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula estadísticas globales (mediana, MAD, sigma_hat) usando una muestra regular
    de máximo ~200k píxeles para velocidad instantánea sin pérdida de representatividad.
    """
    total_pix = img.shape[0] * img.shape[1]
    if total_pix > 200_000:
        step = int(np.ceil(np.sqrt(total_pix / 200_000)))
        sample = img[::step, ::step]
    else:
        sample = img

    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    sigma_hat = float(1.4826 * mad)
    return med, mad, sigma_hat


def _apply_mtf(x: np.ndarray, m: float) -> np.ndarray:
    """
    Transformación directa analítica MTF:
    MTF(m, x) = (m - 1.0) * x / ((2.0 * m - 1.0) * x - m)
    """
    if abs(m - 0.5) < 1e-4:
        return x
    denom = (2.0 * m - 1.0) * x - m
    denom = np.where(np.abs(denom) < 1e-7, np.sign(denom) * 1e-7 + (denom == 0) * 1e-7, denom)
    out = (m - 1.0) * x / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_inv_mtf(y: np.ndarray, m: float) -> np.ndarray:
    """
    Transformación inversa analítica MTF:
    MTF^-1(m, y) = m * y / ((2.0 * m - 1.0) * y - (m - 1.0))
    """
    if abs(m - 0.5) < 1e-4:
        return y
    denom = (2.0 * m - 1.0) * y - (m - 1.0)
    denom = np.where(np.abs(denom) < 1e-7, np.sign(denom) * 1e-7 + (denom == 0) * 1e-7, denom)
    out = (m * y) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _run_single_scale_tiled(
    img_rgb: np.ndarray,
    session: "ort.InferenceSession",
    input_name: str,
    output_name: str,
    tile_size: int = 512,
    overlap: int = 64,
    progress_offset: int = 0,
    total_tiles_all: int = 0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Ejecuta una pasada completa de inferencia NAFNet tesela por tesela con ventana Hann 2D.
    Tolerante a fallos: degrada automáticamente a CPU si la GPU satura su memoria.
    """
    orig_h, orig_w = img_rgb.shape[:2]

    pad_h = max(0, tile_size - orig_h)
    pad_w = max(0, tile_size - orig_w)
    if pad_h > 0 or pad_w > 0:
        padded_img = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        padded_img = img_rgb

    h_pad, w_pad = padded_img.shape[:2]
    y_coords = _get_tile_coords(h_pad, tile_size, overlap)
    x_coords = _get_tile_coords(w_pad, tile_size, overlap)

    hann_win = compute_hann_2d_window(tile_size, tile_size)

    accumulator_img = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    accumulator_weights = np.zeros((h_pad, w_pad, 1), dtype=np.float32)

    current_tile = 0
    for y in y_coords:
        for x in x_coords:
            if cancel_check and cancel_check():
                raise InterruptedError("Inferencia NAFNet cancelada por el usuario.")

            tile_in = padded_img[y:y + tile_size, x:x + tile_size]
            tensor_in = np.transpose(tile_in, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

            try:
                outputs = session.run([output_name], {input_name: tensor_in})
            except Exception as run_err:
                active_providers = session.get_providers()
                if active_providers and active_providers[0] != "CPUExecutionProvider" and model_path:
                    logger.warning(f"Error GPU en NAFNet ({run_err}). Degradando a CPU...")
                    notify_gpu_fallback(f"NAFNet AI Denoise: {run_err}")
                    session = _create_inference_session(model_path, use_cache=True, force_cpu=True)
                    input_name = session.get_inputs()[0].name
                    output_name = session.get_outputs()[0].name
                    outputs = session.run([output_name], {input_name: tensor_in})
                else:
                    raise run_err
            out_tile = outputs[0].squeeze(0)
            out_tile = np.transpose(out_tile, (1, 2, 0))

            if HAS_RUST_AI and hasattr(astroloom_rs, 'accumulate_tile_hann_inplace_rs'):
                astroloom_rs.accumulate_tile_hann_inplace_rs(
                    accumulator_img, accumulator_weights, np.ascontiguousarray(out_tile, dtype=np.float32), hann_win, y, x
                )
            else:
                accumulator_img[y:y + tile_size, x:x + tile_size] += out_tile * hann_win
                accumulator_weights[y:y + tile_size, x:x + tile_size] += hann_win

            current_tile += 1
            if progress_callback and total_tiles_all > 0:
                progress_callback(progress_offset + current_tile, total_tiles_all)

    if HAS_RUST_AI and hasattr(astroloom_rs, 'reconstruct_hann_tiling_rs'):
        clean_out = astroloom_rs.reconstruct_hann_tiling_rs(
            accumulator_img, accumulator_weights, orig_h, orig_w
        )
    else:
        clean_padded = np.divide(
            accumulator_img,
            accumulator_weights,
            out=np.zeros_like(accumulator_img),
            where=accumulator_weights > 1e-12
        )
        clean_out = clean_padded[:orig_h, :orig_w]
    return np.clip(clean_out, 0.0, 1.0).astype(np.float32)


def run_nafnet_tiled_inference(
    img_rgb: np.ndarray,
    model_path: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 64,
    clean_chroma: bool = True,
    chroma_strength: float = 0.85,
    strength: float = 1.0,
    multiscale_chroma: bool = True,
    auto_mtf: bool = True,
    detail_recovery: float = 0.0,
    sky_mask: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> np.ndarray:
    """
    Ejecuta reducción de ruido profunda con NAFNet sobre la imagen mediante Tiling y Ventana 2D de Hann.

    Parámetros:
        img_rgb: Imagen RGB float32 [0.0, 1.0] o uint8 [0, 255].
        model_path: Ruta opcional al archivo .onnx.
        tile_size: Dimensión cuadrada de cada bloque (múltiplo de 16, default 512).
        overlap: Píxeles de solapamiento entre bloques contiguos (default 64).
        clean_chroma: Si es True, mantiene retrocompatibilidad para limpieza de croma.
        chroma_strength: Intensidad de eliminación de ruido cromático [0.0, 1.0].
        strength: Intensidad de reducción residual [0.0, 1.35].
        multiscale_chroma: Si es True, ejecuta inferencia a dos escalas (1.0x luminancia + 0.5x croma).
        auto_mtf: Si es True, estira automáticamente datos lineales para la red y revierte al final.
        detail_recovery: Factor de reinyección de microestructuras [0.0, 1.0].
        sky_mask: Máscara espacial opcional [0.0, 1.0] para modular intensidad (cielo vs suelo).
        progress_callback: fn(bloque_actual, total_bloques).
        cancel_check: fn() -> bool, True para abortar.

    Retorna:
        np.ndarray: Imagen RGB float32 [0.0, 1.0] reducida de ruido con preservación fotométrica exacta.
    """
    if not HAS_ONNX:
        raise RuntimeError("onnxruntime no está instalado en el entorno.")

    path = model_path or get_nafnet_model_path()
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("No se encontró el modelo ONNX de NAFNet.")

    # Asegurar formato estándar [0.0, 1.0] float32 RGB
    if img_rgb.dtype == np.uint8:
        base = img_rgb.astype(np.float32) / 255.0
    else:
        base = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)

    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)

    # Invarianza fotométrica rápida: si la fuerza es nula, devolver entrada intacta
    if strength <= 1e-5 and chroma_strength <= 1e-5:
        return base

    orig_h, orig_w = base.shape[:2]

    # Pre-estirado MTF global reversible si los datos son lineales o sub-estirados
    s_bp = 0.0
    m_param = 0.5
    is_mtf_applied = False

    if auto_mtf:
        med, mad, sigma_hat = _compute_image_stats(base)
        if med < 0.15:
            s_bp = max(0.0, float(med - 2.8 * sigma_hat))
            s_bp = min(s_bp, max(0.0, med - 1e-4))
            denom = max(1e-6, 1.0 - s_bp)
            base_norm = np.clip((base - s_bp) / denom, 0.0, 1.0).astype(np.float32)
            med_norm = float(np.clip((med - s_bp) / denom, 1e-4, 0.9999))
            t_target = 0.22
            denom_m = med_norm * (2.0 * t_target - 1.0) - t_target
            if abs(denom_m) > 1e-7:
                m_cand = (med_norm * (t_target - 1.0)) / denom_m
                m_param = float(np.clip(m_cand, 0.001, 0.999))
                base_stretched = _apply_mtf(base_norm, m_param)
                is_mtf_applied = True
            else:
                base_stretched = base_norm
        else:
            base_stretched = base
    else:
        base_stretched = base

    # Inicializar sesión ONNX (reutiliza caché global si ya está cargada)
    session = _create_inference_session(path, use_cache=True)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Cálculo de bloques totales para reporte de progreso continuo
    pad_h_1x = max(0, tile_size - orig_h)
    pad_w_1x = max(0, tile_size - orig_w)
    tiles_1x = len(_get_tile_coords(orig_h + pad_h_1x, tile_size, overlap)) * len(_get_tile_coords(orig_w + pad_w_1x, tile_size, overlap))

    use_multiscale = multiscale_chroma and (chroma_strength > 1e-4) and (orig_h >= 32 and orig_w >= 32)
    if use_multiscale:
        h_half = max(16, orig_h // 2)
        w_half = max(16, orig_w // 2)
        pad_h_05x = max(0, tile_size - h_half)
        pad_w_05x = max(0, tile_size - w_half)
        tiles_05x = len(_get_tile_coords(h_half + pad_h_05x, tile_size, overlap)) * len(_get_tile_coords(w_half + pad_w_05x, tile_size, overlap))
    else:
        tiles_05x = 0

    total_tiles_all = tiles_1x + tiles_05x

    # Pasada 1: Resolución nativa (1.0x) para luminancia y detalle fino
    out_1x = _run_single_scale_tiled(
        base_stretched, session, input_name, output_name,
        tile_size=tile_size, overlap=overlap,
        progress_offset=0, total_tiles_all=total_tiles_all,
        progress_callback=progress_callback, cancel_check=cancel_check,
        model_path=path
    )

    # Pasada 2: Escala media (0.5x) para croma desacoplado
    if use_multiscale:
        small_in = cv2.resize(base_stretched, (w_half, h_half), interpolation=cv2.INTER_AREA)
        out_05x_small = _run_single_scale_tiled(
            small_in, session, input_name, output_name,
            tile_size=tile_size, overlap=overlap,
            progress_offset=tiles_1x, total_tiles_all=total_tiles_all,
            progress_callback=progress_callback, cancel_check=cancel_check,
            model_path=path
        )
        out_05x = cv2.resize(out_05x_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Fusión en espacio YCrCb: Y de 1.0x + Cr/Cb de 0.5x
        ycrcb_1x = cv2.cvtColor(out_1x, cv2.COLOR_RGB2YCrCb)
        ycrcb_05x = cv2.cvtColor(out_05x, cv2.COLOR_RGB2YCrCb)
        blend_c = float(np.clip(chroma_strength, 0.0, 1.0))
        ycrcb_fused = np.empty_like(ycrcb_1x)
        ycrcb_fused[..., 0] = ycrcb_1x[..., 0]
        ycrcb_fused[..., 1] = ycrcb_1x[..., 1] * (1.0 - blend_c) + ycrcb_05x[..., 1] * blend_c
        ycrcb_fused[..., 2] = ycrcb_1x[..., 2] * (1.0 - blend_c) + ycrcb_05x[..., 2] * blend_c
        clean_stretched = cv2.cvtColor(ycrcb_fused, cv2.COLOR_YCrCb2RGB)
    else:
        if clean_chroma and chroma_strength > 1e-4:
            clean_stretched = preprocess_chroma_clean(out_1x, color_strength=chroma_strength)
        else:
            clean_stretched = out_1x

    # Modulación residual (strength) y máscara de cielo
    residual = base_stretched - clean_stretched
    eff_strength = float(strength)

    if sky_mask is not None:
        sm = sky_mask
        if sm.shape[:2] != (orig_h, orig_w):
            sm = cv2.resize(sm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if sm.ndim == 2:
            sm = sm[..., np.newaxis]
        sm = np.clip(sm, 0.0, 1.0).astype(np.float32)
        eff_strength = eff_strength * (0.45 + 0.55 * sm)

    blended_stretched = np.clip(base_stretched - eff_strength * residual, 0.0, 1.0).astype(np.float32)

    # Recuperación opcional de microdetalle
    if detail_recovery > 1e-4:
        y_orig = cv2.cvtColor(base_stretched, cv2.COLOR_RGB2GRAY)
        g1 = cv2.GaussianBlur(y_orig, (0, 0), sigmaX=0.8)
        g2 = cv2.GaussianBlur(y_orig, (0, 0), sigmaX=2.0)
        d_hf = g1 - g2
        _, _, sigma_y = _compute_image_stats(y_orig)
        hf_mask = (np.abs(d_hf) > (2.5 * sigma_y)).astype(np.float32)
        d_hf_protected = d_hf * hf_mask * float(detail_recovery)

        ycrcb_blended = cv2.cvtColor(blended_stretched, cv2.COLOR_RGB2YCrCb)
        ycrcb_blended[..., 0] = np.clip(ycrcb_blended[..., 0] + d_hf_protected, 0.0, 1.0)
        blended_stretched = cv2.cvtColor(ycrcb_blended, cv2.COLOR_YCrCb2RGB)

    # Reversión analítica de MTF y des-normalización a fotometría original
    if is_mtf_applied:
        inv_mtf = _apply_inv_mtf(blended_stretched, m_param)
        clean_out = np.clip(inv_mtf * (1.0 - s_bp) + s_bp, 0.0, 1.0).astype(np.float32)
    else:
        clean_out = blended_stretched

    return clean_out
