# core/ai_denoise.py
"""
core/ai_denoise.py - Reducción de Ruido Profunda con IA (NAFNet ONNX) en AstroLoom.

Implementa inferencia acelerada mediante red neuronal NAFNet (Nonlinear Activation Free Network)
optimizada para restauración de imagen y desacoplada en bloques (tiling) con ventana 2D de Hann
para garantizar continuidad fotométrica sin costuras.
"""

from typing import Optional, Callable, List, Tuple
import os
import logging
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

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

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(root, "models", DEFAULT_NAFNET_FILENAME),
        os.path.join(root, "models", "nafnet-sidd-w32.onnx"),
        os.path.join(root, "models", "NAFNet-SIDD-width32.onnx"),
        os.path.join(root, "assets", "models", DEFAULT_NAFNET_FILENAME),
        os.path.join(root, "assets", "models", "nafnet-sidd-w32.onnx"),
        os.path.join(root, DEFAULT_NAFNET_FILENAME)
    ]
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
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    y_w = np.sin(np.pi * (np.arange(height, dtype=np.float32) + 0.5) / height) ** 2
    x_w = np.sin(np.pi * (np.arange(width, dtype=np.float32) + 0.5) / width) ** 2
    return np.outer(y_w, x_w).astype(np.float32)[..., np.newaxis]


def _get_tile_coords(length: int, tile_size: int, overlap: int) -> List[int]:
    """Genera las coordenadas de inicio de los bloques cubriendo la longitud completa."""
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


def _create_inference_session(model_path: str) -> "ort.InferenceSession":
    """Inicializa la sesión ONNX con el proveedor de hardware más rápido disponible."""
    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available_providers:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_cores = os.cpu_count() or 4
    opts.intra_op_num_threads = min(16, max(1, cpu_cores))
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def run_nafnet_tiled_inference(
    img_rgb: np.ndarray,
    model_path: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 64,
    clean_chroma: bool = True,
    chroma_strength: float = 0.5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> np.ndarray:
    """
    Ejecuta reducción de ruido profunda con NAFNet sobre la imagen mediante Tiling y Ventana 2D de Hann.

    Parámetros:
        img_rgb: Imagen RGB float32 [0.0, 1.0].
        model_path: Ruta opcional al archivo .onnx.
        tile_size: Dimensión cuadrada de cada bloque (múltiplo de 16, default 512).
        overlap: Píxeles de solapamiento entre bloques contiguos (default 64).
        clean_chroma: Si es True, ejecuta filtrado cromático suave preliminar.
        chroma_strength: Fuerza de limpieza cromática [0.0, 1.0].
        progress_callback: fn(bloque_actual, total_bloques).
        cancel_check: fn() -> bool, True para abortar.

    Retorna:
        np.ndarray: Imagen RGB float32 [0.0, 1.0] reducida de ruido con microestructura intacta.
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

    # 1. Limpieza de croma previa opcional
    if clean_chroma and chroma_strength > 1e-4:
        base = preprocess_chroma_clean(base, color_strength=chroma_strength)

    orig_h, orig_w = base.shape[:2]

    # Relleno reflectivo si la imagen es menor que el tamaño de bloque
    pad_h = max(0, tile_size - orig_h)
    pad_w = max(0, tile_size - orig_w)
    if pad_h > 0 or pad_w > 0:
        padded_img = cv2.copyMakeBorder(base, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        padded_img = base

    h_pad, w_pad = padded_img.shape[:2]

    y_coords = _get_tile_coords(h_pad, tile_size, overlap)
    x_coords = _get_tile_coords(w_pad, tile_size, overlap)
    total_tiles = len(y_coords) * len(x_coords)

    hann_win = compute_hann_2d_window(tile_size, tile_size)

    accumulator_img = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    accumulator_weights = np.zeros((h_pad, w_pad, 1), dtype=np.float32)

    session = _create_inference_session(path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    current_tile = 0
    for y in y_coords:
        for x in x_coords:
            if cancel_check and cancel_check():
                raise InterruptedError("Inferencia NAFNet cancelada por el usuario.")

            # Extraer bloque
            tile_in = padded_img[y:y + tile_size, x:x + tile_size]

            # Formato tensor NCHW float32
            tensor_in = np.transpose(tile_in, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

            # Inferencia ONNX
            outputs = session.run([output_name], {input_name: tensor_in})
            out_tile = outputs[0].squeeze(0)  # (3, H, W)
            out_tile = np.transpose(out_tile, (1, 2, 0))  # (H, W, 3)

            # Ponderación con ventana 2D de Hann
            accumulator_img[y:y + tile_size, x:x + tile_size] += out_tile * hann_win
            accumulator_weights[y:y + tile_size, x:x + tile_size] += hann_win

            current_tile += 1
            if progress_callback:
                progress_callback(current_tile, total_tiles)

    # Reconstrucción normalizada sin división por cero
    clean_padded = np.divide(
        accumulator_img,
        accumulator_weights,
        out=np.zeros_like(accumulator_img),
        where=accumulator_weights > 1e-12
    )

    # Recortar si se rellenó
    clean_out = clean_padded[:orig_h, :orig_w]
    return np.clip(clean_out, 0.0, 1.0).astype(np.float32)
