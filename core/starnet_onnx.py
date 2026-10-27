# core/starnet_onnx.py
"""
core/starnet_onnx.py - Inferencia nativa DirectML con StarNet v2 ONNX en AstroLoom.

Ejecuta el modelo oficial de StarNet v2 mediante ONNX Runtime con aceleración DirectML (GPU)
o CPU fallback, eliminando dependencias de binarios externos, OpenCV en C++ o archivos TIFF temporales.
Implementa tiling con ventana 2D de Hann y ciclo MTF per-canal para imágenes lineales.
"""

from typing import Optional, Callable, Tuple, List
import os
import shutil
import logging
import zipfile
import urllib.request
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from core.config_manager import get_config_val

logger = logging.getLogger("AstroLoom.StarNetONNX")

DEFAULT_STARNET_URL = "https://download.starnetastro.com/starnet2_win_2.6.2-0242_ORT_x64_cli.zip"
DEFAULT_STARNET_FILENAME = "starnet2.onnx"
DEFAULT_TILE_SIZE = 512
DEFAULT_STRIDE = 256


def get_starnet_model_path() -> Optional[str]:
    """
    Localiza el archivo de pesos ONNX de StarNet v2 por orden de prioridad:
    1. Configuración global ('starnet_onnx_path').
    2. Directorio 'models/starnet2.onnx' en la raíz de AstroLoom.
    3. Directorio 'models/StarNet2_weights.onnx'.
    4. Instalación estándar de StarNet 2 en Windows ('C:\\Program Files\\StarNet2\\lib\\starnet2\\...').
    5. Rutas locales o portables en el directorio de AstroLoom o de software.
    """
    custom = get_config_val("starnet_onnx_path", "")
    if custom and os.path.isfile(custom):
        return os.path.abspath(custom)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(root, "models", DEFAULT_STARNET_FILENAME),
        os.path.join(root, "models", "StarNet2_weights.onnx"),
        r"C:\Program Files\StarNet2\lib\starnet2\StarNet2_weights.onnx",
        r"D:\software\StarNet2\lib\starnet2\StarNet2_weights.onnx",
        os.path.join(root, "StarNet2_weights.onnx"),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    return None


def is_starnet_onnx_available() -> bool:
    """Indica si onnxruntime y el archivo de pesos ONNX de StarNet están disponibles."""
    if not HAS_ONNX:
        return False
    path = get_starnet_model_path()
    return path is not None and os.path.isfile(path)


def download_starnet_model(
    target_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Obtiene el archivo ONNX de StarNet v2 (~131 MB).
    Si existe una instalación local de StarNet2 en el sistema, lo copia de forma instantánea.
    De lo contrario, descarga el paquete oficial de StarNet, extrae los pesos a 'models/starnet2.onnx'
    y limpia los archivos temporales.
    """
    if not target_path:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        models_dir = os.path.join(root, "models")
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, DEFAULT_STARNET_FILENAME)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    # 1. Comprobación de instalación local previa
    local_source = r"C:\Program Files\StarNet2\lib\starnet2\StarNet2_weights.onnx"
    if os.path.isfile(local_source):
        logger.info(f"Copiando pesos desde instalación local existente: {local_source}")
        shutil.copy2(local_source, target_path)
        if progress_callback:
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            progress_callback(100, size_mb, size_mb)
        return target_path

    # 2. Descarga oficial del ZIP y extracción exclusiva del archivo .onnx
    zip_temp = target_path + ".download.zip"
    req = urllib.request.Request(
        DEFAULT_STARNET_URL,
        headers={"User-Agent": "AstroLoom/1.0 StarNetDownloader"}
    )

    try:
        with urllib.request.urlopen(req) as resp, open(zip_temp, "wb") as f_out:
            total_size = int(resp.headers.get("Content-Length", 0))
            total_mb = total_size / (1024 * 1024) if total_size > 0 else 184.0
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB

            while True:
                if cancel_check and cancel_check():
                    raise InterruptedError("Descarga cancelada por el usuario.")

                chunk = resp.read(block_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                downloaded_mb = downloaded / (1024 * 1024)
                pct = int((downloaded / total_size) * 100) if total_size > 0 else 0
                if progress_callback:
                    progress_callback(pct, downloaded_mb, total_mb)

        # 3. Extraer únicamente StarNet2_weights.onnx
        with zipfile.ZipFile(zip_temp, 'r') as zf:
            matching = [name for name in zf.namelist() if name.endswith("StarNet2_weights.onnx")]
            if not matching:
                raise FileNotFoundError("El archivo ZIP oficial no contiene StarNet2_weights.onnx.")
            with zf.open(matching[0]) as source_file, open(target_path, "wb") as dest_file:
                shutil.copyfileobj(source_file, dest_file)

        logger.info(f"Modelo StarNet v2 ONNX instalado con éxito en: {target_path}")
        return target_path

    finally:
        if os.path.exists(zip_temp):
            try:
                os.remove(zip_temp)
            except OSError:
                pass


def compute_hann_2d_window(height: int, width: int) -> np.ndarray:
    """Calcula una ventana de ponderación 2D de Hann separable para mezcla suave sin costuras."""
    y_w = np.sin(np.pi * (np.arange(height, dtype=np.float32) + 0.5) / height) ** 2
    x_w = np.sin(np.pi * (np.arange(width, dtype=np.float32) + 0.5) / width) ** 2
    return np.outer(y_w, x_w).astype(np.float32)[..., np.newaxis]


def _get_tile_coords(length: int, tile_size: int, stride: int) -> List[int]:
    """Genera coordenadas de inicio de bloques cubriendo uniformemente la dimensión."""
    if length <= tile_size:
        return [0]
    coords = []
    pos = 0
    while pos + tile_size < length:
        coords.append(pos)
        pos += stride
    coords.append(length - tile_size)
    return sorted(list(set(coords)))


def _create_inference_session(model_path: str) -> "ort.InferenceSession":
    """Inicializa la sesión ONNX con aceleración GPU (DirectML) y fallback a CPU."""
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


def _mtf_curve(x: np.ndarray, m: float) -> np.ndarray:
    """Función analítica MTF: s(x) = (m - 1) * x / ((2m - 1) * x - m)."""
    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m
    return np.where(np.abs(den) > 1e-7, num / den, 0.0)


def _calc_channel_mtf(
    channel: np.ndarray,
    target_bg: float = 0.25,
    shadows_factor: float = -2.80
) -> Tuple[float, float]:
    """Calcula el punto negro (bp) y el valor de medios tonos (m) para una curva MTF analítica."""
    sample = channel[::4, ::4]
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    bp = float(np.clip(med + shadows_factor * 1.4826 * mad, 0.0, 0.95))

    norm_med = max(1e-6, (med - bp) / max(1e-6, 1.0 - bp))
    denom = norm_med * (2.0 * target_bg - 1.0) - target_bg
    m = (norm_med * (target_bg - 1.0)) / denom if abs(denom) > 1e-7 else 0.5
    m = float(np.clip(m, 0.0050, 0.9999))
    return bp, m


def run_starnet_onnx(
    img_rgb: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    stride: int = DEFAULT_STRIDE,
    model_path: Optional[str] = None,
    is_linear: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta StarNet v2 de forma nativa mediante ONNX Runtime DirectML.

    Parámetros:
        img_rgb: Imagen de entrada en float32 [0.0, 1.0] (RGB).
        sky_mask: Máscara opcional (float32 [0.0, 1.0]) donde 1=cielo y 0=suelo.
        stride: Salto de desplazamiento entre bloques (default 256 px).
        model_path: Ruta opcional a los pesos .onnx.
        is_linear: Si es True, ejecuta el ciclo analítico MTF per-canal oficial de StarNet.
        progress_callback: fn(mensaje) para reportar estado y porcentaje a la UI.
        cancel_check: fn() -> bool, devuelve True si el usuario solicita cancelar.

    Retorna:
        starless_final: Imagen sin estrellas (suelo restaurado si se pasó sky_mask).
        stars_only: Capa aislada de estrellas en espacio lineal (RGB).
    """
    if not HAS_ONNX:
        raise RuntimeError("onnxruntime no está instalado en el entorno.")

    path = model_path or get_starnet_model_path()
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "No se encontró el archivo de modelo ONNX de StarNet v2 ('starnet2.onnx').\n"
            "Puedes descargarlo automáticamente desde la interfaz."
        )

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    # 1. Asegurar formato de entrada RGB float32 [0.0, 1.0]
    if img_rgb.dtype == np.uint8:
        base = img_rgb.astype(np.float32) / 255.0
    elif img_rgb.dtype == np.uint16:
        base = img_rgb.astype(np.float32) / 65535.0
    else:
        base = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)

    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)

    h, w = base.shape[:2]
    log(f"Iniciando inferencia nativa StarNet v2 ONNX ({w}x{h} px)...")

    # 2. Protección del suelo: neutralizar el paisaje con el fondo del cielo si hay máscara
    input_data = base.copy()
    mask_3d = None
    if sky_mask is not None:
        if sky_mask.shape[:2] != (h, w):
            mask_aligned = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_aligned = sky_mask

        sky_pixels = input_data[mask_aligned > 0.5]
        bg_color = np.median(sky_pixels, axis=0) if len(sky_pixels) > 0 else np.array([0.005, 0.005, 0.005], dtype=np.float32)
        mask_3d = mask_aligned[..., np.newaxis] if mask_aligned.ndim == 2 else mask_aligned
        input_data = (input_data * mask_3d) + (bg_color * (1.0 - mask_3d))

    # 3. Preparación para modo lineal: ciclo MTF per-canal
    bps: List[float] = []
    ms: List[float] = []
    if is_linear:
        log("Calculando estirado analítico MTF per-canal (shadows_factor=-2.80)...")
        stretched = np.zeros_like(input_data)
        for c in range(3):
            ch = input_data[..., c]
            bp, m = _calc_channel_mtf(ch, target_bg=0.25, shadows_factor=-2.80)
            bps.append(bp)
            ms.append(m)
            c_norm = np.clip((ch - bp) / max(1e-6, 1.0 - bp), 0.0, 1.0)
            stretched[..., c] = np.clip(_mtf_curve(c_norm, m), 0.0, 1.0)
        net_input = stretched
    else:
        net_input = input_data

    # 4. Tiling con recorte de bordes (descarta márgenes convolucionales inestables de 512x512)
    tile_size = DEFAULT_TILE_SIZE
    stride = max(32, min(tile_size - 16, stride))
    border = (tile_size - stride) // 2

    # Margen reflectivo perimetral para que todos los píxeles (incluidos bordes) tengan contexto completo
    padded_img = cv2.copyMakeBorder(net_input, border, border, border, border, cv2.BORDER_REFLECT)

    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))
    total_tiles = len(y_steps) * len(x_steps)
    log(f"Dividiendo imagen en {total_tiles} bloques ({tile_size}x{tile_size}, paso={stride}, margen={border}px)...")

    # 5. Inferencia con sesión ONNX
    sess = _create_inference_session(path)
    raw_starless = np.zeros((h, w, 3), dtype=np.float32)

    tile_idx = 0
    for y in y_steps:
        h_chunk = min(stride, h - y)
        for x in x_steps:
            if cancel_check and cancel_check():
                log("Separación cancelada por el usuario.")
                raise InterruptedError("Operación cancelada por el usuario.")

            w_chunk = min(stride, w - x)

            # Extraer parche de 512x512 con margen de contexto
            patch_in = padded_img[y : y + tile_size, x : x + tile_size]
            if patch_in.shape[:2] != (tile_size, tile_size):
                patch_in = cv2.copyMakeBorder(
                    patch_in, 0, tile_size - patch_in.shape[0], 0, tile_size - patch_in.shape[1], cv2.BORDER_REFLECT
                )

            in_batch = np.expand_dims(patch_in, axis=0)  # [1, 512, 512, 3]
            out_batch = sess.run(None, {'x:0': in_batch})[0]
            tile_out = np.clip(out_batch[0], 0.0, 1.0)

            # Conservar exclusivamente el núcleo central estable y descartar los bordes distorsionados
            raw_starless[y : y + h_chunk, x : x + w_chunk] = tile_out[border : border + h_chunk, border : border + w_chunk]

            tile_idx += 1
            if tile_idx % 5 == 0 or tile_idx == total_tiles:
                pct = int((tile_idx / total_tiles) * 100)
                log(f"Procesando StarNetv2 AI: bloque {tile_idx}/{total_tiles} ({pct}%)...")

    # 6. Des-estirado analítico MTF inverso para imágenes lineales
    if is_linear:
        log("Revirtiendo estirado analítico MTF a espacio lineal...")
        starless_linear = np.zeros_like(raw_starless)
        for c in range(3):
            ch = input_data[..., c]
            bp = bps[c]
            m_inv = 1.0 - ms[c]
            s_norm = _mtf_curve(raw_starless[..., c], m_inv)
            s_unnorm = s_norm * (1.0 - bp) + bp
            # Restaurar sombras recortadas
            s_restored = np.where(ch <= bp, ch, s_unnorm)
            starless_linear[..., c] = np.clip(s_restored, 0.0, 1.0)
        final_starless = starless_linear
    else:
        final_starless = np.clip(raw_starless, 0.0, 1.0)

    # 7. Restaurar suelo real intacto si se suministró sky_mask
    if mask_3d is not None:
        starless_final = (final_starless * mask_3d) + (base * (1.0 - mask_3d))
    else:
        starless_final = final_starless

    starless_final = np.clip(starless_final, 0.0, 1.0).astype(np.float32)

    # 8. Capa de Estrellas puras = max(0, original - starless)
    stars_only = np.clip(base - starless_final, 0.0, 1.0).astype(np.float32)

    log("Desacoplo de estrellas completado exitosamente con DirectML.")
    return starless_final, stars_only
