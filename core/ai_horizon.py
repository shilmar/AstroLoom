# core/ai_horizon.py
"""
core/ai_horizon.py - Detección y segmentación automática de horizonte y suelo mediante IA ligera.
Utiliza inferencia en ONNX Runtime (U²-Net especializada en cielo/suelo) con refinado
subpíxel mediante Filtro Guiado (Guided Filter) de luminancia y fallback algorítmico continuo.
"""
from typing import Optional, Tuple
import os
import time
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from core.config_manager import get_config_val


def get_model_path() -> Optional[str]:
    """
    Localiza el archivo del modelo ONNX de segmentación de cielo por orden de prioridad:
    1. Ruta personalizada en configuración (`ai_sky_model_path`).
    2. Directorio `models/skyseg.onnx` en la raíz del proyecto.
    3. Rutas alternativas en `assets/models/`.
    """
    custom = get_config_val("ai_sky_model_path", "")
    if custom and os.path.isfile(custom):
        return os.path.abspath(custom)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(root, "models", "skyseg.onnx"),
        os.path.join(root, "models", "u2netp.onnx"),
        os.path.join(root, "assets", "models", "skyseg.onnx"),
        os.path.join(root, "assets", "models", "u2netp.onnx"),
        os.path.join(root, "skyseg.onnx")
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def is_ai_model_available() -> bool:
    """Verifica si onnxruntime y el archivo del modelo están disponibles."""
    if not HAS_ONNX:
        return False
    path = get_model_path()
    return path is not None and os.path.isfile(path)


DEFAULT_MODEL_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
DEFAULT_MODEL_FILENAME = "skyseg.onnx"


def download_ai_model(
    target_path: Optional[str] = None,
    progress_callback=None,
    cancel_check=None
) -> str:
    """
    Descarga el modelo ONNX desde Hugging Face con reporte de progreso en bloques.
    Escribe en un archivo temporal (.tmp) y renombra atómicamente al terminar.
    
    Parámetros:
        target_path: Ruta destino opcional (por defecto models/skyseg.onnx).
        progress_callback: fn(porcentaje: int, descargado_mb: float, total_mb: float).
        cancel_check: fn() -> bool, True para abortar.
        
    Retorna:
        str: Ruta absoluta del modelo descargado.
    """
    import urllib.request

    if not target_path:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        models_dir = os.path.join(root, "models")
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, DEFAULT_MODEL_FILENAME)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    temp_path = target_path + ".tmp"
    url = DEFAULT_MODEL_URL

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AstroLoom/1.0 SkySegmentationDownloader"}
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 167.8
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
    return target_path



def fast_guided_filter(
    guide_gray: np.ndarray, 
    src_mask: np.ndarray, 
    radius: int = 15, 
    eps: float = 1e-3
) -> np.ndarray:
    """
    Filtro Guiado ultrarrápido (He et al., ECCV 2010) en NumPy/OpenCV.
    Alinea los bordes de la probabilidad de máscara con las aristas reales
    de árboles, rocas y montañas de la imagen guía con precisión subpíxel.
    
    Parámetros:
        guide_gray: Imagen 2D float32 [0.0, 1.0] (luminancia de alta resolución).
        src_mask: Máscara 2D float32 [0.0, 1.0] (probabilidad coarse).
        radius: Radio del filtro en píxeles.
        eps: Regularización para evitar división por cero en regiones uniformes.
        
    Retorna:
        Máscara 2D float32 [0.0, 1.0] refinada respetando los bordes.
    """
    r = max(1, int(radius))
    ksize = (r, r)
    
    mean_I = cv2.boxFilter(guide_gray, -1, ksize)
    mean_p = cv2.boxFilter(src_mask, -1, ksize)
    mean_Ip = cv2.boxFilter(guide_gray * src_mask, -1, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_gray * guide_gray, -1, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    q = mean_a * guide_gray + mean_b
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def detect_sky_mask_ai(
    image_rgb: np.ndarray,
    feather_radius: int = 7,
    threshold: float = 0.5,
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Segmentación semántica profunda del cielo y suelo usando el modelo ONNX U²-Net.
    
    Parámetros:
        image_rgb: Imagen RGB float32 [0.0, 1.0] o uint8 [0, 255].
        feather_radius: Radio del filtro guiado para bordes finos de vegetación.
        threshold: Umbral de decisión cielo/suelo [0.0, 1.0] (default 0.5).
        model_path: Ruta explícita al modelo ONNX (opcional).
        
    Retorna:
        np.ndarray: Máscara continua float32 [0.0, 1.0] (1.0 = cielo, 0.0 = suelo).
    """
    if not HAS_ONNX:
        raise RuntimeError("onnxruntime no está instalado en el entorno.")

    path = model_path or get_model_path()
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el modelo ONNX de segmentación de cielo.")

    # Asegurar formato float32 [0.0, 1.0]
    if image_rgb.dtype == np.uint8:
        rgb_f = image_rgb.astype(np.float32) / 255.0
    else:
        rgb_f = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    if rgb_f.ndim == 2:
        rgb_f = np.stack([rgb_f] * 3, axis=-1)

    h_orig, w_orig = rgb_f.shape[:2]

    # Preprocesado: Redimensionar a 320x320 con normalización ImageNet
    net_in = cv2.resize(rgb_f, (320, 320), interpolation=cv2.INTER_AREA)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    net_in = (net_in - mean) / std
    net_in = net_in.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    # Inferencia ONNX
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: net_in})

    # d0 es la predicción combinada de U²-Net
    pred = outputs[0].squeeze()  # (320, 320)

    # Normalización del mapa de probabilidad
    p_min, p_max = float(pred.min()), float(pred.max())
    if p_max - p_min > 1e-6:
        pred_norm = (pred - p_min) / (p_max - p_min)
    else:
        pred_norm = np.zeros_like(pred)

    # Reescalar a resolución nativa
    mask_coarse = cv2.resize(pred_norm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Luminancia de guía para el Guided Filter
    gray_guide = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY) if rgb_f.shape[2] == 3 else rgb_f[..., 0]

    # Refinado guiado de microestructuras (ramas y follaje)
    r_guide = max(5, int(feather_radius * 2))
    refined = fast_guided_filter(gray_guide, mask_coarse, radius=r_guide, eps=1e-3)

    # Aplicar umbralización suave o binaria adaptativa centrada en threshold
    if threshold != 0.5:
        offset = 0.5 - float(threshold)
        refined = np.clip(refined + offset, 0.0, 1.0)

    return refined.astype(np.float32)


def detect_sky_mask_algorithmic(
    image_rgb: np.ndarray,
    feather_radius: int = 7
) -> np.ndarray:
    """
    Fallback algorítmico matemático de alta velocidad (cuando no hay modelo IA disponible).
    Atenúa estrellas mediante apertura morfológica y extrae el perfil de horizonte por
    gradiente vertical y umbralización adaptativa, refinado con Filtro Guiado.
    """
    if image_rgb.dtype == np.uint8:
        rgb_f = image_rgb.astype(np.float32) / 255.0
    else:
        rgb_f = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    if rgb_f.ndim == 3:
        gray = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_f.copy()

    h, w = gray.shape[:2]

    # 1. Supresión de estrellas (apertura morfológica con elipse de 7x7)
    k_star = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    gray_nostars = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k_star)

    # 2. Análisis vertical de transición cielo/suelo
    # El cielo nocturno (con estrellas/airglow/polución) tiene mayor luminancia media que el suelo en sombra
    gray_8u = (np.clip(gray_nostars, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Umbral de Otsu adaptativo sobre imagen sin estrellas
    otsu_val, _ = cv2.threshold(gray_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coarse_sky = (gray_8u > otsu_val).astype(np.float32)

    # Verificación de orientación: en fotografía de paisaje el cielo está arriba
    top_mean = float(coarse_sky[:h // 4, :].mean())
    bot_mean = float(coarse_sky[3 * h // 4:, :].mean())
    if bot_mean > top_mean:
        # Invertir si por iluminación local el suelo quedó blanco
        coarse_sky = 1.0 - coarse_sky

    # 3. Refinado con Filtro Guiado
    r_guide = max(5, int(feather_radius * 2))
    refined = fast_guided_filter(gray, coarse_sky, radius=r_guide, eps=1e-3)
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def auto_detect_sky_mask(
    image_rgb: np.ndarray,
    feather_radius: int = 7,
    threshold: float = 0.5,
    force_algorithmic: bool = False
) -> Tuple[np.ndarray, str]:
    """
    Función de alto nivel para detección automática en 1 clic.
    Intenta ejecutar la red neuronal ONNX y, si no está disponible o falla,
    hace fallback automático al algoritmo fotométrico de gradientes.
    
    Retorna:
        Tuple[np.ndarray, str]: (mascara_float32, nombre_del_metodo_utilizado)
    """
    t_start = time.perf_counter()
    if not force_algorithmic and is_ai_model_available():
        try:
            mask = detect_sky_mask_ai(
                image_rgb, 
                feather_radius=feather_radius, 
                threshold=threshold
            )
            elapsed = time.perf_counter() - t_start
            return mask, f"IA U²-Net ({elapsed:.2f}s)"
        except Exception as e:
            # Fallback en caso de error imprevisto en tiempo de inferencia
            pass

    mask = detect_sky_mask_algorithmic(image_rgb, feather_radius=feather_radius)
    elapsed = time.perf_counter() - t_start
    return mask, f"Algorítmico ({elapsed:.2f}s)"
