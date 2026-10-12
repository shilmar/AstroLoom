# core/startrails.py
"""
core/startrails.py - Motor de generación y fusión de trazas de estrellas (Startrails / Circumpolares).

Implementa:
1. Máximo Estándar (Lighten / Max) en streaming de memoria eficiente O(1).
2. Efecto Cometa / Estela Progresiva (Comet / Meteor Fade) con decaimiento lineal, suave o exponencial.
3. Fusión de Suelo Limpio (Separación Cielo / Suelo mediante máscara con eliminación de ruido en sombras).
4. Sustracción de Darks previa a la acumulación.

Hoja de Ruta (Futuras funcionalidades anotadas):
- Fase 2: Relleno de huecos e interpolación angular de arcos (Gap Filling).
- Fase 2: Detección y supresión de trazas anómalas de satélites y aviones.
- Fase 2: Generación y exportación de secuencia acumulativa de fotogramas (Time-Lapse / Vídeo).
"""
import os
import gc
from typing import Optional, Callable, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from core.stacking import load_image_as_float32


def compute_comet_weights(
    n_frames: int,
    tail_frames: int,
    decay_type: str = "linear",
    direction: str = "backward",
    min_floor: float = 0.0
) -> np.ndarray:
    """
    Calcula el vector 1D de ponderación temporal para simular el efecto de cabeza de cometa y estela.

    :param n_frames: Número total de tomas en la serie.
    :param tail_frames: Longitud efectiva de la estela en fotogramas (>= 1).
    :param decay_type: Perfil de decaimiento: "linear", "smooth" (coseno) o "exponential".
    :param direction: "backward" (la toma más reciente es la cabeza y el pasado se desvanece),
                      "forward" (la toma más antigua es la cabeza y el futuro se desvanece),
                      "bidirectional" (doble sentido: brillo máximo central y colas afiladas en ambos extremos).
    :param min_floor: Cota mínima de luminancia de fondo (entre 0.0 y 0.50).
    :return: Array 1D float32 de longitud n_frames con pesos en [min_floor, 1.0].
    """
    if n_frames <= 1:
        return np.ones(n_frames, dtype=np.float32)

    tail_len = max(1, min(n_frames, int(tail_frames)))
    min_floor = float(np.clip(min_floor, 0.0, 0.50))
    weights = np.full(n_frames, min_floor, dtype=np.float32)

    c = (n_frames - 1) / 2.0
    half_len = max(1.0, c * (tail_len / float(n_frames)))
    d_min = 0.5 if (n_frames % 2 == 0) else 0.0
    d_max = max(1e-5, half_len - d_min)

    for i in range(n_frames):
        if direction == "backward":
            d = float((n_frames - 1) - i)
            lim = float(tail_len)
            r = min(1.0, max(0.0, d / max(lim, 1e-5)))
        elif direction == "forward":
            d = float(i)
            lim = float(tail_len)
            r = min(1.0, max(0.0, d / max(lim, 1e-5)))
        elif direction == "bidirectional":
            # Máximo brillo en el centro de la sesión y colas que se desvanecen hacia ambos extremos
            d = float(abs(i - c))
            if d <= half_len:
                r = min(1.0, max(0.0, d - d_min) / d_max)
            else:
                r = 1.0
        else:
            d = float((n_frames - 1) - i)
            lim = float(tail_len)
            r = min(1.0, max(0.0, d / max(lim, 1e-5)))

        if r < 1.0 or (r == 1.0 and min_floor > 0.0):
            if decay_type == "smooth":
                factor = float(np.cos(r * (np.pi / 2.0)) ** 2)
            elif decay_type == "exponential":
                factor = float(np.exp(-3.0 * r))
            else:
                factor = 1.0 - r

            w = min_floor + (1.0 - min_floor) * factor
            weights[i] = float(np.clip(w, min_floor, 1.0))
        else:
            weights[i] = min_floor

    return weights


def build_master_dark(
    dark_files: List[str],
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None
) -> Optional[np.ndarray]:
    """Calcula el Master Dark promediado a partir de la lista de tomas oscuras."""
    if not dark_files:
        return None

    n = len(dark_files)
    dark_acc = None
    for i, path in enumerate(dark_files):
        if abort_flag and abort_flag():
            raise InterruptedError("Procesamiento cancelado por el usuario.")
        if progress_callback:
            progress_callback(int(5 + (i / n) * 15), f"Cargando dark ({i+1}/{n}): {os.path.basename(path)}")
        d_img = load_image_as_float32(path)
        if dark_acc is None:
            dark_acc = d_img.astype(np.float64)
        else:
            dark_acc += d_img

    master_dark = (dark_acc / float(n)).astype(np.float32)
    return master_dark


def clean_frame_streaks(
    current_img: np.ndarray,
    prev_img: Optional[np.ndarray],
    next_img: Optional[np.ndarray],
    sky_mask: Optional[np.ndarray] = None,
    sensitivity: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    Detecta y suprime trazas lineales transitorias (satélites y aviones) en el fotograma actual
    mediante análisis morfológico de elongación geométrica y comparación temporal con fotogramas vecinos.

    :param current_img: Fotograma actual float32 RGB [0, 1].
    :param prev_img: Fotograma anterior (o None si es el primero).
    :param next_img: Fotograma posterior (o None si es el último).
    :param sky_mask: Máscara binaria o float opcional (1.0 = cielo, 0.0 = suelo) para excluir vegetación y suelo móvil.
    :param sensitivity: Sensibilidad de detección (0.1 a 1.0, por defecto 0.5).
    :return: (imagen_limpia, número_de_trazas_eliminadas)
    """
    if prev_img is None and next_img is None:
        return current_img, 0

    if prev_img is not None and next_img is not None:
        ref_bg = np.minimum(prev_img, next_img)
    elif prev_img is not None:
        ref_bg = prev_img
    else:
        ref_bg = next_img

    # 1. Diferencia positiva de luminancia respecto a los fotogramas vecinos
    diff = np.maximum(0.0, current_img - ref_bg)
    diff_lum = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # 2. Si se proporciona máscara de horizonte, enfocar exclusivamente en el cielo
    # Ignora el movimiento de copas de árboles, follaje o hierba que se muevan con el viento
    h, w = diff_lum.shape
    sky_mask_bool = None
    if sky_mask is not None:
        if isinstance(sky_mask, np.ndarray) and sky_mask.dtype == bool:
            if sky_mask.shape[:2] == (h, w):
                sky_mask_bool = sky_mask
            else:
                sm_res = cv2.resize(sky_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                sky_mask_bool = (sm_res > 0)
        else:
            sm_res = sky_mask.astype(np.float32)
            if sm_res.shape[:2] != (h, w):
                sm_res = cv2.resize(sm_res, (w, h), interpolation=cv2.INTER_NEAREST)
            if sm_res.ndim == 3:
                sm_res = sm_res[..., 0]
            if sm_res.max() > 1.05:
                sm_res /= 255.0
            sky_mask_bool = (sm_res > 0.3)
        diff_lum_analysis = diff_lum.copy()
        diff_lum_analysis[~sky_mask_bool] = 0.0
    else:
        diff_lum_analysis = diff_lum

    # 3. Umbral adaptativo sobre el ruido del cielo (submuestreo optimizado)
    if sky_mask_bool is not None and np.any(sky_mask_bool):
        sub_mask = sky_mask_bool[::8, ::8]
        sub_diff = diff_lum[::8, ::8]
        sample_vals = sub_diff[sub_mask] if np.any(sub_mask) else sub_diff
    else:
        sample_vals = diff_lum[::8, ::8]

    bg_val = float(np.median(sample_vals))
    mad_noise = float(1.4826 * np.median(np.abs(sample_vals - bg_val)))
    sigma_factor = max(2.5, 6.0 - (float(sensitivity) * 4.0))
    thresh_val = bg_val + max(mad_noise * sigma_factor, 0.015)

    binary = (diff_lum_analysis > thresh_val).astype(np.uint8)

    # 4. Cierre morfológico suave (3x3) para mantener estroboscópicos unidos sin fusionar estrellas vecinas
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # 5. Análisis de componentes conectadas para discriminar estrellas vs trazas largas y elongadas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return current_img, 0

    # Parámetros calibrados según la sensibilidad
    sens_f = float(np.clip(sensitivity, 0.1, 1.0))
    min_length = max(8.0, 24.0 - (sens_f * 18.0))
    min_elongation = max(2.3, 3.6 - (sens_f * 1.5))

    streak_mask = np.zeros((h, w), dtype=np.uint8)
    found_streaks = 0

    for lbl in range(1, num_labels):
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        diag = np.sqrt(float(cw * cw + ch * ch))

        if diag >= min_length:
            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]

            patch_lbl = labels[y:y+ch, x:x+cw]
            pts = np.argwhere(patch_lbl == lbl)
            if len(pts) >= 4:
                pts_xy = np.fliplr(pts).reshape(-1, 1, 2)
                rect = cv2.minAreaRect(pts_xy)
                rw, rh = rect[1]
                length = max(rw, rh)
                width = min(rw, rh)
                elongation = length / max(width, 0.5)

                if elongation >= min_elongation and length >= min_length:
                    # Asignación ultrarrápida usando el parche local O(1)
                    streak_mask[y:y+ch, x:x+cw][patch_lbl == lbl] = 255
                    found_streaks += 1

    if found_streaks == 0:
        return current_img, 0

    # 6. Dilatar ligeramente la máscara para cubrir el halo óptico difuso
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    streak_mask_dilated = cv2.dilate(streak_mask, k_dilate)

    # 7. Inpainting / Sustitución de los píxeles de la traza por el fondo limpio
    cleaned_img = current_img.copy()
    cleaned_img[streak_mask_dilated > 0] = ref_bg[streak_mask_dilated > 0]

    return cleaned_img, found_streaks


def extract_dilated_stars(
    img: np.ndarray,
    radius: int = 3,
    threshold_factor: float = 3.0,
    sky_mask_bool: Optional[np.ndarray] = None,
    kernel: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Aísla las estrellas del fotograma mediante umbral adaptativo sobre el fondo y las dilata
    para el puenteo continuo (Gap Filling).
    
    :param img: Fotograma float32 RGB [0, 1].
    :param radius: Radio de dilatación en píxeles.
    :param threshold_factor: Multiplicador sobre el MAD de ruido.
    :param sky_mask_bool: Máscara booleana precalculada (True = cielo, False = suelo).
    :param kernel: Elemento estructurante morfológico precalculado (opcional).
    :return: Matriz float32 RGB con las estrellas dilatadas.
    """
    h, w = img.shape[:2]
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if sky_mask_bool is not None and sky_mask_bool.shape[:2] == (h, w):
        sub_mask = sky_mask_bool[::8, ::8]
        sub_lum = lum[::8, ::8]
        sample = sub_lum[sub_mask] if np.any(sub_mask) else sub_lum
    else:
        sample = lum[::8, ::8]

    bg = float(np.median(sample))
    mad = float(1.4826 * np.median(np.abs(sample - bg)))
    thresh = bg + max(mad * threshold_factor, 0.015)

    stars = np.maximum(0.0, img - thresh)

    if sky_mask_bool is not None and sky_mask_bool.shape[:2] == (h, w):
        stars[~sky_mask_bool] = 0.0

    if kernel is None:
        k_size = max(3, int(radius * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    dilated = cv2.dilate(stars, kernel)
    return dilated


def compute_gap_bridge(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    radius: int = 3,
    threshold_factor: float = 3.0,
    sky_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calcula un puente continuo de luminancia estelar entre dos fotogramas consecutivos
    para eliminar los huecos periódicos (gap filling) provocados por el tiempo de obturación
    e intervalómetro.
    
    :param prev_img: Fotograma anterior (float32 RGB [0, 1]).
    :param curr_img: Fotograma actual (float32 RGB [0, 1]).
    :param radius: Radio de dilatación en píxeles (cierra saltos de hasta 2*radius píxeles).
    :param threshold_factor: Factor multiplicador sobre el MAD de ruido para aislar estrellas.
    :param sky_mask: Máscara opcional donde > 0 indica cielo. Si se proporciona, el suelo no se dilata.
    :return: Matriz float32 RGB con los puentes de estrellas calculados (0 en fondo y suelo).
    """
    h, w = prev_img.shape[:2]
    sky_mask_bool = None
    if sky_mask is not None:
        if isinstance(sky_mask, np.ndarray) and sky_mask.dtype == bool:
            if sky_mask.shape[:2] == (h, w):
                sky_mask_bool = sky_mask
            else:
                sm_res = cv2.resize(sky_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                sky_mask_bool = (sm_res > 0)
        else:
            sm = sky_mask.astype(np.float32)
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
            if sm.ndim == 3:
                sm = sm[..., 0]
            if sm.max() > 1.05:
                sm /= 255.0
            sky_mask_bool = (sm > 0.3)

    k_size = max(3, int(radius * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    d_p = extract_dilated_stars(prev_img, radius=radius, threshold_factor=threshold_factor, sky_mask_bool=sky_mask_bool, kernel=kernel)
    d_c = extract_dilated_stars(curr_img, radius=radius, threshold_factor=threshold_factor, sky_mask_bool=sky_mask_bool, kernel=kernel)

    bridge_stars = np.minimum(d_p, d_c)
    bridge_stars[bridge_stars < 0.005] = 0.0
    return bridge_stars


def compute_target_video_dims(w: int, h: int, res_mode: str) -> Tuple[int, int]:
    """
    Calcula dimensiones pares de vídeo preservando la relación de aspecto.
    Garantiza que ambos ejes sean múltiplos de 2 para compatibilidad total con codificadores H.264/MPEG-4.
    """
    is_portrait = (h > w)
    if res_mode == "1080p":
        max_long = 1920
        max_short = 1080
    elif res_mode == "4k":
        max_long = 3840
        max_short = 2160
    else:  # "original"
        return (w // 2) * 2, (h // 2) * 2

    if is_portrait:
        max_w, max_h = max_short, max_long
    else:
        max_w, max_h = max_long, max_short

    scale = min(1.0, min(float(max_w) / float(w), float(max_h) / float(h)))
    tw = max(2, (int(round(w * scale)) // 2) * 2)
    th = max(2, (int(round(h * scale)) // 2) * 2)
    return tw, th


def enhance_star_trails_color(
    img_rgb: np.ndarray,
    saturation_boost: float = 1.5,
    core_recovery: float = 0.5,
    sky_mask: Optional[np.ndarray] = None,
    bg_protect: bool = True
) -> np.ndarray:
    """
    Realza la saturación cromática de las trazas de estrellas y recupera el color
    espectral en los núcleos sobreexpuestos o quemados (blancos) sin afectar al fondo del cielo.

    :param img_rgb: Imagen RGB float32 [0.0, 1.0].
    :param saturation_boost: Factor multiplicador de saturación estelar (1.0 = neutro, 1.2–2.5 = realce vivo).
    :param core_recovery: Grado de recuperación de color en núcleos quemados (0.0 = neutro, 1.0 = máxima infusión).
    :param sky_mask: Máscara opcional (1.0 = cielo, 0.0 = suelo). Si se proporciona, el suelo no se modifica.
    :param bg_protect: Si es True, aísla el fondo oscuro para evitar amplificar ruido cromático.
    :return: Imagen float32 RGB con colores estelares preservados y realzados.
    """
    if saturation_boost <= 1.0 and core_recovery <= 0.0:
        return img_rgb

    img_f32 = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)
    h, w = img_f32.shape[:2]

    # 1. Separación de luminosidad y análisis de umbral de fondo estelar
    lum = cv2.cvtColor(img_f32, cv2.COLOR_RGB2GRAY)
    sample = lum[::8, ::8]
    med_bg = float(np.median(sample))
    mad_bg = float(1.4826 * np.median(np.abs(sample - med_bg)))
    star_thresh = med_bg + max(mad_bg * 2.5, 0.025)

    # 2. Máscara de peso estelar suave (0.0 en fondo, 1.0 en trazas de estrellas)
    if bg_protect:
        star_weight = np.clip((lum - star_thresh) / max(0.04, star_thresh), 0.0, 1.0)
        star_weight = star_weight * star_weight * (3.0 - 2.0 * star_weight)  # Smoothstep cúbico
    else:
        star_weight = np.ones((h, w), dtype=np.float32)

    # Si hay máscara de horizonte, asegurar que el peso sea 0 en el suelo
    sm = None
    if sky_mask is not None:
        sm = sky_mask.astype(np.float32)
        if sm.max() > 1.05:
            sm /= 255.0
        if sm.ndim == 3:
            sm = sm[..., 0]
        if sm.shape[:2] != (h, w):
            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
        star_weight *= sm

    # 3. Conversión a espacio HSV
    hsv = cv2.cvtColor(img_f32, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 4. Recuperación de color en núcleos quemados (Core Color Recovery)
    if core_recovery > 0.0:
        core_mask = (V > 0.75) & (S < 0.35) & (star_weight > 0.15)
        valid_star_color = (star_weight > 0.20) & (S > 0.15) & (V > star_thresh)

        if np.any(core_mask) and np.any(valid_star_color):
            rad_h = np.deg2rad(H)
            u = S * np.cos(rad_h)
            v = S * np.sin(rad_h)

            k_size = max(11, int((min(h, w) // 100) | 1))
            if k_size % 2 == 0:
                k_size += 1

            u_color_src = np.where(valid_star_color, u, 0.0).astype(np.float32)
            v_color_src = np.where(valid_star_color, v, 0.0).astype(np.float32)
            norm_src = valid_star_color.astype(np.float32)

            sigma = float(k_size) / 2.5
            u_diffused = cv2.GaussianBlur(u_color_src, (k_size, k_size), sigmaX=sigma)
            v_diffused = cv2.GaussianBlur(v_color_src, (k_size, k_size), sigmaX=sigma)
            norm_diffused = cv2.GaussianBlur(norm_src, (k_size, k_size), sigmaX=sigma)
            norm_diffused = np.maximum(norm_diffused, 1e-5)

            u_recovered = u_diffused / norm_diffused
            v_recovered = v_diffused / norm_diffused

            recovered_S = np.clip(np.sqrt(u_recovered**2 + v_recovered**2), 0.0, 1.0)
            recovered_H = np.rad2deg(np.arctan2(v_recovered, u_recovered)) % 360.0

            # Solo infundir si hay suficiente soporte de color vecino (norm_diffused > 0.05)
            confidence = np.clip(norm_diffused * 2.0, 0.0, 1.0)
            infuse_factor = np.clip(core_recovery * (core_mask.astype(np.float32)) * confidence, 0.0, 1.0)

            H = np.where(infuse_factor > 0, (1.0 - infuse_factor) * H + infuse_factor * recovered_H, H)
            target_core_S = np.clip(recovered_S * 1.25, 0.15, 0.75)
            S = np.where(infuse_factor > 0, np.maximum(S, target_core_S * infuse_factor), S)

    # 5. Amplificación de Saturación Estelar
    if saturation_boost > 1.0:
        boost_mult = 1.0 + (saturation_boost - 1.0) * star_weight
        S = np.clip(S * boost_mult, 0.0, 1.0)

    # 6. Reconstrucción RGB
    hsv[..., 0] = H
    hsv[..., 1] = S
    hsv[..., 2] = V

    enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if sm is not None:
        sm3 = np.repeat(sm[..., np.newaxis], 3, axis=2)
        out = (enhanced_rgb * sm3) + (img_f32 * (1.0 - sm3))
    else:
        out = enhanced_rgb

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def generate_startrail(
    files: List[str],
    mode: str = "lighten",
    comet_params: Optional[Dict[str, Any]] = None,
    dark_files: Optional[List[str]] = None,
    use_clean_ground: bool = False,
    mask: Optional[np.ndarray] = None,
    ground_mode: str = "average",
    ref_idx: int = 0,
    feather_radius: int = 5,
    suppress_streaks: bool = False,
    streak_sensitivity: float = 0.5,
    min_streak_length: int = 40,
    fill_gaps: bool = False,
    gap_fill_radius: int = 3,
    gap_fill_threshold_factor: float = 3.0,
    timelapse_params: Optional[Dict[str, Any]] = None,
    color_boost_params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Genera una imagen compuesta de trazas de estrellas (Startrails) con streaming eficiente de memoria.

    :param files: Lista de rutas a fotogramas ordenados temporalmente.
    :param mode: "lighten" (máximo clásico) o "comet" (estela decreciente).
    :param comet_params: Diccionario con { "tail_pct": 30, "decay": "linear", "direction": "backward", "min_floor": 0.0 }
    :param dark_files: Lista opcional de tomas dark para sustracción térmica.
    :param use_clean_ground: Si es True, separa cielo y suelo usando la máscara para dejar el suelo limpio sin ruido.
    :param mask: Máscara binaria o en escala de grises (255 / 1.0 = cielo, 0 = suelo).
    :param ground_mode: "average" (media temporal con reducción drástica de ruido) o "ref" (toma de referencia).
    :param ref_idx: Índice del fotograma de referencia para el suelo y encuadre.
    :param feather_radius: Radio en píxeles para el suavizado gaussiano del borde de la máscara.
    :param suppress_streaks: Si es True, detecta y suprime satélites y aviones automáticamente.
    :param streak_sensitivity: Sensibilidad de detección de trazas (0.1 a 1.0).
    :param min_streak_length: Longitud mínima en píxeles del trazo para considerarlo satélite/avión.
    :param fill_gaps: Si es True, conecta de forma continua las estrellas entre fotogramas consecutivos (Gap Filling).
    :param gap_fill_radius: Radio en píxeles del salto temporal a rellenar (1 a 8 px, típicamente 3 o 4 px).
    :param gap_fill_threshold_factor: Factor de umbral adaptativo sobre el ruido para discriminar estrellas.
    :param progress_callback: Función callback (porcentaje: int, mensaje: str).
    :param abort_flag: Función que devuelve True si el usuario solicita cancelar el proceso.
    :return: Imagen final float32 RGB en rango [0.0, 1.0].
    """
    if not files:
        raise ValueError("No se han proporcionado archivos para generar trazas.")

    n_files = len(files)
    ref_idx = max(0, min(n_files - 1, int(ref_idx)))

    if progress_callback:
        progress_callback(2, f"Iniciando procesado de {n_files} fotogramas...")

    # 1. Master Dark si se han especificado tomas oscuras
    master_dark = None
    if dark_files and len(dark_files) > 0:
        if progress_callback:
            progress_callback(5, "Calculando Master Dark...")
        master_dark = build_master_dark(dark_files, progress_callback, abort_flag)

    # 2. Ponderación para efecto cometa
    comet_weights = None
    if mode == "comet":
        cp = comet_params or {}
        tail_pct = float(cp.get("tail_pct", 35.0))
        tail_frames = max(2, int(n_files * (tail_pct / 100.0)))
        decay_type = cp.get("decay", "linear")
        direction = cp.get("direction", "backward")
        min_floor = float(cp.get("min_floor", 0.0))
        comet_weights = compute_comet_weights(n_files, tail_frames, decay_type, direction, min_floor)

    # 3. Preparación de la máscara de horizonte si el suelo limpio está activo o para gap/streaks
    mask_3c = None
    sky_mask_bool = None
    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        if mask_f.ndim == 3:
            mask_f = mask_f[..., 0]

        if use_clean_ground:
            if feather_radius > 0:
                k = max(3, int(feather_radius * 4) | 1)
                mask_feathered = cv2.GaussianBlur(mask_f, (k, k), float(feather_radius))
            else:
                mask_feathered = mask_f

            mask_feathered = np.clip(mask_feathered, 0.0, 1.0)
            mask_3c = np.repeat(mask_feathered[:, :, np.newaxis], 3, axis=2)

        sky_mask_bool = (mask_f > 0.3)

    gap_kernel = None
    if fill_gaps:
        k_size = max(3, int(gap_fill_radius * 2 + 1))
        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # 4. Configuración opcional de Time-Lapse
    timelapse_enabled = False
    output_video_path = None
    fps = 24.0
    res_mode = "1080p"
    save_frames = False
    frames_dir = None

    if timelapse_params and timelapse_params.get("enabled", False):
        timelapse_enabled = True
        output_video_path = timelapse_params.get("output_video_path")
        if not output_video_path:
            first_dir = os.path.dirname(files[0]) if files[0] else os.getcwd()
            if not first_dir:
                first_dir = os.getcwd()
            output_video_path = os.path.join(first_dir, "timelapse_startrails.mp4")
        fps = float(timelapse_params.get("fps", 24.0))
        res_mode = str(timelapse_params.get("resolution", "1080p")).lower()
        save_frames = bool(timelapse_params.get("save_frames", False))
        frames_dir = timelapse_params.get("frames_dir")
        if save_frames and not frames_dir:
            frames_dir = os.path.splitext(output_video_path)[0] + "_frames"

    # 5. Configuración opcional de Realce de Color Estelar
    color_boost_enabled = False
    sat_boost = 1.5
    core_recovery = 0.5
    if color_boost_params and color_boost_params.get("enabled", False):
        color_boost_enabled = True
        sat_boost = float(color_boost_params.get("saturation", 1.5))
        core_recovery = float(color_boost_params.get("core_recovery", 0.5))

    video_writer = None
    actual_video_path = None
    tw, th = None, None
    ref_ground = None

    if timelapse_enabled and use_clean_ground and mask_3c is not None:
        try:
            ref_ground = load_image_as_float32(files[ref_idx])
            if master_dark is not None and master_dark.shape == ref_ground.shape:
                ref_ground = np.maximum(0.0, ref_ground - master_dark)
        except Exception as e:
            print(f"[Startrails] Error al precargar referencia para suelo de timelapse: {e}")
            ref_ground = None

    def _write_timelapse_frame(idx: int):
        nonlocal video_writer, tw, th, actual_video_path
        if not timelapse_enabled or sky_acc is None:
            return

        if video_writer is None:
            h_acc, w_acc = sky_acc.shape[:2]
            tw, th = compute_target_video_dims(w_acc, h_acc, res_mode)
            out_dir = os.path.dirname(os.path.abspath(output_video_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            if save_frames and frames_dir:
                os.makedirs(frames_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (tw, th))
            if not video_writer.isOpened():
                print(f"[Startrails] Advertencia: No se pudo abrir VideoWriter para '{output_video_path}'")
                video_writer = None
                return
            actual_video_path = output_video_path

        if use_clean_ground and mask_3c is not None and ref_ground is not None:
            gh, gw = sky_acc.shape[:2]
            cur_mask = mask_3c
            if cur_mask.shape[:2] != (gh, gw):
                cur_mask = cv2.resize(cur_mask, (gw, gh), interpolation=cv2.INTER_LINEAR)
                if cur_mask.ndim == 2:
                    cur_mask = cur_mask[:, :, np.newaxis]
            cur_ref = ref_ground
            if cur_ref.shape[:2] != (gh, gw):
                cur_ref = cv2.resize(cur_ref, (gw, gh), interpolation=cv2.INTER_LINEAR)
            tl_frame = (sky_acc * cur_mask) + (cur_ref * (1.0 - cur_mask))
        else:
            tl_frame = sky_acc

        if (tw, th) != (tl_frame.shape[1], tl_frame.shape[0]):
            interp = cv2.INTER_AREA if (tw < tl_frame.shape[1] and th < tl_frame.shape[0]) else cv2.INTER_LINEAR
            frame_resized = cv2.resize(tl_frame, (tw, th), interpolation=interp)
        else:
            frame_resized = tl_frame

        if color_boost_enabled:
            frame_resized = enhance_star_trails_color(
                frame_resized,
                saturation_boost=sat_boost,
                core_recovery=core_recovery,
                sky_mask=None,
                bg_protect=True
            )

        bgr_u8 = (np.clip(frame_resized, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_RGB2BGR)

        video_writer.write(bgr_u8)

        if save_frames and frames_dir:
            frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(frame_path, bgr_u8, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 6. Acumulación progresiva streaming
    sky_acc = None
    ground_sum = None
    ref_frame = None

    try:
        if suppress_streaks:
            # Búfer deslizante inteligente O(1) con 2 referencias simultáneas para cada fotograma
            loaded_frames: Dict[int, np.ndarray] = {}

            def get_frame(idx: int) -> np.ndarray:
                if abort_flag and abort_flag():
                    raise InterruptedError("Procesamiento de trazas cancelado por el usuario.")
                if idx not in loaded_frames:
                    img = load_image_as_float32(files[idx])
                    if master_dark is not None and master_dark.shape == img.shape:
                        img = np.maximum(0.0, img - master_dark)
                    loaded_frames[idx] = img
                return loaded_frames[idx]

            prev_clean: Optional[np.ndarray] = None
            prev_weight: float = 1.0
            prev_dilated_stars: Optional[np.ndarray] = None

            for i in range(n_files):
                if abort_flag and abort_flag():
                    raise InterruptedError("Procesamiento de trazas cancelado por el usuario.")

                fname = os.path.basename(files[i])
                pct = int(10 + (i / n_files) * 80)

                curr = get_frame(i)
                h, w = curr.shape[:2]

                # Ajustar dimensiones de la máscara una sola vez si difiere en resolución
                if sky_mask_bool is not None and sky_mask_bool.shape[:2] != (h, w):
                    sm_res = cv2.resize(sky_mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    sky_mask_bool = (sm_res > 0)

                if i == 0 and n_files >= 3:
                    ref_A = get_frame(1)
                    ref_B = get_frame(2)
                elif i == n_files - 1 and n_files >= 3:
                    ref_A = get_frame(n_files - 3)
                    ref_B = get_frame(n_files - 2)
                elif n_files >= 2:
                    ref_A = get_frame(i - 1 if i > 0 else 1)
                    ref_B = get_frame(i + 1 if i < n_files - 1 else 0)
                else:
                    ref_A = None
                    ref_B = None

                # Detección y supresión de satélites / aviones
                cleaned_img, num_streaks = clean_frame_streaks(
                    curr, ref_A, ref_B,
                    sky_mask=sky_mask_bool,
                    sensitivity=streak_sensitivity
                )

                # Liberar memoria de fotogramas pasados que ya no se utilizarán
                keys_to_del = [k for k in list(loaded_frames.keys()) if k < i - 1]
                for k in keys_to_del:
                    del loaded_frames[k]

                tl_tag = " [Time-Lapse]" if timelapse_enabled else ""
                if num_streaks > 0 and progress_callback:
                    progress_callback(pct, f"Toma ({i+1}/{n_files}): suprimidas {num_streaks} trazas de satélite/avión en {fname}{tl_tag}")
                elif progress_callback:
                    progress_callback(pct, f"Integrando toma ({i+1}/{n_files}): {fname}{tl_tag}")

                if i == ref_idx:
                    ref_frame = cleaned_img.copy()

                curr_weight = comet_weights[i] if (mode == "comet" and comet_weights is not None) else 1.0
                if mode == "comet" and comet_weights is not None:
                    frame_sky = cleaned_img * curr_weight
                else:
                    frame_sky = cleaned_img

                if sky_acc is None:
                    sky_acc = frame_sky.copy()
                else:
                    if fill_gaps:
                        curr_dilated_stars = extract_dilated_stars(
                            cleaned_img,
                            radius=gap_fill_radius,
                            threshold_factor=gap_fill_threshold_factor,
                            sky_mask_bool=sky_mask_bool,
                            kernel=gap_kernel
                        )
                        if prev_dilated_stars is not None:
                            bridge = np.minimum(prev_dilated_stars, curr_dilated_stars)
                            bridge[bridge < 0.005] = 0.0
                            if mode == "comet" and comet_weights is not None:
                                w_bridge = 0.5 * (prev_weight + curr_weight)
                                bridge = bridge * w_bridge
                            np.maximum(sky_acc, bridge, out=sky_acc)
                        prev_dilated_stars = curr_dilated_stars

                    np.maximum(sky_acc, frame_sky, out=sky_acc)

                if fill_gaps and prev_dilated_stars is None:
                    prev_dilated_stars = extract_dilated_stars(
                        cleaned_img,
                        radius=gap_fill_radius,
                        threshold_factor=gap_fill_threshold_factor,
                        sky_mask_bool=sky_mask_bool,
                        kernel=gap_kernel
                    )

                prev_clean = cleaned_img
                prev_weight = curr_weight

                if timelapse_enabled:
                    _write_timelapse_frame(i)

                if use_clean_ground and mask_3c is not None and ground_mode == "average":
                    if ground_sum is None:
                        ground_sum = cleaned_img.astype(np.float64)
                    else:
                        ground_sum += cleaned_img

            loaded_frames.clear()

        else:
            # Acumulación directa estándar con prefetching asíncrono y arrastre de estrellas dilatadas
            prev_img: Optional[np.ndarray] = None
            prev_weight: float = 1.0
            prev_dilated_stars: Optional[np.ndarray] = None

            prefetch_exec = ThreadPoolExecutor(max_workers=1)
            try:
                future_next = prefetch_exec.submit(load_image_as_float32, files[0])

                for i, path in enumerate(files):
                    if abort_flag and abort_flag():
                        raise InterruptedError("Procesamiento de trazas cancelado por el usuario.")

                    fname = os.path.basename(files[i])
                    pct = int(10 + (i / n_files) * 80)
                    tl_tag = " [Time-Lapse]" if timelapse_enabled else ""
                    if progress_callback:
                        progress_callback(pct, f"Integrando toma ({i+1}/{n_files}): {fname}{tl_tag}")

                    img = future_next.result()

                    # Lanzar en segundo plano la precarga de la siguiente toma
                    if i + 1 < n_files:
                        future_next = prefetch_exec.submit(load_image_as_float32, files[i + 1])
                    else:
                        future_next = None

                    if master_dark is not None and master_dark.shape == img.shape:
                        img = np.maximum(0.0, img - master_dark)

                    h, w = img.shape[:2]
                    if sky_mask_bool is not None and sky_mask_bool.shape[:2] != (h, w):
                        sm_res = cv2.resize(sky_mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        sky_mask_bool = (sm_res > 0)

                    if i == ref_idx:
                        ref_frame = img.copy()

                    curr_weight = comet_weights[i] if (mode == "comet" and comet_weights is not None) else 1.0
                    if mode == "comet" and comet_weights is not None:
                        frame_sky = img * curr_weight
                    else:
                        frame_sky = img

                    if sky_acc is None:
                        sky_acc = frame_sky.copy()
                    else:
                        if fill_gaps:
                            curr_dilated_stars = extract_dilated_stars(
                                img,
                                radius=gap_fill_radius,
                                threshold_factor=gap_fill_threshold_factor,
                                sky_mask_bool=sky_mask_bool,
                                kernel=gap_kernel
                            )
                            if prev_dilated_stars is not None:
                                bridge = np.minimum(prev_dilated_stars, curr_dilated_stars)
                                bridge[bridge < 0.005] = 0.0
                                if mode == "comet" and comet_weights is not None:
                                    w_bridge = 0.5 * (prev_weight + curr_weight)
                                    bridge = bridge * w_bridge
                                np.maximum(sky_acc, bridge, out=sky_acc)
                            prev_dilated_stars = curr_dilated_stars

                        np.maximum(sky_acc, frame_sky, out=sky_acc)

                    if fill_gaps and prev_dilated_stars is None:
                        prev_dilated_stars = extract_dilated_stars(
                            img,
                            radius=gap_fill_radius,
                            threshold_factor=gap_fill_threshold_factor,
                            sky_mask_bool=sky_mask_bool,
                            kernel=gap_kernel
                        )

                    prev_img = img
                    prev_weight = curr_weight

                    if timelapse_enabled:
                        _write_timelapse_frame(i)

                    if use_clean_ground and mask_3c is not None and ground_mode == "average":
                        if ground_sum is None:
                            ground_sum = img.astype(np.float64)
                        else:
                            ground_sum += img
            finally:
                prefetch_exec.shutdown(wait=False, cancel_futures=True)

    finally:
        if video_writer is not None:
            video_writer.release()

    if sky_acc is None:
        raise RuntimeError("No se pudo acumular ninguna imagen válida.")

    # 7. Realce y preservación de color estelar (Star Color Boost)
    if color_boost_enabled:
        if progress_callback:
            progress_callback(93, "Realzando y preservando saturación de trazas estelares...")
        sky_acc = enhance_star_trails_color(
            sky_acc,
            saturation_boost=sat_boost,
            core_recovery=core_recovery,
            sky_mask=sky_mask_bool,
            bg_protect=True
        )

    # 8. Fusión con suelo limpio o retorno directo
    if use_clean_ground and mask_3c is not None:
        if progress_callback:
            progress_callback(92, "Fusionando cielo de trazas con suelo anti-ruido...")

        # Verificar dimensiones de la máscara respecto a la imagen
        h, w = sky_acc.shape[:2]
        if mask_3c.shape[:2] != (h, w):
            mask_3c = cv2.resize(mask_3c, (w, h), interpolation=cv2.INTER_LINEAR)
            if mask_3c.ndim == 2:
                mask_3c = mask_3c[:, :, np.newaxis]

        if ground_mode == "average" and ground_sum is not None:
            ground_clean = (ground_sum / float(n_files)).astype(np.float32)
        else:
            # Modo toma de referencia única
            if ref_frame is not None:
                ground_clean = ref_frame
            else:
                ground_clean = load_image_as_float32(files[ref_idx])
                if master_dark is not None and master_dark.shape == ground_clean.shape:
                    ground_clean = np.maximum(0.0, ground_clean - master_dark)

        # Fusión alfa ponderada
        final_img = (sky_acc * mask_3c) + (ground_clean * (1.0 - mask_3c))
    else:
        final_img = sky_acc

    final_img = np.clip(final_img, 0.0, 1.0).astype(np.float32)

    if actual_video_path and os.path.exists(actual_video_path):
        if os.path.getsize(actual_video_path) == 0:
            actual_video_path = None
    else:
        actual_video_path = None

    if progress_callback:
        msg = f"Composición de trazas finalizada con éxito ({n_files} tomas)."
        if actual_video_path:
            msg += f" Vídeo time-lapse exportado: {os.path.basename(actual_video_path)}"
        progress_callback(100, msg)

    return final_img, actual_video_path
