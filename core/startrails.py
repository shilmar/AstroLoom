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
import time
import datetime
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from core.stacking import load_image_as_float32, scan_raw_batch_metadata

logger = logging.getLogger("AstroLoom.Startrails")

# Aceleración nativa en Rust vía PyO3 / Rayon
try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


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


def build_master_flat(
    flat_files: List[str],
    master_dark: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None
) -> Optional[np.ndarray]:
    """Calcula el Master Flat promediado y normalizado a media 1.0 por canal a partir de tomas de campo plano."""
    if not flat_files:
        return None

    n = len(flat_files)
    flat_acc = None
    for i, path in enumerate(flat_files):
        if abort_flag and abort_flag():
            raise InterruptedError("Procesamiento cancelado por el usuario.")
        if progress_callback:
            progress_callback(int(5 + (i / n) * 15), f"Cargando flat ({i+1}/{n}): {os.path.basename(path)}")
        f_img = load_image_as_float32(path)
        if flat_acc is None:
            flat_acc = f_img.astype(np.float64)
        else:
            flat_acc += f_img

    master_flat = (flat_acc / float(n)).astype(np.float32)
    if master_dark is not None and master_dark.shape == master_flat.shape:
        master_flat = np.maximum(0.0, master_flat - master_dark)

    # Normalizar cada canal a media 1.0 para mantener el balance cromático
    if master_flat.ndim == 3:
        for ch in range(master_flat.shape[2]):
            m_val = float(np.mean(master_flat[:, :, ch]))
            if m_val > 1e-5:
                master_flat[:, :, ch] /= m_val
            else:
                master_flat[:, :, ch].fill(1.0)
    else:
        m_val = float(np.mean(master_flat))
        if m_val > 1e-5:
            master_flat /= m_val
        else:
            master_flat.fill(1.0)

    return master_flat


def extract_shot_exif(filepath: str, meta_batch_item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extrae tiempo de exposición, ISO, apertura (FNumber) y fecha/hora de un archivo fotográfico, RAW o FITS.
    Retorna un diccionario con valores numéricos y cadenas formateadas listas para la UI.
    """
    exp: Optional[float] = None
    iso: Optional[int] = None
    aperture: Optional[float] = None
    timestamp: Optional[float] = None
    dt_obj: Optional[datetime.datetime] = None

    if meta_batch_item:
        exp = meta_batch_item.get("exposure_time")
        iso = meta_batch_item.get("iso")
        ts = meta_batch_item.get("timestamp")
        if ts is not None and isinstance(ts, (int, float)) and ts > 0:
            timestamp = float(ts)
            try:
                dt_obj = datetime.datetime.fromtimestamp(timestamp)
            except Exception:
                pass

    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".nef", ".cr2", ".cr3", ".arw", ".dng", ".raw", ".orf", ".rw2"]:
        try:
            import rawpy
            with rawpy.imread(filepath) as raw:
                other = getattr(raw, "other", None)
                if other:
                    if exp is None:
                        s = getattr(other, "shutter_speed", None) or getattr(other, "shutter", None)
                        if s is not None and float(s) > 0:
                            exp = float(s)
                    if iso is None:
                        i = getattr(other, "iso_speed", None) or getattr(other, "iso", None)
                        if i is not None and float(i) > 0:
                            iso = int(round(float(i)))
                    if aperture is None:
                        a = getattr(other, "aperture", None)
                        if a is not None and float(a) > 0:
                            aperture = float(a)
                    if dt_obj is None:
                        t = getattr(other, "timestamp", None)
                        if t is not None and float(t) > 0:
                            timestamp = float(t)
                            try:
                                dt_obj = datetime.datetime.fromtimestamp(timestamp)
                            except Exception:
                                pass
        except Exception:
            pass

    if ext in [".fits", ".fit", ".fts"]:
        try:
            from astropy.io import fits
            with fits.open(filepath) as hdul:
                hdr = hdul[0].header
                if exp is None:
                    exp = float(hdr.get("EXPTIME", hdr.get("EXPOSURE", 0.0))) or None
                if iso is None and "ISO" in hdr:
                    try:
                        iso = int(hdr["ISO"])
                    except Exception:
                        pass
                if dt_obj is None:
                    d_obs = hdr.get("DATE-OBS") or hdr.get("DATE")
                    if d_obs:
                        try:
                            cleaned = str(d_obs).replace("T", " ").split(".")[0]
                            dt_obj = datetime.datetime.fromisoformat(cleaned)
                            timestamp = dt_obj.timestamp()
                        except Exception:
                            pass
        except Exception:
            pass

    if exp is None or iso is None or aperture is None or dt_obj is None:
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                exif_raw = img._getexif() if hasattr(img, "_getexif") else None
                if exif_raw:
                    if exp is None and 33434 in exif_raw:
                        exp = float(exif_raw[33434])
                    if iso is None and 34855 in exif_raw:
                        v = exif_raw[34855]
                        iso = int(v[0] if isinstance(v, (list, tuple)) else v)
                    if aperture is None and 33437 in exif_raw:
                        aperture = float(exif_raw[33437])
                    if dt_obj is None:
                        for tag_id in (36867, 36868, 306):
                            if tag_id in exif_raw and exif_raw[tag_id]:
                                try:
                                    s_val = str(exif_raw[tag_id]).strip()
                                    if len(s_val) >= 19 and s_val[4] == ":" and s_val[7] == ":":
                                        parts = s_val.split(" ")
                                        date_part = parts[0].replace(":", "-")
                                        s_val = f"{date_part} {parts[1]}"
                                    dt_obj = datetime.datetime.fromisoformat(s_val[:19])
                                    timestamp = dt_obj.timestamp()
                                    break
                                except Exception:
                                    pass
        except Exception:
            pass

    # Fallback final a fecha de modificación del fichero
    if dt_obj is None and os.path.exists(filepath):
        try:
            mtime = os.path.getmtime(filepath)
            if mtime > 0:
                timestamp = float(mtime)
                dt_obj = datetime.datetime.fromtimestamp(timestamp)
        except Exception:
            pass

    if exp is not None and exp > 0:
        if exp >= 1.0:
            exp_str = f"{exp:.1f}s" if exp % 1 != 0 else f"{int(exp)}s"
        else:
            exp_str = f"1/{int(round(1.0 / exp))}s"
    else:
        exp_str = "—"

    iso_str = f"{int(iso)}" if (iso is not None and iso > 0) else "—"
    aperture_str = f"f/{aperture:.1f}" if (aperture is not None and aperture > 0) else "—"
    time_str = dt_obj.strftime("%H:%M:%S") if dt_obj else "—"
    datetime_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S") if dt_obj else "—"

    return {
        "exp": exp,
        "exp_str": exp_str,
        "iso": iso,
        "iso_str": iso_str,
        "aperture": aperture,
        "aperture_str": aperture_str,
        "timestamp": timestamp,
        "time_str": time_str,
        "datetime_str": datetime_str
    }


def clean_frame_streaks(
    current_img: np.ndarray,
    prev_img: Optional[np.ndarray],
    next_img: Optional[np.ndarray],
    sky_mask: Optional[np.ndarray] = None,
    sensitivity: float = 0.5,
    method: str = "hough",
    min_line_length: Optional[int] = None,
    line_thickness: int = 3
) -> Tuple[np.ndarray, int]:
    """
    Detecta y suprime trazas lineales transitorias (satélites artificiales y aviones) en el fotograma actual
    mediante Transformada de Hough probabilística (cv2.HoughLinesP), análisis morfológico o modo híbrido,
    sustituyendo los píxeles afectados por el fondo limpio interpolado de los fotogramas vecinos.

    :param current_img: Fotograma actual float32 RGB [0, 1].
    :param prev_img: Fotograma anterior (o None si es el primero).
    :param next_img: Fotograma posterior (o None si es el último).
    :param sky_mask: Máscara binaria o float opcional (1.0 = cielo, 0.0 = suelo) para excluir vegetación y suelo móvil.
    :param sensitivity: Sensibilidad de detección (0.1 a 1.0, por defecto 0.5).
    :param method: 'hough' (Transformada de Hough probabilística), 'morphological' (componentes conectadas y elongación)
                   o 'hybrid' (unión de ambos métodos).
    :param min_line_length: Longitud mínima en píxeles del trazo (si es None, se calcula adaptativamente).
    :param line_thickness: Grosor en píxeles para trazar la línea antes de la dilatación del halo (3 por defecto).
    :return: (imagen_limpia, número_de_trazas_eliminadas)
    """
    if prev_img is None and next_img is None:
        return current_img, 0

    h, w = current_img.shape[:2]
    curr_lum = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY) if current_img.ndim == 3 else current_img

    # 1. Comparación diferencial temporal en luminancia monocanal (ahorro drástico de RAM y tiempo)
    if prev_img is not None and next_img is not None:
        # shortcut: Comparación diferencial temporal con fotogramas vecinos (k-1, k+1); actualizar a estimación de vector de velocidad si se integran trazas con seguimiento sideral.
        prev_lum = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY) if prev_img.ndim == 3 else prev_img
        next_lum = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY) if next_img.ndim == 3 else next_img
        ref_bg_lum = np.minimum(prev_lum, next_lum)
    elif prev_img is not None:
        ref_bg_lum = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY) if prev_img.ndim == 3 else prev_img
    else:
        ref_bg_lum = cv2.cvtColor(next_img, cv2.COLOR_RGB2GRAY) if next_img.ndim == 3 else next_img

    diff_lum = np.maximum(0.0, curr_lum - ref_bg_lum)

    # 2. Si se proporciona máscara de horizonte, enfocar exclusivamente en el cielo
    # Ignora el movimiento de copas de árboles, follaje o hierba que se muevan con el viento
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

    # 3. Escala al 50% para análisis rápido y libre de artefactos de interpolación fraccional
    use_half_scale = (max(h, w) > 800)
    scale = 0.5 if use_half_scale else 1.0

    if use_half_scale:
        pw, ph = w // 2, h // 2
        diff_proc = cv2.resize(diff_lum_analysis, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        pw, ph = w, h
        diff_proc = diff_lum_analysis

    # 4. Umbral adaptativo sobre el ruido del cielo
    sample_vals = diff_proc[::8, ::8]
    bg_val = float(np.median(sample_vals))
    mad_noise = float(1.4826 * np.median(np.abs(sample_vals - bg_val)))
    sens_f = float(np.clip(sensitivity, 0.1, 1.0))
    sigma_factor = max(1.8, 5.0 - (sens_f * 3.5))
    thresh_val = bg_val + max(mad_noise * sigma_factor, 0.003)

    binary = (diff_proc > thresh_val).astype(np.uint8) * 255

    # 5. Filtrado vectorial rápido de ruido puntual y estrellas aisladas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return current_img, 0

    cw = stats[1:, cv2.CC_STAT_WIDTH]
    ch = stats[1:, cv2.CC_STAT_HEIGHT]
    min_diag_sq = 9 if use_half_scale else 16
    keep_indices = np.where((cw * cw + ch * ch) >= min_diag_sq)[0]
    if len(keep_indices) == 0:
        return current_img, 0

    lut = np.zeros(num_labels, dtype=np.uint8)
    lut[keep_indices + 1] = 255
    clean_bin = lut[labels]

    if not np.any(clean_bin):
        return current_img, 0

    # 6. Detección de trazas lineales por Hough y/o Morfología
    min_dim = min(ph, pw)
    mask_hough = np.zeros((h, w), dtype=np.uint8)
    hough_streak_count = 0

    if method in ("hough", "hybrid"):
        if min_line_length is not None:
            calc_min_len = max(8, min(int(min_line_length * scale), int(min_dim * 0.5)))
        else:
            base_len = 60.0 - sens_f * 25.0
            calc_min_len = max(10, min(int(base_len * scale), int(min_dim * 0.35)))
        hough_thresh = max(8, int(calc_min_len * 0.6))
        max_line_gap = max(4, int((12 + sens_f * 16) * scale))

        lines = cv2.HoughLinesP(
            clean_bin,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_thresh,
            minLineLength=calc_min_len,
            maxLineGap=max_line_gap
        )
        if lines is not None and len(lines) > 0:
            thick = max(2, int(line_thickness * (2 if use_half_scale else 1)))
            for line in lines:
                x1, y1, x2, y2 = line.ravel()[:4]
                if use_half_scale:
                    fx1, fy1 = int(x1 * 2), int(y1 * 2)
                    fx2, fy2 = int(x2 * 2), int(y2 * 2)
                else:
                    fx1, fy1, fx2, fy2 = int(x1), int(y1), int(x2), int(y2)
                cv2.line(mask_hough, (fx1, fy1), (fx2, fy2), 255, thickness=thick)
            hough_streak_count = len(lines)

    mask_morph = np.zeros((h, w), dtype=np.uint8)
    morph_streak_count = 0

    if method in ("morphological", "hybrid"):
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(clean_bin, cv2.MORPH_CLOSE, k_close)
        num_labels_m, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels_m > 1:
            min_length_m = max(6.0, (20.0 - (sens_f * 15.0)) * scale)
            min_elongation_m = max(2.0, 3.2 - (sens_f * 1.5))
            for lbl in range(1, num_labels_m):
                cw_m = stats_m[lbl, cv2.CC_STAT_WIDTH]
                ch_m = stats_m[lbl, cv2.CC_STAT_HEIGHT]
                diag_m = np.sqrt(float(cw_m * cw_m + ch_m * ch_m))
                if diag_m >= min_length_m:
                    x = stats_m[lbl, cv2.CC_STAT_LEFT]
                    y = stats_m[lbl, cv2.CC_STAT_TOP]
                    patch_lbl = labels_m[y:y+ch_m, x:x+cw_m]
                    pts = np.argwhere(patch_lbl == lbl)
                    if len(pts) >= 4:
                        pts_xy = np.fliplr(pts).reshape(-1, 1, 2)
                        rect = cv2.minAreaRect(pts_xy)
                        rw, rh = rect[1]
                        length = max(rw, rh)
                        width = min(rw, rh)
                        elongation = length / max(width, 0.5)
                        if elongation >= min_elongation_m and length >= min_length_m:
                            if use_half_scale:
                                box = cv2.boxPoints(rect)
                                box_full = (box * 2).astype(np.int32)
                                cv2.fillPoly(mask_morph, [box_full], 255)
                            else:
                                mask_morph[y:y+ch_m, x:x+cw_m][patch_lbl == lbl] = 255
                            morph_streak_count += 1

    if method == "hough":
        final_mask = mask_hough
    elif method == "morphological":
        final_mask = mask_morph
    else:  # hybrid
        final_mask = np.bitwise_or(mask_hough, mask_morph)

    if not np.any(final_mask):
        return current_img, 0

    # Contar componentes conexas independientes para métricas y logs limpios
    num_labels_final, _, _, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    found_streaks = max(1, num_labels_final - 1)

    # 7. Dilatar para cubrir el halo óptico difuso (bloom)
    k_size = (7, 7) if scale < 1.0 else (5, 5)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
    streak_mask_dilated = cv2.dilate(final_mask, k_dilate)

    # 8. Inpainting / Sustitución de los píxeles de la traza por el fondo limpio en RGB
    if prev_img is not None and next_img is not None:
        ref_bg = np.minimum(prev_img, next_img)
    elif prev_img is not None:
        ref_bg = prev_img
    else:
        ref_bg = next_img

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

    if HAS_RUST and hasattr(astroloom_rs, "recover_star_trail_colors_rs"):
        m2d = None
        if sky_mask is not None:
            sm = sky_mask.astype(np.float32)
            if sm.max() > 1.05:
                sm /= 255.0
            if sm.ndim == 3:
                sm = sm[..., 0]
            if sm.shape[:2] != img_rgb.shape[:2]:
                sm = cv2.resize(sm, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            m2d = np.ascontiguousarray(sm, dtype=np.float32)

        return astroloom_rs.recover_star_trail_colors_rs(
            np.ascontiguousarray(np.clip(img_rgb, 0.0, 1.0), dtype=np.float32),
            saturation_boost=float(saturation_boost),
            core_recovery=float(core_recovery),
            sky_mask=m2d,
            bg_protect=bool(bg_protect)
        )

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
    flat_files: Optional[List[str]] = None,
    use_clean_ground: bool = False,
    mask: Optional[np.ndarray] = None,
    ground_mode: str = "average",
    ref_idx: int = 0,
    feather_radius: int = 5,
    suppress_streaks: bool = False,
    streak_sensitivity: float = 0.5,
    min_streak_length: int = 40,
    streak_method: str = "hough",
    fill_gaps: bool = False,
    gap_fill_radius: int = 3,
    gap_fill_threshold_factor: float = 3.0,
    timelapse_params: Optional[Dict[str, Any]] = None,
    color_boost_params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None,
    return_metrics: bool = False
) -> Tuple[Any, ...]:
    """
    Genera una imagen compuesta de trazas de estrellas (Startrails) con streaming eficiente de memoria.

    :param files: Lista de rutas a fotogramas ordenados temporalmente.
    :param mode: "lighten" (máximo clásico) o "comet" (estela decreciente).
    :param comet_params: Diccionario con { "tail_pct": 30, "decay": "linear", "direction": "backward", "min_floor": 0.0 }
    :param dark_files: Lista opcional de tomas dark para sustracción térmica.
    :param flat_files: Lista opcional de tomas flat para corrección de viñeteo y motas de polvo.
    :param use_clean_ground: Si es True, separa cielo y suelo usando la máscara para dejar el suelo limpio sin ruido.
    :param mask: Máscara binaria o en escala de grises (255 / 1.0 = cielo, 0 = suelo).
    :param ground_mode: "average" (media temporal con reducción drástica de ruido) o "ref" (toma de referencia).
    :param ref_idx: Índice del fotograma de referencia para el suelo y encuadre.
    :param feather_radius: Radio en píxeles para el suavizado gaussiano del borde de la máscara.
    :param suppress_streaks: Si es True, detecta y suprime satélites y aviones automáticamente.
    :param streak_sensitivity: Sensibilidad de detección de trazas (0.1 a 1.0).
    :param min_streak_length: Longitud mínima en píxeles del trazo para considerarlo satélite/avión.
    :param streak_method: Algoritmo de purga ("hough" para Transformada de Hough probabilística, "morphological" o "hybrid").
    :param fill_gaps: Si es True, conecta de forma continua las estrellas entre fotogramas consecutivos (Gap Filling).
    :param gap_fill_radius: Radio en píxeles del salto temporal a rellenar (1 a 8 px, típicamente 3 o 4 px).
    :param gap_fill_threshold_factor: Factor de umbral adaptativo sobre el ruido para discriminar estrellas.
    :param progress_callback: Función callback (porcentaje: int, mensaje: str).
    :param abort_flag: Función que devuelve True si el usuario solicita cancelar el proceso.
    :return: Imagen final float32 RGB en rango [0.0, 1.0].
    """
    if not files:
        raise ValueError("No se han proporcionado archivos para generar trazas.")

    t_start = time.perf_counter()
    n_files = len(files)

    # Validación concurrente multihilo de cabeceras de lote en Rust (Fase 17)
    t0_meta = time.perf_counter()
    meta_batch = scan_raw_batch_metadata(files)
    invalid = [f for f in files if meta_batch.get(f) and not meta_batch[f].get("is_valid", True)]
    if invalid:
        files = [f for f in files if f not in invalid]
        if not files:
            raise ValueError(f"Ninguno de los archivos proporcionados es válido o legible (fallaron {len(invalid)} tomas).")
        n_files = len(files)
    t_meta = time.perf_counter() - t0_meta
    logger.info(f"[STARTRAILS] {n_files} tomas validadas en {t_meta:.3f}s")

    ref_idx = max(0, min(n_files - 1, int(ref_idx)))

    if progress_callback:
        progress_callback(2, f"Iniciando procesado de {n_files} fotogramas...")

    # 1. Master Dark y Master Flat si se han especificado tomas de calibración
    master_dark = None
    t_dark = 0.0
    if dark_files and len(dark_files) > 0:
        if progress_callback:
            progress_callback(4, "Calculando Master Dark...")
        t0_dark = time.perf_counter()
        master_dark = build_master_dark(dark_files, progress_callback, abort_flag)
        t_dark = time.perf_counter() - t0_dark
        logger.info(f"[STARTRAILS] Master Dark completado en {t_dark:.2f}s ({len(dark_files)} tomas)")

    master_flat = None
    t_flat = 0.0
    if flat_files and len(flat_files) > 0:
        if progress_callback:
            progress_callback(7, "Calculando Master Flat...")
        t0_flat = time.perf_counter()
        master_flat = build_master_flat(flat_files, master_dark=master_dark, progress_callback=progress_callback, abort_flag=abort_flag)
        t_flat = time.perf_counter() - t0_flat
        logger.info(f"[STARTRAILS] Master Flat completado en {t_flat:.2f}s ({len(flat_files)} tomas)")

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
            if master_flat is not None and master_flat.shape == ref_ground.shape:
                safe_flat = np.where(master_flat > 1e-4, master_flat, 1.0)
                ref_ground = np.clip(ref_ground / safe_flat, 0.0, 1.0)
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

    # 6. Acumulación progresiva streaming con precarga concurrente multihilo
    sky_acc = None
    ground_sum = None
    ref_frame = None

    t_io = 0.0
    t_streak_detection = 0.0
    total_streaks = 0
    t_gap_fill = 0.0
    t_accum = 0.0
    t_timelapse = 0.0
    t_ground = 0.0

    # Algoritmo de demosaicing optimizado para astrofotografía (PPG rápido sin pérdida)
    demosaic_algo = None
    try:
        import rawpy
        demosaic_algo = rawpy.DemosaicAlgorithm.PPG
    except Exception:
        demosaic_algo = None

    # Pipeline de precarga multihilo desacoplado (escalado multi-core en C++/LibRaw)
    max_workers = min(6, max(2, (os.cpu_count() or 4) - 1))
    prefetch_window = max_workers + 1
    prefetch_exec = ThreadPoolExecutor(max_workers=max_workers)

    futures: Dict[int, Any] = {}
    loaded_frames: Dict[int, np.ndarray] = {}

    def _worker_load_frame(idx: int) -> np.ndarray:
        img = load_image_as_float32(files[idx], demosaic_algorithm=demosaic_algo)
        if master_dark is not None and master_dark.shape == img.shape:
            img = np.maximum(0.0, img - master_dark)
        if master_flat is not None and master_flat.shape == img.shape:
            safe_flat = np.where(master_flat > 1e-4, master_flat, 1.0)
            img = np.clip(img / safe_flat, 0.0, 1.0)
        return img

    def _enqueue(idx: int):
        if idx < n_files and idx not in futures and idx not in loaded_frames:
            futures[idx] = prefetch_exec.submit(_worker_load_frame, idx)

    # Encolar la ventana inicial de fotogramas
    for j in range(min(n_files, prefetch_window)):
        _enqueue(j)

    def get_frame(idx: int) -> np.ndarray:
        nonlocal t_io
        if abort_flag and abort_flag():
            raise InterruptedError("Procesamiento de trazas cancelado por el usuario.")
        if idx not in loaded_frames:
            if idx not in futures:
                _enqueue(idx)
            t0_wait = time.perf_counter()
            loaded_frames[idx] = futures[idx].result()
            t_io += time.perf_counter() - t0_wait
            del futures[idx]

            # Mantener la ventana de precarga llena hacia adelante
            furthest = max(futures.keys()) if futures else idx
            while len(futures) < prefetch_window and (furthest + 1) < n_files:
                furthest += 1
                _enqueue(furthest)

        return loaded_frames[idx]

    try:
        if suppress_streaks:
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

                # Detección y supresión de satélites / aviones (Hough / Morfológico)
                t0_detect = time.perf_counter()
                cleaned_img, num_streaks = clean_frame_streaks(
                    curr, ref_A, ref_B,
                    sky_mask=sky_mask_bool,
                    sensitivity=streak_sensitivity,
                    method=streak_method,
                    min_line_length=min_streak_length
                )
                dt_detect = time.perf_counter() - t0_detect
                t_streak_detection += dt_detect
                total_streaks += num_streaks

                # Liberar memoria de fotogramas pasados que ya no se utilizarán
                keys_to_del = [k for k in list(loaded_frames.keys()) if k < i - 1]
                for k in keys_to_del:
                    del loaded_frames[k]

                tl_tag = " [Time-Lapse]" if timelapse_enabled else ""
                if num_streaks > 0:
                    logger.info(
                        f"[DETECCIÓN TRAZAS] Toma {i+1}/{n_files} ({fname}): "
                        f"{num_streaks} trazas suprimidas en {dt_detect*1000:.1f}ms"
                    )
                    if progress_callback:
                        progress_callback(
                            pct,
                            f"Toma ({i+1}/{n_files}): suprimidas {num_streaks} trazas ({dt_detect*1000:.1f}ms) en {fname}{tl_tag}"
                        )
                elif progress_callback:
                    progress_callback(pct, f"Integrando toma ({i+1}/{n_files}): {fname}{tl_tag}")

                if i == ref_idx:
                    ref_frame = cleaned_img.copy()

                curr_weight = comet_weights[i] if (mode == "comet" and comet_weights is not None) else 1.0
                if mode == "comet" and comet_weights is not None:
                    frame_sky = cleaned_img * curr_weight
                else:
                    frame_sky = cleaned_img

                t0_step_acc = time.perf_counter()
                if sky_acc is None:
                    sky_acc = frame_sky.copy()
                elif HAS_RUST and hasattr(astroloom_rs, "accumulate_startrail_step_rs"):
                    curr_dilated_stars = None
                    if fill_gaps:
                        t0_gap = time.perf_counter()
                        curr_dilated_stars = extract_dilated_stars(
                            cleaned_img,
                            radius=gap_fill_radius,
                            threshold_factor=gap_fill_threshold_factor,
                            sky_mask_bool=sky_mask_bool,
                            kernel=gap_kernel
                        )
                        t_gap_fill += time.perf_counter() - t0_gap
                    w_bridge = (0.5 * (prev_weight + curr_weight)) if (mode == "comet" and comet_weights is not None) else 1.0
                    astroloom_rs.accumulate_startrail_step_rs(
                        sky_acc,
                        np.ascontiguousarray(cleaned_img, dtype=np.float32),
                        weight=float(curr_weight),
                        prev_dilated=prev_dilated_stars,
                        curr_dilated=curr_dilated_stars,
                        w_bridge=float(w_bridge)
                    )
                    if fill_gaps:
                        prev_dilated_stars = curr_dilated_stars
                else:
                    if fill_gaps:
                        t0_gap = time.perf_counter()
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
                        t_gap_fill += time.perf_counter() - t0_gap

                    np.maximum(sky_acc, frame_sky, out=sky_acc)

                if fill_gaps and prev_dilated_stars is None:
                    t0_gap = time.perf_counter()
                    prev_dilated_stars = extract_dilated_stars(
                        cleaned_img,
                        radius=gap_fill_radius,
                        threshold_factor=gap_fill_threshold_factor,
                        sky_mask_bool=sky_mask_bool,
                        kernel=gap_kernel
                    )
                    t_gap_fill += time.perf_counter() - t0_gap

                t_accum += time.perf_counter() - t0_step_acc

                prev_clean = cleaned_img
                prev_weight = curr_weight

                if timelapse_enabled:
                    t0_tl = time.perf_counter()
                    _write_timelapse_frame(i)
                    t_timelapse += time.perf_counter() - t0_tl

                if use_clean_ground and mask_3c is not None and ground_mode == "average":
                    t0_gnd = time.perf_counter()
                    if ground_sum is None:
                        ground_sum = cleaned_img.astype(np.float64)
                    else:
                        ground_sum += cleaned_img
                    t_ground += time.perf_counter() - t0_gnd

            loaded_frames.clear()

        else:
            # Acumulación directa estándar con prefetching multihilo concurrente
            prev_img: Optional[np.ndarray] = None
            prev_weight: float = 1.0
            prev_dilated_stars: Optional[np.ndarray] = None

            for i in range(n_files):
                if abort_flag and abort_flag():
                    raise InterruptedError("Procesamiento de trazas cancelado por el usuario.")

                fname = os.path.basename(files[i])
                pct = int(10 + (i / n_files) * 80)
                tl_tag = " [Time-Lapse]" if timelapse_enabled else ""
                if progress_callback:
                    progress_callback(pct, f"Integrando toma ({i+1}/{n_files}): {fname}{tl_tag}")

                img = get_frame(i)

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

                t0_step_acc = time.perf_counter()
                if sky_acc is None:
                    sky_acc = frame_sky.copy()
                elif HAS_RUST and hasattr(astroloom_rs, "accumulate_startrail_step_rs"):
                    curr_dilated_stars = None
                    if fill_gaps:
                        t0_gap = time.perf_counter()
                        curr_dilated_stars = extract_dilated_stars(
                            img,
                            radius=gap_fill_radius,
                            threshold_factor=gap_fill_threshold_factor,
                            sky_mask_bool=sky_mask_bool,
                            kernel=gap_kernel
                        )
                        t_gap_fill += time.perf_counter() - t0_gap
                    w_bridge = (0.5 * (prev_weight + curr_weight)) if (mode == "comet" and comet_weights is not None) else 1.0
                    astroloom_rs.accumulate_startrail_step_rs(
                        sky_acc,
                        np.ascontiguousarray(img, dtype=np.float32),
                        weight=float(curr_weight),
                        prev_dilated=prev_dilated_stars,
                        curr_dilated=curr_dilated_stars,
                        w_bridge=float(w_bridge)
                    )
                    if fill_gaps:
                        prev_dilated_stars = curr_dilated_stars
                else:
                    if fill_gaps:
                        t0_gap = time.perf_counter()
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
                        t_gap_fill += time.perf_counter() - t0_gap

                    np.maximum(sky_acc, frame_sky, out=sky_acc)

                if fill_gaps and prev_dilated_stars is None:
                    t0_gap = time.perf_counter()
                    prev_dilated_stars = extract_dilated_stars(
                        img,
                        radius=gap_fill_radius,
                        threshold_factor=gap_fill_threshold_factor,
                        sky_mask_bool=sky_mask_bool,
                        kernel=gap_kernel
                    )
                    t_gap_fill += time.perf_counter() - t0_gap

                t_accum += time.perf_counter() - t0_step_acc

                prev_img = img
                prev_weight = curr_weight

                if timelapse_enabled:
                    t0_tl = time.perf_counter()
                    _write_timelapse_frame(i)
                    t_timelapse += time.perf_counter() - t0_tl

                if use_clean_ground and mask_3c is not None and ground_mode == "average":
                    t0_gnd = time.perf_counter()
                    if ground_sum is None:
                        ground_sum = img.astype(np.float64)
                    else:
                        ground_sum += img
                    t_ground += time.perf_counter() - t0_gnd

                # Liberar memoria de la toma procesada
                if i in loaded_frames:
                    del loaded_frames[i]

    finally:
        prefetch_exec.shutdown(wait=False, cancel_futures=True)
        if video_writer is not None:
            video_writer.release()
            video_writer.release()

    if sky_acc is None:
        raise RuntimeError("No se pudo acumular ninguna imagen válida.")

    # 7. Realce y preservación de color estelar (Star Color Boost)
    t_color_boost = 0.0
    if color_boost_enabled:
        if progress_callback:
            progress_callback(93, "Realzando y preservando saturación de trazas estelares...")
        t0_cb = time.perf_counter()
        sky_acc = enhance_star_trails_color(
            sky_acc,
            saturation_boost=sat_boost,
            core_recovery=core_recovery,
            sky_mask=sky_mask_bool,
            bg_protect=True
        )
        t_color_boost = time.perf_counter() - t0_cb

    # 8. Fusión con suelo limpio o retorno directo
    t_fusion = 0.0
    if use_clean_ground and mask_3c is not None:
        if progress_callback:
            progress_callback(92, "Fusionando cielo de trazas con suelo anti-ruido...")
        t0_fus = time.perf_counter()

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
                if master_flat is not None and master_flat.shape == ground_clean.shape:
                    safe_flat = np.where(master_flat > 1e-4, master_flat, 1.0)
                    ground_clean = np.clip(ground_clean / safe_flat, 0.0, 1.0)

        # Fusión alfa ponderada
        final_img = (sky_acc * mask_3c) + (ground_clean * (1.0 - mask_3c))
        t_fusion = time.perf_counter() - t0_fus
    else:
        final_img = sky_acc

    final_img = np.clip(final_img, 0.0, 1.0).astype(np.float32)

    if actual_video_path and os.path.exists(actual_video_path):
        if os.path.getsize(actual_video_path) == 0:
            actual_video_path = None
    else:
        actual_video_path = None

    t_total = time.perf_counter() - t_start
    ms_per_frame = (t_total / n_files) * 1000.0 if n_files > 0 else 0.0
    fps_rate = (n_files / t_total) if t_total > 0 else 0.0

    logger.info(
        f"✓ [GENERACIÓN STARTRAILS] Finalizada en {t_total:.2f}s ({n_files} tomas, "
        f"{ms_per_frame:.1f} ms/toma, {fps_rate:.1f} fps)"
    )

    breakdown = [
        f"  • I/O y Carga: {t_io:.2f}s ({100.0 * t_io / max(t_total, 1e-6):.1f}%)"
    ]
    if suppress_streaks:
        ms_st = (t_streak_detection / n_files) * 1000.0 if n_files > 0 else 0.0
        breakdown.append(
            f"  • Detección y Supresión Trazas: {t_streak_detection:.2f}s "
            f"({100.0 * t_streak_detection / max(t_total, 1e-6):.1f}%, {ms_st:.1f} ms/toma, {total_streaks} suprimidas)"
        )
    if fill_gaps:
        breakdown.append(
            f"  • Puenteo Gap Filling: {t_gap_fill:.2f}s ({100.0 * t_gap_fill / max(t_total, 1e-6):.1f}%)"
        )
    breakdown.append(
        f"  • Acumulación Cielo: {t_accum:.2f}s ({100.0 * t_accum / max(t_total, 1e-6):.1f}%)"
    )
    if timelapse_enabled:
        breakdown.append(
            f"  • Renderizado Time-Lapse: {t_timelapse:.2f}s ({100.0 * t_timelapse / max(t_total, 1e-6):.1f}%)"
        )
    if use_clean_ground:
        breakdown.append(
            f"  • Suelo y Fusión Final: {(t_ground + t_fusion):.2f}s ({100.0 * (t_ground + t_fusion) / max(t_total, 1e-6):.1f}%)"
        )
    if color_boost_enabled:
        breakdown.append(
            f"  • Realce Color Estelar: {t_color_boost:.2f}s ({100.0 * t_color_boost / max(t_total, 1e-6):.1f}%)"
        )

    logger.info("[STARTRAILS RENDIMIENTO]:\n" + "\n".join(breakdown))

    if progress_callback:
        msg = f"Composición de trazas finalizada en {t_total:.2f}s ({n_files} tomas, {fps_rate:.1f} fps)."
        if actual_video_path:
            msg += f" Vídeo time-lapse exportado: {os.path.basename(actual_video_path)}"
        progress_callback(100, msg)

    metrics = {
        "t_total": t_total,
        "t_io": t_io,
        "t_meta": t_meta,
        "t_dark": t_dark,
        "t_flat": t_flat,
        "t_streak_detection": t_streak_detection,
        "total_streaks": total_streaks,
        "t_gap_fill": t_gap_fill,
        "t_accum": t_accum,
        "t_timelapse": t_timelapse,
        "t_ground": t_ground,
        "t_fusion": t_fusion,
        "t_color_boost": t_color_boost,
        "t_compute": max(0.0, t_total - t_io),
        "ms_per_frame": ms_per_frame,
        "fps": fps_rate,
        "n_files": n_files
    }

    if return_metrics:
        return final_img, actual_video_path, metrics
    return final_img, actual_video_path
