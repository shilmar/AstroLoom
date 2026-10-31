# core/meteors.py
"""
core/meteors.py - Motor de Procesado y Composición de Lluvias de Meteoros para AstroLoom.

Proporciona algoritmos científicos para:
1. Detección automática de trazas transitorias y bólidos mediante substracción temporal diferencial.
2. Discriminación de falsos positivos (aviones por balizas estroboscópicas y satélites).
3. Registro y reproyección astronómica homográfica de cada meteoro hacia el sistema de
   coordenadas de la toma de referencia (compensando la rotación diurna terrestre).
4. Estimación del punto radiante celeste y clasificación de bólidos pertenecientes a la lluvia vs. esporádicos.
5. Fusión fotométrica de flujo neto sin acumulación de ruido de fondo sobre el cielo apilado.
6. Guardado y restauración de sesiones completas de lluvia de meteoros (.astrometeor).
"""

import os
import gc
import json
import time
import threading
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
import tifffile
from PIL import Image, ExifTags

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from core.stacking import load_image_as_float32, get_image_dimensions
from core.panorama import extract_bright_stars, build_star_asterisms, match_star_asterisms

try:
    import pytoshop
    from pytoshop import layers as psd_layers, enums as psd_enums
    HAS_PYTOSHOP = True
except Exception:
    HAS_PYTOSHOP = False


# =============================================================================
# 1. MODELOS DE DATOS Y CONFIGURACIÓN
# =============================================================================

# Catálogo astronómico de coordenadas celestes aproximadas de lluvias de meteoros anuales
METEOR_SHOWER_PRESETS = {
    "Perseidas": {"name": "Perseidas (Agosto)", "ra_deg": 46.0, "dec_deg": 58.0, "peak_date": "12-Ago"},
    "Gemínidas": {"name": "Gemínidas (Diciembre)", "ra_deg": 112.0, "dec_deg": 33.0, "peak_date": "14-Dic"},
    "Cuadrántidas": {"name": "Cuadrántidas (Enero)", "ra_deg": 230.0, "dec_deg": 49.0, "peak_date": "03-Ene"},
    "Líridas": {"name": "Líridas (Abril)", "ra_deg": 271.0, "dec_deg": 34.0, "peak_date": "22-Abr"},
    "Oriónidas": {"name": "Oriónidas (Octubre)", "ra_deg": 95.0, "dec_deg": 16.0, "peak_date": "21-Oct"},
    "Eta Acuáridas": {"name": "Eta Acuáridas (Mayo)", "ra_deg": 338.0, "dec_deg": -1.0, "peak_date": "06-May"},
    "Leónidas": {"name": "Leónidas (Noviembre)", "ra_deg": 152.0, "dec_deg": 22.0, "peak_date": "17-Nov"},
    "Auto-Detectar": {"name": "Auto-Detectar Radiante (Intersección)", "ra_deg": 0.0, "dec_deg": 0.0, "peak_date": "-"}
}


@dataclass
class MeteorCandidate:
    id: int
    frame_index: int
    filepath: str
    filename: str
    timestamp: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)          # (x1, y1, x2, y2)
    start_pt: Tuple[float, float] = (0.0, 0.0)             # (x, y) en fotograma original
    end_pt: Tuple[float, float] = (0.0, 0.0)               # (x, y) en fotograma original
    length_px: float = 0.0
    thickness_px: float = 0.0
    angle_deg: float = 0.0                                 # Orientación en grados (0 a 180)
    peak_flux: float = 0.0                                 # Brillo máximo alcanzado
    mean_flux: float = 0.0
    is_meteor: bool = True                                 # Incluido por defecto si parece meteoro
    is_sporadic: bool = False                              # Si no converge al radiante
    is_airplane: bool = False                              # Balizas estroboscópicas detectadas
    is_satellite: bool = False                             # Traza recta sin perfil meteórico
    confidence_score: float = 0.85
    reprojected_start: Optional[Tuple[float, float]] = None # En coordenadas de la toma de referencia
    reprojected_end: Optional[Tuple[float, float]] = None
    dist_to_radiant_px: Optional[float] = None
    divergence_angle_deg: Optional[float] = None
    line_continuity: float = 1.0                           # Fracción [0.0, 1.0] de iluminación continua a lo largo del haz


@dataclass
class MeteorSessionConfig:
    shower_name: str = "Perseidas"
    ref_frame_idx: int = 0
    sensitivity_sigma: float = 4.2        # Sensibilidad de detección (3.0 a 6.0)
    min_length_px: float = 24.0           # Longitud mínima de la traza para ser candidata
    max_length_px: float = 2500.0         # Longitud máxima
    filter_airplanes: bool = True         # Descarte automático de balizas rojas/verdes
    filter_satellites: bool = True        # Descarte de trazas uniformes ultra-largas
    radiant_tolerance_deg: float = 7.0    # Tolerancia de divergencia hacia el radiante
    include_sporadics: bool = False       # Si se incluyen o excluyen meteoros esporádicos en el compuesto
    boost_flux: float = 1.0               # Multiplicador de realce de brillo de meteoros
    feather_px: int = 12                  # Radio de desvanecimiento suave de bordes del parche


@dataclass
class MeteorDetectionResult:
    candidates: List[MeteorCandidate]
    radiant_point_ref: Optional[Tuple[float, float]] = None # (x, y) en coordenadas de referencia
    total_frames_scanned: int = 0
    meteors_count: int = 0
    airplanes_count: int = 0
    satellites_count: int = 0


# =============================================================================
# 2. EXTRACCIÓN DE METADATOS Y TIMESTAMP EXIF
# =============================================================================

def extract_frame_timestamp(filepath: str) -> float:
    """Extrae la fecha y hora de captura desde los metadatos EXIF o fecha de archivo."""
    try:
        with Image.open(filepath) as img:
            exif = img.getexif()
            if exif:
                dt_str = exif.get(306) or exif.get(36867)  # DateTime / DateTimeOriginal
                if dt_str:
                    t_struct = time.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S")
                    return time.mktime(t_struct)
    except Exception:
        pass
    try:
        return os.path.getmtime(filepath)
    except Exception:
        return 0.0


# =============================================================================
# 3. KERNEL NUMBA DE DIFERENCIA POSITIVA Y CARGA ULTRA-RÁPIDA
# =============================================================================

@njit(fastmath=True, nogil=True)
def _numba_positive_diff_and_stats(arr_curr: np.ndarray, arr_bg: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Kernel Numba compilado sin GIL que calcula en una pasada continua de memoria:
    1. La diferencia diferencial positiva max(0, curr - bg).
    2. La media y varianza estocástica de los residuos para el umbral adaptativo.
    """
    h = arr_curr.shape[0]
    w = arr_curr.shape[1]
    total_pixels = float(h * w)
    out_diff = np.zeros((h, w), dtype=np.float32)
    sum_diff = 0.0
    sum_sq = 0.0
    for i in range(h):
        for j in range(w):
            d = arr_curr[i, j] - arr_bg[i, j]
            if d > 0.0:
                out_diff[i, j] = d
                sum_diff += d
                sum_sq += d * d
    mean_val = sum_diff / total_pixels
    var_val = (sum_sq / total_pixels) - (mean_val * mean_val)
    if var_val < 0.0:
        var_val = 0.0
    std_val = np.sqrt(var_val)
    return out_diff, mean_val, std_val


def _numpy_positive_diff_and_stats(arr_curr: np.ndarray, arr_bg: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fallback vectorial en NumPy si Numba no está disponible."""
    diff = np.maximum(0.0, arr_curr - arr_bg)
    mean_val = float(np.mean(diff))
    std_val = float(np.std(diff))
    return diff, mean_val, std_val


def compute_positive_diff_stats(arr_curr: np.ndarray, arr_bg: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Despachador acelerado con Numba (nogil) o NumPy según disponibilidad."""
    if HAS_NUMBA:
        return _numba_positive_diff_and_stats(arr_curr, arr_bg)
    return _numpy_positive_diff_and_stats(arr_curr, arr_bg)


if HAS_NUMBA:
    try:
        _dummy = np.zeros((10, 10), dtype=np.float32)
        _numba_positive_diff_and_stats(_dummy, _dummy)
    except Exception:
        pass


def load_fast_meteor_frame(filepath: str, max_dim: int = 2400) -> Tuple[np.ndarray, float]:
    """
    Carga ultra-rápida de una toma para detección y cribado de trazas.
    En archivos RAW usa decodificación directa en binning (half_size=True, 8-bit),
    proporcionando una aceleración de 5x a 10x respecto a la interpolación AHD completa.
    Devuelve la imagen RGB float32 [0.0, 1.0] y el factor de escala respecto a resolución nativa.
    """
    ext = os.path.splitext(filepath)[1].lower()
    raw_exts = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raw", ".raf", ".orf", ".rw2"}

    if ext in raw_exts:
        try:
            import rawpy
            with rawpy.imread(filepath) as raw:
                # Con half_size=True, rawpy decodifica en un factor 2:1 exacto en ambos ejes
                # tras aplicar la orientación EXIF (flip), garantizando consistencia absoluta de escala.
                rgb = raw.postprocess(
                    half_size=True,
                    no_auto_bright=True,
                    use_camera_wb=True,
                    output_bps=8
                )
                scale_factor = 2.0
                img_f32 = rgb.astype(np.float32) / 255.0
                return img_f32, scale_factor
        except Exception:
            pass

    # Formatos estándar (TIFF, FITS, JPEG, PNG)
    img = load_image_as_float32(filepath)
    h, w = img.shape[:2]
    scale_factor = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        w_new = max(1, int(round(w * scale)))
        h_new = max(1, int(round(h * scale)))
        img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        scale_factor = float(w) / float(w_new)
    return img, scale_factor


class ThreadSafeFrameCache:
    """Caché LRU concurrente con hilo seguro para tomas de previsualización/escaneo."""
    def __init__(self, max_items: int = 12):
        self.lock = threading.Lock()
        self.max_items = max_items
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.order: List[str] = []

    def get_or_load(self, path: str) -> Tuple[np.ndarray, float]:
        with self.lock:
            if path in self.cache:
                return self.cache[path]

        # Decodificación fuera del lock para liberar concurrencia de otros hilos
        img, scale = load_fast_meteor_frame(path)

        with self.lock:
            if path not in self.cache:
                self.cache[path] = (img, scale)
                self.order.append(path)
                while len(self.order) > self.max_items:
                    old = self.order.pop(0)
                    self.cache.pop(old, None)
            return self.cache[path]


# =============================================================================
# 4. DETECCIÓN AUTOMÁTICA DE TRAZAS EN FOTOGRAMAS
# =============================================================================

def merge_collinear_segments(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], float, float]],
    max_gap: float = 24.0,
    angle_tol: float = 10.0,
    min_seed_len: float = 12.0
) -> List[Tuple[Tuple[float, float], Tuple[float, float], float, float, List[Tuple[float, float]]]]:
    """
    Fusiona segmentos colineales fragmentados por ruido térmico o balizas estroboscópicas
    que pertenezcan a la misma trayectoria cinemática recta.
    """
    if not segments:
        return []

    items = []
    for p1, p2, l, w in segments:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = float(np.hypot(dx, dy))
        ang = float(np.degrees(np.arctan2(dy, dx))) % 180.0
        items.append({
            'p1': p1, 'p2': p2, 'ang': ang, 'len': length, 'w': w,
            'pts': [p1, p2]
        })

    merged = []
    used = set()
    for i in range(len(items)):
        if i in used:
            continue
        cur = items[i]
        used.add(i)
        track_pts = list(cur['pts'])
        ang_base = cur['ang']
        max_w = cur['w']

        changed = True
        while changed:
            changed = False
            p_arr = np.array(track_pts, dtype=np.float32)
            dists = np.linalg.norm(p_arr[:, None, :] - p_arr[None, :, :], axis=-1)
            im, jm = np.unravel_index(np.argmax(dists), dists.shape)
            p_start = p_arr[im]
            p_end = p_arr[jm]
            curr_len = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))

            for j in range(len(items)):
                if j in used:
                    continue
                cand = items[j]
                ang_diff = abs(cand['ang'] - ang_base)
                if ang_diff > 90:
                    ang_diff = 180 - ang_diff
                if ang_diff > angle_tol:
                    continue

                d_end1 = float(np.hypot(cand['p1'][0] - p_end[0], cand['p1'][1] - p_end[1]))
                d_end2 = float(np.hypot(cand['p2'][0] - p_end[0], cand['p2'][1] - p_end[1]))
                d_start1 = float(np.hypot(cand['p1'][0] - p_start[0], cand['p1'][1] - p_start[1]))
                d_start2 = float(np.hypot(cand['p2'][0] - p_start[0], cand['p2'][1] - p_start[1]))
                min_d = min(d_end1, d_end2, d_start1, d_start2)

                # Evitar unir dos pequeñas estrellas/ruidos dispersos a través de cielo oscuro
                if min_d > 4.0 and curr_len < min_seed_len and cand['len'] < min_seed_len:
                    continue

                if min_d <= max_gap:
                    track_pts.extend(cand['pts'])
                    used.add(j)
                    max_w = max(max_w, cand['w'])
                    changed = True
                    break

        p_arr = np.array(track_pts, dtype=np.float32)
        dists = np.linalg.norm(p_arr[:, None, :] - p_arr[None, :, :], axis=-1)
        im, jm = np.unravel_index(np.argmax(dists), dists.shape)
        pt_start = (float(p_arr[im][0]), float(p_arr[im][1]))
        pt_end = (float(p_arr[jm][0]), float(p_arr[jm][1]))
        tot_len = float(np.hypot(pt_end[0] - pt_start[0], pt_end[1] - pt_start[1]))
        merged.append((pt_start, pt_end, tot_len, max_w, track_pts))

    return merged


def detect_streaks_in_frame(
    img_curr: np.ndarray,
    img_bg: np.ndarray,
    filepath: str,
    frame_idx: int,
    start_id: int = 1,
    sensitivity_sigma: float = 4.2,
    min_length_px: float = 24.0,
    max_length_px: float = 2500.0,
    filter_airplanes: bool = True,
    filter_satellites: bool = True,
    scale_factor: float = 1.0,
    ground_mask: Optional[np.ndarray] = None
) -> List[MeteorCandidate]:
    """
    Detecta trazas transitorias calculando la diferencia fotométrica positiva respecto
    a la imagen de fondo estelar. Analiza la geometría, perfil de brillo y firma espectral.
    Escala automáticamente las coordenadas candidatas al espacio de resolución nativo.
    Inmune al desplazamiento estelar por rotación terrestre gracias a estimación de ruido robusta (MAD).
    """
    h, w = img_curr.shape[:2]
    # 1. Luminancia en escala de grises
    if img_curr.ndim == 3:
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY)
        gray_bg = cv2.cvtColor(img_bg, cv2.COLOR_RGB2GRAY) if img_bg.ndim == 3 else img_bg
    else:
        gray_curr = img_curr
        gray_bg = img_bg

    if gray_curr.shape != gray_bg.shape:
        gray_bg = cv2.resize(gray_bg, (gray_curr.shape[1], gray_curr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 2. Estimación robusta del ruido del cielo (inmune a estrellas desplazadas por rotación terrestre)
    raw_diff = gray_curr - gray_bg
    sample = raw_diff[::4, ::4]
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    sigma_robust = float(1.4826 * mad)
    thresh_val = med + (sensitivity_sigma * 0.72) * max(sigma_robust, 0.0025)

    # 3. Procesar máscara de exclusión de suelo / horizonte
    sky_mask_u8 = None
    if ground_mask is not None and ground_mask.size > 0:
        g_raw = ground_mask.astype(np.float32)
        if g_raw.ndim == 3:
            g_raw = cv2.cvtColor(g_raw, cv2.COLOR_RGB2GRAY)
        if np.max(g_raw) > 1.05:
            max_val = 65535.0 if np.max(g_raw) > 255.0 else 255.0
            g_raw /= max_val
        h_m, w_m = g_raw.shape[:2]
        top_mean = float(np.mean(g_raw[:max(1, int(h_m * 0.15)), :]))
        bot_mean = float(np.mean(g_raw[-max(1, int(h_m * 0.15)):, :]))
        # Polaridad adaptativa: el cielo siempre está hacia el cenit
        sky_f = (g_raw > 0.5) if (top_mean >= bot_mean) else (g_raw < 0.5)
        sky_mask_u8 = cv2.resize(sky_f.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # Margen de seguridad contra movimiento de vegetación/viento en la frontera de horizonte
        k_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        sky_mask_u8 = cv2.erode(sky_mask_u8, k_safe, borderType=cv2.BORDER_REPLICATE)

    diff = np.maximum(0.0, raw_diff)
    if sky_mask_u8 is not None:
        diff *= sky_mask_u8

    diff_smooth = cv2.GaussianBlur(diff, (3, 3), 0.8)
    _, bin_mask = cv2.threshold(diff_smooth, thresh_val, 255.0, cv2.THRESH_BINARY)
    bin_u8 = bin_mask.astype(np.uint8)

    # Conectar trazas rotas con elemento morfológico elíptico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_closed = cv2.morphologyEx(bin_u8, cv2.MORPH_CLOSE, kernel)
    if sky_mask_u8 is not None:
        bin_closed &= (sky_mask_u8 * 255)

    contours, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fname = os.path.basename(filepath)
    ts = extract_frame_timestamp(filepath)

    raw_segments = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (dim1, dim2), angle = rect
        length = max(dim1, dim2)
        width = min(dim1, dim2)
        aspect_ratio = length / max(width, 1.0)

        # Filtrar estrellas y ruido circular (estrellas tienen aspect < 2.0)
        if length >= 6.0 and aspect_ratio >= 2.0:
            if sky_mask_u8 is not None:
                c_ix = int(round(max(0, min(w - 1, cx))))
                c_iy = int(round(max(0, min(h - 1, cy))))
                if sky_mask_u8[c_iy, c_ix] == 0:
                    continue

            box_pts = cv2.boxPoints(rect)
            dists = np.linalg.norm(box_pts[:, None, :] - box_pts[None, :, :], axis=-1)
            im, jm = np.unravel_index(np.argmax(dists), dists.shape)
            pt_a = (float(box_pts[im][0]), float(box_pts[im][1]))
            pt_b = (float(box_pts[jm][0]), float(box_pts[jm][1]))
            raw_segments.append((pt_a, pt_b, length, width))

    # 3. Fusión de segmentos colineales (reunir trazas micro-fragmentadas y balizas de aviones)
    merged_tracks = merge_collinear_segments(raw_segments, max_gap=24.0, angle_tol=10.0, min_seed_len=12.0)

    min_len_eff = min_length_px / scale_factor
    max_len_eff = max_length_px / scale_factor

    candidates = []
    cand_id = start_id

    for pt_a, pt_b, length, width, all_track_pts in merged_tracks:
        if length < min_len_eff or length > max_len_eff:
            continue

        aspect_ratio = length / max(width, 1.0)
        if aspect_ratio < 3.2:
            continue

        if sky_mask_u8 is not None:
            mid_ix = int(round(max(0, min(w - 1, (pt_a[0] + pt_b[0]) / 2.0))))
            mid_iy = int(round(max(0, min(h - 1, (pt_a[1] + pt_b[1]) / 2.0))))
            if sky_mask_u8[mid_iy, mid_ix] == 0:
                continue

            a_ix = int(round(max(0, min(w - 1, pt_a[0]))))
            a_iy = int(round(max(0, min(h - 1, pt_a[1]))))
            b_ix = int(round(max(0, min(w - 1, pt_b[0]))))
            b_iy = int(round(max(0, min(h - 1, pt_b[1]))))
            if sky_mask_u8[a_iy, a_ix] == 0 or sky_mask_u8[b_iy, b_ix] == 0:
                continue

        # 4. Segunda pasada física: Verificación de perfil de línea continua y huecos oscuros
        # Rechaza alineaciones accidentales de estrellas separadas por cielo negro.
        num_samples = max(8, int(round(length)))
        sample_xs = np.linspace(pt_a[0], pt_b[0], num_samples)
        sample_ys = np.linspace(pt_a[1], pt_b[1], num_samples)

        dx_l = pt_b[0] - pt_a[0]
        dy_l = pt_b[1] - pt_a[1]
        l_norm = np.hypot(dx_l, dy_l)
        if l_norm > 1e-4:
            nx = -dy_l / l_norm
            ny = dx_l / l_norm
        else:
            nx, ny = 0.0, 0.0

        v_profile = np.zeros(num_samples, dtype=np.float32)
        max_trans_off = max(1.5, min(float(width) / 2.0, 4.0))
        for offset in np.linspace(-max_trans_off, max_trans_off, 5):
            sx = np.clip(np.round(sample_xs + nx * offset).astype(int), 0, w - 1)
            sy = np.clip(np.round(sample_ys + ny * offset).astype(int), 0, h - 1)
            v_profile = np.maximum(v_profile, diff[sy, sx])

        is_lit = (v_profile > (thresh_val * 0.70))
        lit_indices = np.where(is_lit)[0]
        if len(lit_indices) < 4:
            continue

        internal_lit = is_lit[lit_indices[0]:lit_indices[-1] + 1]
        fill_ratio = float(np.mean(internal_lit))

        max_dark_gap = 0
        curr_gap = 0
        for lit in internal_lit:
            if not lit:
                curr_gap += 1
                if curr_gap > max_dark_gap:
                    max_dark_gap = curr_gap
            else:
                curr_gap = 0

        # Un meteoro o satélite real forma un trazo continuo (> 45% iluminado) sin huecos oscuros internos > 16 px
        if fill_ratio < 0.45 or max_dark_gap > 16:
            continue

        # Extraer bounding box expandido para inspección
        xs = [p[0] for p in all_track_pts]
        ys = [p[1] for p in all_track_pts]
        pad = 12
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(w, int(max(xs)) + pad)
        y2 = min(h, int(max(ys)) + pad)

        patch_curr = img_curr[y1:y2, x1:x2]
        patch_diff = diff[y1:y2, x1:x2]

        peak_flux = float(np.max(patch_diff)) if patch_diff.size > 0 else 0.0
        mean_flux = float(np.mean(patch_diff[patch_diff > thresh_val])) if np.any(patch_diff > thresh_val) else peak_flux

        # Determinar ángulo respecto al eje X
        dx = pt_b[0] - pt_a[0]
        dy = pt_b[1] - pt_a[1]
        deg = float(np.degrees(np.arctan2(dy, dx))) % 180.0

        # Discriminador de avión / satélite
        is_plane = False
        is_sat = False
        conf = 0.88

        track_mask = (patch_diff > thresh_val)
        track_px_count = int(np.sum(track_mask))

        if patch_curr.ndim == 3 and track_px_count >= 6:
            # 1. Extraer fondo local en el parche fuera de la traza para contraste cromático diferencial
            non_track = ~track_mask
            if np.sum(non_track) > 20:
                bg_local = np.median(patch_curr[non_track], axis=0)
            else:
                bg_local = np.median(patch_curr, axis=(0, 1))

            r_tr = patch_curr[track_mask, 0]
            g_tr = patch_curr[track_mask, 1]
            b_tr = patch_curr[track_mask, 2]

            # Exceso cromático diferencial sobre el cielo nocturno local (inmune a baja exposición / fotos oscuras)
            d_r = r_tr - bg_local[0]
            d_g = g_tr - bg_local[1]
            d_b = b_tr - bg_local[2]

            chroma_thresh = max(1.8 * sigma_robust, 0.012)

            # Balizas rojas (luz de navegación babor / destello estroboscópico de panza o deriva)
            red_pixels = (d_r > chroma_thresh) & (d_r > 1.35 * np.maximum(d_g, 0.0)) & (d_r > 1.35 * np.maximum(d_b, 0.0))
            n_red = int(np.sum(red_pixels))

            # Balizas verdes (luz de navegación estribor, verde esmeralda ~520nm)
            green_pixels = (d_g > chroma_thresh) & (d_g > 1.25 * np.maximum(d_r, 0.0))
            n_green = int(np.sum(green_pixels))

            # A. Criterio bicolor simultáneo (Firma definitiva de avión: luces de ambas alas/balizas)
            if n_red >= 4 and n_green >= 4:
                is_plane = True
                conf = 0.10

            # B. Criterio de baliza estroboscópica roja pulsante (destellos periódicos de panza/deriva)
            elif n_red >= 6:
                num_s = max(16, int(round(length)))
                xs = np.clip(np.round(np.linspace(pt_a[0], pt_b[0], num_s) - x1).astype(int), 0, patch_curr.shape[1] - 1)
                ys = np.clip(np.round(np.linspace(pt_a[1], pt_b[1], num_s) - y1).astype(int), 0, patch_curr.shape[0] - 1)
                line_r_diff = patch_curr[ys, xs, 0] - bg_local[0]

                # Contar destellos con prominencia real (al menos 35% de caída entre destellos sucesivos)
                peaks = 0
                max_r = float(np.max(line_r_diff))
                if max_r > chroma_thresh:
                    prom_thresh = max_r * 0.35
                    for p_idx in range(2, num_s - 2):
                        val = line_r_diff[p_idx]
                        if val > chroma_thresh:
                            left_min = np.min(line_r_diff[max(0, p_idx - 6):p_idx])
                            right_min = np.min(line_r_diff[p_idx + 1:min(num_s, p_idx + 7)])
                            if (val - left_min >= prom_thresh) and (val - right_min >= prom_thresh):
                                peaks += 1
                if peaks >= 3:
                    is_plane = True
                    conf = 0.15

        # Discriminador de satélites: trazas ultralargas (> 900px) con brillo plano/constante (sin llamarada)
        flare_ratio = peak_flux / max(mean_flux, 1e-4)
        if (length * scale_factor) > 900 and aspect_ratio > 25 and not is_plane:
            # Los satélites tienen brillo uniforme constante (flare_ratio bajo < 2.2)
            # Los meteoros presentan una llamarada terminal de ablación muy pronunciada (flare_ratio >= 2.5)
            if flare_ratio < 2.2:
                is_sat = True
                conf = 0.30

        # Si el usuario DESMARCÓ el filtro (ej. filter_satellites=False o filter_airplanes=False),
        # se respeta su decisión y se mantiene la traza como incluida (is_meteor = True)
        cand_is_meteor = True
        if is_plane and filter_airplanes:
            cand_is_meteor = False
        if is_sat and filter_satellites:
            cand_is_meteor = False

        cand = MeteorCandidate(
            id=cand_id,
            frame_index=frame_idx,
            filepath=filepath,
            filename=fname,
            timestamp=ts,
            bbox=(
                int(round(x1 * scale_factor)),
                int(round(y1 * scale_factor)),
                int(round(x2 * scale_factor)),
                int(round(y2 * scale_factor))
            ),
            start_pt=(float(pt_a[0] * scale_factor), float(pt_a[1] * scale_factor)),
            end_pt=(float(pt_b[0] * scale_factor), float(pt_b[1] * scale_factor)),
            length_px=float(length * scale_factor),
            thickness_px=float(width * scale_factor),
            angle_deg=deg,
            peak_flux=peak_flux,
            mean_flux=mean_flux,
            is_meteor=cand_is_meteor,
            is_airplane=is_plane,
            is_satellite=is_sat,
            confidence_score=conf,
            line_continuity=round(fill_ratio, 2)
        )
        candidates.append(cand)
        cand_id += 1

    return candidates


# =============================================================================
# 5. REPROYECCIÓN ASTRONÓMICA Y ESTIMACIÓN DEL RADIANTE
# =============================================================================

def compute_frame_homography_to_ref(
    img_target: np.ndarray,
    img_ref: Optional[np.ndarray] = None,
    ref_asterisms: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    scale_factor: float = 1.0
) -> Optional[np.ndarray]:
    """
    Calcula la matriz homográfica 3x3 de transformación estelar que mapea
    coordenadas de la imagen destino a la imagen maestra de referencia.
    Utiliza asterismos estelares invariantes para máxima solidez nocturna.
    Si ref_asterisms está presente, reutiliza los asterismos precalculados de la referencia.
    Escala la matriz homográfica al espacio de coordenadas nativo si scale_factor != 1.0.
    """
    try:
        pts_target = extract_bright_stars(img_target, max_stars=90)

        feats_ref, quads_ref, pts_ref = (None, None, None)
        if ref_asterisms is not None:
            feats_ref, quads_ref, pts_ref = ref_asterisms
        elif img_ref is not None:
            pts_ref = extract_bright_stars(img_ref, max_stars=90)
            if len(pts_ref) >= 6:
                feats_ref, quads_ref = build_star_asterisms(pts_ref)

        if pts_ref is not None and feats_ref is not None and len(pts_target) >= 6 and len(pts_ref) >= 6:
            feats_tgt, quads_tgt = build_star_asterisms(pts_target)
            m_ref, m_tgt, inliers = match_star_asterisms(
                feats_ref, quads_ref, pts_ref,
                feats_tgt, quads_tgt, pts_target,
                tolerance=0.030, min_votes=2
            )
            if inliers >= 4:
                H_down, _ = cv2.findHomography(m_tgt, m_ref, cv2.RANSAC, 4.0)
                if H_down is not None:
                    if abs(scale_factor - 1.0) > 1e-4:
                        S = np.diag([scale_factor, scale_factor, 1.0])
                        S_inv = np.diag([1.0 / scale_factor, 1.0 / scale_factor, 1.0])
                        H_full = S @ H_down @ S_inv
                        return (H_full / H_full[2, 2]).astype(np.float32)
                    return H_down.astype(np.float32)

        # Fallback a detector ORB si asterismos no alcanzaron inliers suficientes
        if img_ref is not None:
            orb = cv2.ORB_create(nfeatures=1500)
            g1 = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY) if img_target.ndim == 3 else img_target
            g2 = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY) if img_ref.ndim == 3 else img_ref
            u1 = (np.clip(g1, 0, 1) * 255).astype(np.uint8) if g1.dtype != np.uint8 else g1
            u2 = (np.clip(g2, 0, 1) * 255).astype(np.uint8) if g2.dtype != np.uint8 else g2
            kp1, des1 = orb.detectAndCompute(u1, None)
            kp2, des2 = orb.detectAndCompute(u2, None)
            if des1 is not None and des2 is not None and len(des1) > 8 and len(des2) > 8:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                if len(matches) >= 6:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H_down, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                    if H_down is not None:
                        if abs(scale_factor - 1.0) > 1e-4:
                            S = np.diag([scale_factor, scale_factor, 1.0])
                            S_inv = np.diag([1.0 / scale_factor, 1.0 / scale_factor, 1.0])
                            H_full = S @ H_down @ S_inv
                            return (H_full / H_full[2, 2]).astype(np.float32)
                        return H_down.astype(np.float32)
    except Exception:
        pass
    return None


def reproject_candidate_to_ref(
    cand: MeteorCandidate,
    H_to_ref: Optional[np.ndarray]
) -> None:
    """Proyecta los puntos extremos de la traza al sistema de coordenadas de la referencia."""
    if H_to_ref is None:
        cand.reprojected_start = cand.start_pt
        cand.reprojected_end = cand.end_pt
        return

    pts = np.array([[cand.start_pt[0], cand.start_pt[1]],
                    [cand.end_pt[0], cand.end_pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H_to_ref)
    cand.reprojected_start = (float(warped[0, 0, 0]), float(warped[0, 0, 1]))
    cand.reprojected_end = (float(warped[1, 0, 0]), float(warped[1, 0, 1]))


def estimate_radiant_intersection(
    candidates: List[MeteorCandidate],
    canvas_shape: Tuple[int, int]
) -> Optional[Tuple[float, float]]:
    """
    Encuentra el punto radiante celeste (x_R, y_R) óptimo mediante mínimos cuadrados
    ortogonales, intersecando las rectas proyectadas hacia atrás de los meteoros confirmados.
    """
    active = [c for c in candidates if c.is_meteor and not c.is_airplane and not c.is_satellite]
    if len(active) < 2:
        return None

    A_mat = []
    B_vec = []

    for c in active:
        p1 = c.reprojected_start or c.start_pt
        p2 = c.reprojected_end or c.end_pt

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length < 5.0:
            continue

        # Ecuación de la recta: A*x + B*y = C
        # Normalizada: nx*x + ny*y = d
        nx = -dy / length
        ny = dx / length
        d = nx * p1[0] + ny * p1[1]

        A_mat.append([nx, ny])
        B_vec.append(d)

    if len(A_mat) < 2:
        return None

    A = np.array(A_mat, dtype=np.float64)
    B = np.array(B_vec, dtype=np.float64)

    try:
        # Resolver por mínimos cuadrados (A * [x, y]^T = B)
        sol, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        rx, ry = float(sol[0]), float(sol[1])

        # Verificar si el radiante cae dentro de un radio razonable del encuadre
        h, w = canvas_shape[:2]
        if -w <= rx <= 2 * w and -h <= ry <= 2 * h:
            return (rx, ry)
    except Exception:
        pass

    return None


def refine_satellite_classifications(
    candidates: List[MeteorCandidate],
    filter_satellites: bool = True
) -> None:
    """
    Refina la clasificación de satélites mediante dos leyes físicas de alta certidumbre:
    1. Persistencia orbital multicuadro (Propuesta 2): Identifica satélites que recorren
       la misma trayectoria colineal a lo largo de 2 o 3 tomas consecutivas.
    2. Cruces de trazas en el mismo fotograma (Propuesta 3A): Los meteoros no se intersectan
       en la misma exposición; las trazas cruzadas corresponden a satélites o aviones.
    """
    if not candidates:
        return

    n = len(candidates)

    # 1. Persistencia cinemática inter-fotograma (tomas consecutivas t y t+1 / t+2)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = candidates[i], candidates[j]
            if abs(c1.frame_index - c2.frame_index) in (1, 2):
                if c1.length_px >= 35.0 and c2.length_px >= 35.0:
                    ang_diff = abs(c1.angle_deg - c2.angle_deg)
                    if ang_diff > 90.0:
                        ang_diff = 180.0 - ang_diff
                    if ang_diff <= 4.5:
                        p1, p2 = c1.start_pt, c1.end_pt
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        L = float(np.hypot(dx, dy))
                        if L > 1e-4:
                            mb = ((c2.start_pt[0] + c2.end_pt[0]) / 2.0, (c2.start_pt[1] + c2.end_pt[1]) / 2.0)
                            d_perp = abs(dy * mb[0] - dx * mb[1] + p2[0] * p1[1] - p2[1] * p1[0]) / L
                            if d_perp <= 20.0:
                                if c1.is_airplane or c2.is_airplane:
                                    c1.is_airplane = True
                                    c2.is_airplane = True
                                    c1.is_satellite = False
                                    c2.is_satellite = False
                                else:
                                    c1.is_satellite = True
                                    c2.is_satellite = True
                                if filter_satellites:
                                    c1.is_meteor = False
                                    c2.is_meteor = False
                                c1.confidence_score = 0.20
                                c2.confidence_score = 0.20

    # 2. Cruces espaciales dentro del mismo fotograma (forma de X / intersección orbital)
    by_frame: Dict[int, List[MeteorCandidate]] = {}
    for c in candidates:
        by_frame.setdefault(c.frame_index, []).append(c)

    for f_idx, f_cands in by_frame.items():
        nf = len(f_cands)
        for i in range(nf):
            for j in range(i + 1, nf):
                c1, c2 = f_cands[i], f_cands[j]
                if c1.length_px >= 35.0 and c2.length_px >= 35.0:
                    ang_diff = abs(c1.angle_deg - c2.angle_deg)
                    if ang_diff > 90.0:
                        ang_diff = 180.0 - ang_diff
                    if 15.0 <= ang_diff <= 165.0:
                        p1, p2 = c1.start_pt, c1.end_pt
                        q1, q2 = c2.start_pt, c2.end_pt
                        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
                        dx2, dy2 = q2[0] - q1[0], q2[1] - q1[1]
                        a1, b1 = -dy1, dx1
                        c_val1 = -dy1 * p1[0] + dx1 * p1[1]
                        a2, b2 = -dy2, dx2
                        c_val2 = -dy2 * q1[0] + dx2 * q1[1]
                        det = a1 * b2 - a2 * b1
                        if abs(det) > 1e-6:
                            xi = (c_val1 * b2 - c_val2 * b1) / det
                            yi = (a1 * c_val2 - a2 * c_val1) / det
                            l1_sq = dx1 * dx1 + dy1 * dy1
                            l2_sq = dx2 * dx2 + dy2 * dy2
                            t1 = np.clip(((xi - p1[0]) * dx1 + (yi - p1[1]) * dy1) / l1_sq, 0.0, 1.0)
                            dist1 = float(np.hypot(xi - (p1[0] + t1 * dx1), yi - (p1[1] + t1 * dy1)))
                            t2 = np.clip(((xi - q1[0]) * dx2 + (yi - q1[1]) * dy2) / l2_sq, 0.0, 1.0)
                            dist2 = float(np.hypot(xi - (q1[0] + t2 * dx2), yi - (q1[1] + t2 * dy2)))
                            if dist1 <= 200.0 and dist2 <= 200.0:
                                if c1.is_airplane or c2.is_airplane:
                                    c1.is_airplane = True
                                    c2.is_airplane = True
                                    c1.is_satellite = False
                                    c2.is_satellite = False
                                else:
                                    c1.is_satellite = True
                                    c2.is_satellite = True
                                if filter_satellites:
                                    c1.is_meteor = False
                                    c2.is_meteor = False
                                c1.confidence_score = 0.20
                                c2.confidence_score = 0.20


def classify_sporadics(
    candidates: List[MeteorCandidate],
    radiant_pt: Tuple[float, float],
    tolerance_deg: float = 7.0
) -> None:
    """
    Calcula la divergencia angular de cada meteoro respecto al radiante y marca
    como 'esporádico' a aquellos cuya prolongación no converge al punto de fuga.
    """
    rx, ry = radiant_pt
    for c in candidates:
        if not c.is_meteor:
            continue

        p1 = c.reprojected_start or c.start_pt
        p2 = c.reprojected_end or c.end_pt

        # Vector de la traza
        v_trace = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        norm_t = np.linalg.norm(v_trace)
        if norm_t < 1e-4:
            continue
        v_trace /= norm_t

        # Centro del meteoro
        mx = 0.5 * (p1[0] + p2[0])
        my = 0.5 * (p1[1] + p2[1])

        # Vector desde el radiante hacia el meteoro
        v_rad = np.array([mx - rx, my - ry], dtype=np.float32)
        dist_rad = np.linalg.norm(v_rad)
        c.dist_to_radiant_px = float(dist_rad)

        if dist_rad < 1e-4:
            c.divergence_angle_deg = 0.0
            c.is_sporadic = False
            continue

        v_rad /= dist_rad

        # Ángulo entre el vector del radiante y la dirección del meteoro
        cos_ang = float(np.clip(np.dot(v_rad, v_trace), -1.0, 1.0))
        # El meteoro puede estar orientado de p1 a p2 o viceversa, evaluar ambas direcciones
        ang1 = float(np.degrees(np.arccos(cos_ang)))
        ang2 = abs(180.0 - ang1)
        divergence = min(ang1, ang2)

        c.divergence_angle_deg = divergence
        c.is_sporadic = (divergence > tolerance_deg)


# =============================================================================
# 5. FUSIÓN FOTOMÉTRICA DE FLUJO NETO
# =============================================================================

def compose_meteor_shower_image(
    base_image: np.ndarray,
    candidates: List[MeteorCandidate],
    homographies_map: Dict[int, Optional[np.ndarray]],
    boost_flux: float = 1.0,
    include_sporadics: bool = False,
    ground_mask: Optional[np.ndarray] = None,
    feather_px: int = 10,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> np.ndarray:
    """
    Funde los meteoros aprobados sobre la imagen base limpia. Aísla el flujo fotométrico
    neto de cada traza (restando el pedestal de cielo local) y la deforma geométricamente
    al sistema de coordenadas de referencia sin acumular ruido estocástico de fondo.
    """
    out_img = base_image.copy().astype(np.float32)
    h_out, w_out = out_img.shape[:2]

    active = [c for c in candidates if c.is_meteor and (include_sporadics or not c.is_sporadic)]
    # La selección explícita del usuario en la tabla de culling (c.is_meteor) es la máxima autoridad:
    if not active:
        active = [c for c in candidates if c.is_meteor]

    total = len(active)

    if total == 0:
        return out_img

    for idx, c in enumerate(active):
        if progress_callback:
            progress_callback(int((idx / total) * 100), f"Fundiendo meteoro {idx + 1}/{total}...")

        try:
            # Cargar fotograma de origen del meteoro
            src_img = load_image_as_float32(c.filepath)
            h_src, w_src = src_img.shape[:2]

            pad = max(24, int(round(c.thickness_px * 2.0)) + feather_px)
            x1, y1, x2, y2 = c.bbox
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = min(w_src, x2 + pad)
            y2_p = min(h_src, y2 + pad)

            patch = src_img[y1_p:y2_p, x1_p:x2_p]
            if patch.size == 0:
                continue

            # Coordenadas locales en el parche
            u_start = c.start_pt[0] - x1_p
            v_start = c.start_pt[1] - y1_p
            u_end = c.end_pt[0] - x1_p
            v_end = c.end_pt[1] - y1_p
            p_start = (int(round(u_start)), int(round(v_start)))
            p_end = (int(round(u_end)), int(round(v_end)))

            # 1. Crear máscara de cápsula vectorial ajustada a la traza del meteoro
            thick = max(6, int(round(c.thickness_px * 1.5)) + 4)
            mask_capsule = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
            cv2.line(mask_capsule, p_start, p_end, 255, thick)
            cv2.circle(mask_capsule, p_start, thick // 2, 255, -1)
            cv2.circle(mask_capsule, p_end, thick // 2, 255, -1)

            k_blur_cap = max(3, (thick // 2) * 2 + 1)
            mask_capsule_f = cv2.GaussianBlur(
                mask_capsule.astype(np.float32) / 255.0, (k_blur_cap, k_blur_cap), thick / 3.0
            )

            # 2. Estimar fondo local en el parche y en la imagen base por canal RGB (percentil 30)
            is_sky = (mask_capsule == 0)
            if np.sum(is_sky) > 25:
                sky_pixels = patch[is_sky]
                bg_src = np.percentile(sky_pixels, 30, axis=0)
            else:
                bg_src = np.percentile(patch, 25, axis=(0, 1) if patch.ndim > 2 else 0)

            # Posición equivalente en la imagen base para igualación exacta de pedestal
            p1_ref = c.reprojected_start or c.start_pt
            p2_ref = c.reprojected_end or c.end_pt
            margin = pad + 16
            bx1 = max(0, int(min(p1_ref[0], p2_ref[0]) - margin))
            by1 = max(0, int(min(p1_ref[1], p2_ref[1]) - margin))
            bx2 = min(w_out, int(max(p1_ref[0], p2_ref[0]) + margin))
            by2 = min(h_out, int(max(p1_ref[1], p2_ref[1]) + margin))
            base_patch = out_img[by1:by2, bx1:bx2]
            if base_patch.size > 25:
                bg_base = np.percentile(base_patch, 30, axis=(0, 1) if base_patch.ndim > 2 else 0)
            else:
                bg_base = np.percentile(out_img, 30, axis=(0, 1) if out_img.ndim > 2 else 0)

            # 3. Puerta de ruido (Noise Gate) basada en la luminancia diferencial
            delta = patch - bg_src
            if delta.ndim == 3 and delta.shape[2] == 3:
                delta_lum = 0.299 * delta[..., 0] + 0.587 * delta[..., 1] + 0.114 * delta[..., 2]
                sky_delta_lum = delta_lum[is_sky] if np.sum(is_sky) > 25 else delta_lum
            else:
                delta_lum = delta.squeeze() if delta.ndim == 3 else delta
                sky_delta_lum = delta_lum[is_sky] if np.sum(is_sky) > 25 else delta_lum

            neg_sky = sky_delta_lum[sky_delta_lum < 0]
            if len(neg_sky) > 10:
                sigma_sky = float(1.4826 * np.median(np.abs(neg_sky)))
            else:
                sigma_sky = float(np.std(sky_delta_lum)) if sky_delta_lum.size > 0 else 0.003

            noise_floor = max(2.0 * sigma_sky, 0.003)
            noise_ceil = max(3.5 * sigma_sky, 0.008)

            # Máscara adaptativa: 0.0 en ruido de cielo, 1.0 en traza del meteoro
            mask_gate = np.clip((delta_lum - noise_floor) / max(noise_ceil - noise_floor, 1e-5), 0.0, 1.0)
            mask_final = mask_capsule_f * mask_gate
            mask_final = cv2.GaussianBlur(mask_final, (3, 3), 0.8)

            # 4. Calibración fotométrica con realce de flujo
            patch_calibrated = bg_base + delta * boost_flux

            # 5. Proyectar al sistema de coordenadas de la imagen de referencia
            H = homographies_map.get(c.frame_index)
            if H is not None:
                T = np.eye(3, dtype=np.float32)
                T[0, 2] = x1_p
                T[1, 2] = y1_p
                H_combined = H @ T

                warped_patch = cv2.warpPerspective(
                    patch_calibrated, H_combined, (w_out, h_out),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                )
                warped_mask = cv2.warpPerspective(
                    mask_final, H_combined, (w_out, h_out),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                )
            else:
                warped_patch = np.zeros_like(out_img)
                warped_patch[y1_p:y2_p, x1_p:x2_p] = patch_calibrated
                warped_mask = np.zeros((h_out, w_out), dtype=np.float32)
                warped_mask[y1_p:y2_p, x1_p:x2_p] = mask_final

            # 6. Fusión en modo Aclarar (Lighten) ponderada por la máscara adaptativa
            if warped_mask.ndim == 2 and out_img.ndim == 3:
                warped_mask_3d = warped_mask[..., None]
            else:
                warped_mask_3d = warped_mask

            excess = np.maximum(0.0, warped_patch - out_img)
            out_img += warped_mask_3d * excess

            del src_img, patch, warped_patch, warped_mask
        except Exception:
            continue

    # 6. Preservar suelo estático limpio si hay máscara de tierra definida
    if ground_mask is not None and ground_mask.size > 0:
        g_raw = ground_mask.astype(np.float32)
        if g_raw.ndim == 3:
            g_raw = cv2.cvtColor(g_raw, cv2.COLOR_RGB2GRAY)
        if np.max(g_raw) > 1.05:
            max_val = 65535.0 if np.max(g_raw) > 255.0 else 255.0
            g_raw /= max_val
        h_m, w_m = g_raw.shape[:2]
        top_mean = float(np.mean(g_raw[:max(1, int(h_m * 0.15)), :]))
        bot_mean = float(np.mean(g_raw[-max(1, int(h_m * 0.15)):, :]))
        # En la composición gmask debe ser 1.0 en suelo y 0.0 en cielo
        is_top_sky = (top_mean >= bot_mean)
        gmask = (1.0 - g_raw) if is_top_sky else g_raw

        if gmask.shape[:2] != out_img.shape[:2]:
            gmask = cv2.resize(gmask, (out_img.shape[1], out_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        if feather_px > 0:
            k = 2 * feather_px + 1
            gmask = cv2.GaussianBlur(gmask, (k, k), feather_px / 2.0)

        if gmask.ndim == 2 and out_img.ndim == 3:
            gmask = gmask[..., None]

        # Restablecer suelo sin trazas
        out_img = out_img * (1.0 - gmask) + base_image * gmask

    return np.clip(out_img, 0.0, 1.0)


# =============================================================================
# 6. EXPORTACIÓN MULTICAPA PSD (PHOTOSHOP)
# =============================================================================

def export_meteor_layered_psd(
    output_path: str,
    base_image: np.ndarray,
    candidates: List[MeteorCandidate],
    homographies_map: Dict[int, Optional[np.ndarray]],
    include_sporadics: bool = False,
    boost_flux: float = 1.0
) -> bool:
    """
    Exporta la composición en un archivo PSD de Photoshop de 16 bits donde cada
    meteoro se aloja en su propia capa independiente con modo de fusión Trama (Screen).
    """
    if not HAS_PYTOSHOP:
        return False

    try:
        h, w = base_image.shape[:2]
        psd = pytoshop.core.PsdFile(num_channels=3, height=h, width=w, depth=16)

        # 1. Capa Fondo
        bg_16 = (np.clip(base_image, 0.0, 1.0) * 65535.0).astype(np.uint16)

        bg_layer = psd_layers.LayerRecord(
            name="Cielo y Suelo Base",
            top=0, left=0, bottom=h, right=w,
            blend_mode=psd_enums.BlendMode.normal,
            channels={
                0: psd_layers.ChannelImageData(image=bg_16[..., 0], compression=psd_enums.Compression.raw),
                1: psd_layers.ChannelImageData(image=bg_16[..., 1], compression=psd_enums.Compression.raw),
                2: psd_layers.ChannelImageData(image=bg_16[..., 2], compression=psd_enums.Compression.raw),
            }
        )
        psd.layer_and_mask_info.layer_info.layer_records.append(bg_layer)

        # 2. Capas individuales por cada meteoro
        active = [c for c in candidates if c.is_meteor and (include_sporadics or not c.is_sporadic)]
        if not active:
            active = [c for c in candidates if c.is_meteor]

        for idx, c in enumerate(active):
            try:
                src_img = load_image_as_float32(c.filepath)
                h_src, w_src = src_img.shape[:2]
                pad = max(24, int(round(c.thickness_px * 2.0)))
                x1, y1, x2, y2 = c.bbox
                x1_p = max(0, x1 - pad)
                y1_p = max(0, y1 - pad)
                x2_p = min(w_src, x2 + pad)
                y2_p = min(h_src, y2 + pad)

                patch = src_img[y1_p:y2_p, x1_p:x2_p]
                if patch.size == 0:
                    continue

                u_start = c.start_pt[0] - x1_p
                v_start = c.start_pt[1] - y1_p
                u_end = c.end_pt[0] - x1_p
                v_end = c.end_pt[1] - y1_p
                p_start = (int(round(u_start)), int(round(v_start)))
                p_end = (int(round(u_end)), int(round(v_end)))

                thick = max(6, int(round(c.thickness_px * 1.5)) + 4)
                mask_capsule = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
                cv2.line(mask_capsule, p_start, p_end, 255, thick)
                cv2.circle(mask_capsule, p_start, thick // 2, 255, -1)
                cv2.circle(mask_capsule, p_end, thick // 2, 255, -1)

                k_blur_cap = max(3, (thick // 2) * 2 + 1)
                mask_capsule_f = cv2.GaussianBlur(
                    mask_capsule.astype(np.float32) / 255.0, (k_blur_cap, k_blur_cap), thick / 3.0
                )

                # Estimar fondo local por canal RGB (percentil 30)
                is_sky = (mask_capsule == 0)
                if np.sum(is_sky) > 25:
                    bg_src = np.percentile(patch[is_sky], 30, axis=0)
                else:
                    bg_src = np.percentile(patch, 25, axis=(0, 1) if patch.ndim > 2 else 0)

                delta = patch - bg_src
                if delta.ndim == 3 and delta.shape[2] == 3:
                    delta_lum = 0.299 * delta[..., 0] + 0.587 * delta[..., 1] + 0.114 * delta[..., 2]
                    sky_delta_lum = delta_lum[is_sky] if np.sum(is_sky) > 25 else delta_lum
                else:
                    delta_lum = delta.squeeze() if delta.ndim == 3 else delta
                    sky_delta_lum = delta_lum[is_sky] if np.sum(is_sky) > 25 else delta_lum

                neg_sky = sky_delta_lum[sky_delta_lum < 0]
                if len(neg_sky) > 10:
                    sigma_sky = float(1.4826 * np.median(np.abs(neg_sky)))
                else:
                    sigma_sky = float(np.std(sky_delta_lum)) if sky_delta_lum.size > 0 else 0.003

                noise_floor = max(2.0 * sigma_sky, 0.003)
                noise_ceil = max(3.5 * sigma_sky, 0.008)

                mask_gate = np.clip((delta_lum - noise_floor) / max(noise_ceil - noise_floor, 1e-5), 0.0, 1.0)
                mask_final = mask_capsule_f * mask_gate
                mask_final = cv2.GaussianBlur(mask_final, (3, 3), 0.8)

                # Flujo neto aislado (0.0 estricto en el fondo del cielo)
                net_flux = np.maximum(0.0, delta) * boost_flux
                if net_flux.ndim == 3:
                    net_flux = net_flux * mask_final[..., None]
                else:
                    net_flux = net_flux * mask_final

                H = homographies_map.get(c.frame_index)
                if H is not None:
                    T = np.eye(3, dtype=np.float32)
                    T[0, 2] = x1_p
                    T[1, 2] = y1_p
                    H_c = H @ T
                    warped = cv2.warpPerspective(net_flux, H_c, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                else:
                    warped = np.zeros_like(base_image)
                    warped[y1_p:y2_p, x1_p:x2_p] = net_flux

                # Acotar el tamaño de la capa al área real del meteoro para un PSD ligero y manejable
                active_mask = (warped > 0.001)
                if active_mask.ndim == 3:
                    active_mask = np.any(active_mask, axis=2)

                ys, xs = np.where(active_mask)
                if len(ys) == 0:
                    continue

                pad_layer = 16
                top = max(0, int(np.min(ys)) - pad_layer)
                bottom = min(h, int(np.max(ys)) + pad_layer + 1)
                left = max(0, int(np.min(xs)) - pad_layer)
                right = min(w, int(np.max(xs)) + pad_layer + 1)

                cropped_layer = warped[top:bottom, left:right]
                m_16 = (np.clip(cropped_layer, 0.0, 1.0) * 65535.0).astype(np.uint16)

                m_layer = psd_layers.LayerRecord(
                    name=f"Meteoro #{c.id:02d} ({c.filename})",
                    top=top, left=left, bottom=bottom, right=right,
                    blend_mode=psd_enums.BlendMode.lighten,
                    channels={
                        0: psd_layers.ChannelImageData(image=m_16[..., 0], compression=psd_enums.Compression.raw),
                        1: psd_layers.ChannelImageData(image=m_16[..., 1], compression=psd_enums.Compression.raw),
                        2: psd_layers.ChannelImageData(image=m_16[..., 2], compression=psd_enums.Compression.raw),
                    }
                )
                psd.layer_and_mask_info.layer_info.layer_records.append(m_layer)
            except Exception:
                continue

        # Vista compuesta base para miniaturas rápidas
        from pytoshop import image_data as psd_image_data
        psd.image_data = psd_image_data.ImageData(
            channels=np.transpose(bg_16, (2, 0, 1)),
            compression=psd_enums.Compression.raw
        )

        with open(output_path, "wb") as f:
            psd.write(f)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# 7. SERIALIZACIÓN DE SESIÓN (.astrometeor)
# =============================================================================

def save_meteor_session(filepath: str, session_data: Dict[str, Any]) -> bool:
    """Guarda la configuración y la lista de meteoros analizados en formato JSON."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_meteor_session(filepath: str) -> Optional[Dict[str, Any]]:
    """Carga y valida un archivo de sesión .astrometeor."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None
