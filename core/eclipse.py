# core/eclipse.py
"""
core/eclipse.py - Motor matemático y algoritmos especializados para eclipses solares y lunares.

Agradecimientos y Referencias Científicas:
- Miloslav Druckmüller (Brno University of Technology): Fundamentos matemáticos pioneros de
  normalización de gradiente adaptativo y filtrado multiescala MGN (Morgan & Druckmüller 2014).
- Pere Guerra (PeterWar): Por la librería de código abierto Agentic-Eclipse-Library (AEL),
  sus investigaciones sobre artefactos de isófotas en fusión HDR, balance coronal neutro,
  preservación cromática de protuberancias y desacoplo de croma (https://github.com/PeterWar/Agentic-Eclipse-Library).

Incluye:
1. Extracción de metadatos EXIF (tiempo de exposición, ISO, apertura).
2. Detección de limbo de alta precisión (Sub-pixel Radial Edge Scanner + Ajuste Algebraico
   con rechazo RANSAC/MAD de protuberancias y perlas de Baily).
3. Traslación afín subpíxel y registro de series de bracketing por limbo.
4. Fusión HDR lineal con estirado de rango dinámico (Asinh) para levantar la corona externa.
5. Balance de blancos coronal neutro automático en anillo medio.
6. Filtro MGN multiescala con denoise de croma local y preservación de emisión H-alfa.
7. Filtro NRGF con compuerta radial (Radial Gate) y sustracción de ruido de fondo.
8. Realce de filamentos finos mediante filtro tangencial / bilateral enmascarado.
"""
import os
import cv2
import numpy as np
import datetime
from typing import Tuple, List, Dict, Optional, Any, Union
from PIL import Image, ExifTags
from scipy.optimize import minimize

try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from core.stacking import load_image_as_float32


def extract_exposure_info(filepath: str) -> Dict[str, Any]:
    """
    Extrae la información de exposición (tiempo de obturación, ISO, apertura, timestamp)
    desde los metadatos EXIF de la imagen (RAW, TIFF, JPEG).
    """
    info = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "exposure_s": 1.0,
        "exp_time": 1.0,
        "shutter_str": "1.0s",
        "iso": 100,
        "aperture": 0.0,
        "timestamp": 0.0,
        "time_str": "--:--:--",
    }

    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw']:
        try:
            import rawpy
            with rawpy.imread(filepath) as raw:
                if hasattr(raw, 'other'):
                    other = raw.other
                    shutter = getattr(other, 'shutter_speed', None) or getattr(other, 'shutter', None)
                    if shutter is not None and float(shutter) > 0:
                        info["exposure_s"] = float(shutter)
                    iso = getattr(other, 'iso_speed', None) or getattr(other, 'iso', None)
                    if iso is not None and float(iso) > 0:
                        info["iso"] = int(round(float(iso)))
                    aperture = getattr(other, 'aperture', None)
                    if aperture is not None and float(aperture) > 0:
                        info["aperture"] = float(aperture)

                    ts = getattr(other, 'timestamp', None)
                    if ts is not None:
                        if isinstance(ts, datetime.datetime):
                            info["timestamp"] = float(ts.timestamp())
                            info["time_str"] = ts.strftime("%H:%M:%S")
                        elif isinstance(ts, (int, float)) and ts > 0:
                            info["timestamp"] = float(ts)
                            try:
                                info["time_str"] = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                            except Exception:
                                pass
        except Exception:
            pass
    else:
        try:
            with Image.open(filepath) as img:
                exif_raw = img._getexif()
                if exif_raw:
                    exif = {
                        ExifTags.TAGS.get(k, k): v
                        for k, v in exif_raw.items()
                    }

                    exp = exif.get("ExposureTime")
                    if exp is not None:
                        try:
                            exp_val = float(exp)
                            if exp_val > 0:
                                info["exposure_s"] = exp_val
                        except (ValueError, TypeError, ZeroDivisionError):
                            pass

                    iso = exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")
                    if iso is not None:
                        try:
                            if isinstance(iso, (list, tuple)):
                                info["iso"] = int(iso[0])
                            else:
                                info["iso"] = int(iso)
                        except (ValueError, TypeError):
                            pass

                    fnum = exif.get("FNumber")
                    if fnum is not None:
                        try:
                            info["aperture"] = float(fnum)
                        except (ValueError, TypeError):
                            pass

                    date_str = exif.get("DateTimeOriginal") or exif.get("DateTimeDigitized") or exif.get("DateTime")
                    if date_str and isinstance(date_str, str):
                        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                            try:
                                dt = datetime.datetime.strptime(date_str.strip(), fmt)
                                info["timestamp"] = float(dt.timestamp())
                                info["time_str"] = dt.strftime("%H:%M:%S")
                                break
                            except ValueError:
                                pass
        except Exception:
            pass

    # Fallback si no hubo timestamp EXIF
    if info.get("timestamp", 0.0) <= 0.0:
        try:
            mtime = os.path.getmtime(filepath)
            info["timestamp"] = float(mtime)
            info["time_str"] = datetime.datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        except Exception:
            pass

    exp_val = info["exposure_s"]
    info["exp_time"] = exp_val
    if exp_val < 0.99:
        denom = int(round(1.0 / exp_val))
        info["shutter_str"] = f"1/{denom}s"
    else:
        if abs(exp_val - round(exp_val)) < 0.05:
            info["shutter_str"] = f"{int(round(exp_val))}s"
        else:
            info["shutter_str"] = f"{exp_val:.2f}s"

    return info


def _fit_circle_algebraic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Ajuste algebraico de círculo por mínimos cuadrados (Pratt / Kasa)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx = c[0] / 2.0
    cy = c[1] / 2.0
    r = np.sqrt(max(c[2] + cx**2 + cy**2, 1e-4))
    return float(cx), float(cy), float(r)


def _radial_limb_scan(
    img_gray: np.ndarray,
    init_cx: float,
    init_cy: float,
    r_min: float,
    r_max: float,
    num_rays: int = 360,
    mode: str = "solar"
) -> Optional[Tuple[float, float, float]]:
    """
    Escanea rayos radiales desde un centro aproximado para detectar la posición
    exacta del limbo con precisión subpíxel y rechazo robusto de protuberancias / Baily's beads.
    Calcula residuales respecto al círculo ajustado y filtra por MAD para eliminar protuberancias.
    """
    if r_max <= r_min + 3:
        return None

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    r_steps = np.arange(r_min, r_max, 0.5, dtype=np.float32)

    xs = init_cx + np.outer(np.cos(angles), r_steps)
    ys = init_cy + np.outer(np.sin(angles), r_steps)

    samples = cv2.remap(img_gray, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    grad = np.diff(samples, axis=1)

    if mode == "solar":
        max_idx = np.argmax(grad, axis=1)
        peak_vals = np.take_along_axis(grad, max_idx[:, None], axis=1).ravel()
    else:
        max_idx = np.argmin(grad, axis=1)
        peak_vals = -np.take_along_axis(grad, max_idx[:, None], axis=1).ravel()

    detected_r = r_steps[max_idx]

    valid_grad = peak_vals > np.percentile(peak_vals, 15)
    if np.sum(valid_grad) < 20:
        return None

    r_valid = detected_r[valid_grad]
    a_valid = angles[valid_grad]

    xin = init_cx + r_valid * np.cos(a_valid)
    yin = init_cy + r_valid * np.sin(a_valid)

    # 1. Ajuste preliminar del círculo
    cxt, cyt, rt = _fit_circle_algebraic(xin, yin)

    # 2. Rechazo robusto de protuberancias basado en residual radial al círculo
    dist = np.sqrt((xin - cxt)**2 + (yin - cyt)**2)
    residual = np.abs(dist - rt)
    med_res = float(np.median(residual))
    mad_res = float(1.4826 * np.median(np.abs(residual - med_res)))
    inliers = residual < max(2.5 * mad_res, 3.0)

    if np.sum(inliers) < 15:
        return float(cxt), float(cyt), float(rt)

    cx_final, cy_final, r_final = _fit_circle_algebraic(xin[inliers], yin[inliers])
    return float(cx_final), float(cy_final), float(r_final)


def optimize_center_circular_flux(
    gray: np.ndarray,
    init_cx: float,
    init_cy: float,
    radius: float,
    mode: str = "solar",
    max_search_dist: Optional[float] = None
) -> Tuple[float, float]:
    """
    Optimiza las coordenadas del centro (cx, cy) a precisión subpíxel milimétrica (< 0.1 px)
    con un radio fijado r = radius, maximizando el flujo del gradiente radial perpendicular al limbo.
    
    Para modo solar:
      - Maximiza el gradiente radial exterior hacia la corona/fotosfera (percentil 75).
      - Penaliza fuertemente cualquier contaminación luminosa dentro del disco lunar (r < R).
      - Inmune a destellos potentes (anillo de diamante y perlas de Baily) que distorsionan medianas globales.
      - Acotado rígidamente dentro de los límites de la imagen y ventana de búsqueda máxima.
    """
    h, w = gray.shape[:2]
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    r_in = max(1.0, radius - 2.5)
    r_out = radius + 2.5
    r_int = radius * 0.50

    max_dev = float(max_search_dist) if (max_search_dist and max_search_dist > 0) else min(35.0, max(12.0, radius * 0.12))

    def loss(p):
        cx, cy = p[0], p[1]
        dist_dev = np.hypot(cx - init_cx, cy - init_cy)
        if dist_dev > max_dev or cx < 0 or cx >= w or cy < 0 or cy >= h:
            return 1e6 + dist_dev * 1000.0

        xs_in = cx + r_in * cos_a
        ys_in = cy + r_in * sin_a
        xs_out = cx + r_out * cos_a
        ys_out = cy + r_out * sin_a
        v_in = cv2.remap(gray, xs_in.astype(np.float32)[:, None], ys_in.astype(np.float32)[:, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()
        v_out = cv2.remap(gray, xs_out.astype(np.float32)[:, None], ys_out.astype(np.float32)[:, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()
        
        if mode == "solar":
            diff = v_out - v_in
            # En anillo de diamante o creciente, el gradiente relevante está en el 25% superior de ángulos
            grad = float(np.percentile(diff, 75))
            xs_int = cx + r_int * cos_a
            ys_int = cy + r_int * sin_a
            v_int = cv2.remap(gray, xs_int.astype(np.float32)[:, None], ys_int.astype(np.float32)[:, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel()
            pen_in = float(np.mean(v_int))
            eff_pen = min(pen_in * 0.5, max(0.0, grad * 0.4))
            return -(grad - eff_pen)
        else:
            diff = v_in - v_out
            return -float(np.percentile(diff, 60))

    best_c = (init_cx, init_cy)
    best_loss = loss(best_c)
    search_range = int(min(max_dev, 24.0))
    step = 2 if search_range <= 16 else 3
    for dx in range(-search_range, search_range + 1, step):
        for dy in range(-search_range, search_range + 1, step):
            c = (init_cx + dx, init_cy + dy)
            l = loss(c)
            if l < best_loss:
                best_loss = l
                best_c = c

    res = minimize(loss, list(best_c), method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-4, "maxiter": 50})
    final_cx, final_cy = float(res.x[0]), float(res.x[1])
    if np.hypot(final_cx - init_cx, final_cy - init_cy) > max_dev or final_cx < 0 or final_cx >= w or final_cy < 0 or final_cy >= h or loss((final_cx, final_cy)) > best_loss:
        return float(best_c[0]), float(best_c[1])
    return final_cx, final_cy


def fit_circular_arc_convex_hull(
    hull: np.ndarray, 
    expected_radius: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Ajusta un arco circular sobre el contorno exterior (convex hull) de un creciente solar,
    anillo de diamante o perlas de Baily mediante el teorema analítico de cuerda-sagita.
    
    Evita la trampa de minEnclosingCircle que sitúa el centro en el punto medio de la cuerda de la hoz.
    Calcula el centro astronómico real del disco solar/lunar que genera la curvatura con precisión subpíxel.
    """
    if len(hull) < 3:
        (cx, cy), r = cv2.minEnclosingCircle(hull)
        return float(cx), float(cy), float(expected_radius or r)

    # 1. Encontrar los dos puntos más alejados del contorno (extremos de la cuerda)
    diff = hull[:, np.newaxis, :] - hull[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    max_idx = np.unravel_index(np.argmax(dists_sq), dists_sq.shape)
    p0 = hull[max_idx[0]]
    p1 = hull[max_idx[1]]

    chord_dx = float(p1[0] - p0[0])
    chord_dy = float(p1[1] - p0[1])
    chord_len = float(np.hypot(chord_dx, chord_dy))
    if chord_len < 2.0:
        return float(p0[0]), float(p0[1]), float(expected_radius or 10.0)

    mid = (p0 + p1) / 2.0
    normal = np.array([-chord_dy, chord_dx], dtype=np.float32) / chord_len

    # Sagita: distancia perpendicular máxima de los puntos del contorno a la cuerda
    # Producto cruz 2D: |dx * (p0_y - y) - dy * (p0_x - x)| / chord_len
    line_dists = np.abs(chord_dx * (p0[1] - hull[:, 1]) - chord_dy * (p0[0] - hull[:, 0])) / chord_len
    h_sagitta = float(np.max(line_dists))

    if expected_radius and expected_radius > (chord_len * 0.45):
        R = float(expected_radius)
    else:
        # Teorema de la cuerda y la sagita: R = h/2 + c^2 / (8h)
        if h_sagitta > 1.0:
            R = (h_sagitta / 2.0) + (chord_len ** 2) / (8.0 * h_sagitta)
        else:
            R = max(chord_len * 0.70, 20.0)

    # Distancia del punto medio de la cuerda al centro del círculo
    if R >= chord_len * 0.5:
        d_center = float(np.sqrt(max(0.0, R**2 - (chord_len / 2.0)**2)))
    else:
        d_center = 0.0

    # Las dos soluciones geométricas perpendiculares a la cuerda
    cand1 = mid + normal * d_center
    cand2 = mid - normal * d_center

    # Seleccionar la solución que minimiza la desviación con el contorno
    loss1 = float(np.sum((np.hypot(hull[:, 0] - cand1[0], hull[:, 1] - cand1[1]) - R) ** 2))
    loss2 = float(np.sum((np.hypot(hull[:, 0] - cand2[0], hull[:, 1] - cand2[1]) - R) ** 2))
    best_c = cand1 if loss1 < loss2 else cand2

    # Refinamiento Gauss-Newton rápido (5 iteraciones)
    cx, cy = float(best_c[0]), float(best_c[1])
    for _ in range(5):
        dists = np.hypot(hull[:, 0] - cx, hull[:, 1] - cy)
        err = dists - R
        d_safe = np.maximum(dists, 1e-4)
        dx = (hull[:, 0] - cx) / d_safe
        dy = (hull[:, 1] - cy) / d_safe
        cx += float(np.mean(err * dx)) * 0.5
        cy += float(np.mean(err * dy)) * 0.5

    return float(cx), float(cy), float(R)


def detect_eclipse_disk(
    image: np.ndarray,
    mode: str = "solar",
    expected_radius: Optional[float] = None,
    expected_center: Optional[Tuple[float, float]] = None
) -> Tuple[float, float, float]:
    """
    Detecta automáticamente las coordenadas del centro (cx, cy) y el radio (r)
    del disco del eclipse (Sol o Luna) con precisión subpíxel.
    
    1. Localización inicial topológica:
       - Solar: detección del disco lunar como hueco interior dentro del anillo de corona brillante.
       - Lunar: detección del contorno exterior de la Luna brillante.
    2. Si expected_radius / expected_center están disponibles:
       - Ancla exactamente el radio físico de la Luna (invariable durante la totalidad).
       - Ancla la búsqueda en torno al centro temporal esperado, evitando saltos en destellos de diamante.
       - Optimiza (cx, cy) con flujo de gradiente radial mediano, ignorando al 100% protuberancias.
    3. Si expected_radius es None:
       - Realiza escaneo radial subpíxel en dos pasadas con filtrado MAD de residuales.
       - Refina el centro subpíxel resultante mediante flujo circular.
    """
    h, w = image.shape[:2]
    if image.ndim == 3:
        gray = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    else:
        gray = image.copy()
    gray = gray.astype(np.float32)

    # Coarse detection via downsampled image (1000 px max, sin sobreescalar)
    scale = min(1.0, 1000.0 / float(max(h, w)))
    small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
    small_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    exp_c_s = (expected_center[0] * scale, expected_center[1] * scale) if expected_center is not None else None

    init_cx, init_cy, init_r = None, None, None

    if mode == "solar":
        best_hole = None
        max_val = float(small_gray.max())

        # 1. Detección multiumbral del disco lunar (hueco oscuro cerrado dentro de la corona)
        # Inmune a destellos potentes (anillo de diamante) donde max_val=1.0 pero la corona está a 0.015-0.080
        hole_thresholds = [0.012, 0.018, 0.025, 0.038, 0.060, 0.090, 0.140]
        kernel_hole = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        best_hole_cand = None
        best_hole_score = -1e9

        for th_h in hole_thresholds:
            bright = (small_gray > th_h).astype(np.uint8) * 255
            bright_closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel_hole)
            cnts_h, hierarchy = cv2.findContours(bright_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if hierarchy is None:
                continue

            for i, h_info in enumerate(hierarchy[0]):
                if h_info[3] != -1:  # Hueco interior
                    c_h = cnts_h[i]
                    area_h = cv2.contourArea(c_h)
                    if area_h < 100:
                        continue
                    (hcx_t, hcy_t), hr_t = cv2.minEnclosingCircle(c_h)
                    perim_h = cv2.arcLength(c_h, True)
                    circ_h = 4 * np.pi * area_h / (perim_h ** 2 + 1e-6)
                    if circ_h < 0.45 or hr_t < 15:
                        continue

                    score_h = circ_h * 2.0
                    if expected_radius:
                        exp_r_s = expected_radius * scale
                        r_err = abs(hr_t - exp_r_s) / max(1.0, exp_r_s)
                        if r_err > 0.45:
                            continue
                        score_h += (1.0 - r_err) * 3.0

                    if exp_c_s is not None:
                        c_dist = np.hypot(hcx_t - exp_c_s[0], hcy_t - exp_c_s[1])
                        if c_dist > max(25.0, (expected_radius * scale * 0.40) if expected_radius else 25.0):
                            continue
                        score_h += max(0.0, 1.0 - c_dist / 50.0) * 4.0

                    if score_h > best_hole_score:
                        best_hole_score = score_h
                        best_hole_cand = (hcx_t, hcy_t, hr_t, c_h)

        if best_hole_cand is not None:
            hcx, hcy, hr, best_hole = best_hole_cand
            init_cx = float(hcx / scale)
            init_cy = float(hcy / scale)
            init_r = float(hr / scale)
        else:
            # 2. En fases parciales o anillo de diamante sin corona cerrada:
            if max_val > 0.02:
                thresh_val = max(0.015, max_val * 0.35)
                bright_crescent = (small_gray > thresh_val).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(bright_crescent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c_best = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c_best) > 15:
                        hull = cv2.convexHull(c_best, returnPoints=True)[:, 0, :].astype(np.float32)
                        diff_pts = hull[:, np.newaxis, :] - hull[np.newaxis, :, :]
                        chord_len_s = float(np.sqrt(np.max(np.sum(diff_pts ** 2, axis=-1))))
                        exp_r_scaled = (expected_radius * scale) if (expected_radius and expected_radius > 10) else None
                        ref_r_s = exp_r_scaled or chord_len_s

                        # Si la cuerda es amplia (>= 0.60 * R), es un creciente real con arco exterior:
                        if chord_len_s >= ref_r_s * 0.60:
                            hcx, hcy, hr = fit_circular_arc_convex_hull(hull, expected_radius=exp_r_scaled)
                            init_cx = float(hcx / scale)
                            init_cy = float(hcy / scale)
                            init_r = float(hr / scale)
                        else:
                            # Destello concentrado (anillo de diamante o perla puntual):
                            if expected_center is not None and expected_radius:
                                init_cx = float(expected_center[0])
                                init_cy = float(expected_center[1])
                                init_r = float(expected_radius)
                            elif expected_radius and expected_radius > 10:
                                flare_center = np.mean(hull, axis=0)
                                best_cand_c = flare_center
                                best_cand_val = 1e9
                                for a_c in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                                    c_cand = flare_center + np.array([np.cos(a_c), np.sin(a_c)]) * (expected_radius * scale)
                                    cx_i = int(c_cand[0])
                                    cy_i = int(c_cand[1])
                                    if 0 <= cx_i < small_w and 0 <= cy_i < small_h:
                                        val_int = float(small_gray[cy_i, cx_i])
                                        if val_int < best_cand_val:
                                            best_cand_val = val_int
                                            best_cand_c = c_cand
                                init_cx = float(best_cand_c[0] / scale)
                                init_cy = float(best_cand_c[1] / scale)
                                init_r = float(expected_radius)
                            else:
                                hcx, hcy, hr = fit_circular_arc_convex_hull(hull, expected_radius=exp_r_scaled)
                                init_cx = float(hcx / scale)
                                init_cy = float(hcy / scale)
                                init_r = float(hr / scale)
    else:
        # En eclipse lunar, la Luna es el objeto brillante circular.
        # Evitamos percentiles globales fijos (p80) que seleccionan nubes o resplandor difuso.
        small_u8 = np.clip(small_gray, 0, 255).astype(np.uint8) if small_gray.max() > 1.5 else np.clip(small_gray * 255.0, 0, 255).astype(np.uint8)

        # 1. Localizar el punto de brillo lunar máximo (filtrando ruido puntual con Gaussiano)
        peak_blur = cv2.GaussianBlur(small_u8, (31, 31), 0)
        _, _, _, (peak_x, peak_y) = cv2.minMaxLoc(peak_blur)

        # 2. Detección de círculos mediante transformada de Hough multiescala/gradiente
        blur = cv2.GaussianBlur(small_u8, (9, 9), 2)
        min_r = max(5, int(15 * scale))
        max_r = max(min_r + 20, int(min(small_w, small_h) * 0.45))

        for p2 in [35, 28, 22, 16, 12]:
            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, int(20 * scale)),
                param1=80, param2=p2, minRadius=min_r, maxRadius=max_r
            )
            if circles is not None and len(circles[0]) > 0:
                cands = circles[0]
                # Priorizar el círculo que contenga o esté más próximo a la zona de brillo lunar
                valid_cands = []
                for c in cands:
                    cx_c, cy_c, r_c = c[0], c[1], c[2]
                    dist = np.hypot(cx_c - peak_x, cy_c - peak_y)
                    if dist <= max(r_c * 1.6, 40.0):
                        valid_cands.append((dist, cx_c, cy_c, r_c))
                if valid_cands:
                    valid_cands.sort(key=lambda item: item[0])
                    best = valid_cands[0]
                    init_cx = float(best[1] / scale)
                    init_cy = float(best[2] / scale)
                    init_r = float(best[3] / scale)
                    break
                best = min(cands, key=lambda c: np.hypot(c[0] - small_w / 2, c[1] - small_h / 2))
                init_cx = float(best[0] / scale)
                init_cy = float(best[1] / scale)
                init_r = float(best[2] / scale)
                break

        # 3. Fallback: umbralización de Otsu en una ventana local centrada en el brillo lunar
        if init_cx is None:
            roi_size = int(max(min(small_w, small_h) * 0.3, 100))
            x0 = max(0, peak_x - roi_size)
            y0 = max(0, peak_y - roi_size)
            x1 = min(small_w, peak_x + roi_size)
            y1 = min(small_h, peak_y + roi_size)
            roi = small_u8[y0:y1, x0:x1]
            _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c_best = max(cnts, key=cv2.contourArea)
                (hcx, hcy), hr = cv2.minEnclosingCircle(c_best)
                init_cx = float((hcx + x0) / scale)
                init_cy = float((hcy + y0) / scale)
                init_r = float(hr / scale)

    if init_cx is None or init_r is None or init_r < 10:
        if expected_center is not None:
            init_cx = float(expected_center[0])
            init_cy = float(expected_center[1])
        else:
            init_cx = w / 2.0
            init_cy = h / 2.0
        init_r = expected_radius if (expected_radius and expected_radius > 10) else min(w, h) * 0.18

    # Si expected_radius está disponible y es válido, optimizar centro anclando R
    if expected_radius is not None and expected_radius > 10:
        max_search = min(20.0, max(8.0, expected_radius * 0.08)) if expected_center is not None else None
        opt_cx, opt_cy = optimize_center_circular_flux(gray, init_cx, init_cy, expected_radius, mode=mode, max_search_dist=max_search)
        if expected_center is not None:
            if np.hypot(opt_cx - expected_center[0], opt_cy - expected_center[1]) > 25.0:
                opt_cx, opt_cy = expected_center
        return float(opt_cx), float(opt_cy), float(expected_radius)

    if mode == "solar" and best_hole is None:
        # En fase parcial o contacto solar sin radio fijado previo, optimizamos con el radio estimado del arco
        if init_r and init_r > 15:
            opt_cx, opt_cy = optimize_center_circular_flux(gray, init_cx, init_cy, init_r, mode=mode)
            return float(opt_cx), float(opt_cy), float(init_r)
        return float(init_cx), float(init_cy), float(init_r)

    # Detección completa de centro y radio (Totalidad solar o eclipse lunar)
    r_min = max(5.0, init_r - 35.0)
    r_max = init_r + 35.0

    p1 = _radial_limb_scan(gray, init_cx, init_cy, r_min, r_max, num_rays=360, mode=mode)
    if p1 is not None:
        p1_cx, p1_cy, p1_r = p1
        p2 = _radial_limb_scan(gray, p1_cx, p1_cy, p1_r - 15.0, p1_r + 15.0, num_rays=360, mode=mode)
        p_use = p2 if p2 is not None else p1
        final_cx, final_cy = optimize_center_circular_flux(gray, p_use[0], p_use[1], p_use[2], mode=mode)
        final_r = p_use[2]
    else:
        final_cx, final_cy, final_r = init_cx, init_cy, init_r

    return float(final_cx), float(final_cy), float(final_r)


def refine_disk_subpixel(
    image: np.ndarray,
    approx_cx: float,
    approx_cy: float,
    approx_radius: float,
    mode: str = "solar"
) -> Tuple[float, float, float]:
    """
    Refina las coordenadas de centro y radio a precisión subpíxel milimétrica (< 0.1 px)
    a partir de una estimación previa del usuario, directamente sobre la imagen a resolución completa.
    """
    if image.ndim == 3:
        gray = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    else:
        gray = image.copy()
    gray = gray.astype(np.float32)

    fcx, fcy = optimize_center_circular_flux(gray, approx_cx, approx_cy, approx_radius, mode=mode)
    return float(fcx), float(fcy), float(approx_radius)



def register_frame_to_center(
    image: np.ndarray,
    current_center: Tuple[float, float],
    target_center: Tuple[float, float],
    interpolation: int = cv2.INTER_LANCZOS4,
    border_mode: int = cv2.BORDER_REPLICATE
) -> np.ndarray:
    """
    Desplaza una toma mediante transformación afín subpíxel para que su centro
    del limbo (current_center) coincida exactamente con target_center.
    """
    dx = float(target_center[0] - current_center[0])
    dy = float(target_center[1] - current_center[1])

    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return image.copy()

    h, w = image.shape[:2]
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=interpolation,
        borderMode=border_mode
    )


def crop_frame_roi(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    crop_size: int = 3000
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Recorta un recuadro cuadrado (ROI) centrado en el limbo del eclipse de tamaño crop_size,
    garantizando dimensiones idénticas en todas las tomas y actualizando el centro local.
    """
    h, w = image.shape[:2]
    c_w = min(int(crop_size), w)
    c_h = min(int(crop_size), h)
    half_w = c_w // 2
    half_h = c_h // 2

    x0 = int(round(center_x - half_w))
    y0 = int(round(center_y - half_h))

    if x0 < 0:
        x0 = 0
    elif x0 + c_w > w:
        x0 = max(0, w - c_w)

    if y0 < 0:
        y0 = 0
    elif y0 + c_h > h:
        y0 = max(0, h - c_h)

    x1 = x0 + c_w
    y1 = y0 + c_h

    cropped = image[y0:y1, x0:x1].copy()
    new_c = (float(center_x - x0), float(center_y - y0), float(radius))
    return cropped, new_c


def align_eclipse_bracketing(
    images: List[np.ndarray],
    centers: List[Tuple[float, float]],
    ref_idx: int = 0
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    Alinea una lista de imágenes de bracketing al centro del limbo de la toma de referencia.
    """
    if not images or len(images) != len(centers):
        raise ValueError("El número de imágenes y centros debe coincidir.")

    ref_center = centers[ref_idx]
    aligned_images = []
    shifts = []

    for img, c in zip(images, centers):
        dx = float(ref_center[0] - c[0])
        dy = float(ref_center[1] - c[1])
        shifts.append((dx, dy))

        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            aligned_images.append(img.copy())
        else:
            aligned = register_frame_to_center(img, c, ref_center, border_mode=cv2.BORDER_REPLICATE)
            aligned_images.append(aligned)

    return aligned_images, shifts


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Función de interpolación hermítica cúbica suave en [edge0, edge1]."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_shoulder(peak: np.ndarray, threshold: float = 0.70, ceiling: float = 1.0) -> np.ndarray:
    """Atenuación suave exponencial en altas luces (evita corte duro o meseta a blanco)."""
    res = np.copy(peak)
    mask = peak > threshold
    p = peak[mask] - threshold
    span = max(ceiling - threshold, 1e-4)
    res[mask] = threshold + span * (1.0 - np.exp(-p / span))
    return res


def apply_asinh_stretch(
    image: np.ndarray,
    stretch_factor: float = 30.0,
    black_point: float = 0.0,
    shoulder_threshold: float = 0.70,
    highlight_ceiling: float = 1.0
) -> np.ndarray:
    """
    Estirado de arco seno hiperbólico (asinh) con hombro suave de altas luces (soft highlight shoulder).
    Preserva los ratios cromáticos exactos R:G:B y comprime el rango dinámico
    extremo de la corona solar: levanta la corona externa tenue sin empastar
    ni saturar protuberancias a blanco puro.
    """
    if stretch_factor <= 1.0:
        return np.clip(image - black_point, 0.0, 1.0).astype(np.float32)

    img = np.maximum(image - black_point, 0.0).astype(np.float32)
    denom = np.arcsinh(float(stretch_factor))
    u = np.arcsinh(float(stretch_factor) * img) / denom

    if img.ndim == 3:
        peak = np.max(u, axis=-1)
        sh = soft_shoulder(peak, threshold=shoulder_threshold, ceiling=highlight_ceiling)
        factor = np.divide(sh, peak, out=np.ones_like(peak), where=peak > 1e-6)
        stretched = u * factor[..., None]
    else:
        stretched = soft_shoulder(u, threshold=shoulder_threshold, ceiling=highlight_ceiling)

    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def balance_coronal_white_balance(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    ring_range: Tuple[float, float] = (1.2, 1.8)
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Neutraliza el balance de blancos de la corona solar estimando la mediana de color
    en el anillo coronal medio (1.2 - 1.8 R), eliminando dominantes cálidas o del
    cielo crepuscular según las directrices de AEL (white_balance_on_corona).
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 3:
        return img, (1.0, 1.0, 1.0)

    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.hypot(x - center_x, y - center_y)
    r_lo = radius * float(ring_range[0])
    r_hi = radius * float(ring_range[1])
    ring = (dist >= r_lo) & (dist <= r_hi)

    if np.sum(ring) < 100:
        return img, (1.0, 1.0, 1.0)

    med = np.median(img[ring], axis=0)
    g_val = float(med[1]) if med[1] > 1e-6 else float(np.mean(med))
    f = g_val / np.maximum(med, 1e-6)
    f = np.clip(f, 0.4, 2.5)

    balanced = np.clip(img * f[None, None, :], 0.0, 1.0).astype(np.float32)
    return balanced, (float(f[0]), float(f[1]), float(f[2]))


def fuse_hdr_bracketing(
    images: List[np.ndarray],
    exposures_s: List[float],
    saturation_threshold: float = 0.90,
    noise_floor: float = 0.015,
    asinh_stretch: float = 0.0,
    mode: str = "solar",
    lunar_mask_params: Optional[Tuple[float, float, float]] = None,
    subtract_sky: bool = True,
    auto_wb_corona: bool = True,
    gradual_handover: bool = True
) -> np.ndarray:
    """
    Fusión de alto rango dinámico (HDR) para tomas de bracketing de eclipses.
    - Modo solar: Fusión lineal fotométrica con pesos unificados en luminancia (inmune a
      aberración cromática en saturación), sustracción de pedestal de cielo por fotograma
      y protección de disco lunar frente a dispersión de tomas largas.
    - Modo lunar: Fusión por exposición multiescala (Mertens-Kautz-Van Reeth),
      óptima para el contraste extremo entre la zona iluminada y la umbra rojiza.
    """
    if not images or len(images) != len(exposures_s):
        raise ValueError("La lista de imágenes y tiempos de exposición debe ser consistente.")

    if mode == "lunar":
        imgs_for_mertens = []
        for img in images:
            if img.dtype == np.uint8:
                c_img = img
            else:
                max_v = float(img.max()) if img.size > 0 else 1.0
                if max_v <= 1.05:
                    c_img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
                else:
                    c_img = np.clip(img, 0.0, 255.0).astype(np.uint8)
            imgs_for_mertens.append(c_img)

        mertens = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=1.0,
            exposure_weight=1.0
        )
        fused = mertens.process(imgs_for_mertens)
        fused = np.clip(fused, 0.0, 1.0).astype(np.float32)

        if asinh_stretch > 1.0:
            fused = apply_asinh_stretch(fused, stretch_factor=asinh_stretch)

        return fused

    order = np.argsort(exposures_s)
    sorted_images = [np.asarray(images[i], dtype=np.float32) for i in order]
    sorted_exp = [float(exposures_s[i]) for i in order]

    shape = sorted_images[0].shape
    weighted_flux = np.zeros(shape, dtype=np.float64)
    sum_weights = np.zeros(shape, dtype=np.float64)

    sat_lo = max(0.40, saturation_threshold - 0.20)
    sat_hi = saturation_threshold

    # Máscara lunar para evitar que flare óptico de tomas largas invada el disco
    moon_weight = None
    if lunar_mask_params is not None:
        mcx, mcy, mr = lunar_mask_params
        y_idx, x_idx = np.indices(shape[:2], dtype=np.float32)
        dist = np.hypot(x_idx - mcx, y_idx - mcy)
        moon_weight = _smoothstep(dist, mr - 1.5, mr + 1.0)

    for i, (img, t) in enumerate(zip(sorted_images, sorted_exp)):
        t = max(t, 1e-6)
        shorter_t = sorted_exp[i - 1] if i > 0 else None

        # 1. Sustracción de pedestal/fondo por fotograma para anivelar el cielo
        if subtract_sky:
            if img.ndim == 3:
                bg = np.percentile(img, 0.5, axis=(0, 1), keepdims=True)
            else:
                bg = np.percentile(img, 0.5)
            clean_img = np.maximum(img - bg, 0.0)
        else:
            clean_img = img

        flux = clean_img.astype(np.float64) / t

        # 2. Pesos unificados sobre luminancia y canal máximo (evita franjas de color)
        if img.ndim == 3:
            lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            max_c = np.max(img, axis=-1)
        else:
            lum = img
            max_c = img

        w_low = _smoothstep(lum, 0.0, noise_floor * 2.0)
        w_high = 1.0 - _smoothstep(max_c, sat_lo, sat_hi)
        w = (w_low * w_high).astype(np.float64)

        # Modulación gradual del peso de la toma más larga (AEL 1.4.0 / handover)
        # Atenúa la caída abrupta de varianza distribuyendo el peso con la exposición más corta,
        # erradicando falsos anillos y contornos en isófotas de saturación en filtros posteriores.
        if gradual_handover and shorter_t is not None and shorter_t > 0:
            taper = w_high
            weight_factor = taper + (1.0 - taper) * (t / shorter_t)
            w = w / np.maximum(weight_factor, 1e-4)

        if moon_weight is not None:
            w = w * moon_weight

        if img.ndim == 3:
            w_3d = w[..., None]
            weighted_flux += flux * w_3d
            sum_weights += w_3d
        else:
            weighted_flux += flux * w
            sum_weights += w

    zero_mask = sum_weights < 1e-6
    if np.any(zero_mask):
        shortest_img = sorted_images[0]
        shortest_t = sorted_exp[0]
        longest_img = sorted_images[-1]
        longest_t = sorted_exp[-1]

        if sorted_images[0].ndim == 3:
            shortest_max = np.max(shortest_img, axis=-1, keepdims=True)
        else:
            shortest_max = shortest_img

        fallback_flux = np.where(
            shortest_max >= sat_lo,
            shortest_img.astype(np.float64) / shortest_t,
            longest_img.astype(np.float64) / longest_t
        )
        weighted_flux = np.where(zero_mask, fallback_flux, weighted_flux)
        sum_weights = np.where(zero_mask, 1.0, sum_weights)

    merged_flux = weighted_flux / sum_weights

    if moon_weight is not None:
        if merged_flux.ndim == 3:
            merged_flux = merged_flux * moon_weight[..., None]
        else:
            merged_flux = merged_flux * moon_weight

    # Normalización sobre píxeles válidos fuera del disco lunar
    if moon_weight is not None:
        valid_px = moon_weight > 0.5
        p_high = float(np.percentile(merged_flux[valid_px], 99.999))
        p_low = float(np.percentile(merged_flux[valid_px], 0.05))
    else:
        p_high = float(np.percentile(merged_flux, 99.999))
        p_low = float(np.percentile(merged_flux, 0.05))

    denom = max(p_high - p_low, 1e-8)
    hdr_norm = np.clip((merged_flux - p_low) / denom, 0.0, 1.0).astype(np.float32)

    # Balance de blancos coronal automático (anillo 1.2 - 1.8 R) para eliminar dominantes de cielo
    if auto_wb_corona and lunar_mask_params is not None and hdr_norm.ndim == 3:
        mcx, mcy, mr = lunar_mask_params
        hdr_norm, _ = balance_coronal_white_balance(hdr_norm, mcx, mcy, mr, ring_range=(1.2, 1.8))

    if asinh_stretch > 1.0:
        hdr_norm = apply_asinh_stretch(hdr_norm, stretch_factor=asinh_stretch)

    return hdr_norm


def display_detail_rings(
    detail: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    width_px: float = 2.0,
    min_count: int = 50,
    smooth_bins: int = 3,
    noise_floor: float = 1e-5,
    return_info: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Normaliza y mapea el detalle firmado mediante nivel y dispersión medidos anillo por
    anillo en torno al centro solar (AEL 1.3.4 / Issue #9 de Eduardo Piñas).

    1. Calcula mediana y 1.4826·MAD en anillos de ancho width_px (mínimo min_count píxeles).
    2. Suaviza perfiles radiales sobre smooth_bins anillos consecutivos para eliminar escalonamiento.
    3. Interpola linealmente continuo en r y aplica mapeo sigmoidal:
       0.5 + 0.5 * tanh((detail - level(r)) / (2.5 * spread(r)))
    4. Evita que la corona interna se sature a blanco/negro puro al tiempo que conserva
       contraste uniforme a través de todo el radio solar.
    """
    det = np.asarray(detail, dtype=np.float32)
    if det.ndim == 3:
        out = np.empty_like(det)
        for c in range(det.shape[2]):
            out[..., c] = display_detail_rings(
                det[..., c],
                valid_mask=valid_mask,
                center_x=center_x,
                center_y=center_y,
                width_px=width_px,
                min_count=min_count,
                smooth_bins=smooth_bins,
                noise_floor=noise_floor,
                return_info=False
            )
        if return_info:
            return out, {"display": "level and spread per ring (3D)", "ring_width_px": width_px}
        return out

    if HAS_RUST and hasattr(astroloom_rs, "display_detail_rings_rs"):
        det_c = np.ascontiguousarray(det, dtype=np.float32)
        mask_c = np.ascontiguousarray(valid_mask.astype(bool)) if valid_mask is not None else None
        res = np.asarray(astroloom_rs.display_detail_rings_rs(
            det_c,
            mask_c,
            float(center_x),
            float(center_y),
            float(width_px),
            int(min_count),
            int(smooth_bins),
            float(noise_floor)
        ), dtype=np.float32)
        if return_info:
            return res, {"display": "level and spread per ring (Rust)", "ring_width_px": width_px}
        return res

    # Fallback NumPy
    h, w = det.shape[:2]
    good = np.isfinite(det)
    if valid_mask is not None:
        good = good & (valid_mask > 0)

    if not np.any(good):
        out = np.zeros_like(det)
        return (out, {"display": "empty"}) if return_info else out

    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    dx = x_idx - center_x
    dy = y_idx - center_y
    r = np.sqrt(dx * dx + dy * dy)

    r_good = r[good]
    det_good = det[good].astype(np.float64)
    w_px = max(float(width_px), 0.1)
    k = np.floor(r_good / w_px).astype(np.int32)
    max_k = int(np.max(k)) + 1 if k.size > 0 else 1

    cnt = np.bincount(k, minlength=max_k)
    rsum = np.bincount(k, weights=r_good, minlength=max_k)

    sort_idx = np.argsort(k, kind="stable")
    ds = det_good[sort_idx]
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    sel = np.flatnonzero(cnt >= min_count)

    if len(sel) < 3:
        med_val = float(np.median(det_good))
        mad_val = max(float(1.4826 * np.median(np.abs(det_good - med_val))), float(noise_floor))
        out = np.where(good, 0.5 + 0.5 * np.tanh((det - med_val) / (2.5 * mad_val)), 0.0).astype(np.float32)
        info = {"display": "one spread for whole image", "scale": mad_val}
        return (out, info) if return_info else out

    med = []
    mad = []
    for i in sel:
        seg = ds[start[i]: start[i] + cnt[i]]
        m_val = float(np.median(seg))
        med.append(m_val)
        mad.append(float(1.4826 * np.median(np.abs(seg - m_val))))

    med = np.array(med, dtype=np.float64)
    mad = np.array(mad, dtype=np.float64)
    rm = rsum[sel] / cnt[sel]

    if smooth_bins > 1:
        ker = np.ones(smooth_bins, dtype=np.float64)
        norm = np.convolve(np.ones(len(med), dtype=np.float64), ker, mode="same")
        med = np.convolve(med, ker, mode="same") / norm
        mad = np.convolve(mad, ker, mode="same") / norm

    level = np.interp(r, rm, med).astype(np.float32)
    spread = np.maximum(np.interp(r, rm, mad), float(noise_floor)).astype(np.float32)

    out = np.where(good, 0.5 + 0.5 * np.tanh((det - level) / (2.5 * spread)), 0.0).astype(np.float32)
    info = {
        "display": "level and spread per ring (NumPy)",
        "ring_width_px": width_px,
        "ring_min_count": min_count,
        "display_annuli": int(len(sel))
    }
    return (out, info) if return_info else out


def apply_nrgf(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    bin_px: float = 1.0,
    robust: bool = True,
    blend_ratio: float = 0.55,
    radius_min: float = 10.0,
    radius_max_factor: float = 3.5,
    use_detail_rings: bool = False,
    partial_annuli: str = "observed"
) -> np.ndarray:
    """
    Normalized Radial Gradient Filter (NRGF) con compuerta radial (Radial Gate),
    interpolación radial subpíxel continua y modulación de amplio rango dinámico para
    el máximo estiramiento y contraste de los filamentos coronales ("hilos").

    1. Modela perfiles de brillo radial mediano mu(r) y dispersión sigma(r).
    2. Suaviza e interpola linealmente de forma continua para erradicar cualquier
       escalonamiento o bandas concéntricas.
    3. Detalle normalizado (I - mu) / sigma filtrado bilateralmente (preserva filamentos
       coherentes y anula el ruido de píxel suelto).
    4. Compuerta radial suave que desvanece la modulación a 0.0 hacia radius_max_factor,
       manteniendo el cielo exterior negro aterciopelado y limpio al 100%.
    5. Opcional: normalización anular por anillos (use_detail_rings, AEL 1.3.4) para
       evitar saturación a blanco o negro en la corona interna.
    6. Opcional: encadenamiento de sectores angulares (partial_annuli='chain', AEL 1.4.1)
       para eliminar saltos en el limbo o recortes de sensor cuando la Luna está descentrada.
    """
    if partial_annuli not in ("observed", "chain"):
        raise ValueError("partial_annuli debe ser 'observed' o 'chain'")

    if partial_annuli == "observed" and HAS_RUST and hasattr(astroloom_rs, "apply_nrgf_continuous_rs") and hasattr(astroloom_rs, "compute_nrgf_profiles_and_detail_rs"):
        img_c = np.ascontiguousarray(image, dtype=np.float32)
        if img_c.ndim == 3:
            lum = 0.2126 * img_c[..., 0] + 0.7152 * img_c[..., 1] + 0.0722 * img_c[..., 2]
        else:
            lum = img_c
        lum_c = np.ascontiguousarray(lum, dtype=np.float32)
        raw_detail, sigma_prof, sky_noise = astroloom_rs.compute_nrgf_profiles_and_detail_rs(
            lum_c, float(center_x), float(center_y), float(radius_min), float(radius_max_factor), bool(robust)
        )
        clean_detail = cv2.bilateralFilter(raw_detail, d=7, sigmaColor=0.35, sigmaSpace=4.0)
        if use_detail_rings:
            rings_norm = display_detail_rings(
                clean_detail,
                valid_mask=(raw_detail != 0.0),
                center_x=float(center_x),
                center_y=float(center_y)
            )
            clean_detail = (rings_norm - 0.5) * 5.0

        return np.asarray(astroloom_rs.apply_nrgf_continuous_rs(
            img_c, clean_detail, sigma_prof, float(sky_noise),
            float(center_x), float(center_y), float(radius_min), float(radius_max_factor), float(blend_ratio)
        ), dtype=np.float32)

    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]

    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    dx = x_coords - center_x
    dy = y_coords - center_y
    r_map = np.sqrt(dx * dx + dy * dy)

    # 1. Compuerta radial continua en la periferia de la corona
    r_max = radius_min * max(1.2, float(radius_max_factor))
    gate_start = radius_min * (1.0 + (float(radius_max_factor) - 1.0) * 0.75)
    t = np.clip((r_map - gate_start) / max(r_max - gate_start, 1.0), 0.0, 1.0)
    radial_gate = 1.0 - (t * t * (3.0 - 2.0 * t))
    radial_gate = np.where(r_map > r_max, 0.0, radial_gate)

    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()
    lum = lum.astype(np.float32)

    # 2. Estimar fondo de cielo y ruido fuera del radio de corona
    sky_mask = (r_map >= r_max)
    if np.sum(sky_mask) > 100:
        sky_samples = lum[sky_mask]
        sky_bg = float(np.median(sky_samples))
        sky_noise = float(1.4826 * np.median(np.abs(sky_samples - sky_bg)))
    else:
        far_r = np.percentile(r_map, 90.0)
        sky_samples = lum[r_map >= far_r]
        sky_bg = float(np.median(sky_samples))
        sky_noise = float(1.4826 * np.median(np.abs(sky_samples - sky_bg)))

    sky_noise = max(sky_noise, 1e-4)

    bins = r_map.astype(np.int32)
    max_bin = int(r_map.max()) + 2
    counts = np.bincount(bins.ravel(), minlength=max_bin)

    flat_bins = bins.ravel()
    flat_lum = lum.ravel()

    mu_prof = np.zeros(max_bin, dtype=np.float32)
    sigma_prof = np.ones(max_bin, dtype=np.float32)

    if robust:
        sort_idx = np.argsort(flat_bins, kind="stable")
        sorted_bins = flat_bins[sort_idx]
        sorted_lum = flat_lum[sort_idx]
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

        valid_bins = np.nonzero(counts >= 30)[0]
        for b in valid_bins:
            if b < radius_min * 0.95:
                continue
            seg = sorted_lum[starts[b]: starts[b] + counts[b]]
            med = np.median(seg)
            mad = 1.4826 * np.median(np.abs(seg - med))
            mu_prof[b] = med
            sigma_prof[b] = mad

        # Encadenamiento angular anti-halos (partial_annuli='chain', AEL 1.4.1)
        if partial_annuli == "chain" and len(valid_bins) > 1:
            theta_map = np.arctan2(dy, dx)
            sector_map = (np.floor((theta_map + np.pi) / (2.0 * np.pi) * 360.0).astype(np.int32) % 360).ravel()
            sorted_sectors = sector_map[sort_idx]

            for i in range(1, len(valid_bins)):
                b_prev = valid_bins[i - 1]
                b_curr = valid_bins[i]
                if b_curr != b_prev + 1:
                    continue
                sec_prev = sorted_sectors[starts[b_prev]: starts[b_prev] + counts[b_prev]]
                sec_curr = sorted_sectors[starts[b_curr]: starts[b_curr] + counts[b_curr]]
                cov_prev = np.bincount(sec_prev, minlength=360) > 0
                cov_curr = np.bincount(sec_curr, minlength=360) > 0
                if not np.array_equal(cov_prev, cov_curr):
                    common = cov_prev & cov_curr
                    if np.any(common):
                        lum_prev = sorted_lum[starts[b_prev]: starts[b_prev] + counts[b_prev]]
                        lum_curr = sorted_lum[starts[b_curr]: starts[b_curr] + counts[b_curr]]
                        mask_p = common[sec_prev]
                        mask_c = common[sec_curr]
                        if np.sum(mask_p) >= 10 and np.sum(mask_c) >= 10:
                            step_mu = float(np.median(lum_curr[mask_c])) - float(np.median(lum_prev[mask_p]))
                            mu_prof[b_curr] = mu_prof[b_prev] + step_mu
                            mad_c = float(1.4826 * np.median(np.abs(lum_curr[mask_c] - np.median(lum_curr[mask_c]))))
                            mad_p = float(1.4826 * np.median(np.abs(lum_prev[mask_p] - np.median(lum_prev[mask_p]))))
                            step_sd = mad_c - mad_p
                            sigma_prof[b_curr] = max(sigma_prof[b_prev] + step_sd, 1e-4)
    else:
        sum_lum = np.bincount(flat_bins, weights=flat_lum, minlength=max_bin)
        sum_sq = np.bincount(flat_bins, weights=flat_lum.astype(np.float64) ** 2, minlength=max_bin)
        valid_mask = counts >= 30
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_vals = np.where(valid_mask, sum_lum / np.maximum(counts, 1), 0.0)
            var_vals = np.where(valid_mask, (sum_sq / np.maximum(counts, 1)) - (mean_vals ** 2), 0.0)
            mu_prof = mean_vals.astype(np.float32)
            sigma_prof = np.sqrt(np.maximum(var_vals, 1e-8)).astype(np.float32)

    # Suavizado de perfiles radiales
    if max_bin > 15:
        mu_prof = cv2.GaussianBlur(mu_prof[:, None], (1, 25), 6).ravel()
        sigma_prof = cv2.GaussianBlur(sigma_prof[:, None], (1, 25), 6).ravel()

    # Piso de ruido sobre sigma
    min_sigma = max(sky_noise * 2.0, 0.0025)
    sigma_prof = np.maximum(sigma_prof, min_sigma)

    # 3. Interpolación continua subpíxel a lo largo de r_map para eliminar por completo
    # el efecto de bandas concéntricas discretas
    bin_centers = np.arange(max_bin, dtype=np.float32)
    mu_map = np.interp(r_map.ravel(), bin_centers, mu_prof).reshape(h, w).astype(np.float32)
    sigma_map = np.interp(r_map.ravel(), bin_centers, sigma_prof).reshape(h, w).astype(np.float32)

    # Máscara fuera del disco lunar
    disk_mask = r_map >= radius_min

    # 4. Detalle normalizado con filtrado bilateral anti-ruido (preserva hilos continuos)
    raw_detail = np.where(disk_mask, (lum - mu_map) / sigma_map, 0.0).astype(np.float32)
    clean_detail = cv2.bilateralFilter(raw_detail, d=7, sigmaColor=0.35, sigmaSpace=4.0)
    if use_detail_rings:
        rings_norm = display_detail_rings(
            clean_detail,
            valid_mask=disk_mask,
            center_x=center_x,
            center_y=center_y
        )
        clipped_detail = (rings_norm - 0.5) * 2.0
    else:
        clipped_detail = np.clip(clean_detail, -2.5, 2.5) / 2.5

    # 5. Factor SNR adaptativo y modulación con máximo estiramiento de filamentos
    snr_factor = np.clip((sigma_map - 1.2 * sky_noise) / (sigma_map + 1e-6), 0.0, 1.0)
    blend = float(np.clip(blend_ratio, 0.0, 1.0))

    streamer_mod = clipped_detail * radial_gate * snr_factor * blend

    # Protección de margen superior (headroom) para evitar saturación en meseta blanca
    upper_headroom = np.where(streamer_mod > 0, np.maximum(1.0 - lum, 0.0), 1.0)
    final_mod = streamer_mod * (0.4 + 0.6 * np.sqrt(upper_headroom))

    new_lum = np.clip(lum + final_mod, 0.0, 1.0)
    new_lum = np.where(disk_mask, new_lum, 0.0).astype(np.float32)

    # 6. Recomposición cromática lineal radiométrica (preserva tonos y sat exactos sin distorsión HSV)
    if img.ndim == 3:
        ratio = np.divide(new_lum, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        ratio = np.clip(ratio, 0.0, 4.0)
        result = img * ratio[..., None]
    else:
        result = new_lum

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_mgn(
    image: np.ndarray,
    sigmas: Tuple[float, ...] = (1.5, 3.0, 6.0, 12.0, 24.0, 48.0),
    k: float = 0.70,
    gamma: float = 2.8,
    blend_ratio: float = 0.50,
    noise_floor: float = 0.003,
    chroma_denoise_sigma: float = 16.0,
    lunar_mask_params: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Multi-scale Gaussian Normalization (MGN - Morgan & Druckmüller 2014) con preservación
    de la corona exterior, denoise cromático local (AEL) y protección de protuberancias en el limbo.
    Normaliza el contraste de la corona solar simultáneamente en múltiples escalas espaciales,
    haciendo visibles filamentos radiales finos, bucles magnéticos cerrados y la corona exterior tenue,
    eliminando el grano de ruido coloreado del fondo sin deslavar detalles de emisión H-alfa.
    """
    if HAS_RUST and hasattr(astroloom_rs, "combine_mgn_scales_rs") and hasattr(astroloom_rs, "compute_mgn_diff_sq_rs"):
        img = np.asarray(image, dtype=np.float32)
        img_c = np.ascontiguousarray(img, dtype=np.float32)
        if img.ndim == 3:
            lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        else:
            lum = img.copy()
        lum_c = np.ascontiguousarray(lum, dtype=np.float32)
        h, w = lum_c.shape[:2]

        if lunar_mask_params is not None:
            mcx, mcy, mr = lunar_mask_params
            m_bool = astroloom_rs.create_lunar_mask_rs(h, w, float(mcx), float(mcy), float(mr - 2.5))
        else:
            m_bool = np.ones((h, w), dtype=bool)

        m_f32 = m_bool.astype(np.float32)
        masked_lum = np.where(m_bool, lum_c, 0.0).astype(np.float32)
        acc = np.zeros((h, w), dtype=np.float32)
        total_w = 0.0

        # Invarianza de escala (AEL 1.3.6 / PR #8): calcula data_scale y normaliza lum para gamma
        valid_lum = lum_c[m_bool] if np.any(m_bool) else lum_c
        if valid_lum.size > 0:
            min_v = float(np.min(valid_lum))
            max_v = float(np.max(valid_lum))
            data_scale = (max_v - min_v) if (max_v - min_v) > 1e-12 else 1.0
        else:
            min_v = 0.0
            data_scale = 1.0

        norm_lum = np.clip((lum_c - min_v) / data_scale, 0.0, 1.0).astype(np.float32)

        for s in sigmas:
            s_val = float(s)
            num = cv2.GaussianBlur(masked_lum, (0, 0), s_val)
            den = cv2.GaussianBlur(m_f32, (0, 0), s_val)
            d_sq = astroloom_rs.compute_mgn_diff_sq_rs(lum_c, m_bool, num, den)
            num_v = cv2.GaussianBlur(d_sq, (0, 0), s_val)
            astroloom_rs.combine_mgn_scales_rs(
                acc, lum_c, num, den, num_v, m_bool, float(k), float(noise_floor), float(data_scale)
            )
            total_w += 1.0

        mgn_lum = np.asarray(astroloom_rs.finalize_mgn_lum_rs(
            acc, float(total_w), norm_lum, float(gamma), float(blend_ratio),
            m_bool if lunar_mask_params is not None else None
        ), dtype=np.float32)

        if img.ndim == 3:
            scaled_rgb, chroma = astroloom_rs.compute_mgn_scaled_and_chroma_rs(
                img_c, mgn_lum, lum_c, float(noise_floor)
            )
            if chroma_denoise_sigma > 0.5:
                chroma_smooth = cv2.GaussianBlur(chroma, (0, 0), float(chroma_denoise_sigma))
                denoised_rgb = np.clip(chroma_smooth * mgn_lum[..., None], 0.0, 1.0)
                if lunar_mask_params is not None:
                    mcx, mcy, mr = lunar_mask_params
                    prom_mask = astroloom_rs.compute_mgn_prominence_mask_rs(
                        scaled_rgb, float(mcx), float(mcy), float(mr * 1.35)
                    )
                    prom_w = cv2.GaussianBlur(prom_mask, (0, 0), 2.0)
                    result = astroloom_rs.blend_mgn_prominences_and_mask_rs(
                        scaled_rgb, denoised_rgb, prom_w, m_bool
                    )
                else:
                    result = denoised_rgb
            else:
                result = scaled_rgb
                if lunar_mask_params is not None:
                    result = np.where(m_bool[..., None], result, 0.0)
        else:
            result = mgn_lum

        return np.asarray(result, dtype=np.float32)

    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]

    # Excluir interior del disco lunar protegiendo el margen cromosférico y protuberancias (mr - 2.5px)
    if lunar_mask_params is not None:
        mcx, mcy, mr = lunar_mask_params
        y_idx, x_idx = np.indices((h, w), dtype=np.float32)
        dist = np.hypot(x_idx - mcx, y_idx - mcy)
        m_bool = dist >= (mr - 2.5)
    else:
        m_bool = np.ones((h, w), dtype=bool)

    m = m_bool.astype(np.float32)
    acc = np.zeros((h, w), dtype=np.float32)
    total_w = 0.0

    # Invarianza de escala (AEL 1.3.6 / PR #8)
    valid_lum = lum[m_bool] if np.any(m_bool) else lum
    if valid_lum.size > 0:
        min_v = float(np.min(valid_lum))
        max_v = float(np.max(valid_lum))
        data_scale = (max_v - min_v) if (max_v - min_v) > 1e-12 else 1.0
    else:
        min_v = 0.0
        data_scale = 1.0

    eff_nf = float(noise_floor) * data_scale
    nf_sq = eff_nf * eff_nf
    min_sd = 1e-7 * data_scale

    for s in sigmas:
        s_val = float(s)
        # Convolución normalizada: (G * (I * m)) / (G * m)
        num = cv2.GaussianBlur(np.where(m_bool, lum, 0.0).astype(np.float32), (0, 0), s_val)
        den = cv2.GaussianBlur(m, (0, 0), s_val)
        mu = np.where(m_bool & (den >= 0.15), num / np.maximum(den, 1e-12), np.nan)

        d = np.where(np.isfinite(mu), lum - mu, 0.0).astype(np.float32)
        num_v = cv2.GaussianBlur(np.where(m_bool, d * d, 0.0).astype(np.float32), (0, 0), s_val)
        var = np.where(m_bool & (den >= 0.15), num_v / np.maximum(den, 1e-12), 0.0)
        # Piso de ruido relativo invariante a la escala
        sd = np.sqrt(np.maximum(var, 0.0) + nf_sq)

        c = np.where(np.isfinite(mu) & (sd > min_sd), (lum - mu) / sd, 0.0)
        acc += np.arctan(k * c).astype(np.float32)
        total_w += 1.0

    acc /= max(total_w, 1.0)

    # Escalar arctan[-pi/2, pi/2] a [0, 1]
    acc_norm = np.clip((acc + (np.pi / 2.0)) / np.pi, 0.0, 1.0)

    # Contexto global por compresión gamma normalizado
    norm_lum = np.clip((lum - min_v) / data_scale, 0.0, 1.0).astype(np.float32)
    g = norm_lum ** (1.0 / max(gamma, 0.1))

    # Mezcla armónica entre detalle multiescala normalizado y contexto global
    h_weight = float(np.clip(1.0 - blend_ratio, 0.0, 1.0))
    mgn_lum = np.clip(h_weight * g + (1.0 - h_weight) * acc_norm, 0.0, 1.0)

    if lunar_mask_params is not None:
        mgn_lum = np.where(m_bool, mgn_lum, 0.0)

    # Recomposición cromática lineal con margen adaptativo y denoise de croma local (AEL)
    if img.ndim == 3:
        ratio = np.divide(mgn_lum, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        snr = np.clip(lum / max(float(noise_floor) * 3.0, 1e-3), 0.0, 1.0)
        max_ratio = 4.0 + 20.0 * snr
        ratio = np.minimum(ratio, max_ratio)
        scaled_rgb = np.clip(img * ratio[..., None], 0.0, 1.0)

        if chroma_denoise_sigma > 0.5:
            # Extracción y filtrado espacial de croma (elimina el grano de color del fondo)
            chroma = np.divide(
                scaled_rgb,
                np.maximum(mgn_lum[..., None], 1e-4),
                out=np.zeros_like(scaled_rgb),
                where=mgn_lum[..., None] > 1e-5
            )
            chroma_smooth = cv2.GaussianBlur(chroma, (0, 0), float(chroma_denoise_sigma))
            denoised_rgb = np.clip(chroma_smooth * mgn_lum[..., None], 0.0, 1.0)

            # Protección estricta de protuberancias y limbo interior (H-alfa nativa sin desenfocar)
            if lunar_mask_params is not None:
                mcx, mcy, mr = lunar_mask_params
                prom_mask = (dist < mr * 1.35) & (scaled_rgb[..., 0] / np.maximum(scaled_rgb[..., 1], 1e-4) > 1.15)
                prom_w = cv2.GaussianBlur(prom_mask.astype(np.float32), (0, 0), 2.0)[..., None]
                result = prom_w * scaled_rgb + (1.0 - prom_w) * denoised_rgb
            else:
                result = denoised_rgb
        else:
            result = scaled_rgb

        result = np.clip(result, 0.0, 1.0)
        if lunar_mask_params is not None:
            result = np.where(m_bool[..., None], result, 0.0)
    else:
        result = mgn_lum

    return result.astype(np.float32)


def apply_coronal_highpass(
    image: np.ndarray,
    strength: float = 0.5,
    radius: int = 5,
    radius_min: float = 10.0,
    center_x: float = 0.0,
    center_y: float = 0.0,
    radius_max_factor: float = 2.6,
    use_detail_rings: bool = False
) -> np.ndarray:
    """
    Realza los filamentos finos mediante filtro de detalle bilateral enmascarado
    dentro de la zona coronal activa para evitar cualquier elevación de ruido en el fondo.
    Soporta normalización por anillos (AEL 1.3.4 / use_detail_rings) para evitar quemado.
    """
    if strength <= 0.01:
        return image.copy()

    img = np.clip(image, 0.0, 1.0).astype(np.float32)
    h, w = img.shape[:2]
    k = max(3, int(radius) | 1)
    smooth = cv2.bilateralFilter(img, d=k, sigmaColor=0.15, sigmaSpace=k * 1.5)
    detail = img - smooth

    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    r_map = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    r_max = radius_min * max(1.2, float(radius_max_factor))
    gate = (1.0 - _smoothstep(r_map, r_max * 0.8, r_max)) * _smoothstep(r_map, radius_min * 0.95, radius_min * 1.05)

    if img.ndim == 3:
        gate = gate[..., None]

    if use_detail_rings:
        disk_valid = (r_map >= radius_min * 0.95) & (r_map <= r_max)
        rings_norm = display_detail_rings(detail, valid_mask=disk_valid, center_x=center_x, center_y=center_y)
        norm_detail = (rings_norm - 0.5) * 2.0
        enhanced = img + (norm_detail * float(strength) * gate)
    else:
        enhanced = img + (detail * float(strength) * gate)

    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def compute_red_excess_mask(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_min: float,
    radius_max: Optional[float] = None,
    threshold_ratio: float = 1.12,
    dilation_px: int = 6
) -> np.ndarray:
    """
    Detecta protuberancias H-alfa en el limbo solar midiendo el exceso del cociente R/G
    anillo por anillo (AEL 1.4.2 / Issue de artefactos Larson-Sekanina).
    Aplica dilatación elíptica para cubrir el penacho de emisión y evitar réplicas oscuras.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=bool)

    h, w = img.shape[:2]
    r_min = max(float(radius_min), 1.0)
    r_max = float(radius_max) if radius_max is not None else (r_min * 1.4)
    r_max = max(r_max, r_min + 1.0)

    if HAS_RUST and hasattr(astroloom_rs, "compute_red_excess_mask_rs"):
        rgb_c = np.ascontiguousarray(img[..., :3], dtype=np.float32)
        raw_mask = np.asarray(astroloom_rs.compute_red_excess_mask_rs(
            rgb_c, float(center_x), float(center_y), r_min, r_max, float(threshold_ratio)
        ), dtype=bool)
    else:
        y_idx, x_idx = np.indices((h, w), dtype=np.float32)
        dist = np.hypot(x_idx - center_x, y_idx - center_y)
        in_zone = (dist >= r_min) & (dist <= r_max)

        r_ch = img[..., 0]
        g_ch = img[..., 1]
        ratio = np.divide(r_ch, np.maximum(g_ch, 1e-4), out=np.zeros_like(r_ch), where=in_zone)

        raw_mask = np.zeros((h, w), dtype=bool)
        if np.any(in_zone):
            width_px = 2.0
            k = np.floor((dist[in_zone] - r_min) / width_px).astype(np.int32)
            max_k = int(np.max(k)) + 1 if k.size > 0 else 1
            rat_good = ratio[in_zone]

            cnt = np.bincount(k, minlength=max_k)
            sort_idx = np.argsort(k, kind="stable")
            rs = rat_good[sort_idx]
            start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            sel = np.flatnonzero(cnt >= 10)

            bin_medians = np.ones(max_k, dtype=np.float32)
            for i in sel:
                seg = rs[start[i]: start[i] + cnt[i]]
                bin_medians[i] = float(np.median(seg))

            all_bins = np.floor((dist - r_min) / width_px).astype(np.int32)
            all_bins = np.clip(all_bins, 0, max_k - 1)
            med_map = bin_medians[all_bins]
            raw_mask = in_zone & (ratio > threshold_ratio * med_map)

    if dilation_px > 0 and np.any(raw_mask):
        ksize = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_DILATE, kernel)
        return dilated.astype(bool)

    return raw_mask


def apply_steerable_g2(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_min: float = 10.0,
    sigmas: Tuple[float, ...] = (1.5, 3.0, 6.0, 12.0),
    blend_ratio: float = 0.50,
    radius_max_factor: float = 3.5,
    use_detail_rings: bool = True
) -> np.ndarray:
    """
    Filtro de segunda derivada orientable analítica (Freeman & Adelson 1991 / AEL 1.4.1)
    guiado por el tensor de estructura local (Weickert 1999).
    Sigue arcos magnéticos y bucles cerrados cerca del limbo y streamers curvados
    con máxima resolución y menos del 50% del grano aleatorio respecto a ACHF.
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    r_map = np.hypot(x_coords - center_x, y_coords - center_y)

    r_max = radius_min * max(1.2, float(radius_max_factor))
    radial_gate = (1.0 - _smoothstep(r_map, r_max * 0.8, r_max)) * _smoothstep(r_map, radius_min * 0.95, radius_min * 1.05)
    valid_mask = (r_map >= radius_min * 0.95) & (r_map <= r_max)

    acc = np.zeros((h, w), dtype=np.float32)
    lum_c = np.ascontiguousarray(lum, dtype=np.float32)

    for s in sigmas:
        s_val = float(s)
        rho = s_val * 2.0

        blur = cv2.GaussianBlur(lum_c, (0, 0), s_val)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

        hxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
        hyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
        hxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)

        j11 = cv2.GaussianBlur(gx * gx, (0, 0), rho)
        j22 = cv2.GaussianBlur(gy * gy, (0, 0), rho)
        j12 = cv2.GaussianBlur(gx * gy, (0, 0), rho)

        scale_weight = s_val

        if HAS_RUST and hasattr(astroloom_rs, "accumulate_steerable_g2_rs"):
            astroloom_rs.accumulate_steerable_g2_rs(
                acc,
                np.ascontiguousarray(hxx, dtype=np.float32),
                np.ascontiguousarray(hyy, dtype=np.float32),
                np.ascontiguousarray(hxy, dtype=np.float32),
                np.ascontiguousarray(j11, dtype=np.float32),
                np.ascontiguousarray(j22, dtype=np.float32),
                np.ascontiguousarray(j12, dtype=np.float32),
                float(scale_weight)
            )
        else:
            big_x = j11 - j22
            big_y = 2.0 * j12
            r_sq = big_x * big_x + big_y * big_y
            r_geom = np.sqrt(np.maximum(r_sq, 1e-12))
            g2 = 0.5 * (hxx + hyy) + np.where(
                r_sq > 1e-12,
                0.5 * (big_x * (hxx - hyy) + 2.0 * big_y * hxy) / r_geom,
                0.0
            )
            acc += (-g2) * scale_weight

    total_w = sum(sigmas) if sigmas else 1.0
    acc /= max(total_w, 1e-6)

    if use_detail_rings:
        rings_norm = display_detail_rings(
            acc,
            valid_mask=valid_mask,
            center_x=center_x,
            center_y=center_y
        )
        clipped_detail = (rings_norm - 0.5) * 2.0
    else:
        s0 = float(np.std(acc[valid_mask])) if np.any(valid_mask) else 1.0
        clipped_detail = np.tanh(acc / max(s0 * 2.0, 1e-5))

    streamer_mod = clipped_detail * radial_gate * float(np.clip(blend_ratio, 0.0, 1.0))
    upper_headroom = np.where(streamer_mod > 0, np.maximum(1.0 - lum, 0.0), 1.0)
    final_mod = streamer_mod * (0.4 + 0.6 * np.sqrt(upper_headroom))

    new_lum = np.clip(lum + final_mod, 0.0, 1.0)
    new_lum = np.where(r_map >= (radius_min - 2.5), new_lum, 0.0).astype(np.float32)

    if img.ndim == 3:
        ratio = np.divide(new_lum, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        ratio = np.clip(ratio, 0.0, 4.0)
        result = img * ratio[..., None]
        result = np.where((r_map >= (radius_min - 2.5))[..., None], result, 0.0)
    else:
        result = new_lum

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_larson_sekanina(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    delta_theta_deg: float = 1.0,
    blend_ratio: float = 0.50,
    protect_prominences: bool = True,
    radius_min: float = 10.0,
    radius_max_factor: float = 3.5,
    use_detail_rings: bool = True
) -> np.ndarray:
    """
    Filtro de gradiente rotacional de Larson–Sekanina (1984) evaluado con interpolación
    cúbica de Keys exacta (a = -0.5) y máscara de protección de protuberancias H-alfa (AEL 1.4.2).
    Realza estructuras angulares tenues, chorros y filamentos sin copias fantasma oscuras.
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    r_map = np.hypot(x_coords - center_x, y_coords - center_y)

    prom_mask = None
    if protect_prominences and img.ndim == 3 and img.shape[2] >= 3:
        prom_mask = compute_red_excess_mask(
            img, center_x, center_y, radius_min=radius_min, radius_max=radius_min * 1.4
        )

    lum_c = np.ascontiguousarray(lum, dtype=np.float32)
    if HAS_RUST and hasattr(astroloom_rs, "compute_larson_sekanina_rs"):
        mask_c = np.ascontiguousarray(prom_mask) if prom_mask is not None else None
        ls = np.asarray(astroloom_rs.compute_larson_sekanina_rs(
            lum_c,
            float(center_x),
            float(center_y),
            float(delta_theta_deg),
            float(radius_min),
            mask_c,
            1e-4
        ), dtype=np.float32)
    else:
        rad = np.radians(float(delta_theta_deg))
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        dx = x_coords - center_x
        dy = y_coords - center_y

        xp = center_x + dx * cos_a - dy * sin_a
        yp = center_y + dx * sin_a + dy * cos_a
        xm = center_x + dx * cos_a + dy * sin_a
        ym = center_y - dx * sin_a + dy * cos_a

        def _keys_w_np(t):
            at = np.abs(t)
            w_arr = np.zeros_like(at)
            m1 = at <= 1.0
            m2 = (at > 1.0) & (at < 2.0)
            w_arr[m1] = 1.5 * at[m1]**3 - 2.5 * at[m1]**2 + 1.0
            w_arr[m2] = -0.5 * at[m2]**3 + 2.5 * at[m2]**2 - 4.0 * at[m2] + 2.0
            return w_arr

        def _reflect101_np(idx, length):
            i = idx.copy()
            i = np.where(i < 0, -i, i)
            i = np.where(i >= length, 2 * length - 2 - i, i)
            return np.clip(i, 0, length - 1)

        def _sample_keys_np(src, xq, yq):
            ix = np.floor(xq).astype(np.int32)
            iy = np.floor(yq).astype(np.int32)
            fx = xq - ix
            fy = yq - iy
            wx = [_keys_w_np(1.0 + fx), _keys_w_np(fx), _keys_w_np(1.0 - fx), _keys_w_np(2.0 - fx)]
            wy = [_keys_w_np(1.0 + fy), _keys_w_np(fy), _keys_w_np(1.0 - fy), _keys_w_np(2.0 - fy)]
            sampled = np.zeros_like(xq, dtype=np.float32)
            for m in range(4):
                py = _reflect101_np(iy - 1 + m, h)
                row_sum = np.zeros_like(xq, dtype=np.float32)
                for n in range(4):
                    px = _reflect101_np(ix - 1 + n, w)
                    row_sum += src[py, px] * wx[n]
                sampled += row_sum * wy[m]
            return sampled

        val_p = np.maximum(_sample_keys_np(lum, xp, yp), 1e-4)
        val_m = np.maximum(_sample_keys_np(lum, xm, ym), 1e-4)
        val_c = np.maximum(lum, 1e-4)

        ls = 2.0 * np.log(val_c) - np.log(val_p) - np.log(val_m)
        r_min_eff = max(float(radius_min) - 2.5, 0.0)
        excl = (r_map < r_min_eff)
        if prom_mask is not None:
            excl = excl | prom_mask
        ls = np.where(excl, 0.0, ls).astype(np.float32)

    r_max = radius_min * max(1.2, float(radius_max_factor))
    radial_gate = (1.0 - _smoothstep(r_map, r_max * 0.8, r_max)) * _smoothstep(r_map, radius_min * 0.95, radius_min * 1.05)
    valid_mask = (r_map >= radius_min * 0.95) & (r_map <= r_max)
    if prom_mask is not None:
        valid_mask = valid_mask & (~prom_mask)

    if use_detail_rings:
        rings_norm = display_detail_rings(
            ls,
            valid_mask=valid_mask,
            center_x=center_x,
            center_y=center_y
        )
        clipped_detail = (rings_norm - 0.5) * 2.0
    else:
        s0 = float(np.std(ls[valid_mask])) if np.any(valid_mask) else 1.0
        clipped_detail = np.tanh(ls / max(s0 * 2.0, 1e-5))

    streamer_mod = clipped_detail * radial_gate * float(np.clip(blend_ratio, 0.0, 1.0))
    upper_headroom = np.where(streamer_mod > 0, np.maximum(1.0 - lum, 0.0), 1.0)
    final_mod = streamer_mod * (0.4 + 0.6 * np.sqrt(upper_headroom))

    new_lum = np.clip(lum + final_mod, 0.0, 1.0)
    new_lum = np.where(r_map >= (radius_min - 2.5), new_lum, 0.0).astype(np.float32)

    if img.ndim == 3:
        ratio = np.divide(new_lum, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        ratio = np.clip(ratio, 0.0, 4.0)
        result = img * ratio[..., None]
        result = np.where((r_map >= (radius_min - 2.5))[..., None], result, 0.0)
    else:
        result = new_lum

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_swap_filter(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_min: float = 10.0,
    window_deg: float = 15.0,
    c0: float = 0.75,
    blend_ratio: float = 1.0
) -> np.ndarray:
    """
    Filtro de compresión angular y radial SWAP (Seaton et al. 2023, Solar Physics / PROBA2).
    Genera una representación global de alto contraste de la corona solar idéntica a la de
    observatorios espaciales solares, comprimiendo simultáneamente bucles internos y streamers externos.
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]
    max_r = int(np.hypot(max(center_x, w - center_x), max(center_y, h - center_y))) + 2
    n_theta = 1440

    # 1. Mapeo a coordenadas polares (r, theta) con buffer pre-inicializado (evita basura de memoria/NaN en OpenCV 5.0)
    dst_polar = np.zeros((max_r, n_theta), dtype=np.float32)
    polar = cv2.warpPolar(lum, (n_theta, max_r), (center_x, center_y), max_r, cv2.WARP_POLAR_LINEAR, dst=dst_polar)
    polar = np.nan_to_num(polar, copy=False)

    # 2. Mediana móvil a lo largo de theta (ventana angular +- window_deg)
    win = int(round(2.0 * window_deg / 360.0 * n_theta))
    if win % 2 == 0:
        win += 1
    win = max(win, 3)

    if HAS_RUST and hasattr(astroloom_rs, "polar_circular_median_filter_rs"):
        polar_c = np.ascontiguousarray(polar, dtype=np.float32)
        f_polar = np.asarray(astroloom_rs.polar_circular_median_filter_rs(polar_c, win), dtype=np.float32)
    else:
        from scipy.ndimage import median_filter
        f_polar = median_filter(polar, size=(1, win), mode="wrap")

    # 3. Remapeo inverso a coordenadas cartesianas y suavizado cartesiano (sigma = 4.0 px)
    dst_cart = np.zeros((h, w), dtype=np.float32)
    cart_f = cv2.warpPolar(f_polar, (w, h), (center_x, center_y), max_r, cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP, dst=dst_cart)
    cart_f = np.nan_to_num(cart_f, copy=False)
    f_bg = cv2.GaussianBlur(cart_f, (0, 0), 4.0)

    # 4. Potencia reducida I / (F + t0)^c0
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    r_map = np.hypot(x_coords - center_x, y_coords - center_y)
    mask_solar = (r_map >= max(radius_min - 2.5, 0.0))

    valid_f = f_bg[mask_solar & (f_bg > 0)]
    t0 = float(np.median(valid_f)) if valid_f.size > 0 else 1e-4

    denom = np.power(np.maximum(f_bg + t0, 1e-6), float(c0))
    i_swap = np.where(mask_solar, lum / denom, 0.0).astype(np.float32)

    # 5. Normalización por raíz cuarta y mapeo al rango visual
    valid_swap = i_swap[mask_solar & (i_swap > 0)]
    med_swap = float(np.median(valid_swap)) if valid_swap.size > 0 else 1.0
    i_norm = np.power(np.maximum(i_swap / max(med_swap, 1e-6), 0.0), 0.25)

    scaled = np.clip((i_norm - 0.5) / (2.0 - 0.5), 0.0, 1.0)
    scaled = np.where(mask_solar, scaled, 0.0).astype(np.float32)

    # Recomposición cromática o autónoma
    if img.ndim == 3:
        ratio = np.divide(scaled, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        ratio = np.clip(ratio, 0.0, 4.0)
        swap_rgb = np.clip(img * ratio[..., None], 0.0, 1.0)
        swap_rgb = np.where(mask_solar[..., None], swap_rgb, 0.0)
        blend = float(np.clip(blend_ratio, 0.0, 1.0))
        final = blend * swap_rgb + (1.0 - blend) * img
    else:
        blend = float(np.clip(blend_ratio, 0.0, 1.0))
        final = blend * scaled + (1.0 - blend) * lum

    return np.clip(final, 0.0, 1.0).astype(np.float32)


# ============================================================================
# FILTROS Y EXPORTACIÓN AVANZADA DE FASE 3 (AEL v1.3.4 - v1.4.2)
# ============================================================================

_B3K = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


def _paired_b3_step(
    c: np.ndarray,
    m: np.ndarray,
    step: int,
    bilateral: bool = False,
    guide: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Paso à-trous B3 con soporte bilateral de pares simétricos (AEL 1.3.4 / 1.3.5 / Issue #4).
    Acelerado por Rust (Rayon) con fallback numérico exacto en NumPy.
    Garantiza que ningún tap opuesto falte para anular sesgo en bordes de datos y el limbo lunar.
    """
    if HAS_RUST and hasattr(astroloom_rs, "paired_b3_step_rs"):
        c_c = np.ascontiguousarray(c, dtype=np.float32)
        m_c = np.ascontiguousarray(m, dtype=bool)
        g_c = np.ascontiguousarray(guide, dtype=np.float32) if guide is not None else None
        o, compl, p = astroloom_rs.paired_b3_step_rs(c_c, m_c, int(step), bool(bilateral), g_c)
        return np.asarray(o, dtype=np.float32), np.asarray(compl, dtype=np.float32), np.asarray(p, dtype=bool)

    # Fallback NumPy
    h, w = c.shape
    out = np.zeros_like(c)
    compl = np.zeros_like(c)
    pair = np.zeros_like(m)
    ctg_arr = c if guide is None else guide

    for y in range(h):
        for x in range(w):
            if not m[y, x]:
                continue
            ctr = float(c[y, x])
            ctg = float(ctg_arr[y, x])
            mo, mo2, ma = 0.0, 0.0, 0.0
            taps_g = []
            for dy in range(-2, 3):
                y1 = y + dy * step
                y2 = y - dy * step
                for dx in range(-2, 3):
                    x1 = x + dx * step
                    x2 = x - dx * step
                    k = float(_B3K[dy + 2] * _B3K[dx + 2])
                    if dy == 0 and dx == 0:
                        g = 1.0
                    else:
                        ok1 = (0 <= y1 < h) and (0 <= x1 < w) and m[y1, x1]
                        ok2 = (0 <= y2 < h) and (0 <= x2 < w) and m[y2, x2]
                        g = 1.0 if (ok1 and ok2) else 0.0
                    taps_g.append((dy, dx, k, g, y1, x1))
                    if bilateral and g > 0:
                        vg = float(ctg_arr[y1, x1])
                        dl = vg - ctg
                        mo += k * g * dl
                        mo2 += k * g * dl * dl
                        ma += k * g
            if bilateral:
                ma_safe = max(ma, 1e-20)
                mean = mo / ma_safe
                vv = max(mo2 / ma_safe - mean * mean, 1e-20)

            num, den, ks, kn = 0.0, 0.0, 0.0, 0.0
            for dy, dx, k, g, y1, x1 in taps_g:
                if g <= 0:
                    continue
                v = float(c[y1, x1])
                vg = float(ctg_arr[y1, x1])
                if dy != 0 or dx != 0:
                    kn += k * g
                if bilateral:
                    diff = ctg - vg
                    wt = k * np.exp(-0.5 * diff * diff / vv)
                else:
                    wt = k
                num += wt * (v - ctr)
                den += wt
                ks += k * g
            out[y, x] = ctr + num / max(den, 1e-20)
            compl[y, x] = ks
            pair[y, x] = kn > 0

    return out, compl, pair


def _b3_normalized(a: np.ndarray, m: np.ndarray, step: int) -> np.ndarray:
    """
    Convolución separable B3 à-trous 2D normalizada por soporte observado:
    conv(a * m, step) / max(conv(m, step), 1e-20).
    Acelerada por Rust con fallback separable en NumPy.
    """
    if HAS_RUST and hasattr(astroloom_rs, "b3_normalized_rs"):
        a_c = np.ascontiguousarray(a, dtype=np.float32)
        m_c = np.ascontiguousarray(m, dtype=bool)
        return np.asarray(astroloom_rs.b3_normalized_rs(a_c, m_c, int(step)), dtype=np.float32)

    h, w = a.shape
    step_i = int(step)

    def conv_b3(src):
        tmp = np.zeros_like(src)
        for dy, ky in zip(range(-2, 3), _B3K):
            yy = np.clip(np.arange(h) + dy * step_i, 0, h - 1)
            valid_y = (np.arange(h) + dy * step_i >= 0) & (np.arange(h) + dy * step_i < h)
            tmp[valid_y] += ky * src[yy[valid_y]]
        out = np.zeros_like(src)
        for dx, kx in zip(range(-2, 3), _B3K):
            xx = np.clip(np.arange(w) + dx * step_i, 0, w - 1)
            valid_x = (np.arange(w) + dx * step_i >= 0) & (np.arange(w) + dx * step_i < w)
            out[:, valid_x] += kx * tmp[:, xx[valid_x]]
        return out

    ca = conv_b3(np.where(m, a, 0.0).astype(np.float32))
    cm = conv_b3(m.astype(np.float32))
    return ca / np.maximum(cm, 1e-20)


def apply_wow_edge_aware(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_min: float = 10.0,
    n_scales: int = 7,
    guided: bool = True,
    completeness: Tuple[float, float] = (0.6, 0.995),
    blend_ratio: float = 0.50,
    use_detail_rings: bool = True
) -> np.ndarray:
    """
    Filtro WOW bilateral guiado (Wavelet-Optimized Whitening Edge-Aware / AEL 1.3.5 / Issue #4).
    Descompone la corona en escalas à-trous B3 con pares simétricos y pesos de rango guiados
    para mantener el limbo lunar limpio sin generar halos ni invertir estructuras sutiles.
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    dist = np.hypot(x_coords - center_x, y_coords - center_y)
    valid_mask = dist >= max(radius_min - 2.5, 0.0)

    c = np.where(valid_mask, lum, 0.0).astype(np.float32)
    out = np.zeros_like(c)
    lo, hi = completeness

    for s in range(n_scales):
        step = 2 ** s
        guide = _paired_b3_step(c, valid_mask, step)[0] if guided else None
        nxt, compl, pair = _paired_b3_step(c, valid_mask, step, bilateral=True, guide=guide)
        nxt = np.where(valid_mask, nxt, 0.0).astype(np.float32)

        wave = np.where(valid_mask, c - nxt, 0.0).astype(np.float32)
        eps_thresh = 8.0 * np.finfo(np.float32).eps * np.maximum(np.abs(c), np.abs(nxt))
        wave[np.abs(wave) <= eps_thresh] = 0.0

        local_power = _b3_normalized(wave * wave, valid_mask, step)
        denom = np.sqrt(np.maximum(local_power, 1e-20))
        g = np.where(valid_mask & (compl > 0), wave / denom, 0.0).astype(np.float32)

        if s >= 3:
            q = np.clip((compl - lo) / (hi - lo), 0.0, 1.0)
            weight = (q * q * (3.0 - 2.0 * q) * valid_mask).astype(np.float32)
        else:
            weight = (valid_mask & (compl > 0)).astype(np.float32)

        whole = valid_mask & (compl >= 0.999)
        if whole.any():
            g -= float(np.mean(g[whole], dtype=np.float64))

        sigma_bg = float(min(150.0, max(32.0, 4.0 * step)))
        bg_num = cv2.GaussianBlur(g * weight * pair, (0, 0), sigma_bg)
        bg_den = cv2.GaussianBlur((weight * pair).astype(np.float32), (0, 0), sigma_bg)
        local_bg = bg_num / np.maximum(bg_den, 1e-6)

        g -= pair * local_bg
        out += weight * g
        c = nxt

    raw_detail = np.where(valid_mask, out, 0.0).astype(np.float32)

    # Normalización del detalle
    if use_detail_rings:
        rings_norm = display_detail_rings(
            raw_detail,
            valid_mask=valid_mask,
            center_x=center_x,
            center_y=center_y
        )
        norm_detail = rings_norm
    else:
        mad = float(1.4826 * np.median(np.abs(raw_detail[valid_mask] - np.median(raw_detail[valid_mask]))))
        norm_detail = 0.5 + 0.5 * np.tanh(raw_detail / max(2.5 * mad, 1e-5))

    blend = float(np.clip(blend_ratio, 0.0, 1.0))
    if img.ndim == 3:
        detail_3ch = np.repeat(norm_detail[..., None], 3, axis=2)
        overlay_res = np.where(
            img < 0.5,
            2.0 * img * detail_3ch,
            1.0 - 2.0 * (1.0 - img) * (1.0 - detail_3ch)
        )
        final = (1.0 - blend) * img + blend * overlay_res
        final = np.where(valid_mask[..., None], final, 0.0)
    else:
        overlay_res = np.where(
            lum < 0.5,
            2.0 * lum * norm_detail,
            1.0 - 2.0 * (1.0 - lum) * (1.0 - norm_detail)
        )
        final = (1.0 - blend) * lum + blend * overlay_res
        final = np.where(valid_mask, final, 0.0)

    return np.clip(final, 0.0, 1.0).astype(np.float32)


def nafe_neighbourhood(n: int = 129) -> np.ndarray:
    """
    Vecindario difuso de NAFE (Druckmüller 2013, ApJS 207, 25, eq. 12-14).
    Compuesto por 12 componentes gaussianas sigma_m = 2^(m/2) px (1.4 a 64 px).
    """
    if n % 2 != 1:
        raise ValueError("n debe ser impar")
    k = np.arange(n) - n // 2
    r2 = (k[:, None] ** 2 + k[None, :] ** 2).astype(np.float64)
    sig = 2.0 ** (np.arange(1, 13) / 2.0)
    g = sum(np.exp(-0.5 * r2 / (s ** 2)) / (2.0 * np.pi * np.sqrt(s)) for s in sig)
    return (g * (2.0 * np.pi / np.sum(sig ** -0.5))).astype(np.float32)


def apply_nafe(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_min: float = 10.0,
    sigma: Optional[float] = None,
    n: int = 129,
    level_step: float = 0.01,
    blend_ratio: float = 0.79,
    use_overlay: bool = True
) -> np.ndarray:
    """
    Noise Adaptive Fuzzy Equalization (NAFE - Miloslav Druckmüller 2013, ApJS 207, 25).
    Ecualización difusa con compuerta de ruido gaussiana a lo largo del eje de intensidad,
    calculada de forma acelerada mediante niveles de intensidad y convoluciones FFT 2D.
    """
    from scipy import fft as sfft
    from scipy.special import ndtr

    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    dist = np.hypot(x_coords - center_x, y_coords - center_y)
    m = dist >= max(radius_min - 2.5, 0.0)

    if not m.any():
        return img.copy()

    # Si sigma no se proporciona, estimar a partir del piso de ruido en la corona lejana
    if sigma is None or sigma <= 0:
        far_r = np.percentile(dist[m], 85.0)
        sky_samples = lum[dist >= far_r]
        if sky_samples.size > 50:
            med_sky = float(np.median(sky_samples))
            mad_sky = float(1.4826 * np.median(np.abs(sky_samples - med_sky)))
            sigma = max(12.0 * mad_sky, 0.02)
        else:
            sigma = 0.035

    half = n // 2
    a0 = np.where(m, lum, 0.0).astype(np.float32)
    mf = m.astype(np.float32)
    ph = sfft.next_fast_len(h + n - 1, real=True)
    pw = sfft.next_fast_len(w + n - 1, real=True)
    kf = sfft.rfft2(nafe_neighbourhood(n), s=(ph, pw), workers=-1)

    def conv(x):
        y = sfft.irfft2(sfft.rfft2(x, s=(ph, pw), workers=-1) * kf, s=(ph, pw), workers=-1)
        return y[half:half + h, half:half + w]

    norm = np.maximum(conv(mf), 1e-12)
    v = a0[m].astype(np.float64)
    lo, hi = float(v.min()), float(v.max())
    t = [lo - sigma]
    while t[-1] < hi or len(t) < 2:
        t.append(t[-1] + max(sigma / 2.0, level_step * (t[-1] - lo)))
    t = np.asarray(t)

    idx = np.flatnonzero(m.ravel())
    k = np.clip(np.searchsorted(t, v, side="right") - 1, 0, len(t) - 2)
    order = np.argsort(k, kind="stable")
    idx, v_ord, k_ord = idx[order], v[order], k[order]
    cuts = np.searchsorted(k_ord, np.arange(len(t)))
    out = np.full(h * w, np.nan, dtype=np.float32)

    def level(tj):
        return (conv(mf * ndtr((tj - a0) / sigma).astype(np.float32)) / norm).ravel()

    prev = None
    for j in range(1, len(t)):
        lo_j, hi_j = cuts[j - 1], cuts[j]
        nxt = cuts[j + 1] if j + 1 < len(t) else hi_j
        if hi_j == lo_j and nxt == hi_j:
            prev = None
            continue
        if prev is None:
            prev = level(t[j - 1])
        cur = level(t[j])
        if hi_j > lo_j:
            p = idx[lo_j:hi_j]
            f = (v_ord[lo_j:hi_j] - t[j - 1]) / (t[j] - t[j - 1])
            out[p] = prev[p] + f * (cur[p] - prev[p])
        prev = cur

    equalized = np.clip(np.nan_to_num(out.reshape(h, w), nan=0.5), 0.0, 1.0).astype(np.float32)

    blend = float(np.clip(blend_ratio, 0.0, 1.0))
    if use_overlay:
        if img.ndim == 3:
            eq_3ch = np.repeat(equalized[..., None], 3, axis=2)
            overlay_res = np.where(
                img < 0.5,
                2.0 * img * eq_3ch,
                1.0 - 2.0 * (1.0 - img) * (1.0 - eq_3ch)
            )
            final = (1.0 - blend) * img + blend * overlay_res
            final = np.where(m[..., None], final, 0.0)
        else:
            overlay_res = np.where(
                lum < 0.5,
                2.0 * lum * equalized,
                1.0 - 2.0 * (1.0 - lum) * (1.0 - equalized)
            )
            final = (1.0 - blend) * lum + blend * overlay_res
            final = np.where(m, final, 0.0)
    else:
        if img.ndim == 3:
            final = (1.0 - blend) * img + blend * np.repeat(equalized[..., None], 3, axis=2)
            final = np.where(m[..., None], final, 0.0)
        else:
            final = (1.0 - blend) * lum + blend * equalized
            final = np.where(m, final, 0.0)

    return np.clip(final, 0.0, 1.0).astype(np.float32)


def export_coronal_layers_psd(
    filepath: str,
    base_image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_min: float,
    layers_data: Optional[Dict[str, np.ndarray]] = None,
    depth: int = 16,
    log_fn: Optional[Any] = None
) -> Tuple[bool, str, int]:
    """
    Exporta una composición coronal completa en archivo PSD nativo de Photoshop (16-bit)
    con capas y modos de fusión estandarizados de la literatura astronómica:
    - 00 - Revelado Final (Referencia AstroLoom) [Normal, oculta por defecto]
    - 01 - NRGF [Overlay, 55%]
    - 02 - WOW Bilateral Guiado [Overlay, 50%]
    - 03 - MGN Multiescala (Morgan & Druckmüller) [Overlay, 50%]
    - 09 - NAFE Difuso [Overlay, 79%, oculta]
    - 11 - Steerable G2 [Overlay, 50%]
    - 15 - Larson-Sekanina [Overlay, 20%, oculta]
    - 16 - Filtro SWAP PROBA2 [Normal, 100%, oculta]
    - Fondo HDR Base [Normal, 100%, visible]
    """
    def _log(msg: str):
        if log_fn:
            log_fn(msg)

    if base_image is None or base_image.size == 0:
        return False, "La imagen base está vacía.", 0

    base_rgb = np.clip(base_image, 0.0, 1.0).astype(np.float32)
    if base_rgb.ndim == 2:
        base_rgb = np.repeat(base_rgb[..., None], 3, axis=2)

    h, w = base_rgb.shape[:2]
    _log(f"[PSD CORONAL] Generando capas estandarizadas ({w}x{h} px, {depth}-bit)...")

    data = layers_data.copy() if layers_data is not None else {}

    # 1. Obtener o calcular capas estándar
    if "nrgf" not in data:
        _log("[PSD CORONAL] Procesando capa 01 - NRGF...")
        data["nrgf"] = apply_nrgf(base_rgb, center_x, center_y, radius_min=radius_min, blend_ratio=0.55)

    if "wow" not in data:
        _log("[PSD CORONAL] Procesando capa 02 - WOW Bilateral Guiado...")
        data["wow"] = apply_wow_edge_aware(base_rgb, center_x, center_y, radius_min=radius_min, blend_ratio=0.50)

    if "mgn" not in data:
        _log("[PSD CORONAL] Procesando capa 03 - MGN Multiescala...")
        lunar_mask = (center_x, center_y, radius_min) if radius_min > 5 else None
        data["mgn"] = apply_mgn(
            base_rgb,
            sigmas=(1.5, 3.0, 6.0, 12.0, 24.0, 48.0),
            k=0.70,
            gamma=2.8,
            blend_ratio=0.50,
            chroma_denoise_sigma=16.0,
            lunar_mask_params=lunar_mask
        )

    if "nafe" not in data:
        _log("[PSD CORONAL] Procesando capa 09 - NAFE Difuso...")
        data["nafe"] = apply_nafe(base_rgb, center_x, center_y, radius_min=radius_min, blend_ratio=0.79)

    if "steerable_g2" not in data:
        _log("[PSD CORONAL] Procesando capa 11 - Steerable G2...")
        data["steerable_g2"] = apply_steerable_g2(base_rgb, center_x, center_y, radius_min=radius_min, blend_ratio=0.50)

    if "larson_sekanina" not in data:
        _log("[PSD CORONAL] Procesando capa 15 - Larson-Sekanina...")
        data["larson_sekanina"] = apply_larson_sekanina(base_rgb, center_x, center_y, radius_min=radius_min, blend_ratio=0.20)

    if "swap" not in data:
        _log("[PSD CORONAL] Procesando capa 16 - Filtro SWAP PROBA2...")
        data["swap"] = apply_swap_filter(base_rgb, center_x, center_y, radius_min=radius_min, blend_ratio=1.0)

    def _prep_rgb(arr: np.ndarray) -> np.ndarray:
        clipped = np.clip(arr, 0.0, 1.0).astype(np.float32)
        if clipped.ndim == 2:
            clipped = np.repeat(clipped[..., None], 3, axis=2)
        if clipped.shape[:2] != (h, w):
            clipped = cv2.resize(clipped, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(clipped, dtype=np.float32)

    # Construcción de la pila de capas PSD (de abajo a arriba en Photoshop)
    layer_records = []

    # Fondo HDR Base
    layer_records.append({
        "name": "Fondo HDR Base",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "norm",
        "opacity": 255,
        "visible": True,
        "rgb": _prep_rgb(base_rgb)
    })

    # 16. Filtro SWAP
    layer_records.append({
        "name": "16 - Filtro SWAP PROBA2",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "norm",
        "opacity": 255,
        "visible": False,
        "rgb": _prep_rgb(data["swap"])
    })

    # 15. Larson-Sekanina
    layer_records.append({
        "name": "15 - Larson-Sekanina",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "over",
        "opacity": 51,  # ~20%
        "visible": False,
        "rgb": _prep_rgb(data["larson_sekanina"])
    })

    # 11. Steerable G2
    layer_records.append({
        "name": "11 - Steerable G2 (Freeman & Adelson)",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "over",
        "opacity": 128,  # 50%
        "visible": True,
        "rgb": _prep_rgb(data["steerable_g2"])
    })

    # 09. NAFE Difuso
    layer_records.append({
        "name": "09 - NAFE Difuso (Druckmüller 2013)",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "over",
        "opacity": 201,  # ~79%
        "visible": False,
        "rgb": _prep_rgb(data["nafe"])
    })

    # 03. MGN Multiescala
    layer_records.append({
        "name": "03 - MGN Multiescala (Morgan & Druckmüller)",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "over",
        "opacity": 128,  # 50%
        "visible": True,
        "rgb": _prep_rgb(data["mgn"])
    })

    # 02. WOW Bilateral Guiado
    layer_records.append({
        "name": "02 - WOW Bilateral Guiado",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "over",
        "opacity": 128,  # 50%
        "visible": True,
        "rgb": _prep_rgb(data["wow"])
    })

    # 01. NRGF
    layer_records.append({
        "name": "01 - NRGF",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "over",
        "opacity": 140,  # ~55%
        "visible": True,
        "rgb": _prep_rgb(data["nrgf"])
    })

    # 00. Capa Cúspide de Referencia (Revelado Final)
    comp_img = data.get("composite", data["steerable_g2"])
    layer_records.append({
        "name": "00 - Revelado Final (Referencia AstroLoom)",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "norm",
        "opacity": 255,
        "visible": False,
        "rgb": _prep_rgb(comp_img)
    })

    _log(f"[PSD CORONAL] Guardando {len(layer_records)} capas en disco...")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if HAS_RUST and hasattr(astroloom_rs, "write_psd_file"):
        comp_contiguous = _prep_rgb(comp_img)
        astroloom_rs.write_psd_file(
            filepath,
            w,
            h,
            depth,
            comp_contiguous,
            layer_records,
            None
        )
    else:
        # Fallback a core.psd_export si está disponible
        from core.psd_export import export_developer_psd
        export_developer_psd(filepath, comp_img, base_rgb)

    file_size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
    desc = f"PSD Multicapa ({len(layer_records)} capas coronales, {depth}-bit, {file_size_mb:.1f} MB)"
    _log(f"[PSD CORONAL FINALIZADO] {os.path.basename(filepath)} ({desc})")

    return True, desc, len(layer_records)
