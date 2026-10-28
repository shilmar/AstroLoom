# core/eclipse.py
"""
core/eclipse.py - Motor matemático y algoritmos especializados para eclipses solares y lunares.

Incluye:
1. Extracción de metadatos EXIF (tiempo de exposición, ISO, apertura).
2. Detección de limbo de alta precisión (Sub-pixel Radial Edge Scanner + Ajuste Algebraico
   con rechazo RANSAC/MAD de protuberancias y perlas de Baily).
3. Traslación afín subpíxel y registro de series de bracketing por limbo.
4. Fusión HDR lineal con estirado de rango dinámico (Asinh) para levantar la corona externa.
5. Filtro NRGF con compuerta radial (Radial Gate) y sustracción de ruido de fondo para
   eliminar al 100% el grano del sensor en el cielo exterior.
6. Realce de filamentos finos mediante filtro tangencial / bilateral enmascarado.
"""
import os
import cv2
import numpy as np
import datetime
from typing import Tuple, List, Dict, Optional, Any
from PIL import Image, ExifTags
from scipy.optimize import minimize

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


def fuse_hdr_bracketing(
    images: List[np.ndarray],
    exposures_s: List[float],
    saturation_threshold: float = 0.90,
    noise_floor: float = 0.015,
    asinh_stretch: float = 0.0,
    mode: str = "solar",
    lunar_mask_params: Optional[Tuple[float, float, float]] = None,
    subtract_sky: bool = True
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

    sat_lo = max(0.5, saturation_threshold - 0.10)
    sat_hi = saturation_threshold

    # Máscara lunar para evitar que flare óptico de tomas largas invada el disco
    moon_weight = None
    if lunar_mask_params is not None:
        mcx, mcy, mr = lunar_mask_params
        y_idx, x_idx = np.indices(shape[:2], dtype=np.float32)
        dist = np.hypot(x_idx - mcx, y_idx - mcy)
        moon_weight = _smoothstep(dist, mr - 1.5, mr + 1.0)

    for img, t in zip(sorted_images, sorted_exp):
        t = max(t, 1e-6)

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

    if asinh_stretch > 1.0:
        hdr_norm = apply_asinh_stretch(hdr_norm, stretch_factor=asinh_stretch)

    return hdr_norm


def apply_nrgf(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    bin_px: float = 1.0,
    robust: bool = True,
    blend_ratio: float = 0.55,
    radius_min: float = 10.0,
    radius_max_factor: float = 3.5
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
    """
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
    k: float = 0.7,
    gamma: float = 2.8,
    blend_ratio: float = 0.50,
    lunar_mask_params: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Multi-scale Gaussian Normalization (MGN - Morgan & Druckmüller 2014).
    Normaliza el contraste de la corona solar simultáneamente en múltiples escalas espaciales,
    haciendo visibles filamentos radiales finos, bucles magnéticos cerrados y la corona exterior tenue,
    sin artefactos de coordenadas polares (gramófono) ni compuertas que cercenen el fondo de cielo.
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]

    # Excluir interior del disco lunar para evitar halos y dispersión espuria
    if lunar_mask_params is not None:
        mcx, mcy, mr = lunar_mask_params
        y_idx, x_idx = np.indices((h, w), dtype=np.float32)
        dist = np.hypot(x_idx - mcx, y_idx - mcy)
        m_bool = dist >= (mr - 1.0)
    else:
        m_bool = np.ones((h, w), dtype=bool)

    m = m_bool.astype(np.float32)
    acc = np.zeros((h, w), dtype=np.float32)
    total_w = 0.0

    for s in sigmas:
        s_val = float(s)
        # Convolución normalizada: (G * (I * m)) / (G * m)
        num = cv2.GaussianBlur(np.where(m_bool, lum, 0.0).astype(np.float32), (0, 0), s_val)
        den = cv2.GaussianBlur(m, (0, 0), s_val)
        mu = np.where(m_bool & (den >= 0.15), num / np.maximum(den, 1e-12), np.nan)

        d = np.where(np.isfinite(mu), lum - mu, 0.0).astype(np.float32)
        num_v = cv2.GaussianBlur(np.where(m_bool, d * d, 0.0).astype(np.float32), (0, 0), s_val)
        var = np.where(m_bool & (den >= 0.15), num_v / np.maximum(den, 1e-12), 0.0)
        sd = np.sqrt(np.maximum(var, 1e-8))

        c = np.where(np.isfinite(mu) & (sd > 1e-5), (lum - mu) / sd, 0.0)
        acc += np.arctan(k * c).astype(np.float32)
        total_w += 1.0

    acc /= max(total_w, 1.0)

    # Escalar arctan[-pi/2, pi/2] a [0, 1]
    acc_norm = np.clip((acc + (np.pi / 2.0)) / np.pi, 0.0, 1.0)

    # Contexto global por compresión gamma
    g = np.clip(lum, 0.0, 1.0) ** (1.0 / max(gamma, 0.1))

    # Mezcla armónica entre detalle multiescala normalizado y contexto global
    h_weight = float(np.clip(1.0 - blend_ratio, 0.0, 1.0))
    mgn_lum = np.clip(h_weight * g + (1.0 - h_weight) * acc_norm, 0.0, 1.0)

    if lunar_mask_params is not None:
        mgn_lum = np.where(m_bool, mgn_lum, 0.0)

    # Recomposición cromática lineal radiométrica (preserva H y S intactos)
    if img.ndim == 3:
        ratio = np.divide(mgn_lum, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        ratio = np.clip(ratio, 0.0, 5.0)
        result = img * ratio[..., None]
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
    radius_max_factor: float = 2.6
) -> np.ndarray:
    """
    Realza los filamentos finos mediante filtro de detalle bilateral enmascarado
    dentro de la zona coronal activa para evitar cualquier elevación de ruido en el fondo.
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

    enhanced = img + (detail * float(strength) * gate)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)
