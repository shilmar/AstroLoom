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
import astroloom_rs

HAS_RUST = True

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
    if ext in ('.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw', '.orf', '.raf', '.rw2', '.pef'):
        try:
            meta = astroloom_rs.scan_raw_metadata_rs(filepath)
            if meta.get("is_valid"):
                exp_t = float(meta.get("exposure_time", 0.0) or 0.0)
                if exp_t > 0.0:
                    info["exposure_s"] = exp_t
                iso_val = int(meta.get("iso", 0) or 0)
                if iso_val > 0:
                    info["iso"] = iso_val
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
    gray_f32 = np.ascontiguousarray(gray, dtype=np.float32)
    return astroloom_rs.optimize_center_circular_flux_rs(
        gray_f32,
        float(init_cx),
        float(init_cy),
        float(radius),
        mode,
        float(max_search_dist) if max_search_dist is not None else None
    )


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


def _prestretch_limb_luminance(gray: np.ndarray, force: bool = False) -> np.ndarray:
    """
    Pre-estira temporalmente la luminancia lineal de tomas ultrarrápidas muy oscuras
    (donde incluso el 0.1% más brillante de la corona está por debajo de 0.06)
    para revelar el borde lunar sin alterar los datos originales de fusión HDR.
    Usa p75 como suelo de cielo para no levantar el ruido de lectura por encima de hole_thresholds.
    """
    p_75, p_999 = np.percentile(gray, [75.0, 99.9])
    if not force and p_999 >= 0.06:
        return gray
    span = max(float(p_999 - p_75), 1e-6)
    g_norm = np.clip((gray - float(p_75)) / span, 0.0, 1.0)
    k = 40.0 if p_999 < 0.015 else 20.0
    stretched = (np.arcsinh(k * g_norm) / np.arcsinh(k)).astype(np.float32)
    return cv2.GaussianBlur(stretched, (0, 0), 1.5)


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
    if mode == "solar":
        gray = _prestretch_limb_luminance(gray)

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
        if mode == "solar" and best_hole is not None:
            p_scan = _radial_limb_scan(
                gray, init_cx, init_cy,
                max(5.0, expected_radius - 25.0), expected_radius + 25.0,
                num_rays=360, mode=mode
            )
            if p_scan is not None:
                scx, scy, sr = p_scan
                if 0.85 * expected_radius <= sr <= 1.15 * expected_radius and np.hypot(scx - init_cx, scy - init_cy) <= 35.0:
                    return float(scx), float(scy), float(expected_radius)
        max_search = min(35.0, max(12.0, expected_radius * 0.10))
        opt_cx, opt_cy = optimize_center_circular_flux(gray, init_cx, init_cy, expected_radius, mode=mode, max_search_dist=max_search)
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
    if mode == "solar":
        gray = _prestretch_limb_luminance(gray)

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
    ref_idx: int = 0,
    consume_input: bool = False
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    Alinea una lista de imágenes de bracketing al centro del limbo de la toma de referencia.
    Si consume_input=True, libera cada toma original de `images` tras alinearla para evitar duplicar RAM.
    """
    if not images or len(images) != len(centers):
        raise ValueError("El número de imágenes y centros debe coincidir.")

    ref_center = centers[ref_idx]
    aligned_images = []
    shifts = []

    for i in range(len(images)):
        img = images[i]
        c = centers[i]
        if consume_input:
            images[i] = None
        dx = float(ref_center[0] - c[0])
        dy = float(ref_center[1] - c[1])
        shifts.append((dx, dy))

        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            aligned_images.append(img if consume_input else img.copy())
        else:
            aligned = register_frame_to_center(img, c, ref_center, border_mode=cv2.BORDER_REPLICATE)
            aligned_images.append(aligned)

    return aligned_images, shifts


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Función de interpolación hermítica cúbica suave en [edge0, edge1]."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_asinh_stretch(
    image: np.ndarray,
    stretch_factor: float = 30.0,
    black_point: float = 0.0,
    shoulder_threshold: float = 0.70,
    highlight_ceiling: float = 1.0,
    white_point: float = 1.0,
    color_preserve: float = 0.0
) -> np.ndarray:
    """
    Estirado de arco seno hiperbólico (asinh) con hombro suave de altas luces (soft highlight shoulder).
    Soporta punto blanco extendido (white_point > 1.0) y preservación cromática Lupton (color_preserve en [0, 1])
    para mantener el detalle y el color H-alfa de protuberancias solares incluso con estirados altos.
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    wp = max(float(white_point), 1.0)
    cp = float(np.clip(color_preserve, 0.0, 1.0))
    return np.asarray(astroloom_rs.apply_asinh_stretch_rs(
        img,
        float(stretch_factor),
        float(black_point),
        float(shoulder_threshold),
        float(highlight_ceiling),
        wp,
        cp
    ), dtype=np.float32)


def balance_coronal_white_balance(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    ring_range: Tuple[float, float] = (1.2, 1.8),
    clip_upper: bool = True
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Neutraliza el balance de blancos de la corona solar estimando la mediana de color
    en el anillo coronal medio (1.2 - 1.8 R), eliminando dominantes cálidas o del
    cielo crepuscular según las directrices de AEL (white_balance_on_corona).
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    bal, gains = astroloom_rs.balance_coronal_white_balance_rs(
        img,
        float(center_x),
        float(center_y),
        float(radius),
        float(ring_range[0]),
        float(ring_range[1]),
        bool(clip_upper)
    )
    return np.asarray(bal, dtype=np.float32), (float(gains[0]), float(gains[1]), float(gains[2]))


def balance_coronal_thermal_dual(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    corona_warmth: float = 0.0,
    sky_temp: float = 0.0,
    transition_range: Tuple[float, float] = (1.5, 4.0),
    clip_upper: bool = True
) -> np.ndarray:
    """
    Ajuste térmico lineal dual para la corona y el fondo de cielo exterior.
    - corona_warmth: [-1.0, 1.0] modula la calidez cromática de la corona (dispersión de Thomson).
      Valores > 0 confieren un tono perlado/dorado suave; < 0 un tono más frío/azulado.
    - sky_temp: [-1.0, 1.0] modula la temperatura del cielo periférico.
      Valores < 0 profundizan el azul marino crepuscular; > 0 calientan el fondo.
    - transition_range: (r_in, r_out) múltiplos del radio lunar para el traspaso gradual continuo.
    """
    if image.ndim != 3 or (abs(corona_warmth) < 1e-4 and abs(sky_temp) < 1e-4):
        return image.copy()

    img = np.ascontiguousarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.hypot(x - center_x, y - center_y)

    r_ref = max(float(radius), 1.0)
    r_norm = dist / r_ref

    r_in, r_out = float(transition_range[0]), float(transition_range[1])
    t = _smoothstep(r_norm, r_in, r_out)

    cw = float(np.clip(corona_warmth, -1.0, 1.0))
    st = float(np.clip(sky_temp, -1.0, 1.0))
    k = (1.0 - t) * cw + t * st

    # Ganancias RGB térmicas con conservación aproximada de luminancia fotométrica:
    # 0.2126 * (0.30 * k) - 0.7152 * (0.06 * k) - 0.0722 * (0.30 * k) ~ 0
    g_r = 1.0 + 0.30 * k
    g_g = 1.0 - 0.06 * k
    g_b = 1.0 - 0.30 * k

    result = np.empty_like(img)
    for c_idx, g_val in enumerate((g_r, g_g, g_b)):
        ch = img[..., c_idx] * g_val
        result[..., c_idx] = np.clip(ch, 0.0, 1.0) if clip_upper else np.maximum(ch, 0.0)

    return result.astype(np.float32)


def subtract_twilight_gradient(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    r_min_factor: float = 6.0,
    degree: int = 1,
    robust: bool = True,
    clip_upper: bool = True
) -> np.ndarray:
    """
    Modela y sustrae el plano o gradiente crepuscular 2D en el cielo exterior
    fuera de la corona activa (distancia >= r_min_factor * radius).

    Ajusta una superficie polinómica de grado 1 (plano inclinado 2D) o 2 en
    espacio lineal fotométrico por mínimos cuadrados o regresión robusta,
    sustrayendo la inclinación lumínica sin alterar el nivel medio del fondo.
    """
    if image is None:
        return image

    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]

    # 1. Mapa de distancias radiales respecto al centro del eclipse
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    dx = x_coords - float(center_x)
    dy = y_coords - float(center_y)
    r_map = np.sqrt(dx * dx + dy * dy)

    r_ref = max(float(radius), 1.0)
    r_cut = r_ref * max(float(r_min_factor), 1.2)

    sky_mask = (r_map >= r_cut)
    if np.sum(sky_mask) < 200:
        r_cut = float(np.percentile(r_map, 85.0))
        sky_mask = (r_map >= r_cut)
        if np.sum(sky_mask) < 100:
            return img.copy()

    # 2. Filtrado de muestras para evitar estrellas hiperbrillantes
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img

    sky_lum = lum[sky_mask]
    p_lo = float(np.percentile(sky_lum, 1.0))
    p_hi = float(np.percentile(sky_lum, 97.0))
    valid_sky = sky_mask & (lum >= p_lo) & (lum <= p_hi)
    if np.sum(valid_sky) < 50:
        valid_sky = sky_mask

    # 3. Matriz de diseño polinómica 2D normalizada en [-1, 1]
    u = ((x_coords - (w * 0.5)) / (w * 0.5)).astype(np.float32)
    v = ((y_coords - (h * 0.5)) / (h * 0.5)).astype(np.float32)

    u_sub = u[valid_sky]
    v_sub = v[valid_sky]
    ones = np.ones_like(u_sub)

    if degree >= 2:
        A_sub = np.column_stack([ones, u_sub, v_sub, u_sub ** 2, v_sub ** 2, u_sub * v_sub])
        A_full = np.column_stack([
            np.ones(h * w, dtype=np.float32),
            u.ravel(), v.ravel(),
            (u ** 2).ravel(), (v ** 2).ravel(), (u * v).ravel()
        ])
    else:
        A_sub = np.column_stack([ones, u_sub, v_sub])
        A_full = np.column_stack([
            np.ones(h * w, dtype=np.float32),
            u.ravel(), v.ravel()
        ])

    result = img.copy()

    # 4. Ajuste y sustracción por canal manteniendo el nivel medio
    channels = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    for c_idx, ch in enumerate(channels):
        y_sub = ch[valid_sky].astype(np.float64)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_sub.astype(np.float64), y_sub, rcond=None)
            if robust and len(y_sub) > 60:
                res = y_sub - A_sub.astype(np.float64) @ coeffs
                med_res = np.median(res)
                mad_res = 1.4826 * np.median(np.abs(res - med_res))
                inliers = np.abs(res - med_res) <= (2.5 * max(mad_res, 1e-5))
                if np.sum(inliers) >= 30:
                    coeffs, _, _, _ = np.linalg.lstsq(A_sub[inliers].astype(np.float64), y_sub[inliers], rcond=None)

            surf_full = (A_full.astype(np.float64) @ coeffs).reshape(h, w).astype(np.float32)
            mean_bg = float(np.median(surf_full[valid_sky]))
            tilt = surf_full - mean_bg

            sub_ch = ch - tilt
            clipped = (np.clip(sub_ch, 0.0, 1.0) if clip_upper else np.maximum(sub_ch, 0.0)).astype(np.float32)
            if img.ndim == 2:
                result = clipped
            else:
                result[..., c_idx] = clipped
        except Exception:
            pass

    return result


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
    gradual_handover: bool = True,
    corona_warmth: float = 0.0,
    sky_temp: float = 0.0,
    equalize_sky_gradient: bool = False,
    clip_highlights: bool = True
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

    # Pre-calcular pedestales de cielo si procede
    bg_pedestals = []
    if subtract_sky:
        for img in sorted_images:
            if img.ndim == 3:
                bg_pedestals.append(np.percentile(img, 0.5, axis=(0, 1)).astype(np.float32).tolist())
            else:
                bg_pedestals.append([float(np.percentile(img, 0.5))])

    hdr_norm = astroloom_rs.fuse_hdr_bracketing_solar_rs(
        sorted_images,
        sorted_exp,
        float(saturation_threshold),
        float(noise_floor),
        lunar_mask_params,
        bool(gradual_handover),
        bg_pedestals if subtract_sky else None,
        bool(clip_highlights)
    )
    # Balance de blancos coronal automático (anillo 1.2 - 1.8 R) para eliminar dominantes de cielo
    if auto_wb_corona and lunar_mask_params is not None and hdr_norm.ndim == 3:
        mcx, mcy, mr = lunar_mask_params
        hdr_norm, _ = balance_coronal_white_balance(
            hdr_norm, mcx, mcy, mr, ring_range=(1.2, 1.8), clip_upper=clip_highlights
        )

    if (abs(corona_warmth) > 1e-4 or abs(sky_temp) > 1e-4) and lunar_mask_params is not None and hdr_norm.ndim == 3:
        mcx, mcy, mr = lunar_mask_params
        hdr_norm = balance_coronal_thermal_dual(
            hdr_norm, mcx, mcy, mr, corona_warmth=corona_warmth, sky_temp=sky_temp, clip_upper=clip_highlights
        )

    # Ecualización opcional de gradiente crepuscular 2D en espacio lineal antes del estirado
    if equalize_sky_gradient and lunar_mask_params is not None:
        mcx, mcy, mr = lunar_mask_params
        hdr_norm = subtract_twilight_gradient(
            hdr_norm, mcx, mcy, mr, r_min_factor=6.0, degree=1, clip_upper=clip_highlights
        )

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

    if partial_annuli == "observed":
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
            h_im, w_im = lum.shape[:2]
            y_coords, x_coords = np.indices((h_im, w_im), dtype=np.float32)
            disk_mask = np.hypot(x_coords - float(center_x), y_coords - float(center_y)) >= float(radius_min)
            rings_norm = display_detail_rings(
                clean_detail,
                valid_mask=disk_mask,
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

    # Protección bilateral de margen dinámico (headroom superior para altas luces e inferior para sombras)
    upper = np.maximum(1.0 - lum, 0.0)
    lower = np.maximum(lum, 0.0)
    mod_pos = streamer_mod * (0.4 + 0.6 * np.sqrt(upper))
    mod_neg = streamer_mod * lower * (0.3 + 0.7 * np.sqrt(lower))
    final_mod = np.where(streamer_mod > 0, mod_pos, mod_neg)

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


def apply_fnrgf(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    fourier_order: int = 4,
    bin_px: float = 1.0,
    robust: bool = True,
    blend_ratio: float = 0.55,
    radius_min: float = 10.0,
    radius_max_factor: float = 3.5,
    use_detail_rings: bool = False
) -> np.ndarray:
    """
    Fourier Normalizing Radial Graded Filter (FNRGF) (Druckmüllerová, Morgan & Habbal 2011).
    Modela la variación azimutal suave de los streamers y agujeros coronales mediante
    descomposición espectral angular ortogonal libre del efecto de cuadrícula cartesiana:
        B(r, theta) = a0(r) + sum_{k=1}^N [ ak(r) cos(k theta) + bk(r) sin(k theta) ]

    A diferencia de una integración continua sobre píxeles cartesianos (que resuena con la 
    simetría cuádruple de la rejilla cuadrada del sensor creando cruces espurias a 45°),
    este método muestrea la corona en M sectores angulares equiespaciados uniformemente,
    garantizando ortogonalidad trigonométrica exacta y modulando el detalle de filamentos
    de forma relativa para preservar intacto el fondo estelar y la estructura global.
    """
    if image is None:
        return None

    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]

    # 1. Coordenadas polares
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    dx = x_coords - float(center_x)
    dy = y_coords - float(center_y)
    r_map = np.sqrt(dx * dx + dy * dy)
    theta_map = (np.arctan2(dy, dx) + (2.0 * np.pi)) % (2.0 * np.pi)

    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()
    lum = lum.astype(np.float32)

    r_min = max(float(radius_min), 1.0)
    r_max = r_min * max(1.2, float(radius_max_factor))
    disk_mask = r_map >= r_min

    # 2. Estimación de ruido de cielo exterior
    sky_mask = (r_map >= r_max)
    if np.sum(sky_mask) > 100:
        sky_samples = lum[sky_mask]
        sky_bg = float(np.median(sky_samples))
        sky_noise = float(1.4826 * np.median(np.abs(sky_samples - sky_bg)))
    else:
        far_r = np.percentile(r_map, 85.0)
        sky_samples = lum[r_map >= far_r]
        sky_bg = float(np.median(sky_samples))
        sky_noise = float(1.4826 * np.median(np.abs(sky_samples - sky_bg)))
    sky_noise = max(sky_noise, 1e-4)

    # 3. Discretización angular uniforme (M sectores para ortogonalidad exacta anti-rejilla)
    M = 180  # 180 sectores = paso de 2.0 grados
    bin_size = max(float(bin_px), 0.5)
    r_bins = np.clip((r_map / bin_size).astype(np.int32), 0, None)
    max_bin = int(r_bins.max()) + 1
    th_bins = np.clip((theta_map / (2.0 * np.pi) * M).astype(np.int32), 0, M - 1)

    cell_idx = r_bins * M + th_bins
    flat_cells = cell_idx.ravel()
    flat_lum = lum.ravel()
    total_cells = max_bin * M

    cell_counts = np.bincount(flat_cells, minlength=total_cells)
    cell_sums = np.bincount(flat_cells, weights=flat_lum, minlength=total_cells)

    sec_grid = np.where(cell_counts >= 1, cell_sums / np.maximum(cell_counts, 1), 0.0).reshape(max_bin, M)

    # Rellenar sectores vacíos por interpolación circular para radios con cobertura parcial
    for b in range(max_bin):
        valid = cell_counts.reshape(max_bin, M)[b] >= 1
        if not np.all(valid) and np.any(valid):
            row = sec_grid[b]
            angles = np.arange(M)
            row[~valid] = np.interp(angles[~valid], angles[valid], row[valid], period=M)

    # 4. Transformada de Fourier 1D por fila (armónicos azimutales puros)
    N = int(np.clip(fourier_order, 1, 8))
    rfft = np.fft.rfft(sec_grid, axis=1)

    # Filtramos todo armónico superior a N
    if rfft.shape[1] > (N + 1):
        rfft[:, (N + 1):] = 0.0

    # Reconstrucción del fondo azimutal suave en la rejilla angular
    smooth_grid = np.fft.irfft(rfft, n=M, axis=1).astype(np.float32)

    # Suavizado radial gaussiano para asegurar continuidad sin bandas concéntricas
    if max_bin > 15:
        smooth_grid = cv2.GaussianBlur(smooth_grid, (1, 25), 6.0)

    # Muestreo del fondo azimutal para cada píxel de la imagen
    bkg_fourier = smooth_grid[r_bins, th_bins]

    # 5. Extracción de detalle y dispersión local
    diff = np.where(disk_mask, lum - bkg_fourier, 0.0).astype(np.float32)

    # Estimación de dispersión robusta sigma(r) por anillos
    flat_r = r_bins.ravel()
    flat_diff = np.abs(diff).ravel()
    counts_r = np.bincount(flat_r, minlength=max_bin)
    sum_diff_sq = np.bincount(flat_r, weights=flat_diff.astype(np.float64) ** 2, minlength=max_bin)
    var_vals = np.where(counts_r >= 30, sum_diff_sq / np.maximum(counts_r, 1), 0.0)
    sigma_prof = np.sqrt(np.maximum(var_vals, 1e-8)).astype(np.float32)

    if max_bin > 15:
        sigma_prof = cv2.GaussianBlur(sigma_prof[:, None], (1, 25), 6.0).ravel()

    min_sigma = max(sky_noise * 1.5, 0.002)
    sigma_prof = np.maximum(sigma_prof, min_sigma)
    sigma_map = sigma_prof[r_bins]

    raw_detail = np.where(disk_mask, diff / np.maximum(sigma_map, 1e-5), 0.0).astype(np.float32)
    clean_detail = cv2.bilateralFilter(raw_detail, d=7, sigmaColor=0.35, sigmaSpace=4.0)

    if use_detail_rings:
        rings_norm = display_detail_rings(
            clean_detail,
            valid_mask=disk_mask,
            center_x=float(center_x),
            center_y=float(center_y)
        )
        detail_factor = np.tanh((rings_norm - 0.5) * 3.0)
    else:
        detail_factor = np.tanh(clean_detail / 2.0)

    # 6. Compuerta radial suave y modulación fotométrica relativa
    gate_start = r_min * (1.0 + (float(radius_max_factor) - 1.0) * 0.70)
    t = np.clip((r_map - gate_start) / max(r_max - gate_start, 1.0), 0.0, 1.0)
    radial_gate = 1.0 - (t * t * (3.0 - 2.0 * t))
    radial_gate = np.where(r_map > r_max, 0.0, radial_gate)

    snr_factor = np.clip((sigma_map - 1.2 * sky_noise) / (sigma_map + 1e-6), 0.0, 1.0)
    blend = float(np.clip(blend_ratio, 0.0, 1.0))
    mod_strength = blend * radial_gate * snr_factor

    # Modo científico 100% mapa de contraste FNRGF
    if blend >= 0.98:
        map_fnrgf = np.clip(0.5 + 0.5 * detail_factor, 0.0, 1.0)
        map_fnrgf = np.where(disk_mask, map_fnrgf, 0.0).astype(np.float32)
        if img.ndim == 3:
            return np.repeat(map_fnrgf[..., None], 3, axis=2)
        return map_fnrgf

    # Modo fotográfico (realce relativo sin black-clipping ni alteración del fondo)
    new_lum = lum * (1.0 + 0.80 * detail_factor * mod_strength)
    new_lum = np.clip(new_lum, 0.0, 1.0)
    new_lum = np.where(disk_mask, new_lum, 0.0).astype(np.float32)

    # 7. Recomposición cromática lineal exacta
    if img.ndim == 3:
        ratio = np.divide(new_lum, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        ratio = np.clip(ratio, 0.0, 4.0)
        result = img * ratio[..., None]
        result = np.where(disk_mask[..., None], result, 0.0)
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

    rgb_c = np.ascontiguousarray(img[..., :3], dtype=np.float32)
    raw_mask = np.asarray(astroloom_rs.compute_red_excess_mask_rs(
        rgb_c, float(center_x), float(center_y), r_min, r_max, float(threshold_ratio)
    ), dtype=bool)

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
    # Margen dinámico bilateral: headroom superior para altas luces y headroom inferior
    # proporcional a la luminancia local para evitar clipping destructivo a negro (black clipping)
    upper = np.maximum(1.0 - lum, 0.0)
    lower = np.maximum(lum, 0.0)
    mod_pos = streamer_mod * (0.4 + 0.6 * np.sqrt(upper))
    mod_neg = streamer_mod * lower * (0.3 + 0.7 * np.sqrt(lower))
    final_mod = np.where(streamer_mod > 0, mod_pos, mod_neg)

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

    # Suavizado gaussiano multiescala según SNR radial (Larson & Sekanina 1984 / AEL 1.4.2)
    # Suprime el ruido de disparo y fluctuaciones estocásticas que causan speckles, islas y huecos oscuros
    r_norm = r_map / max(float(radius_min), 1.0)
    g1 = cv2.GaussianBlur(lum, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(lum, (0, 0), 2.5)
    w_g = np.clip((r_norm - 1.2) / 1.5, 0.0, 1.0)
    lum_eval = (g1 * (1.0 - w_g) + g2 * w_g).astype(np.float32)

    lum_c = np.ascontiguousarray(lum_eval, dtype=np.float32)
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

    # Realce asimétrico de crestas (AEL / Larson-Sekanina): destaca filamentos y chorros
    # coronales positivos sin sustraer luz destructiva de valles, cavidades o absorciones de nubes
    pos_detail = np.maximum(clipped_detail, 0.0)
    streamer_mod = pos_detail * radial_gate * float(np.clip(blend_ratio, 0.0, 1.0))
    upper = np.maximum(1.0 - lum, 0.0)
    final_mod = streamer_mod * (0.4 + 0.6 * np.sqrt(upper))

    # Compensación de déficit azimutal (nubes y valles difusos): evita colapso a mancha oscura frente a filamentos
    bg_env = cv2.GaussianBlur(lum, (0, 0), sigmaX=max(float(radius_min) * 0.10, 25.0))
    cloud_deficit = np.maximum(bg_env - lum, 0.0) * radial_gate
    lum_base = lum + 0.50 * cloud_deficit

    new_lum = np.clip(lum_base + final_mod, 0.0, 1.0)
    new_lum[r_map < radius_min] = 0.0

    if img.ndim == 3:
        ratio = np.divide(new_lum, np.maximum(lum, 1e-4), out=np.zeros_like(lum), where=lum > 1e-6)
        ratio = np.clip(ratio, 0.0, 4.0)
        result = img * ratio[..., None]
        result[r_map < radius_min] = 0.0
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
    blend_ratio: float = 1.0,
    radius_max_factor: float = 3.5
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

    # 1. Mapeo a coordenadas polares (r, theta)
    # cv2.warpPolar con dsize=(max_r, n_theta) asigna cols -> r y rows -> theta.
    # Al transponer (.T), la matriz polar queda con shape (max_r, n_theta)
    # donde las filas corresponden al radio r y las columnas al ángulo theta.
    dst_polar = np.zeros((n_theta, max_r), dtype=np.float32)
    raw_polar = cv2.warpPolar(lum, (max_r, n_theta), (center_x, center_y), max_r, cv2.WARP_POLAR_LINEAR, dst=dst_polar)
    polar = np.nan_to_num(raw_polar.T, copy=False)

    # 2. Mediana móvil a lo largo de theta (columnas) para cada círculo de radio r (filas)
    win = int(round(2.0 * window_deg / 360.0 * n_theta))
    if win % 2 == 0:
        win += 1
    win = max(win, 3)

    polar_c = np.ascontiguousarray(polar, dtype=np.float32)
    f_polar = np.asarray(astroloom_rs.polar_circular_median_filter_rs(polar_c, win), dtype=np.float32)

    # 3. Remapeo inverso a coordenadas cartesianas y suavizado cartesiano (sigma = 4.0 px)
    # f_polar.T tiene shape (n_theta, max_r) como espera WARP_INVERSE_MAP
    dst_cart = np.zeros((h, w), dtype=np.float32)
    cart_f = cv2.warpPolar(np.ascontiguousarray(f_polar.T), (w, h), (center_x, center_y), max_r, cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP, dst=dst_cart)
    cart_f = np.nan_to_num(cart_f, copy=False)
    f_bg = cv2.GaussianBlur(cart_f, (0, 0), 4.0)

    # 4. Potencia reducida I / (F + t0)^c0 y compuerta radial continua
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    r_map = np.hypot(x_coords - center_x, y_coords - center_y)
    mask_solar = (r_map >= max(radius_min - 2.5, 0.0))

    r_max = radius_min * max(1.2, float(radius_max_factor))
    gate_start = radius_min * (1.0 + (float(radius_max_factor) - 1.0) * 0.75)
    t = np.clip((r_map - gate_start) / max(r_max - gate_start, 1.0), 0.0, 1.0)
    radial_gate = 1.0 - (t * t * (3.0 - 2.0 * t))
    radial_gate = np.where(r_map > r_max, 0.0, radial_gate)

    corona_mask = mask_solar & (r_map <= r_max)
    valid_f = f_bg[corona_mask & (f_bg > 0)]
    t0 = float(np.median(valid_f)) if valid_f.size > 0 else 1e-4

    denom = np.power(np.maximum(f_bg + t0, 1e-6), float(c0))
    i_swap = np.where(mask_solar, lum / denom, 0.0).astype(np.float32)

    # 5. Normalización de alto contraste y transición suave hacia el cielo exterior
    c_vals = i_swap[corona_mask & (i_swap > 0)]
    if c_vals.size > 10:
        p_low = float(np.percentile(c_vals, 2.0))
        p_high = float(np.percentile(c_vals, 98.0))
        scaled_c = np.clip((i_swap - p_low) / max(p_high - p_low, 1e-4), 0.0, 1.0)
    else:
        valid_swap = i_swap[mask_solar & (i_swap > 0)]
        med_swap = float(np.median(valid_swap)) if valid_swap.size > 0 else 1.0
        i_norm = np.power(np.maximum(i_swap / max(med_swap, 1e-6), 0.0), 0.25)
        scaled_c = np.clip((i_norm - 0.5) / 1.5, 0.0, 1.0)

    eff_scaled = scaled_c * radial_gate + lum * (1.0 - radial_gate)
    scaled = np.where(mask_solar, eff_scaled, 0.0).astype(np.float32)

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

def _b3_normalized(a: np.ndarray, m: np.ndarray, step: int) -> np.ndarray:
    """
    Convolución separable B3 à-trous 2D normalizada por soporte observado:
    conv(a * m, step) / max(conv(m, step), 1e-20).
    """
    a_c = np.ascontiguousarray(a, dtype=np.float32)
    m_c = np.ascontiguousarray(m, dtype=bool)
    return np.asarray(astroloom_rs.b3_normalized_rs(a_c, m_c, int(step)), dtype=np.float32)


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
    Acelerado 100% de forma nativa en Rust con Rayon.
    """
    img_c = np.ascontiguousarray(image, dtype=np.float32)
    lo, hi = completeness
    res = astroloom_rs.apply_wow_edge_aware_rs(
        img_c,
        float(center_x),
        float(center_y),
        float(radius_min),
        int(n_scales),
        bool(guided),
        float(lo),
        float(hi),
        float(blend_ratio),
        bool(use_detail_rings)
    )
    return np.asarray(res, dtype=np.float32)


def compute_dark_structure_mask(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius_mask: float,
    radius_max_factor: float = 3.5,
    layers: int = 5,
    denoise_fine_scales: int = 1,
    threshold_sigma: float = 0.0,
    prominence_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calcula una máscara normalizada [0.0, 1.0] que aísla selectivamente las estructuras
    oscuras (cavidades coronales, canales inter-streamers, valles de absorción).

    Utiliza la descomposición multiescala à-trous B3 acelerada en Rust.
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim == 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    h, w = lum.shape[:2]
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    r_map = np.hypot(x_coords - center_x, y_coords - center_y)

    r_max = radius_mask * max(1.2, float(radius_max_factor))
    disk_valid = (r_map >= radius_mask * 0.98) & (r_map <= r_max * 1.05)

    # 1. Descomposición multiescala à-trous B3 para fondo regional
    current = lum.copy()
    for s in range(max(1, int(layers))):
        step = 1 << s
        current = _b3_normalized(current, disk_valid, step)
    base_smooth = current

    # 2. Suavizado fino de alta frecuencia para descartar ruido de disparo
    if denoise_fine_scales > 0:
        fine = lum.copy()
        for s in range(int(denoise_fine_scales)):
            step = 1 << s
            fine = _b3_normalized(fine, disk_valid, step)
    else:
        fine = lum

    # 3. Aislamiento del residuo de valles / estructuras por debajo del fondo
    dark_delta = np.maximum(0.0, base_smooth - fine)

    # 4. Umbral adaptativo por MAD si se requiere supresión extra de ruido
    if threshold_sigma > 0.0:
        valid_deltas = dark_delta[disk_valid]
        if len(valid_deltas) > 0:
            median_d = np.median(valid_deltas)
            mad = np.median(np.abs(valid_deltas - median_d))
            tau = float(threshold_sigma) * 1.4826 * mad
            dark_delta = np.maximum(0.0, dark_delta - tau)

    # 5. Compuerta radial suave (protege disco lunar interior y cielo exterior)
    radial_gate = (
        (1.0 - _smoothstep(r_map, r_max * 0.85, r_max))
        * _smoothstep(r_map, radius_mask * 0.98, radius_mask * 1.05)
    )
    mask = dark_delta * radial_gate

    # 6. Protección de protuberancias H-alfa cromáticas
    if prominence_mask is not None:
        p_mask = np.clip(prominence_mask.astype(np.float32), 0.0, 1.0)
        mask *= (1.0 - p_mask)

    # Normalización al rango [0, 1] basada en el percentil 99.8 para evitar saturación
    valid_vals = mask[disk_valid]
    if len(valid_vals) > 0:
        p99 = float(np.percentile(valid_vals, 99.8))
        if p99 > 1e-6:
            mask = np.clip(mask / p99, 0.0, 1.0)

    return mask.astype(np.float32)


def apply_dark_structure_enhancement(
    image: np.ndarray,
    amount: float = 0.40,
    center_x: float = 0.0,
    center_y: float = 0.0,
    radius_mask: float = 100.0,
    radius_max_factor: float = 3.5,
    layers: int = 5,
    method: str = "mtf",
    denoise_fine_scales: int = 1,
    threshold_sigma: float = 0.0,
    prominence_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Aplica realce selectivo de estructuras oscuras (Dark Structure Enhancement / DSE)
    sobre la corona solar en espacio float32 [0.0, 1.0].

    Métodos:
      - 'mtf': Ajuste racional de tonos medios (Midtones Transfer Function). Imposible causar black-clipping.
      - 'subtractive': Modulación sustractiva ponderada por la luminancia local.
    """
    if amount <= 0.001:
        return image.copy()

    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    is_rgb = img.ndim == 3 and img.shape[2] == 3

    if is_rgb:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img.copy()

    mask = compute_dark_structure_mask(
        image=lum,
        center_x=center_x,
        center_y=center_y,
        radius_mask=radius_mask,
        radius_max_factor=radius_max_factor,
        layers=layers,
        denoise_fine_scales=denoise_fine_scales,
        threshold_sigma=threshold_sigma,
        prominence_mask=prominence_mask
    )

    amt = float(np.clip(amount, 0.0, 1.0))

    if method == "mtf":
        m = 0.5 + (0.45 * amt * mask)
        m_clamped = np.clip(m, 1e-4, 1.0 - 1e-4).astype(np.float32)
        denom = (2.0 * m_clamped - 1.0) * lum - m_clamped
        denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom)
        lum_new = np.clip((m_clamped - 1.0) * lum / denom, 0.0, 1.0)
    elif method == "subtractive":
        lum_new = np.maximum(0.0, lum * (1.0 - amt * mask))
    else:
        raise ValueError(f"Método de realce desconocido: {method}")

    if is_rgb:
        ratio = lum_new / np.maximum(lum, 1e-6)
        enhanced = np.clip(img * ratio[..., None], 0.0, 1.0)
    else:
        enhanced = np.clip(lum_new, 0.0, 1.0)

    return enhanced.astype(np.float32)


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
    Acelerado nativamente en Rust con Rayon y realfft.
    """
    img_c = np.ascontiguousarray(image, dtype=np.float32)
    res = astroloom_rs.apply_nafe_rs(
        img_c,
        float(center_x),
        float(center_y),
        float(radius_min),
        float(sigma) if sigma is not None and sigma > 0 else None,
        int(n),
        float(level_step),
        float(blend_ratio),
        bool(use_overlay)
    )
    return np.asarray(res, dtype=np.float32)


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

    if "fnrgf" not in data:
        _log("[PSD CORONAL] Procesando capa 01b - FNRGF Azimutal de Fourier...")
        data["fnrgf"] = apply_fnrgf(base_rgb, center_x, center_y, radius_min=radius_min, blend_ratio=0.55)

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

    if "dark_structure" in data and data["dark_structure"] is None:
        _log("[PSD CORONAL] Procesando capa 08b - Estructuras Oscuras (Cavidades)...")
        data["dark_structure"] = apply_dark_structure_enhancement(
            base_rgb, center_x=center_x, center_y=center_y, radius_mask=radius_min, amount=0.40
        )

    def _prep_rgb(arr: np.ndarray) -> np.ndarray:
        clipped = np.clip(arr, 0.0, 1.0).astype(np.float32)
        if clipped.ndim == 2:
            clipped = np.repeat(clipped[..., None], 3, axis=2)
        if clipped.shape[:2] != (h, w):
            clipped = cv2.resize(clipped, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(clipped, dtype=np.float32)

    # Construcción de la pila de capas PSD (de abajo a arriba)
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

    # 08b. Estructuras Oscuras (Cavidades)
    if "dark_structure" in data and data["dark_structure"] is not None:
        layer_records.append({
            "name": "08b - Estructuras Oscuras (Cavidades)",
            "top": 0, "left": 0, "bottom": h, "right": w,
            "blend_mode": "mul",
            "opacity": 102,  # 40%
            "visible": False,
            "rgb": _prep_rgb(data["dark_structure"])
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

    # 01b. FNRGF
    layer_records.append({
        "name": "01b - FNRGF Azimutal de Fourier",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "over",
        "opacity": 140,  # ~55%
        "visible": False,
        "rgb": _prep_rgb(data["fnrgf"])
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

    file_size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
    desc = f"PSD Multicapa ({len(layer_records)} capas coronales, {depth}-bit, {file_size_mb:.1f} MB)"
    _log(f"[PSD CORONAL FINALIZADO] {os.path.basename(filepath)} ({desc})")

    return True, desc, len(layer_records)
