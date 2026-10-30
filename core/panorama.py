# core/panorama.py
"""
core/panorama.py - Módulo de Composición Panorámica y Mosaicos Nocturnos (Nightscape Stitcher).

Diseñado específicamente para fotografía de paisaje nocturno y astrofotografía:
1. Extracción de metadatos de paneles (focal, timestamp, ISO, orientación).
2. Detección y descripción de rasgos mejorada para astrofotografía (CLAHE + estirado asinh/gamma
   sobre una copia local para capturar tanto estrellas como relieves terrestres en sombras profundas).
3. Soporte nativo multifila (multi-row) mediante grafo 2D de conectividad y emparejamiento bidireccional.
4. Estimación robusta de homografías mediante RANSAC con umbrales adaptados a cielos oscuros.
5. Proyecciones cartográficas: Cilíndrica (Cylindrical), Esférica (Spherical) y Plana (Rectilinear/Plane).
6. Compensación de viñeteo / ganancia entre paneles y corrección de horizonte.
7. Fusión multicanal piramidal (Laplacian Multi-Band Blending) y degradado por mapa de distancias.
8. Algoritmo O(H x W) de Recorte Automático de Rectángulo Máximo Inscrito (Zero-border auto-crop).
9. Mantenimiento del flujo de trabajo de 16/32-bit float para exportación directa al Revelador.
"""

import os
import gc
import json
import time
import tempfile
import shutil
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import tifffile
from PIL import Image, ExifTags
from scipy.spatial import cKDTree

from core.stacking import load_image_as_float32, get_image_dimensions


@dataclass
class PanoramaPanelMetadata:
    """Metadatos de cada tesela o panel que compone la panorámica."""
    filepath: str
    filename: str
    index: int = 0
    row: int = 0
    col: int = 0
    width: int = 0
    height: int = 0
    focal_length_mm: Optional[float] = None
    iso: Optional[int] = None
    exposure_s: Optional[float] = None
    time_str: str = "--:--:--"
    timestamp: float = 0.0
    is_included: bool = True


@dataclass
class PanoramaSettings:
    """Configuración del proceso de cosido panorámico."""
    projection: str = "spherical"          # "spherical", "cylindrical", "plane"
    astro_mode: bool = True                # Pre-realce CLAHE para cielo oscuro y suelo
    confidence_thresh: float = 0.45        # Umbral de confianza / Lowe ratio (0.3 a 0.8)
    min_inliers: int = 6                   # Mínimo número de puntos coincidentes (6 para noche)
    blend_mode: str = "multiband"          # "multiband" (pirámide laplaciana), "feather" (degradado suave)
    blend_bands: int = 5                   # Niveles de la pirámide laplaciana (5 para transición imperceptible)
    exposure_comp: bool = True             # Compensación de ganancia / viñeteo
    wave_correction: bool = True           # Nivelación horizontal de horizonte
    auto_crop: bool = True                 # Recorte automático del rectángulo máximo inscrito
    layout_mode: str = "auto"              # "auto" (grafo 2D) o "grid" (asistente matriz MxN)
    grid_rows: int = 1
    grid_cols: int = 1
    focal_length_override: Optional[float] = None  # Focal en mm si no está en EXIF
    burst_stacking: bool = False           # Activar apilado previo por panel
    shots_per_panel: int = 0               # 0 = auto por tiempo, >0 = tomas fijas por panel
    mount_mode: str = "tripod"             # "tripod" (trípode fijo) o "tracker" (star tracker)
    use_auto_mask: bool = True             # Desacoplamiento cielo/suelo con auto-máscara IA y anti-ghosting
    vignette_amount: float = 0.5           # Fuerza de compensación de viñeteo (0.0 a 1.0)
    star_matching: bool = True             # Alineación guiada por identificación de estrellas (asterismos)
    arch_curvature: float = 0.0            # Deformación de arco / curvatura de horizonte (-0.25 a +0.25)


@dataclass
class PanoramaResult:
    """Resultado del cosido panorámico."""
    image: np.ndarray                      # Imagen cosida completa float32 RGB [0.0, 1.0]
    cropped_image: Optional[np.ndarray]    # Imagen tras auto-recorte (o None si no aplica)
    crop_rect: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    mask: np.ndarray                       # Máscara binaria del área cosida (uint8 0/255)
    panels_stitched: int
    total_panels: int
    resolution: Tuple[int, int]            # (ancho, alto)
    projection: str
    processing_time_s: float
    message: str = "Éxito"


def extract_panel_metadata(filepath: str, index: int = 0) -> PanoramaPanelMetadata:
    """
    Extrae la información esencial de cada panel (focal, tiempo de captura, dimensiones, etc.).
    Soporta archivos RAW (NEF, CR2, CR3, ARW, DNG), TIFF y JPG.
    """
    filename = os.path.basename(filepath)
    meta = PanoramaPanelMetadata(
        filepath=filepath,
        filename=filename,
        index=index
    )

    # 1. Dimensiones rápidas
    try:
        h, w = get_image_dimensions(filepath)
        meta.width = w
        meta.height = h
    except Exception:
        meta.width = 1920
        meta.height = 1080

    ext = os.path.splitext(filepath)[1].lower()

    # 2. RAW / TIFF mediante tifffile (acceso rápido a etiquetas EXIF sub-IFD)
    if ext in ['.nef', '.cr2', '.dng', '.tif', '.tiff', '.raw', '.arw']:
        try:
            with tifffile.TiffFile(filepath) as tif:
                if tif.pages:
                    p = tif.pages[0]
                    # Fecha tag 306 (DateTime) o tag 36867 (DateTimeOriginal)
                    dt_val = None
                    if 306 in p.tags:
                        dt_val = str(p.tags[306].value)
                    elif 36867 in p.tags:
                        dt_val = str(p.tags[36867].value)
                    if dt_val:
                        parts = dt_val.split()
                        if len(parts) >= 2:
                            meta.time_str = parts[1]

                    # EXIF sub-IFD (tag 34665)
                    if 34665 in p.tags:
                        exif = p.tags[34665].value
                        if isinstance(exif, dict):
                            focal = exif.get("FocalLengthIn35mmFilm") or exif.get("FocalLength")
                            if focal:
                                if isinstance(focal, (tuple, list)) and len(focal) == 2:
                                    meta.focal_length_mm = float(focal[0]) / max(1.0, float(focal[1]))
                                else:
                                    meta.focal_length_mm = float(focal)

                            iso = exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")
                            if iso:
                                if isinstance(iso, (tuple, list)):
                                    meta.iso = int(iso[0])
                                else:
                                    meta.iso = int(iso)

                            exp = exif.get("ExposureTime")
                            if exp:
                                if isinstance(exp, (tuple, list)) and len(exp) == 2:
                                    meta.exposure_s = float(exp[0]) / max(0.0001, float(exp[1]))
                                else:
                                    meta.exposure_s = float(exp)
        except Exception:
            pass

    # 3. Fallback con PIL para JPG o archivos estándar
    if meta.focal_length_mm is None or meta.time_str == "--:--:--":
        try:
            with Image.open(filepath) as img:
                exif_data = img._getexif()
                if exif_data:
                    exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
                    focal = exif.get("FocalLengthIn35mmFilm") or exif.get("FocalLength")
                    if focal and meta.focal_length_mm is None:
                        if isinstance(focal, (tuple, list)) and len(focal) == 2:
                            meta.focal_length_mm = float(focal[0]) / max(1.0, float(focal[1]))
                        else:
                            meta.focal_length_mm = float(focal)

                    dt = exif.get("DateTimeOriginal") or exif.get("DateTime")
                    if dt and meta.time_str == "--:--:--":
                        parts = str(dt).split()
                        if len(parts) >= 2:
                            meta.time_str = parts[1]

                    if meta.iso is None:
                        meta.iso = exif.get("ISOSpeedRatings")
        except Exception:
            pass

    # 4. Fallback de fecha por fecha de modificación
    if meta.time_str == "--:--:--":
        try:
            mtime = os.path.getmtime(filepath)
            meta.timestamp = float(mtime)
            meta.time_str = time.strftime("%H:%M:%S", time.localtime(mtime))
        except Exception:
            pass

    return meta


def extract_astro_features(
    image_gray: np.ndarray,
    astro_mode: bool = True,
    max_features: int = 4000
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Extrae puntos clave y descriptores SIFT con pre-realce astro específico.
    Aplica CLAHE + estirado suave sobre sombras para capturar estrellas y textura
    terrestre con alta repetibilidad.
    """
    if astro_mode:
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        prep = clahe.apply(image_gray)
    else:
        prep = image_gray

    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.010,
        edgeThreshold=12,
        sigma=1.4
    )
    kps, descs = sift.detectAndCompute(prep, None)
    return kps, descs


def find_maximum_inscribed_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calcula el rectángulo de área máxima completamente contenido dentro de la máscara binaria
    utilizando el algoritmo O(H x W) basado en histogramas acumulativos.
    Garantiza un auto-recorte perfecto eliminando al 100% los bordes negros irregulares.
    Retorna (x, y, ancho, alto).
    """
    h, w = mask.shape[:2]
    # Reducción de escala para cálculo ultra-rápido si la máscara es gigantesca
    scale = 1.0
    if max(h, w) > 2000:
        scale = 1000.0 / float(max(h, w))
        sh = max(10, int(h * scale))
        sw = max(10, int(w * scale))
        bin_mask = (cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        ch, cw = sh, sw
    else:
        bin_mask = (mask > 127).astype(np.uint8)
        ch, cw = h, w

    heights = np.zeros(cw, dtype=np.int32)
    max_area = 0
    best_rect = (0, 0, cw, ch)

    for r in range(ch):
        heights = np.where(bin_mask[r] == 1, heights + 1, 0)

        stack = []
        for c in range(cw + 1):
            curr_h = heights[c] if c < cw else 0
            while stack and heights[stack[-1]] >= curr_h:
                bar_idx = stack.pop()
                bar_h = heights[bar_idx]
                width = c if not stack else (c - stack[-1] - 1)
                area = bar_h * width
                if area > max_area:
                    max_area = area
                    top_y = r - bar_h + 1
                    left_x = stack[-1] + 1 if stack else 0
                    best_rect = (left_x, top_y, width, bar_h)
            stack.append(c)

    if scale != 1.0:
        bx, by, bw, bh = best_rect
        rx = int(round(bx / scale))
        ry = int(round(by / scale))
        rw = int(round(bw / scale))
        rh = int(round(bh / scale))
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))
        rw = max(1, min(w - rx, rw))
        rh = max(1, min(h - ry, rh))
        return rx, ry, rw, rh

    return best_rect


def warp_image_cylindrical(
    image: np.ndarray,
    focal_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica proyección cilíndrica nativa a una imagen y genera su máscara válida.
    x_cyl = f * tan( (x - cx) / f ) + cx
    y_cyl = (y - cy) / cos( (x - cx) / f ) + cy
    """
    h, w = image.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    # Malla de coordenadas de destino
    x_c = np.arange(w, dtype=np.float32)
    y_c = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_c, y_c)

    theta = (x_grid - cx) / focal_px
    x_orig = focal_px * np.tan(theta) + cx
    y_orig = (y_grid - cy) / np.cos(theta) + cy

    # Válidos solo donde cos(theta) > 0 y dentro de la imagen original
    valid = (np.cos(theta) > 0.05) & (x_orig >= 0) & (x_orig < w - 1) & (y_orig >= 0) & (y_orig < h - 1)

    map_x = x_orig.astype(np.float32)
    map_y = y_orig.astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = (valid.astype(np.uint8)) * 255
    return warped, mask


def warp_image_spherical(
    image: np.ndarray,
    focal_px: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica proyección esférica nativa a una imagen y genera su máscara válida.
    """
    h, w = image.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    x_c = np.arange(w, dtype=np.float32)
    y_c = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_c, y_c)

    theta = (x_grid - cx) / focal_px
    phi = (y_grid - cy) / focal_px

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_orig = focal_px * (sin_theta / (cos_theta * cos_phi + 1e-7)) + cx
    y_orig = focal_px * (sin_phi / (cos_theta * cos_phi + 1e-7)) + cy

    valid = (cos_theta > 0.1) & (x_orig >= 0) & (x_orig < w - 1) & (y_orig >= 0) & (y_orig < h - 1)

    map_x = x_orig.astype(np.float32)
    map_y = y_orig.astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = (valid.astype(np.uint8)) * 255
    return warped, mask


def extract_bright_stars(
    image: np.ndarray,
    max_stars: int = 70,
    min_distance: int = 12
) -> np.ndarray:
    """
    Detecta y extrae centroides subpíxel de las estrellas más brillantes.
    Aísla fuentes puntuales mediante filtrado morfológico Top-Hat y suprime
    nebulosas, gradientes y paisaje terrestre.
    Retorna un array (N, 2) de coordenadas [x, y].
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if gray.dtype != np.uint8:
        gray_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        gray_u8 = gray

    k_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k_tophat)

    p_thresh = max(18, int(np.percentile(tophat, 98.8)))
    _, thresh_img = cv2.threshold(tophat, p_thresh, 255, cv2.THRESH_TOZERO)

    dil = cv2.dilate(thresh_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    peaks = (thresh_img == dil) & (thresh_img > 0)
    ys, xs = np.where(peaks)

    if len(xs) < 4:
        p_thresh = max(10, int(np.percentile(tophat, 97.0)))
        _, thresh_img = cv2.threshold(tophat, p_thresh, 255, cv2.THRESH_TOZERO)
        dil = cv2.dilate(thresh_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        peaks = (thresh_img == dil) & (thresh_img > 0)
        ys, xs = np.where(peaks)
        if len(xs) < 4:
            return np.empty((0, 2), dtype=np.float32)

    intensities = tophat[ys, xs]
    sort_idx = np.argsort(intensities)[::-1]
    xs, ys = xs[sort_idx], ys[sort_idx]

    selected = []
    min_dist_sq = min_distance * min_distance
    for x, y in zip(xs, ys):
        too_close = False
        for sx, sy in selected:
            if (x - sx) ** 2 + (y - sy) ** 2 < min_dist_sq:
                too_close = True
                break
        if not too_close:
            selected.append((float(x), float(y)))
            if len(selected) >= max_stars:
                break

    if len(selected) < 4:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array(selected, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    refined = cv2.cornerSubPix(gray_u8, pts, (3, 3), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def build_star_asterisms(
    pts: np.ndarray,
    k_neighbors: int = 6
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Construye asterismos de 4 estrellas (quads) normalizados respecto a la base de
    máxima separación, invariantes a escala, traslación y rotación 2D.
    Retorna (features (M, 4), quads con índices en pts).
    """
    if len(pts) < 4:
        return np.empty((0, 4), dtype=np.float32), []

    tree = cKDTree(pts)
    _, idxs = tree.query(pts, k=min(k_neighbors + 1, len(pts)))

    feats = []
    quads = []
    seen = set()

    for i in range(len(pts)):
        nbrs = idxs[i][1:]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                for c in range(b + 1, len(nbrs)):
                    q_indices = tuple(sorted([i, nbrs[a], nbrs[b], nbrs[c]]))
                    if q_indices in seen:
                        continue
                    seen.add(q_indices)

                    q_pts = pts[list(q_indices)]
                    max_d = -1.0
                    best_pair = (0, 1)
                    for p_i in range(4):
                        for p_j in range(p_i + 1, 4):
                            d = np.linalg.norm(q_pts[p_i] - q_pts[p_j])
                            if d > max_d:
                                max_d = d
                                best_pair = (p_i, p_j)

                    iA, iB = best_pair
                    other = [idx for idx in range(4) if idx not in best_pair]
                    iC, iD = other[0], other[1]

                    vA, vB = q_pts[iA], q_pts[iB]
                    vC, vD = q_pts[iC], q_pts[iD]

                    # Orientación positiva de C respecto a la línea AB
                    if (vB[0] - vA[0]) * (vC[1] - vA[1]) - (vB[1] - vA[1]) * (vC[0] - vA[0]) < 0:
                        vA, vB = vB, vA
                        iA, iB = iB, iA

                    dx = vB[0] - vA[0]
                    dy = vB[1] - vA[1]
                    L = np.hypot(dx, dy)
                    if L < 5.0:
                        continue

                    cos_th = dx / L
                    sin_th = dy / L

                    rxC, ryC = (vC[0] - vA[0]) / L, (vC[1] - vA[1]) / L
                    xC = rxC * cos_th + ryC * sin_th
                    yC = -rxC * sin_th + ryC * cos_th

                    rxD, ryD = (vD[0] - vA[0]) / L, (vD[1] - vA[1]) / L
                    xD = rxD * cos_th + ryD * sin_th
                    yD = -rxD * sin_th + ryD * cos_th

                    if xC > xD or (abs(xC - xD) < 1e-4 and yC > yD):
                        xC, xD = xD, xC
                        yC, yD = yD, yC
                        iC, iD = iD, iC

                    feats.append([xC, yC, xD, yD])
                    quads.append((q_indices[iA], q_indices[iB], q_indices[iC], q_indices[iD]))

    return np.array(feats, dtype=np.float32), quads


def match_star_asterisms(
    feats1: np.ndarray,
    quads1: List[Tuple[int, int, int, int]],
    pts1: np.ndarray,
    feats2: np.ndarray,
    quads2: List[Tuple[int, int, int, int]],
    pts2: np.ndarray,
    tolerance: float = 0.020,
    min_votes: int = 2
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Empareja asterismos de 4 estrellas entre dos paneles mediante árbol k-d
    y verifica correspondencias con RANSAC rígido/afín.
    Retorna (pts1_matched, pts2_matched, num_inliers).
    """
    if len(feats1) == 0 or len(feats2) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0

    tree2 = cKDTree(feats2)
    votes = {}
    for i1, feat in enumerate(feats1):
        for i2 in tree2.query_ball_point(feat, r=tolerance):
            for p_a, p_b in zip(quads1[i1], quads2[i2]):
                votes[(p_a, p_b)] = votes.get((p_a, p_b), 0) + 1

    best_1, best_2 = {}, {}
    for (s1, s2), c in votes.items():
        if c >= min_votes:
            if s1 not in best_1 or c > best_1[s1][1]:
                best_1[s1] = (s2, c)
            if s2 not in best_2 or c > best_2[s2][1]:
                best_2[s2] = (s1, c)

    mutual = [(s1, s2) for s1, (s2, _) in best_1.items() if s2 in best_2 and best_2[s2][0] == s1]
    if len(mutual) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0

    pts1_m = pts1[[m[0] for m in mutual]]
    pts2_m = pts2[[m[1] for m in mutual]]

    M, inliers = cv2.estimateAffinePartial2D(pts2_m, pts1_m, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if inliers is None or np.sum(inliers) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0

    inl_mask = inliers.ravel() > 0
    return pts1_m[inl_mask], pts2_m[inl_mask], int(np.sum(inl_mask))


def estimate_camera_rotation_from_stars(
    pts1: np.ndarray,
    pts2: np.ndarray,
    w: int,
    h: int,
    focal_px: float
) -> Tuple[Optional[np.ndarray], int]:
    """
    Estima la matriz de rotación 3D en SO(3) entre dos cámaras utilizando los rayos unitarios
    de las estrellas emparejadas sobre la esfera celeste (descomposición SVD de Kabsch-Umeyama).
    Resuelve: ray_1 ≈ R_2_to_1 @ ray_2.
    Retorna (R_2_to_1, num_inliers).
    """
    if len(pts1) < 4 or len(pts2) < 4:
        return None, 0

    r1 = np.column_stack([pts1[:, 0] - w / 2.0, pts1[:, 1] - h / 2.0, np.full(len(pts1), focal_px)])
    r1 /= np.linalg.norm(r1, axis=1, keepdims=True)

    r2 = np.column_stack([pts2[:, 0] - w / 2.0, pts2[:, 1] - h / 2.0, np.full(len(pts2), focal_px)])
    r2 /= np.linalg.norm(r2, axis=1, keepdims=True)

    H = r1.T @ r2
    U, S, Vt = np.linalg.svd(H)
    R_est = U @ Vt
    if np.linalg.det(R_est) < 0:
        U[:, -1] *= -1
        R_est = U @ Vt

    dots = np.sum(r1 * (r2 @ R_est.T), axis=1)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    inliers = angles < np.radians(1.2)
    n_inliers = int(np.sum(inliers))

    if n_inliers < 4:
        return None, 0

    if n_inliers < len(pts1):
        H_clean = r1[inliers].T @ r2[inliers]
        U, S, Vt = np.linalg.svd(H_clean)
        R_est = U @ Vt
        if np.linalg.det(R_est) < 0:
            U[:, -1] *= -1
            R_est = U @ Vt

    return R_est, n_inliers


def _find_reachable_nodes(start_node: int, edges: Dict[Tuple[int, int], Any]) -> set:
    """Retorna el conjunto de nodos alcanzables desde start_node en el grafo de aristas."""
    visited = {start_node}
    queue = [start_node]
    while queue:
        curr = queue.pop(0)
        for (s, d) in edges.keys():
            if s == curr and d not in visited:
                visited.add(d)
                queue.append(d)
    return visited


def _level_and_center_rotations(rmats: List[np.ndarray]) -> List[np.ndarray]:
    """
    Alinea el eje medio de rotación del barrido con el eje vertical (Y)
    y centra el panel central en yaw = 0.
    """
    n = len(rmats)
    if n < 2:
        return [r.astype(np.float32) for r in rmats]
    try:
        axes = []
        for i in range(n - 1):
            opt_curr = rmats[i].T @ np.array([0, 0, 1], dtype=np.float32)
            opt_next = rmats[i + 1].T @ np.array([0, 0, 1], dtype=np.float32)
            axis_w = np.cross(opt_curr, opt_next)
            norm_ax = np.linalg.norm(axis_w)
            if norm_ax > 1e-4:
                axes.append(axis_w / norm_ax)
        if axes:
            tripod_axis = np.median(axes, axis=0)
            tripod_axis /= np.linalg.norm(tripod_axis)
            target_up = np.array([0, -1, 0], dtype=np.float32)
            if np.dot(tripod_axis, target_up) < 0:
                target_up = -target_up
            v_rot = np.cross(tripod_axis, target_up)
            cos_a = np.dot(tripod_axis, target_up)
            sin_a = np.linalg.norm(v_rot)
            if sin_a > 1e-4:
                k = v_rot / sin_a
                K_mat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
                R_align = np.eye(3, dtype=np.float32) + sin_a * K_mat + (1.0 - cos_a) * (K_mat @ K_mat)
            else:
                R_align = np.eye(3, dtype=np.float32)
            R_new = [r @ R_align.T for r in rmats]

            mid_idx = n // 2
            opt_mid = R_new[mid_idx].T @ np.array([0, 0, 1], dtype=np.float32)
            yaw_mid = np.arctan2(opt_mid[0], opt_mid[2])
            R_yaw_center = np.array([
                [np.cos(-yaw_mid), 0, np.sin(-yaw_mid)],
                [0, 1, 0],
                [-np.sin(-yaw_mid), 0, np.cos(-yaw_mid)]
            ], dtype=np.float32)
            return [(r @ R_yaw_center.T).astype(np.float32) for r in R_new]
    except Exception:
        pass
    return [r.astype(np.float32) for r in rmats]


def apply_arch_curvature_deformation(image: np.ndarray, curvature: float = 0.0) -> np.ndarray:
    """
    Aplica una deformación parabólica vertical suave (Curvature / Bow warp)
    para nivelar el horizonte y arquear la Vía Láctea, idéntico a las herramientas de
    curvatura de Autopano Giga y Photoshop.
    Valores negativos (<0) corrigen el efecto cuenco elevando el centro,
    lo que aplana los árboles y acentúa el arco estelar hacia el cenit.
    """
    if abs(curvature) < 1e-4 or image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    nx = (xs - w / 2.0) / (w / 2.0)
    dy = curvature * h * (1.0 - nx ** 2)
    map_x = xs
    map_y = ys - dy
    return cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _build_overlap_graph(
    features_list: List[Tuple[List[cv2.KeyPoint], np.ndarray]],
    min_inliers: int = 15,
    ratio_thresh: float = 0.75,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None,
    star_edges_affine: Optional[Dict[Tuple[int, int], Tuple[np.ndarray, int]]] = None
) -> Dict[Tuple[int, int], Tuple[np.ndarray, int]]:
    """
    Construye el grafo 2D de solapamiento entre todas las parejas de imágenes.
    Soporta de forma nativa panorámicas de 1 fila (1D) y matrices multifila (multi-row 2D).
    Retorna un diccionario de aristas: (src_idx, dst_idx) -> (H_src_to_dst, num_inliers).
    """
    n = len(features_list)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    edges = dict(star_edges_affine) if star_edges_affine else {}

    total_pairs = (n * (n - 1)) // 2
    pair_count = 0

    for i in range(n):
        kps_i, desc_i = features_list[i]
        if desc_i is None or len(desc_i) < 10:
            continue

        for j in range(i + 1, n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")

            pair_count += 1
            if progress_callback and pair_count % max(1, total_pairs // 10) == 0:
                prog = int(15 + 35 * (pair_count / max(1, total_pairs)))
                progress_callback(prog, f"Emparejando paneles ({pair_count}/{total_pairs})...")

            if (j, i) in edges and (i, j) in edges:
                continue

            kps_j, desc_j = features_list[j]
            if desc_j is None or len(desc_j) < 10:
                continue

            # Emparejamiento Knn con test de ratio de Lowe
            matches = bf.knnMatch(desc_i, desc_j, k=2)
            good = [m for m, n_match in matches if m.distance < ratio_thresh * n_match.distance]

            if len(good) < min_inliers:
                continue

            pts_i = np.float32([kps_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_j = np.float32([kps_j[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Estimación robusta con modelo afín / rígido para evitar deformaciones proyectivas singulares
            M, inliers_mask = cv2.estimateAffinePartial2D(pts_j, pts_i, method=cv2.RANSAC, ransacReprojThreshold=8.0)

            if M is not None and inliers_mask is not None:
                n_inliers = int(np.sum(inliers_mask))
                if n_inliers >= min_inliers:
                    H_j_to_i = np.vstack([M, [0, 0, 1]])
                    try:
                        H_i_to_j = np.linalg.inv(H_j_to_i)
                        edges[(j, i)] = (H_j_to_i, n_inliers)
                        edges[(i, j)] = (H_i_to_j, n_inliers)
                    except np.linalg.LinAlgError:
                        pass

    return edges


def _compute_global_homographies(
    n_images: int,
    edges: Dict[Tuple[int, int], Tuple[np.ndarray, int]],
    anchor_idx: Optional[int] = None
) -> Tuple[Dict[int, np.ndarray], int]:
    """
    Calcula las matrices de transformación global de cada imagen hacia el plano del lienzo
    mediante recorrido en anchura (BFS) sobre el grafo de máxima conectividad.
    """
    if anchor_idx is None:
        # Seleccionar automáticamente el nodo que abarca el mayor número de paneles
        best_anchor = n_images // 2
        best_reach = 0
        for cand in range(n_images):
            visited_cand = {cand}
            q_cand = [cand]
            while q_cand:
                c = q_cand.pop(0)
                for (s, d) in edges.keys():
                    if s == c and d not in visited_cand:
                        visited_cand.add(d)
                        q_cand.append(d)
            if len(visited_cand) > best_reach:
                best_reach = len(visited_cand)
                best_anchor = cand
                if best_reach == n_images:
                    break
        anchor_idx = best_anchor

    # Matriz identidad para el panel de referencia
    global_H = {anchor_idx: np.eye(3, dtype=np.float64)}
    visited = {anchor_idx}
    queue = [anchor_idx]

    while queue:
        curr = queue.pop(0)
        curr_H = global_H[curr]

        # Encontrar vecinos conectados ordenados por mayor número de inliers
        neighbors = []
        for (src, dst), (H_src_to_dst, inliers) in edges.items():
            if src == curr and dst not in visited:
                neighbors.append((dst, H_src_to_dst, inliers))

        neighbors.sort(key=lambda x: x[2], reverse=True)

        for nxt, H_curr_to_nxt, _ in neighbors:
            # H_nxt_to_anchor = H_curr_to_anchor * H_nxt_to_curr
            # Donde H_nxt_to_curr = inv(H_curr_to_nxt)
            try:
                H_nxt_to_curr = np.linalg.inv(H_curr_to_nxt)
                global_H[nxt] = curr_H @ H_nxt_to_curr
                visited.add(nxt)
                queue.append(nxt)
            except np.linalg.LinAlgError:
                continue

    return global_H, anchor_idx


def blend_multiband_pyramid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    num_bands: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fusión de múltiples imágenes mediante Pirámide Laplaciana Multibanda.
    Separa las frecuencias bajas (iluminación, degradados y cielo nocturno continuo)
    de las altas frecuencias (estrellas, ramas y siluetas del terreno nítidas).
    """
    n = len(images)
    if n == 0:
        raise ValueError("No hay imágenes para fusionar.")
    if n == 1:
        return images[0], masks[0]

    h, w, c = images[0].shape

    # 1. Mapas de pesos suavizados basados en Distance Transform
    weight_maps = []
    for m in masks:
        bin_m = (m > 127).astype(np.uint8)
        dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
        # Normalizar para evitar divisiones inestables
        max_d = np.max(dist)
        if max_d > 0:
            dist /= max_d
        weight_maps.append(dist)

    total_weight = np.sum(np.stack(weight_maps, axis=-1), axis=-1, keepdims=True)
    total_weight = np.where(total_weight > 1e-6, total_weight, 1.0)

    norm_weights = [(w_m[..., np.newaxis] / total_weight).astype(np.float32) for w_m in weight_maps]

    # 2. Pirámide Gaussiana para los pesos y Laplaciana para las imágenes
    lap_pyrs = []
    weight_pyrs = []

    for i in range(n):
        # Construir pirámide gaussiana de pesos
        g_w = norm_weights[i]
        pyr_w = [g_w]
        for _ in range(num_bands - 1):
            g_w = cv2.pyrDown(g_w)
            if len(g_w.shape) == 2:
                g_w = g_w[..., np.newaxis]
            pyr_w.append(g_w)
        weight_pyrs.append(pyr_w)

        # Construir pirámide laplaciana de la imagen
        g_img = images[i].astype(np.float32)
        pyr_g_img = [g_img]
        for _ in range(num_bands - 1):
            g_img = cv2.pyrDown(g_img)
            pyr_g_img.append(g_img)

        pyr_lap = []
        for level in range(num_bands - 1):
            size = (pyr_g_img[level].shape[1], pyr_g_img[level].shape[0])
            expanded = cv2.pyrUp(pyr_g_img[level + 1], dstsize=size)
            pyr_lap.append(pyr_g_img[level] - expanded)
        pyr_lap.append(pyr_g_img[-1])
        lap_pyrs.append(pyr_lap)

    # 3. Fusión en cada nivel de la pirámide
    blended_pyr = []
    for level in range(num_bands):
        level_sum = np.zeros_like(lap_pyrs[0][level])
        for i in range(n):
            level_sum += lap_pyrs[i][level] * weight_pyrs[i][level]
        blended_pyr.append(level_sum)

    # 4. Reconstrucción ascendente desde el nivel base
    reconstructed = blended_pyr[-1]
    for level in range(num_bands - 2, -1, -1):
        size = (blended_pyr[level].shape[1], blended_pyr[level].shape[0])
        reconstructed = cv2.pyrUp(reconstructed, dstsize=size) + blended_pyr[level]

    final_rgb = np.clip(reconstructed, 0.0, 1.0)
    # Máscara total unificada
    union_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        union_mask = cv2.bitwise_or(union_mask, m)

    return final_rgb, union_mask


def stitch_nightscape_panorama(
    panels_paths: List[str],
    settings: Optional[PanoramaSettings] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None,
    panels_metadata: Optional[List[PanoramaPanelMetadata]] = None
) -> PanoramaResult:
    """
    Canalización completa de cosido panorámico nocturno (Nightscape Panorama Stitcher).
    
    Pasos:
    1. Ingesta de paneles y extracción de metadatos EXIF.
    2. Carga de vistas previas ligeras con pre-realce astro (CLAHE) para calibración de cámaras.
    3. Calibración global de rotación 3D (SO(3)) y longitud focal mediante ajuste de haces (Bundle Adjustment).
    4. Corrección horizontal de horizonte (Wave Correction) para nivelar la panorámica.
    5. Fallback automático a grafo 2D de correspondencias afines (SIFT + RANSAC BFS) si la calibración global falla.
    6. Proyección cartográfica de alta fidelidad: Esférica, Cilíndrica o Plana (PyRotationWarper).
    7. Fusión en streaming continuo de memoria O(1) con ponderación suave por mapa de distancias potenciado.
    8. Cálculo de Rectángulo Máximo Inscrito (Auto-recorte).
    """
    t0 = time.time()
    cv2.ocl.setUseOpenCL(False)  # Evitar desbordamiento de memoria GPU en arrays panorámicos
    if settings is None:
        settings = PanoramaSettings()

    n = len(panels_paths)
    if n < 2:
        raise ValueError("Se requieren al menos 2 imágenes para componer una panorámica.")

    # -------------------------------------------------------------
    # 1. Extracción de metadatos de los paneles
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(5, "Extrayendo metadatos EXIF de los paneles...")

    metas = panels_metadata if (panels_metadata is not None and len(panels_metadata) == n) else [extract_panel_metadata(p, idx) for idx, p in enumerate(panels_paths)]
    full_w = metas[0].width
    full_h = metas[0].height

    # -------------------------------------------------------------
    # 2. Carga de vistas previas ligeras para calibración de cámara
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(10, f"Generando vistas previas optimizadas para {n} paneles...")

    imgs_small = []
    for i, path in enumerate(panels_paths):
        if abort_flag and abort_flag():
            raise InterruptedError("Operación cancelada por el usuario.")

        ext = os.path.splitext(path)[1].lower()
        if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw']:
            try:
                import rawpy
                with rawpy.imread(path) as r:
                    rgb_half = r.postprocess(half_size=True, use_camera_wb=True)
                small = cv2.resize(rgb_half, (rgb_half.shape[1] // 2, rgb_half.shape[0] // 2))
                del rgb_half
            except Exception:
                img_f = load_image_as_float32(path)
                u8 = (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)
                small = cv2.resize(u8, (u8.shape[1] // 4, u8.shape[0] // 4))
                del img_f, u8
        else:
            img_f = load_image_as_float32(path)
            u8 = (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)
            small = cv2.resize(u8, (u8.shape[1] // 4, u8.shape[0] // 4))
            del img_f, u8

        # Pre-realce CLAHE para cielo estrellado y detalles en sombras
        if settings.astro_mode:
            lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            small = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

        imgs_small.append(small)
        gc.collect()

    small_h, small_w = imgs_small[0].shape[:2]
    scale_to_full = full_w / float(small_w)

    # -------------------------------------------------------------
    # 2.5 Segmentación de Horizonte IA (Desacoplamiento Cielo / Suelo)
    # -------------------------------------------------------------
    sky_masks_sm = None
    use_mask_opt = getattr(settings, 'use_auto_mask', True)
    if settings.astro_mode or use_mask_opt:
        try:
            if progress_callback:
                progress_callback(18, "Segmentando horizonte y bóveda celeste con IA...")
            from core.ai_horizon import auto_detect_sky_mask
            sky_masks_sm = []
            for sm in imgs_small:
                m_sky, _ = auto_detect_sky_mask(sm)
                sky_masks_sm.append(m_sky)
        except Exception:
            sky_masks_sm = None

    # -------------------------------------------------------------
    # 3. Estimación de Rotación 3D Global (Bundle Adjustment / Asterismos Estelares)
    # -------------------------------------------------------------
    cameras = None
    rmats_corr = None
    connected_indices = list(range(n))

    # Estimación de distancia focal de trabajo y completa
    # Nota: En sensores 35mm (Full Frame), la cota máxima del sensor siempre corresponde a 36.0 mm
    sensor_max_px = float(max(full_w, full_h))
    if settings.focal_length_override is not None and settings.focal_length_override > 1.0:
        focal_full = float(settings.focal_length_override) * (sensor_max_px / 36.0)
    elif metas[0].focal_length_mm is not None and metas[0].focal_length_mm > 1.0:
        focal_full = float(metas[0].focal_length_mm) * (sensor_max_px / 36.0)
    else:
        focal_full = float(sensor_max_px * 1.0)
    focal_work = focal_full / scale_to_full

    star_edges_affine = {}
    star_edges_rot = {}

    # 3.1 Calibración Estelar Pura guiada por Máscara IA (Evita interferencia de árboles y rotación terrestre)
    if settings.projection != "plane":
        if sky_masks_sm is not None:
            try:
                if progress_callback:
                    progress_callback(25, "Calibrando cámaras sobre estrellas (horizonte protegido)...")
                sift_sky = cv2.SIFT_create(contrastThreshold=0.015, edgeThreshold=15)
                feats_sky = [
                    cv2.detail.computeImageFeatures2(sift_sky, im, (m > 0.5).astype(np.uint8) * 255)
                    for im, m in zip(imgs_small, sky_masks_sm)
                ]
                matcher_sky = cv2.detail_BestOf2NearestMatcher(False, settings.confidence_thresh)
                pairwise_sky = matcher_sky.apply2(feats_sky)
                estimator_sky = cv2.detail_HomographyBasedEstimator()
                cams_init = []
                ok_est, cams_init = estimator_sky.apply(feats_sky, pairwise_sky, cams_init)
                if ok_est and len(cams_init) == n:
                    for c in cams_init:
                        c.R = c.R.astype(np.float32)
                    adjuster_sky = cv2.detail_BundleAdjusterRay()
                    adjuster_sky.setConfThresh(0.35)
                    ok_ba, cams_ba = adjuster_sky.apply(feats_sky, pairwise_sky, cams_init)
                    if ok_ba and len(cams_ba) == n:
                        cameras = list(cams_ba)
                        scale_work_to_full = full_w / (cameras[0].ppx * 2.0)
                        focal_full = float(np.median([c.focal * scale_work_to_full for c in cameras]))
                        rmats_corr = _level_and_center_rotations([c.R for c in cameras])
                        if settings.wave_correction:
                            try:
                                rmats_corr = cv2.detail.waveCorrect([r.astype(np.float32) for r in rmats_corr], cv2.detail.WAVE_CORRECT_HORIZ)
                            except Exception:
                                pass
                        if progress_callback:
                            progress_callback(35, "Alineación estelar completada con ajuste de haces de rayos.")
            except Exception:
                cameras = None
                rmats_corr = None

        # Fallback a Stitcher estándar conjunto si la máscara o calibración estelar no converge
        if cameras is None or rmats_corr is None:
            if progress_callback:
                progress_callback(25, "Estimando geometría 3D con ajuste de haces global...")
            stitch_mode = cv2.Stitcher_PANORAMA
            stitcher = cv2.Stitcher_create(stitch_mode)
            status = stitcher.estimateTransform(imgs_small)

            if status == 0 and len(stitcher.cameras()) == n:
                cameras = list(stitcher.cameras())
                scale_work_to_full = full_w / (cameras[0].ppx * 2.0)
                focal_full = float(np.median([c.focal * scale_work_to_full for c in cameras]))
                rmats_corr = _level_and_center_rotations([c.R for c in cameras])
                if settings.wave_correction:
                    try:
                        rmats_corr = cv2.detail.waveCorrect([r.astype(np.float32) for r in rmats_corr], cv2.detail.WAVE_CORRECT_HORIZ)
                    except Exception:
                        pass
                if progress_callback:
                    progress_callback(35, "Alineación 3D global completada con ajuste de haces óptimo.")

    # 3.2 Si el Bundle Adjuster no converge (cielo oscuro puro sin horizonte), aplicar Identificación Estelar
    if (cameras is None or rmats_corr is None) and settings.star_matching:
        if progress_callback:
            progress_callback(25, "Identificando estrellas brillantes y asterismos estelares...")

        stars_per_panel = [extract_bright_stars(sm, max_stars=120) for sm in imgs_small]
        quads_per_panel = [build_star_asterisms(pts, k_neighbors=7) for pts in stars_per_panel]
        min_star_inliers = max(4, min(settings.min_inliers, 5))

        for i in range(n):
            for j in range(i + 1, n):
                if abort_flag and abort_flag():
                    raise InterruptedError("Operación cancelada por el usuario.")

                if len(stars_per_panel[i]) < 4 or len(stars_per_panel[j]) < 4:
                    continue

                pts_i_m, pts_j_m, n_inl = match_star_asterisms(
                    quads_per_panel[i][0], quads_per_panel[i][1], stars_per_panel[i],
                    quads_per_panel[j][0], quads_per_panel[j][1], stars_per_panel[j],
                    tolerance=0.038, min_votes=2
                )

                if n_inl >= min_star_inliers:
                    # Guardar homografía afín escalada a tamaño completo para fallback 2D
                    pts_i_full = pts_i_m * scale_to_full
                    pts_j_full = pts_j_m * scale_to_full
                    M_aff, aff_inl = cv2.estimateAffinePartial2D(pts_j_full, pts_i_full, method=cv2.RANSAC, ransacReprojThreshold=8.0)
                    if M_aff is not None:
                        H_j_to_i = np.vstack([M_aff, [0, 0, 1]])
                        try:
                            H_i_to_j = np.linalg.inv(H_j_to_i)
                            star_edges_affine[(j, i)] = (H_j_to_i, n_inl)
                            star_edges_affine[(i, j)] = (H_i_to_j, n_inl)
                        except np.linalg.LinAlgError:
                            pass

                    # Estimar rotación 3D sobre la esfera celeste con Kabsch SVD
                    R_j_to_i, n_rot_inl = estimate_camera_rotation_from_stars(
                        pts_i_m, pts_j_m, small_w, small_h, focal_work
                    )
                    if R_j_to_i is not None and n_rot_inl >= min_star_inliers:
                        star_edges_rot[(j, i)] = (R_j_to_i, n_rot_inl)
                        star_edges_rot[(i, j)] = (R_j_to_i.T, n_rot_inl)

        # Si los asterismos conectan todos los paneles, resolver rotación global exacta
        if star_edges_rot:
            best_anchor = n // 2
            best_reach = 0
            for cand in range(n):
                reach = len(_find_reachable_nodes(cand, star_edges_rot))
                if reach > best_reach:
                    best_reach = reach
                    best_anchor = cand

            if best_reach == n and settings.projection != "plane":
                R_global = {best_anchor: np.eye(3, dtype=np.float64)}
                queue = [best_anchor]
                visited = {best_anchor}
                while queue:
                    curr = queue.pop(0)
                    for j in range(n):
                        if (j, curr) in star_edges_rot and j not in visited:
                            visited.add(j)
                            R_global[j] = star_edges_rot[(j, curr)][0] @ R_global[curr]
                            queue.append(j)

                rmats_corr = _level_and_center_rotations([R_global[i] for i in range(n)])
                cameras = [True] * n
                if progress_callback:
                    progress_callback(35, f"Alineación por asterismos estelares completada ({len(star_edges_rot) // 2} solapamientos).")

    # Fallback a grafo 2D de homografías si la calibración 3D no converge o es proyección plana
    if cameras is None or rmats_corr is None:
        if progress_callback:
            progress_callback(30, "Alineando paneles mediante grafo 2D de correspondencias...")
        features_list = []
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)
        for i, sm in enumerate(imgs_small):
            gray = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
            kps, des = sift.detectAndCompute(gray, None)
            kps_full = [
                cv2.KeyPoint(kp.pt[0] * scale_to_full, kp.pt[1] * scale_to_full, kp.size * scale_to_full,
                             kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in kps
            ]
            features_list.append((kps_full, des))

        edges = _build_overlap_graph(
            features_list,
            min_inliers=settings.min_inliers,
            ratio_thresh=0.82,
            star_edges_affine=star_edges_affine
        )
        if not edges:
            raise ValueError("No se encontraron solapamientos suficientes entre los paneles.")
        global_H, anchor_idx = _compute_global_homographies(n, edges)
        connected_indices = sorted(list(global_H.keys()))

    # -------------------------------------------------------------
    # 4. Configuración del Lienzo y Proyección Geométrica
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(50, "Calculando dimensiones del lienzo panorámico...")

    proj_type = settings.projection
    if proj_type not in ("spherical", "cylindrical", "plane"):
        proj_type = "spherical"

    MAX_CANVAS_PIXELS = 65_000_000  # 65 Megapíxeles para estabilidad garantizada de RAM en cualquier PC

    if cameras is not None and rmats_corr is not None:
        warper = cv2.PyRotationWarper(proj_type, focal_full)
        corners = []
        sizes = []
        for i in range(n):
            K = np.array([
                [focal_full, 0, full_w / 2.0],
                [0, focal_full, full_h / 2.0],
                [0, 0, 1]
            ], dtype=np.float32)
            roi = warper.warpRoi((full_w, full_h), K, rmats_corr[i].astype(np.float32))
            corners.append((roi[0], roi[1]))
            sizes.append((roi[2], roi[3]))

        min_x = min(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_x = max(c[0] + s[0] for c, s in zip(corners, sizes))
        max_y = max(c[1] + s[1] for c, s in zip(corners, sizes))

        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)

        scale_canvas = 1.0
        max_dim = max(canvas_w, canvas_h)
        if canvas_w * canvas_h > MAX_CANVAS_PIXELS or max_dim > 24000:
            scale_mp = np.sqrt(MAX_CANVAS_PIXELS / (canvas_w * canvas_h)) if canvas_w * canvas_h > MAX_CANVAS_PIXELS else 1.0
            scale_dim = 24000.0 / max_dim if max_dim > 24000 else 1.0
            scale_canvas = min(scale_mp, scale_dim)
            focal_full *= scale_canvas
            warper = cv2.PyRotationWarper(proj_type, focal_full)
            corners = []
            sizes = []
            cur_w = int(full_w * scale_canvas)
            cur_h = int(full_h * scale_canvas)
            for i in range(n):
                K = np.array([
                    [focal_full, 0, cur_w / 2.0],
                    [0, focal_full, cur_h / 2.0],
                    [0, 0, 1]
                ], dtype=np.float32)
                roi = warper.warpRoi((cur_w, cur_h), K, rmats_corr[i].astype(np.float32))
                corners.append((roi[0], roi[1]))
                sizes.append((roi[2], roi[3]))

            min_x = min(c[0] for c in corners)
            min_y = min(c[1] for c in corners)
            max_x = max(c[0] + s[0] for c, s in zip(corners, sizes))
            max_y = max(c[1] + s[1] for c, s in zip(corners, sizes))
            canvas_w = int(max_x - min_x)
            canvas_h = int(max_y - min_y)

        cur_w = int(full_w * scale_canvas)
        cur_h = int(full_h * scale_canvas)
        K = np.array([
            [focal_full, 0, cur_w / 2.0],
            [0, focal_full, cur_h / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)

        # -------------------------------------------------------------
        # 5. Compensación de Ganancia Fotométrica (Gain Compensator)
        # -------------------------------------------------------------
        # Nivelación de exposición y balance entre paneles contiguos para eliminar
        # por completo las marcas de luz o saltos de brillo verticales en el cielo.
        gain_comp = cv2.detail_GainCompensator() if settings.exposure_comp else None
        if gain_comp is not None and 'imgs_small' in locals():
            try:
                gc_scale = 0.20
                gc_w, gc_h = max(10, int(cur_w * gc_scale)), max(10, int(cur_h * gc_scale))
                gc_focal = focal_full * gc_scale
                gc_warper = cv2.PyRotationWarper(proj_type, gc_focal)
                K_gc = np.array([
                    [gc_focal, 0, gc_w / 2.0],
                    [0, gc_focal, gc_h / 2.0],
                    [0, 0, 1]
                ], dtype=np.float32)
                gc_corners, gc_imgs, gc_masks = [], [], []
                for idx_c in range(n):
                    im_sm = cv2.resize(imgs_small[idx_c], (gc_w, gc_h))
                    c_gc, w_im_gc = gc_warper.warp(im_sm, K_gc, rmats_corr[idx_c].astype(np.float32), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                    w_m_gc = (np.max(w_im_gc, axis=2) > 1).astype(np.uint8) * 255
                    gc_corners.append(c_gc)
                    gc_imgs.append(w_im_gc)
                    gc_masks.append(w_m_gc)
                gain_comp.feed(gc_corners, gc_imgs, gc_masks)
                del gc_corners, gc_imgs, gc_masks
            except Exception:
                gain_comp = None

        panel_gains = [1.0] * n
        if gain_comp is not None:
            try:
                m_gains = gain_comp.getMatGains()
                if len(m_gains) == n:
                    panel_gains = [float(g[0, 0]) for g in m_gains]
            except Exception:
                panel_gains = [1.0] * n

        if 'imgs_small' in locals():
            del imgs_small
            gc.collect()

        if progress_callback:
            progress_callback(60, f"Calculando partición óptima de costuras ({canvas_w}x{canvas_h}, {canvas_w*canvas_h/1e6:.1f} MP)...")

        # -------------------------------------------------------------
        # 6. Partición Voronoi y Fusión Multibanda Calibrada
        # -------------------------------------------------------------
        best_dist = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        best_idx = np.full((canvas_h, canvas_w), -1, dtype=np.int16)
        panel_mask = np.ones((cur_h, cur_w), dtype=np.uint8) * 255

        for i in range(n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")
            c_m, w_mask = warper.warp(panel_mask, K, rmats_corr[i].astype(np.float32), cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            dist = cv2.distanceTransform(w_mask, cv2.DIST_L2, 3)
            gx = c_m[0] - min_x
            gy = c_m[1] - min_y
            h_w, w_w = dist.shape
            x1, x2 = max(0, gx), min(canvas_w, gx + w_w)
            y1, y2 = max(0, gy), min(canvas_h, gy + h_w)
            ix1, ix2 = x1 - gx, x2 - gx
            iy1, iy2 = y1 - gy, y2 - gy

            sub_d = dist[iy1:iy2, ix1:ix2]
            sub_best = best_dist[y1:y2, x1:x2]
            better = sub_d > sub_best
            sub_best[better] = sub_d[better]
            best_dist[y1:y2, x1:x2] = sub_best
            best_idx[y1:y2, x1:x2][better] = i
            del c_m, w_mask, dist

        # Rellenar cualquier píxel no asignado en fronteras con el vecino más cercano
        unassigned = (best_idx < 0) & (best_dist > 0)
        if np.any(unassigned):
            _, nearest_lbl = cv2.distanceTransformWithLabels((best_idx >= 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

        # Fusión Multibanda con Desacoplamiento Cielo / Suelo (Anti-Ghosting de horizonte)
        num_bands = max(1, min(6, settings.blend_bands if settings.blend_bands > 0 else 5))
        blender_sky = cv2.detail_MultiBandBlender(0, num_bands)
        blender_sky.prepare((0, 0, canvas_w, canvas_h))

        has_ground = (sky_masks_sm is not None)
        if has_ground:
            ground_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
            sky_mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for i in range(n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")

            if progress_callback:
                prog = int(65 + 25 * (i / n))
                progress_callback(prog, f"Proyectando y fusionando panel {i + 1}/{n}: {os.path.basename(panels_paths[i])}...")

            img_f = load_image_as_float32(panels_paths[i])
            if scale_canvas != 1.0:
                img_f = cv2.resize(img_f, (cur_w, cur_h), interpolation=cv2.INTER_AREA)

            ih, iw = img_f.shape[:2]

            # Compensación adaptativa de caída óptica / viñeteo calibrada
            v_amt = getattr(settings, 'vignette_amount', 0.5) if settings.exposure_comp else 0.0
            if v_amt > 0.01:
                vig_factor = float(v_amt) * 1.0
                Y_g, X_g = np.ogrid[:ih, :iw]
                cx_img, cy_img = iw / 2.0, ih / 2.0
                r_sq = ((X_g - cx_img)**2 + (Y_g - cy_img)**2) / float(cx_img**2 + cy_img**2)
                vignette_corr = (1.0 + vig_factor * r_sq + 0.05 * (r_sq**2)).astype(np.float32)[..., np.newaxis]
                img_f = np.clip(img_f * vignette_corr, 0.0, 1.0)

            corner, w_img = warper.warp(img_f, K, rmats_corr[i].astype(np.float32), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            del img_f

            if abs(panel_gains[i] - 1.0) > 1e-4:
                w_img = np.clip(w_img * panel_gains[i], 0.0, 1.0)

            gx = corner[0] - min_x
            gy = corner[1] - min_y
            h_w, w_w = w_img.shape[:2]

            x1, x2 = max(0, gx), min(canvas_w, gx + w_w)
            y1, y2 = max(0, gy), min(canvas_h, gy + h_w)
            ix1, ix2 = x1 - gx, x2 - gx
            iy1, iy2 = y1 - gy, y2 - gy

            local_seam = np.zeros((h_w, w_w), dtype=np.uint8)
            local_part = (best_idx[y1:y2, x1:x2] == i)
            local_seam[iy1:iy2, ix1:ix2] = np.where(local_part, 255, 0).astype(np.uint8)
            seam_dilated = cv2.dilate(local_seam, None)

            w_img_16s = (np.clip(w_img, 0.0, 1.0) * 32767.0).astype(np.int16)
            blender_sky.feed(w_img_16s, seam_dilated, (gx, gy))
            del w_img_16s, local_seam, seam_dilated

            if has_ground:
                m_sky_full = cv2.resize(sky_masks_sm[i], (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
                _, w_mask_sky = warper.warp(m_sky_full, K, rmats_corr[i].astype(np.float32), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                ground_canvas[y1:y2, x1:x2][local_part] = w_img[iy1:iy2, ix1:ix2][local_part]
                sky_mask_canvas[y1:y2, x1:x2][local_part] = w_mask_sky[iy1:iy2, ix1:ix2][local_part]
                del m_sky_full, w_mask_sky

            del w_img
            gc.collect()

        dst_sky, _ = blender_sky.blend(None, None)
        stitched_sky = np.clip(dst_sky.astype(np.float32) / 32767.0, 0.0, 1.0)
        del blender_sky, dst_sky

        total_mask = (best_idx >= 0).astype(np.uint8) * 255
        del best_dist, best_idx

        if has_ground and np.mean(sky_mask_canvas) <= 0.98:
            if progress_callback:
                progress_callback(92, "Componiendo cielo y suelo con protección de horizonte y follaje...")
            from core.stacking import compose_sky_ground_with_anti_ghosting
            blended_img = compose_sky_ground_with_anti_ghosting(stitched_sky, ground_canvas, sky_mask_canvas)
            del ground_canvas, sky_mask_canvas, stitched_sky
        else:
            blended_img = stitched_sky
            if has_ground:
                del ground_canvas, sky_mask_canvas

        blended_img[total_mask == 0] = 0.0
        gc.collect()

    else:
        if 'imgs_small' in locals():
            del imgs_small
            gc.collect()
        # Fallback Homography Projection
        all_corners_x = []
        all_corners_y = []
        for idx in connected_indices:
            H = global_H[idx]
            corners = np.array([
                [0, 0, 1],
                [full_w, 0, 1],
                [full_w, full_h, 1],
                [0, full_h, 1]
            ], dtype=np.float64).T
            transformed = H @ corners
            transformed /= transformed[2, :]
            all_corners_x.extend(transformed[0, :])
            all_corners_y.extend(transformed[1, :])

        min_x = np.floor(np.min(all_corners_x))
        max_x = np.ceil(np.max(all_corners_x))
        min_y = np.floor(np.min(all_corners_y))
        max_y = np.ceil(np.max(all_corners_y))

        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)

        scale_factor = 1.0
        if canvas_w * canvas_h > MAX_CANVAS_PIXELS:
            scale_factor = np.sqrt(MAX_CANVAS_PIXELS / (canvas_w * canvas_h))
            canvas_w = int(canvas_w * scale_factor)
            canvas_h = int(canvas_h * scale_factor)

        T = np.array([
            [scale_factor, 0, -min_x * scale_factor],
            [0, scale_factor, -min_y * scale_factor],
            [0, 0, 1]
        ], dtype=np.float64)

        accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        total_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        num_conn = len(connected_indices)
        for i, idx in enumerate(connected_indices):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")
            if progress_callback:
                prog = int(60 + 30 * (i / num_conn))
                progress_callback(prog, f"Proyectando y fusionando panel {i + 1}/{num_conn}...")

            img_f = load_image_as_float32(panels_paths[idx])
            h, w = img_f.shape[:2]
            mask_src = np.ones((h, w), dtype=np.uint8) * 255
            H_final = T @ global_H[idx]

            warped_img = cv2.warpPerspective(img_f, H_final, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
            warped_mask = cv2.warpPerspective(mask_src, H_final, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST)
            del img_f, mask_src

            dist = cv2.distanceTransform((warped_mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
            max_d = np.max(dist)
            if max_d > 0:
                dist /= max_d
            dist = np.power(dist, 1.8)
            w_map = dist[..., np.newaxis].astype(np.float32)

            accum += warped_img * w_map
            weight_sum += w_map
            total_mask = cv2.bitwise_or(total_mask, warped_mask)

            del warped_img, warped_mask, dist, w_map
            gc.collect()

        valid_norm = weight_sum > 1e-5
        accum[valid_norm.squeeze()] /= weight_sum[valid_norm.squeeze()]
        blended_img = np.clip(accum, 0.0, 1.0)
        del weight_sum
        gc.collect()

    # 5.5 Deformación de Arco / Curvatura de Horizonte (Bow deformation)
    if abs(settings.arch_curvature) > 1e-4:
        if progress_callback:
            progress_callback(94, "Aplicando deformación de arco y curvatura de horizonte...")
        blended_img = apply_arch_curvature_deformation(blended_img, settings.arch_curvature)
        if total_mask is not None:
            total_mask = apply_arch_curvature_deformation(total_mask, settings.arch_curvature)

    # -------------------------------------------------------------
    # 6. Recorte Automático de Rectángulo Máximo Inscrito
    # -------------------------------------------------------------
    cropped_img = None
    crop_rect = None

    if settings.auto_crop:
        if progress_callback:
            progress_callback(95, "Calculando rectángulo máximo inscrito (auto-recorte)...")

        # Erosión morfológica de seguridad para eliminar muescas de proyección esférica y peldaños
        ksize = max(15, int(min(canvas_w, canvas_h) * 0.004))
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        eroded_mask = cv2.erode(total_mask, kernel)

        # Fallback a total_mask si la máscara erosionada resultara excesivamente reducida
        mask_for_crop = eroded_mask if np.count_nonzero(eroded_mask) > 1000 else total_mask

        rx, ry, rw, rh = find_maximum_inscribed_rectangle(mask_for_crop)
        crop_rect = (rx, ry, rw, rh)
        cropped_img = blended_img[ry:ry+rh, rx:rx+rw].copy()

    total_time = time.time() - t0

    if progress_callback:
        progress_callback(100, f"Panorámica completada en {total_time:.1f}s.")

    return PanoramaResult(
        image=blended_img,
        cropped_image=cropped_img,
        crop_rect=crop_rect,
        mask=total_mask,
        panels_stitched=len(connected_indices),
        total_panels=n,
        resolution=(canvas_w, canvas_h),
        projection=settings.projection,
        processing_time_s=total_time,
        message=f"Se cosieron exitosamente {len(connected_indices)} de {n} paneles en {total_time:.1f} segundos."
    )


def save_panorama_session(filepath: str, data: Dict[str, Any]) -> bool:
    """Guarda la sesión del proyecto de panorámica (.panosess)."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_panorama_session(filepath: str) -> Optional[Dict[str, Any]]:
    """Carga un archivo de sesión de panorámica (.panosess)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def group_panel_paths(
    paths: List[str],
    shots_per_panel: int = 0,
    time_threshold_s: float = 15.0
) -> List[List[str]]:
    """
    Agrupa una lista de rutas de archivos en teselas/paneles para apilado previo.
    - Si shots_per_panel > 0: divide en lotes de tamaño exacto.
    - Si shots_per_panel == 0: agrupa automáticamente por saltos de tiempo EXIF.
    """
    if not paths:
        return []
    if shots_per_panel > 0:
        return [paths[i:i + shots_per_panel] for i in range(0, len(paths), shots_per_panel)]

    metas = [extract_panel_metadata(p, idx) for idx, p in enumerate(paths)]
    sorted_pairs = sorted(zip(paths, metas), key=lambda x: (x[1].timestamp, x[0]))

    groups: List[List[str]] = []
    current_group: List[str] = []

    for i, (path, meta) in enumerate(sorted_pairs):
        if not current_group:
            current_group.append(path)
        else:
            prev_meta = sorted_pairs[i - 1][1]
            dt = abs(meta.timestamp - prev_meta.timestamp)
            if dt == 0 and meta.time_str != "--:--:--" and prev_meta.time_str != "--:--:--":
                try:
                    def _to_sec(ts):
                        p = [int(x) for x in ts.split(':')]
                        return p[0] * 3600 + p[1] * 60 + p[2]
                    dt = abs(_to_sec(meta.time_str) - _to_sec(prev_meta.time_str))
                except Exception:
                    dt = 0
            if dt <= time_threshold_s:
                current_group.append(path)
            else:
                groups.append(current_group)
                current_group = [path]

    if current_group:
        groups.append(current_group)

    return groups


def stack_panel_group(
    paths: List[str],
    mount_mode: str = "tripod",
    use_mask: bool = False,
    abort_flag: Optional[Callable[[], bool]] = None
) -> np.ndarray:
    """
    Apila un conjunto de tomas de un mismo panel para generar una tesela maestra de alto SNR.
    - mount_mode: 'tripod' (trípode fijo) o 'tracker' (star tracker).
    - use_mask: Si True, separa cielo y suelo mediante auto-máscara de horizonte y anti-ghosting.
    Retorna un array float32 RGB [0.0, 1.0].
    """
    n = len(paths)
    if n == 0:
        raise ValueError("Lista de tomas vacía para el panel.")
    if n == 1:
        return load_image_as_float32(paths[0])

    ref_img = load_image_as_float32(paths[0])
    h, w = ref_img.shape[:2]

    if not use_mask:
        # Apilado directo ultrarrápido (Smart Mean) para ráfagas cortas
        accum = ref_img.astype(np.float32)
        for i in range(1, n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")
            nxt = load_image_as_float32(paths[i])
            accum += nxt
            del nxt
        res = accum / float(n)
        del accum
        return np.clip(res, 0.0, 1.0)

    # Con separación inteligente cielo / suelo
    from core.ai_horizon import auto_detect_sky_mask
    from core.stacking import compose_sky_ground_with_anti_ghosting

    sky_mask, _ = auto_detect_sky_mask(ref_img)
    if sky_mask.shape[:2] != (h, w):
        sky_mask = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    if mount_mode == "tracker":
        # En star tracker, el cielo está quieto respecto al sensor y el suelo se mueve.
        sky_accum = ref_img.astype(np.float32)
        for i in range(1, n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")
            nxt = load_image_as_float32(paths[i])
            sky_accum += nxt
            del nxt
        stacked_sky = np.clip(sky_accum / float(n), 0.0, 1.0)
        del sky_accum
        res = compose_sky_ground_with_anti_ghosting(stacked_sky, ref_img, sky_mask)
        del stacked_sky, ref_img
        return np.clip(res, 0.0, 1.0)

    else:
        # En trípode fijo: el suelo está inmóvil y las estrellas se desplazan.
        ground_accum = ref_img.astype(np.float32)
        sky_accum = ref_img.astype(np.float32)

        sky_bin = (sky_mask > 0.5).astype(np.uint8) * 255
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.01)
        ref_gray = (cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
        kps_ref, descs_ref = sift.detectAndCompute(ref_gray, mask=sky_bin)
        bf = cv2.BFMatcher(cv2.NORM_L2)

        for i in range(1, n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")
            nxt = load_image_as_float32(paths[i])
            ground_accum += nxt

            warped_sky = None
            if descs_ref is not None and len(descs_ref) >= 8:
                nxt_gray = (cv2.cvtColor(nxt, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
                kps_nxt, descs_nxt = sift.detectAndCompute(nxt_gray, mask=sky_bin)
                if descs_nxt is not None and len(descs_nxt) >= 8:
                    matches = bf.knnMatch(descs_ref, descs_nxt, k=2)
                    good = [m for m, nm in matches if m.distance < 0.75 * nm.distance]
                    if len(good) >= 6:
                        pts_ref = np.float32([kps_ref[m.queryIdx].pt for m in good])
                        pts_nxt = np.float32([kps_nxt[m.trainIdx].pt for m in good])
                        M, inliers = cv2.estimateAffinePartial2D(pts_nxt, pts_ref, method=cv2.RANSAC)
                        if M is not None:
                            warped_sky = cv2.warpAffine(nxt, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            if warped_sky is not None:
                sky_accum += warped_sky
                del warped_sky
            else:
                sky_accum += nxt
            del nxt

        stacked_ground = np.clip(ground_accum / float(n), 0.0, 1.0)
        stacked_sky = np.clip(sky_accum / float(n), 0.0, 1.0)
        del ground_accum, sky_accum, ref_img

        res = compose_sky_ground_with_anti_ghosting(stacked_sky, stacked_ground, sky_mask)
        del stacked_sky, stacked_ground
        return np.clip(res, 0.0, 1.0)


def stack_and_stitch_nightscape_panorama(
    grouped_panels: List[List[str]],
    settings: Optional[PanoramaSettings] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None
) -> PanoramaResult:
    """
    Canalización combinada: apila primero cada grupo de tomas por tesela y luego
    cose las imágenes maestras resultantes en la panorámica final.
    """
    if settings is None:
        settings = PanoramaSettings()

    total_groups = len(grouped_panels)
    if total_groups < 2:
        raise ValueError("Se requieren al menos 2 teselas para componer una panorámica.")

    temp_dir = tempfile.mkdtemp(prefix="astroloom_pano_burst_")
    master_paths = []
    master_metas = []

    try:
        for idx, group in enumerate(grouped_panels):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")

            if progress_callback:
                prog = int(5 + 40 * (idx / total_groups))
                progress_callback(prog, f"[Fase 1/2] Apilando tesela {idx + 1}/{total_groups} ({len(group)} tomas)...")

            orig_meta = extract_panel_metadata(group[0], idx)

            stacked_img = stack_panel_group(
                group,
                mount_mode=settings.mount_mode,
                use_mask=settings.use_auto_mask,
                abort_flag=abort_flag
            )

            master_filename = f"master_panel_{idx:03d}.tif"
            master_filepath = os.path.join(temp_dir, master_filename)
            tifffile.imwrite(master_filepath, stacked_img.astype(np.float32), photometric='rgb')
            del stacked_img
            gc.collect()

            orig_meta.filepath = master_filepath
            orig_meta.filename = master_filename
            master_paths.append(master_filepath)
            master_metas.append(orig_meta)

        def remapped_progress(p: int, msg: str):
            if progress_callback:
                p_global = int(45 + 0.55 * p)
                progress_callback(p_global, f"[Fase 2/2] {msg}")

        result = stitch_nightscape_panorama(
            panels_paths=master_paths,
            settings=settings,
            progress_callback=remapped_progress,
            abort_flag=abort_flag,
            panels_metadata=master_metas
        )
        return result

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def estimate_series_vignetting(paths: List[str]) -> float:
    """
    Estima la caída porcentual mediana de luz en bordes y esquinas (0.0 a 1.0)
    a partir del fondo del cielo en una serie panorámica.
    """
    if not paths:
        return 0.35
    step = max(1, len(paths) // 8)
    sample_paths = paths[::step][:8]
    falloffs = []

    for p in sample_paths:
        try:
            ext = os.path.splitext(p)[1].lower()
            im = None
            if ext in ('.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw'):
                try:
                    import rawpy
                    with rawpy.imread(p) as r:
                        th = r.extract_thumb()
                        if th.format == rawpy.ThumbFormat.JPEG:
                            im = cv2.imdecode(np.frombuffer(th.data, np.uint8), cv2.IMREAD_GRAYSCALE)
                except Exception:
                    im = None

            if im is None:
                im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

            if im is None:
                continue

            small = cv2.resize(im, (120, 180))
            sh, sw = small.shape
            sky = small[:int(sh * 0.75), :]
            # Supresión morfológica de estrellas (filtro mínimo 5x5)
            bg = cv2.erode(sky, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

            cy, cx = sh // 2, sw // 2
            Y, X = np.ogrid[:sky.shape[0], :sw]
            r_norm = np.sqrt((X - cx)**2 + (Y - cy)**2) / np.sqrt(cx**2 + cy**2)

            center_val = float(np.median(bg[r_norm < 0.28]))
            edge_val = float(np.median(bg[(r_norm > 0.65) & (r_norm < 0.95)]))

            if center_val > 5.0:
                fo = max(0.0, min(0.85, 1.0 - (edge_val / center_val)))
                falloffs.append(fo)
        except Exception:
            continue

    if not falloffs:
        return 0.35
    return float(np.median(falloffs))
