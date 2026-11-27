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
import logging
import tempfile
import shutil
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import tifffile
from PIL import Image, ExifTags

logger = logging.getLogger("AstroLoom.Panorámicas")

import astroloom_rs

HAS_RUST = True

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
    use_plate_solving: bool = True         # Plate solving local offline (catálogo ALCAT001)
    plate_catalog_profile: str = "lite"    # Perfil de catálogo: "lite", "standard", "deep"
    view_pitch_deg: float = 0.0            # Inclinación 3D de la esfera de proyección (-45° a +45°)
    view_roll_deg: float = 0.0             # Rotación / nivelación 3D del horizonte (-20° a +20°)
    view_yaw_deg: float = 0.0              # Centrado horizontal 3D de la panorámica (-60° a +60°)
    arch_curvature: float = 0.0            # Deformación de arco / curvatura de horizonte (-0.25 a +0.25)
    preview_scale: float = 1.0             # 1.0 = alta resolución completa, <1.0 (ej. 0.25) = borrador rápido
    annotate_constellations: bool = False  # Generar capa RGBA de constelaciones y estrellas nombradas


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
    annotation_rgba: Optional[np.ndarray] = None  # Capa RGBA uint8 de constelaciones y estrellas


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


def find_maximum_inscribed_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Calcula el rectángulo de área máxima completamente contenido dentro de la máscara binaria
    utilizando el algoritmo O(H x W) basado en histogramas acumulativos.
    Garantiza un auto-recorte perfecto eliminando al 100% los bordes negros irregulares.
    Retorna (x, y, ancho, alto).
    """
    mask_u8 = np.ascontiguousarray(mask if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8) * 255)
    return astroloom_rs.find_max_inscribed_rect_rs(mask_u8)


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

    pts_c = np.ascontiguousarray(pts, dtype=np.float32)
    feats, quads_arr = astroloom_rs.build_star_asterisms_rs(pts_c, k_neighbors)
    return feats, [tuple(q) for q in quads_arr]


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
    Empareja asterismos de 4 estrellas entre dos paneles mediante árbol k-d en Rust
    y verifica correspondencias con RANSAC rígido/afín.
    """
    if len(feats1) == 0 or len(feats2) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0

    quads1_arr = np.ascontiguousarray(quads1, dtype=np.int32)
    quads2_arr = np.ascontiguousarray(quads2, dtype=np.int32)
    feats1_c = np.ascontiguousarray(feats1, dtype=np.float32)
    feats2_c = np.ascontiguousarray(feats2, dtype=np.float32)
    m1, m2 = astroloom_rs.match_star_asterisms_rs(
        feats1_c, quads1_arr, feats2_c, quads2_arr, float(tolerance), int(min_votes)
    )
    if len(m1) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), 0
    pts1_m = pts1[m1]
    pts2_m = pts2[m2]
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


def _level_and_center_rotations(
    rmats: List[np.ndarray],
    level_horizon: bool = True,
    half_fov_v_rad: float = 0.65,
    max_elev_deg: float = 76.0,
) -> List[np.ndarray]:
    """
    Nivela el horizonte astronómico/terrestre y centra el barrido en yaw = 0 sobre SO(3).
    Nota: OpenCV PyRotationWarper proyecta p_world = R @ p_cam, por lo que toda rotación
    global de la esfera panorámica DEBE multiplicarse por la izquierda (R_global @ R_i)
    para preservar intacta la alineación relativa entre paneles.
    """
    n = len(rmats)
    if n < 1:
        return []
    try:
        R_step = [r.astype(np.float64) for r in rmats]

        if level_horizon and n >= 2:
            horiz_vecs = []
            cam_ups = []
            opts = [r @ np.array([0.0, 0.0, 1.0]) for r in R_step]
            for r in R_step:
                horiz_vecs.append(r @ np.array([1.0, 0.0, 0.0]))
                cam_ups.append(r @ np.array([0.0, -1.0, 0.0]))
            for i in range(n - 1):
                dz = opts[i + 1] - opts[i]
                nrm = np.linalg.norm(dz)
                if nrm > 1e-4:
                    horiz_vecs.append(dz / nrm)

            if len(horiz_vecs) >= 2:
                _, _, Vt = np.linalg.svd(np.asarray(horiz_vecs, dtype=np.float64))
                zenith = Vt[-1] / max(1e-12, float(np.linalg.norm(Vt[-1])))
                if np.dot(zenith, np.mean(cam_ups, axis=0)) < 0.0:
                    zenith = -zenith

                target_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
                v_rot = np.cross(zenith, target_up)
                sin_a = float(np.linalg.norm(v_rot))
                cos_a = float(np.clip(np.dot(zenith, target_up), -1.0, 1.0))
                if sin_a > 1e-6:
                    k = v_rot / sin_a
                    K_mat = np.array(
                        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
                        dtype=np.float64,
                    )
                    R_align = np.eye(3, dtype=np.float64) + sin_a * K_mat + (1.0 - cos_a) * (K_mat @ K_mat)
                    R_step = [R_align @ r for r in R_step]

        # Centrar acimut medio en yaw = 0 (+Z) para no cruzar la costura ±180° de la esfera
        opts_lvl = [r @ np.array([0.0, 0.0, 1.0]) for r in R_step]
        mean_xz = np.mean([[o[0], o[2]] for o in opts_lvl], axis=0)
        if np.hypot(mean_xz[0], mean_xz[1]) > 1e-5:
            yaw_mid = float(np.arctan2(mean_xz[0], mean_xz[1]))
            cy, sy = np.cos(-yaw_mid), np.sin(-yaw_mid)
            R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
            R_step = [R_yaw @ r for r in R_step]

        # Guardia polar: evitar que bordes superiores rocen el polo ±90° en proyección esférica
        opts_centered = [r @ np.array([0.0, 0.0, 1.0]) for r in R_step]
        elevs = [float(np.arcsin(np.clip(-o[1], -1.0, 1.0))) for o in opts_centered]
        max_lim = np.radians(max_elev_deg)
        max_top = max(elevs) + half_fov_v_rad
        min_bot = min(elevs) - half_fov_v_rad
        if max_top > max_lim:
            pitch_down = min(max_top - max_lim, max(0.0, 0.5 * (max_top + min_bot)))
            if pitch_down > 1e-4:
                cp, sp = np.cos(-pitch_down), np.sin(-pitch_down)
                R_p = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
                R_step = [R_p @ r for r in R_step]
        elif min_bot < -max_lim:
            pitch_up = min(-max_lim - min_bot, max(0.0, -0.5 * (max_top + min_bot)))
            if pitch_up > 1e-4:
                cp, sp = np.cos(pitch_up), np.sin(pitch_up)
                R_p = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
                R_step = [R_p @ r for r in R_step]

        return [r.astype(np.float32) for r in R_step]
    except Exception:
        return [r.astype(np.float32) for r in rmats]


def _unproject_radtan_rays(
    px: np.ndarray,
    py: np.ndarray,
    focal: float,
    k1: float,
    k2: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Desproyecta coordenadas de píxel a rayos unitarios 3D invirtiendo distorsión radial (k1, k2)."""
    dx = (px - cx) * (1.0 / focal)
    dy = (py - cy) * (1.0 / focal)
    rd = np.hypot(dx, dy)
    ru = rd.copy()
    for _ in range(4):
        r2 = ru * ru
        r4 = r2 * r2
        f_val = ru * (1.0 + k1 * r2 + k2 * r4) - rd
        df = np.maximum(1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4, 1e-6)
        ru = ru - f_val / df
    scale = np.where(rd > 1e-9, ru / np.maximum(rd, 1e-9), 1.0)
    vx = dx * scale
    vy = dy * scale
    inv_norm = 1.0 / np.sqrt(vx * vx + vy * vy + 1.0)
    return np.stack([vx * inv_norm, vy * inv_norm, inv_norm], axis=-1)


def _refine_astro_camera_rig(
    R_init: List[np.ndarray],
    pair_matches: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
    full_w: int,
    full_h: int,
    f_init: float,
    k1_init: float = 0.0,
    anchor_idx: Optional[int] = None,
) -> Tuple[List[np.ndarray], float, float, float, float, float]:
    """
    Ajuste de haces conjunto (Bundle Adjustment) sobre la esfera celeste:
    optimiza simultáneamente las rotaciones 3D SO(3) de todos los paneles y los
    parámetros intrínsecos compartidos de la lente (focal, k1, k2, dcx, dcy).
    """
    n = len(R_init)
    if anchor_idx is None:
        anchor_idx = n // 2
    r_ref = R_init[anchor_idx].astype(np.float64)
    R_norm = [r_ref.T @ r.astype(np.float64) for r in R_init]

    if not pair_matches:
        return R_norm, float(f_init), float(k1_init), 0.0, 0.0, 0.0

    p_i_list, p_j_list, xa_l, ya_l, xb_l, yb_l, w_l = [], [], [], [], [], [], []
    for i, j, pts_a, pts_b, weights in pair_matches:
        m = len(pts_a)
        if m == 0:
            continue
        p_i_list.append(np.full(m, i, dtype=np.int32))
        p_j_list.append(np.full(m, j, dtype=np.int32))
        xa_l.append(pts_a[:, 0])
        ya_l.append(pts_a[:, 1])
        xb_l.append(pts_b[:, 0])
        yb_l.append(pts_b[:, 1])
        w_l.append(weights)

    if not p_i_list:
        return R_norm, float(f_init), float(k1_init), 0.0, 0.0, 0.0

    p_i = np.concatenate(p_i_list)
    p_j = np.concatenate(p_j_list)
    xa = np.concatenate(xa_l).astype(np.float64)
    ya = np.concatenate(ya_l).astype(np.float64)
    xb = np.concatenate(xb_l).astype(np.float64)
    yb = np.concatenate(yb_l).astype(np.float64)
    sqrt_w = np.sqrt(np.clip(np.concatenate(w_l).astype(np.float64), 0.05, 1.0))

    if len(xa) < 6:
        return R_norm, float(f_init), float(k1_init), 0.0, 0.0, 0.0

    try:
        rot_indices = [idx for idx in range(n) if idx != anchor_idx]
        fit_intrinsics = len(xa) >= 18

        if fit_intrinsics:
            x0_list = [float(f_init), float(k1_init), 0.0, 0.0, 0.0]
            lo_list = [f_init * 0.82, -0.35, -0.25, -full_w * 0.04, -full_h * 0.04]
            hi_list = [f_init * 1.18, 0.25, 0.25, full_w * 0.04, full_h * 0.04]
        else:
            x0_list, lo_list, hi_list = [], [], []

        for idx in rot_indices:
            rv, _ = cv2.Rodrigues(R_norm[idx])
            x0_list.extend(rv.ravel().tolist())
            lo_list.extend([-np.inf, -np.inf, -np.inf])
            hi_list.extend([np.inf, np.inf, np.inf])

        x0 = np.asarray(x0_list, dtype=np.float64)
        lo_arr = np.asarray(lo_list, dtype=np.float64)
        hi_arr = np.asarray(hi_list, dtype=np.float64)
        cx_base = full_w * 0.5
        cy_base = full_h * 0.5
        base_off = 5 if fit_intrinsics else 0

        def _res_fn(params: np.ndarray) -> np.ndarray:
            if fit_intrinsics:
                f_v, k1_v, k2_v, dcx_v, dcy_v = params[:5]
            else:
                f_v, k1_v, k2_v, dcx_v, dcy_v = float(f_init), float(k1_init), 0.0, 0.0, 0.0
            Rs = [None] * n
            Rs[anchor_idx] = np.eye(3, dtype=np.float64)
            for k_idx, pan_idx in enumerate(rot_indices):
                rv = params[base_off + 3 * k_idx : base_off + 3 * (k_idx + 1)]
                Rs[pan_idx], _ = cv2.Rodrigues(rv)
            Rs_arr = np.asarray(Rs, dtype=np.float64)
            va = _unproject_radtan_rays(xa, ya, f_v, k1_v, k2_v, cx_base + dcx_v, cy_base + dcy_v)
            vb = _unproject_radtan_rays(xb, yb, f_v, k1_v, k2_v, cx_base + dcx_v, cy_base + dcy_v)
            wa = np.einsum("nij,nj->ni", Rs_arr[p_i], va)
            wb = np.einsum("nij,nj->ni", Rs_arr[p_j], vb)
            return ((wa - wb) * (f_v * sqrt_w[:, None])).ravel()

        # Levenberg-Marquardt con ponderación robusta IRLS soft_l1 y proyección de cotas
        f_scale = 1.5
        eps_jac = 1e-6

        def _soft_l1_cost(rv_vec: np.ndarray) -> float:
            z = (rv_vec / f_scale) ** 2
            return float((f_scale ** 2) * np.sum(np.sqrt(1.0 + z) - 1.0))

        x_cur = np.clip(x0, lo_arr, hi_arr)
        r_cur = _res_fn(x_cur)
        cost_cur = _soft_l1_cost(r_cur)
        lam = 1e-3
        n_p = len(x_cur)

        for _ in range(35):
            w_irls = 1.0 / np.sqrt(1.0 + (r_cur / f_scale) ** 2)
            J = np.empty((len(r_cur), n_p), dtype=np.float64)
            for j_col in range(n_p):
                step = eps_jac * max(1.0, abs(x_cur[j_col]))
                if x_cur[j_col] + step > hi_arr[j_col]:
                    step = -step
                xp = x_cur.copy()
                xp[j_col] += step
                J[:, j_col] = (_res_fn(xp) - r_cur) / step

            Jw = J * w_irls[:, None]
            H = Jw.T @ J
            g = Jw.T @ r_cur
            diag = np.maximum(np.diag(H), 1e-6)

            accepted = False
            for _ in range(10):
                A = H + lam * np.diag(diag)
                try:
                    dx = -np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    dx = -np.linalg.lstsq(A, g, rcond=None)[0]
                x_new = np.clip(x_cur + dx, lo_arr, hi_arr)
                r_new = _res_fn(x_new)
                cost_new = _soft_l1_cost(r_new)
                if cost_new < cost_cur:
                    improvement = cost_cur - cost_new
                    x_cur, r_cur, cost_cur = x_new, r_new, cost_new
                    lam = max(lam * 0.3, 1e-9)
                    accepted = True
                    if improvement < 1e-9 * max(1.0, cost_cur):
                        lam = 0.0
                    break
                lam = min(lam * 5.0, 1e9)
            if not accepted or lam == 0.0:
                break

        if fit_intrinsics:
            f_out, k1_out, k2_out, dcx_out, dcy_out = [float(v) for v in x_cur[:5]]
        else:
            f_out, k1_out, k2_out, dcx_out, dcy_out = float(f_init), float(k1_init), 0.0, 0.0, 0.0

        R_out = [np.eye(3, dtype=np.float64) for _ in range(n)]
        for k_idx, pan_idx in enumerate(rot_indices):
            rv = x_cur[base_off + 3 * k_idx : base_off + 3 * (k_idx + 1)]
            R_out[pan_idx], _ = cv2.Rodrigues(rv)
        return R_out, f_out, k1_out, k2_out, dcx_out, dcy_out
    except Exception as e_ba:
        logger.warning(f"Refinamiento conjunto de haces omitido, usando poses base: {e_ba}")
        return R_norm, float(f_init), float(k1_init), 0.0, 0.0, 0.0


def _warp_panel_with_distortion(
    warper: Any,
    src: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    dcx: float = 0.0,
    dcy: float = 0.0,
    interp: int = cv2.INTER_LINEAR,
) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Proyecta un panel directamente al lienzo panorámico incorporando el modelo
    de distorsión radial de gran angular (1 + k1*r^2 + k2*r^4) y descentrado (dcx, dcy)
    en una única pasada de remuestreo (sin doble interpolación).
    """
    if abs(k1) < 1e-6 and abs(k2) < 1e-6 and abs(dcx) < 1e-2 and abs(dcy) < 1e-2:
        return warper.warp(src, K, R, interp, cv2.BORDER_CONSTANT)

    h_s, w_s = src.shape[:2]
    roi, xmap, ymap = warper.buildMaps((w_s, h_s), K, R)
    f = np.float32(K[0, 0])
    inv_f = np.float32(1.0 / max(1e-6, float(K[0, 0])))
    cx0 = np.float32(K[0, 2])
    cy0 = np.float32(K[1, 2])

    valid = (xmap >= 0.0) & (ymap >= 0.0)
    xu = (xmap - cx0) * inv_f
    yu = (ymap - cy0) * inv_f
    r2 = xu * xu + yu * yu
    f_dist = f * (np.float32(1.0) + np.float32(k1) * r2 + np.float32(k2) * (r2 * r2))
    x_dist = np.where(valid, np.float32(float(cx0) + dcx) + xu * f_dist, np.float32(-1.0))
    y_dist = np.where(valid, np.float32(float(cy0) + dcy) + yu * f_dist, np.float32(-1.0))
    dst = cv2.remap(src, x_dist, y_dist, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (int(roi[0]), int(roi[1])), dst


def _apply_view_orientation(
    rmats: List[np.ndarray],
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[np.ndarray]:
    """
    Aplica una rotación 3D virtual (Pitch, Roll, Yaw) sobre la esfera panorámica
    antes del warp, permitiendo aplanar el horizonte y moldear el arco sin doble remuestreo.
    """
    if abs(pitch_deg) < 1e-3 and abs(roll_deg) < 1e-3 and abs(yaw_deg) < 1e-3:
        return rmats

    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    y = np.radians(yaw_deg)

    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    cy, sy = np.cos(y), np.sin(y)

    R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    R_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)

    R_view = R_roll @ R_pitch @ R_yaw
    # OpenCV PyRotationWarper usa p_world = R @ p_cam => rotar la esfera se multiplica por la izquierda
    return [(R_view @ rm.astype(np.float64)).astype(np.float32) for rm in rmats]


def apply_arch_curvature_deformation(image: np.ndarray, curvature: float = 0.0) -> np.ndarray:
    """
    Aplica una deformación parabólica vertical suave (Curvature / Bow warp)
    para nivelar el horizonte y arquear la Vía Láctea.
    Valores negativos (<0) corrigen el efecto cuenco elevando el centro,
    lo que aplana los árboles y acentúa el arco estelar hacia el cenit.
    """
    if abs(curvature) < 1e-4 or image is None or image.size == 0:
        return image

    if HAS_RUST:
        try:
            if image.ndim == 3 and image.dtype == np.float32 and hasattr(astroloom_rs, "apply_arch_curvature_rgb_rs"):
                img_c = np.ascontiguousarray(image)
                return astroloom_rs.apply_arch_curvature_rgb_rs(img_c, float(curvature))
            elif image.ndim == 2 and image.dtype == np.uint8 and hasattr(astroloom_rs, "apply_arch_curvature_mask_rs"):
                mask_c = np.ascontiguousarray(image)
                return astroloom_rs.apply_arch_curvature_mask_rs(mask_c, float(curvature))
        except Exception:
            pass

    # Fallback sin meshgrid (ahorro de memoria 1D broadcasting)
    h, w = image.shape[:2]
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    nx = (np.arange(w, dtype=np.float32) - w / 2.0) / (w / 2.0)
    dy = np.float32(curvature * h) * (1.0 - nx ** 2)
    map_y = np.arange(h, dtype=np.float32)[:, None] - dy[None, :]
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

    logger.info(f"Iniciando cosido de {n} paneles (proyección={settings.projection}, astro_mode={settings.astro_mode})...")

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
    need_clean_small = (
        (getattr(settings, 'preview_scale', 1.0) < 1.0)
        or getattr(settings, 'use_plate_solving', False)
        or getattr(settings, 'exposure_comp', False)
    )
    clean_imgs_small = [] if need_clean_small else None
    for i, path in enumerate(panels_paths):
        if abort_flag and abort_flag():
            raise InterruptedError("Operación cancelada por el usuario.")

        ext = os.path.splitext(path)[1].lower()
        if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw']:
            try:
                import rawpy
                from core.config_manager import get_config_val
                from core.icc_manager import get_rawpy_colorspace
                raw_cs = get_rawpy_colorspace(get_config_val("icc_profile", "srgb"))
                with rawpy.imread(path) as r:
                    # Mismos parámetros fotométricos que load_image_as_float32 para igualar borrador y alta resolución
                    rgb_half = r.postprocess(
                        half_size=True,
                        gamma=(2.222, 4.5),
                        no_auto_bright=True,
                        use_camera_wb=True,
                        output_color=raw_cs,
                        adjust_maximum_thr=0.0,
                    )
                small = cv2.resize(rgb_half, (rgb_half.shape[1] // 2, rgb_half.shape[0] // 2), interpolation=cv2.INTER_AREA)
                del rgb_half
            except Exception:
                img_f = load_image_as_float32(path)
                u8 = (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)
                small = cv2.resize(u8, (u8.shape[1] // 4, u8.shape[0] // 4), interpolation=cv2.INTER_AREA)
                del img_f, u8
        else:
            img_f = load_image_as_float32(path)
            u8 = (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)
            small = cv2.resize(u8, (u8.shape[1] // 4, u8.shape[0] // 4), interpolation=cv2.INTER_AREA)
            del img_f, u8

        # Preservar miniatura limpia para borrador rápido sin distorsión CLAHE de detección
        if clean_imgs_small is not None:
            clean_imgs_small.append(small.copy())

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
    astro_pair_matches: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    lens_k1, lens_k2, lens_dcx_full, lens_dcy_full = 0.0, 0.0, 0.0, 0.0
    plate_sols: List[Optional[Dict[str, Any]]] = []
    pano_anchor_idx: int = n // 2

    # 3.0 Plate Solving Local Offline (Quads 4D + catálogo ALCAT001) + Ajuste Conjunto de Lente y Poses SO(3)
    if getattr(settings, 'use_plate_solving', False) and clean_imgs_small is not None:
        try:
            from core.plate_solver import (
                HAS_RUST_PLATE_SOLVER,
                get_best_available_catalog,
                solve_frame,
                build_star_anchors_between_frames,
            )
            req_profile = getattr(settings, 'plate_catalog_profile', 'auto')
            if req_profile == "auto":
                f_mm = settings.focal_length_override or metas[0].focal_length_mm or 24.0
                if f_mm <= 26.0:
                    req_profile = "lite"
                elif f_mm <= 55.0:
                    req_profile = "standard"
                else:
                    req_profile = "deep"
            cat_path = get_best_available_catalog(req_profile)
            if HAS_RUST_PLATE_SOLVER and cat_path:
                if progress_callback:
                    progress_callback(21, "Ejecutando plate solving astrométrico sobre el cielo...")
                plate_sols: List[Optional[Dict[str, Any]]] = []
                for i in range(n):
                    if abort_flag and abort_flag():
                        raise InterruptedError("Operación cancelada por el usuario.")
                    sm_mask = sky_masks_sm[i] if (sky_masks_sm is not None and i < len(sky_masks_sm)) else None
                    sol = solve_frame(
                        clean_imgs_small[i],
                        sky_mask=sm_mask,
                        focal_px_hint=focal_work,
                        catalog_path=cat_path,
                        scale_to_full=scale_to_full,
                        filepath=panels_paths[i],
                    )
                    plate_sols.append(sol)

                solved_valid = [s for s in plate_sols if s is not None]
                if len(solved_valid) >= 2:
                    focal_full = float(np.median([s["focal_px"] for s in solved_valid]))
                    k1_med = float(np.median([s.get("k1", 0.0) for s in solved_valid]))
                    focal_work = focal_full / scale_to_full
                    logger.info(
                        f"Plate solving: {len(solved_valid)}/{n} paneles resueltos "
                        f"(focal calibrada={focal_full:.1f} px, k1={k1_med:.4f})."
                    )

                    min_ps_anchors = max(4, min(settings.min_inliers, 6))
                    for i in range(n):
                        if plate_sols[i] is None:
                            continue
                        for j in range(i + 1, n):
                            if plate_sols[j] is None:
                                continue
                            pair = build_star_anchors_between_frames(
                                plate_sols[i],
                                plate_sols[j],
                                os.path.basename(panels_paths[i]),
                                os.path.basename(panels_paths[j]),
                            )
                            anchors = pair["anchors"]
                            if len(anchors) >= min_ps_anchors:
                                R_j_to_i = pair["rotation_b_to_a"]
                                star_edges_rot[(j, i)] = (R_j_to_i, len(anchors))
                                star_edges_rot[(i, j)] = (R_j_to_i.T, len(anchors))

                                pts_a = np.array([[a["x_a"], a["y_a"]] for a in anchors], dtype=np.float32)
                                pts_b = np.array([[a["x_b"], a["y_b"]] for a in anchors], dtype=np.float32)
                                w_anc = np.array([a["weight"] for a in anchors], dtype=np.float32)
                                astro_pair_matches.append((i, j, pts_a, pts_b, w_anc))

                                M_aff, _ = cv2.estimateAffinePartial2D(
                                    pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=8.0
                                )
                                if M_aff is not None:
                                    H_j_to_i = np.vstack([M_aff, [0, 0, 1]])
                                    try:
                                        H_i_to_j = np.linalg.inv(H_j_to_i)
                                        star_edges_affine[(j, i)] = (H_j_to_i, len(anchors))
                                        star_edges_affine[(i, j)] = (H_i_to_j, len(anchors))
                                    except np.linalg.LinAlgError:
                                        pass

                    if len(solved_valid) == n and settings.projection != "plane":
                        anchor_idx = n // 2
                        pano_anchor_idx = anchor_idx
                        r_anchor = np.asarray(plate_sols[anchor_idx]["rotation_icrf"], dtype=np.float64)
                        r_init = [
                            r_anchor.T @ np.asarray(plate_sols[i]["rotation_icrf"], dtype=np.float64)
                            for i in range(n)
                        ]
                        r_ref, focal_full, lens_k1, lens_k2, lens_dcx_full, lens_dcy_full = _refine_astro_camera_rig(
                            r_init,
                            astro_pair_matches,
                            full_w,
                            full_h,
                            focal_full,
                            k1_med,
                            anchor_idx=anchor_idx,
                        )
                        focal_work = focal_full / scale_to_full
                        half_fov_v = float(np.arctan(0.5 * full_h / max(100.0, focal_full)))
                        rmats_corr = _level_and_center_rotations(
                            r_ref,
                            level_horizon=settings.wave_correction,
                            half_fov_v_rad=half_fov_v,
                        )
                        cameras = [True] * n
                        logger.info(
                            f"Ajuste astrométrico conjunto: f={focal_full:.1f} px, "
                            f"k1={lens_k1:.4f}, k2={lens_k2:.4f}, dcx={lens_dcx_full:.1f}, dcy={lens_dcy_full:.1f}."
                        )
                        if progress_callback:
                            progress_callback(35, f"Alineación astrométrica 3D completada ({n}/{n} paneles resueltos).")
        except InterruptedError:
            raise
        except Exception as e_ps:
            logger.warning(f"Plate solving omitido por error, usando fallback clásico: {e_ps}")

    # 3.1 Identificación Estelar por Asterismos (si Plate Solving fue parcial o está desactivado)
    if (cameras is None or rmats_corr is None) and settings.star_matching and settings.projection != "plane":
        if progress_callback:
            progress_callback(25, "Identificando estrellas brillantes y asterismos estelares...")

        stars_per_panel = [extract_bright_stars(sm, max_stars=120) for sm in imgs_small]
        quads_per_panel = [build_star_asterisms(pts, k_neighbors=7) for pts in stars_per_panel]
        min_star_inliers = max(4, min(settings.min_inliers, 5))

        for i in range(n):
            for j in range(i + 1, n):
                if abort_flag and abort_flag():
                    raise InterruptedError("Operación cancelada por el usuario.")
                if (j, i) in star_edges_rot and (i, j) in star_edges_rot:
                    continue
                if len(stars_per_panel[i]) < 4 or len(stars_per_panel[j]) < 4:
                    continue

                pts_i_m, pts_j_m, n_inl = match_star_asterisms(
                    quads_per_panel[i][0], quads_per_panel[i][1], stars_per_panel[i],
                    quads_per_panel[j][0], quads_per_panel[j][1], stars_per_panel[j],
                    tolerance=0.038, min_votes=2
                )

                if n_inl >= min_star_inliers:
                    pts_i_full = pts_i_m * scale_to_full
                    pts_j_full = pts_j_m * scale_to_full
                    astro_pair_matches.append((i, j, pts_i_full, pts_j_full, np.ones(len(pts_i_full), dtype=np.float32)))
                    M_aff, _ = cv2.estimateAffinePartial2D(pts_j_full, pts_i_full, method=cv2.RANSAC, ransacReprojThreshold=8.0)
                    if M_aff is not None:
                        H_j_to_i = np.vstack([M_aff, [0, 0, 1]])
                        try:
                            H_i_to_j = np.linalg.inv(H_j_to_i)
                            star_edges_affine[(j, i)] = (H_j_to_i, n_inl)
                            star_edges_affine[(i, j)] = (H_i_to_j, n_inl)
                        except np.linalg.LinAlgError:
                            pass

                    R_j_to_i, n_rot_inl = estimate_camera_rotation_from_stars(
                        pts_i_m, pts_j_m, small_w, small_h, focal_work
                    )
                    if R_j_to_i is not None and n_rot_inl >= min_star_inliers:
                        star_edges_rot[(j, i)] = (R_j_to_i, n_rot_inl)
                        star_edges_rot[(i, j)] = (R_j_to_i.T, n_rot_inl)

        if star_edges_rot:
            best_anchor = n // 2
            best_reach = 0
            for cand in range(n):
                reach = len(_find_reachable_nodes(cand, star_edges_rot))
                if reach > best_reach:
                    best_reach = reach
                    best_anchor = cand

            if best_reach == n:
                R_global = {best_anchor: np.eye(3, dtype=np.float64)}
                queue = [best_anchor]
                visited = {best_anchor}
                while queue:
                    curr = queue.pop(0)
                    # Ordenar vecinos por mayor número de estrellas coincidentes para minimizar deriva BFS
                    nbrs = [
                        (j, star_edges_rot[(j, curr)][0], star_edges_rot[(j, curr)][1])
                        for j in range(n)
                        if (j, curr) in star_edges_rot and j not in visited
                    ]
                    nbrs.sort(key=lambda item: item[2], reverse=True)
                    for j, R_j_to_curr, _ in nbrs:
                        visited.add(j)
                        R_global[j] = R_global[curr] @ R_j_to_curr
                        queue.append(j)

                r_ref, focal_full, lens_k1, lens_k2, lens_dcx_full, lens_dcy_full = _refine_astro_camera_rig(
                    [R_global[i] for i in range(n)],
                    astro_pair_matches,
                    full_w,
                    full_h,
                    focal_full,
                    0.0,
                    anchor_idx=best_anchor,
                )
                focal_work = focal_full / scale_to_full
                half_fov_v = float(np.arctan(0.5 * full_h / max(100.0, focal_full)))
                rmats_corr = _level_and_center_rotations(
                    r_ref,
                    level_horizon=settings.wave_correction,
                    half_fov_v_rad=half_fov_v,
                )
                cameras = [True] * n
                if progress_callback:
                    progress_callback(35, f"Alineación por asterismos estelares completada ({len(star_edges_rot) // 2} solapamientos).")

    # 3.2 Fallback a calibración SIFT + Ajuste de Haces de Rayos (cielos nublados, crepúsculo o sin estrellas)
    if (cameras is None or rmats_corr is None) and settings.projection != "plane":
        if sky_masks_sm is not None:
            try:
                if progress_callback:
                    progress_callback(25, "Calibrando cámaras sobre rasgos del cielo (horizonte protegido)...")
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
                        half_fov_v = float(np.arctan(0.5 * full_h / max(100.0, focal_full)))
                        rmats_corr = _level_and_center_rotations(
                            [c.R for c in cameras],
                            level_horizon=settings.wave_correction,
                            half_fov_v_rad=half_fov_v,
                        )
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
                half_fov_v = float(np.arctan(0.5 * full_h / max(100.0, focal_full)))
                rmats_corr = _level_and_center_rotations(
                    [c.R for c in cameras],
                    level_horizon=settings.wave_correction,
                    half_fov_v_rad=half_fov_v,
                )
                if progress_callback:
                    progress_callback(35, "Alineación 3D global completada con ajuste de haces óptimo.")

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
    valid_projs = ("spherical", "cylindrical", "plane", "stereographic", "mercator", "paniniA2B1")
    if proj_type not in valid_projs:
        proj_type = "spherical"

    MAX_CANVAS_PIXELS = 65_000_000  # 65 Megapíxeles para estabilidad garantizada de RAM en cualquier PC

    if cameras is not None and rmats_corr is not None:
        # Aplicar orientación 3D virtual (Pitch / Roll / Yaw) antes de proyectar
        rmats_corr = _apply_view_orientation(
            rmats_corr,
            pitch_deg=getattr(settings, 'view_pitch_deg', 0.0),
            roll_deg=getattr(settings, 'view_roll_deg', 0.0),
            yaw_deg=getattr(settings, 'view_yaw_deg', 0.0),
        )

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

        # Fallback seguro a esférica si stereographic/panini/mercator diverge en FOV extremo (>160°)
        max_panel_dim = max((max(s[0], s[1]) for s in sizes), default=0)
        if (
            canvas_w <= 0
            or canvas_h <= 0
            or max(canvas_w, canvas_h) > max(full_w, full_h) * 18
            or max_panel_dim > max(full_w, full_h) * 8
        ) and proj_type != "spherical":
            logger.warning(f"Proyección '{proj_type}' excedió límites de lienzo ({canvas_w}x{canvas_h}); usando 'spherical'.")
            proj_type = "spherical"
            warper = cv2.PyRotationWarper(proj_type, focal_full)
            corners, sizes = [], []
            for i in range(n):
                K = np.array([[focal_full, 0, full_w / 2.0], [0, focal_full, full_h / 2.0], [0, 0, 1]], dtype=np.float32)
                roi = warper.warpRoi((full_w, full_h), K, rmats_corr[i].astype(np.float32))
                corners.append((roi[0], roi[1]))
                sizes.append((roi[2], roi[3]))
            min_x = min(c[0] for c in corners)
            min_y = min(c[1] for c in corners)
            max_x = max(c[0] + s[0] for c, s in zip(corners, sizes))
            max_y = max(c[1] + s[1] for c, s in zip(corners, sizes))
            canvas_w = int(max_x - min_x)
            canvas_h = int(max_y - min_y)

        base_scale = float(settings.preview_scale) if getattr(settings, 'preview_scale', 1.0) < 1.0 else 1.0
        max_dim = max(canvas_w, canvas_h)
        need_rescale = (base_scale < 1.0) or (canvas_w * canvas_h > MAX_CANVAS_PIXELS) or (max_dim > 24000)

        if need_rescale:
            if base_scale < 1.0:
                scale_canvas = base_scale
            else:
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
        else:
            scale_canvas = 1.0

        cur_w = int(full_w * scale_canvas)
        cur_h = int(full_h * scale_canvas)
        cur_dcx = lens_dcx_full * scale_canvas
        cur_dcy = lens_dcy_full * scale_canvas
        K = np.array([
            [focal_full, 0, cur_w / 2.0],
            [0, focal_full, cur_h / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)

        # -------------------------------------------------------------
        # 5. Compensación de Ganancia Fotométrica por Bloques (BlocksGainCompensator)
        # -------------------------------------------------------------
        # Nivelación suave 2D sobre imágenes limpias (sin CLAHE) con viñeteo aplicado,
        # eliminando tanto viñeteo residual como gradientes de contaminación lumínica entre paneles.
        panel_gain_maps: List[Optional[np.ndarray]] = [None] * n
        panel_gains = [1.0] * n
        src_for_gain = clean_imgs_small if (clean_imgs_small is not None and len(clean_imgs_small) == n) else (imgs_small if 'imgs_small' in locals() else None)
        if settings.exposure_comp and src_for_gain is not None:
            try:
                gc_scale = min(0.25, 320.0 / max(1.0, float(max(cur_w, cur_h))))
                gc_w, gc_h = max(32, int(cur_w * gc_scale)), max(32, int(cur_h * gc_scale))
                gc_focal = focal_full * gc_scale
                gc_dcx = cur_dcx * gc_scale
                gc_dcy = cur_dcy * gc_scale
                gc_warper = cv2.PyRotationWarper(proj_type, gc_focal)
                K_gc = np.array([
                    [gc_focal, 0, gc_w / 2.0],
                    [0, gc_focal, gc_h / 2.0],
                    [0, 0, 1]
                ], dtype=np.float32)
                v_amt_gc = getattr(settings, 'vignette_amount', 0.35)
                gc_corners, gc_imgs, gc_masks = [], [], []
                max_gc_dim = 0
                for idx_c in range(n):
                    im_sm = cv2.resize(src_for_gain[idx_c], (gc_w, gc_h), interpolation=cv2.INTER_AREA)
                    if v_amt_gc > 0.01:
                        im_f_gc = np.ascontiguousarray(im_sm.astype(np.float32) * (1.0 / 255.0))
                        if HAS_RUST and hasattr(astroloom_rs, "apply_vignette_and_gain_rs"):
                            astroloom_rs.apply_vignette_and_gain_rs(im_f_gc, float(v_amt_gc), 1.0)
                        im_sm = (np.clip(im_f_gc, 0.0, 1.0) * 255.0).astype(np.uint8)
                    c_gc, w_im_gc = _warp_panel_with_distortion(
                        gc_warper, im_sm, K_gc, rmats_corr[idx_c].astype(np.float32),
                        lens_k1, lens_k2, gc_dcx, gc_dcy, cv2.INTER_LINEAR
                    )
                    max_gc_dim = max(max_gc_dim, w_im_gc.shape[0], w_im_gc.shape[1])
                    w_m_gc = (np.max(w_im_gc, axis=2) > 1).astype(np.uint8) * 255
                    gc_corners.append(c_gc)
                    gc_imgs.append(w_im_gc)
                    gc_masks.append(w_m_gc)

                # Limitar bloques a ~6x6 por panel para que BlocksGainCompensator use <2 MB RAM y <30 ms
                bl_size = max(48, int(max_gc_dim // 5))
                if hasattr(cv2, "detail_BlocksGainCompensator") and max_gc_dim <= 1200:
                    gain_comp = cv2.detail_BlocksGainCompensator(bl_size, bl_size, 1)
                else:
                    gain_comp = cv2.detail_GainCompensator()
                gain_comp.feed(gc_corners, gc_imgs, gc_masks)
                m_gains = gain_comp.getMatGains()
                if len(m_gains) == n:
                    for idx_c in range(n):
                        g_arr = np.asarray(m_gains[idx_c], dtype=np.float32)
                        if g_arr.ndim == 2 and (g_arr.shape[0] > 1 or g_arr.shape[1] > 1):
                            panel_gain_maps[idx_c] = np.clip(g_arr, 0.65, 1.55)
                            panel_gains[idx_c] = 1.0
                        else:
                            panel_gains[idx_c] = float(np.clip(g_arr.flat[0], 0.65, 1.55))
                del gc_corners, gc_imgs, gc_masks
            except Exception as e_gc:
                logger.warning(f"Compensador de ganancia por bloques omitido: {e_gc}")
                panel_gain_maps = [None] * n
                panel_gains = [1.0] * n

        if 'imgs_small' in locals():
            del imgs_small
            gc.collect()

        if progress_callback:
            progress_callback(60, f"Calculando partición óptima de costuras ({canvas_w}x{canvas_h}, {canvas_w*canvas_h/1e6:.1f} MP)...")
        logger.info(f"Lienzo proyectado: {canvas_w}x{canvas_h} px ({canvas_w*canvas_h/1e6:.1f} MP, escala={scale_canvas:.3f}).")

        # -------------------------------------------------------------
        # 6. Partición Voronoi y Fusión Multibanda Calibrada
        # -------------------------------------------------------------
        best_dist = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        best_idx = np.full((canvas_h, canvas_w), -1, dtype=np.int16)
        panel_mask = np.zeros((cur_h, cur_w), dtype=np.uint8)
        if cur_h > 4 and cur_w > 4:
            panel_mask[2:-2, 2:-2] = 255
        else:
            panel_mask[:, :] = 255

        for i in range(n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")
            c_m, w_mask = _warp_panel_with_distortion(
                warper, panel_mask, K, rmats_corr[i].astype(np.float32),
                lens_k1, lens_k2, cur_dcx, cur_dcy, cv2.INTER_NEAREST
            )
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

        # Fusión Multibanda con Desacoplamiento Cielo / Suelo (Anti-Ghosting de horizonte)
        num_bands = max(1, min(6, settings.blend_bands if settings.blend_bands > 0 else 5))
        blender_sky = cv2.detail_MultiBandBlender(0, num_bands)
        blender_sky.prepare((0, 0, canvas_w, canvas_h))

        # Escala segura int16 (CV_16S) con margen de seguridad (headroom) para evitar overflow
        # en la reconstrucción y suma de octavas de la pirámide laplaciana de OpenCV
        BLENDER_SCALE = 8192.0

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

            if getattr(settings, 'preview_scale', 1.0) < 1.0 and clean_imgs_small is not None and i < len(clean_imgs_small):
                img_f = (clean_imgs_small[i].astype(np.float32) / 255.0)
                if img_f.shape[:2] != (cur_h, cur_w):
                    img_f = cv2.resize(img_f, (cur_w, cur_h), interpolation=cv2.INTER_AREA)
            else:
                img_f = load_image_as_float32(panels_paths[i])
                if scale_canvas != 1.0:
                    img_f = cv2.resize(img_f, (cur_w, cur_h), interpolation=cv2.INTER_AREA)

            ih, iw = img_f.shape[:2]

            # Compensación adaptativa de caída óptica / viñeteo calibrada y ganancia fotométrica
            v_amt = getattr(settings, 'vignette_amount', 0.5) if settings.exposure_comp else 0.0
            p_gain = float(panel_gains[i]) if i < len(panel_gains) else 1.0

            applied_in_rust = False
            if HAS_RUST and hasattr(astroloom_rs, "apply_vignette_and_gain_rs"):
                try:
                    img_f = np.ascontiguousarray(img_f, dtype=np.float32)
                    astroloom_rs.apply_vignette_and_gain_rs(img_f, float(v_amt), p_gain)
                    applied_in_rust = True
                except Exception:
                    applied_in_rust = False

            if not applied_in_rust:
                if v_amt > 0.01:
                    vig_factor = float(v_amt) * 1.0
                    Y_g, X_g = np.ogrid[:ih, :iw]
                    cx_img, cy_img = iw / 2.0, ih / 2.0
                    r_sq = ((X_g - cx_img)**2 + (Y_g - cy_img)**2) / float(cx_img**2 + cy_img**2)
                    vignette_corr = (1.0 + vig_factor * r_sq + 0.05 * (r_sq**2)).astype(np.float32)[..., np.newaxis]
                    img_f = np.clip(img_f * vignette_corr, 0.0, 1.0)

            corner, w_img = _warp_panel_with_distortion(
                warper, img_f, K, rmats_corr[i].astype(np.float32),
                lens_k1, lens_k2, cur_dcx, cur_dcy, cv2.INTER_LINEAR
            )
            del img_f

            if panel_gain_maps[i] is not None:
                h_gm, w_gm = w_img.shape[:2]
                g_smooth = cv2.resize(panel_gain_maps[i], (w_gm, h_gm), interpolation=cv2.INTER_CUBIC)
                g_smooth = np.clip(g_smooth, 0.65, 1.55)
                w_img = np.clip(w_img * g_smooth[..., np.newaxis], 0.0, 1.0)
            elif not applied_in_rust and abs(panel_gains[i] - 1.0) > 1e-4:
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

            w_img_16s = (np.clip(w_img, 0.0, 1.0) * BLENDER_SCALE).astype(np.int16)
            blender_sky.feed(w_img_16s, seam_dilated, (gx, gy))
            del w_img_16s, local_seam, seam_dilated

            if has_ground:
                m_sky_full = cv2.resize(sky_masks_sm[i], (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
                _, w_mask_sky = _warp_panel_with_distortion(
                    warper, m_sky_full, K, rmats_corr[i].astype(np.float32),
                    lens_k1, lens_k2, cur_dcx, cur_dcy, cv2.INTER_LINEAR
                )
                ground_canvas[y1:y2, x1:x2][local_part] = w_img[iy1:iy2, ix1:ix2][local_part]
                sky_mask_canvas[y1:y2, x1:x2][local_part] = w_mask_sky[iy1:iy2, ix1:ix2][local_part]
                del m_sky_full, w_mask_sky

            del w_img
            gc.collect()

        dst_sky, _ = blender_sky.blend(None, None)
        stitched_sky = np.clip(dst_sky.astype(np.float32) / BLENDER_SCALE, 0.0, 1.0)
        del blender_sky, dst_sky

        total_mask = (best_idx >= 0).astype(np.uint8) * 255
        del best_dist, best_idx

        sky_mask_final_pano: Optional[np.ndarray] = None
        if has_ground and np.mean(sky_mask_canvas) <= 0.98:
            if progress_callback:
                progress_callback(92, "Componiendo cielo y suelo con protección de horizonte y follaje...")
            from core.stacking import compose_sky_ground_with_anti_ghosting
            blended_img = compose_sky_ground_with_anti_ghosting(stitched_sky, ground_canvas, sky_mask_canvas)
            if getattr(settings, 'annotate_constellations', False):
                sky_mask_final_pano = sky_mask_canvas
            else:
                del sky_mask_canvas
            del ground_canvas, stitched_sky
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

        base_scale = float(settings.preview_scale) if getattr(settings, 'preview_scale', 1.0) < 1.0 else 1.0
        scale_factor = base_scale
        if canvas_w * canvas_h * (scale_factor ** 2) > MAX_CANVAS_PIXELS:
            scale_factor *= np.sqrt(MAX_CANVAS_PIXELS / (canvas_w * canvas_h * (scale_factor ** 2)))

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

            if getattr(settings, 'preview_scale', 1.0) < 1.0 and clean_imgs_small is not None and idx < len(clean_imgs_small):
                img_f = (clean_imgs_small[idx].astype(np.float32) / 255.0)
                h, w = img_f.shape[:2]
                mask_src = np.ones((h, w), dtype=np.uint8) * 255
                S = np.diag([scale_to_full, scale_to_full, 1.0])
                H_final = T @ global_H[idx] @ S
            else:
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
        if 'sky_mask_final_pano' in locals() and sky_mask_final_pano is not None:
            sky_mask_final_pano = (
                apply_arch_curvature_deformation((np.clip(sky_mask_final_pano, 0.0, 1.0) * 255.0).astype(np.uint8), settings.arch_curvature).astype(np.float32) / 255.0
            )

    # 5.6 Capa opcional de Constelaciones y Estrellas Nombradas sobre la proyección panorámica
    annotation_rgba = None
    if (
        getattr(settings, 'annotate_constellations', False)
        and cameras is not None
        and rmats_corr is not None
        and plate_sols
        and pano_anchor_idx < len(plate_sols)
        and plate_sols[pano_anchor_idx] is not None
    ):
        try:
            from core.sky_annotations import project_sky_annotations, render_sky_annotations_rgba

            # Promediar R_icrf_to_world sobre los paneles resueltos compensando descentrado (dcx, dcy)
            pp_vec = np.array([lens_dcx_full / max(1.0, focal_full / scale_canvas), lens_dcy_full / max(1.0, focal_full / scale_canvas), 1.0], dtype=np.float64)
            pp_vec /= np.linalg.norm(pp_vec)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            v_pp = np.cross(pp_vec, z_axis)
            s_pp = float(np.linalg.norm(v_pp))
            c_pp = float(np.dot(pp_vec, z_axis))
            if s_pp > 1e-7:
                k_pp = v_pp / s_pp
                Km_pp = np.array([[0.0, -k_pp[2], k_pp[1]], [k_pp[2], 0.0, -k_pp[0]], [-k_pp[1], k_pp[0], 0.0]])
                R_pp = np.eye(3) + s_pp * Km_pp + (1.0 - c_pp) * (Km_pp @ Km_pp)
            else:
                R_pp = np.eye(3)

            H_acc = np.zeros((3, 3), dtype=np.float64)
            for i_s, ps_i in enumerate(plate_sols):
                if ps_i is None:
                    continue
                R_i_est = rmats_corr[i_s].astype(np.float64) @ R_pp @ np.asarray(ps_i["rotation_icrf"], dtype=np.float64).T
                H_acc += R_i_est
            U_h, _, Vt_h = np.linalg.svd(H_acc)
            R_icrf_to_world = U_h @ Vt_h
            if np.linalg.det(R_icrf_to_world) < 0:
                U_h[:, -1] *= -1
                R_icrf_to_world = U_h @ Vt_h

            half_cw = canvas_w * 0.5
            curv_val = float(settings.arch_curvature)
            cx_k, cy_k = float(K[0, 2]), float(K[1, 2])

            def _pano_proj(u_icrf: np.ndarray) -> Optional[Tuple[float, float]]:
                ray_w = R_icrf_to_world @ u_icrf
                up = np.array([0.0, 1.0, 0.0]) if abs(ray_w[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
                ex = np.cross(up, ray_w)
                ex /= np.linalg.norm(ex)
                ey = np.cross(ray_w, ex)
                R_dir = np.column_stack([ex, ey, ray_w]).astype(np.float32)
                wx, wy = warper.warpPoint((cx_k, cy_k), K, R_dir)
                px_c = float(wx) - float(min_x)
                py_c = float(wy) - float(min_y)
                if not np.isfinite(px_c) or not np.isfinite(py_c):
                    return None
                if abs(curv_val) > 1e-4 and half_cw > 1.0:
                    nx = (px_c - half_cw) / half_cw
                    py_c += curv_val * float(canvas_h) * (1.0 - nx * nx)
                if px_c < -canvas_w * 0.15 or px_c > canvas_w * 1.15 or py_c < -canvas_h * 0.15 or py_c > canvas_h * 1.15:
                    return None
                return (px_c, py_c)

            mask_for_annot = sky_mask_final_pano if 'sky_mask_final_pano' in locals() else None
            polylines, labels = project_sky_annotations(
                _pano_proj, canvas_w, canvas_h, sky_mask=mask_for_annot, subdivide_steps=10
            )
            annotation_rgba = render_sky_annotations_rgba(canvas_w, canvas_h, polylines, labels)
        except Exception as e_ann:
            logger.warning(f"No se pudo generar capa de anotaciones en panorámica: {e_ann}")
            annotation_rgba = None

    if 'sky_mask_final_pano' in locals() and sky_mask_final_pano is not None:
        del sky_mask_final_pano

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
    logger.info(f"Cosido completado: {len(connected_indices)}/{n} paneles integrados en {total_time:.1f}s.")

    if progress_callback:
        progress_callback(100, f"Panorámica completada en {total_time:.1f}s.")

    msg_prefix = "[Borrador Rápido] " if getattr(settings, 'preview_scale', 1.0) < 1.0 else ""
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
        message=f"{msg_prefix}Se cosieron exitosamente {len(connected_indices)} de {n} paneles en {total_time:.1f} segundos.",
        annotation_rgba=annotation_rgba,
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

    total_burst_shots = sum(len(g) for g in grouped_panels)
    logger.info(f"Iniciando apilado en ráfaga + cosido: {total_groups} teselas ({total_burst_shots} tomas totales)...")

    temp_dir = tempfile.mkdtemp(prefix="astroloom_pano_burst_")
    master_paths = []
    master_metas = []

    try:
        is_draft = getattr(settings, 'preview_scale', 1.0) < 1.0
        if is_draft:
            if progress_callback:
                progress_callback(10, f"[Borrador] Seleccionando tomas base para {total_groups} teselas...")
            for idx, group in enumerate(grouped_panels):
                orig_meta = extract_panel_metadata(group[0], idx)
                master_paths.append(group[0])
                master_metas.append(orig_meta)
        else:
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
                # Amortiguar caída medida sobre miniatura con gamma para no sobrecorregir esquinas en float32
                fo = max(0.0, min(0.45, (1.0 - (edge_val / center_val)) * 0.55))
                falloffs.append(fo)
        except Exception:
            continue

    if not falloffs:
        return 0.35
    return float(np.median(falloffs))
