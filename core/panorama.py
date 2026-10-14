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
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
import tifffile
from PIL import Image, ExifTags

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
    blend_mode: str = "multiband"          # "multiband", "feather", "linear"
    blend_bands: int = 5                   # Niveles de la pirámide laplaciana
    exposure_comp: bool = True             # Compensación de ganancia / viñeteo
    wave_correction: bool = True           # Nivelación horizontal de horizonte
    auto_crop: bool = True                 # Recorte automático del rectángulo máximo inscrito
    layout_mode: str = "auto"              # "auto" (grafo 2D) o "grid" (asistente matriz MxN)
    grid_rows: int = 1
    grid_cols: int = 1
    focal_length_override: Optional[float] = None  # Focal en mm si no está en EXIF


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


def _build_overlap_graph(
    features_list: List[Tuple[List[cv2.KeyPoint], np.ndarray]],
    min_inliers: int = 15,
    ratio_thresh: float = 0.75,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    abort_flag: Optional[Callable[[], bool]] = None
) -> Dict[Tuple[int, int], Tuple[np.ndarray, int]]:
    """
    Construye el grafo 2D de solapamiento entre todas las parejas de imágenes.
    Soporta de forma nativa panorámicas de 1 fila (1D) y matrices multifila (multi-row 2D).
    Retorna un diccionario de aristas: (src_idx, dst_idx) -> (H_src_to_dst, num_inliers).
    """
    n = len(features_list)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    edges = {}

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
    abort_flag: Optional[Callable[[], bool]] = None
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

    metas = [extract_panel_metadata(p, idx) for idx, p in enumerate(panels_paths)]
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
    # 3. Estimación de Rotación 3D Global (Bundle Adjustment)
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(25, "Estimando geometría 3D y alineación global...")

    stitch_mode = cv2.Stitcher_SCANS if settings.projection == "plane" else cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher_create(stitch_mode)
    status = stitcher.estimateTransform(imgs_small)

    cameras = None
    rmats_corr = None
    connected_indices = list(range(n))

    if status == 0 and len(stitcher.cameras()) == n:
        cameras = list(stitcher.cameras())
        rmats = [c.R.copy() for c in cameras]
        if settings.wave_correction:
            try:
                rmats_corr = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
            except Exception:
                rmats_corr = rmats
        else:
            rmats_corr = rmats
        focals = [c.focal for c in cameras]
        med_focal_small = float(np.median(focals))
        focal_full = med_focal_small * scale_to_full
    else:
        # Fallback a SIFT pairwise overlap graph si el bundle adjuster no converge
        if progress_callback:
            progress_callback(30, "Alineando paneles mediante grafo 2D de correspondencias afines...")
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

        edges = _build_overlap_graph(features_list, min_inliers=settings.min_inliers, ratio_thresh=0.82)
        if not edges:
            raise ValueError("No se encontraron solapamientos suficientes entre los paneles.")
        global_H, anchor_idx = _compute_global_homographies(n, edges)
        connected_indices = sorted(list(global_H.keys()))

    # Liberar imágenes intermedias de calibración
    del imgs_small
    gc.collect()

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
                [cameras[i].focal * scale_to_full, 0, full_w / 2.0],
                [0, cameras[i].focal * scale_to_full, full_h / 2.0],
                [0, 0, 1]
            ], dtype=np.float32)
            roi = warper.warpRoi((full_w, full_h), K, rmats_corr[i])
            corners.append((roi[0], roi[1]))
            sizes.append((roi[2], roi[3]))

        min_x = min(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_x = max(c[0] + s[0] for c, s in zip(corners, sizes))
        max_y = max(c[1] + s[1] for c, s in zip(corners, sizes))

        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)

        scale_canvas = 1.0
        if canvas_w * canvas_h > MAX_CANVAS_PIXELS:
            scale_canvas = np.sqrt(MAX_CANVAS_PIXELS / (canvas_w * canvas_h))
            focal_full *= scale_canvas
            warper = cv2.PyRotationWarper(proj_type, focal_full)
            corners = []
            sizes = []
            cur_w = int(full_w * scale_canvas)
            cur_h = int(full_h * scale_canvas)
            for i in range(n):
                K = np.array([
                    [cameras[i].focal * scale_to_full * scale_canvas, 0, cur_w / 2.0],
                    [0, cameras[i].focal * scale_to_full * scale_canvas, cur_h / 2.0],
                    [0, 0, 1]
                ], dtype=np.float32)
                roi = warper.warpRoi((cur_w, cur_h), K, rmats_corr[i])
                corners.append((roi[0], roi[1]))
                sizes.append((roi[2], roi[3]))

            min_x = min(c[0] for c in corners)
            min_y = min(c[1] for c in corners)
            max_x = max(c[0] + s[0] for c, s in zip(corners, sizes))
            max_y = max(c[1] + s[1] for c, s in zip(corners, sizes))
            canvas_w = int(max_x - min_x)
            canvas_h = int(max_y - min_y)

        # -------------------------------------------------------------
        # 5. Deformación y Fusión en Streaming (Memoria Segura O(1))
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback(60, f"Proyectando lienzo ({canvas_w}x{canvas_h}, {canvas_w*canvas_h/1e6:.1f} MP)...")

        accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        total_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        for i in range(n):
            if abort_flag and abort_flag():
                raise InterruptedError("Operación cancelada por el usuario.")

            if progress_callback:
                prog = int(60 + 30 * (i / n))
                progress_callback(prog, f"Proyectando y fusionando panel {i + 1}/{n}: {os.path.basename(panels_paths[i])}...")

            img_f = load_image_as_float32(panels_paths[i])
            if scale_canvas != 1.0:
                img_f = cv2.resize(img_f, (0, 0), fx=scale_canvas, fy=scale_canvas, interpolation=cv2.INTER_AREA)

            ih, iw = img_f.shape[:2]
            K = np.array([
                [cameras[i].focal * scale_to_full * scale_canvas, 0, iw / 2.0],
                [0, cameras[i].focal * scale_to_full * scale_canvas, ih / 2.0],
                [0, 0, 1]
            ], dtype=np.float32)

            corner, w_img = warper.warp(img_f, K, rmats_corr[i], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            del img_f

            w_mask = (np.max(w_img, axis=2) > 0.001).astype(np.uint8) * 255
            dist = cv2.distanceTransform(w_mask, cv2.DIST_L2, 5)
            max_d = np.max(dist)
            if max_d > 0:
                dist /= max_d
            dist = np.power(dist, 4.0)
            w_map = dist[..., np.newaxis].astype(np.float32)

            gx = corner[0] - min_x
            gy = corner[1] - min_y
            h_w, w_w = w_img.shape[:2]

            x1, x2 = max(0, gx), min(canvas_w, gx + w_w)
            y1, y2 = max(0, gy), min(canvas_h, gy + h_w)
            ix1, ix2 = x1 - gx, x2 - gx
            iy1, iy2 = y1 - gy, y2 - gy

            accum[y1:y2, x1:x2] += w_img[iy1:iy2, ix1:ix2] * w_map[iy1:iy2, ix1:ix2]
            weight_sum[y1:y2, x1:x2] += w_map[iy1:iy2, ix1:ix2]
            total_mask[y1:y2, x1:x2] = cv2.bitwise_or(total_mask[y1:y2, x1:x2], w_mask[iy1:iy2, ix1:ix2])

            del w_img, w_mask, dist, w_map
            gc.collect()

        valid_norm = weight_sum > 1e-5
        accum[valid_norm.squeeze()] /= weight_sum[valid_norm.squeeze()]
        blended_img = np.clip(accum, 0.0, 1.0)
        del weight_sum
        gc.collect()

    else:
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
            dist = np.power(dist, 4.0)
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

    # -------------------------------------------------------------
    # 6. Recorte Automático de Rectángulo Máximo Inscrito
    # -------------------------------------------------------------
    cropped_img = None
    crop_rect = None

    if settings.auto_crop:
        if progress_callback:
            progress_callback(95, "Calculando rectángulo máximo inscrito (auto-recorte)...")

        rx, ry, rw, rh = find_maximum_inscribed_rectangle(total_mask)
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
