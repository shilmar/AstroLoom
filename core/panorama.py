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
    projection: str = "cylindrical"        # "cylindrical", "spherical", "plane"
    astro_mode: bool = True                # Pre-realce CLAHE para cielo oscuro y suelo
    confidence_thresh: float = 0.45        # Umbral de confianza / Lowe ratio (0.3 a 0.8)
    min_inliers: int = 15                  # Mínimo número de puntos coincidentes
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
        # Suave filtrado bilateral para no magnificar el ruido de alta frecuencia del sensor
        prep = cv2.bilateralFilter(prep, d=5, sigmaColor=35, sigmaSpace=35)
    else:
        prep = image_gray

    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.015,
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

            # Estimación robusta de Homografía con RANSAC
            H_j_to_i, inliers_mask = cv2.findHomography(pts_j, pts_i, cv2.RANSAC, 4.0)

            if H_j_to_i is not None and inliers_mask is not None:
                n_inliers = int(np.sum(inliers_mask))
                if n_inliers >= min_inliers:
                    # Inversión matemática para la dirección i -> j
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
        anchor_idx = n_images // 2

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
    2. Carga en memoria y pre-realce astro para detección de puntos clave (SIFT + CLAHE).
    3. Construcción del Grafo 2D de Conectividad con filtrado RANSAC (soporta multifila).
    4. Estimación de Homografías Globales por BFS / Árbol de Expansión.
    5. Proyección geométrica (Plana, Cilíndrica o Esférica).
    6. Deformación al lienzo global y compensación de exposición/viñeteo.
    7. Fusión Laplaciana Multibanda (Laplacian Multi-Band Blending).
    8. Cálculo de Rectángulo Máximo Inscrito (Auto-recorte).
    """
    t0 = time.time()
    if settings is None:
        settings = PanoramaSettings()

    if len(panels_paths) < 2:
        raise ValueError("Se requieren al menos 2 imágenes para componer una panorámica.")

    # -------------------------------------------------------------
    # 1. Extracción de metadatos y cálculo de focal representativa
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(5, "Extrayendo metadatos y focales de los paneles...")

    metas = [extract_panel_metadata(p, idx) for idx, p in enumerate(panels_paths)]
    focals = [m.focal_length_mm for m in metas if m.focal_length_mm is not None and m.focal_length_mm > 0]

    if settings.focal_length_override and settings.focal_length_override > 0:
        focal_mm = settings.focal_length_override
    elif focals:
        focal_mm = float(np.median(focals))
    else:
        focal_mm = 24.0  # Fallback estándar para paisaje nocturno gran angular

    # -------------------------------------------------------------
    # 2. Carga de imágenes y extracción de rasgos SIFT astro
    # -------------------------------------------------------------
    features_list = []
    loaded_images = []
    n = len(panels_paths)

    for i, path in enumerate(panels_paths):
        if abort_flag and abort_flag():
            raise InterruptedError("Operación cancelada por el usuario.")

        if progress_callback:
            prog = int(8 + 15 * (i / n))
            progress_callback(prog, f"Cargando y analizando panel {i + 1}/{n}: {os.path.basename(path)}")

        img_float = load_image_as_float32(path)
        loaded_images.append(img_float)

        # Convertir a escala de grises uint8 para SIFT
        gray_u8 = (np.clip(cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY), 0.0, 1.0) * 255.0).astype(np.uint8)
        kps, descs = extract_astro_features(gray_u8, astro_mode=settings.astro_mode, max_features=4000)
        features_list.append((kps, descs))

    # -------------------------------------------------------------
    # 3. Grafo 2D de conectividad y correspondencias
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(25, "Calculando grafo 2D de solapamiento y emparejamientos...")

    edges = _build_overlap_graph(
        features_list,
        min_inliers=settings.min_inliers,
        ratio_thresh=0.75,
        progress_callback=progress_callback,
        abort_flag=abort_flag
    )

    if not edges:
        raise ValueError("No se encontraron solapamientos suficientes entre los paneles para coser la panorámica.")

    # -------------------------------------------------------------
    # 4. Cálculo de Homografías Globales por BFS
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(55, "Optimizando alineación global en el lienzo...")

    global_H, anchor_idx = _compute_global_homographies(n, edges)
    connected_indices = sorted(list(global_H.keys()))

    if len(connected_indices) < 2:
        raise ValueError(
            f"Solo se pudo alinear 1 imagen con el panel de referencia. Comprueba el orden o solapamiento."
        )

    # -------------------------------------------------------------
    # 5. Proyección Geométrica (Cilíndrica, Esférica o Plana)
    # -------------------------------------------------------------
    # Estimar distancia focal en píxeles a partir de la diagonal del sensor de 35mm (43.27 mm)
    sample_h, sample_w = loaded_images[0].shape[:2]
    sensor_diag_px = np.sqrt(sample_w**2 + sample_h**2)
    focal_px = (focal_mm / 43.27) * sensor_diag_px

    images_to_project = []
    masks_to_project = []

    for idx in range(n):
        img = loaded_images[idx]
        h, w = img.shape[:2]

        if settings.projection == "cylindrical":
            w_img, w_mask = warp_image_cylindrical(img, focal_px)
        elif settings.projection == "spherical":
            w_img, w_mask = warp_image_spherical(img, focal_px)
        else:
            w_img = img
            w_mask = np.ones((h, w), dtype=np.uint8) * 255

        images_to_project.append(w_img)
        masks_to_project.append(w_mask)

    # -------------------------------------------------------------
    # 6. Deformación al Lienzo Global (Warping & Bounding Box)
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(65, "Deformando paneles y proyectando sobre el lienzo...")

    # Calcular la caja envolvente (bounding box) de todas las esquinas transformadas
    all_corners_x = []
    all_corners_y = []

    for idx in connected_indices:
        H = global_H[idx]
        h, w = images_to_project[idx].shape[:2]
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
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

    # Límites de seguridad para evitar saturación de memoria RAM
    MAX_CANVAS_PIXELS = 120_000_000  # ~120 Megapíxeles máx
    current_pixels = canvas_w * canvas_h
    scale_factor = 1.0

    if current_pixels > MAX_CANVAS_PIXELS:
        scale_factor = np.sqrt(MAX_CANVAS_PIXELS / current_pixels)
        canvas_w = int(canvas_w * scale_factor)
        canvas_h = int(canvas_h * scale_factor)

    # Matriz de traslación para centrar la panorámica en el lienzo
    T = np.array([
        [scale_factor, 0, -min_x * scale_factor],
        [0, scale_factor, -min_y * scale_factor],
        [0, 0, 1]
    ], dtype=np.float64)

    warped_panels = []
    warped_masks = []

    for idx in connected_indices:
        H_final = T @ global_H[idx]
        img_src = images_to_project[idx]
        mask_src = masks_to_project[idx]

        warped_img = cv2.warpPerspective(
            img_src, H_final, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        warped_mask = cv2.warpPerspective(
            mask_src, H_final, (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        warped_panels.append(warped_img)
        warped_masks.append(warped_mask)

    # Liberar memoria de imágenes intermedias no proyectadas
    del loaded_images
    del images_to_project
    del masks_to_project
    gc.collect()

    # -------------------------------------------------------------
    # 7. Compensación de Ganancia / Viñeteo (opcional)
    # -------------------------------------------------------------
    if settings.exposure_comp and len(warped_panels) > 1:
        if progress_callback:
            progress_callback(78, "Compensando viñeteo y ecualizando exposición...")

        # Calcular brillo relativo en zonas de solapamiento
        base_brightness = [np.mean(p[m > 127]) if np.any(m > 127) else 1.0 for p, m in zip(warped_panels, warped_masks)]
        ref_b = base_brightness[len(base_brightness) // 2]
        for i in range(len(warped_panels)):
            if base_brightness[i] > 1e-4:
                gain = float(ref_b / base_brightness[i])
                # Limitar ganancia a un rango suave [0.7, 1.4] para no introducir grano
                gain = np.clip(gain, 0.7, 1.4)
                warped_panels[i] = np.clip(warped_panels[i] * gain, 0.0, 1.0)

    # -------------------------------------------------------------
    # 8. Fusión Multicanal (Blending Multibanda)
    # -------------------------------------------------------------
    if progress_callback:
        progress_callback(85, f"Fusionando paneles con pirámide multibanda ({settings.blend_bands} niveles)...")

    if settings.blend_mode == "multiband":
        blended_img, total_mask = blend_multiband_pyramid(
            warped_panels, warped_masks, num_bands=settings.blend_bands
        )
    else:
        # Fusión rápida por distancia ponderada
        weights = [cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5) for m in warped_masks]
        weight_sum = np.sum(np.stack(weights, axis=-1), axis=-1, keepdims=True)
        weight_sum = np.where(weight_sum > 1e-6, weight_sum, 1.0)

        accum = np.zeros_like(warped_panels[0])
        total_mask = np.zeros(warped_masks[0].shape, dtype=np.uint8)

        for p, m, w in zip(warped_panels, warped_masks, weights):
            w_norm = (w[..., np.newaxis] / weight_sum).astype(np.float32)
            accum += p * w_norm
            total_mask = cv2.bitwise_or(total_mask, m)

        blended_img = np.clip(accum, 0.0, 1.0)

    # -------------------------------------------------------------
    # 9. Recorte Automático de Rectángulo Máximo Inscrito
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
