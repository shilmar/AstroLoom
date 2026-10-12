"""
core/eclipse_collage.py - Motor científico y de composición para secuencias y collages de eclipses.

Funcionalidades clave:
  1. Análisis y cribado masivo:
     - Detección de trepidación por viento y nitidez mediante gradiente radial del limbo y varianza laplaciana.
     - Medición matemática exacta del porcentaje de ocultación del disco solar/lunar.
     - Extracción de marcas temporales EXIF y ordenación cronológica.
  2. Recomendador inteligente de fotogramas clave (Keyframe Selector):
     - Selección simétrica automática de tomas en porcentajes diana (ej. 7, 9, 11 o 15 fases).
     - Descarte automático de tomas trepidadas o emborronadas en favor de la toma nítida más cercana.
  3. Normalización y centrado de discos:
     - Centrado geométrico subpíxel a diámetro idéntico en todas las fases.
     - Limpieza de pedestal de fondo y máscara circular suave con canal alfa.
  4. Maquetación espacial:
     - Cinta horizontal simétrica (panorámica / timeline).
     - Arco parabólico de trayectoria celeste.
     - Cuadrícula / matriz (grid poster).
  5. Exportación:
     - Renderizado de imagen plana de 16/32 bits y JPEG.
     - Exportación a archivo PSD nativo de 16 bits por capas (Photoshop).
"""
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
import cv2
import numpy as np

try:
    import pytoshop
    from pytoshop import layers as psd_layers, enums as psd_enums
    HAS_PYTOSHOP = True
except ImportError:
    HAS_PYTOSHOP = False

from core.eclipse import extract_exposure_info, detect_eclipse_disk


@dataclass
class EclipseFrameMetadata:
    """Metadatos de análisis y calidad de una toma de eclipse."""
    filepath: str
    filename: str
    timestamp: float
    time_str: str
    exp_time: float
    iso: int
    center_x: float
    center_y: float
    radius: float
    sharpness_score: float     # 0.0 a 100.0 (mayor es más nítido)
    is_trepidated: bool        # True si se detecta trepidación por viento
    obscuration_pct: float     # 0.0% (sin eclipse) a 100.0% (totalidad)
    is_ingress: bool           # True si es fase creciente (C1-C2), False si es decreciente (C3-C4)
    phase_label: str           # "Parcial Entrada", "Totalidad", "Parcial Salida", etc.
    is_recommended: bool = False


def measure_limb_sharpness_and_shake(
    gray_img: np.ndarray, 
    cx: float, 
    cy: float, 
    radius: float
) -> Tuple[float, bool]:
    """
    Evalúa la nitidez y presencia de trepidación analizando la pendiente del gradiente
    del limbo solar/lunar y la varianza laplaciana de alta frecuencia.
    
    Retorna:
      - sharpness_score: índice de 0.0 a 100.0.
      - is_trepidated: True si la pendiente del limbo es anormalmente suave o asimétrica.
    """
    h, w = gray_img.shape[:2]
    if radius < 10 or cx < 0 or cx >= w or cy < 0 or cy >= h:
        return 0.0, True

    # 1. Recorte ajustado en torno al disco/creciente con margen de seguridad del 35%
    margin = radius * 1.35
    x0 = int(max(0, cx - margin))
    x1 = int(min(w, cx + margin))
    y0 = int(max(0, cy - margin))
    y1 = int(min(h, cy + margin))

    if (x1 - x0) < 15 or (y1 - y0) < 15:
        return 0.0, True

    crop = gray_img[y0:y1, x0:x1]
    max_val = float(crop.max())
    if max_val < 0.01:
        return 0.0, True

    # 2. Gradientes de Sobel para captar la transición en el limbo
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    # 3. Transición de borde superior: seleccionar los píxeles de transición más nítidos
    num_samples = min(1500, max(50, int(crop.size * 0.005)))
    top_grads = np.sort(mag.ravel())[-num_samples:]
    median_top = float(np.median(top_grads))

    # 4. Análisis de trepidación direccional (viento / vibración)
    edge_mask = mag > (median_top * 0.5)
    directional_shake = False
    if np.sum(edge_mask) > 40:
        mean_gx = float(np.mean(np.abs(gx[edge_mask])))
        mean_gy = float(np.mean(np.abs(gy[edge_mask])))
        ratio = max(mean_gx, mean_gy) / max(1e-4, min(mean_gx, mean_gy))
        directional_shake = (ratio > 2.5)

    # 5. Puntuación normalizada 0 - 100
    # En float32 [0.0, 1.0], una transición nítida en 1-2 píxeles da gradiente ~0.35-0.80
    base_score = float(np.clip((median_top / 0.45) * 100.0, 5.0, 100.0))
    if directional_shake:
        base_score *= 0.60

    is_trepidated = directional_shake or (base_score < 45.0)
    return round(base_score, 1), is_trepidated


def calculate_solar_obscuration_pct(
    gray_img: np.ndarray, 
    cx: float, 
    cy: float, 
    radius: float
) -> float:
    """
    Calcula el porcentaje de ocultación del disco solar (0.0% a 100.0%).
    Compara el área luminosa de la media luna visible con el área teórica pi * r^2.
    """
    if radius < 5:
        return 100.0

    h, w = gray_img.shape[:2]
    # Recuadro envolvente del disco
    x0 = int(max(0, cx - radius * 1.25))
    x1 = int(min(w, cx + radius * 1.25))
    y0 = int(max(0, cy - radius * 1.25))
    y1 = int(min(h, cy + radius * 1.25))

    crop = gray_img[y0:y1, x0:x1]
    if crop.size == 0:
        return 100.0

    y_g, x_g = np.mgrid[y0:y1, x0:x1]
    dist_disk = np.hypot(x_g - cx, y_g - cy)
    sun_disc_mask = (dist_disk <= radius)

    inner_pixels = crop[sun_disc_mask]
    if inner_pixels.size == 0:
        return 100.0

    theoretical_pixels = float(np.pi * radius * radius)
    max_val = float(np.percentile(inner_pixels, 99.5))
    if max_val < 0.03:
        # En totalidad completa, la silueta oscura de la Luna tapa todo el disco
        return 100.0

    thresh_val = max(0.015, max_val * 0.20)
    visible_pixels = float(np.sum(inner_pixels > thresh_val))

    # Ratio de sol tapado por la Luna
    covered_ratio = max(0.0, min(1.0, 1.0 - (visible_pixels / max(1.0, theoretical_pixels))))
    return float(covered_ratio * 100.0)


def analyze_eclipse_frame(
    filepath: str, 
    mode: str = "solar", 
    ref_radius: Optional[float] = None
) -> EclipseFrameMetadata:
    """
    Analiza un único fotograma de eclipse:
    Lee EXIF, detecta centro/radio, calcula % de ocultación y mide nitidez/trepidación.
    """
    info = extract_exposure_info(filepath)
    filename = os.path.basename(filepath)
    timestamp = float(info.get("timestamp") or 0.0)
    time_str = str(info.get("time_str") or "--:--:--")
    exp_time = float(info.get("exposure_s") or info.get("exp_time") or 0.001)
    iso = int(info.get("iso") or 100)

    # Carga rápida en escala de grises
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        from core.stacking import load_image_as_float32
        raw_f32 = load_image_as_float32(filepath)
        gray = cv2.cvtColor(raw_f32, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Detección de disco
    cx, cy, r = detect_eclipse_disk(gray, mode=mode, expected_radius=ref_radius)
    if ref_radius is not None and ref_radius > 10:
        r = ref_radius  # Mantener radio unificado del lote si está disponible

    # Medición de nitidez y trepidación
    sharpness, is_trep = measure_limb_sharpness_and_shake(gray, cx, cy, r)

    # Medición de porcentaje de ocultación
    obscuration = calculate_solar_obscuration_pct(gray, cx, cy, r)

    # Clasificación preliminar
    if obscuration >= 98.0:
        phase = "Totalidad / Máximo"
    elif obscuration >= 90.0:
        phase = "Baily / Diamante"
    elif obscuration >= 50.0:
        phase = "Parcial Media"
    else:
        phase = "Parcial Inicial/Final"

    return EclipseFrameMetadata(
        filepath=filepath,
        filename=filename,
        timestamp=timestamp,
        time_str=time_str,
        exp_time=exp_time,
        iso=iso,
        center_x=float(cx),
        center_y=float(cy),
        radius=float(r),
        sharpness_score=sharpness,
        is_trepidated=is_trep,
        obscuration_pct=obscuration,
        is_ingress=True,  # Se recalibra en lote según cronología
        phase_label=phase,
        is_recommended=False
    )


def batch_analyze_eclipse_series(
    filepaths: List[str], 
    mode: str = "solar", 
    callback_progress: Optional[Callable[[int, int, str], None]] = None
) -> List[EclipseFrameMetadata]:
    """
    Analiza un lote completo de tomas de eclipse, las ordena cronológicamente por EXIF,
    identifica el punto de máxima totalidad y clasifica las fases de entrada (ingreso)
    y salida (egreso).
    """
    results: List[EclipseFrameMetadata] = []
    total = len(filepaths)

    # 1. Primera pasada: extraer metadatos de todos los fotogramas
    for i, fp in enumerate(filepaths):
        if callback_progress:
            callback_progress(i + 1, total, f"Analizando {os.path.basename(fp)}...")
        meta = analyze_eclipse_frame(fp, mode=mode)
        results.append(meta)

    # 2. Unificar radio solar/lunar medio para escalas homogéneas
    valid_radii = [m.radius for m in results if 100.0 < m.radius < 800.0]
    median_radius = float(np.median(valid_radii)) if valid_radii else 200.0
    for r in results:
        r.radius = median_radius

    # 3. Ordenación cronológica estricta por EXIF timestamp
    results.sort(key=lambda m: (m.timestamp if m.timestamp > 0 else 0, m.filename))

    # 4. Localizar el instante pico de Totalidad / Máximo
    tot_cands = [m for m in results if m.obscuration_pct >= 95.0]
    if tot_cands:
        t_peak = float(np.median([m.timestamp for m in tot_cands]))
    else:
        max_m = max(results, key=lambda m: m.obscuration_pct)
        t_peak = max_m.timestamp

    # 5. Clasificar entrada vs totalidad vs salida basado en la física temporal del eclipse
    for m in results:
        dt = m.timestamp - t_peak
        if abs(dt) <= 25.0 or m.obscuration_pct >= 98.0:
            m.is_ingress = True
            m.phase_label = f"TOTALIDAD ({m.obscuration_pct:.0f}%)"
        elif dt < 0:
            m.is_ingress = True
            m.phase_label = f"Entrada C1-C2 ({m.obscuration_pct:.0f}%)"
        else:
            m.is_ingress = False
            m.phase_label = f"Salida C3-C4 ({m.obscuration_pct:.0f}%)"

    return results


def recommend_key_frames(
    frames: List[EclipseFrameMetadata], 
    num_targets: int = 9, 
    min_sharpness: float = 50.0
) -> List[EclipseFrameMetadata]:
    """
    Recomendador inteligente de fases:
    1. Agrupa ráfagas de bracketing tomadas en el mismo instante (<= 3 segundos).
    2. En cada ráfaga, selecciona el fotograma óptimo (mayor nitidez sin trepidación).
    3. Distribuye simétricamente las fases objetivo: Entrada, Totalidad central y Salida.
    4. Garantiza ordenación 100% cronológica sin saltos temporales ni fases duplicadas.
    """
    if not frames:
        return []

    # Limpiar recomendaciones previas
    for f in frames:
        f.is_recommended = False

    if len(frames) <= num_targets:
        for f in frames:
            f.is_recommended = True
        return frames

    # 1. Agrupar en eventos / ráfagas temporales (<= 3.0 segundos)
    bursts: List[List[EclipseFrameMetadata]] = []
    current_burst = [frames[0]]
    for f in frames[1:]:
        if abs(f.timestamp - current_burst[-1].timestamp) <= 3.0:
            current_burst.append(f)
        else:
            bursts.append(current_burst)
            current_burst = [f]
    bursts.append(current_burst)

    # 2. Para cada ráfaga, elegir el mejor fotograma representativo
    def pick_best_in_burst(burst_list: List[EclipseFrameMetadata]) -> EclipseFrameMetadata:
        def burst_score(m: EclipseFrameMetadata) -> float:
            score = m.sharpness_score
            if m.is_trepidated:
                score -= 40.0
            if 0.0003 <= m.exp_time <= 0.05:
                score += 10.0
            return score
        return max(burst_list, key=burst_score)

    burst_reps = [pick_best_in_burst(b) for b in bursts]

    # 3. Localizar el instante central de Totalidad
    tot_reps = [m for m in burst_reps if "TOTALIDAD" in m.phase_label or m.obscuration_pct >= 95.0]
    if tot_reps:
        t_peak = float(np.median([m.timestamp for m in tot_reps]))
        tot_best = max(tot_reps, key=lambda m: (not m.is_trepidated, m.sharpness_score))
    else:
        tot_best = max(burst_reps, key=lambda m: m.obscuration_pct)
        t_peak = tot_best.timestamp

    # 4. Dividir las ráfagas en Entrada, Totalidad y Salida
    ingress_reps = [m for m in burst_reps if m.timestamp < t_peak - 20.0 and m != tot_best]
    egress_reps = [m for m in burst_reps if m.timestamp > t_peak + 20.0 and m != tot_best]

    selected: List[EclipseFrameMetadata] = [tot_best]
    rem = num_targets - 1
    flank_in = rem // 2
    flank_out = rem - flank_in

    # Selección en Entrada: distribución equilibrada en tiempo / progresión de eclipse
    if ingress_reps and flank_in > 0:
        target_obscs = np.linspace(15.0, 92.0, flank_in)
        used_fps = {tot_best.filepath}
        for tobsc in target_obscs:
            best_m = None
            best_cost = 999999.0
            for m in ingress_reps:
                if m.filepath in used_fps:
                    continue
                cost = abs(m.obscuration_pct - tobsc) * 2.0 - (m.sharpness_score * 0.4) + (35.0 if m.is_trepidated else 0.0)
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            if best_m is not None:
                used_fps.add(best_m.filepath)
                selected.append(best_m)

    # Selección en Salida: distribución equilibrada en tiempo / progresión de eclipse
    if egress_reps and flank_out > 0:
        target_obscs = np.linspace(92.0, 15.0, flank_out)
        used_fps = {s.filepath for s in selected}
        for tobsc in target_obscs:
            best_m = None
            best_cost = 999999.0
            for m in egress_reps:
                if m.filepath in used_fps:
                    continue
                cost = abs(m.obscuration_pct - tobsc) * 2.0 - (m.sharpness_score * 0.4) + (35.0 if m.is_trepidated else 0.0)
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            if best_m is not None:
                used_fps.add(best_m.filepath)
                selected.append(best_m)

    # Si faltan fotogramas para completar num_targets
    if len(selected) < num_targets:
        selected_fps = {s.filepath for s in selected}
        remaining_cands = [m for m in burst_reps if m.filepath not in selected_fps]
        remaining_cands.sort(key=lambda m: (m.is_trepidated, -m.sharpness_score))
        needed = num_targets - len(selected)
        selected.extend(remaining_cands[:needed])

    for m in selected:
        m.is_recommended = True

    # Garantizar ordenación cronológica estricta de la selección
    selected.sort(key=lambda m: (m.timestamp if m.timestamp > 0 else 0, m.filename))
    return selected


# -------------------------------------------------------------------------
# Centrado y Extracción Normalizada de Discos
# -------------------------------------------------------------------------
def extract_normalized_disk(
    image_rgb: np.ndarray, 
    cx: float, 
    cy: float, 
    radius: float, 
    target_radius: int = 150, 
    margin_factor: float = 1.35
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recorta y centra con precisión subpíxel el disco solar/lunar en un lienzo cuadrado
    normalizado a un radio objetivo uniforme.
    
    Retorna:
      - cropped_rgb: imagen RGB recortada float32 [0.0, 1.0].
      - alpha_mask: máscara suave float32 [0.0, 1.0] con fundido circular en el borde.
    """
    h, w = image_rgb.shape[:2]
    tile_size = int(target_radius * 2 * margin_factor)
    if tile_size % 2 != 0:
        tile_size += 1

    tile_center = tile_size / 2.0
    scale = target_radius / max(1.0, radius)

    # Matriz afín de transformación:
    # 1. Trasladar centro de la foto (cx, cy) al origen (0, 0)
    # 2. Escalar al radio objetivo
    # 3. Trasladar al centro del tile (tile_center, tile_center)
    M = np.array([
        [scale, 0.0, tile_center - cx * scale],
        [0.0, scale, tile_center - cy * scale]
    ], dtype=np.float32)

    cropped_rgb = cv2.warpAffine(
        image_rgb, 
        M, 
        (tile_size, tile_size), 
        flags=cv2.INTER_LANCZOS4, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )

    # Generar máscara alfa circular suave
    y_g, x_g = np.mgrid[:tile_size, :tile_size]
    dist = np.hypot(x_g - tile_center, y_g - tile_center)

    # Radio de corte y ancho de suavizado
    r_cut = target_radius * 1.15
    feather = max(3.0, target_radius * 0.10)
    
    alpha_mask = np.clip((r_cut + feather - dist) / feather, 0.0, 1.0).astype(np.float32)

    return cropped_rgb, alpha_mask


# -------------------------------------------------------------------------
# Maquetación Espacial (Layout Engine)
# -------------------------------------------------------------------------
LAYOUT_HORIZONTAL = "horizontal"
LAYOUT_ARC = "arc"
LAYOUT_GRID = "grid"

@dataclass
class LayerPlacement:
    """Información de posicionamiento para una capa en el collage."""
    frame_meta: EclipseFrameMetadata
    top: int
    left: int
    width: int
    height: int
    scale: float
    rgb_tile: np.ndarray
    alpha_tile: np.ndarray


def compute_collage_layout(
    selected_frames: List[EclipseFrameMetadata],
    layout_type: str = LAYOUT_HORIZONTAL,
    canvas_width: int = 3840,
    target_disk_radius: int = 140,
    gap_spacing: int = 40,
    totality_scale: float = 1.30,
    margin_padding: int = 60
) -> Tuple[int, int, List[LayerPlacement]]:
    """
    Calcula las dimensiones del lienzo global y las coordenadas (top, left)
    de cada fase según la modalidad seleccionada.
    
    Retorna:
      - (total_canvas_w, total_canvas_h, placements)
    """
    from core.stacking import load_image_as_float32
    placements: List[LayerPlacement] = []
    n = len(selected_frames)
    if n == 0:
        return canvas_width, 1080, []

    # Extraer y preparar los tiles recortados
    prepared_tiles = []
    for f in selected_frames:
        img_rgb = load_image_as_float32(f.filepath)
        is_tot = (f.obscuration_pct >= 95.0)
        scale_mod = totality_scale if is_tot else 1.0
        eff_radius = int(target_disk_radius * scale_mod)
        
        # Para totalidad usamos un margen mayor para respetar los filamentos de la corona
        m_factor = 1.90 if is_tot else 1.35
        rgb_tile, alpha_tile = extract_normalized_disk(
            img_rgb, f.center_x, f.center_y, f.radius, target_radius=eff_radius, margin_factor=m_factor
        )
        prepared_tiles.append((f, rgb_tile, alpha_tile, scale_mod))

    if layout_type == LAYOUT_HORIZONTAL:
        # Cinta horizontal secuencial
        total_w = margin_padding * 2
        max_h = 0
        
        for _, tile, _, _ in prepared_tiles:
            th, tw = tile.shape[:2]
            total_w += tw + gap_spacing
            if th > max_h:
                max_h = th
                
        total_w -= gap_spacing  # Restar último gap sobrante
        canvas_w = max(canvas_width, total_w)
        canvas_h = max_h + margin_padding * 2

        # Centrar el bloque completo en X
        start_x = (canvas_w - (total_w - margin_padding * 2)) // 2
        curr_x = start_x

        for f, tile, alpha, sc in prepared_tiles:
            th, tw = tile.shape[:2]
            y_pos = (canvas_h - th) // 2
            placements.append(LayerPlacement(
                frame_meta=f,
                top=int(y_pos),
                left=int(curr_x),
                width=int(tw),
                height=int(th),
                scale=float(sc),
                rgb_tile=tile,
                alpha_tile=alpha
            ))
            curr_x += tw + gap_spacing

        return canvas_w, canvas_h, placements

    elif layout_type == LAYOUT_ARC:
        # Arco parabólico (cenit en el centro)
        canvas_w = max(canvas_width, 3840)
        canvas_h = int(canvas_w * 0.45)
        
        usable_w = canvas_w - margin_padding * 2
        center_x = canvas_w / 2.0
        zenith_y = margin_padding + target_disk_radius * 1.5
        base_y = canvas_h - margin_padding - target_disk_radius * 1.8

        for idx, (f, tile, alpha, sc) in enumerate(prepared_tiles):
            t = (idx / float(max(1, n - 1))) * 2.0 - 1.0  # -1.0 a +1.0
            # Ecuación parabólica: y = zenith + (base - zenith) * t^2
            x_center = center_x + t * (usable_w / 2.0)
            y_center = zenith_y + (base_y - zenith_y) * (t ** 2)

            th, tw = tile.shape[:2]
            top = int(y_center - th / 2.0)
            left = int(x_center - tw / 2.0)

            placements.append(LayerPlacement(
                frame_meta=f,
                top=top,
                left=left,
                width=int(tw),
                height=int(th),
                scale=float(sc),
                rgb_tile=tile,
                alpha_tile=alpha
            ))

        return canvas_w, canvas_h, placements

    else:
        # Cuadrícula / Grid (e.g. columnas calculadas)
        cols = int(math.ceil(math.sqrt(n * (16.0 / 9.0))))
        cols = max(3, min(cols, 6))
        rows = int(math.ceil(n / float(cols)))

        max_tile_w = max(t[1].shape[1] for t in prepared_tiles)
        max_tile_h = max(t[1].shape[0] for t in prepared_tiles)

        cell_w = max_tile_w + gap_spacing
        cell_h = max_tile_h + gap_spacing

        canvas_w = margin_padding * 2 + cols * cell_w
        canvas_h = margin_padding * 2 + rows * cell_h

        for idx, (f, tile, alpha, sc) in enumerate(prepared_tiles):
            r_idx = idx // cols
            c_idx = idx % cols
            
            th, tw = tile.shape[:2]
            cell_center_x = margin_padding + c_idx * cell_w + cell_w // 2
            cell_center_y = margin_padding + r_idx * cell_h + cell_h // 2

            top = int(cell_center_y - th // 2)
            left = int(cell_center_x - tw // 2)

            placements.append(LayerPlacement(
                frame_meta=f,
                top=top,
                left=left,
                width=int(tw),
                height=int(th),
                scale=float(sc),
                rgb_tile=tile,
                alpha_tile=alpha
            ))

        return canvas_w, canvas_h, placements


# -------------------------------------------------------------------------
# Renderizado de Imagen Plana y Exportación PSD por Capas
# -------------------------------------------------------------------------
def render_flat_collage(
    canvas_w: int, 
    canvas_h: int, 
    placements: List[LayerPlacement],
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    draw_captions: bool = True
) -> np.ndarray:
    """
    Genera el lienzo final plano combinando todas las fases con mezcla alfa.
    Opcionalmente imprime marcas horarias debajo de cada disco en tipografía limpia.
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas[..., 0] = bg_color[0]
    canvas[..., 1] = bg_color[1]
    canvas[..., 2] = bg_color[2]

    for p in placements:
        t = max(0, p.top)
        b = min(canvas_h, p.top + p.height)
        l = max(0, p.left)
        r = min(canvas_w, p.left + p.width)

        src_t = t - p.top
        src_b = src_t + (b - t)
        src_l = l - p.left
        src_r = src_l + (r - l)

        tile_crop = p.rgb_tile[src_t:src_b, src_l:src_r]
        alpha_crop = p.alpha_tile[src_t:src_b, src_l:src_r, np.newaxis]

        canvas[t:b, l:r] = canvas[t:b, l:r] * (1.0 - alpha_crop) + tile_crop * alpha_crop

        if draw_captions:
            text = f"{p.frame_meta.time_str}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = int(p.left + (p.width - tw) // 2)
            ty = int(min(canvas_h - 15, p.top + p.height + 20))
            if 0 <= tx < canvas_w and 0 <= ty < canvas_h:
                cv2.putText(canvas, text, (tx, ty), font, font_scale, (0.75, 0.75, 0.75), thickness, cv2.LINE_AA)

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def export_layered_psd(
    filepath: str,
    canvas_w: int,
    canvas_h: int,
    placements: List[LayerPlacement],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    depth: int = 16
) -> bool:
    """
    Exporta el collage como archivo PSD nativo de Photoshop con cada fase en su propia capa.
    Cada capa incluye su nombre descriptivo, posición exacta (top, left) y transparencia.
    """
    if not HAS_PYTOSHOP:
        return False

    psd = pytoshop.core.PsdFile(num_channels=3, height=canvas_h, width=canvas_w, depth=depth)

    # 1. Capa de Fondo
    bg_layer = psd_layers.LayerRecord(
        top=0,
        bottom=canvas_h,
        left=0,
        right=canvas_w,
        name="Fondo Negro",
        opacity=255,
        blend_mode=psd_enums.BlendMode.normal
    )

    if depth == 16:
        max_v = 65535
        dtype = np.uint16
    else:
        max_v = 255
        dtype = np.uint8

    # Canal sólido de fondo
    bg_r = np.full((canvas_h, canvas_w), int(bg_color[0] * (max_v / 255.0)), dtype=dtype)
    bg_g = np.full((canvas_h, canvas_w), int(bg_color[1] * (max_v / 255.0)), dtype=dtype)
    bg_b = np.full((canvas_h, canvas_w), int(bg_color[2] * (max_v / 255.0)), dtype=dtype)

    bg_layer.channels = {
        0: psd_layers.ChannelImageData(image=bg_r, compression=psd_enums.Compression.raw),
        1: psd_layers.ChannelImageData(image=bg_g, compression=psd_enums.Compression.raw),
        2: psd_layers.ChannelImageData(image=bg_b, compression=psd_enums.Compression.raw)
    }
    psd.layer_and_mask_info.layer_info.layer_records.append(bg_layer)

    # 2. Capas de cada fase del eclipse
    for idx, p in enumerate(placements):
        layer_name = f"Fase {idx+1:02d} - {p.frame_meta.phase_label} ({p.frame_meta.time_str})"
        
        # Coordenadas acotadas al lienzo
        top = int(max(0, p.top))
        bottom = int(min(canvas_h, p.top + p.height))
        left = int(max(0, p.left))
        right = int(min(canvas_w, p.left + p.width))

        crop_h = bottom - top
        crop_w = right - left
        if crop_h <= 0 or crop_w <= 0:
            continue

        src_t = top - p.top
        src_b = src_t + crop_h
        src_l = left - p.left
        src_r = src_l + crop_w

        tile_rgb = np.clip(p.rgb_tile[src_t:src_b, src_l:src_r], 0.0, 1.0)
        tile_alpha = np.clip(p.alpha_tile[src_t:src_b, src_l:src_r], 0.0, 1.0)

        # Multiplicar canales RGB por la máscara alfa suave para garantizar que el fondo exterior
        # sea matemáticamente 0 (negro puro), eliminando el ruido del sensor y evitando el desbordamiento uint16
        tile_rgb_masked = tile_rgb * tile_alpha[..., np.newaxis]

        if p.frame_meta.obscuration_pct < 95.0:
            # En fases parciales, suprimir ruido térmico residual del cielo negro
            tile_rgb_masked[tile_rgb_masked < 0.008] = 0.0

        ch_r = (np.clip(tile_rgb_masked[..., 0], 0.0, 1.0) * max_v).astype(dtype)
        ch_g = (np.clip(tile_rgb_masked[..., 1], 0.0, 1.0) * max_v).astype(dtype)
        ch_b = (np.clip(tile_rgb_masked[..., 2], 0.0, 1.0) * max_v).astype(dtype)
        ch_a = (tile_alpha * max_v).astype(dtype)

        # Si es la totalidad, poner en modo Trama (Screen) para fundir la corona suavemente
        blend = psd_enums.BlendMode.screen if p.frame_meta.obscuration_pct >= 95.0 else psd_enums.BlendMode.normal

        rec = psd_layers.LayerRecord(
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            name=layer_name,
            opacity=255,
            blend_mode=blend
        )
        rec.channels = {
            0: psd_layers.ChannelImageData(image=ch_r, compression=psd_enums.Compression.raw),
            1: psd_layers.ChannelImageData(image=ch_g, compression=psd_enums.Compression.raw),
            2: psd_layers.ChannelImageData(image=ch_b, compression=psd_enums.Compression.raw),
            -1: psd_layers.ChannelImageData(image=ch_a, compression=psd_enums.Compression.raw)
        }
        psd.layer_and_mask_info.layer_info.layer_records.append(rec)

    with open(filepath, "wb") as f:
        psd.write(f)

    return True
