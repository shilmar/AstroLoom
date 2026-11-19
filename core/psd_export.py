# core/psd_export.py
"""
core/psd_export.py - Exportador de archivos PSD nativos (Adobe Photoshop) multicapa de 16 y 8 bits.
Permite exportar composiciones astrofotográficas con separación nativa no destructiva de:
- 00 - Revelado Final AstroLoom (Compuesto de Referencia, oculto por defecto para inspección interactiva)
- 04 - Suelo / Paisaje (con Máscara de Capa nativa de Photoshop vinculada, blanco=suelo, negro=cielo)
- 03 - Estrellas desacopladas (con modo Linear Dodge / Add para recomposición matemática perfecta)
- 02 - Fondo Starless / Nebuloso (estirado y contrastado con máscara de cielo nativa)
- 01 - Imagen Base Revelada (fondo de seguridad con revelado completo)
- Capas de ajuste locales con sus respectivas máscaras de pincel como Máscaras de Capa vinculadas

Acelerado por Rust (astroloom_rs) con procesamiento SIMD multihilo Rayon.
Elimina por completo la dependencia externa de `pytoshop`.
"""
import os
from typing import Optional, List, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

try:
    import astroloom_rs
    HAS_PSD_ENGINE = hasattr(astroloom_rs, "write_psd_file")
except Exception:
    HAS_PSD_ENGINE = False

# Compatibilidad con el resto del código que verificaba HAS_PYTOSHOP
HAS_PYTOSHOP = HAS_PSD_ENGINE


def export_developer_psd(
    filepath: str,
    composite_image: np.ndarray,
    base_image: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None,
    sky_mask: Optional[np.ndarray] = None,
    ground_image: Optional[np.ndarray] = None,
    adjustment_layers: Optional[List[Any]] = None,
    depth: int = 16,
    icc_profile_bytes: Optional[bytes] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_fn: Optional[Callable[[int, str], None]] = None
) -> Tuple[bool, str, int]:
    """
    Exporta el revelado completo a un archivo PSD nativo de Adobe Photoshop por capas con máscaras nativas.

    :param filepath: Ruta absoluta de destino (.psd).
    :param composite_image: Imagen procesada final (Float32 RGB [0.0, 1.0]).
    :param base_image: Imagen base estirada/desarrollada completa.
    :param starless_image: Fondo sin estrellas estirado (StarNet), opcional.
    :param stars_image: Imagen de estrellas puras estiradas (StarNet), opcional.
    :param sky_mask: Máscara 2D de cielo (1.0 = cielo, 0.0 = suelo), opcional.
    :param ground_image: Imagen desarrollada con ajustes específicos para el suelo, opcional.
    :param adjustment_layers: Lista opcional de MaskLayer con máscaras locales.
    :param depth: Profundidad de color por canal (16 para uint16, 8 para uint8).
    :param icc_profile_bytes: Bytes del perfil ICC para incrustar en el PSD.
    :param log_fn: Callback opcional para registrar mensajes informativos.
    :param progress_fn: Callback opcional (pct, texto) para notificar el avance del guardado.
    :return: (éxito: bool, mensaje_descriptivo: str, total_capas: int)
    """
    def _log(msg: str):
        if log_fn:
            log_fn(msg)

    def _progress(pct: int, msg: str):
        if progress_fn:
            progress_fn(pct, msg)

    if not HAS_PSD_ENGINE:
        err = "El motor nativo de exportación PSD no está disponible."
        _log(f"[ERROR PSD] {err}")
        return False, err, 0

    if composite_image is None or composite_image.size == 0:
        return False, "La imagen compuesta está vacía.", 0

    h, w = composite_image.shape[:2]
    if depth not in (8, 16):
        depth = 16

    _log(f"[PSD] Creando estructura PSD multicapa ({w}x{h} px, {depth}-bit) con motor Rust...")
    _progress(70, "Estructurando capas PSD...")

    # Preparar máscara de horizonte limpia si existe
    clean_mask = None
    if sky_mask is not None:
        clean_mask = np.squeeze(sky_mask)
        if clean_mask.ndim > 2:
            clean_mask = clean_mask[..., 0]
        clean_mask = clean_mask.astype(np.float32)
        if clean_mask.shape[:2] != (h, w):
            clean_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        clean_mask = np.clip(clean_mask, 0.0, 1.0)

    def _format_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.repeat(arr[..., np.newaxis], 3, axis=2)
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(np.clip(arr, 0.0, 1.0), dtype=np.float32)

    def _format_mask(m: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if m is None:
            return None
        if m.ndim > 2:
            m = m[..., 0]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(np.clip(m, 0.0, 1.0), dtype=np.float32)

    layer_records = []

    # 1. Capa Base / Fondo Revelado (Fondo de la pila en Photoshop)
    source_base = base_image if base_image is not None else composite_image
    base_name = "01 - Imagen Base Revelada" if (starless_image is not None or clean_mask is not None) else "01 - Imagen Revelada"
    layer_records.append({
        "name": base_name,
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "norm",
        "opacity": 255,
        "visible": True,
        "rgb": _format_rgb(source_base)
    })

    # 2. Capa Fondo Nebular (Starless AI) si está disponible (con máscara de cielo nativa vinculada si existe)
    if starless_image is not None:
        layer_records.append({
            "name": "02 - Fondo Starless",
            "top": 0, "left": 0, "bottom": h, "right": w,
            "blend_mode": "norm",
            "opacity": 255,
            "visible": True,
            "rgb": _format_rgb(starless_image),
            "mask": _format_mask(clean_mask)
        })

    # 3. Capa de Estrellas (Stars) con modo Trama (Screen) y máscara de cielo nativa vinculada
    if stars_image is not None:
        layer_records.append({
            "name": "03 - Estrellas (Trama / Screen)",
            "top": 0, "left": 0, "bottom": h, "right": w,
            "blend_mode": "scrn",
            "opacity": 255,
            "visible": True,
            "rgb": _format_rgb(stars_image),
            "mask": _format_mask(clean_mask)
        })

    # 4. Capa Suelo / Terreno con Máscara de Capa nativa vinculada (Suelo=Blanco, Cielo=Negro)
    if clean_mask is not None:
        ground_source = ground_image if ground_image is not None else source_base
        gnd_layer_name = "04 - Suelo / Paisaje" if starless_image is not None else "02 - Suelo / Paisaje"
        ground_mask = np.clip(1.0 - clean_mask, 0.0, 1.0)
        layer_records.append({
            "name": gnd_layer_name,
            "top": 0, "left": 0, "bottom": h, "right": w,
            "blend_mode": "norm",
            "opacity": 255,
            "visible": True,
            "rgb": _format_rgb(ground_source),
            "mask": _format_mask(ground_mask)
        })

    # 5. Capas de ajuste locales o capas ráster de usuario adicionales
    if adjustment_layers:
        user_layers = [
            l for l in adjustment_layers 
            if not getattr(l, 'is_system_layer', False)
        ]
        n_user = len(user_layers)

        def _build_adj_record(idx_adj):
            idx, adj = idx_adj
            is_raster = getattr(adj, 'mask_type', None) == 'image_raster' or getattr(adj, 'layer_kind', None) == 'image'
            bmask = getattr(adj, 'brush_mask', None)
            if bmask is None:
                bmask = getattr(adj, 'custom_mask', None)

            if is_raster and getattr(adj, 'image_data', None) is not None:
                raw_img = getattr(adj, 'image_data')
                if raw_img.ndim == 2:
                    raw_img = np.repeat(raw_img[..., np.newaxis], 3, axis=-1)
                elif raw_img.ndim == 3 and raw_img.shape[2] == 1:
                    raw_img = np.repeat(raw_img, 3, axis=-1)
                if raw_img.shape[:2] != (h, w):
                    raw_img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_AREA)

                # Aplicar transformación espacial 2D (offset, escala X/Y, rotación) si no es identidad
                if not getattr(adj, 'is_transform_identity', lambda: True)():
                    scale_x = max(0.01, float(getattr(adj, 'scale_x', getattr(adj, 'scale', 1.0))))
                    scale_y = max(0.01, float(getattr(adj, 'scale_y', getattr(adj, 'scale', 1.0))))
                    angle_val = float(getattr(adj, 'angle', 0.0))
                    h_r, w_r = raw_img.shape[:2]
                    theta = np.deg2rad(angle_val)
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    a = scale_x * cos_t
                    b = -scale_y * sin_t
                    c = scale_x * sin_t
                    d = scale_y * cos_t
                    cx, cy = w / 2.0, h / 2.0
                    tx = float(getattr(adj, 'offset_x', 0.0)) * w
                    ty = float(getattr(adj, 'offset_y', 0.0)) * h
                    M = np.array([
                        [a * (w / float(w_r)), b * (h / float(h_r)), cx - a * cx - b * cy + tx],
                        [c * (w / float(w_r)), d * (h / float(h_r)), cy - c * cx - d * cy + ty]
                    ], dtype=np.float32)
                    raw_img = cv2.warpAffine(
                        raw_img, M, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0.0, 0.0, 0.0)
                    )
                    pts = np.array([[0.0, 0.0], [float(w_r), 0.0], [float(w_r), float(h_r)], [0.0, float(h_r)]], dtype=np.float32)
                    pts_dst = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
                    valid_alpha = np.zeros((h, w), dtype=np.float32)
                    cv2.fillConvexPoly(valid_alpha, np.int32(np.round(pts_dst)), 1.0)
                    if bmask is not None:
                        bmask = bmask * valid_alpha
                    else:
                        bmask = valid_alpha

                b_mode = getattr(adj, 'blend_mode', 'normal')
                psd_mode = {
                    "normal": "norm",
                    "screen": "scrn",
                    "luminosity": "lum ",
                    "soft_light": "sLit",
                    "overlay": "over",
                    "multiply": "mul "
                }.get(b_mode, "norm")

                return idx, {
                    "name": f"Ráster - {getattr(adj, 'name', f'Capa {idx+1}')}",
                    "top": 0, "left": 0, "bottom": h, "right": w,
                    "blend_mode": psd_mode,
                    "opacity": int(np.clip(getattr(adj, 'opacity', 1.0), 0.0, 1.0) * 255),
                    "visible": getattr(adj, 'enabled', True),
                    "rgb": _format_rgb(raw_img),
                    "mask": _format_mask(bmask) if bmask is not None else None
                }
            elif not is_raster and bmask is not None:
                adj_name = f"Ajuste - {getattr(adj, 'name', f'Capa {idx+1}')}"
                try:
                    from core.mask_layers import apply_local_adjustments_to_image
                    ones_mask = np.ones((h, w), dtype=np.float32)
                    adj_rgb = apply_local_adjustments_to_image(composite_image, ones_mask, adj)
                except Exception:
                    adj_rgb = composite_image

                return idx, {
                    "name": adj_name,
                    "top": 0, "left": 0, "bottom": h, "right": w,
                    "blend_mode": "norm",
                    "opacity": int(np.clip(getattr(adj, 'opacity', 1.0), 0.0, 1.0) * 255),
                    "visible": getattr(adj, 'enabled', True),
                    "rgb": _format_rgb(adj_rgb),
                    "mask": _format_mask(bmask)
                }
            return idx, None

        if n_user > 1:
            max_workers = min(4, n_user)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_build_adj_record, (idx, adj)): (idx, adj) for idx, adj in enumerate(user_layers)}
                finished = 0
                results = []
                for fut in as_completed(futures):
                    finished += 1
                    idx, rec = fut.result()
                    if rec is not None:
                        results.append((idx, rec))
                    layer_name = getattr(futures[fut][1], 'name', f'Capa {idx+1}')
                    pct = 72 + int(15 * (finished / max(1, n_user)))
                    _progress(pct, f"Ajuste {finished}/{n_user}: {layer_name[:12]}")
                results.sort(key=lambda x: x[0])
                for _, rec in results:
                    layer_records.append(rec)
        else:
            for idx, adj in enumerate(user_layers):
                _, rec = _build_adj_record((idx, adj))
                if rec is not None:
                    layer_records.append(rec)

    # 6. Capa Cúspide de Referencia: Resultado Revelado Final AstroLoom (Compuesto)
    # Por defecto 'visible=False' para no alterar el revelado interactivo en Photoshop
    layer_records.append({
        "name": "00 - Revelado Final (Referencia AstroLoom)",
        "top": 0, "left": 0, "bottom": h, "right": w,
        "blend_mode": "norm",
        "opacity": 255,
        "visible": False,
        "rgb": _format_rgb(composite_image)
    })

    total_layers = len(layer_records)
    _log(f"[PSD] Escribiendo archivo en disco ({total_layers} capas generadas con máscaras nativas)...")
    _progress(88, f"Escribiendo {total_layers} capas...")

    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)

    comp_contiguous = _format_rgb(composite_image)
    astroloom_rs.write_psd_file(
        filepath,
        w,
        h,
        depth,
        comp_contiguous,
        layer_records,
        icc_profile_bytes
    )

    file_size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
    desc = f"PSD multicapa ({total_layers} capas, {depth}-bit, {file_size_mb:.1f} MB)"
    _log(f"[PSD FINALIZADO] Guardado con éxito: {os.path.basename(filepath)} ({desc})")
    _progress(100, "PSD guardado con éxito")

    return True, desc, total_layers
