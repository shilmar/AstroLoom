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
import cv2
import numpy as np

try:
    import astroloom_rs
    HAS_PSD_ENGINE = hasattr(astroloom_rs, "write_psd_file")
except Exception:
    HAS_PSD_ENGINE = False

# Compatibilidad con el resto del código que verificaba HAS_PYTOSHOP
HAS_PYTOSHOP = HAS_PSD_ENGINE


def is_psd_export_available() -> bool:
    """Indica si el motor de exportación PSD está disponible en el entorno."""
    return HAS_PSD_ENGINE


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
    log_fn: Optional[Callable[[str], None]] = None
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
    :return: (éxito: bool, mensaje_descriptivo: str, total_capas: int)
    """
    def _log(msg: str):
        if log_fn:
            log_fn(msg)

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

    # 5. Capas de ajuste locales o máscaras de usuario adicionales
    if adjustment_layers:
        user_adjustments = [
            l for l in adjustment_layers 
            if not getattr(l, 'is_system_layer', False) and getattr(l, 'layer_kind', 'adjustment') == 'adjustment'
        ]
        for idx, adj in enumerate(user_adjustments):
            bmask = getattr(adj, 'brush_mask', None)
            if bmask is None:
                bmask = getattr(adj, 'custom_mask', None)
            if bmask is not None:
                adj_name = f"Ajuste - {getattr(adj, 'name', f'Capa {idx+1}')}"
                try:
                    from core.mask_layers import apply_local_adjustments_to_image
                    ones_mask = np.ones((h, w), dtype=np.float32)
                    adj_rgb = apply_local_adjustments_to_image(composite_image, ones_mask, adj)
                except Exception:
                    adj_rgb = composite_image

                layer_records.append({
                    "name": adj_name,
                    "top": 0, "left": 0, "bottom": h, "right": w,
                    "blend_mode": "norm",
                    "opacity": int(np.clip(getattr(adj, 'opacity', 1.0), 0.0, 1.0) * 255),
                    "visible": getattr(adj, 'enabled', True),
                    "rgb": _format_rgb(adj_rgb),
                    "mask": _format_mask(bmask)
                })

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

    return True, desc, total_layers
