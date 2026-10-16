# core/psd_export.py
"""
core/psd_export.py - Exportador de archivos PSD nativos (Photoshop) multicapa de 16 y 8 bits.
Permite exportar composiciones astrofotográficas con separación nativa de:
- Resultado Revelado Final (Compuesto)
- Imagen Base / Fondo original
- Fondo Nebular (Starless AI)
- Capa de Estrellas desacopladas (StarNet AI) con modo Trama / Sobreexponer lineal (Add)
- Suelo / Paisaje aislado por máscara de horizonte
- Máscara de Horizonte (referencia B/N en escala de grises para selecciones directas)
- Capas de ajuste y máscaras de pincel locales
"""
import os
from typing import Optional, List, Any, Callable, Tuple
import cv2
import numpy as np

try:
    import pytoshop
    from pytoshop import layers as psd_layers, enums as psd_enums, tagged_block
    HAS_PYTOSHOP = True
except ImportError:
    HAS_PYTOSHOP = False


def is_psd_export_available() -> bool:
    """Indica si la librería pytoshop está disponible en el entorno."""
    return HAS_PYTOSHOP


def export_developer_psd(
    filepath: str,
    composite_image: np.ndarray,
    base_image: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None,
    sky_mask: Optional[np.ndarray] = None,
    adjustment_layers: Optional[List[Any]] = None,
    depth: int = 16,
    log_fn: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str, int]:
    """
    Exporta el revelado completo a un archivo PSD nativo de Adobe Photoshop por capas.

    :param filepath: Ruta absoluta de destino (.psd).
    :param composite_image: Imagen procesada final (Float32 RGB [0.0, 1.0]).
    :param base_image: Imagen original base antes de ajustes globales.
    :param starless_image: Fondo sin estrellas (StarNet), opcional.
    :param stars_image: Imagen de estrellas puras (StarNet), opcional.
    :param sky_mask: Máscara 2D de cielo (1.0 = cielo, 0.0 = suelo), opcional.
    :param adjustment_layers: Lista opcional de MaskLayer con máscaras locales.
    :param depth: Profundidad de color por canal (16 para uint16, 8 para uint8).
    :param log_fn: Callback opcional para registrar mensajes informativos.
    :return: (éxito: bool, mensaje_descriptivo: str, total_capas: int)
    """
    def _log(msg: str):
        if log_fn:
            log_fn(msg)

    if not HAS_PYTOSHOP:
        err = "La librería 'pytoshop' no está instalada en el entorno. No se puede exportar en formato PSD."
        _log(f"[ERROR PSD] {err}")
        return False, err, 0

    if composite_image is None or composite_image.size == 0:
        return False, "La imagen compuesta está vacía.", 0

    h, w = composite_image.shape[:2]
    if depth not in (8, 16):
        depth = 16

    dtype = np.uint16 if depth == 16 else np.uint8
    max_val = 65535.0 if depth == 16 else 255.0

    _log(f"[PSD] Creando estructura PSD ({w}x{h} px, {depth}-bit)...")

    psd = pytoshop.core.PsdFile(num_channels=3, height=h, width=w, depth=depth)

    def _build_layer_record(
        name: str,
        rgb: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        blend: Any = psd_enums.BlendMode.normal,
        opacity: int = 255,
        visible: bool = True
    ) -> psd_layers.LayerRecord:
        """Crea un LayerRecord con canales R, G, B y Alpha opcional."""
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb_clamped = np.clip(rgb, 0.0, 1.0)
        ch_r = (rgb_clamped[..., 0] * max_val).astype(dtype)
        ch_g = (rgb_clamped[..., 1] * max_val).astype(dtype)
        ch_b = (rgb_clamped[..., 2] * max_val).astype(dtype)

        if alpha is not None:
            if alpha.ndim == 3:
                alpha = alpha[..., 0]
            if alpha.shape[:2] != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            ch_a = (np.clip(alpha, 0.0, 1.0) * max_val).astype(dtype)
        else:
            ch_a = np.full((h, w), int(max_val), dtype=dtype)

        channels = {
            0: psd_layers.ChannelImageData(image=ch_r, compression=psd_enums.Compression.raw),
            1: psd_layers.ChannelImageData(image=ch_g, compression=psd_enums.Compression.raw),
            2: psd_layers.ChannelImageData(image=ch_b, compression=psd_enums.Compression.raw),
            -1: psd_layers.ChannelImageData(image=ch_a, compression=psd_enums.Compression.raw)
        }

        blocks = [tagged_block.UnicodeLayerName(name=name)]

        rec = psd_layers.LayerRecord(
            top=0,
            left=0,
            bottom=h,
            right=w,
            name=name,
            opacity=opacity,
            visible=visible,
            blend_mode=blend,
            blocks=blocks
        )
        rec.channels = channels
        return rec

    layer_records = []

    # 1. Capa Base / Fondo original (Fondo de la pila en Photoshop)
    source_base = base_image if base_image is not None else composite_image
    layer_records.append(_build_layer_record(
        name="01 - Imagen Base (Original)",
        rgb=source_base,
        visible=True
    ))

    # 2. Capa Fondo Nebular (Starless) si está disponible
    if starless_image is not None:
        layer_records.append(_build_layer_record(
            name="02 - Fondo Nebular (Starless AI)",
            rgb=starless_image,
            blend=psd_enums.BlendMode.normal,
            visible=True
        ))

    # 3. Capa de Estrellas (Stars) con modo Linear Dodge (Añadir) para recomposición matemática perfecta
    if stars_image is not None:
        layer_records.append(_build_layer_record(
            name="03 - Estrellas (Linear Dodge / Add)",
            rgb=stars_image,
            blend=psd_enums.BlendMode.linear_dodge,
            visible=True
        ))

    # 4. Capas segmentadas por Máscara de Horizonte (Suelo / Cielo)
    if sky_mask is not None:
        clean_mask = np.squeeze(sky_mask)
        if clean_mask.ndim > 2:
            clean_mask = clean_mask[..., 0]
        clean_mask = clean_mask.astype(np.float32)
        if clean_mask.shape[:2] != (h, w):
            clean_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        ground_alpha = np.clip(1.0 - clean_mask, 0.0, 1.0)
        sky_alpha = np.clip(clean_mask, 0.0, 1.0)

        # Suelo aislado
        layer_records.append(_build_layer_record(
            name="04 - Suelo / Paisaje (Aislado)",
            rgb=source_base,
            alpha=ground_alpha,
            blend=psd_enums.BlendMode.normal,
            visible=True
        ))

        # Cielo / Nebulosa aislada
        sky_rgb = starless_image if starless_image is not None else composite_image
        layer_records.append(_build_layer_record(
            name="05 - Cielo / Estrellas (Aislado)",
            rgb=sky_rgb,
            alpha=sky_alpha,
            blend=psd_enums.BlendMode.normal,
            visible=True
        ))

        # Máscara visual de referencia (B/N en escala de grises, invisible por defecto)
        mask_rgb = np.repeat(clean_mask[..., np.newaxis], 3, axis=2)
        layer_records.append(_build_layer_record(
            name="06 - Referencia Máscara Horizonte (B/N)",
            rgb=mask_rgb,
            blend=psd_enums.BlendMode.normal,
            visible=False
        ))

    # 5. Capas de ajuste locales o máscaras de usuario adicionales
    if adjustment_layers:
        for idx, adj in enumerate(adjustment_layers):
            if hasattr(adj, 'brush_mask') and adj.brush_mask is not None:
                adj_name = f"Ajuste - {getattr(adj, 'name', f'Capa {idx+1}')}"
                bmask = adj.brush_mask
                if bmask.ndim > 2:
                    bmask = bmask[..., 0]
                layer_records.append(_build_layer_record(
                    name=adj_name,
                    rgb=composite_image,
                    alpha=bmask,
                    blend=psd_enums.BlendMode.normal,
                    visible=getattr(adj, 'enabled', True)
                ))

    # 6. Capa Cúspide: Resultado Revelado Final (Compuesto)
    # Permite ver el trabajo completo al abrir el PSD y ocultarla para manipular las capas inferiores.
    layer_records.append(_build_layer_record(
        name="00 - Revelado Final AstroLoom (Compuesto)",
        rgb=composite_image,
        blend=psd_enums.BlendMode.normal,
        visible=True
    ))

    # Asignar capas al objeto PSD
    for rec in layer_records:
        psd.layer_and_mask_info.layer_info.layer_records.append(rec)

    total_layers = len(layer_records)
    _log(f"[PSD] Escribiendo archivo en disco ({total_layers} capas generadas)...")

    # Escritura segura y atómica
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)

    with open(filepath, "wb") as fd:
        psd.write(fd)

    file_size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
    desc = f"PSD multicapa ({total_layers} capas, {depth}-bit, {file_size_mb:.1f} MB)"
    _log(f"[PSD FINALIZADO] Guardado con éxito: {os.path.basename(filepath)} ({desc})")

    return True, desc, total_layers
