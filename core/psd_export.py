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
"""
import os
from typing import Optional, List, Any, Callable, Tuple
import cv2
import numpy as np

try:
    import pytoshop
    from pytoshop import layers as psd_layers, enums as psd_enums, tagged_block, image_resources as psd_resources
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

    _log(f"[PSD] Creando estructura PSD multicapa ({w}x{h} px, {depth}-bit)...")

    psd = pytoshop.core.PsdFile(num_channels=3, height=h, width=w, depth=depth)

    # 1. Incrustar perfil de color ICC si se proporciona
    if icc_profile_bytes:
        try:
            icc_block = psd_resources.GenericImageResourceBlock(name="", resource_id=1039, data=icc_profile_bytes)
            psd.image_resources.blocks.append(icc_block)
            _log("[PSD] Perfil de color ICC incrustado en el archivo PSD.")
        except Exception as e:
            _log(f"[PSD AVISO] No se pudo incrustar el perfil ICC: {e}")

    # 2. Guardar imagen compuesta en psd.image_data para compatibilidad con visores externos y Lightroom
    try:
        comp_clamped = np.clip(composite_image, 0.0, 1.0)
        ch0 = (comp_clamped[..., 0] * max_val).astype(dtype)
        ch1 = (comp_clamped[..., 1] * max_val).astype(dtype)
        ch2 = (comp_clamped[..., 2] * max_val).astype(dtype)
        psd.image_data = pytoshop.image_data.ImageData(channels=np.stack([ch0, ch1, ch2], axis=0))
    except Exception as e:
        _log(f"[PSD AVISO] Fallo menor al generar ImageData compuesto: {e}")

    def _build_layer_record(
        name: str,
        rgb: np.ndarray,
        layer_mask: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        blend: Any = psd_enums.BlendMode.normal,
        opacity: int = 255,
        visible: bool = True
    ) -> psd_layers.LayerRecord:
        """Crea un LayerRecord con canales R, G, B y opcionalmente máscara de capa (user_layer_mask) y/o alfa."""
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., np.newaxis], 3, axis=2)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb_clamped = np.clip(rgb, 0.0, 1.0)
        ch_r = (rgb_clamped[..., 0] * max_val).astype(dtype)
        ch_g = (rgb_clamped[..., 1] * max_val).astype(dtype)
        ch_b = (rgb_clamped[..., 2] * max_val).astype(dtype)

        channels = {
            0: psd_layers.ChannelImageData(image=ch_r, compression=psd_enums.Compression.raw),
            1: psd_layers.ChannelImageData(image=ch_g, compression=psd_enums.Compression.raw),
            2: psd_layers.ChannelImageData(image=ch_b, compression=psd_enums.Compression.raw),
        }

        if alpha is not None:
            if alpha.ndim == 3:
                alpha = alpha[..., 0]
            if alpha.shape[:2] != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            ch_a = (np.clip(alpha, 0.0, 1.0) * max_val).astype(dtype)
            channels[-1] = psd_layers.ChannelImageData(image=ch_a, compression=psd_enums.Compression.raw)

        psd_mask = None
        if layer_mask is not None:
            if layer_mask.ndim == 3:
                layer_mask = layer_mask[..., 0]
            if layer_mask.shape[:2] != (h, w):
                layer_mask = cv2.resize(layer_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            ch_mask = (np.clip(layer_mask, 0.0, 1.0) * max_val).astype(dtype)
            channels[-2] = psd_layers.ChannelImageData(image=ch_mask, compression=psd_enums.Compression.raw)
            psd_mask = psd_layers.LayerMask(
                top=0,
                left=0,
                bottom=h,
                right=w,
                default_color=False,
                layer_mask_disabled=False
            )

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
        if psd_mask is not None:
            rec.mask = psd_mask
        rec.channels = channels
        return rec

    layer_records = []

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

    # 1. Capa Base / Fondo Revelado (Fondo de la pila en Photoshop)
    source_base = base_image if base_image is not None else composite_image
    base_name = "01 - Imagen Base Revelada" if (starless_image is not None or clean_mask is not None) else "01 - Imagen Revelada"
    layer_records.append(_build_layer_record(
        name=base_name,
        rgb=source_base,
        visible=True
    ))

    # 2. Capa Fondo Nebular (Starless AI) si está disponible (con máscara de cielo nativa vinculada si existe)
    if starless_image is not None:
        layer_records.append(_build_layer_record(
            name="02 - Fondo Starless",
            rgb=starless_image,
            layer_mask=clean_mask,
            blend=psd_enums.BlendMode.normal,
            visible=True
        ))

    # 3. Capa de Estrellas (Stars) con modo Trama (Screen) y máscara de cielo nativa vinculada
    if stars_image is not None:
        layer_records.append(_build_layer_record(
            name="03 - Estrellas (Trama / Screen)",
            rgb=stars_image,
            layer_mask=clean_mask,
            blend=psd_enums.BlendMode.screen,
            visible=True
        ))

    # 4. Capa Suelo / Terreno con Máscara de Capa nativa vinculada (Suelo=Blanco, Cielo=Negro)
    if clean_mask is not None:
        ground_source = ground_image if ground_image is not None else source_base
        gnd_layer_name = "04 - Suelo / Paisaje" if starless_image is not None else "02 - Suelo / Paisaje"
        ground_mask = np.clip(1.0 - clean_mask, 0.0, 1.0)
        layer_records.append(_build_layer_record(
            name=gnd_layer_name,
            rgb=ground_source,
            layer_mask=ground_mask,
            blend=psd_enums.BlendMode.normal,
            visible=True
        ))

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

                layer_records.append(_build_layer_record(
                    name=adj_name,
                    rgb=adj_rgb,
                    layer_mask=bmask,
                    blend=psd_enums.BlendMode.normal,
                    opacity=int(np.clip(getattr(adj, 'opacity', 1.0), 0.0, 1.0) * 255),
                    visible=getattr(adj, 'enabled', True)
                ))

    # 6. Capa Cúspide de Referencia: Resultado Revelado Final AstroLoom (Compuesto)
    # Por defecto 'visible=False' (ojo apagado) para que el usuario acceda de inmediato a las
    # capas independientes y pueda activar el ojo cuando desee contrastar con el original de AstroLoom.
    layer_records.append(_build_layer_record(
        name="00 - Revelado Final (Referencia AstroLoom)",
        rgb=composite_image,
        blend=psd_enums.BlendMode.normal,
        visible=False
    ))

    # Asignar capas al objeto PSD
    for rec in layer_records:
        psd.layer_and_mask_info.layer_info.layer_records.append(rec)

    total_layers = len(layer_records)
    _log(f"[PSD] Escribiendo archivo en disco ({total_layers} capas generadas con máscaras nativas)...")

    # Escritura segura y atómica
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)

    with open(filepath, "wb") as fd:
        psd.write(fd)

    file_size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
    desc = f"PSD multicapa ({total_layers} capas, {depth}-bit, {file_size_mb:.1f} MB)"
    _log(f"[PSD FINALIZADO] Guardado con éxito: {os.path.basename(filepath)} ({desc})")

    return True, desc, total_layers
