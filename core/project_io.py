"""
core/project_io.py - Motor de serialización y deserialización de proyectos AstroLoom Developer (.devloom).
Guarda y carga sesiones completas preservando sliders, curvas, máscaras 16-bit y capas StarNet.
Impulsado por Rust con Rayon y PNG nativo de 16 bits (astroloom_rs).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import astroloom_rs

HAS_RUST_PROJECT_IO = True


def save_devloom_project(
    file_path: str,
    developer_state: Dict[str, Any],
    mask_layers: List[Any],
    master_sky_mask: Optional[np.ndarray] = None,
    source_image_path: Optional[str] = None,
    starnet_data: Optional[Dict[str, Any]] = None,
    cropped_base_image: Optional[np.ndarray] = None,
    ai_denoise_data: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> bool:
    """
    Guarda una sesión completa de desarrollo en un contenedor comprimido .devloom.
    
    :param file_path: Ruta de destino del archivo .devloom.
    :param developer_state: Diccionario paramétrico obtenido de _capture_state().
    :param mask_layers: Lista de capas (MaskLayer o dicts).
    :param master_sky_mask: Máscara maestra 2D float32 [0.0, 1.0].
    :param source_image_path: Ruta a la imagen base cargada.
    :param starnet_data: Diccionario opcional con 'image_starless', 'image_stars', 'raw_linear_starless'.
    :param cropped_base_image: Imagen base recortada 32-bit si se aplicó crop sin StarNet.
    :param progress_callback: Función callback(porcentaje: int, mensaje: str).
    """
    file_path = os.path.abspath(file_path)
    proj_dir = os.path.dirname(file_path)
    os.makedirs(proj_dir, exist_ok=True)

    def _notify(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    _notify(5, "Iniciando contenedor de proyecto...")

    # 1. Preparar metadatos y máscaras de capas
    serialized_layers = []
    layer_images_to_encode = []
    for idx, layer in enumerate(mask_layers):
        if hasattr(layer, "to_dict"):
            layer_dict = layer.to_dict()
            brush_arr = getattr(layer, "brush_mask", None)
            raster_arr = getattr(layer, "image_data", None)
        else:
            layer_dict = dict(layer)
            brush_arr = layer_dict.pop("brush_mask", None)
            raster_arr = layer_dict.pop("image_data", None)

        layer_dict.pop("proxy_preview", None)
        layer_dict.pop("proxy_draft", None)
        layer_dict.pop("brush_mask_b64", None)

        if brush_arr is not None and isinstance(brush_arr, np.ndarray):
            brush_file = f"masks/brush_{idx}.png"
            layer_images_to_encode.append((brush_file, brush_arr))
            layer_dict["mask_file"] = brush_file

        if raster_arr is not None and isinstance(raster_arr, np.ndarray):
            raster_file = f"rasters/raster_{idx}.png"
            layer_images_to_encode.append((raster_file, raster_arr))
            layer_dict["raster_file"] = raster_file

        serialized_layers.append(layer_dict)

    master_mask_ref = "masks/master_sky.png" if (master_sky_mask is not None and isinstance(master_sky_mask, np.ndarray)) else None
    has_starnet = False
    starnet_images = []
    if starnet_data:
        sl = starnet_data.get("image_starless")
        st = starnet_data.get("image_stars")
        raw_sl = starnet_data.get("raw_linear_starless")

        if sl is not None and isinstance(sl, np.ndarray):
            has_starnet = True
            starnet_images.append(("starnet/starless.png", sl))

        if st is not None and isinstance(st, np.ndarray):
            has_starnet = True
            starnet_images.append(("starnet/stars.png", st))

        if raw_sl is not None and isinstance(raw_sl, np.ndarray):
            has_starnet = True
            starnet_images.append(("starnet/raw_linear_starless.png", raw_sl))

    has_cropped_base = cropped_base_image is not None and isinstance(cropped_base_image, np.ndarray)

    ai_denoise_images = []
    ai_denoise_meta = {
        "starless_ai_enabled": False,
        "ground_ai_enabled": False,
    }
    if ai_denoise_data:
        sl_ai = ai_denoise_data.get("starless_ai")
        gnd_ai = ai_denoise_data.get("ground_ai")
        if sl_ai is not None and isinstance(sl_ai, np.ndarray):
            ai_denoise_images.append(("ai_denoise/starless_ai.png", sl_ai))
            ai_denoise_meta["starless_ai_enabled"] = bool(ai_denoise_data.get("starless_ai_enabled", False))
        if gnd_ai is not None and isinstance(gnd_ai, np.ndarray):
            ai_denoise_images.append(("ai_denoise/ground_ai.png", gnd_ai))
            ai_denoise_meta["ground_ai_enabled"] = bool(ai_denoise_data.get("ground_ai_enabled", False))

    source_info = {}
    if source_image_path:
        abs_source = os.path.abspath(source_image_path)
        source_info["filename"] = os.path.basename(abs_source)
        source_info["absolute_path"] = abs_source
        try:
            source_info["relative_path"] = os.path.relpath(abs_source, proj_dir).replace("\\", "/")
        except ValueError:
            source_info["relative_path"] = None

    # developer_state no debe contener objetos MaskLayer en memoria
    clean_developer_state = dict(developer_state) if developer_state else {}
    clean_developer_state.pop("mask_layers", None)

    manifest = {
        "version": "1.0",
        "app": "AstroLoom",
        "module": "developer",
        "source_image": source_info,
        "master_sky_mask": master_mask_ref,
        "has_starnet": has_starnet,
        "has_cropped_base": has_cropped_base,
        "starless_is_stretched": clean_developer_state.get("starless_is_stretched", False) if clean_developer_state else False,
        "ai_denoise": ai_denoise_meta,
        "developer_state": clean_developer_state,
        "mask_layers": serialized_layers,
    }
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)

    # 2. Escritura nativa paralela en Rust (Rayon multicore)
    _notify(20, "Comprimiendo capas y máscaras en paralelo (Rust/Rayon)...")
    images_list: List[Tuple[str, np.ndarray]] = []
    if master_mask_ref and master_sky_mask is not None:
        images_list.append((master_mask_ref, np.ascontiguousarray(master_sky_mask, dtype=np.float32)))

    for b_name, b_arr in layer_images_to_encode:
        images_list.append((b_name, np.ascontiguousarray(b_arr, dtype=np.float32)))

    for s_name, s_arr in starnet_images:
        images_list.append((s_name, np.ascontiguousarray(s_arr, dtype=np.float32)))

    for ai_name, ai_arr in ai_denoise_images:
        images_list.append((ai_name, np.ascontiguousarray(ai_arr, dtype=np.float32)))

    if has_cropped_base and cropped_base_image is not None:
        images_list.append(("base/cropped_base.png", np.ascontiguousarray(cropped_base_image, dtype=np.float32)))

    ok = astroloom_rs.write_devloom_project_rs(file_path, manifest_json, images_list)
    _notify(100, "Proyecto guardado con éxito.")
    return bool(ok)


def load_devloom_project(
    file_path: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Carga y deserializa una sesión de revelado .devloom.
    
    Retorna un diccionario con:
      - 'manifest': metadatos generales
      - 'source_image': dict con filename, relative_path, absolute_path
      - 'developer_state': dict para _restore_state()
      - 'master_sky_mask': np.ndarray float32 [0.0, 1.0] o None
      - 'mask_layers': lista de diccionarios de capas reconstruidos con 'brush_mask'
      - 'starnet': dict con 'image_starless', 'image_stars', 'raw_linear_starless'
    """
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo de proyecto no encontrado: {file_path}")

    def _notify(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    _notify(5, "Abriendo archivo .devloom...")

    # 1. Lectura nativa paralela en Rust (Rayon multicore)
    _notify(20, "Descomprimiendo y decodificando capas en paralelo (Rust/Rayon)...")
    manifest_json_str, images_dict = astroloom_rs.read_devloom_project_rs(file_path)
    manifest = json.loads(manifest_json_str)

    master_sky = None
    mask_ref = manifest.get("master_sky_mask")
    if mask_ref and mask_ref in images_dict:
        master_sky = images_dict[mask_ref]

    layers = manifest.get("mask_layers", [])
    for layer in layers:
        mask_file = layer.get("mask_file")
        if mask_file and mask_file in images_dict:
            layer["brush_mask"] = images_dict[mask_file]
        raster_file = layer.get("raster_file")
        if raster_file and raster_file in images_dict:
            layer["image_data"] = images_dict[raster_file]

    starnet_out = {
        "image_starless": images_dict.get("starnet/starless.png"),
        "image_stars": images_dict.get("starnet/stars.png"),
        "raw_linear_starless": images_dict.get("starnet/raw_linear_starless.png"),
    }

    ai_meta = manifest.get("ai_denoise") or {}
    ai_denoise_out = {
        "starless_ai": images_dict.get("ai_denoise/starless_ai.png"),
        "ground_ai": images_dict.get("ai_denoise/ground_ai.png"),
        "starless_ai_enabled": bool(ai_meta.get("starless_ai_enabled", False)),
        "ground_ai_enabled": bool(ai_meta.get("ground_ai_enabled", False)),
    }

    cropped_base = images_dict.get("base/cropped_base.png")

    _notify(100, "Lectura de proyecto finalizada.")
    return {
        "manifest": manifest,
        "source_image": manifest.get("source_image", {}),
        "developer_state": manifest.get("developer_state", {}),
        "master_sky_mask": master_sky,
        "mask_layers": layers,
        "starnet": starnet_out,
        "ai_denoise": ai_denoise_out,
        "cropped_base_image": cropped_base,
        "has_starnet": manifest.get("has_starnet", False),
        "starless_is_stretched": manifest.get("starless_is_stretched", False),
    }


def resolve_source_image_path(source_info: Dict[str, Any], devloom_filepath: str) -> Optional[str]:
    """
    Intenta localizar la imagen base referenciada por el proyecto buscando:
    1. Ruta relativa respecto a la ubicación del archivo .devloom.
    2. Nombre de archivo en la misma carpeta que el .devloom.
    3. Ruta absoluta guardada originalmente.
    
    Retorna la ruta válida si existe en disco, o None.
    """
    if not source_info:
        return None

    proj_dir = os.path.dirname(os.path.abspath(devloom_filepath))
    rel_path = source_info.get("relative_path")
    filename = source_info.get("filename")
    abs_path = source_info.get("absolute_path")

    candidates = []
    if rel_path:
        candidates.append(os.path.normpath(os.path.join(proj_dir, rel_path)))
    if filename:
        candidates.append(os.path.join(proj_dir, filename))
    if abs_path:
        candidates.append(abs_path)

    for p in candidates:
        if p and os.path.exists(p):
            return p

    return None
