"""
core/project_io.py - Motor de serialización y deserialización de proyectos AstroLoom Developer (.devloom).
Guarda y carga sesiones completas preservando sliders, curvas, máscaras 16-bit y capas StarNet.
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np


def save_devloom_project(
    file_path: str,
    developer_state: Dict[str, Any],
    mask_layers: List[Any],
    master_sky_mask: Optional[np.ndarray] = None,
    source_image_path: Optional[str] = None,
    starnet_data: Optional[Dict[str, Any]] = None,
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
    :param progress_callback: Función callback(porcentaje: int, mensaje: str).
    """
    file_path = os.path.abspath(file_path)
    proj_dir = os.path.dirname(file_path)
    os.makedirs(proj_dir, exist_ok=True)

    def _notify(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    _notify(5, "Iniciando contenedor de proyecto...")

    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Guardar máscara maestra si existe (16-bit PNG)
        master_mask_ref = None
        if master_sky_mask is not None and isinstance(master_sky_mask, np.ndarray):
            _notify(15, "Comprimiendo máscara maestra de horizonte (16-bit)...")
            master_mask_ref = "masks/master_sky.png"
            u16 = np.clip(master_sky_mask * 65535.0, 0, 65535).astype(np.uint16)
            ok, buf = cv2.imencode(".png", u16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            if ok:
                zf.writestr(master_mask_ref, buf.tobytes())

        # 2. Guardar máscaras de capas (16-bit PNG)
        _notify(25, "Comprimiendo máscaras de capas de ajuste locales...")
        serialized_layers = []
        for idx, layer in enumerate(mask_layers):
            if hasattr(layer, "to_dict"):
                layer_dict = layer.to_dict()
                brush_arr = getattr(layer, "brush_mask", None)
            else:
                layer_dict = dict(layer)
                brush_arr = layer_dict.pop("brush_mask", None)

            # Limpiar base64 para mantener project.json ligero
            layer_dict.pop("brush_mask_b64", None)

            if brush_arr is not None and isinstance(brush_arr, np.ndarray):
                brush_file = f"masks/brush_{idx}.png"
                u16_b = np.clip(brush_arr * 65535.0, 0, 65535).astype(np.uint16)
                ok, buf = cv2.imencode(".png", u16_b, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                if ok:
                    zf.writestr(brush_file, buf.tobytes())
                    layer_dict["mask_file"] = brush_file

            serialized_layers.append(layer_dict)

        # 3. Guardar capas StarNet si existen
        has_starnet = False
        if starnet_data:
            sl = starnet_data.get("image_starless")
            st = starnet_data.get("image_stars")
            raw_sl = starnet_data.get("raw_linear_starless")

            if sl is not None and isinstance(sl, np.ndarray):
                has_starnet = True
                _notify(45, "Comprimiendo capa StarNet Fondo Nebular (16-bit)...")
                u16_sl = np.clip(sl * 65535.0, 0, 65535).astype(np.uint16)
                bgr_sl = cv2.cvtColor(u16_sl, cv2.COLOR_RGB2BGR) if u16_sl.ndim == 3 else u16_sl
                ok, buf = cv2.imencode(".png", bgr_sl, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                if ok:
                    zf.writestr("starnet/starless.png", buf.tobytes())

            if st is not None and isinstance(st, np.ndarray):
                has_starnet = True
                _notify(65, "Comprimiendo capa StarNet Estrellas (16-bit)...")
                u16_st = np.clip(st * 65535.0, 0, 65535).astype(np.uint16)
                bgr_st = cv2.cvtColor(u16_st, cv2.COLOR_RGB2BGR) if u16_st.ndim == 3 else u16_st
                ok, buf = cv2.imencode(".png", bgr_st, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                if ok:
                    zf.writestr("starnet/stars.png", buf.tobytes())

            if raw_sl is not None and isinstance(raw_sl, np.ndarray):
                _notify(80, "Comprimiendo capa StarNet Lineal (16-bit)...")
                u16_rsl = np.clip(raw_sl * 65535.0, 0, 65535).astype(np.uint16)
                bgr_rsl = cv2.cvtColor(u16_rsl, cv2.COLOR_RGB2BGR) if u16_rsl.ndim == 3 else u16_rsl
                ok, buf = cv2.imencode(".png", bgr_rsl, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                if ok:
                    zf.writestr("starnet/raw_linear_starless.png", buf.tobytes())

        # 4. Formatear rutas de imagen base
        _notify(90, "Escribiendo project.json y metadatos...")
        source_info = {}
        if source_image_path:
            abs_source = os.path.abspath(source_image_path)
            source_info["filename"] = os.path.basename(abs_source)
            source_info["absolute_path"] = abs_source
            try:
                source_info["relative_path"] = os.path.relpath(abs_source, proj_dir).replace("\\", "/")
            except ValueError:
                source_info["relative_path"] = None

        manifest = {
            "version": "1.0",
            "app": "AstroLoom",
            "module": "developer",
            "source_image": source_info,
            "master_sky_mask": master_mask_ref,
            "has_starnet": has_starnet,
            "starless_is_stretched": developer_state.get("starless_is_stretched", False) if developer_state else False,
            "developer_state": developer_state,
            "mask_layers": serialized_layers,
        }

        zf.writestr("project.json", json.dumps(manifest, indent=2, ensure_ascii=False))

    _notify(100, "Proyecto guardado con éxito.")
    return True


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

    with zipfile.ZipFile(file_path, "r") as zf:
        namelist = set(zf.namelist())
        if "project.json" not in namelist:
            raise ValueError("El archivo no es un contenedor .devloom válido (falta project.json).")

        _notify(15, "Leyendo project.json y parámetros...")
        manifest = json.loads(zf.read("project.json").decode("utf-8"))

        # 1. Cargar máscara maestra de cielo si existe
        master_sky = None
        mask_ref = manifest.get("master_sky_mask")
        if mask_ref and mask_ref in namelist:
            _notify(30, "Decodificando máscara maestra de horizonte (16-bit)...")
            raw_b = zf.read(mask_ref)
            arr = np.frombuffer(raw_b, np.uint8)
            dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                div = 65535.0 if dec.dtype == np.uint16 else 255.0
                master_sky = dec.astype(np.float32) / div

        # 2. Cargar capas de ajuste y máscaras de pincel
        _notify(45, "Decodificando máscaras de capas locales (16-bit)...")
        layers = manifest.get("mask_layers", [])
        for layer in layers:
            mask_file = layer.get("mask_file")
            if mask_file and mask_file in namelist:
                raw_b = zf.read(mask_file)
                arr = np.frombuffer(raw_b, np.uint8)
                dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if dec is not None:
                    div = 65535.0 if dec.dtype == np.uint16 else 255.0
                    layer["brush_mask"] = dec.astype(np.float32) / div

        # 3. Cargar capas StarNet si existen
        starnet_out = {
            "image_starless": None,
            "image_stars": None,
            "raw_linear_starless": None,
        }

        if "starnet/starless.png" in namelist:
            _notify(60, "Extrayendo capa StarNet Fondo Nebular (16-bit)...")
            raw_b = zf.read("starnet/starless.png")
            arr = np.frombuffer(raw_b, np.uint8)
            dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                if dec.ndim == 3:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                div = 65535.0 if dec.dtype == np.uint16 else 255.0
                starnet_out["image_starless"] = dec.astype(np.float32) / div

        if "starnet/stars.png" in namelist:
            _notify(80, "Extrayendo capa StarNet Estrellas (16-bit)...")
            raw_b = zf.read("starnet/stars.png")
            arr = np.frombuffer(raw_b, np.uint8)
            dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                if dec.ndim == 3:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                div = 65535.0 if dec.dtype == np.uint16 else 255.0
                starnet_out["image_stars"] = dec.astype(np.float32) / div

        if "starnet/raw_linear_starless.png" in namelist:
            _notify(90, "Extrayendo capa StarNet Lineal (16-bit)...")
            raw_b = zf.read("starnet/raw_linear_starless.png")
            arr = np.frombuffer(raw_b, np.uint8)
            dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                if dec.ndim == 3:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                div = 65535.0 if dec.dtype == np.uint16 else 255.0
                starnet_out["raw_linear_starless"] = dec.astype(np.float32) / div

    _notify(100, "Lectura de proyecto finalizada.")
    return {
        "manifest": manifest,
        "source_image": manifest.get("source_image", {}),
        "developer_state": manifest.get("developer_state", {}),
        "master_sky_mask": master_sky,
        "mask_layers": layers,
        "starnet": starnet_out,
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
