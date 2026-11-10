"""
core/project_io.py - Motor de serialización y deserialización de proyectos AstroLoom Developer (.devloom).
Guarda y carga sesiones completas preservando sliders, curvas, máscaras 16-bit y capas StarNet.
Acelerado por Rust con Rayon y PNG nativo de 16 bits (astroloom_rs) con respaldo transparente
en OpenCV + zipfile de Python.
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import astroloom_rs
    HAS_RUST_PROJECT_IO = hasattr(astroloom_rs, "write_devloom_project_rs") and hasattr(
        astroloom_rs, "read_devloom_project_rs"
    )
except Exception:
    HAS_RUST_PROJECT_IO = False


def save_devloom_project(
    file_path: str,
    developer_state: Dict[str, Any],
    mask_layers: List[Any],
    master_sky_mask: Optional[np.ndarray] = None,
    source_image_path: Optional[str] = None,
    starnet_data: Optional[Dict[str, Any]] = None,
    cropped_base_image: Optional[np.ndarray] = None,
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
        else:
            layer_dict = dict(layer)
            brush_arr = layer_dict.pop("brush_mask", None)

        layer_dict.pop("brush_mask_b64", None)

        if brush_arr is not None and isinstance(brush_arr, np.ndarray):
            brush_file = f"masks/brush_{idx}.png"
            layer_images_to_encode.append((brush_file, brush_arr))
            layer_dict["mask_file"] = brush_file

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
        "has_cropped_base": has_cropped_base,
        "starless_is_stretched": developer_state.get("starless_is_stretched", False) if developer_state else False,
        "developer_state": developer_state,
        "mask_layers": serialized_layers,
    }
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)

    # 2. Vía nativa ultra-rápida en Rust (Rayon multicore)
    if HAS_RUST_PROJECT_IO:
        try:
            _notify(20, "Comprimiendo capas y máscaras en paralelo (Rust/Rayon)...")
            images_list: List[Tuple[str, np.ndarray]] = []
            if master_mask_ref and master_sky_mask is not None:
                images_list.append((master_mask_ref, np.ascontiguousarray(master_sky_mask, dtype=np.float32)))

            for b_name, b_arr in layer_images_to_encode:
                images_list.append((b_name, np.ascontiguousarray(b_arr, dtype=np.float32)))

            for s_name, s_arr in starnet_images:
                images_list.append((s_name, np.ascontiguousarray(s_arr, dtype=np.float32)))

            if has_cropped_base and cropped_base_image is not None:
                images_list.append(("base/cropped_base.png", np.ascontiguousarray(cropped_base_image, dtype=np.float32)))

            ok = astroloom_rs.write_devloom_project_rs(file_path, manifest_json, images_list)
            if ok:
                _notify(100, "Proyecto guardado con éxito.")
                return True
        except Exception:
            # Fallback transparente a Python puro si surge algún problema en Rust
            pass

    # 3. Respaldo en Python puro (zipfile + OpenCV)
    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if master_mask_ref and master_sky_mask is not None:
            _notify(15, "Comprimiendo máscara maestra de horizonte (16-bit)...")
            u16 = np.clip(master_sky_mask * 65535.0, 0, 65535).astype(np.uint16)
            ok, buf = cv2.imencode(".png", u16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            if ok:
                zf.writestr(master_mask_ref, buf.tobytes())

        _notify(25, "Comprimiendo máscaras de capas de ajuste locales...")
        for b_name, b_arr in layer_images_to_encode:
            u16_b = np.clip(b_arr * 65535.0, 0, 65535).astype(np.uint16)
            ok, buf = cv2.imencode(".png", u16_b, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            if ok:
                zf.writestr(b_name, buf.tobytes())

        if has_starnet:
            for s_name, s_arr in starnet_images:
                _notify(60, f"Comprimiendo capa StarNet ({s_name})...")
                u16_s = np.clip(s_arr * 65535.0, 0, 65535).astype(np.uint16)
                bgr_s = cv2.cvtColor(u16_s, cv2.COLOR_RGB2BGR) if u16_s.ndim == 3 else u16_s
                ok, buf = cv2.imencode(".png", bgr_s, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                if ok:
                    zf.writestr(s_name, buf.tobytes())

        if has_cropped_base and cropped_base_image is not None:
            _notify(85, "Comprimiendo imagen base recortada (16-bit)...")
            u16_cb = np.clip(cropped_base_image * 65535.0, 0, 65535).astype(np.uint16)
            bgr_cb = cv2.cvtColor(u16_cb, cv2.COLOR_RGB2BGR) if u16_cb.ndim == 3 else u16_cb
            ok, buf = cv2.imencode(".png", bgr_cb, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            if ok:
                zf.writestr("base/cropped_base.png", buf.tobytes())

        _notify(90, "Escribiendo project.json y metadatos...")
        zf.writestr("project.json", manifest_json)

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

    # 1. Vía nativa ultra-rápida en Rust (Rayon multicore)
    if HAS_RUST_PROJECT_IO:
        try:
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

            starnet_out = {
                "image_starless": images_dict.get("starnet/starless.png"),
                "image_stars": images_dict.get("starnet/stars.png"),
                "raw_linear_starless": images_dict.get("starnet/raw_linear_starless.png"),
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
                "cropped_base_image": cropped_base,
                "has_starnet": manifest.get("has_starnet", False),
                "starless_is_stretched": manifest.get("starless_is_stretched", False),
            }
        except Exception:
            # Fallback transparente si falla la descompresión en Rust
            pass

    # 2. Respaldo en Python puro (zipfile + OpenCV)
    with zipfile.ZipFile(file_path, "r") as zf:
        namelist = set(zf.namelist())
        if "project.json" not in namelist:
            raise ValueError("El archivo no es un contenedor .devloom válido (falta project.json).")

        _notify(15, "Leyendo project.json y parámetros...")
        manifest = json.loads(zf.read("project.json").decode("utf-8"))

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

        cropped_base = None
        if "base/cropped_base.png" in namelist:
            _notify(95, "Extrayendo imagen base recortada (16-bit)...")
            raw_b = zf.read("base/cropped_base.png")
            arr = np.frombuffer(raw_b, np.uint8)
            dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if dec is not None:
                if dec.ndim == 3:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                div = 65535.0 if dec.dtype == np.uint16 else 255.0
                cropped_base = dec.astype(np.float32) / div

    _notify(100, "Lectura de proyecto finalizada.")
    return {
        "manifest": manifest,
        "source_image": manifest.get("source_image", {}),
        "developer_state": manifest.get("developer_state", {}),
        "master_sky_mask": master_sky,
        "mask_layers": layers,
        "starnet": starnet_out,
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
