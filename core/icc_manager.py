"""
core/icc_manager.py - Gestor integral de perfiles de color ICC para Apilator.
Permite incrustar etiquetas estándar ICC (Tag 34675 / InterColorProfile) en archivos TIFF,
garantizando compatibilidad colorimétrica total en Photoshop, Lightroom, PixInsight, Siril, etc.
"""

import os
import io
import tifffile
import numpy as np
import rawpy
from PIL import ImageCms

ICC_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "icc_profiles")

PROFILE_MAP = {
    "srgb": "sRGB.icc",
    "adobe": "AdobeRGB1998.icc",
    "prophoto": "ProPhotoRGB.icc"
}

RAWPY_COLORSPACE_MAP = {
    "srgb": rawpy.ColorSpace.sRGB,
    "adobe": rawpy.ColorSpace.Adobe,
    "prophoto": rawpy.ColorSpace.ProPhoto
}


def get_rawpy_colorspace(color_space: str = "srgb") -> rawpy.ColorSpace:
    """Devuelve el espacio de color correspondiente para la decodificación de LibRaw."""
    if not color_space:
        return rawpy.ColorSpace.sRGB
    key = str(color_space).lower().strip()
    return RAWPY_COLORSPACE_MAP.get(key, rawpy.ColorSpace.sRGB)


def get_icc_profile_bytes(color_space: str = "srgb") -> bytes:
    """
    Obtiene los bytes en crudo del perfil de color ICC especificado ('srgb', 'adobe', 'prophoto').
    Incluye fallback dinámico a generación mediante LittleCMS si el archivo físico no estuviese disponible.
    """
    key = str(color_space).lower().strip() if color_space else "srgb"
    filename = PROFILE_MAP.get(key, "sRGB.icc")
    profile_path = os.path.join(ICC_PROFILES_DIR, filename)

    if os.path.exists(profile_path):
        try:
            with open(profile_path, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"[ICC Manager] Advertencia: Error leyendo {profile_path}: {e}")

    # Fallback dinámico con LittleCMS para sRGB
    try:
        cms_prof = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        return cms_prof.tobytes()
    except Exception as e:
        print(f"[ICC Manager] Error en fallback de perfil sRGB: {e}")
        return b""


def extract_icc_profile(filepath: str) -> bytes | None:
    """
    Extrae el perfil ICC incrustado en una toma de entrada (formato TIFF / Tag 34675) si existe.
    Permite preservar el espacio de color de origen del usuario al procesar ráfagas precalibradas.
    """
    if not filepath or not os.path.exists(filepath):
        return None
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.tif', '.tiff']:
        try:
            with tifffile.TiffFile(filepath) as tif:
                tag = tif.pages[0].tags.get(34675)  # Tag 34675: InterColorProfile
                if tag is not None and tag.value:
                    return bytes(tag.value)
        except Exception:
            pass
            
    return None


def get_profile_description(icc_bytes: bytes) -> str:
    """Devuelve la descripción legible del perfil ICC a partir de sus bytes."""
    if not icc_bytes:
        return "Sin perfil incrustado"
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        return prof.profile.profile_description or "Perfil ICC Personalizado"
    except Exception:
        return "Perfil ICC Binario"


def save_tiff_with_icc(
    output_path: str,
    img_data: np.ndarray,
    color_space: str = None,
    icc_bytes: bytes = None,
    compression: str = "zlib",
    photometric: str = "rgb"
) -> None:
    """
    Guarda una imagen en formato TIFF (flotante de 32 bits o entero de 16 bits)
    incrustando de forma atómica el perfil de color ICC adecuado.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Determinar los bytes del perfil ICC a incrustar
    final_icc = icc_bytes
    if final_icc is None:
        if color_space is None:
            from core.config_manager import get_config_val
            color_space = get_config_val("icc_profile", "srgb")
        final_icc = get_icc_profile_bytes(color_space)

    # 2. Sanitización contra NaNs e Infs en datos de 32 bits
    clean_data = img_data
    if np.issubdtype(clean_data.dtype, np.floating):
        clean_data = np.ascontiguousarray(clean_data, dtype=np.float32)
        if np.isnan(clean_data).any() or np.isinf(clean_data).any():
            clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=1.0, neginf=0.0)
        clean_data = np.clip(clean_data, 0.0, 1.0)
    elif clean_data.dtype == np.uint16:
        clean_data = np.ascontiguousarray(clean_data, dtype=np.uint16)

    # 3. Eliminar archivo previo si existe para evitar bloqueos
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    # 4. Escritura en disco con etiqueta ICC Profile
    if final_icc and len(final_icc) > 0:
        tifffile.imwrite(
            output_path,
            clean_data,
            compression=compression,
            photometric=photometric,
            iccprofile=final_icc
        )
    else:
        tifffile.imwrite(
            output_path,
            clean_data,
            compression=compression,
            photometric=photometric
        )
