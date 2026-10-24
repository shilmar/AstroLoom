"""
core/icc_manager.py - Gestor integral de perfiles de color ICC para Apilator.
Permite incrustar etiquetas estándar ICC (Tag 34675 / InterColorProfile) en archivos TIFF,
garantizando compatibilidad colorimétrica total.
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


def normalize_colorspace_name(color_space: str | None) -> str:
    """Normaliza el identificador del espacio de color a una clave canónica ('srgb', 'adobe', 'prophoto')."""
    if not color_space:
        return "srgb"
    key = str(color_space).lower().strip()
    if "prophoto" in key or "romm" in key:
        return "prophoto"
    if "adobe" in key:
        return "adobe"
    return "srgb"


def get_profile_description(icc_bytes: bytes) -> str:
    """Devuelve la descripción legible del perfil ICC a partir de sus bytes."""
    if not icc_bytes:
        return "Sin perfil incrustado"
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        return prof.profile.profile_description or "Perfil ICC Personalizado"
    except Exception:
        return "Perfil ICC Binario"


def identify_colorspace_from_icc(icc_bytes: bytes) -> str:
    """Identifica el espacio canónico ('srgb', 'adobe', 'prophoto') a partir de los bytes ICC."""
    if not icc_bytes:
        return "srgb"
    desc = get_profile_description(icc_bytes)
    return normalize_colorspace_name(desc)


# =============================================================================
# ÁLGEBRA LINEAL Y MATRICES DE TRANSFORMACIÓN DE COLOR (CIE D65 / D50)
# =============================================================================

# Matriz sRGB lineal (D65) <-> Adobe RGB (1998, D65)
M_sRGB_to_Adobe = np.array([
    [0.7151626, 0.2848374, 0.0000000],
    [0.0000000, 1.0000000, 0.0000000],
    [0.0000000, 0.0411705, 0.9588295]
], dtype=np.float32)
M_sRGB_to_Adobe /= M_sRGB_to_Adobe.sum(axis=1, keepdims=True)
M_Adobe_to_sRGB = np.linalg.inv(M_sRGB_to_Adobe).astype(np.float32)

# Matriz sRGB lineal (D65) <-> ProPhoto RGB (D50, adaptación cromática Bradford)
M_sRGB_to_ProPhoto = np.array([
    [0.5293459, 0.3300728, 0.1405812],
    [0.0983744, 0.8734611, 0.0281647],
    [0.0168832, 0.1176725, 0.8654443]
], dtype=np.float32)
M_sRGB_to_ProPhoto /= M_sRGB_to_ProPhoto.sum(axis=1, keepdims=True)
M_ProPhoto_to_sRGB = np.linalg.inv(M_sRGB_to_ProPhoto).astype(np.float32)

# Matriz Adobe RGB (D65) <-> ProPhoto RGB (D50)
M_Adobe_to_ProPhoto = (M_sRGB_to_ProPhoto @ M_Adobe_to_sRGB).astype(np.float32)
M_Adobe_to_ProPhoto /= M_Adobe_to_ProPhoto.sum(axis=1, keepdims=True)
M_ProPhoto_to_Adobe = np.linalg.inv(M_Adobe_to_ProPhoto).astype(np.float32)


# =============================================================================
# CURVAS TONALES Y DECODIFICACIÓN / CODIFICACIÓN LINEAL
# =============================================================================

def srgb_to_linear(s: np.ndarray) -> np.ndarray:
    """Lineariza valores sRGB según estándar IEC 61966-2-1."""
    s = np.clip(s, 0.0, 1.0)
    mask = s <= 0.04045
    lin = np.empty_like(s)
    lin[mask] = s[mask] / 12.92
    lin[~mask] = ((s[~mask] + 0.055) / 1.055) ** 2.4
    return lin


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Aplica curva de codificación sRGB estándar."""
    lin = np.clip(lin, 0.0, 1.0)
    mask = lin <= 0.0031308
    out = np.empty_like(lin)
    out[mask] = lin[mask] * 12.92
    out[~mask] = 1.055 * (lin[~mask] ** (1.0 / 2.4)) - 0.055
    return np.clip(out, 0.0, 1.0)


def adobe_to_linear(a: np.ndarray) -> np.ndarray:
    """Lineariza Adobe RGB (1998) con exponente 2.19921875 (~gamma 2.2)."""
    a = np.clip(a, 0.0, 1.0)
    return a ** 2.19921875


def linear_to_adobe(lin: np.ndarray) -> np.ndarray:
    """Aplica curva de codificación Adobe RGB (1998)."""
    lin = np.clip(lin, 0.0, 1.0)
    return lin ** (1.0 / 2.19921875)


def prophoto_to_linear(p: np.ndarray) -> np.ndarray:
    """Lineariza ProPhoto RGB (ROMM RGB, ISO 22028-2) con tramo lineal y gamma 1.8."""
    p = np.clip(p, 0.0, 1.0)
    mask = p < 0.031248  # 16.0 * 0.001953
    lin = np.empty_like(p)
    lin[mask] = p[mask] / 16.0
    lin[~mask] = p[~mask] ** 1.8
    return lin


def linear_to_prophoto(lin: np.ndarray) -> np.ndarray:
    """Aplica curva de codificación ProPhoto RGB (ROMM RGB)."""
    lin = np.clip(lin, 0.0, 1.0)
    mask = lin < 0.001953
    out = np.empty_like(lin)
    out[mask] = 16.0 * lin[mask]
    out[~mask] = lin[~mask] ** (1.0 / 1.8)
    return np.clip(out, 0.0, 1.0)


# Caché de tablas Look-Up (LUT) de 16 bits para aceleración instantánea
_LUT_U16_CACHE: dict[str, np.ndarray] = {}
_LUT_U8_CACHE: dict[str, np.ndarray] = {}


def _get_u16_linear_lut(space: str) -> np.ndarray:
    """Obtiene tabla precalculada [65536] uint16 -> float32 lineal."""
    if space not in _LUT_U16_CACHE:
        axis = np.arange(65536, dtype=np.float32) / 65535.0
        if space == "srgb":
            _LUT_U16_CACHE["srgb"] = srgb_to_linear(axis)
        elif space == "adobe":
            _LUT_U16_CACHE["adobe"] = adobe_to_linear(axis)
        elif space == "prophoto":
            _LUT_U16_CACHE["prophoto"] = prophoto_to_linear(axis)
        else:
            _LUT_U16_CACHE[space] = axis
    return _LUT_U16_CACHE[space]


def _get_u8_linear_lut(space: str) -> np.ndarray:
    """Obtiene tabla precalculada [256] uint8 -> float32 lineal."""
    if space not in _LUT_U8_CACHE:
        axis = np.arange(256, dtype=np.float32) / 255.0
        if space == "srgb":
            _LUT_U8_CACHE["srgb"] = srgb_to_linear(axis)
        elif space == "adobe":
            _LUT_U8_CACHE["adobe"] = adobe_to_linear(axis)
        elif space == "prophoto":
            _LUT_U8_CACHE["prophoto"] = prophoto_to_linear(axis)
        else:
            _LUT_U8_CACHE[space] = axis
    return _LUT_U8_CACHE[space]


COLOR_TRANSFORMS = {
    ("srgb", "adobe"): (srgb_to_linear, M_sRGB_to_Adobe, linear_to_adobe),
    ("adobe", "srgb"): (adobe_to_linear, M_Adobe_to_sRGB, linear_to_srgb),
    ("srgb", "prophoto"): (srgb_to_linear, M_sRGB_to_ProPhoto, linear_to_prophoto),
    ("prophoto", "srgb"): (prophoto_to_linear, M_ProPhoto_to_sRGB, linear_to_srgb),
    ("adobe", "prophoto"): (adobe_to_linear, M_Adobe_to_ProPhoto, linear_to_prophoto),
    ("prophoto", "adobe"): (prophoto_to_linear, M_ProPhoto_to_Adobe, linear_to_adobe),
}


def convert_colorspace(
    img_data: np.ndarray,
    src_space: str = "srgb",
    dst_space: str = "srgb"
) -> np.ndarray:
    """
    Convierte una imagen (float32, uint16 o uint8) entre espacios de color
    (sRGB, Adobe RGB 1998, ProPhoto RGB) preservando el rango dinámico y el tipo de dato.
    Acelerado mediante LUTs para 16-bit / 8-bit y álgebra matricial vectorial NumPy.
    """
    if img_data is None or img_data.size == 0:
        return img_data

    src_k = normalize_colorspace_name(src_space)
    dst_k = normalize_colorspace_name(dst_space)

    if src_k == dst_k:
        return img_data

    pair = (src_k, dst_k)
    if pair not in COLOR_TRANSFORMS:
        return img_data

    to_lin_fn, mat, from_lin_fn = COLOR_TRANSFORMS[pair]

    orig_dtype = img_data.dtype
    has_alpha = (img_data.ndim == 3 and img_data.shape[2] == 4)
    rgb_part = img_data[..., :3] if has_alpha else img_data

    if orig_dtype == np.uint16:
        lut = _get_u16_linear_lut(src_k)
        lin = lut[rgb_part]
        lin_conv = lin @ mat.T
        out_f = from_lin_fn(lin_conv)
        conv_rgb = np.clip(out_f * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    elif orig_dtype == np.uint8:
        lut = _get_u8_linear_lut(src_k)
        lin = lut[rgb_part]
        lin_conv = lin @ mat.T
        out_f = from_lin_fn(lin_conv)
        conv_rgb = np.clip(out_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        f_in = np.clip(rgb_part.astype(np.float32), 0.0, 1.0)
        lin = to_lin_fn(f_in)
        lin_conv = lin @ mat.T
        out_f = from_lin_fn(lin_conv)
        conv_rgb = np.clip(out_f, 0.0, 1.0).astype(orig_dtype)

    if has_alpha:
        alpha = img_data[..., 3:]
        return np.concatenate([conv_rgb, alpha], axis=-1)
    return conv_rgb


def save_tiff_with_icc(
    output_path: str,
    img_data: np.ndarray,
    color_space: str = None,
    icc_bytes: bytes = None,
    compression: str = "zlib",
    photometric: str = "rgb",
    source_color_space: str = "srgb",
    convert_color: bool = True,
    **kwargs
) -> None:
    """
    Guarda una imagen en formato TIFF (flotante de 32 bits o entero de 16 bits)
    convirtiendo de forma precisa el espacio de color (p. ej. sRGB -> ProPhoto/AdobeRGB)
    e incrustando atómicamente el perfil ICC correspondiente (Tag 34675).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Determinar el perfil de destino canónico
    if color_space is None and "profile_name" in kwargs:
        color_space = kwargs["profile_name"]

    if color_space is None:
        if icc_bytes is not None and len(icc_bytes) > 0:
            target_space = identify_colorspace_from_icc(icc_bytes)
        else:
            from core.config_manager import get_config_val
            color_space = get_config_val("icc_profile", "srgb")
            target_space = normalize_colorspace_name(color_space)
    else:
        target_space = normalize_colorspace_name(color_space)

    # 2. Determinar los bytes ICC finales a incrustar
    final_icc = icc_bytes
    if final_icc is None or len(final_icc) == 0:
        final_icc = get_icc_profile_bytes(target_space)

    # 3. Conversión de coordenadas de color si el espacio origen difiere del destino
    clean_data = img_data
    src_space = normalize_colorspace_name(source_color_space)
    if convert_color and src_space != target_space:
        clean_data = convert_colorspace(clean_data, src_space=src_space, dst_space=target_space)

    # 4. Sanitización contra NaNs e Infs y acondicionamiento de tipo
    if np.issubdtype(clean_data.dtype, np.floating):
        clean_data = np.ascontiguousarray(clean_data, dtype=np.float32)
        if np.isnan(clean_data).any() or np.isinf(clean_data).any():
            clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=1.0, neginf=0.0)
        clean_data = np.clip(clean_data, 0.0, 1.0)
    elif clean_data.dtype == np.uint16:
        clean_data = np.ascontiguousarray(np.clip(clean_data, 0, 65535), dtype=np.uint16)
    elif clean_data.dtype == np.uint8:
        clean_data = np.ascontiguousarray(np.clip(clean_data, 0, 255), dtype=np.uint8)

    # 5. Eliminar archivo previo si existe para evitar bloqueos
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    # 6. Escritura en disco con etiqueta ICC Profile
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
