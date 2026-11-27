# gui/export_utils.py
"""
gui/export_utils.py - Utilidades compartidas de diálogo y guardado de imágenes
(TIFF 16/32-bit con ICC, PNG 16-bit, JPEG 8-bit) para todas las pestañas de AstroLoom.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog, QWidget

from core.icc_manager import save_tiff_with_icc


def ask_export_path(
    parent: QWidget,
    title: str,
    default_base: str,
    force_format: Optional[str] = None,
    psd_label: str = "Photoshop PSD (*.psd)",
    default_dir: Optional[str] = None,
    default_format: str = "tiff16",
) -> Optional[Tuple[str, str]]:
    """
    Muestra el diálogo unificado de guardado y resuelve (ruta_normalizada, formato)
    donde formato es uno de: 'psd', 'png', 'jpg', 'tiff32', 'tiff16'.
    Devuelve None si el usuario cancela.
    """
    filtros = (
        "TIFF 16-bit (*.tif *.tiff);;"
        "TIFF 32-bit Float (*.tif *.tiff);;"
        f"{psd_label};;"
        "PNG (*.png);;"
        "JPEG (*.jpg *.jpeg)"
    )

    fmt_init = (force_format or default_format).lower()
    if fmt_init == "psd":
        default_filter = psd_label
        default_name = f"{default_base}.psd"
    elif fmt_init == "tiff32":
        default_filter = "TIFF 32-bit Float (*.tif *.tiff)"
        default_name = f"{default_base}_32bit.tif"
    elif fmt_init == "png":
        default_filter = "PNG (*.png)"
        default_name = f"{default_base}.png"
    elif fmt_init in ("jpg", "jpeg"):
        default_filter = "JPEG (*.jpg *.jpeg)"
        default_name = f"{default_base}.jpg"
    else:
        default_filter = "TIFF 16-bit (*.tif *.tiff)"
        default_name = f"{default_base}_16bit.tif"

    if default_dir:
        os.makedirs(default_dir, exist_ok=True)
        initial_path = os.path.join(default_dir, default_name)
    else:
        initial_path = default_name

    path, selected_filter = QFileDialog.getSaveFileName(
        parent,
        title,
        initial_path,
        filtros,
        selectedFilter=default_filter,
    )
    if not path:
        return None

    path = os.path.abspath(os.path.normpath(str(path).strip()))
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        if "Photoshop" in selected_filter or "*.psd" in selected_filter:
            ext = ".psd"
        elif "PNG" in selected_filter or "*.png" in selected_filter:
            ext = ".png"
        elif "JPEG" in selected_filter or "*.jpg" in selected_filter:
            ext = ".jpg"
        else:
            ext = ".tif"
        path += ext

    if ext == ".psd" or ("Photoshop" in selected_filter and ext not in (".tif", ".tiff", ".jpg", ".jpeg", ".png")):
        fmt = "psd"
    elif ext == ".png" or "PNG" in selected_filter:
        fmt = "png"
    elif ext in (".jpg", ".jpeg") or ("JPEG" in selected_filter and ext not in (".tif", ".tiff")):
        fmt = "jpg"
    elif "32-bit" in selected_filter or (ext in (".tif", ".tiff") and "32-bit" in path):
        fmt = "tiff32"
    else:
        fmt = "tiff16"

    return path, fmt


def save_flat_image(path: str, img: np.ndarray, fmt: str) -> str:
    """
    Guarda una imagen float32 [0, 1] (RGB o monocroma) en disco en el formato indicado
    ('tiff16', 'tiff32', 'png', 'jpg') y devuelve una descripción legible del formato.
    """
    photometric = "minisblack" if img.ndim == 2 else "rgb"

    if fmt == "jpg":
        u8 = np.clip(img * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR) if u8.ndim == 3 else cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        return "JPEG 8-bit (Calidad 96)"

    if fmt == "png":
        u16 = np.ascontiguousarray(np.clip(img * 65535.0 + 0.5, 0.0, 65535.0), dtype=np.uint16)
        bgr16 = cv2.cvtColor(u16, cv2.COLOR_RGB2BGR) if img.ndim == 3 else u16
        cv2.imwrite(path, bgr16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        return "PNG 16-bit"

    if fmt == "tiff32":
        out32 = np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)
        save_tiff_with_icc(path, out32, compression="zlib", photometric=photometric)
        return "TIFF 32-bit float (HDR)"

    out16 = np.ascontiguousarray(np.clip(img * 65535.0 + 0.5, 0.0, 65535.0), dtype=np.uint16)
    save_tiff_with_icc(path, out16, compression="zlib", photometric=photometric)
    return "TIFF 16-bit entero"
