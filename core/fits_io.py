# core/fits_io.py
"""
core/fits_io.py - Lector nativo ultra-rápido de archivos FITS (.fits, .fit, .fts).
Impulsado por el motor nativo en Rust (astroloom_rs) con SIMD y multihilo Rayon.
"""
import os
from typing import Tuple
import numpy as np
import astroloom_rs

HAS_RUST_FITS = True


def get_fits_dimensions(filepath: str) -> Tuple[int, int]:
    """
    Obtiene (alto, ancho) del archivo FITS leyendo únicamente los bloques de cabecera en <0.2 ms.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo FITS no encontrado: {filepath}")
    return astroloom_rs.get_fits_dimensions_rs(filepath)


def read_fits(filepath: str) -> np.ndarray:
    """
    Lee un archivo astronómico FITS (.fits, .fit, .fts) y devuelve la matriz en float32 nativo.
    Soporta 2D (escala de grises) y 3D (cubos multicanal / RGB).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo FITS no encontrado: {filepath}")
    return astroloom_rs.read_fits_f32(filepath)
