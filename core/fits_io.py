# core/fits_io.py
"""
core/fits_io.py - Lector nativo ultra-rápido de archivos FITS (.fits, .fit, .fts).
Elimina completamente la dependencia externa pesada de `astropy`.
Acelerado por Rust con SIMD y multihilo Rayon (astroloom_rs) y respaldo transparente
en Python puro + NumPy.
"""
import os
from typing import Tuple, Optional
import numpy as np

try:
    import astroloom_rs
    HAS_RUST_FITS = hasattr(astroloom_rs, "read_fits_f32")
except Exception:
    HAS_RUST_FITS = False


def get_fits_dimensions(filepath: str) -> Tuple[int, int]:
    """
    Obtiene (alto, ancho) del archivo FITS leyendo únicamente los bloques de cabecera en <0.2 ms.
    """
    if HAS_RUST_FITS and hasattr(astroloom_rs, "get_fits_dimensions_rs"):
        try:
            return astroloom_rs.get_fits_dimensions_rs(filepath)
        except Exception:
            pass

    # Respaldo en Python puro
    with open(filepath, "rb") as f:
        h, w = _parse_header_dims(f)
        if h > 0 and w > 0:
            return h, w
    raise ValueError(f"No se pudieron leer las dimensiones FITS de '{filepath}'")


def read_fits(filepath: str) -> np.ndarray:
    """
    Lee un archivo astronómico FITS (.fits, .fit, .fts) y devuelve la matriz en float32 nativo.
    Soporta 2D (escala de grises) y 3D (cubos multicanal / RGB).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo FITS no encontrado: {filepath}")

    if HAS_RUST_FITS:
        try:
            return astroloom_rs.read_fits_f32(filepath)
        except Exception as e:
            # Fallback a respaldo si ocurre un error menor en Rust
            pass

    return _read_fits_fallback(filepath)


def _parse_header_dims(f) -> Tuple[int, int]:
    """Lee bloques de 2880 bytes buscando NAXIS1 y NAXIS2."""
    h, w = 0, 0
    while True:
        block = f.read(2880)
        if len(block) < 2880:
            break
        for i in range(0, 2880, 80):
            card = block[i:i+80].decode('ascii', errors='ignore')
            key = card[:8].strip()
            if key == "END":
                if h > 0 and w > 0:
                    return h, w
                break
            if card[8:9] == "=":
                val_str = card[9:].split('/')[0].strip()
                if key == "NAXIS1":
                    w = int(val_str)
                elif key == "NAXIS2":
                    h = int(val_str)
    return h, w


def _read_fits_fallback(filepath: str) -> np.ndarray:
    """Lector ligero FITS en Python puro + NumPy sin ninguna librería externa."""
    with open(filepath, "rb") as f:
        bitpix = 16
        naxis = 0
        naxis1, naxis2, naxis3 = 0, 0, 1
        bscale = 1.0
        bzero = 0.0

        while True:
            block = f.read(2880)
            if len(block) < 2880:
                raise ValueError("Cabecera FITS incompleta")
            end_found = False
            for i in range(0, 2880, 80):
                card = block[i:i+80].decode('ascii', errors='ignore')
                key = card[:8].strip()
                if key == "END":
                    end_found = True
                    break
                if card[8:9] == "=":
                    v = card[9:].split('/')[0].replace("'", "").strip()
                    if key == "BITPIX":
                        bitpix = int(v)
                    elif key == "NAXIS":
                        naxis = int(v)
                    elif key == "NAXIS1":
                        naxis1 = int(v)
                    elif key == "NAXIS2":
                        naxis2 = int(v)
                    elif key == "NAXIS3":
                        naxis3 = int(v)
                    elif key == "BSCALE":
                        bscale = float(v.replace('D', 'E').replace('d', 'e'))
                    elif key == "BZERO":
                        bzero = float(v.replace('D', 'E').replace('d', 'e'))

            if end_found:
                if naxis >= 2 and naxis1 > 0 and naxis2 > 0:
                    break
                # Si NAXIS == 0, buscar siguiente bloque de extensión
                continue

        dtype_map = {
            8: '>u1',
            16: '>i2',
            32: '>i4',
            -32: '>f4',
            -64: '>f8'
        }
        dt = dtype_map.get(bitpix)
        if not dt:
            raise ValueError(f"BITPIX {bitpix} no soportado en respaldo")

        count = naxis1 * naxis2 * (naxis3 if naxis >= 3 else 1)
        raw = np.fromfile(f, dtype=dt, count=count)
        data = raw.astype(np.float32)
        if bscale != 1.0:
            data *= np.float32(bscale)
        if bzero != 0.0:
            data += np.float32(bzero)

        if naxis >= 3 and naxis3 > 1:
            return data.reshape((naxis3, naxis2, naxis1))
        return data.reshape((naxis2, naxis1))
