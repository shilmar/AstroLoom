# core/histogram.py
"""
core/histogram.py - Cálculo optimizado de histogramas en vivo (LightCraft 1-pass pattern).
Calcula simultáneamente los 4 canales (R, G, B y Luma Rec.709) en una sola pasada fusionada
sin asignaciones intermedias de memoria, con aceleración Numba JIT y respaldo NumPy.
"""
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(fastmath=True, nogil=True)
    def _hist_fused_f32_3d(img: np.ndarray, step: int) -> np.ndarray:
        h, w = img.shape[0], img.shape[1]
        hist = np.zeros((4, 256), dtype=np.uint32)
        for y in range(0, h, step):
            for x in range(0, w, step):
                rf = img[y, x, 0] * 255.0
                gf = img[y, x, 1] * 255.0
                bf = img[y, x, 2] * 255.0
                r = 0 if rf < 0.0 else (255 if rf > 255.0 else int(rf))
                g = 0 if gf < 0.0 else (255 if gf > 255.0 else int(gf))
                b = 0 if bf < 0.0 else (255 if bf > 255.0 else int(bf))
                # Fórmula entera Rec.709 de LightCraft: (54*r + 183*g + 19*b) >> 8
                lum = (r * 54 + g * 183 + b * 19) >> 8
                if lum > 255:
                    lum = 255
                hist[0, r] += 1
                hist[1, g] += 1
                hist[2, b] += 1
                hist[3, lum] += 1
        return hist

    @njit(fastmath=True, nogil=True)
    def _hist_fused_u8_3d(img: np.ndarray, step: int) -> np.ndarray:
        h, w = img.shape[0], img.shape[1]
        hist = np.zeros((4, 256), dtype=np.uint32)
        for y in range(0, h, step):
            for x in range(0, w, step):
                r = int(img[y, x, 0])
                g = int(img[y, x, 1])
                b = int(img[y, x, 2])
                lum = (r * 54 + g * 183 + b * 19) >> 8
                if lum > 255:
                    lum = 255
                hist[0, r] += 1
                hist[1, g] += 1
                hist[2, b] += 1
                hist[3, lum] += 1
        return hist

    @njit(fastmath=True, nogil=True)
    def _hist_fused_u16_3d(img: np.ndarray, step: int) -> np.ndarray:
        h, w = img.shape[0], img.shape[1]
        hist = np.zeros((4, 256), dtype=np.uint32)
        for y in range(0, h, step):
            for x in range(0, w, step):
                r = int(img[y, x, 0] >> 8)
                g = int(img[y, x, 1] >> 8)
                b = int(img[y, x, 2] >> 8)
                lum = (r * 54 + g * 183 + b * 19) >> 8
                if lum > 255:
                    lum = 255
                hist[0, r] += 1
                hist[1, g] += 1
                hist[2, b] += 1
                hist[3, lum] += 1
        return hist

    @njit(fastmath=True, nogil=True)
    def _hist_fused_mono(img: np.ndarray, step: int, is_float: bool, is_u16: bool) -> np.ndarray:
        h, w = img.shape[0], img.shape[1]
        hist = np.zeros((4, 256), dtype=np.uint32)
        for y in range(0, h, step):
            for x in range(0, w, step):
                if is_float:
                    vf = img[y, x] * 255.0
                    v = 0 if vf < 0.0 else (255 if vf > 255.0 else int(vf))
                elif is_u16:
                    v = int(img[y, x] >> 8)
                else:
                    v = int(img[y, x])
                hist[0, v] += 1
                hist[1, v] += 1
                hist[2, v] += 1
                hist[3, v] += 1
        return hist


def _hist_numpy_fallback(img: np.ndarray, step: int) -> np.ndarray:
    """Respaldo vectorizado puro mediante OpenCV/NumPy si Numba no está disponible."""
    sample = img[::step, ::step]
    if sample.dtype == np.uint8:
        u8 = sample
    elif sample.dtype == np.uint16:
        u8 = (sample >> 8).astype(np.uint8)
    else:
        u8 = (np.clip(sample, 0.0, 1.0) * 255.0).astype(np.uint8)

    hist = np.zeros((4, 256), dtype=np.uint32)
    if u8.ndim == 2 or (u8.ndim == 3 and u8.shape[2] == 1):
        ch = u8.squeeze()
        h_ch = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel()
        hist[0] = h_ch
        hist[1] = h_ch
        hist[2] = h_ch
        hist[3] = h_ch
    else:
        hist[0] = cv2.calcHist([u8], [0], None, [256], [0, 256]).ravel()
        hist[1] = cv2.calcHist([u8], [1], None, [256], [0, 256]).ravel()
        hist[2] = cv2.calcHist([u8], [2], None, [256], [0, 256]).ravel()
        r = u8[..., 0].astype(np.uint32)
        g = u8[..., 1].astype(np.uint32)
        b = u8[..., 2].astype(np.uint32)
        luma = ((r * 54 + g * 183 + b * 19) >> 8).astype(np.uint8)
        hist[3] = cv2.calcHist([luma], [0], None, [256], [0, 256]).ravel()
    return hist


def compute_live_histogram(
    img: Optional[np.ndarray],
    target_samples: int = 65536
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool, bool]:
    """
    Calcula de forma ultrarrápida (< 0.2 ms) los histogramas logarítmicos normalizados
    para R, G, B y Luma (Rec.709), junto con las banderas de recorte de sombras y luces.
    Retorna: (hist_r, hist_g, hist_b, hist_lum, has_shadow_clip, has_highlight_clip)
    """
    if img is None:
        return None, None, None, None, False, False

    if hasattr(img, "get"):
        try:
            img = img.get()
        except Exception:
            pass

    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim < 2:
        return None, None, None, None, False, False

    h, w = img.shape[0], img.shape[1]
    total_pixels = h * w
    if total_pixels <= target_samples or target_samples <= 0:
        step = 1
    else:
        step = max(1, int(np.sqrt(total_pixels / target_samples)))

    hist_raw = None
    if _HAS_NUMBA:
        try:
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                flat_mono = img.squeeze() if img.ndim == 3 else img
                is_f = flat_mono.dtype in (np.float32, np.float64)
                is_16 = flat_mono.dtype == np.uint16
                hist_raw = _hist_fused_mono(flat_mono, step, is_f, is_16)
            elif img.ndim == 3 and img.shape[2] >= 3:
                if img.dtype in (np.float32, np.float64):
                    hist_raw = _hist_fused_f32_3d(img, step)
                elif img.dtype == np.uint8:
                    hist_raw = _hist_fused_u8_3d(img, step)
                elif img.dtype == np.uint16:
                    hist_raw = _hist_fused_u16_3d(img, step)
        except Exception:
            hist_raw = None

    if hist_raw is None:
        hist_raw = _hist_numpy_fallback(img, step)

    has_shadow_clip = bool(hist_raw[3, 0] > 0 or hist_raw[0, 0] > 0)
    has_highlight_clip = bool(hist_raw[3, 255] > 0 or hist_raw[0, 255] > 0)

    # Compresión logarítmica astronómica
    r_log = np.log1p(hist_raw[0].astype(np.float32))
    g_log = np.log1p(hist_raw[1].astype(np.float32))
    b_log = np.log1p(hist_raw[2].astype(np.float32))
    l_log = np.log1p(hist_raw[3].astype(np.float32))

    max_val = max(float(r_log.max()), float(g_log.max()), float(b_log.max()), float(l_log.max()), 1e-5)

    return (
        (r_log / max_val).astype(np.float32),
        (g_log / max_val).astype(np.float32),
        (b_log / max_val).astype(np.float32),
        (l_log / max_val).astype(np.float32),
        has_shadow_clip,
        has_highlight_clip
    )


# Precalentamiento JIT al importar el módulo
if _HAS_NUMBA:
    try:
        _d3 = np.zeros((2, 2, 3), dtype=np.float32)
        _hist_fused_f32_3d(_d3, 1)
        _d3_u8 = np.zeros((2, 2, 3), dtype=np.uint8)
        _hist_fused_u8_3d(_d3_u8, 1)
        _d2 = np.zeros((2, 2), dtype=np.float32)
        _hist_fused_mono(_d2, 1, True, False)
    except Exception:
        pass
