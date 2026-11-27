# core/ghs.py
"""
Módulo de Estirado Hiperbólico Generalizado (Generalized Hyperbolic Stretch - GHS).
Implementación matemática rigurosa basada en la formulación de David Payne y Mike Cranfield (GHS Astro).
Permite expandir el rango dinámico de astrofotografías lineales con máximo contraste local centrado en
el Punto de Simetría (SP), previniendo el hinchado de estrellas (star bloat) y el quemado de altas luces mediante
protección analítica (HP/LP) y preservando la fidelidad cromática en espacio de luminancia.
"""

from typing import Optional, Dict
import numpy as np
import cv2
import astroloom_rs

HAS_RUST = True


def compute_ghs_lut(
    sp: float,
    d: float,
    b: float,
    hp: float = 0.0,
    lp: float = 0.0,
    bp: float = 0.0,
    lut_size: int = 65536
) -> np.ndarray:
    """
    Precomputa una Look-Up Table (LUT) de 16 bits para Estirado Hiperbólico Generalizado (GHS).

    Parámetros:
    - sp: Punto de Simetría (Symmetry Point) en rango [0.0, 1.0]. Punto de máxima pendiente/contraste.
    - d: Factor de estirado (Stretch factor / intensidad). d >= 0.0 (0.0 = transformación identidad).
    - b: Parámetro de forma / concentración local. b >= 0.0.
         (b=0 exponencial puro, b>0 hiperbólico con compresión asintótica de altas luces).
    - hp: Factor de protección de altas luces (Highlight Protection) en rango [0.0, 1.0].
          hp=0.0 sin protección adicional, hp>0.0 extiende linealmente la pendiente antes del blanco.
    - lp: Factor de protección de sombras (Lowlight Protection) en rango [0.0, 1.0].
    - bp: Punto negro previo al estirado en rango [0.0, 0.95].
    - lut_size: Resolución de la tabla (por defecto 65,536 puntos para precisión de 16 bits).

    Retorna:
    - np.ndarray de float32 de forma (lut_size,) con valores estrictamente acotados en [0.0, 1.0].
    """
    return astroloom_rs.compute_ghs_lut_rs(
        float(sp), float(d), float(b),
        float(hp), float(lp), float(bp),
        int(lut_size)
    )


def apply_ghs_stretch(
    img: np.ndarray,
    sp: float,
    d: float,
    b: float,
    hp: float = 0.0,
    lp: float = 0.0,
    bp: float = 0.0,
    color_mode: str = "luma",
    lut_size: int = 65536,
    precomputed_lut: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Aplica Estirado Hiperbólico Generalizado a una imagen 2D o 3D (RGB) en float32 [0.0, 1.0].
    """
    lut = precomputed_lut if precomputed_lut is not None else compute_ghs_lut(
        sp=sp, d=d, b=b, hp=hp, lp=lp, bp=bp, lut_size=lut_size
    )
    img_c = np.ascontiguousarray(img, dtype=np.float32)
    lut_c = np.ascontiguousarray(lut, dtype=np.float32)
    return astroloom_rs.apply_ghs_stretch_rs(img_c, lut_c, str(color_mode))


def estimate_ghs_params(
    img_float: np.ndarray, 
    target_background: float = 0.18,
    sky_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Estima analíticamente los parámetros óptimos de GHS para una imagen lineal de astrofotografía:
    - SP: Centrado en la mediana del fondo del cielo (conciencia de máscara si se provee).
    - BP: Derivado de la dispersión de ruido (mediana - 2.5 * MAD), protegiendo sombras del suelo.
    - D: Calculado para elevar el fondo a target_background (~0.18) sin reventar estrellas.
    - b: Configurado en 2.0 para una compresión equilibrada.
    - HP: 0.15 para protección inicial de centros de estrellas.
    """
    sample = img_float[::4, ::4]
    if sample.ndim == 3:
        lum_sample = 0.2126 * sample[..., 0] + 0.7152 * sample[..., 1] + 0.0722 * sample[..., 2]
    else:
        lum_sample = sample

    h_s, w_s = lum_sample.shape[:2]
    med_all = float(np.median(lum_sample))
    mad_all = float(np.median(np.abs(lum_sample - med_all)))

    if sky_mask is not None:
        if sky_mask.shape[:2] != (h_s, w_s):
            m_s = cv2.resize(sky_mask.astype(np.float32), (w_s, h_s), interpolation=cv2.INTER_NEAREST)
        else:
            m_s = sky_mask.astype(np.float32)
        sky_pixels = lum_sample[m_s > 0.50]
        if len(sky_pixels) >= 50:
            med_sky = float(np.median(sky_pixels))
            mad_sky = float(np.median(np.abs(sky_pixels - med_sky)))
            sp = float(np.clip(med_sky, 0.0001, 0.90))
            bp = float(np.clip(min(med_sky - 2.5 * mad_sky, med_all - 2.5 * mad_all), 0.0, 0.90))
        else:
            sp = float(np.clip(med_all, 0.0001, 0.90))
            bp = float(np.clip(med_all - 2.5 * mad_all, 0.0, 0.90))
    else:
        sp = float(np.clip(med_all, 0.0001, 0.90))
        bp = float(np.clip(med_all - 2.5 * mad_all, 0.0, 0.90))

    b = 2.0
    hp = 0.15
    lp = 0.0

    sp_eff = max(1e-5, (sp - bp) / max(1e-6, 1.0 - bp))

    # Búsqueda dicotómica de D para alcanzar target_background en el fondo
    low_d, high_d = 0.1, 20.0
    best_d = 5.0
    for _ in range(25):
        mid_d = (low_d + high_d) / 2.0
        ds = mid_d * b
        q0 = (1.0 + ds * sp_eff)**(-1.0 / b)
        q1 = 2.0 - (1.0 + ds * (1.0 - sp_eff))**(-1.0 / b)
        y_val = (1.0 - q0) / max(1e-7, (q1 - q0))
        if y_val < target_background:
            low_d = mid_d
        else:
            high_d = mid_d
        best_d = mid_d

    return {
        "bp": float(bp),
        "sp": float(sp),
        "d": float(np.clip(best_d, 0.5, 15.0)),
        "b": float(b),
        "hp": float(hp),
        "lp": float(lp)
    }
