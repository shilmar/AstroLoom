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
    x = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)

    # 1. Ajuste preliminar de Punto Negro (BP)
    if bp > 0.0:
        x_adj = np.clip((x - bp) / max(1e-6, 1.0 - bp), 0.0, 1.0)
        sp_eff = float(np.clip((sp - bp) / max(1e-6, 1.0 - bp), 1e-5, 1.0 - 1e-5))
    elif bp < 0.0:
        x_adj = np.clip(x - bp, 0.0, 1.0)
        sp_eff = float(np.clip(sp - bp, 1e-5, 1.0 - 1e-5))
    else:
        x_adj = x
        sp_eff = float(np.clip(sp, 1e-5, 1.0 - 1e-5))

    # Si la intensidad de estirado es nula, devolver directamente la rampa lineal
    if d <= 1e-6:
        return x_adj.astype(np.float32)

    d = float(max(0.0, d))
    b = float(max(0.0, b))
    hp = float(np.clip(hp, 0.0, 0.99))
    lp = float(np.clip(lp, 0.0, 0.99))

    # Umbrales analíticos de protección
    hp_thresh = 1.0 - hp * (1.0 - sp_eff)
    lp_thresh = lp * sp_eff

    ds = d * max(1e-6, b) if b > 0.0 else d
    bs = max(1e-6, b)

    # Curva GHS base por ramas inferior y superior simétricas
    def q_raw(val):
        diff_lower = np.maximum(0.0, sp_eff - val)
        diff_upper = np.maximum(0.0, val - sp_eff)
        if b <= 1e-6:
            ql = np.exp(-d * diff_lower)
            qu = 2.0 - np.exp(-d * diff_upper)
        else:
            ql = (1.0 + ds * diff_lower)**(-1.0 / bs)
            qu = 2.0 - (1.0 + ds * diff_upper)**(-1.0 / bs)
        return np.where(val < sp_eff, ql, qu)

    # Primera derivada analítica para empalme C1 suave en zonas de protección
    def q_deriv(val):
        diff_lower = np.maximum(0.0, sp_eff - val)
        diff_upper = np.maximum(0.0, val - sp_eff)
        if b <= 1e-6:
            dl = d * np.exp(-d * diff_lower)
            du = d * np.exp(-d * diff_upper)
        else:
            dl = d * (1.0 + ds * diff_lower)**(-1.0 / bs - 1.0)
            du = d * (1.0 + ds * diff_upper)**(-1.0 / bs - 1.0)
        return np.where(val < sp_eff, dl, du)

    q_hp = q_raw(hp_thresh)
    s_hp = q_deriv(hp_thresh)

    q_lp = q_raw(lp_thresh)
    s_lp = q_deriv(lp_thresh)

    def eval_q(v):
        base = q_raw(v)
        res = np.where(v > hp_thresh, q_hp + s_hp * (v - hp_thresh), base)
        res = np.where(v < lp_thresh, q_lp - s_lp * (lp_thresh - v), res)
        return res

    q0 = eval_q(0.0)
    q1 = eval_q(1.0)

    qv = eval_q(x_adj)
    denom = max(1e-7, float(q1 - q0))
    lut = np.clip((qv - q0) / denom, 0.0, 1.0).astype(np.float32)
    return lut


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

    Parámetros:
    - img: Imagen en float32 [0.0, 1.0] (H, W) o (H, W, 3).
    - sp, d, b, hp, lp, bp: Parámetros del modelo GHS.
    - color_mode: "luma" (preserva saturación y matices calculando el estirado sobre la luminancia Rec. 709)
                  o "rgb" (aplica el estirado a cada canal RGB de forma vinculada).
    - precomputed_lut: LUT ya calculada para evitar recomputaciones redundantes.
    """
    if precomputed_lut is not None:
        lut = precomputed_lut
    else:
        lut = compute_ghs_lut(sp=sp, d=d, b=b, hp=hp, lp=lp, bp=bp, lut_size=lut_size)

    max_idx = len(lut) - 1

    # Caso monocromo / 2D
    if img.ndim == 2:
        idx = np.clip(img * max_idx, 0, max_idx).astype(np.uint16)
        return lut[idx].astype(np.float32)

    # Caso RGB
    if color_mode == "luma":
        # Luminancia estándar Rec. 709
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        lum = np.clip(lum, 0.0, 1.0)
        idx_lum = np.clip(lum * max_idx, 0, max_idx).astype(np.uint16)
        lum_stretched = lut[idx_lum]
        ratio = (lum_stretched + 1e-6) / (lum + 1e-6)
        out = img * ratio[..., np.newaxis]
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    else:
        # Modo RGB directo canal por canal
        idx = np.clip(img * max_idx, 0, max_idx).astype(np.uint16)
        return lut[idx].astype(np.float32)


def estimate_ghs_params(img_float: np.ndarray, target_background: float = 0.18) -> Dict[str, float]:
    """
    Estima analíticamente los parámetros óptimos de GHS para una imagen lineal de astrofotografía:
    - SP: Centrado en la mediana del fondo del cielo.
    - BP: Derivado de la dispersión de ruido (mediana - 2.5 * MAD).
    - D: Calculado para elevar el fondo a target_background (~0.18) sin reventar estrellas.
    - b: Configurado en 2.0 para una compresión equilibrada.
    - HP: 0.15 para protección inicial de centros de estrellas.
    """
    sample = img_float[::4, ::4]
    if sample.ndim == 3:
        lum_sample = 0.2126 * sample[..., 0] + 0.7152 * sample[..., 1] + 0.0722 * sample[..., 2]
    else:
        lum_sample = sample

    med = float(np.median(lum_sample))
    mad = float(np.median(np.abs(lum_sample - med)))

    bp = float(np.clip(med - 2.5 * mad, 0.0, 0.90))
    sp = float(np.clip(med, 0.0001, 0.90))

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
