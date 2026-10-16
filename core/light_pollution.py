# core/light_pollution.py
"""
Módulo de Reducción Paramétrica de Contaminación Lumínica Terrestre.
Modela analíticamente cúpulas de luz generadas por asentamientos urbanos en el horizonte mediante superficies
elípticas/super-gaussianas 2D, con posición de centro (X, Y), radio horizontal/vertical y neutralización cromática
selectiva (vapor de sodio, lámparas de mercurio, LED cálido/frío o color de muestra interactivo).
"""

from typing import Optional, Tuple, Dict
import numpy as np
import cv2

# Preajustes cromáticos estándar de fuentes de iluminación urbana
LP_PRESET_TINTS: Dict[str, Tuple[float, float, float]] = {
    "neutral": (1.0, 1.0, 1.0),                  # Atenuación neutra monocromática
    "sodium": (1.0, 0.68, 0.22),                 # Vapor de sodio / ámbar intenso (589 nm)
    "warm_led": (1.0, 0.84, 0.52),               # Alumbrado LED urbano cálido (3000K-3500K)
    "cold_led": (0.75, 0.92, 1.0),               # LED frío comercial / vapor de mercurio (5000K+)
}


def compute_light_pollution_dome(
    h: int,
    w: int,
    center_x: float = 0.50,
    center_y: float = 0.90,
    width: float = 0.60,
    height: float = 0.45,
    exponent: float = 2.0
) -> np.ndarray:
    """
    Calcula una máscara 2D normalizada [0.0, 1.0] que representa el perfil espacial de la cúpula de luz.

    Parámetros:
    - h, w: Dimensiones de la imagen en píxeles.
    - center_x: Posición horizontal del centro de la ciudad en rango normalizado [0.0, 1.0] (0=izq, 1=der).
    - center_y: Posición vertical del centro (horizonte) en rango normalizado [0.0, 1.0] (0=cenit, 1=base).
    - width: Dispersión/radio horizontal (sigma_x) en escala relativa (típicamente 0.10 a 2.00).
    - height: Dispersión/altura vertical hacia el cenit (sigma_y) en escala relativa (0.10 a 1.50).
    - exponent: Forma de la campana (2.0 = Gaussiana clásica, 1.0 = Exponencial puntiaguda, 3.0 = Cúpula aplanada).
    """
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    dx = (xx - center_x) / max(0.01, width)
    dy = (yy - center_y) / max(0.01, height)

    # La cúpula se propaga hacia arriba en el cielo (yy <= center_y).
    # Por debajo del centro de horizonte (yy > center_y), se mantiene el valor máximo para que
    # al modular con la máscara de horizonte no se produzcan saltos ni halos oscuros en la silueta.
    dy = np.where(yy > center_y, 0.0, dy)

    dist_sq = dx**2 + dy**2
    if abs(exponent - 2.0) < 1e-3:
        dome = np.exp(-0.5 * dist_sq * 4.0)
    else:
        dome = np.exp(-0.5 * (dist_sq * 4.0)**(max(0.5, exponent) / 2.0))

    return dome.astype(np.float32)


def apply_parametric_light_pollution(
    img_rgb: np.ndarray,
    strength: float = 0.0,
    center_x: float = 0.50,
    center_y: float = 0.90,
    width: float = 0.60,
    height: float = 0.45,
    tint_rgb: Tuple[float, float, float] = (1.0, 0.68, 0.22),
    tint_strength: float = 0.50,
    sky_mask: Optional[np.ndarray] = None,
    mode: str = "attenuate",
    legacy_1d: bool = False
) -> np.ndarray:
    """
    Aplica reducción paramétrica 2D de contaminación lumínica en float32 [0.0, 1.0].

    Parámetros:
    - img_rgb: Imagen RGB en float32 [0.0, 1.0] (H, W, 3).
    - strength: Intensidad de reducción en [0.0, 1.0] (0.0 = apagado).
    - center_x, center_y: Coordenadas normalizadas [0.0, 1.0] del foco de la cúpula.
    - width, height: Amplitud horizontal y vertical de la cúpula.
    - tint_rgb: Tinte cromático dominante de la luz parásita (R, G, B).
    - tint_strength: Grado de neutralización cromática (0.0 = neutro, 1.0 = neutralización cromática completa).
    - sky_mask: Máscara de cielo en float32 [0.0, 1.0] (1.0 = cielo, 0.0 = suelo).
    - mode: "attenuate" (atenuación proporcional suave) o "subtractive" (sustracción física del resplandor aditivo).
    - legacy_1d: Si es True, ejecuta el algoritmo de rampa vertical 1D clásica (compatibilidad).
    """
    if strength <= 1e-4:
        return img_rgb

    h, w = img_rgb.shape[:2]

    # Modo compatible 1D clásico (rampa coseno vertical)
    if legacy_1d:
        y_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)
        grad_y = np.clip((y_coords - 0.50) / 0.50, 0.0, 1.0)
        curve_1d = 0.5 * (1.0 - np.cos(np.pi * grad_y)) * (strength * 0.70)
        weight_2d = np.tile(curve_1d[:, np.newaxis], (1, w))
        if sky_mask is not None:
            m = np.squeeze(sky_mask).astype(np.float32)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            weight_2d = weight_2d * m
        weight_3d = np.repeat(weight_2d[..., np.newaxis], 3, axis=2)
        attenuated = img_rgb * (1.0 - weight_3d)
        return np.clip(attenuated, 0.0, 1.0).astype(np.float32)

    # 1. Campo espacial 2D de la cúpula
    dome_2d = compute_light_pollution_dome(
        h, w,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height
    )

    # 2. Modulación con máscara de cielo si está disponible
    if sky_mask is not None:
        m = np.squeeze(sky_mask).astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        dome_2d = dome_2d * m

    # 3. Vectorización de tinte cromático normalizado
    t_arr = np.array(tint_rgb, dtype=np.float32)
    max_t = max(1e-4, float(np.max(t_arr)))
    t_norm = t_arr / max_t

    # Ponderación entre neutralidad blanca (1, 1, 1) y tinte cromático
    c_eff = (1.0 - tint_strength) * np.ones(3, dtype=np.float32) + tint_strength * t_norm
    c_eff = c_eff / max(1e-4, float(np.max(c_eff)))

    # Campo ponderado 3D (H, W, 3)
    weight_3d = dome_2d[..., np.newaxis] * c_eff[np.newaxis, np.newaxis, :]

    # 4. Aplicación de la reducción
    if mode == "subtractive":
        # Sustracción aditiva calibrada al pedestal de fondo
        # strength modula la cantidad de flujo parásito eliminado
        pedestal_scale = 0.12 * strength
        out = img_rgb - (weight_3d * pedestal_scale)
    else:
        # Modo por defecto "attenuate" (atenuación multiplicativa de brillo y saturación espuria)
        # Factor máximo de atenuación al 100% de fuerza: 75%
        decay_factor = weight_3d * (strength * 0.75)
        out = img_rgb * (1.0 - decay_factor)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def estimate_light_pollution_center(
    img_rgb: np.ndarray,
    sky_mask: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Analiza la imagen para detectar automáticamente la ubicación más probable del centro de la cúpula de luz
    en el horizonte (X, Y normalizados en [0.0, 1.0]).
    """
    h, w = img_rgb.shape[:2]
    # Muestreo a baja resolución para análisis robusto inmune a estrellas individuales
    small_h, small_w = 120, 160
    small_img = cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Luminancia ponderada
    lum = 0.2126 * small_img[..., 0] + 0.7152 * small_img[..., 1] + 0.0722 * small_img[..., 2]

    if sky_mask is not None:
        small_mask = cv2.resize(sky_mask, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        lum = lum * small_mask

    # Desenfoque gaussiano amplio para aislar el resplandor difuso del cielo
    blurred = cv2.GaussianBlur(lum, (31, 31), 10.0)

    # Buscar el pico en el 60% inferior de la toma (zona cercana al horizonte)
    start_y = int(small_h * 0.40)
    sub_region = blurred[start_y:, :]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sub_region)

    best_x = float(max_loc[0]) / float(small_w)
    best_y = float(max_loc[1] + start_y) / float(small_h)

    return (
        float(np.clip(best_x, 0.05, 0.95)),
        float(np.clip(best_y, 0.50, 0.98))
    )
