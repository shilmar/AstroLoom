# core/orton.py
"""
Módulo de Efecto Orton Astronómico y Resplandor Específico para Polvo Galáctico (Galactic Dust Glow).
Diseñado específicamente para astrofotografía:
1. Conciencia estelar: opera sobre capas sin estrellas (Starless) o aísla picos puntuales estelares
   mediante filtrado morfológico de alta frecuencia, protegiendo las estrellas para que no se
   hinchen ni pierdan su nitidez puntual.
2. Umbral de emisión: aísla las estructuras de polvo cósmico y nubes moleculares protegiendo
   el fondo negro espacial de volverse lechoso o lavado.
3. Difusión multi-escala: desenfoque compuesto de caída natural (celestial bloom).
4. Coloración cromática: tintes dorado cósmico, neutro, H-alfa rubí y azul de reflexión.
5. Fusión tonal: modos Luz Suave (Soft Light), Trama (Screen) y Aditivo acumulativo.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import cv2

ORTON_PRESET_TINTS: Dict[str, Tuple[float, float, float]] = {
    "warm_gold": (1.0, 0.85, 0.55),   # Núcleo galáctico (Sagitario/Escorpio) y polvo molecular (Rho Ophiuchi)
    "neutral":   (1.0, 1.0, 1.0),     # Luminancia neutra monocromática sin desviación de balance
    "cool_blue": (0.70, 0.85, 1.0),   # Zonas de reflexión nebular y cúmulos estelares jóvenes
    "halpha":    (1.0, 0.40, 0.65),   # Emisión de hidrógeno ionizado (H-alfa / rubí cósmico)
}


def apply_astronomical_orton(
    img_rgb: np.ndarray,
    strength: float = 0.0,
    radius: float = 40.0,
    threshold: float = 0.15,
    exponent: float = 1.5,
    tint_type: str = "warm_gold",
    custom_tint: Tuple[float, float, float] = (1.0, 0.85, 0.55),
    tint_strength: float = 0.40,
    blend_mode: str = "soft_light",
    sky_mask: Optional[np.ndarray] = None,
    is_starless: bool = False
) -> np.ndarray:
    """
    Aplica el Efecto Orton Astronómico y Glow de Polvo Galáctico sobre una imagen RGB en float32 [0.0, 1.0].

    Parámetros:
    - img_rgb: Imagen RGB en float32 [0.0, 1.0] (H, W, 3).
    - strength: Intensidad de fusión del resplandor en [0.0, 1.0] (0.0 = desactivado).
    - radius: Radio del desenfoque difuso en píxeles (sigma espacial, 10.0 a 150.0).
    - threshold: Umbral mínimo de luminancia a partir del cual el polvo y nebulosas emiten resplandor [0.0, 0.60].
    - exponent: Curva de respuesta no lineal del resplandor (1.0 = lineal, 2.0 = cuadrático/focalizado).
    - tint_type: Clave en ORTON_PRESET_TINTS ("warm_gold", "neutral", "cool_blue", "halpha") o "custom".
    - custom_tint: Tupla (R, G, B) utilizada cuando tint_type == "custom".
    - tint_strength: Grado de modulación cromática del resplandor en [0.0, 1.0].
    - blend_mode: "soft_light" (contraste misterioso y aterciopelado), "screen" (velo etéreo luminoso) o "additive".
    - sky_mask: Máscara 2D de cielo [0.0, 1.0] (1.0 = cielo, 0.0 = suelo) para proteger el paisaje terrestre.
    - is_starless: Si es True, la imagen de entrada ya carece de estrellas (starless desacoplado con StarNet),
                   por lo que no requiere supresión morfológica de estrellas.
    """
    if strength <= 1e-4:
        return img_rgb

    h, w = img_rgb.shape[:2]

    # 1. Luminancia perceptiva ponderada
    luma = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]

    # 2. Conciencia estelar: si la imagen tiene estrellas, aislamos la emisión difusa
    if is_starless:
        glow_base = img_rgb.copy()
        diffuse_luma = luma
    else:
        # Supresión de picos puntuales estelares mediante filtro morfológico de apertura
        # Kernel elíptico pequeño (5x5) para remover estrellas puntuales sin dañar nebulosas
        k_size = max(3, int(round(min(h, w) * 0.005)) | 1)
        k_size = min(k_size, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        diffuse_luma = cv2.morphologyEx(luma, cv2.MORPH_OPEN, kernel)

        # Máscara de estrellas detectadas
        star_peak = np.clip(luma - diffuse_luma, 0.0, 1.0)
        # Atenuamos las estrellas en la fuente del resplandor para evitar que se expandan
        star_attenuation = 1.0 - (star_peak * 0.90)
        glow_base = img_rgb * star_attenuation[..., np.newaxis]

    # 3. Puerta de emisión por umbral: sólo el polvo y zonas iluminadas generan glow
    # Las zonas profundas del cielo (luma < threshold) quedan completamente protegidas
    denom = max(1e-4, 1.0 - threshold)
    emission = np.clip((diffuse_luma - threshold) / denom, 0.0, 1.0)
    if abs(exponent - 1.0) > 1e-3:
        emission = np.power(emission, max(0.5, exponent))

    glow_source = glow_base * emission[..., np.newaxis]

    # 4. Difusión multi-escala (Celestial Bloom)
    # Combinamos dos componentes gaussianos: uno medio (sigma=radius) y otro amplio (sigma=2.5*radius)
    # para producir una caída exponencial orgánica y realista
    sigma1 = max(1.0, float(radius))
    sigma2 = max(2.5, float(radius * 2.5))

    # Optimizamos el cálculo reduciendo resolución si sigma es muy amplio
    if min(h, w) > 800 and sigma1 > 20.0:
        scale = 0.5
        small_w = max(64, int(w * scale))
        small_h = max(64, int(h * scale))
        small_src = cv2.resize(glow_source, (small_w, small_h), interpolation=cv2.INTER_AREA)

        blur1 = cv2.GaussianBlur(small_src, (0, 0), sigmaX=sigma1 * scale, sigmaY=sigma1 * scale)
        blur2 = cv2.GaussianBlur(small_src, (0, 0), sigmaX=sigma2 * scale, sigmaY=sigma2 * scale)
        small_bloom = (blur1 * 0.65) + (blur2 * 0.35)

        bloom = cv2.resize(small_bloom, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        blur1 = cv2.GaussianBlur(glow_source, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
        blur2 = cv2.GaussianBlur(glow_source, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
        bloom = (blur1 * 0.65) + (blur2 * 0.35)

    # 5. Modulación cromática del Glow
    raw_tint = ORTON_PRESET_TINTS.get(tint_type, custom_tint)
    t_arr = np.array(raw_tint, dtype=np.float32)
    max_t = max(1e-4, float(np.max(t_arr)))
    t_norm = t_arr / max_t

    c_eff = (1.0 - tint_strength) * np.ones(3, dtype=np.float32) + tint_strength * t_norm
    c_eff = c_eff / max(1e-4, float(np.max(c_eff)))
    bloom = bloom * c_eff[np.newaxis, np.newaxis, :]

    # 6. Protección del paisaje terrestre con sky_mask
    if sky_mask is not None:
        m = np.squeeze(sky_mask).astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        bloom = bloom * m[..., np.newaxis]

    # 7. Fusión Tonal según blend_mode
    if blend_mode == "screen":
        # Modo Trama (Screen): 1 - (1 - I) * (1 - G)
        blended = 1.0 - (1.0 - img_rgb) * (1.0 - bloom)
    elif blend_mode == "additive":
        # Modo Aditivo directo: I + G
        blended = img_rgb + bloom
    else:
        # Modo por defecto "soft_light": fórmula continua de Pegtop
        # (1 - 2*G)*I^2 + 2*G*I
        blended = (1.0 - 2.0 * bloom) * (img_rgb ** 2) + 2.0 * bloom * img_rgb

    # 8. Interpolación ponderada por la intensidad (strength)
    out = (1.0 - strength) * img_rgb + strength * blended

    return np.clip(out, 0.0, 1.0).astype(np.float32)
