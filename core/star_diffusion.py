# core/star_diffusion.py
"""
Módulo de Realce Espectral de Estrellas y Filtro Difusor Digital (Digital Star Diffusion / Mist Filter).
Emula de forma analítica y digital los prestigiosos filtros ópticos físicos de difusión estelar
(como Kenko Pro1D Pro Softon-A, Kase Star Diffusion o Lee Soft Mist) y la función "Enhance Star Light" de Sequator:

1. Aislamiento selectivo de estrellas prominentes (magnitudes 1ª a 4ª) mediante análisis morfológico
   de picos de alta frecuencia (Top-Hat) o a través de la capa de estrellas desacoplada con StarNet++.
2. Protección de estrellas débiles: las millones de estrellas diminutas del fondo permanecen puntuales y nítidas,
   sin generar niebla ni inflarse.
3. Recuperación cromática espectral: extrae el color intrínseco de los bordes del disco estelar (clases espectrales
   O, B, A, F, G, K, M) e infunde halos cromáticos enriquecidos (ej. el naranja cálido de Betelgeuse o el azul de Rigel).
4. Difusión óptica multi-escala: emulación de dispersión gaussiana continua con caída natural de luz difusa.
5. Fusión suave mediante modos Trama (Screen), Luz Suave o Aditivo, respetando máscaras de cielo y horizonte.
"""

from typing import Optional, Tuple
import numpy as np
import cv2


def apply_star_diffusion(
    img_rgb: np.ndarray,
    strength: float = 0.0,
    radius: float = 20.0,
    threshold: float = 0.45,
    color_boost: float = 0.50,
    halo_mode: str = "screen",
    sky_mask: Optional[np.ndarray] = None,
    stars_layer: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Aplica el Filtro Difusor Digital y Realce Espectral de Estrellas sobre una imagen RGB en float32 [0.0, 1.0].

    Parámetros:
    - img_rgb: Imagen RGB en float32 [0.0, 1.0] (H, W, 3).
    - strength: Intensidad de fusión del filtro en [0.0, 1.0] (0.0 = desactivado / identidad).
    - radius: Radio espacial base del halo difuso en píxeles (5.0 a 60.0 px).
    - threshold: Umbral mínimo de magnitud estelar relativa [0.15, 0.85].
                 Valores altos (0.60-0.80) restringen el halo a las estrellas más brillantes de las constelaciones.
                 Valores medios (0.35-0.55) incluyen estrellas secundarias.
    - color_boost: Saturación cromática espectral del halo [0.0, 1.0].
                   0.0 = halo blanco neutro; 1.0 = máxima saturación del color intrínseco de la estrella.
    - halo_mode: Modo de fusión del resplandor: "screen" (Trama, etéreo y sin quemar),
                 "soft_light" (Luz Suave, aterciopelado) o "additive" (Aditivo, radiante).
    - sky_mask: Máscara 2D de cielo [0.0, 1.0] para evitar que luces terrestres o el paisaje generen halos.
    - stars_layer: Opcional. Capa RGB de solo estrellas (float32) desacoplada con StarNet++.
                   Si está presente, se aíslan las estrellas directamente desde ella sin riesgo de capturar nebulosas.
    """
    if strength <= 1e-4:
        return img_rgb

    h, w = img_rgb.shape[:2]

    # 1. Aislamiento de picos estelares puntuales
    if stars_layer is not None:
        # Modo con estrellas desacopladas: la capa ya contiene únicamente estrellas sobre fondo negro
        st_src = stars_layer.copy()
        if st_src.shape[:2] != (h, w):
            st_src = cv2.resize(st_src, (w, h), interpolation=cv2.INTER_LINEAR)
        star_luma = 0.2126 * st_src[..., 0] + 0.7152 * st_src[..., 1] + 0.0722 * st_src[..., 2]
        star_peaks = star_luma
        emission_base = st_src
    else:
        # Modo imagen única: filtro morfológico Top-Hat (Luma - Apertura) para aislar estrellas puntuales
        # ignorando por completo el gas de la Vía Láctea y nebulosas extendidas
        luma = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
        k_size = max(5, int(round(min(h, w) * 0.007)) | 1)
        k_size = min(k_size, 19)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        diffuse = cv2.morphologyEx(luma, cv2.MORPH_OPEN, kernel)
        star_peaks = np.clip(luma - diffuse, 0.0, 1.0)
        emission_base = img_rgb.copy()

    # 2. Protección del horizonte y paisaje terrestre mediante sky_mask
    if sky_mask is not None:
        m = np.squeeze(sky_mask).astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        star_peaks = star_peaks * m
        emission_base = emission_base * m[..., np.newaxis]

    # 3. Puerta de magnitud relativa por umbral con curva suave Hermite (smoothstep)
    # Las estrellas tenues de fondo (peaks < threshold) reciben peso 0.0 absoluto
    thr = max(0.05, float(threshold))
    knee = 0.12
    t = np.clip((star_peaks - thr) / knee, 0.0, 1.0)
    star_weight = t * t * (3.0 - 2.0 * t)

    # Si no hay estrellas por encima del umbral, retornamos sin modificar
    if np.max(star_weight) < 1e-4:
        return img_rgb

    # 4. Fuente de emisión estelar
    star_source = emission_base * star_weight[..., np.newaxis]

    # 5. Recuperación cromática espectral (Star Spectral Color Recovery)
    if color_boost > 1e-4:
        # Las estrellas muy brillantes tienen núcleos saturados a blanco puro (V > 0.65, S < 0.25).
        # Los píxeles del perímetro de la estrella conservan su matiz (H) y saturación (S) reales.
        # Difundimos el color cromático localmente para que el halo refleje el tipo espectral de la estrella.
        hsv = cv2.cvtColor(np.clip(emission_base, 0.0, 1.0), cv2.COLOR_RGB2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        rad_h = np.deg2rad(H)
        u = S * np.cos(rad_h)
        v = S * np.sin(rad_h)

        valid_color = (star_peaks > 0.12) & (S > 0.06)
        if np.any(valid_color):
            u_src = np.where(valid_color, u, 0.0).astype(np.float32)
            v_src = np.where(valid_color, v, 0.0).astype(np.float32)
            weight_src = valid_color.astype(np.float32)

            k_col = 7
            u_b = cv2.GaussianBlur(u_src, (k_col, k_col), sigmaX=2.0)
            v_b = cv2.GaussianBlur(v_src, (k_col, k_col), sigmaX=2.0)
            w_b = cv2.GaussianBlur(weight_src, (k_col, k_col), sigmaX=2.0) + 1e-4

            u_rec = u_b / w_b
            v_rec = v_b / w_b
            rec_S = np.clip(np.sqrt(u_rec**2 + v_rec**2) * (1.0 + color_boost * 2.0), 0.0, 1.0)
            rec_H = (np.rad2deg(np.arctan2(v_rec, u_rec)) % 360.0)

            # Infundir en núcleos quemados o con baja saturación
            is_desaturated_star = (V > 0.50) & (star_peaks > (thr * 0.8))
            infusion = np.clip(is_desaturated_star.astype(np.float32) * (color_boost * 1.2), 0.0, 1.0)

            final_H = np.where(infusion > 0, rec_H, H)
            final_S = np.where(infusion > 0, np.maximum(S, rec_S), S * (1.0 + color_boost * 1.5))
            final_hsv = np.stack([final_H, np.clip(final_S, 0.0, 1.0), V], axis=-1)
            boosted_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            star_source = boosted_rgb * star_weight[..., np.newaxis]

    # 6. Difusión óptica multi-escala del halo estelar (Star Mist Halo)
    # Semilla de micro-difracción: dilatación elíptica suave de 3x3 para expandir la dispersión
    k_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed_source = cv2.dilate(star_source, k_seed)

    sigma1 = max(1.0, float(radius * 0.55))
    sigma2 = max(2.5, float(radius * 1.75))

    # Optimización downscale si la resolución o el radio justifican aceleración
    if (min(h, w) > 250 and sigma1 > 3.0):
        scale = 0.5
        small_w = max(64, int(w * scale))
        small_h = max(64, int(h * scale))
        small_src = cv2.resize(seed_source, (small_w, small_h), interpolation=cv2.INTER_AREA)

        b1 = cv2.GaussianBlur(small_src, (0, 0), sigmaX=sigma1 * scale, sigmaY=sigma1 * scale)
        b2 = cv2.GaussianBlur(small_src, (0, 0), sigmaX=sigma2 * scale, sigmaY=sigma2 * scale)
        small_halo = (b1 * 0.60) + (b2 * 0.40)

        halo = cv2.resize(small_halo, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        b1 = cv2.GaussianBlur(seed_source, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
        b2 = cv2.GaussianBlur(seed_source, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
        halo = (b1 * 0.60) + (b2 * 0.40)

    # Factor de escala energético para halo visible y proporcionado
    halo = halo * 3.8

    # Re-aplicar máscara de cielo al halo resultante para que no desborde hacia el suelo
    if sky_mask is not None:
        halo = halo * m[..., np.newaxis]

    # 7. Fusión Tonal según halo_mode
    if halo_mode == "soft_light":
        # Modo Luz Suave: I + 2.0 * H * I * (1.0 - I)
        # Resplandor aterciopelado con cero oscurecimiento
        blended = img_rgb + 2.0 * halo * img_rgb * (1.0 - img_rgb)
    elif halo_mode == "additive":
        # Modo Aditivo directo: I + H
        blended = img_rgb + halo
    else:
        # Modo por defecto "screen": Trama 1.0 - (1.0 - I) * (1.0 - H)
        # Velo etéreo, luminoso y progresivo sin artefactos de recorte
        blended = 1.0 - (1.0 - img_rgb) * (1.0 - np.clip(halo, 0.0, 1.0))

    # 8. Interpolación ponderada por la intensidad (strength)
    out = (1.0 - strength) * img_rgb + strength * blended

    return np.clip(out, 0.0, 1.0).astype(np.float32)
