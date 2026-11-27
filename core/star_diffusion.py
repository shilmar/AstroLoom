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
import astroloom_rs

HAS_RUST = True


def apply_star_diffusion(
    img_rgb: np.ndarray,
    strength: float = 0.0,
    radius: float = 20.0,
    threshold: float = 0.45,
    color_boost: float = 0.50,
    halo_mode: str = "screen",
    sky_mask: Optional[np.ndarray] = None,
    stars_layer: Optional[np.ndarray] = None,
    ref_min_dim: Optional[int] = None,
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
        st_src = stars_layer
        if st_src.shape[:2] != (h, w):
            st_src = cv2.resize(st_src, (w, h), interpolation=cv2.INTER_LINEAR)
        star_peaks = 0.2126 * st_src[..., 0] + 0.7152 * st_src[..., 1] + 0.0722 * st_src[..., 2]
        emission_base = st_src
    else:
        # Modo imagen única: filtro morfológico Top-Hat (Luma - Apertura) para aislar estrellas puntuales
        # ignorando por completo el gas de la Vía Láctea y nebulosas extendidas
        luma = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
        base_dim = int(ref_min_dim) if ref_min_dim is not None else min(h, w)
        k_size = max(5, int(round(base_dim * 0.007)) | 1)
        k_size = min(k_size, 19)
        if k_size >= 7 and min(h, w) >= 256:
            # Apertura en 0.5x para extraer el fondo difuso ~16x más rápido
            sw, sh = max(32, w // 2), max(32, h // 2)
            ks = max(3, (k_size // 2) | 1)
            luma_s = cv2.resize(luma, (sw, sh), interpolation=cv2.INTER_AREA)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            diffuse = cv2.resize(cv2.morphologyEx(luma_s, cv2.MORPH_OPEN, kernel), (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            diffuse = cv2.morphologyEx(luma, cv2.MORPH_OPEN, kernel)
        star_peaks = np.clip(luma - diffuse, 0.0, 1.0)
        emission_base = img_rgb

    # 2. Protección del horizonte y paisaje terrestre mediante sky_mask
    m = None
    if sky_mask is not None:
        m = np.squeeze(sky_mask).astype(np.float32, copy=False)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3-6a. Puerta de magnitud + recuperación cromática + dilatación 3x3 fusionadas en Rust
    m_c = np.ascontiguousarray(m, dtype=np.float32) if m is not None else None
    seed_source = astroloom_rs.prepare_star_diffusion_seed_rs(
        np.ascontiguousarray(emission_base, dtype=np.float32),
        np.ascontiguousarray(star_peaks, dtype=np.float32),
        float(threshold),
        float(color_boost),
        m_c,
    )
    if seed_source is None:
        return img_rgb

    sigma1 = max(1.0, float(radius * 0.55))
    sigma2 = max(2.5, float(radius * 1.75))

    # Optimización downscale adaptativo según sigma (equivalente visualmente a escala completa)
    min_dim = min(h, w)
    if min_dim > 400 and sigma1 > 10.0:
        scale1 = 0.25
    elif min_dim > 250 and sigma1 > 3.0:
        scale1 = 0.5
    else:
        scale1 = 1.0

    if scale1 < 1.0:
        s1_w = max(64, int(w * scale1))
        s1_h = max(64, int(h * scale1))
        src1 = cv2.resize(seed_source, (s1_w, s1_h), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(src1, (0, 0), sigmaX=sigma1 * scale1, sigmaY=sigma1 * scale1)

        if min_dim > 512 and sigma2 > 28.0:
            scale2 = 0.125
        elif min_dim > 400 and sigma2 > 10.0:
            scale2 = 0.25
        else:
            scale2 = scale1

        if scale2 < scale1:
            s2_w = max(32, int(w * scale2))
            s2_h = max(32, int(h * scale2))
            src2 = cv2.resize(src1, (s2_w, s2_h), interpolation=cv2.INTER_AREA)
            b2_s = cv2.GaussianBlur(src2, (0, 0), sigmaX=sigma2 * scale2, sigmaY=sigma2 * scale2)
            b2 = cv2.resize(b2_s, (s1_w, s1_h), interpolation=cv2.INTER_LINEAR)
        else:
            b2 = cv2.GaussianBlur(src1, (0, 0), sigmaX=sigma2 * scale1, sigmaY=sigma2 * scale1)

        small_halo = (b1 * 0.60) + (b2 * 0.40)
        halo = cv2.resize(small_halo, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        b1 = cv2.GaussianBlur(seed_source, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
        b2 = cv2.GaussianBlur(seed_source, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
        halo = (b1 * 0.60) + (b2 * 0.40)

    # Factor de escala energético para halo visible y proporcionado
    halo *= 3.8

    return astroloom_rs.blend_orton_and_diffusion_rs(
        np.ascontiguousarray(img_rgb, dtype=np.float32),
        np.ascontiguousarray(halo, dtype=np.float32),
        tint=None,
        sky_mask=m_c,
        blend_mode=str(halo_mode),
        strength=float(strength)
    )
