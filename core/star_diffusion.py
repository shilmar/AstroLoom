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

try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


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
    # 5. Recuperación cromática espectral dispersa (solo píxeles activos, evita HSV sobre fondo negro)
    star_source = None
    if color_boost > 1e-4:
        act_y, act_x = np.nonzero(star_peaks > 0.12)
        if act_y.size > 0:
            rgb_act = np.clip(emission_base[act_y, act_x], 0.0, 1.0).reshape(-1, 1, 3)
            hsv_act = cv2.cvtColor(rgb_act, cv2.COLOR_RGB2HSV).reshape(-1, 3)
            H_act, S_act, _ = hsv_act[:, 0], hsv_act[:, 1], hsv_act[:, 2]

            valid_sat = S_act > 0.06
            if np.any(valid_sat):
                vy = act_y[valid_sat]
                vx = act_x[valid_sat]
                rad_h = np.deg2rad(H_act[valid_sat])
                s_v = S_act[valid_sat]

                # Empaquetamos (u, v, weight) en 1 array de 3 canales para 1 solo GaussianBlur
                uvw = np.zeros((h, w, 3), dtype=np.float32)
                uvw[vy, vx, 0] = s_v * np.cos(rad_h)
                uvw[vy, vx, 1] = s_v * np.sin(rad_h)
                uvw[vy, vx, 2] = 1.0

                uvw_b = cv2.GaussianBlur(uvw, (7, 7), sigmaX=2.0)

                # Solo reconstruimos HSV->RGB donde star_weight > 0 (típ. <1% de píxeles)
                ey, ex = np.nonzero(star_weight > 1e-4)
                if ey.size > 0:
                    rgb_em = np.clip(emission_base[ey, ex], 0.0, 1.0).reshape(-1, 1, 3)
                    hsv_em = cv2.cvtColor(rgb_em, cv2.COLOR_RGB2HSV).reshape(-1, 3)
                    H_em, S_em, V_em = hsv_em[:, 0], hsv_em[:, 1], hsv_em[:, 2]

                    wb = uvw_b[ey, ex, 2] + 1e-4
                    u_rec = uvw_b[ey, ex, 0] / wb
                    v_rec = uvw_b[ey, ex, 1] / wb
                    rec_S = np.clip(np.sqrt(u_rec * u_rec + v_rec * v_rec) * (1.0 + color_boost * 2.0), 0.0, 1.0)
                    rec_H = np.rad2deg(np.arctan2(v_rec, u_rec)) % 360.0

                    is_desat = (V_em > 0.50) & (star_peaks[ey, ex] > (thr * 0.8))
                    final_H = np.where(is_desat, rec_H, H_em)
                    final_S = np.clip(
                        np.where(is_desat, np.maximum(S_em, rec_S), S_em * (1.0 + color_boost * 1.5)),
                        0.0, 1.0
                    )
                    final_hsv = np.stack([final_H, final_S, V_em], axis=-1).astype(np.float32).reshape(-1, 1, 3)
                    boosted_em = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)

                    star_source = np.zeros((h, w, 3), dtype=np.float32)
                    star_source[ey, ex] = boosted_em * star_weight[ey, ex, np.newaxis]

    if star_source is None:
        star_source = emission_base * star_weight[..., np.newaxis]

    # 6. Difusión óptica multi-escala del halo estelar (Star Mist Halo)
    # Semilla de micro-difracción: dilatación elíptica suave de 3x3 para expandir la dispersión
    k_seed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed_source = cv2.dilate(star_source, k_seed)

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
    halo = halo * 3.8

    if HAS_RUST and hasattr(astroloom_rs, "blend_orton_and_diffusion_rs"):
        try:
            m_c = None
            if sky_mask is not None:
                m = np.squeeze(sky_mask).astype(np.float32)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                m_c = np.ascontiguousarray(m, dtype=np.float32)

            return astroloom_rs.blend_orton_and_diffusion_rs(
                np.ascontiguousarray(img_rgb, dtype=np.float32),
                np.ascontiguousarray(halo, dtype=np.float32),
                tint=None,
                sky_mask=m_c,
                blend_mode=str(halo_mode),
                strength=float(strength)
            )
        except Exception:
            pass

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
