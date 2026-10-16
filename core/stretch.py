# core/stretch.py
import numpy as np
import cv2

from core.ghs import compute_ghs_lut, apply_ghs_stretch, estimate_ghs_params
from core.light_pollution import (
    compute_light_pollution_dome,
    apply_parametric_light_pollution,
    estimate_light_pollution_center,
    LP_PRESET_TINTS
)
from core.orton import (
    apply_astronomical_orton,
    ORTON_PRESET_TINTS
)
from core.star_diffusion import apply_star_diffusion

def adjust_saturation_dual(
    img_rgb: np.ndarray, 
    sat_sky: float = 1.0, 
    sat_ground: float = 1.0, 
    sky_mask: np.ndarray = None
) -> np.ndarray:
    """
    Ajusta la saturación en espacio HSV en float32 de forma independiente para cielo y suelo.
    sat_sky, sat_ground: 1.0 es original, 0.0 es escala de grises, 2.0 es el doble de saturación.
    """
    if sat_sky == 1.0 and (sat_ground == 1.0 or sky_mask is None):
        return img_rgb

    # Conversión a HSV en rango [0.0, 1.0]
    hsv = cv2.cvtColor(np.clip(img_rgb, 0.0, 1.0), cv2.COLOR_RGB2HSV)

    if sky_mask is None or sat_sky == sat_ground:
        # Ajuste global directo
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_sky, 0.0, 1.0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 1. Crear versión saturada para cielo y versión saturada para suelo
    hsv_sky = hsv.copy()
    hsv_sky[..., 1] = np.clip(hsv_sky[..., 1] * sat_sky, 0.0, 1.0)
    rgb_sky = cv2.cvtColor(hsv_sky, cv2.COLOR_HSV2RGB)

    hsv_gnd = hsv.copy()
    hsv_gnd[..., 1] = np.clip(hsv_gnd[..., 1] * sat_ground, 0.0, 1.0)
    rgb_gnd = cv2.cvtColor(hsv_gnd, cv2.COLOR_HSV2RGB)

    # 2. Suavizado de máscara para evitar artefactos o halos en la silueta (horizonte/árboles)
    h, w = img_rgb.shape[:2]
    if sky_mask.shape != (h, w):
        mask_aligned = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask_aligned = sky_mask

    # Desenfoque gaussiano suave adaptado a la resolución (kernel impar ~15-25 px)
    ksize = int(max(15, (min(h, w) // 150) | 1))
    smooth_mask = cv2.GaussianBlur(mask_aligned, (ksize, ksize), sigmaX=ksize / 3.0)
    smooth_mask = np.repeat(smooth_mask[..., np.newaxis], 3, axis=2)

    # 3. Mezcla lineal continua
    blended = (rgb_sky * smooth_mask) + (rgb_gnd * (1.0 - smooth_mask))
    return np.clip(blended, 0.0, 1.0)

def calculate_mtf_params(img_float: np.ndarray, target_background: float = 0.20):
    sample = img_float[::4, ::4]
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    
    bp = float(np.clip(med - 1.5 * mad, 0.0, 0.95))
    
    norm_med = max(1e-6, (med - bp) / max(1e-6, 1.0 - bp))
    denom = norm_med * (2.0 * target_background - 1.0) - target_background
    if abs(denom) < 1e-7:
        m = 0.5
    else:
        m = (norm_med * (target_background - 1.0)) / denom
    
    # Límite mínimo para evitar sobreexposición destructiva en presencia de cúpulas de luz
    m = float(np.clip(m, 0.0050, 0.9999))
    return bp, m

def mtf_curve(x: np.ndarray, m: float) -> np.ndarray:
    """Función de transferencia de medios tonos estándar (Midtone Transfer Function)."""
    # MTF(x, m) = (m - 1) * x / ((2m - 1) * x - m)
    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m
    return np.where(np.abs(den) > 1e-7, num / den, 0.0)

def auto_mtf_stretch(img_float: np.ndarray, target_background: float = 0.20):
    bp, m = calculate_mtf_params(img_float, target_background)
    stretched = manual_stretch(img_float, black_point=bp, midtone=m)
    return stretched, bp, m

def manual_stretch(img_float: np.ndarray, black_point: float = 0.0, midtone: float = 0.1):
    # 1. Recorte y re-escalado del punto negro
    clipped = np.clip((img_float - black_point) / max(1e-6, 1.0 - black_point), 0.0, 1.0)
    # 2. Aplicación directa de curva MTF con el 'm' elegido
    stretched = mtf_curve(clipped, max(1e-5, min(0.9999, midtone)))
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)
    
def apply_white_balance(img_rgb: np.ndarray, temp_factor: float = 0.0, tint_factor: float = 0.0) -> np.ndarray:
    """
    Aplica balance de blancos multiplicativo sobre float32 lineal.
    - temp_factor: [-1.0, 1.0] (negativo enfría/azul, positivo calienta/rojo)
    - tint_factor: [-1.0, 1.0] (negativo tira a verde, positivo tira a magenta)
    """
    if temp_factor == 0.0 and tint_factor == 0.0:
        return img_rgb

    # Canales RGB base
    r_gain = 1.0 + (temp_factor * 0.5) + (tint_factor * 0.25)
    g_gain = 1.0 - (tint_factor * 0.5)
    b_gain = 1.0 - (temp_factor * 0.5) + (tint_factor * 0.25)

    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    # Normalizar para que la luminancia promedio no sufra saltos bruscos
    gains /= np.mean(gains)

    balanced = img_rgb * gains
    return np.clip(balanced, 0.0, 1.0)
    
def adjust_vibrance(img_rgb: np.ndarray, vibrance: float = 0.0) -> np.ndarray:
    """
    Ajusta la intensidad (Vibrance) de forma selectiva.
    vibrance va de -1.0 a 1.0 (0.0 = neutro).
    Satura más los tonos tenues y protege los tonos ya saturados.
    """
    if abs(vibrance) < 1e-4:
        return img_rgb

    # Trabajamos en espacio HSV
    hsv = cv2.cvtColor(np.clip(img_rgb, 0.0, 1.0), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    if vibrance > 0:
        # Cuanto menos saturado esté el píxel, más incremento recibe
        factor = 1.0 + vibrance * (1.0 - sat)
        hsv[..., 1] = np.clip(sat * factor, 0.0, 1.0)
    else:
        # Reducción suave
        hsv[..., 1] = np.clip(sat * (1.0 + vibrance), 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def adjust_contrast(img_rgb: np.ndarray, contrast: float = 0.0, pivot: float = None) -> np.ndarray:
    """
    Ajuste de contraste sigmoidal en 'S' suave y progresivo en rango [0.0, 1.0].
    Garantiza estrictamente f(0) = 0 y f(1) = 1, eliminando el empaste o recorte
    de sombras y transiciones bruscas cerca de contrast = 0.
    
    - contrast: [-1.0, 1.0] (0.0 neutro).
    - pivot: punto de inflexión tonal (por defecto 0.35 para proteger sombras profundas).
    """
    if abs(contrast) < 1e-4:
        return img_rgb

    # Anclaje de pivote seguro: por defecto 0.35 para centrarse en gas y nebulosas
    # sin perturbar el pedestal de fondo de cielo
    if pivot is None:
        pivot = 0.35
    else:
        pivot = float(np.clip(pivot, 0.10, 0.70))

    x = np.clip(img_rgb, 0.0, 1.0)

    if contrast > 0:
        # k modula la pendiente: contraste de +0.01 genera un incremento suave e imperceptible
        # mientras que +1.0 produce una curva pronunciada sin recortes
        k = 1.0 + (contrast * 9.0)

        # Función sigmoidal normalizada con f(0)=0 y f(1)=1
        # Usamos formulación algebraica continua libre de overflow exp()
        # s(x) = (x / pivot)^k si x < pivot, 1 - ((1 - x)/(1 - pivot))^k si x >= pivot
        # Para garantizar derivada suave usamos una mezcla sinusoidal/polinómica
        sig = 1.0 / (1.0 + np.exp(-k * (x - pivot)))
        sig_0 = 1.0 / (1.0 + np.exp(k * pivot))
        sig_1 = 1.0 / (1.0 + np.exp(-k * (1.0 - pivot)))

        norm_s = (sig - sig_0) / max(1e-7, (sig_1 - sig_0))

        # Mezcla ponderada por el valor de contraste para asegurar continuidad en 0.00
        out = (1.0 - contrast) * x + contrast * norm_s
    else:
        # Reducción suave de contraste hacia el pivote
        factor = 1.0 + (contrast * 0.5)  # contrast es negativo
        out = pivot + (x - pivot) * factor

    return np.clip(out, 0.0, 1.0).astype(np.float32)
    
# --- Ecualizador Tonal de 4 Bandas (Starless) ---
def adjust_tonal_bands(
    img_rgb: np.ndarray,
    blacks: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
    whites: float = 0.0
) -> np.ndarray:
    """
    Ecualizador de 4 bandas actuando sobre la imagen ya estirada (perceptual [0.0, 1.0]):
    - blacks: [-1.0, 1.0] ajusta el anclaje del fondo profundo (0.0 a 0.25).
    - shadows: [-1.0, 1.0] modula el polvo y nebulosas tenues (0.15 a 0.50).
    - highlights: [-1.0, 1.0] modula estructuras brillantes (0.45 a 0.80).
    - whites: [-1.0, 1.0] expande o comprime los picos de luminosidad (0.70 a 1.0).
    """
    if all(abs(v) < 1e-4 for v in (blacks, shadows, highlights, whites)):
        return img_rgb

    # 1. Luminancia perceptiva (segura y delimitada en 0..1)
    lum = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    lum = np.clip(lum, 0.0, 1.0)

    # 2. Pesos polinómicos suaves y continuos (sin funciones trigonométricas ni potencias fraccionarias)
    # A) Negros: máximo en 0.0, decrece hasta 0.25
    w_blacks = np.clip((0.25 - lum) / 0.25, 0.0, 1.0) ** 2

    # B) Sombras: campana suave entre 0.10 y 0.55 (pico en ~0.30)
    w_shadows = np.clip((lum - 0.10) / 0.20, 0.0, 1.0) * np.clip((0.55 - lum) / 0.25, 0.0, 1.0) * 4.0

    # C) Altas Luces: campana entre 0.40 y 0.85 (pico en ~0.65)
    w_highlights = np.clip((lum - 0.40) / 0.25, 0.0, 1.0) * np.clip((0.85 - lum) / 0.20, 0.0, 1.0) * 4.0

    # D) Blancos: sube a partir de 0.65 hasta 1.0
    w_whites = np.clip((lum - 0.65) / 0.35, 0.0, 1.0) ** 2

    # 3. Deltas tonales (con escala de fuerza equilibrada e intuitiva)
    delta_lum = (
        (blacks * 0.12 * w_blacks) +
        (shadows * 0.18 * w_shadows) +
        (highlights * 0.18 * w_highlights) +
        (whites * 0.15 * w_whites)
    )

    new_lum = np.clip(lum + delta_lum, 0.0, 1.0)

    # 4. Modulación cromática segura (evitando cualquier división por cero)
    eps = 1e-6
    ratio = (new_lum + eps) / (lum + eps)
    out_rgb = img_rgb * ratio[..., np.newaxis]

    return np.clip(out_rgb, 0.0, 1.0).astype(np.float32)

# --- Reducción de Ruido Multimodelo (Starless) ---

def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Implementación rápida en NumPy del Guided Filter."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, -1, ksize)
    mean_p = cv2.boxFilter(src, -1, ksize)
    mean_Ip = cv2.boxFilter(guide * src, -1, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, -1, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    q = mean_a * guide + mean_b
    return q


def apply_denoise(
    img_rgb: np.ndarray,
    strength: float = 0.0,
    is_full_res: bool = False
) -> np.ndarray:
    """
    Aplica reducción de ruido mediante Guided Filter nativo en float32 [0.0, 1.0].
    Preserva bordes finos, nebulosas y filamentos sin cuantizar ni generar banding.
    
    - strength: [0.0, 1.0] (0% a 100%)
    - is_full_res: escala el radio geométrico para resolución completa nativa.
    """
    if strength <= 1e-4:
        return img_rgb

    # Factor de escala geométrico para compensar diferencia proxy vs sensor completo
    scale = 2.5 if is_full_res else 1.0

    # Radio de vecindad: crecimiento progresivo y suave (de 2 a ~6 px en proxy, escalado en full)
    radius = int(max(2, round((2.0 + 4.0 * (strength ** 0.8)) * scale)))

    # Epsilon: respuesta logarítmica/potencial suave.
    # En 1% es casi imperceptible (1e-6) y en 100% alcanza un filtrado firme (1.5e-3)
    # sin lavar detalles estructurales.
    eps = float(1e-6 * (1500.0 ** strength))

    # Guía en escala de grises float32 (sin cuantizar a uint8)
    gray_guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    channels = []
    for c in range(3):
        ch_denoised = _guided_filter(gray_guide, img_rgb[..., c], radius=radius, eps=eps)
        channels.append(ch_denoised)

    out = np.stack(channels, axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
    
def apply_curve_lut(img_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Aplica una curva tonal mediante tabla de consulta (LUT de 256 elementos [0..1]).
    Rendimiento sub-milisegundo vía cv2.LUT.
    """
    if lut is None or img_rgb is None:
        return img_rgb

    # Escalado a uint8 -> mapeo LUT -> recuperación float32
    u8 = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    lut_u8 = (np.clip(lut, 0.0, 1.0) * 255.0).astype(np.uint8)

    mapped_u8 = cv2.LUT(u8, lut_u8)
    return (mapped_u8.astype(np.float32) / 255.0)
    
def apply_light_pollution_gradient(
    img_rgb: np.ndarray, 
    strength: float = 0.0, 
    height_ratio: float = 0.50,
    sky_mask: np.ndarray = None
) -> np.ndarray:
    """
    Atenúa la cúpula de luz en el horizonte modulando la luminancia de forma neutra.
    strength: 0.0 (desactivado) a 1.0 (máxima atenuación).
    """
    if strength <= 1e-4:
        return img_rgb

    h, w = img_rgb.shape[:2]
    
    # 1. Perfil vertical normalizado a 1D
    y_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)
    
    # 0 arriba, 1 en la base del horizonte
    grad_y = np.clip((y_coords - (1.0 - height_ratio)) / max(1e-4, height_ratio), 0.0, 1.0)
    
    # Curva suave coseno (sin saltos bruscos)
    curve_1d = 0.5 * (1.0 - np.cos(np.pi * grad_y)) * (strength * 0.70)
    
    # 2. Expandir explícitamente a toda la cuadrícula (h, w)
    weight_2d = np.tile(curve_1d[:, np.newaxis], (1, w))

    # 3. Aplicar máscara si existe
    if sky_mask is not None:
        m = np.squeeze(sky_mask).astype(np.float32)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        weight_2d = weight_2d * m

    # 4. Pasar a 3 canales (h, w, 3)
    weight_3d = np.repeat(weight_2d[..., np.newaxis], 3, axis=2)

    # 5. Atenuación multiplicativa neutra sobre los 3 canales por igual
    attenuated = img_rgb * (1.0 - weight_3d)
    return np.clip(attenuated, 0.0, 1.0).astype(np.float32)
    
def extract_background_polynomial(
    img_rgb: np.ndarray, 
    sky_mask: np.ndarray = None, 
    degree: int = 2
) -> np.ndarray:
    """
    Modela el fondo del cielo mediante una superficie polinómica 2D (grado 1 o 2).
    Al ajustarse solo con píxeles puros de cielo, es 100% inmune a siluetas de árboles y suelo.
    """
    h, w, c = img_rgb.shape
    
    # Reducir resolución para estimación ultrarrápida y resistente a estrellas
    scale = max(1, min(h, w) // 300)
    small_h, small_w = h // scale, w // scale
    
    small_img = cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    if sky_mask is not None:
        m = np.squeeze(sky_mask).astype(np.float32)
        if m.shape != (small_h, small_w):
            small_mask = cv2.resize(m, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        else:
            small_mask = m
        # Retirar borde del suelo para muestreo puro
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        valid_sky = (cv2.erode((small_mask > 0.5).astype(np.uint8), k) > 0)
    else:
        valid_sky = np.ones((small_h, small_w), dtype=bool)

    # Coordenadas normalizadas [-1, 1]
    y, x = np.mgrid[:small_h, :small_w]
    x_norm = (x / (small_w - 1.0)) * 2.0 - 1.0
    y_norm = (y / (small_h - 1.0)) * 2.0 - 1.0

    # Términos del polinomio cuadrático: 1, x, y, x^2, y^2, x*y
    if degree == 1:
        A = np.column_stack([np.ones(x_norm.size), x_norm.ravel(), y_norm.ravel()])
    else:
        A = np.column_stack([
            np.ones(x_norm.size), 
            x_norm.ravel(), y_norm.ravel(),
            (x_norm**2).ravel(), (y_norm**2).ravel(), (x_norm * y_norm).ravel()
        ])

    valid_flat = valid_sky.ravel()
    A_valid = A[valid_flat]
    
    corrected_channels = []
    
    for ch in range(c):
        vals = small_img[..., ch].ravel()[valid_flat]
        
        # Rechazo de estrellas brillantes (percentil 10 a 65 del cielo)
        p_low, p_high = np.percentile(vals, [5, 60])
        samples = (vals >= p_low) & (vals <= p_high)
        
        # Ajuste analítico por mínimos cuadrados
        coeff, _, _, _ = np.linalg.lstsq(A_valid[samples], vals[samples], rcond=None)
        
        # Evaluar superficie completa a escala real
        y_f, x_f = np.mgrid[:h, :w]
        x_fn = (x_f / (w - 1.0)) * 2.0 - 1.0
        y_fn = (y_f / (h - 1.0)) * 2.0 - 1.0
        
        if degree == 1:
            bg_full = coeff[0] + coeff[1]*x_fn + coeff[2]*y_fn
        else:
            bg_full = (coeff[0] + coeff[1]*x_fn + coeff[2]*y_fn + 
                       coeff[3]*(x_fn**2) + coeff[4]*(y_fn**2) + coeff[5]*(x_fn*y_fn))
                       
        # Substracción preservando el valor medio
        pedestal = float(np.median(vals[samples]))
        ch_corr = img_rgb[..., ch] - bg_full + pedestal
        corrected_channels.append(ch_corr)
        
    corrected = np.stack(corrected_channels, axis=-1)
    
    if sky_mask is not None:
        m_full = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        ksize = int(max(15, (min(h, w) // 150) | 1))
        if ksize % 2 == 0: ksize += 1
        m_smooth = cv2.GaussianBlur(m_full, (ksize, ksize), sigmaX=ksize/3.0)[..., np.newaxis]
        return np.clip((corrected * m_smooth) + (img_rgb * (1.0 - m_smooth)), 0.0, 1.0).astype(np.float32)
        
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)
    
def apply_clarity(img_rgb: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Contraste local (Claridad) en frecuencias medias.
    strength: -2.0 a +2.0 (0.0 neutro).
    """
    if abs(strength) < 1e-4:
        return img_rgb

    # Luminancia perceptual
    lum = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    
    h, w = img_rgb.shape[:2]
    r = int(max(9, (min(h, w) // 35) | 1))
    if r % 2 == 0:
        r += 1

    lum_blur = cv2.GaussianBlur(lum, (r, r), sigmaX=r / 3.0)
    high_freq = lum - lum_blur

    # Con strength hasta 2.0 usamos una respuesta suave para evitar saturación dura
    factor = strength * 0.85
    new_lum = np.clip(lum + high_freq * factor, 1e-6, 1.0)

    ratio = (new_lum / np.maximum(1e-6, lum))[..., np.newaxis]
    return np.clip(img_rgb * ratio, 0.0, 1.0).astype(np.float32)


def apply_dehaze(img_rgb: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Borrar Neblina (Dehaze) con compensación de velo atmosférico y saturación.
    strength: -2.0 a +2.0 (0.0 neutro).
    """
    if abs(strength) < 1e-4:
        return img_rgb

    dark_ch = np.min(img_rgb, axis=2)
    h, w = img_rgb.shape[:2]
    r = int(max(15, (min(h, w) // 25) | 1))
    if r % 2 == 0:
        r += 1
        
    haze_map = cv2.GaussianBlur(dark_ch, (r, r), sigmaX=r / 2.5)

    if strength > 0:
        # Transmisión escalada para tolerar valores de hasta 2.0 sin colapsar a 0
        t = np.clip(1.0 - (strength * 0.45) * haze_map, 0.12, 1.0)[..., np.newaxis]
        airlight = float(np.percentile(haze_map, 95)) * min(0.65, 0.35 * strength)
        out = (img_rgb - airlight * (1.0 - t)) / t
        
        # Realce dinámico de saturación en zonas rescatadas de la bruma
        lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., np.newaxis]
        out = lum + (out - lum) * (1.0 + strength * 0.25)
    else:
        factor = min(0.85, abs(strength) * 0.35)
        out = img_rgb * (1.0 - factor) + haze_map[..., np.newaxis] * factor

    return np.clip(out, 0.0, 1.0).astype(np.float32)
    
def descomponer_ondiculas(imagen: np.ndarray, niveles: int = 5) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Descompone la imagen en planos de detalle por ondículas (filtros dilatados)
    usando un núcleo B-Spline cúbico separable.
    """
    nucleo_base = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float32)
    actual = imagen.copy()
    planos_detalle = []

    for nivel in range(niveles):
        salto = 2 ** nivel
        longitud_k = 4 * salto + 1
        k1d = np.zeros(longitud_k, dtype=np.float32)
        for i, val in enumerate(nucleo_base):
            k1d[i * salto] = val

        # Filtrado paso bajo separable
        paso_bajo = cv2.sepFilter2D(actual, -1, k1d, k1d, borderType=cv2.BORDER_REFLECT)
        
        # El detalle es la resta entre la escala actual y la suavizada
        detalle = actual - paso_bajo
        planos_detalle.append(detalle)
        actual = paso_bajo

    # 'actual' contiene la capa residual de frecuencia ultra-baja (fondo y cúpula)
    return planos_detalle, actual


def realce_multiescala_ondiculas(
    imagen_rgb: np.ndarray,
    fuerza_estructura: float = 0.0,
    reduccion_fondo: float = 0.0,
    sky_mask: np.ndarray = None
) -> np.ndarray:
    """
    Aplica realce de estructuras de gas/polvo y atenuación de cúpula residual.
    fuerza_estructura: -1.0 a +1.0 (escalas intermedias)
    reduccion_fondo: 0.0 a 1.0 (capa residual de fondo)
    """
    if abs(fuerza_estructura) < 1e-4 and reduccion_fondo < 1e-4:
        return imagen_rgb

    # Descomposición en 5 niveles de escala
    detalles, residual = descomponer_ondiculas(imagen_rgb, niveles=5)

    # 1. Modulación de detalles: las escalas 3 y 4 corresponden a las nebulosas y polvo
    # Se aplica una ganancia suave progresiva
    pesos = [
        1.0,                                    # Escala 1 (ruido / micro-estrellas)
        1.0,                                    # Escala 2 (estrellas pequeñas)
        1.0 + fuerza_estructura * 0.4,          # Escala 3 (transiciones de gas)
        1.0 + fuerza_estructura * 0.8,          # Escala 4 (filamentos de la Vía Láctea)
        1.0 + fuerza_estructura * 0.6           # Escala 5 (carriles anchos de polvo)
    ]

    # Reconstrucción de la suma de detalles
    suma_detalles = np.zeros_like(imagen_rgb)
    for d, peso in zip(detalles, pesos):
        suma_detalles += d * peso

    # 2. Atenuación del plano residual (la cúpula fija)
    if reduccion_fondo > 1e-4:
        pedestal = np.percentile(residual, 5, axis=(0, 1))
        residual_atenuado = residual - (residual - pedestal) * (reduccion_fondo * 0.5)
        residual = residual_atenuado

    reconstruida = np.clip(suma_detalles + residual, 0.0, 1.0)

    # Si hay máscara de cielo, aplicamos selectivamente
    if sky_mask is not None:
        h, w = imagen_rgb.shape[:2]
        m = cv2.resize(np.squeeze(sky_mask).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if m.ndim == 2:
            m = m[..., np.newaxis]
        return np.clip(reconstruida * m + imagen_rgb * (1.0 - m), 0.0, 1.0).astype(np.float32)

    return reconstruida.astype(np.float32)


def apply_hue_vs_sat_curve(img_rgb: np.ndarray, lut_hue_sat: np.ndarray) -> np.ndarray:
    """
    Aplica una curva selectiva de saturación en función del tono (Hue vs Saturation).
    - lut_hue_sat: array float32 de 256 elementos donde:
      * 0.5 representa factor neutro (1.0x)
      * < 0.5 reduce saturación hacia 0.0x
      * > 0.5 amplifica saturación hasta 2.0x
    """
    if lut_hue_sat is None or len(lut_hue_sat) != 256:
        return img_rgb
    if np.all(np.abs(lut_hue_sat - 0.5) < 1e-3):
        return img_rgb

    clamped = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(clamped, cv2.COLOR_RGB2HSV)

    # El canal H en OpenCV float32 está en grados [0.0, 360.0)
    h_channel = hsv[..., 0]
    h_idx = np.clip((h_channel / 360.0) * 255.0, 0, 255).astype(np.int32)

    # Multiplicador: valor de LUT * 2.0 (0.5 -> 1.0x, 1.0 -> 2.0x, 0.0 -> 0.0x)
    mult = lut_hue_sat[h_idx] * 2.0
    hsv[..., 1] = np.clip(hsv[..., 1] * mult, 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_luma_vs_sat_curve(img_rgb: np.ndarray, lut_luma_sat: np.ndarray) -> np.ndarray:
    """
    Aplica una curva selectiva de saturación en función de la luminancia (Luma vs Saturation).
    Permite desaturar sombras para eliminar ruido de crominancia o realzar nebulosas en tonos medios.
    - lut_luma_sat: array float32 de 256 elementos donde 0.5 representa 1.0x (neutro).
    """
    if lut_luma_sat is None or len(lut_luma_sat) != 256:
        return img_rgb
    if np.all(np.abs(lut_luma_sat - 0.5) < 1e-3):
        return img_rgb

    clamped = np.clip(img_rgb, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(clamped, cv2.COLOR_RGB2HSV)

    # Luminancia perceptual BT.709
    luma = 0.2126 * clamped[..., 0] + 0.7152 * clamped[..., 1] + 0.0722 * clamped[..., 2]
    l_idx = np.clip(luma * 255.0, 0, 255).astype(np.int32)

    mult = lut_luma_sat[l_idx] * 2.0
    hsv[..., 1] = np.clip(hsv[..., 1] * mult, 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)