# core/mask_layers.py
"""
core/mask_layers.py - Sistema de capas y máscaras de ajuste local.
Permite apilar múltiples máscaras (Pincel, Suelo/Cielo, Luminancia, Rango de Color) con ajustes
locales paramétricos independientes y conciencia de capas StarNet (Compuesta, Suelo, Fondo Starless, Estrellas).
"""
import uuid
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
from core.masking import generate_luminance_mask, generate_color_range_mask
from core.stretch import apply_white_balance, adjust_contrast

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _numba_fused_layer_adjust(
        orig_img, adj_source, mask, weight_scale,
        gain_r, gain_g, gain_b,
        shadows, highlights,
        has_curve, curve_lut
    ):
        h, w, c = orig_img.shape
        out = np.empty_like(orig_img)
        lut_len = len(curve_lut) - 1 if has_curve else 0

        for i in prange(h):
            for j in range(w):
                m = mask[i, j] * weight_scale
                if m <= 1e-4:
                    out[i, j, 0] = orig_img[i, j, 0]
                    out[i, j, 1] = orig_img[i, j, 1]
                    out[i, j, 2] = orig_img[i, j, 2]
                    continue

                for k in range(3):
                    val = adj_source[i, j, k]
                    if k == 0:
                        val *= gain_r
                    elif k == 1:
                        val *= gain_g
                    else:
                        val *= gain_b

                    if val < 0.0:
                        val = 0.0
                    elif val > 1.0:
                        val = 1.0

                    if abs(shadows) > 1e-3:
                        lift = (1.0 - val) * (1.0 - val)
                        val = val * (1.0 + shadows * lift)

                    if abs(highlights) > 1e-3:
                        hi = val * val
                        val = val + (highlights * hi * 0.4)

                    if val < 0.0:
                        val = 0.0
                    elif val > 1.0:
                        val = 1.0

                    if has_curve:
                        idx = int(val * lut_len)
                        if idx < 0:
                            idx = 0
                        elif idx > lut_len:
                            idx = lut_len
                        val = curve_lut[idx]

                    out[i, j, k] = val * m + orig_img[i, j, k] * (1.0 - m)

        return out

    @njit(parallel=True, fastmath=True)
    def _numba_screen_blend(canvas, stars, opacity):
        h, w, c = canvas.shape
        out = np.empty_like(canvas)
        for i in prange(h):
            for j in range(w):
                for k in range(c):
                    s = stars[i, j, k] * opacity
                    if s < 0.0:
                        s = 0.0
                    elif s > 1.0:
                        s = 1.0
                    cv = canvas[i, j, k]
                    out[i, j, k] = cv + s - cv * s
        return out


# Ámbitos de aplicación (Target Scope)
SCOPE_TARGET_ALL = "all"            # Toda la imagen (Compuesta)
SCOPE_TARGET_GROUND = "ground"      # Solo Suelo (aislado por la máscara de suelo)
SCOPE_TARGET_STARLESS = "starless"  # Solo Fondo Starless (estrellas desacopladas)
SCOPE_TARGET_STARS = "stars"        # Solo Capa de Estrellas

# Naturaleza de la Capa (Layer Kind)
LAYER_KIND_ADJUSTMENT = "adjustment"  # Capa de ajuste paramétrico con máscara
LAYER_KIND_IMAGE = "image"            # Capa portadora de imagen (Fondo Starless, Estrellas, Suelo)

# Tipos de Máscara de Ajuste
MASK_TYPE_BRUSH = "brush"           # Pincel interactivo con dureza y opacidad de trazo
MASK_TYPE_GROUND = "ground"         # Máscara Suelo (invertida del cielo)
MASK_TYPE_SKY = "sky"               # Máscara Cielo
MASK_TYPE_LUMA = "luminance"        # Rango de Luminancia (Sombras/Medios/Luces)
MASK_TYPE_COLOR = "color"           # Rango de Color (HSV)
MASK_TYPE_CUSTOM = "custom"         # Trazos libres / raster personalizado

# Tipos Especiales de Capas de Imagen (StarNet / Pipeline)
MASK_TYPE_IMAGE_BASE = "image_base"          # 🖼️ Fondo (Imagen Base inicial)
MASK_TYPE_IMAGE_STARLESS = "image_starless"  # 🌌 Fondo Nebular (Starless AI)
MASK_TYPE_IMAGE_STARS = "image_stars"        # ⭐ Capa de Estrellas (StarNet AI)
MASK_TYPE_IMAGE_GROUND = "image_ground"      # 🏞️ Capa de Suelo / Terreno (Paisaje)


@dataclass
class MaskLayer:
    """Representa una capa (de ajuste o de imagen especial) con su propia máscara y parámetros."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Nueva Máscara"
    layer_kind: str = LAYER_KIND_ADJUSTMENT   # "adjustment" o "image"
    mask_type: str = MASK_TYPE_BRUSH
    target_scope: str = SCOPE_TARGET_ALL
    enabled: bool = True
    inverted: bool = False
    opacity: float = 1.0  # 0.0 a 1.0
    is_system_layer: bool = False  # Si es True, no se puede eliminar con el botón de borrar

    # Parámetros para máscara de Pincel interactivo
    brush_radius: int = 35          # Radio en píxeles (5 a 250 px)
    brush_hardness: float = 0.5     # Dureza: 0.0 (suave/difuminado) a 1.0 (borde neto)
    brush_flow: float = 0.8         # Flujo / opacidad por pasada (0.05 a 1.00)
    brush_mode: str = "paint"       # "paint" o "erase"
    brush_mask: Optional[np.ndarray] = None  # Float32 2D [0.0, 1.0]

    # Parámetros para máscara de rango de luminancia
    luma_min: float = 0.0
    luma_max: float = 1.0
    feather: float = 0.08

    # Parámetros para máscara de rango cromático
    color_hue: float = 0.0          # 0.0 a 360.0 grados
    color_tolerance: float = 30.0   # Tolerancia angular en grados
    color_min_sat: float = 0.08     # Saturación mínima

    # Ajustes locales de revelado
    exposure: float = 0.0           # -3.00 EV a +3.00 EV
    contrast: float = 0.0           # -1.00 a +1.00
    shadows: float = 0.0            # -1.00 a +1.00 (recuperar o profundizar)
    highlights: float = 0.0         # -1.00 a +1.00 (atenuar o iluminar luces)
    saturation: float = 1.0         # 0.00x a 2.50x
    temp: float = 0.0               # -1.0000 a +1.0000 (factor de balance cálido/frío)
    tint: float = 0.0               # -1.0000 a +1.0000 (factor de balance verde/magenta)

    # Curva tonal interactiva de capa
    curve_points: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 1.0)])
    curve_lut: Optional[np.ndarray] = None

    def is_curve_identity(self) -> bool:
        """Determina si la curva es la diagonal neutra identidad [(0,0), (1,1)]."""
        if self.curve_points and len(self.curve_points) == 2:
            p0, p1 = self.curve_points[0], self.curve_points[1]
            if (abs(p0[0] - 0.0) < 1e-3 and abs(p0[1] - 0.0) < 1e-3 and
                abs(p1[0] - 1.0) < 1e-3 and abs(p1[1] - 1.0) < 1e-3):
                return True
        elif not self.curve_points:
            return True
        return False

    def has_curve(self) -> bool:
        """Indica si la capa tiene aplicada una curva no lineal efectiva."""
        return not self.is_curve_identity()

    # Máscara rasterizada en memoria float32 [0.0, 1.0] (no serializable a JSON)
    custom_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la capa a un diccionario JSON preservando la máscara de pincel y curva."""
        d = asdict(self)
        d.pop("custom_mask", None)
        d.pop("brush_mask", None)
        d.pop("curve_lut", None)
        if self.brush_mask is not None:
            try:
                u8 = (np.clip(self.brush_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                ok, buf = cv2.imencode(".png", u8)
                if ok:
                    d["brush_mask_b64"] = base64.b64encode(buf).decode("ascii")
            except Exception:
                pass
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskLayer":
        """Instancia una capa a partir de un diccionario JSON."""
        data_clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "curve_points" in data_clean and data_clean["curve_points"]:
            data_clean["curve_points"] = [tuple(p) for p in data_clean["curve_points"]]
        inst = cls(**data_clean)
        b64_str = data.get("brush_mask_b64")
        if b64_str:
            try:
                buf = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
                dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if dec is not None:
                    inst.brush_mask = dec.astype(np.float32) / 255.0
            except Exception:
                pass
        if not inst.is_curve_identity() and inst.curve_lut is None:
            inst.curve_lut = compute_curve_lut_from_points(inst.curve_points)
        return inst


def compute_layer_mask(
    layer: MaskLayer,
    base_rgb: np.ndarray,
    master_sky_mask: Optional[np.ndarray] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Calcula o evalúa la máscara 2D [0.0, 1.0] correspondiente a la capa.
    Aplica inversión si la capa lo requiere. Soporta recorte espacial cuando el visor
    se encuentra en modo Zoom 100%.
    """
    h, w = base_rgb.shape[:2]

    def _slice_or_resize_mask(m_src: np.ndarray) -> np.ndarray:
        if m_src is None:
            return np.zeros((h, w), dtype=np.float32)
        if m_src.shape[:2] == (h, w):
            return m_src.copy()
        if crop_rect is not None and full_image_shape is not None:
            x1, y1, x2, y2 = crop_rect
            h_f, w_f = full_image_shape
            h_m, w_m = m_src.shape[:2]
            sx = w_m / float(w_f)
            sy = h_m / float(h_f)
            mx1 = max(0, min(w_m, int(round(x1 * sx))))
            my1 = max(0, min(h_m, int(round(y1 * sy))))
            mx2 = max(0, min(w_m, int(round(x2 * sx))))
            my2 = max(0, min(h_m, int(round(y2 * sy))))
            m_sub = m_src[my1:my2, mx1:mx2]
            if m_sub.shape[:2] != (h, w):
                return cv2.resize(m_sub, (w, h), interpolation=cv2.INTER_LINEAR)
            return m_sub.copy()
        return cv2.resize(m_src, (w, h), interpolation=cv2.INTER_LINEAR)

    if layer.mask_type == MASK_TYPE_BRUSH:
        if layer.brush_mask is not None:
            m = _slice_or_resize_mask(layer.brush_mask)
        else:
            m = np.zeros((h, w), dtype=np.float32)

    elif layer.mask_type == MASK_TYPE_GROUND:
        if master_sky_mask is not None:
            m_sky = master_sky_mask[..., 0] if master_sky_mask.ndim == 3 else master_sky_mask
            m_sky = _slice_or_resize_mask(m_sky)
            m = np.clip(1.0 - m_sky, 0.0, 1.0)
        else:
            m = np.ones((h, w), dtype=np.float32)

    elif layer.mask_type == MASK_TYPE_SKY:
        if master_sky_mask is not None:
            m_sky = master_sky_mask[..., 0] if master_sky_mask.ndim == 3 else master_sky_mask
            m = _slice_or_resize_mask(m_sky)
        else:
            m = np.ones((h, w), dtype=np.float32)

    elif layer.mask_type == MASK_TYPE_LUMA:
        m = generate_luminance_mask(
            base_rgb,
            luma_min=layer.luma_min,
            luma_max=layer.luma_max,
            feather=layer.feather,
            invert=False
        )

    elif layer.mask_type == MASK_TYPE_COLOR:
        m = generate_color_range_mask(
            base_rgb,
            target_hue_deg=layer.color_hue,
            hue_tolerance_deg=layer.color_tolerance,
            min_saturation=layer.color_min_sat,
            feather=layer.feather,
            invert=False
        )

    elif layer.mask_type == MASK_TYPE_CUSTOM and layer.custom_mask is not None:
        m = _slice_or_resize_mask(layer.custom_mask)
    else:
        m = np.ones((h, w), dtype=np.float32)

    # Inversión de máscara
    if layer.inverted:
        m = 1.0 - m

    return np.clip(m, 0.0, 1.0).astype(np.float32)


def compute_curve_lut_from_points(points: Optional[List[Tuple[float, float]]], resolution: int = 256) -> np.ndarray:
    """Calcula una tabla LUT interpolada monótona PCHIP de 256 niveles a partir de puntos de control."""
    if not points:
        return np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    pts = sorted(points, key=lambda p: p[0])
    clean_pts = []
    for p in pts:
        if not clean_pts or abs(p[0] - clean_pts[-1][0]) > 1e-4:
            clean_pts.append(p)
    xs = [float(p[0]) for p in clean_pts]
    ys = [float(p[1]) for p in clean_pts]
    if xs[0] > 0.001:
        xs.insert(0, 0.0)
        ys.insert(0, ys[0])
    else:
        xs[0] = 0.0
    if xs[-1] < 0.999:
        xs.append(1.0)
        ys.append(ys[-1])
    else:
        xs[-1] = 1.0
    grid_x = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(xs, ys)
        lut_arr = np.clip(pchip(grid_x), 0.0, 1.0).astype(np.float32)
    except Exception:
        lut_arr = np.clip(np.interp(grid_x, xs, ys), 0.0, 1.0).astype(np.float32)
    return lut_arr


def apply_pre_curve_adjustments(image_rgb: np.ndarray, layer: MaskLayer, fast_draft: bool = False) -> np.ndarray:
    """
    Aplica secuencialmente los ajustes pre-curva de la capa (Balance de Blancos, Exposición EV,
    Sombras, Luces, Contraste, Saturación) sin aplicar la curva tonal LUT ni ponderación de máscara.
    Útil para alimentar histogramas interactivos de entrada a la curva tonal.
    """
    has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
               abs(layer.exposure) > 1e-3 or abs(layer.shadows) > 1e-3 or 
               abs(layer.highlights) > 1e-3 or abs(layer.contrast) > 1e-3 or 
               abs(layer.saturation - 1.0) > 1e-3)
    if not has_adj:
        return np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    adj = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

    # 1. Balance de blancos local
    if abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4:
        adj = apply_white_balance(adj, layer.temp, layer.tint)

    # 2. Exposición local (EV)
    if abs(layer.exposure) > 1e-3:
        adj = adj * (2.0 ** layer.exposure)

    # 3. Sombras locales
    if abs(layer.shadows) > 1e-3:
        lift = (1.0 - np.clip(adj, 0.0, 1.0)) ** 2
        adj = adj * (1.0 + layer.shadows * lift)

    # 4. Luces locales (Highlights)
    if abs(layer.highlights) > 1e-3:
        hi = np.clip(adj, 0.0, 1.0) ** 2
        adj = adj + (layer.highlights * hi * 0.4)

    # 5. Contraste local
    if abs(layer.contrast) > 1e-3:
        adj = adjust_contrast(adj, layer.contrast, fast_draft=fast_draft)

    # 6. Saturación local
    if abs(layer.saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(np.clip(adj, 0.0, 1.0), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * layer.saturation, 0.0, 1.0)
        adj = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return np.clip(adj, 0.0, 1.0).astype(np.float32)


def apply_local_adjustments_to_image(
    image_rgb: np.ndarray,
    effective_mask: np.ndarray,
    layer: MaskLayer,
    fast_draft: bool = False
) -> np.ndarray:
    """
    Aplica los controles locales de una capa sobre image_rgb ponderados por effective_mask y opacidad.
    """
    if not layer.enabled or layer.opacity <= 1e-4:
        return image_rgb

    # Ponderación global de la máscara por opacidad
    # Omitir procesamiento costoso si todos los controles locales y curvas están en valores neutros
    has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
               abs(layer.exposure) > 1e-3 or abs(layer.shadows) > 1e-3 or 
               abs(layer.highlights) > 1e-3 or abs(layer.contrast) > 1e-3 or 
               abs(layer.saturation - 1.0) > 1e-3 or not layer.is_curve_identity())
    if not has_adj:
        return image_rgb

    # Ponderación global de la máscara por opacidad
    weight = np.clip(effective_mask * layer.opacity, 0.0, 1.0).astype(np.float32)
    if np.max(weight) <= 1e-4:
        return image_rgb

    if HAS_NUMBA:
        orig_c = np.ascontiguousarray(np.clip(image_rgb, 0.0, 1.0), dtype=np.float32)
        has_sat = abs(layer.saturation - 1.0) > 1e-3
        if has_sat:
            hsv = cv2.cvtColor(orig_c, cv2.COLOR_RGB2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * layer.saturation, 0.0, 1.0)
            adj_source = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            adj_source = orig_c

        temp_factor = layer.temp
        tint_factor = layer.tint
        if abs(temp_factor) > 1e-4 or abs(tint_factor) > 1e-4:
            r_gain = 1.0 + (temp_factor * 0.5) + (tint_factor * 0.25)
            g_gain = 1.0 - (tint_factor * 0.5)
            b_gain = 1.0 - (temp_factor * 0.5) + (tint_factor * 0.25)
            mean_gain = (r_gain + g_gain + b_gain) / 3.0
            r_gain /= mean_gain
            g_gain /= mean_gain
            b_gain /= mean_gain
        else:
            r_gain = g_gain = b_gain = 1.0

        if abs(layer.exposure) > 1e-3:
            ev_factor = 2.0 ** layer.exposure
            r_gain *= ev_factor
            g_gain *= ev_factor
            b_gain *= ev_factor

        has_curve = not layer.is_curve_identity()
        has_contrast = abs(layer.contrast) > 1e-3
        if has_curve or has_contrast:
            if has_curve:
                if layer.curve_lut is None:
                    layer.curve_lut = compute_curve_lut_from_points(layer.curve_points)
                lut = layer.curve_lut.copy()
            else:
                lut = np.linspace(0.0, 1.0, 256, dtype=np.float32)

            if has_contrast:
                lut = adjust_contrast(lut, layer.contrast, fast_draft=False)
            lut = np.ascontiguousarray(np.clip(lut, 0.0, 1.0), dtype=np.float32)
            use_lut = True
        else:
            lut = np.empty(1, dtype=np.float32)
            use_lut = False

        mask_2d = effective_mask[..., 0] if effective_mask.ndim == 3 else effective_mask
        mask_c = np.ascontiguousarray(mask_2d, dtype=np.float32)

        return _numba_fused_layer_adjust(
            orig_c, adj_source, mask_c, float(layer.opacity),
            float(r_gain), float(g_gain), float(b_gain),
            float(layer.shadows), float(layer.highlights),
            use_lut, lut
        )

    adj = apply_pre_curve_adjustments(image_rgb, layer, fast_draft=fast_draft)

    # 7. Curva tonal interactiva propia de la capa (LUT de 256 niveles acelerada por hardware)
    if not layer.is_curve_identity():
        if layer.curve_lut is None:
            layer.curve_lut = compute_curve_lut_from_points(layer.curve_points)
        u8 = (np.clip(adj, 0.0, 1.0) * 255.0).astype(np.uint8)
        lut_u8 = (np.clip(layer.curve_lut, 0.0, 1.0) * 255.0).astype(np.uint8)
        adj = (cv2.LUT(u8, lut_u8).astype(np.float32) / 255.0)

    adj = np.clip(adj, 0.0, 1.0)

    # Fusión ponderada por canal
    w_3d = weight[..., np.newaxis] if weight.ndim == 2 else weight
    result = (adj * w_3d) + (image_rgb * (1.0 - w_3d))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def compute_effective_layer_mask(
    layer: MaskLayer,
    composite_image: np.ndarray,
    master_sky_mask: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Calcula la máscara final efectiva considerando su tipo, parámetros e inversión.
    En el modelo de capas de Photoshop, el ámbito de afectación lo determina de forma
    natural la posición de la capa en la pila (afecta y lee del lienzo que tiene debajo).
    """
    m_base = compute_layer_mask(
        layer, composite_image,
        master_sky_mask=master_sky_mask,
        crop_rect=crop_rect,
        full_image_shape=full_image_shape
    )

    # Compatibilidad con proyectos legados que tengan un target_scope explícito distinto de 'all'
    if layer.target_scope == SCOPE_TARGET_ALL or not layer.target_scope:
        return m_base

    img_h, img_w = composite_image.shape[:2]
    sky_m = None
    if master_sky_mask is not None:
        sky_m = master_sky_mask[..., 0] if master_sky_mask.ndim == 3 else master_sky_mask
        if sky_m.shape[:2] != (img_h, img_w):
            if crop_rect is not None and full_image_shape is not None:
                x1, y1, x2, y2 = crop_rect
                h_f, w_f = full_image_shape
                h_m, w_m = sky_m.shape[:2]
                sx = w_m / float(w_f)
                sy = h_m / float(h_f)
                sx1 = max(0, min(w_m, int(round(x1 * sx))))
                sy1 = max(0, min(h_m, int(round(y1 * sy))))
                sx2 = max(0, min(w_m, int(round(x2 * sx))))
                sy2 = max(0, min(h_m, int(round(y2 * sy))))
                sky_sub = sky_m[sy1:sy2, sx1:sx2]
                if sky_sub.shape[:2] != (img_h, img_w):
                    sky_m = cv2.resize(sky_sub, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                else:
                    sky_m = sky_sub
            else:
                sky_m = cv2.resize(sky_m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    if layer.target_scope == SCOPE_TARGET_GROUND and sky_m is not None:
        return np.clip(m_base * (1.0 - sky_m), 0.0, 1.0).astype(np.float32)
    elif layer.target_scope == SCOPE_TARGET_STARLESS and sky_m is not None:
        return np.clip(m_base * sky_m, 0.0, 1.0).astype(np.float32)
    elif layer.target_scope == SCOPE_TARGET_STARS and stars_image is not None:
        st = stars_image
        star_luma = np.mean(st, axis=2) if st.ndim == 3 else st
        star_gate = _smoothstep(0.02, 0.15, star_luma)
        if star_gate.shape[:2] != (img_h, img_w):
            star_gate = cv2.resize(star_gate, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(m_base * star_gate, 0.0, 1.0).astype(np.float32)

    return m_base


def apply_all_mask_layers(
    composite_image: np.ndarray,
    mask_layers: List[MaskLayer],
    master_sky_mask: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_image_shape: Optional[Tuple[int, int]] = None,
    fast_draft: bool = False
) -> np.ndarray:
    """
    Aplica secuencialmente la composición de capas (estilo Photoshop),
    integrando capas de imagen (Fondo Starless, Suelo, Estrellas) y capas de ajuste locales.
    Soporta modo Zoom 100% mediante crop_rect y full_image_shape para alineación exacta 1:1.
    """
    if not mask_layers:
        return composite_image

    has_image_layers = any(l.layer_kind == LAYER_KIND_IMAGE for l in mask_layers)
    if not has_image_layers:
        # Modo clásico de solo máscaras de ajuste sobre la imagen compuesta
        img = composite_image.copy()
        for layer in mask_layers:
            if not layer.enabled or layer.opacity <= 1e-4:
                continue
            m_eff = compute_effective_layer_mask(
                layer, img,
                master_sky_mask=master_sky_mask,
                starless_image=starless_image,
                stars_image=stars_image,
                crop_rect=crop_rect,
                full_image_shape=full_image_shape
            )
            img = apply_local_adjustments_to_image(img, m_eff, layer, fast_draft=fast_draft)
        return img

    # =========================================================================
    # MODO COMPOSITOR MULTICAPA COMPLETO (Estilo Photoshop: evaluación Bottom-Up)
    # =========================================================================
    # En la interfaz de usuario, la fila 0 es la capa superior (TOP), y la última fila es la base (BOTTOM).
    # Para evaluar la pila de abajo hacia arriba, invertimos el orden de las capas.
    eval_layers = list(reversed(mask_layers))

    # Base inicial del lienzo (Fondo estirado y ajustado por el pipeline)
    canvas = composite_image.copy()
    img_h, img_w = canvas.shape[:2]

    # Máscara 2D de cielo y suelo normalizada
    sky_m = None
    gnd_m = None
    if master_sky_mask is not None:
        sky_m = master_sky_mask[..., 0] if master_sky_mask.ndim == 3 else master_sky_mask
        if sky_m.shape[:2] != (img_h, img_w):
            if crop_rect is not None and full_image_shape is not None:
                x1, y1, x2, y2 = crop_rect
                h_f, w_f = full_image_shape
                h_m, w_m = sky_m.shape[:2]
                sx = w_m / float(w_f)
                sy = h_m / float(h_f)
                sx1 = max(0, min(w_m, int(round(x1 * sx))))
                sy1 = max(0, min(h_m, int(round(y1 * sy))))
                sx2 = max(0, min(w_m, int(round(x2 * sx))))
                sy2 = max(0, min(h_m, int(round(y2 * sy))))
                sky_sub = sky_m[sy1:sy2, sx1:sx2]
                if sky_sub.shape[:2] != (img_h, img_w):
                    sky_m = cv2.resize(sky_sub, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                else:
                    sky_m = sky_sub
            else:
                sky_m = cv2.resize(sky_m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        gnd_m = np.clip(1.0 - sky_m, 0.0, 1.0).astype(np.float32)

    for layer in eval_layers:
        if not layer.enabled:
            continue

        if layer.mask_type in (MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_BASE):
            # Capa 🖼️ Fondo Base o 🌌 Fondo Starless
            has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
                       abs(layer.exposure) > 1e-3 or abs(layer.contrast) > 1e-3 or 
                       abs(layer.saturation - 1.0) > 1e-3 or abs(layer.shadows) > 1e-3 or 
                       abs(layer.highlights) > 1e-3 or layer.has_curve())
            if has_adj:
                canvas = apply_local_adjustments_to_image(canvas, np.ones((img_h, img_w), dtype=np.float32), layer, fast_draft=fast_draft)
            if layer.opacity < 0.999:
                canvas = canvas * layer.opacity

        elif layer.mask_type == MASK_TYPE_IMAGE_GROUND:
            # Capa 🏞️ Suelo / Terreno
            if gnd_m is not None:
                eff_m = np.clip(gnd_m * layer.opacity, 0.0, 1.0)
                # Si el suelo está encima de las estrellas, restaura el suelo limpio sin estrellas debajo
                w_3d = eff_m[..., np.newaxis]
                canvas = (composite_image * w_3d) + (canvas * (1.0 - w_3d))
                canvas = apply_local_adjustments_to_image(canvas, eff_m, layer, fast_draft=fast_draft)

        elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
            # Capa ⭐ Estrellas
            if stars_image is not None and layer.opacity > 1e-4:
                st = stars_image.copy()
                if st.shape[:2] != (img_h, img_w):
                    if crop_rect is not None and full_image_shape is not None:
                        x1, y1, x2, y2 = crop_rect
                        h_f, w_f = full_image_shape
                        h_s, w_s = st.shape[:2]
                        sx = w_s / float(w_f)
                        sy = h_s / float(h_f)
                        sx1 = max(0, min(w_s, int(round(x1 * sx))))
                        sy1 = max(0, min(h_s, int(round(y1 * sy))))
                        sx2 = max(0, min(w_s, int(round(x2 * sx))))
                        sy2 = max(0, min(h_s, int(round(y2 * sy))))
                        st_sub = st[sy1:sy2, sx1:sx2]
                        if st_sub.shape[:2] != (img_h, img_w):
                            st = cv2.resize(st_sub, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                        else:
                            st = st_sub
                    else:
                        st = cv2.resize(st, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                # Ajustes cromáticos propios de la capa de estrellas
                has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
                           abs(layer.exposure) > 1e-3 or abs(layer.contrast) > 1e-3 or 
                           abs(layer.saturation - 1.0) > 1e-3 or layer.has_curve())
                if has_adj:
                    st = apply_local_adjustments_to_image(st, np.ones((img_h, img_w), dtype=np.float32), layer, fast_draft=fast_draft)

                # Mezcla Screen (Trama estilo Photoshop): 1 - (1 - canvas) * (1 - st * opacity)
                # Garantiza que las estrellas no quemen núcleos y se fundan de forma natural
                if HAS_NUMBA:
                    st_c = np.ascontiguousarray(st, dtype=np.float32)
                    canvas = _numba_screen_blend(canvas, st_c, float(layer.opacity))
                else:
                    st_weighted = np.clip(st * layer.opacity, 0.0, 1.0)
                    canvas = 1.0 - (1.0 - canvas) * (1.0 - st_weighted)

        else:
            # Capas de ajuste estándar (Pincel, Luminancia, Color, etc.)
            if layer.opacity > 1e-4:
                m_eff = compute_effective_layer_mask(
                    layer, canvas,
                    master_sky_mask=master_sky_mask,
                    starless_image=starless_image,
                    stars_image=stars_image,
                    crop_rect=crop_rect,
                    full_image_shape=full_image_shape
                )
                canvas = apply_local_adjustments_to_image(canvas, m_eff, layer, fast_draft=fast_draft)

    return np.ascontiguousarray(np.clip(canvas, 0.0, 1.0), dtype=np.float32)


def get_canvas_below_layer(
    composite_image: np.ndarray,
    mask_layers: List[MaskLayer],
    target_layer_id: str,
    master_sky_mask: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Calcula el estado del lienzo directamente por debajo de la capa indicada (target_layer_id).
    En el modelo de Photoshop, una capa evalúa su máscara (ej. Luminancia/Color) y sus ajustes
    sobre el lienzo compuesto por todas las capas que tiene situadas por debajo en la pila.
    """
    if not mask_layers or not target_layer_id:
        return composite_image

    idx = next((i for i, l in enumerate(mask_layers) if l.id == target_layer_id), -1)
    if idx < 0:
        return composite_image

    # En la lista de UI (fila 0 = TOP, fila N-1 = BOTTOM):
    # Las capas estrictamente por debajo de 'target_layer_id' son las que tienen índice > idx
    layers_below = mask_layers[idx + 1 :]
    if not layers_below:
        return composite_image

    return apply_all_mask_layers(
        composite_image,
        layers_below,
        master_sky_mask=master_sky_mask,
        starless_image=starless_image,
        stars_image=stars_image,
        crop_rect=crop_rect,
        full_image_shape=full_image_shape
    )


