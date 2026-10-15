# core/mask_layers.py
"""
core/mask_layers.py - Sistema de capas y máscaras de ajuste local.
Permite apilar múltiples máscaras (Pincel, Suelo/Cielo, Luminancia, Rango de Color) con ajustes
locales paramétricos independientes y conciencia de capas StarNet (Compuesta, Suelo, Fondo Starless, Estrellas).
"""
import uuid
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from core.masking import generate_luminance_mask, generate_color_range_mask
from core.stretch import apply_white_balance, adjust_contrast


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

    # Máscara rasterizada en memoria float32 [0.0, 1.0] (no serializable a JSON)
    custom_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la capa a un diccionario JSON preservando la máscara de pincel."""
        d = asdict(self)
        d.pop("custom_mask", None)
        d.pop("brush_mask", None)
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
        return inst


def compute_layer_mask(
    layer: MaskLayer,
    base_rgb: np.ndarray,
    master_sky_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calcula o evalúa la máscara 2D [0.0, 1.0] correspondiente a la capa.
    Aplica inversión si la capa lo requiere.
    """
    h, w = base_rgb.shape[:2]

    if layer.mask_type == MASK_TYPE_BRUSH:
        if layer.brush_mask is not None:
            m = layer.brush_mask.copy()
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            m = np.zeros((h, w), dtype=np.float32)

    elif layer.mask_type == MASK_TYPE_GROUND:
        if master_sky_mask is not None:
            m = 1.0 - master_sky_mask
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Si no hay máscara de suelo, todo activo
            m = np.ones((h, w), dtype=np.float32)

    elif layer.mask_type == MASK_TYPE_SKY:
        if master_sky_mask is not None:
            m = master_sky_mask.copy()
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
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
        m = layer.custom_mask
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        m = np.ones((h, w), dtype=np.float32)

    # Inversión de máscara
    if layer.inverted:
        m = 1.0 - m

    return np.clip(m, 0.0, 1.0).astype(np.float32)


def apply_local_adjustments_to_image(
    image_rgb: np.ndarray,
    effective_mask: np.ndarray,
    layer: MaskLayer
) -> np.ndarray:
    """
    Aplica los controles locales de una capa sobre image_rgb ponderados por effective_mask y opacidad.
    """
    if not layer.enabled or layer.opacity <= 1e-4:
        return image_rgb

    # Ponderación global de la máscara por opacidad
    weight = np.clip(effective_mask * layer.opacity, 0.0, 1.0).astype(np.float32)
    if np.max(weight) <= 1e-4:
        return image_rgb

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
        adj = adjust_contrast(adj, layer.contrast)

    # 6. Saturación local
    if abs(layer.saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(np.clip(adj, 0.0, 1.0), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * layer.saturation, 0.0, 1.0)
        adj = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

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
    stars_image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calcula la máscara final efectiva considerando su tipo, parámetros, inversión
    y modulación según el ámbito de aplicación (Target Scope: todo, suelo, sin estrellas, estrellas).
    """
    m_base = compute_layer_mask(layer, composite_image, master_sky_mask=master_sky_mask)
    img_h, img_w = composite_image.shape[:2]

    sky_m = None
    if master_sky_mask is not None:
        sky_m = master_sky_mask[..., 0] if master_sky_mask.ndim == 3 else master_sky_mask
        if sky_m.shape[:2] != (img_h, img_w):
            sky_m = cv2.resize(sky_m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    # Modular según el ámbito de aplicación (Target Scope)
    if layer.target_scope == SCOPE_TARGET_GROUND:
        if sky_m is not None:
            gnd_m = 1.0 - sky_m
            return np.clip(m_base * gnd_m, 0.0, 1.0).astype(np.float32)
        return m_base

    elif layer.target_scope == SCOPE_TARGET_STARLESS:
        if sky_m is not None:
            return np.clip(m_base * sky_m, 0.0, 1.0).astype(np.float32)
        return m_base

    elif layer.target_scope == SCOPE_TARGET_STARS:
        if stars_image is not None:
            st = stars_image
            if st.ndim == 3:
                star_luma = np.mean(st, axis=2)
            else:
                star_luma = st
            star_gate = _smoothstep(0.02, 0.15, star_luma)
            if star_gate.shape[:2] != (img_h, img_w):
                star_gate = cv2.resize(star_gate, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            return np.clip(m_base * star_gate, 0.0, 1.0).astype(np.float32)
        return m_base

    return m_base


def apply_all_mask_layers(
    composite_image: np.ndarray,
    mask_layers: List[MaskLayer],
    master_sky_mask: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Aplica secuencialmente la composición de capas (estilo Photoshop),
    integrando capas de imagen (Fondo Starless, Suelo, Estrellas) y capas de ajuste locales.
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
                stars_image=stars_image
            )
            img = apply_local_adjustments_to_image(img, m_eff, layer)
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
            sky_m = cv2.resize(sky_m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        gnd_m = np.clip(1.0 - sky_m, 0.0, 1.0).astype(np.float32)

    for layer in eval_layers:
        if not layer.enabled:
            continue

        if layer.mask_type == MASK_TYPE_IMAGE_STARLESS:
            # Capa 🌌 Fondo Starless
            has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
                       abs(layer.exposure) > 1e-3 or abs(layer.contrast) > 1e-3 or 
                       abs(layer.saturation - 1.0) > 1e-3 or abs(layer.shadows) > 1e-3 or 
                       abs(layer.highlights) > 1e-3)
            if has_adj:
                canvas = apply_local_adjustments_to_image(canvas, np.ones((img_h, img_w), dtype=np.float32), layer)
            if layer.opacity < 0.999:
                canvas = canvas * layer.opacity

        elif layer.mask_type == MASK_TYPE_IMAGE_GROUND:
            # Capa 🏞️ Suelo / Terreno
            if gnd_m is not None:
                eff_m = np.clip(gnd_m * layer.opacity, 0.0, 1.0)
                # Si el suelo está encima de las estrellas, restaura el suelo limpio sin estrellas debajo
                w_3d = eff_m[..., np.newaxis]
                canvas = (composite_image * w_3d) + (canvas * (1.0 - w_3d))
                canvas = apply_local_adjustments_to_image(canvas, eff_m, layer)

        elif layer.mask_type == MASK_TYPE_IMAGE_STARS:
            # Capa ⭐ Estrellas
            if stars_image is not None and layer.opacity > 1e-4:
                st = stars_image.copy()
                if st.shape[:2] != (img_h, img_w):
                    st = cv2.resize(st, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                # Ajustes cromáticos propios de la capa de estrellas
                has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
                           abs(layer.exposure) > 1e-3 or abs(layer.contrast) > 1e-3 or 
                           abs(layer.saturation - 1.0) > 1e-3)
                if has_adj:
                    st = apply_local_adjustments_to_image(st, np.ones((img_h, img_w), dtype=np.float32), layer)

                # Mezcla Screen (Trama estilo Photoshop): 1 - (1 - canvas) * (1 - st * opacity)
                # Garantiza que las estrellas no quemen núcleos y se fundan de forma natural
                st_weighted = np.clip(st * layer.opacity, 0.0, 1.0)
                canvas = 1.0 - (1.0 - canvas) * (1.0 - st_weighted)

        else:
            # Capas de ajuste estándar (Pincel, Luminancia, Color, etc.)
            if layer.opacity > 1e-4:
                m_eff = compute_effective_layer_mask(
                    layer, canvas,
                    master_sky_mask=master_sky_mask,
                    starless_image=starless_image,
                    stars_image=stars_image
                )
                canvas = apply_local_adjustments_to_image(canvas, m_eff, layer)

    return np.ascontiguousarray(np.clip(canvas, 0.0, 1.0), dtype=np.float32)

