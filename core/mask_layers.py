# core/mask_layers.py
"""
core/mask_layers.py - Sistema de capas de ajuste local con máscaras (estilo Lightroom / Camera Raw).
Permite apilar múltiples máscaras (Suelo/Cielo, Luminancia, Rango de Color) con ajustes
locales paramétricos independientes y conciencia de capas StarNet (Compuesta, Suelo, Fondo Starless, Estrellas).
"""
import uuid
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

# Tipos de Máscara
MASK_TYPE_GROUND = "ground"         # Máscara Suelo (invertida del cielo)
MASK_TYPE_SKY = "sky"               # Máscara Cielo
MASK_TYPE_LUMA = "luminance"        # Rango de Luminancia (Sombras/Medios/Luces)
MASK_TYPE_COLOR = "color"           # Rango de Color (HSV)
MASK_TYPE_CUSTOM = "custom"         # Trazos libres / raster personalizado


@dataclass
class MaskLayer:
    """Representa una capa de ajuste local con su propia máscara y parámetros."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Nueva Máscara"
    mask_type: str = MASK_TYPE_LUMA
    target_scope: str = SCOPE_TARGET_ALL
    enabled: bool = True
    inverted: bool = False
    opacity: float = 1.0  # 0.0 a 1.0

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
    temp: float = 0.0               # -50.0 a +50.0 (K local)
    tint: float = 0.0               # -50.0 a +50.0 (tinte local)

    # Máscara rasterizada en memoria float32 [0.0, 1.0] (no serializable a JSON)
    custom_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la capa a un diccionario excluyendo arrays pesados."""
        d = asdict(self)
        d.pop("custom_mask", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskLayer":
        """Instancia una capa a partir de un diccionario."""
        data_clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**data_clean)


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

    if layer.mask_type == MASK_TYPE_GROUND:
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
    if abs(layer.temp) > 1e-3 or abs(layer.tint) > 1e-3:
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
    Aplica secuencialmente todas las capas de ajuste local activas considerando su ámbito (Target Scope).
    """
    if not mask_layers:
        return composite_image

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

