# core/mask_layers.py
"""
core/mask_layers.py - Sistema de capas y máscaras de ajuste local.
Permite apilar múltiples máscaras (Pincel, Suelo/Cielo, Luminancia, Rango de Color) con ajustes
locales paramétricos independientes y conciencia de capas StarNet (Compuesta, Suelo, Fondo Starless, Estrellas).
"""
import uuid
import copy
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
import cv2
import numpy as np
from core.masking import (
    generate_luminance_mask,
    generate_color_range_mask,
    refine_mask_guided_cross,
    guided_cross
)
from core.stretch import apply_white_balance, adjust_contrast

try:
    import astroloom_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

# Fallback vectorizado puro en NumPy/OpenCV con paridad matemática exacta a Rust
def _numpy_fused_layer_adjust(
    orig_img: np.ndarray,
    adj_source: np.ndarray,
    mask: np.ndarray,
    weight_scale: float,
    gain_r: float,
    gain_g: float,
    gain_b: float,
    shadows: float,
    highlights: float,
    has_curve: bool,
    curve_lut: np.ndarray
) -> np.ndarray:
    adj = adj_source.copy()
    if abs(gain_r - 1.0) > 1e-4 or abs(gain_g - 1.0) > 1e-4 or abs(gain_b - 1.0) > 1e-4:
        adj[..., 0] *= gain_r
        adj[..., 1] *= gain_g
        adj[..., 2] *= gain_b
        adj = np.clip(adj, 0.0, 1.0)

    if abs(shadows) > 1e-3:
        lift = (1.0 - adj) ** 2
        adj = adj * (1.0 + shadows * lift)

    if abs(highlights) > 1e-3:
        hi = adj ** 2
        adj = adj + (highlights * hi * 0.4)

    adj = np.clip(adj, 0.0, 1.0)

    if has_curve:
        lut_len = len(curve_lut) - 1
        idx = np.clip((adj * lut_len).astype(np.int32), 0, lut_len)
        adj = curve_lut[idx]

    m = mask * weight_scale
    if m.ndim == 2:
        m = m[..., np.newaxis]
    return np.clip(adj * m + orig_img * (1.0 - m), 0.0, 1.0).astype(np.float32)


# Ámbitos de aplicación (Target Scope)
SCOPE_TARGET_ALL = "all"            # Toda la imagen (Compuesta)
SCOPE_TARGET_GROUND = "ground"      # Solo Suelo (aislado por la máscara de suelo)
SCOPE_TARGET_STARLESS = "starless"  # Solo Fondo Starless (estrellas desacopladas)
SCOPE_TARGET_STARS = "stars"        # Solo Capa de Estrellas

# Naturaleza de la Capa (Layer Kind)
LAYER_KIND_ADJUSTMENT = "adjustment"  # Capa de ajuste paramétrico con máscara
LAYER_KIND_IMAGE = "image"            # Capa portadora de imagen (Fondo Starless, Estrellas, Suelo, Ráster)

# Modos de Fusión para Capas Ráster (Blend Modes)
BLEND_MODE_NORMAL = "normal"          # Sustitución ponderada por opacidad
BLEND_MODE_SCREEN = "screen"          # Trama (1 - (1 - A) * (1 - B))
BLEND_MODE_LUMINOSITY = "luminosity"  # Inyección de luminancia del filtro preservando crominancia base
BLEND_MODE_SOFT_LIGHT = "soft_light"  # Realce sutil de contraste tonal
BLEND_MODE_OVERLAY = "overlay"        # Superposición de microestructuras
BLEND_MODE_MULTIPLY = "multiply"      # Atenuación de resplandores / máscaras de densidad

ALL_BLEND_MODES = [
    BLEND_MODE_NORMAL,
    BLEND_MODE_SCREEN,
    BLEND_MODE_LUMINOSITY,
    BLEND_MODE_SOFT_LIGHT,
    BLEND_MODE_OVERLAY,
    BLEND_MODE_MULTIPLY,
]

# Tipos de Máscara de Ajuste
MASK_TYPE_BRUSH = "brush"           # Pincel interactivo con dureza y opacidad de trazo
MASK_TYPE_GROUND = "ground"         # Máscara Suelo (invertida del cielo)
MASK_TYPE_SKY = "sky"               # Máscara Cielo
MASK_TYPE_LUMA = "luminance"        # Rango de Luminancia (Sombras/Medios/Luces)
MASK_TYPE_COLOR = "color"           # Rango de Color (HSV)
MASK_TYPE_GRADIENT_LINEAR = "gradient_linear"  # Filtro Graduado Lineal vectorial
MASK_TYPE_GRADIENT_RADIAL = "gradient_radial"  # Filtro Graduado Radial vectorial
MASK_TYPE_CUSTOM = "custom"         # Trazos libres / raster personalizado

# Restricción espacial de máscara (Edge-Snapping al horizonte / Suelo)
LIMIT_MASK_NONE = "none"            # Afecta a toda la escena sin recorte
LIMIT_MASK_SKY = "sky"              # Restringido exclusivamente al cielo
LIMIT_MASK_GROUND = "ground"        # Restringido exclusivamente al suelo

# Tipos Especiales de Capas de Imagen (StarNet / Pipeline / Eclipses)
MASK_TYPE_IMAGE_BASE = "image_base"          # 🖼️ Fondo (Imagen Base inicial)
MASK_TYPE_IMAGE_STARLESS = "image_starless"  # 🌌 Fondo Nebular (Starless AI)
MASK_TYPE_IMAGE_STARS = "image_stars"        # ⭐ Capa de Estrellas (StarNet AI)
MASK_TYPE_IMAGE_GROUND = "image_ground"      # 🏞️ Capa de Suelo / Terreno (Paisaje)
MASK_TYPE_IMAGE_RASTER = "image_raster"      # 🎨 Capa Ráster de Imagen (Filtros Eclipses / Composición)


# Campos pesados o derivados que nunca deben clonarse al serializar a JSON
_MASK_LAYER_EXCLUDED_FIELDS = frozenset({
    "custom_mask",
    "image_data",
    "proxy_preview",
    "proxy_draft",
    "brush_mask",
    "brush_mask_preview",
    "brush_mask_draft",
    "curve_lut",
})


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

    # Propiedades para capas de imagen ráster (layer_kind == "image")
    image_data: Optional[np.ndarray] = None     # Array Float32 RGB [0.0, 1.0] o 2D Mono
    proxy_preview: Optional[np.ndarray] = None  # Mipmap Nivel 1 (<= 1600 px) para vista previa interactiva
    proxy_draft: Optional[np.ndarray] = None    # Mipmap Nivel 2 (<= 1000 px) para arrastre a 60 fps
    blend_mode: str = BLEND_MODE_NORMAL         # Modo de fusión ("normal", "screen", "luminosity", etc.)
    active_target: str = "image"                # "image" o "mask" (foco de edición en miniatura dual)

    # Transformación espacial 2D (Mover y Redimensionar)
    offset_x: float = 0.0                       # Desplazamiento horizontal relativo [-1.0, 1.0] (fracción de ancho)
    offset_y: float = 0.0                       # Desplazamiento vertical relativo [-1.0, 1.0] (fracción de alto)
    scale_x: float = 1.0                        # Factor de escala horizontal (0.05x a 10.0x, 1.0 = 100%)
    scale_y: float = 1.0                        # Factor de escala vertical (0.05x a 10.0x, 1.0 = 100%)
    scale: float = 1.0                          # Factor de escala uniforme (compatibilidad hacia atrás)
    angle: float = 0.0                          # Ángulo de rotación en grados (-180.0° a +180.0°)

    @staticmethod
    def _sanitize_image_data(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Sanea y normaliza arrays de imagen ráster una sola vez al ingresarlos."""
        if img is None:
            return None
        flt = img
        if flt.dtype != np.float32:
            if flt.dtype == np.uint8:
                flt = flt.astype(np.float32) / 255.0
            elif flt.dtype == np.uint16:
                flt = flt.astype(np.float32) / 65535.0
            else:
                flt = flt.astype(np.float32)
        if flt.ndim == 2:
            flt = np.repeat(flt[..., np.newaxis], 3, axis=-1)
        elif flt.ndim == 3 and flt.shape[2] == 1:
            flt = np.repeat(flt, 3, axis=-1)
        if np.isnan(flt).any() or np.isinf(flt).any():
            flt = np.nan_to_num(flt, nan=0.0, posinf=1.0, neginf=0.0)
        return np.ascontiguousarray(flt, dtype=np.float32)

    def _generate_proxies(self) -> None:
        """Genera pirámides de resolución (mipmaps) para capas ráster."""
        if self.image_data is None:
            self.proxy_preview = None
            self.proxy_draft = None
            return
        h, w = self.image_data.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1600:
            scale_p = 1600.0 / float(max_dim)
            pw = max(4, (int(w * scale_p) // 4) * 4)
            ph = max(4, (int(h * scale_p) // 4) * 4)
            self.proxy_preview = np.ascontiguousarray(
                cv2.resize(self.image_data, (pw, ph), interpolation=cv2.INTER_AREA),
                dtype=np.float32
            )
        else:
            self.proxy_preview = self.image_data

        p_h, p_w = self.proxy_preview.shape[:2]
        p_max = max(p_h, p_w)
        if p_max > 1000:
            scale_d = 1000.0 / float(p_max)
            dw = max(4, (int(p_w * scale_d) // 4) * 4)
            dh = max(4, (int(p_h * scale_d) // 4) * 4)
            self.proxy_draft = np.ascontiguousarray(
                cv2.resize(self.proxy_preview, (dw, dh), interpolation=cv2.INTER_AREA),
                dtype=np.float32
            )
        else:
            self.proxy_draft = self.proxy_preview

    def __post_init__(self):
        if self.scale != 1.0 and self.scale_x == 1.0 and self.scale_y == 1.0:
            self.scale_x = self.scale
            self.scale_y = self.scale
        elif (self.scale_x != 1.0 or self.scale_y != 1.0) and self.scale == 1.0:
            self.scale = self.scale_x

        if self.image_data is not None:
            self.image_data = self._sanitize_image_data(self.image_data)
            self._generate_proxies()

    def set_image_data(self, img: Optional[np.ndarray]) -> None:
        """Asigna y sanea una nueva matriz de píxeles ráster, regenerando sus mipmaps."""
        self.image_data = self._sanitize_image_data(img)
        self._generate_proxies()

    def is_transform_identity(self) -> bool:
        """Determina si la capa no tiene transformación espacial (offset 0, escala 1, rotación 0)."""
        return (abs(self.offset_x) < 1e-4 and 
                abs(self.offset_y) < 1e-4 and 
                abs(self.scale_x - 1.0) < 1e-4 and 
                abs(self.scale_y - 1.0) < 1e-4 and 
                abs(self.angle) < 1e-3)

    # Parámetros para máscara de Pincel interactivo
    brush_radius: int = 35          # Radio en píxeles (5 a 250 px)
    brush_hardness: float = 0.5     # Dureza: 0.0 (suave/difuminado) a 1.0 (borde neto)
    brush_flow: float = 0.8         # Flujo / opacidad por pasada (0.05 a 1.00)
    brush_mode: str = "paint"       # "paint" o "erase"
    brush_mask: Optional[np.ndarray] = None  # Float32 2D [0.0, 1.0]
    brush_mask_preview: Optional[np.ndarray] = None  # Mipmap Nivel 1 (<= 1600 px)
    brush_mask_draft: Optional[np.ndarray] = None    # Mipmap Nivel 2 (<= 800 px)
    brush_mask_version: int = 0                      # Contador de versión para invalidación de cachés

    def invalidate_brush_proxies(self) -> None:
        """Invalida los mipmaps de la máscara de pincel e incrementa su versión."""
        self.brush_mask_version += 1
        self.brush_mask_preview = None
        self.brush_mask_draft = None
        self._cached_brush_id = None

    def set_brush_mask(self, mask: Optional[np.ndarray]) -> None:
        """Asigna una nueva máscara de pincel e invalida los proxies cacheados."""
        self.brush_mask = mask
        self.invalidate_brush_proxies()

    def get_brush_proxy(self, level: str) -> Optional[np.ndarray]:
        """Obtiene o genera de forma perezosa el proxy de pincel para 'draft' (<=800px) o 'preview' (<=1600px)."""
        if self.brush_mask is None:
            return None

        # Si el array ha cambiado de identidad en memoria, auto-invalidar
        current_id = id(self.brush_mask)
        if getattr(self, "_cached_brush_id", None) != current_id:
            self.brush_mask_preview = None
            self.brush_mask_draft = None
            self._cached_brush_id = current_id

        h, w = self.brush_mask.shape[:2]
        max_dim = max(h, w)

        if level == "draft":
            if self.brush_mask_draft is not None:
                return self.brush_mask_draft
            if max_dim <= 800:
                self.brush_mask_draft = self.brush_mask
            else:
                scale_d = 800.0 / float(max_dim)
                dw = max(4, (int(w * scale_d) // 4) * 4)
                dh = max(4, (int(h * scale_d) // 4) * 4)
                src = self.brush_mask_preview if self.brush_mask_preview is not None else self.brush_mask
                self.brush_mask_draft = np.ascontiguousarray(
                    cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR),
                    dtype=np.float32
                )
            return self.brush_mask_draft

        # level == "preview"
        if self.brush_mask_preview is not None:
            return self.brush_mask_preview
        if max_dim <= 1600:
            self.brush_mask_preview = self.brush_mask
        else:
            scale_p = 1600.0 / float(max_dim)
            pw = max(4, (int(w * scale_p) // 4) * 4)
            ph = max(4, (int(h * scale_p) // 4) * 4)
            self.brush_mask_preview = np.ascontiguousarray(
                cv2.resize(self.brush_mask, (pw, ph), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32
            )
        return self.brush_mask_preview

    # Parámetros para máscara de rango de luminancia
    luma_min: float = 0.0
    luma_max: float = 1.0
    feather: float = 0.08

    # Parámetros para máscara de rango cromático
    color_hue: float = 0.0          # 0.0 a 360.0 grados
    color_tolerance: float = 30.0   # Tolerancia angular en grados
    color_min_sat: float = 0.08     # Saturación mínima

    # Parámetros para máscara de Gradiente Lineal vectorial
    linear_p0: Tuple[float, float] = (0.5, 0.75)    # Punto inicio / cresta (1.0) coords normalizadas (x, y)
    linear_p1: Tuple[float, float] = (0.5, 0.35)    # Punto fin / borde A (0.0) coords normalizadas (x, y)
    linear_midpoint: float = 0.5                    # Posición media (0.01 a 0.99)
    linear_profile: str = "smoothstep"              # "smoothstep", "linear", "exponential", "reflected"
    linear_symmetric: bool = True                   # Si la banda reflejada es simétrica (Lado B = Lado A)
    linear_span_b: float = 1.0                      # Factor de escala ancho Lado B (0.1x a 3.0x, 1.0 = 100%)

    # Parámetros para máscara de Gradiente Radial vectorial
    radial_center: Tuple[float, float] = (0.5, 0.5) # Centro de la elipse (cx, cy) normalizado [0.0, 1.0]
    radial_radius_x: float = 0.25                   # Semieje horizontal normalizado (0.01 a 2.0)
    radial_radius_y: float = 0.35                   # Semieje vertical normalizado (0.01 a 2.0)
    radial_angle: float = 0.0                       # Inclinación en grados (-180.0° a +180.0°)
    radial_feather: float = 0.5                     # Desvanecimiento interior (0.05 a 1.00)

    # Restricción espacial de máscara (Edge-Snapping al horizonte / Suelo)
    limit_mask: str = LIMIT_MASK_NONE               # "none", "sky", "ground"

    # Refinamiento de bordes por Filtro Cruzado Guiado (guided_cross / LightCraft)
    refine_edges: bool = False      # Activa refinado automático contra la escena
    refine_strength: float = 0.0    # Intensidad del refinado [0.0, 1.0]
    refine_radius: Optional[int] = None  # Radio de filtrado (None = 0.04 * long_edge)

    # Ajustes locales de revelado
    exposure: float = 0.0           # -3.00 EV a +3.00 EV
    contrast: float = 0.0           # -1.00 a +1.00
    shadows: float = 0.0            # -1.00 a +1.00 (recuperar o profundizar)
    highlights: float = 0.0         # -1.00 a +1.00 (atenuar o iluminar luces)
    saturation: float = 1.0         # 0.00x a 2.50x
    temp: float = 0.0               # -1.0000 a +1.0000 (factor de balance cálido/frío)
    tint: float = 0.0               # -1.0000 a +1.0000 (factor de balance verde/magenta)

    # Máscara de Enfoque y Nitidez (USM con detección de bordes)
    sharp_amount: float = 0.0       # 0.0 a 150.0 (0 = desactivado)
    sharp_radius: float = 1.0       # 0.5 a 3.0 px
    sharp_detail: float = 25.0      # 0.0 a 100.0 (amortiguación de micro-ruido)
    sharp_masking: float = 0.0      # 0.0 a 100.0 (umbral de bordes)

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

    @property
    def supports_brush(self) -> bool:
        """Determina si la capa admite trazos de pincel interactivo para definir su máscara espacial."""
        return self.mask_type in (MASK_TYPE_BRUSH, MASK_TYPE_IMAGE_RASTER)

    @property
    def is_raster_image(self) -> bool:
        """Indica si la capa transporta una imagen ráster externa (p. ej. filtro coronal o composición)."""
        return self.mask_type == MASK_TYPE_IMAGE_RASTER

    @property
    def is_gradient(self) -> bool:
        """Indica si la capa es un filtro graduado analítico (lineal o radial)."""
        return self.mask_type in (MASK_TYPE_GRADIENT_LINEAR, MASK_TYPE_GRADIENT_RADIAL)

    def to_dict(self, encode_mask: bool = False) -> Dict[str, Any]:
        """Serializa la capa a un diccionario JSON preservando parámetros y curva sin deepcopy."""
        d: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            if k not in _MASK_LAYER_EXCLUDED_FIELDS and not k.startswith("_"):
                val = getattr(self, k)
                if k == "curve_points" and val is not None:
                    d[k] = list(val)
                else:
                    d[k] = val

        if encode_mask and self.brush_mask is not None:
            try:
                u8 = (np.clip(self.brush_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                ok, buf = cv2.imencode(".png", u8)
                if ok:
                    d["brush_mask_b64"] = base64.b64encode(buf).decode("ascii")
            except Exception:
                pass
        return d

    def shallow_snapshot(self) -> "MaskLayer":
        """Crea una instantánea superficial rápida aislando escalares para el hilo de render."""
        snap = copy.copy(self)
        if snap.curve_points:
            snap.curve_points = list(snap.curve_points)
        return snap

    def __eq__(self, other: object) -> bool:
        """Comparación por valor de capas para detección precisa de cambios en historial sin ValueError en arrays."""
        if not isinstance(other, MaskLayer):
            return False
        if self.id != other.id:
            return False
        if self.brush_mask_version != other.brush_mask_version:
            return False
        if (self.brush_mask is None) != (other.brush_mask is None):
            return False
        if (self.image_data is None) != (other.image_data is None):
            return False
        if self.image_data is not None and other.image_data is not None:
            if self.image_data is not other.image_data and id(self.image_data) != id(other.image_data):
                if self.image_data.shape != other.image_data.shape:
                    return False
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskLayer":
        """Instancia una capa a partir de un diccionario JSON o estructura en memoria."""
        data_clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "curve_points" in data_clean and data_clean["curve_points"]:
            data_clean["curve_points"] = [tuple(p) for p in data_clean["curve_points"]]
        if "linear_p0" in data_clean and data_clean["linear_p0"]:
            data_clean["linear_p0"] = tuple(data_clean["linear_p0"])
        if "linear_p1" in data_clean and data_clean["linear_p1"]:
            data_clean["linear_p1"] = tuple(data_clean["linear_p1"])
        if "radial_center" in data_clean and data_clean["radial_center"]:
            data_clean["radial_center"] = tuple(data_clean["radial_center"])
        inst = cls(**data_clean)
        b64_str = data.get("brush_mask_b64")
        if b64_str:
            try:
                buf = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
                dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if dec is not None:
                    inst.brush_mask = dec.astype(np.float32) / 255.0
                    inst.invalidate_brush_proxies()
            except Exception:
                pass
        elif "brush_mask" in data and isinstance(data["brush_mask"], np.ndarray):
            inst.brush_mask = data["brush_mask"]
            inst.invalidate_brush_proxies()

        if "image_data" in data and isinstance(data["image_data"], np.ndarray):
            inst.set_image_data(data["image_data"])

        if not inst.is_curve_identity() and inst.curve_lut is None:
            inst.curve_lut = compute_curve_lut_from_points(inst.curve_points)
        return inst


def compute_linear_gradient_py(
    h: int, w: int,
    x0: float, y0: float,
    x1: float, y1: float,
    midpoint: float = 0.5,
    profile: str = "smoothstep",
    span_b: float = 1.0,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Calcula analíticamente una máscara de gradiente lineal continua en Python/NumPy."""
    if crop_rect is not None and full_shape is not None:
        cx1, cy1, _, _ = crop_rect
        fh, fw = full_shape
        fh_f = max(1.0, float(fh))
        fw_f = max(1.0, float(fw))
        x_coords = (np.arange(w, dtype=np.float32) + cx1) / fw_f
        y_coords = (np.arange(h, dtype=np.float32) + cy1) / fh_f
        aspect = fw_f / fh_f
    else:
        fh_f = max(1.0, float(h))
        fw_f = max(1.0, float(w))
        x_coords = np.arange(w, dtype=np.float32) / fw_f
        y_coords = np.arange(h, dtype=np.float32) / fh_f
        aspect = fw_f / fh_f

    vx = (x1 - x0) * aspect
    vy = y1 - y0
    len_sq = vx * vx + vy * vy
    if len_sq < 1e-7:
        return np.zeros((h, w), dtype=np.float32)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    dx = (grid_x - x0) * aspect
    dy = grid_y - y0
    proj = (dx * vx + dy * vy) / len_sq

    if profile == "reflected":
        span_b_safe = max(0.01, float(span_b))
        t = np.where(proj >= 0.0, proj, np.abs(proj) / span_b_safe)
        t = np.clip(t, 0.0, 1.0)
        m = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
        return np.clip(m, 0.0, 1.0).astype(np.float32)

    t = np.clip(proj, 0.0, 1.0)

    mid = max(0.01, min(0.99, float(midpoint)))
    if abs(mid - 0.5) > 1e-4:
        mask_low = t < mid
        t_adj = np.empty_like(t)
        t_adj[mask_low] = 0.5 * (t[mask_low] / mid)
        t_adj[~mask_low] = 0.5 + 0.5 * ((t[~mask_low] - mid) / (1.0 - mid))
        t = t_adj

    if profile == "linear":
        m = 1.0 - t
    elif profile == "exponential":
        m = (1.0 - t) ** 2
    else:  # smoothstep
        m = 1.0 - (3.0 * t * t - 2.0 * t * t * t)

    return np.clip(m, 0.0, 1.0).astype(np.float32)


def compute_radial_gradient_py(
    h: int, w: int,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_deg: float = 0.0,
    feather: float = 0.5,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Calcula analíticamente una máscara de gradiente radial elíptico en Python/NumPy."""
    if crop_rect is not None and full_shape is not None:
        cx1, cy1, _, _ = crop_rect
        fh, fw = full_shape
        fh_f = max(1.0, float(fh))
        fw_f = max(1.0, float(fw))
        x_coords = (np.arange(w, dtype=np.float32) + cx1) / fw_f
        y_coords = (np.arange(h, dtype=np.float32) + cy1) / fh_f
        aspect = fw_f / fh_f
    else:
        fh_f = max(1.0, float(h))
        fw_f = max(1.0, float(w))
        x_coords = np.arange(w, dtype=np.float32) / fw_f
        y_coords = np.arange(h, dtype=np.float32) / fh_f
        aspect = fw_f / fh_f

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    dx = grid_x - cx
    dy = grid_y - cy

    rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    inv_aspect = 1.0 / max(1e-5, aspect)
    x_rot = dx * cos_t + (dy * inv_aspect) * sin_t
    y_rot = -(dx * aspect) * sin_t + dy * cos_t

    rx_s = max(1e-5, abs(float(rx)))
    ry_s = max(1e-5, abs(float(ry)))
    d = np.sqrt((x_rot / rx_s) ** 2 + (y_rot / ry_s) ** 2)

    feather_s = max(0.01, min(1.0, float(feather)))
    d_in = max(0.0, 1.0 - feather_s)

    t = np.clip((d - d_in) / feather_s, 0.0, 1.0)
    m = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


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

    def _get_brush_source() -> Optional[np.ndarray]:
        if layer.brush_mask is None:
            return None
        if crop_rect is not None and full_image_shape is not None:
            return layer.brush_mask
        max_dim = max(h, w)
        if max_dim <= 800:
            return layer.get_brush_proxy("draft")
        elif max_dim <= 1600:
            return layer.get_brush_proxy("preview")
        return layer.brush_mask

    if layer.mask_type == MASK_TYPE_BRUSH:
        brush_src = _get_brush_source()
        if brush_src is not None:
            m = _slice_or_resize_mask(brush_src)
        else:
            m = np.zeros((h, w), dtype=np.float32)

    elif layer.mask_type == MASK_TYPE_IMAGE_RASTER:
        brush_src = _get_brush_source()
        if brush_src is not None:
            m = _slice_or_resize_mask(brush_src)
        else:
            m = np.ones((h, w), dtype=np.float32)

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

    elif layer.mask_type == MASK_TYPE_GRADIENT_LINEAR:
        span_b = 1.0 if getattr(layer, "linear_symmetric", True) else float(getattr(layer, "linear_span_b", 1.0))
        grad_key = (
            h, w, crop_rect, full_image_shape,
            float(layer.linear_p0[0]), float(layer.linear_p0[1]),
            float(layer.linear_p1[0]), float(layer.linear_p1[1]),
            float(layer.linear_midpoint),
            str(layer.linear_profile),
            float(span_b),
            bool(getattr(layer, "linear_symmetric", True))
        )
        if getattr(layer, "_cached_grad_key", None) == grad_key and getattr(layer, "_cached_grad_mask", None) is not None:
            m = layer._cached_grad_mask.copy()
        else:
            if HAS_RUST and hasattr(astroloom_rs, "compute_linear_gradient_rs"):
                try:
                    m = astroloom_rs.compute_linear_gradient_rs(
                        h, w,
                        float(layer.linear_p0[0]), float(layer.linear_p0[1]),
                        float(layer.linear_p1[0]), float(layer.linear_p1[1]),
                        float(layer.linear_midpoint),
                        str(layer.linear_profile),
                        float(span_b),
                        crop_rect,
                        full_image_shape
                    )
                except Exception:
                    m = compute_linear_gradient_py(
                        h, w,
                        layer.linear_p0[0], layer.linear_p0[1],
                        layer.linear_p1[0], layer.linear_p1[1],
                        layer.linear_midpoint,
                        layer.linear_profile,
                        span_b,
                        crop_rect,
                        full_image_shape
                    )
            else:
                m = compute_linear_gradient_py(
                    h, w,
                    layer.linear_p0[0], layer.linear_p0[1],
                    layer.linear_p1[0], layer.linear_p1[1],
                    layer.linear_midpoint,
                    layer.linear_profile,
                    span_b,
                    crop_rect,
                    full_image_shape
                )
            layer._cached_grad_key = grad_key
            layer._cached_grad_mask = m

    elif layer.mask_type == MASK_TYPE_GRADIENT_RADIAL:
        grad_key = (
            h, w, crop_rect, full_image_shape,
            float(layer.radial_center[0]), float(layer.radial_center[1]),
            float(layer.radial_radius_x), float(layer.radial_radius_y),
            float(layer.radial_angle),
            float(layer.radial_feather)
        )
        if getattr(layer, "_cached_grad_key", None) == grad_key and getattr(layer, "_cached_grad_mask", None) is not None:
            m = layer._cached_grad_mask.copy()
        else:
            if HAS_RUST and hasattr(astroloom_rs, "compute_radial_gradient_rs"):
                try:
                    m = astroloom_rs.compute_radial_gradient_rs(
                        h, w,
                        float(layer.radial_center[0]), float(layer.radial_center[1]),
                        float(layer.radial_radius_x), float(layer.radial_radius_y),
                        float(layer.radial_angle),
                        float(layer.radial_feather),
                        crop_rect,
                        full_image_shape
                    )
                except Exception:
                    m = compute_radial_gradient_py(
                        h, w,
                        layer.radial_center[0], layer.radial_center[1],
                        layer.radial_radius_x, layer.radial_radius_y,
                        layer.radial_angle,
                        layer.radial_feather,
                        crop_rect,
                        full_image_shape
                    )
            else:
                m = compute_radial_gradient_py(
                    h, w,
                    layer.radial_center[0], layer.radial_center[1],
                    layer.radial_radius_x, layer.radial_radius_y,
                    layer.radial_angle,
                    layer.radial_feather,
                    crop_rect,
                    full_image_shape
                )
            layer._cached_grad_key = grad_key
            layer._cached_grad_mask = m

    elif layer.mask_type == MASK_TYPE_CUSTOM and layer.custom_mask is not None:
        m = _slice_or_resize_mask(layer.custom_mask)
    else:
        m = np.ones((h, w), dtype=np.float32)

    # Restricción espacial de máscara (Edge-Snapping al horizonte / Suelo)
    limit_mode = getattr(layer, "limit_mask", LIMIT_MASK_NONE)
    if limit_mode == LIMIT_MASK_SKY and master_sky_mask is not None:
        m_sky = master_sky_mask[..., 0] if master_sky_mask.ndim == 3 else master_sky_mask
        m_sky = _slice_or_resize_mask(m_sky)
        m = m * np.clip(m_sky, 0.0, 1.0)
    elif limit_mode == LIMIT_MASK_GROUND and master_sky_mask is not None:
        m_sky = master_sky_mask[..., 0] if master_sky_mask.ndim == 3 else master_sky_mask
        m_sky = _slice_or_resize_mask(m_sky)
        m = m * np.clip(1.0 - m_sky, 0.0, 1.0)

    # Inversión de máscara
    if layer.inverted:
        m = 1.0 - m

    # Refinamiento de bordes por Filtro Cruzado Guiado (guided_cross de LightCraft)
    if getattr(layer, "refine_edges", False) and getattr(layer, "refine_strength", 0.0) > 1e-4:
        m = refine_mask_guided_cross(
            m, base_rgb,
            strength=layer.refine_strength,
            radius=getattr(layer, "refine_radius", None)
        )

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
    lut_arr = np.maximum.accumulate(lut_arr)  # Monotonía garantizada
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
               abs(layer.saturation - 1.0) > 1e-3 or not layer.is_curve_identity() or
               getattr(layer, 'sharp_amount', 0.0) > 1e-4)
    if not has_adj:
        return image_rgb

    # Ponderación global de la máscara por opacidad
    # Comprobación ligera sin asignar buffers temporales
    if float(layer.opacity) <= 1e-4 or float(np.max(effective_mask)) * float(layer.opacity) <= 1e-4:
        return image_rgb

    if HAS_RUST:
        if image_rgb.dtype == np.float32 and image_rgb.flags.c_contiguous:
            orig_c = image_rgb
        else:
            orig_c = np.ascontiguousarray(image_rgb, dtype=np.float32)
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
            lut_c = np.ascontiguousarray(np.clip(lut, 0.0, 1.0), dtype=np.float32)
        else:
            lut_c = None

        mask_2d = effective_mask[..., 0] if effective_mask.ndim == 3 else effective_mask
        mask_c = np.ascontiguousarray(mask_2d, dtype=np.float32)

        res = astroloom_rs.apply_layer_adjustments_fused(
            orig_c,
            mask_c,
            lut_c,
            opacity=float(layer.opacity),
            exposure=float(layer.exposure),
            contrast=0.0,
            highlights=float(layer.highlights),
            shadows=float(layer.shadows),
            saturation=float(layer.saturation),
            vibrance=0.0,
            temp=float(layer.temp),
            tint=float(layer.tint)
        )
        if getattr(layer, 'sharp_amount', 0.0) > 1e-3:
            from core.sharpening import compute_edge_mask, apply_unsharp_mask
            edge_m = compute_edge_mask(res, masking=float(layer.sharp_masking), scope="global")
            sharp_res = apply_unsharp_mask(
                res,
                amount=float(layer.sharp_amount),
                radius=float(layer.sharp_radius),
                detail=float(layer.sharp_detail),
                edge_mask=edge_m
            )
            w_3d = (np.clip(mask_2d * layer.opacity, 0.0, 1.0))[..., np.newaxis]
            res = np.clip((sharp_res * w_3d) + (res * (1.0 - w_3d)), 0.0, 1.0).astype(np.float32)
        return res

    # Fallback vectorizado puro en NumPy/OpenCV con paridad matemática exacta a Rust
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

    res = _numpy_fused_layer_adjust(
        orig_c, adj_source, mask_c, float(layer.opacity),
        float(r_gain), float(g_gain), float(b_gain),
        float(layer.shadows), float(layer.highlights),
        use_lut, lut
    )
    if getattr(layer, 'sharp_amount', 0.0) > 1e-3:
        from core.sharpening import compute_edge_mask, apply_unsharp_mask
        edge_m = compute_edge_mask(res, masking=float(layer.sharp_masking), scope="global")
        sharp_res = apply_unsharp_mask(
            res,
            amount=float(layer.sharp_amount),
            radius=float(layer.sharp_radius),
            detail=float(layer.sharp_detail),
            edge_mask=edge_m
        )
        w_3d = (np.clip(mask_2d * layer.opacity, 0.0, 1.0))[..., np.newaxis]
        res = np.clip((sharp_res * w_3d) + (res * (1.0 - w_3d)), 0.0, 1.0).astype(np.float32)
    return res

    adj = apply_pre_curve_adjustments(image_rgb, layer, fast_draft=fast_draft)

    # 7. Curva tonal interactiva propia de la capa (LUT de 256 niveles acelerada por hardware)
    if not layer.is_curve_identity():
        if layer.curve_lut is None:
            layer.curve_lut = compute_curve_lut_from_points(layer.curve_points)
        u8 = (np.clip(adj, 0.0, 1.0) * 255.0).astype(np.uint8)
        lut_u8 = (np.clip(layer.curve_lut, 0.0, 1.0) * 255.0).astype(np.uint8)
        adj = (cv2.LUT(u8, lut_u8).astype(np.float32) / 255.0)

    # 8. Máscara de Enfoque y Nitidez local de capa si está activa
    if getattr(layer, 'sharp_amount', 0.0) > 1e-3:
        from core.sharpening import compute_edge_mask, apply_unsharp_mask
        edge_m = compute_edge_mask(adj, masking=float(layer.sharp_masking), scope="global")
        adj = apply_unsharp_mask(
            adj,
            amount=float(layer.sharp_amount),
            radius=float(layer.sharp_radius),
            detail=float(layer.sharp_detail),
            edge_mask=edge_m
        )

    adj = np.clip(adj, 0.0, 1.0)

    # Fusión ponderada por canal
    weight = np.clip(effective_mask * layer.opacity, 0.0, 1.0).astype(np.float32)
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


# =========================================================================
# OPERACIONES DE FUSIÓN DE CAPAS RÁSTER (BLEND MODES)
# =========================================================================

def apply_normal_blend(canvas: np.ndarray, top: np.ndarray, weight: Union[float, np.ndarray]) -> np.ndarray:
    """Mezcla Normal (Sustitución lineal ponderada por opacidad/máscara)."""
    if isinstance(weight, (int, float)):
        if abs(weight - 1.0) < 1e-5:
            return np.clip(top, 0.0, 1.0).astype(np.float32)
        w = float(weight)
    else:
        w = weight[..., np.newaxis] if weight.ndim == 2 else weight
    return np.clip((1.0 - w) * canvas + w * top, 0.0, 1.0).astype(np.float32)


def apply_screen_blend(canvas: np.ndarray, top: np.ndarray, weight: Union[float, np.ndarray]) -> np.ndarray:
    """Mezcla Trama (Screen: 1 - (1 - A)(1 - B * w)). Ideal para estrellas y protuberancias."""
    if isinstance(weight, (int, float)):
        if HAS_RUST:
            c_c = np.ascontiguousarray(canvas, dtype=np.float32)
            t_c = np.ascontiguousarray(top, dtype=np.float32)
            return np.asarray(astroloom_rs.screen_blend(c_c, t_c, float(weight)), dtype=np.float32)
        w = float(weight)
    else:
        w = weight[..., np.newaxis] if weight.ndim == 2 else weight

    top_w = np.clip(top * w, 0.0, 1.0)
    return np.clip(1.0 - (1.0 - canvas) * (1.0 - top_w), 0.0, 1.0).astype(np.float32)


def apply_luminosity_blend(canvas: np.ndarray, top: np.ndarray, weight: Union[float, np.ndarray]) -> np.ndarray:
    """
    Mezcla de Luminosidad (Photoshop / W3C Compositing):
    Preserva el tono y saturación de la base (canvas) e inyecta la luminancia del filtro (top).
    Aplica el algoritmo estándar ClipColor para preservar el gamut sin viraje cromático.
    """
    if HAS_RUST and hasattr(astroloom_rs, "luminosity_blend_rs") and canvas.ndim == 3 and top.ndim == 3 and canvas.shape == top.shape:
        c_c = np.ascontiguousarray(canvas, dtype=np.float32)
        t_c = np.ascontiguousarray(top, dtype=np.float32)
        if isinstance(weight, (int, float)):
            return np.asarray(astroloom_rs.luminosity_blend_rs(c_c, t_c, float(weight), None), dtype=np.float32)
        elif isinstance(weight, np.ndarray) and weight.ndim == 2 and weight.shape == canvas.shape[:2]:
            m_c = np.ascontiguousarray(weight, dtype=np.float32)
            return np.asarray(astroloom_rs.luminosity_blend_rs(c_c, t_c, 1.0, m_c), dtype=np.float32)

    l_base = 0.2126 * canvas[..., 0] + 0.7152 * canvas[..., 1] + 0.0722 * canvas[..., 2]
    if top.ndim == 2:
        l_top = top
    elif top.ndim == 3 and top.shape[2] == 1:
        l_top = top[..., 0]
    else:
        l_top = 0.2126 * top[..., 0] + 0.7152 * top[..., 1] + 0.0722 * top[..., 2]

    # Diferencia de luminancia
    d = (l_top - l_base)[..., np.newaxis]
    c = canvas + d

    # ClipColor W3C / Photoshop standard
    l = l_top[..., np.newaxis]
    n = np.min(c, axis=-1, keepdims=True)
    x = np.max(c, axis=-1, keepdims=True)

    mask_n = (n < 0.0) & (l != n)
    c = np.where(mask_n, l + ((c - l) * l) / np.maximum(l - n, 1e-6), c)

    mask_x = (x > 1.0) & (x != l)
    c = np.where(mask_x, l + ((c - l) * (1.0 - l)) / np.maximum(x - l, 1e-6), c)

    blended = np.clip(c, 0.0, 1.0)

    if isinstance(weight, (int, float)):
        if abs(weight - 1.0) < 1e-5:
            return blended.astype(np.float32)
        w = float(weight)
    else:
        w = weight[..., np.newaxis] if weight.ndim == 2 else weight

    return np.clip((1.0 - w) * canvas + w * blended, 0.0, 1.0).astype(np.float32)


def apply_soft_light_blend(canvas: np.ndarray, top: np.ndarray, weight: Union[float, np.ndarray]) -> np.ndarray:
    """
    Mezcla de Luz Suave (Soft Light - W3C / Photoshop):
    Realza el microcontraste y gradientes coronales sin quemar luces ni empastar sombras.
    """
    if HAS_RUST and hasattr(astroloom_rs, "soft_light_blend_rs") and canvas.ndim == 3 and top.ndim == 3 and canvas.shape == top.shape:
        c_c = np.ascontiguousarray(canvas, dtype=np.float32)
        t_c = np.ascontiguousarray(top, dtype=np.float32)
        if isinstance(weight, (int, float)):
            return np.asarray(astroloom_rs.soft_light_blend_rs(c_c, t_c, float(weight), None), dtype=np.float32)
        elif isinstance(weight, np.ndarray) and weight.ndim == 2 and weight.shape == canvas.shape[:2]:
            m_c = np.ascontiguousarray(weight, dtype=np.float32)
            return np.asarray(astroloom_rs.soft_light_blend_rs(c_c, t_c, 1.0, m_c), dtype=np.float32)

    cond = (top <= 0.5)
    c_low = canvas - (1.0 - 2.0 * top) * canvas * (1.0 - canvas)
    sqrt_b = np.sqrt(np.maximum(canvas, 0.0))
    c_high = canvas + (2.0 * top - 1.0) * (sqrt_b - canvas)
    blended = np.clip(np.where(cond, c_low, c_high), 0.0, 1.0)

    if isinstance(weight, (int, float)):
        if abs(weight - 1.0) < 1e-5:
            return blended.astype(np.float32)
        w = float(weight)
    else:
        w = weight[..., np.newaxis] if weight.ndim == 2 else weight

    return np.clip((1.0 - w) * canvas + w * blended, 0.0, 1.0).astype(np.float32)


def apply_overlay_blend(canvas: np.ndarray, top: np.ndarray, weight: Union[float, np.ndarray]) -> np.ndarray:
    """
    Mezcla de Superposición (Overlay):
    Combina Multiplicar y Trama según la luminancia de la base para realce de microestructuras.
    """
    cond = (canvas <= 0.5)
    c_low = 2.0 * canvas * top
    c_high = 1.0 - 2.0 * (1.0 - canvas) * (1.0 - top)
    blended = np.clip(np.where(cond, c_low, c_high), 0.0, 1.0)

    if isinstance(weight, (int, float)):
        if abs(weight - 1.0) < 1e-5:
            return blended.astype(np.float32)
        w = float(weight)
    else:
        w = weight[..., np.newaxis] if weight.ndim == 2 else weight

    return np.clip((1.0 - w) * canvas + w * blended, 0.0, 1.0).astype(np.float32)


def apply_multiply_blend(canvas: np.ndarray, top: np.ndarray, weight: Union[float, np.ndarray]) -> np.ndarray:
    """
    Mezcla de Multiplicación (Multiply):
    Atenúa altas luces y aplica máscaras de densidad o gradientes de atenuación.
    """
    blended = canvas * top
    if isinstance(weight, (int, float)):
        if abs(weight - 1.0) < 1e-5:
            return np.clip(blended, 0.0, 1.0).astype(np.float32)
        w = float(weight)
    else:
        w = weight[..., np.newaxis] if weight.ndim == 2 else weight

    return np.clip((1.0 - w) * canvas + w * blended, 0.0, 1.0).astype(np.float32)


def apply_blend_mode(
    canvas: np.ndarray,
    top: np.ndarray,
    blend_mode: str,
    weight: Union[float, np.ndarray] = 1.0
) -> np.ndarray:
    """Despacha la operación de fusión seleccionada sobre el lienzo base."""
    mode = (blend_mode or BLEND_MODE_NORMAL).lower()

    # Aceleración nativa Rayon en Rust (Fase B)
    if HAS_RUST and hasattr(astroloom_rs, "apply_blend_mode_rs") and canvas.ndim == 3 and top.ndim == 3 and canvas.shape == top.shape:
        c_c = np.ascontiguousarray(canvas, dtype=np.float32)
        t_c = np.ascontiguousarray(top, dtype=np.float32)
        if isinstance(weight, (int, float)):
            return np.asarray(astroloom_rs.apply_blend_mode_rs(c_c, t_c, mode, float(weight), None), dtype=np.float32)
        elif isinstance(weight, np.ndarray) and weight.ndim == 2 and weight.shape == canvas.shape[:2]:
            m_c = np.ascontiguousarray(weight, dtype=np.float32)
            return np.asarray(astroloom_rs.apply_blend_mode_rs(c_c, t_c, mode, 1.0, m_c), dtype=np.float32)

    if mode == BLEND_MODE_SCREEN:
        return apply_screen_blend(canvas, top, weight)
    elif mode == BLEND_MODE_LUMINOSITY:
        return apply_luminosity_blend(canvas, top, weight)
    elif mode == BLEND_MODE_SOFT_LIGHT:
        return apply_soft_light_blend(canvas, top, weight)
    elif mode == BLEND_MODE_OVERLAY:
        return apply_overlay_blend(canvas, top, weight)
    elif mode == BLEND_MODE_MULTIPLY:
        return apply_multiply_blend(canvas, top, weight)
    else:
        return apply_normal_blend(canvas, top, weight)


# --- Caché de Composición Incremental (Sub-composite Cache) ---
_SUBCOMPOSITE_CACHE: Dict[Tuple, np.ndarray] = {}
_MAX_SUBCOMPOSITE_CACHE = 64

def clear_subcomposite_cache() -> None:
    """Vacía la caché de composiciones parciales de capas."""
    _SUBCOMPOSITE_CACHE.clear()

def _compute_subcomposite_key(
    composite_image: np.ndarray,
    layers_below: List[MaskLayer],
    master_sky_mask: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    fast_draft: bool = False
) -> Tuple:
    layer_signatures = tuple(
        (
            l.id,
            l.enabled,
            float(l.opacity),
            l.mask_type,
            getattr(l, "blend_mode", "normal"),
            float(l.exposure),
            float(l.contrast),
            float(l.temp),
            float(l.tint),
            float(l.highlights),
            float(l.shadows),
            float(l.saturation),
            float(getattr(l, "sharp_amount", 0.0)),
            float(getattr(l, "sharp_radius", 1.0)),
            float(getattr(l, "sharp_detail", 25.0)),
            float(getattr(l, "sharp_masking", 0.0)),
            tuple(l.curve_points) if l.curve_points else (),
            float(getattr(l, "offset_x", 0.0)),
            float(getattr(l, "offset_y", 0.0)),
            float(getattr(l, "scale_x", 1.0)),
            float(getattr(l, "scale_y", 1.0)),
            float(getattr(l, "angle", 0.0)),
            id(l.image_data),
            id(l.brush_mask),
            getattr(l, "brush_mask_version", 0),
            id(l.custom_mask),
            l.inverted,
            getattr(l, "linear_p0", None),
            getattr(l, "linear_p1", None),
            getattr(l, "linear_span_b", None),
            getattr(l, "linear_midpoint", None),
            getattr(l, "linear_profile", None),
            getattr(l, "linear_symmetric", None),
            getattr(l, "radial_center", None),
            getattr(l, "radial_radius_x", None),
            getattr(l, "radial_radius_y", None),
            getattr(l, "radial_angle", None),
            getattr(l, "radial_feather", None),
        )
        for l in layers_below
    )
    return (
        id(composite_image),
        composite_image.shape,
        id(master_sky_mask) if master_sky_mask is not None else None,
        id(starless_image) if starless_image is not None else None,
        id(stars_image) if stars_image is not None else None,
        crop_rect,
        fast_draft,
        layer_signatures
    )


def apply_all_mask_layers(
    composite_image: np.ndarray,
    mask_layers: List[MaskLayer],
    master_sky_mask: Optional[np.ndarray] = None,
    starless_image: Optional[np.ndarray] = None,
    stars_image: Optional[np.ndarray] = None,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
    full_image_shape: Optional[Tuple[int, int]] = None,
    fast_draft: bool = False,
    abort_flag: Optional[Callable[[], bool]] = None,
) -> Optional[np.ndarray]:
    """
    Aplica secuencialmente la composición de capas (estilo Photoshop),
    integrando capas de imagen (Fondo Starless, Suelo, Estrellas) y capas de ajuste locales.
    Soporta modo Zoom 100% mediante crop_rect y full_image_shape para alineación exacta 1:1.
    """
    if abort_flag and abort_flag():
        return None
    if not mask_layers:
        return composite_image

    has_image_layers = any(l.layer_kind == LAYER_KIND_IMAGE for l in mask_layers)
    if not has_image_layers:
        # Modo clásico de solo máscaras de ajuste sobre la imagen compuesta
        img = composite_image.copy()
        for layer in mask_layers:
            if abort_flag and abort_flag():
                return None
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
    # Aceleración incremental: reutilizar prefijo ya compuesto si está en caché
    start_cut = len(mask_layers)
    canvas = None
    if len(mask_layers) > 1 and not crop_rect:
        for cut in range(1, len(mask_layers)):
            if abort_flag and abort_flag():
                return None
            ck = _compute_subcomposite_key(
                composite_image, mask_layers[cut:], master_sky_mask, starless_image, stars_image, crop_rect, fast_draft
            )
            if ck in _SUBCOMPOSITE_CACHE:
                canvas = _SUBCOMPOSITE_CACHE[ck].copy()
                start_cut = cut
                break
    if canvas is None:
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

    # Iterar de fondo a cima desde start_cut - 1 hasta la cima (índice 0)
    for i in range(start_cut - 1, -1, -1):
        if abort_flag and abort_flag():
            return None
        # Poblar la caché de subcomposición para consultas O(1) de capas superiores
        if i + 1 < len(mask_layers) and not crop_rect:
            ck = _compute_subcomposite_key(
                composite_image, mask_layers[i + 1:], master_sky_mask, starless_image, stars_image, crop_rect, fast_draft
            )
            if ck not in _SUBCOMPOSITE_CACHE:
                if len(_SUBCOMPOSITE_CACHE) >= _MAX_SUBCOMPOSITE_CACHE:
                    _SUBCOMPOSITE_CACHE.pop(next(iter(_SUBCOMPOSITE_CACHE)))
                _SUBCOMPOSITE_CACHE[ck] = canvas.copy()

        layer = mask_layers[i]
        if not layer.enabled:
            continue

        if layer.mask_type in (MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_BASE):
            # Capa 🖼️ Fondo Base o 🌌 Fondo Starless
            has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
                       abs(layer.exposure) > 1e-3 or abs(layer.contrast) > 1e-3 or 
                       abs(layer.saturation - 1.0) > 1e-3 or abs(layer.shadows) > 1e-3 or 
                       abs(layer.highlights) > 1e-3 or layer.has_curve())
            if has_adj:
                eff_starless_m = sky_m if (layer.mask_type == MASK_TYPE_IMAGE_STARLESS and sky_m is not None) else np.ones((img_h, img_w), dtype=np.float32)
                canvas = apply_local_adjustments_to_image(canvas, eff_starless_m, layer, fast_draft=fast_draft)
            if layer.opacity < 0.999:
                if layer.mask_type == MASK_TYPE_IMAGE_STARLESS and sky_m is not None:
                    w = (sky_m * (1.0 - layer.opacity))[..., np.newaxis]
                    canvas = canvas * (1.0 - w)
                else:
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
            # Capa ⭐ Estrellas (ajustes locales preservan balance B-V espectral sin mutar layer)
            if stars_image is not None and layer.opacity > 1e-4:
                if stars_image.shape[:2] != (img_h, img_w):
                    if crop_rect is not None and full_image_shape is not None:
                        x1, y1, x2, y2 = crop_rect
                        h_f, w_f = full_image_shape
                        h_s, w_s = stars_image.shape[:2]
                        sx = w_s / float(w_f)
                        sy = h_s / float(h_f)
                        sx1 = max(0, min(w_s, int(round(x1 * sx))))
                        sy1 = max(0, min(h_s, int(round(y1 * sy))))
                        sx2 = max(0, min(w_s, int(round(x2 * sx))))
                        sy2 = max(0, min(h_s, int(round(y2 * sy))))
                        st_sub = stars_image[sy1:sy2, sx1:sx2]
                        if st_sub.shape[:2] != (img_h, img_w):
                            st = cv2.resize(st_sub, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                        else:
                            st = st_sub
                    else:
                        st = cv2.resize(stars_image, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                else:
                    st = stars_image
                
                # Ajustes cromáticos propios de la capa de estrellas (Temp y Tinte ignorados para preservar física espectral B-V)
                has_adj = (abs(layer.exposure) > 1e-3 or abs(layer.contrast) > 1e-3 or 
                           abs(layer.saturation - 1.0) > 1e-3 or layer.has_curve())
                if has_adj:
                    st_layer = layer.shallow_snapshot()
                    st_layer.temp = 0.0
                    st_layer.tint = 0.0
                    st = apply_local_adjustments_to_image(st, np.ones((img_h, img_w), dtype=np.float32), st_layer, fast_draft=fast_draft)

                # Mezcla Screen (Trama estilo Photoshop): 1 - (1 - canvas) * (1 - st * opacity)
                # Garantiza que las estrellas no quemen núcleos y se fundan de forma natural
                if HAS_RUST:
                    st_c = np.ascontiguousarray(st, dtype=np.float32)
                    canvas = astroloom_rs.screen_blend(canvas, st_c, float(layer.opacity))
                else:
                    st_weighted = np.clip(st * float(layer.opacity), 0.0, 1.0)
                    canvas = 1.0 - (1.0 - canvas) * (1.0 - st_weighted)

        elif layer.mask_type == MASK_TYPE_IMAGE_RASTER:
            # Capa 🎨 Imagen Ráster Genérica (Filtros Eclipses, Composición Externa)
            if layer.image_data is not None and layer.opacity > 1e-4:
                flt_img = layer.image_data
                # shortcut: image_data is sanitized on ingest in MaskLayer.__post_init__, avoid 28ms NaN scan per frame
                if flt_img.dtype != np.float32 or flt_img.ndim != 3 or flt_img.shape[2] != 3:
                    flt_img = MaskLayer._sanitize_image_data(flt_img)
                    layer.image_data = flt_img
                    layer._generate_proxies()

                # Selección del mipmap óptimo (Fase 2: Mipmaps multi-escala por capa)
                is_zoomed = (crop_rect is not None and full_image_shape is not None)
                if not is_zoomed:
                    if fast_draft and getattr(layer, 'proxy_draft', None) is not None:
                        flt_img = layer.proxy_draft
                    elif getattr(layer, 'proxy_preview', None) is not None and (img_w <= 1600 and img_h <= 1600):
                        flt_img = layer.proxy_preview
                    else:
                        flt_img = layer.image_data
                else:
                    flt_img = layer.image_data

                h_r, w_r = flt_img.shape[:2]
                has_transform = not getattr(layer, 'is_transform_identity', lambda: True)()
                is_zoomed = (crop_rect is not None and full_image_shape is not None)

                if is_zoomed:
                    x1, y1, x2, y2 = crop_rect
                    h_f, w_f = float(full_image_shape[0]), float(full_image_shape[1])
                else:
                    x1, y1 = 0.0, 0.0
                    h_f, w_f = float(img_h), float(img_w)

                valid_alpha = None
                if has_transform or (is_zoomed and (abs(w_r - w_f) > 1 or abs(h_r - h_f) > 1)):
                    scale_x = max(0.01, float(getattr(layer, 'scale_x', getattr(layer, 'scale', 1.0))))
                    scale_y = max(0.01, float(getattr(layer, 'scale_y', getattr(layer, 'scale', 1.0))))
                    angle_val = float(getattr(layer, 'angle', 0.0))
                    theta = np.deg2rad(angle_val)
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    a = scale_x * cos_t
                    b = -scale_y * sin_t
                    c = scale_x * sin_t
                    d = scale_y * cos_t

                    cx = w_f / 2.0
                    cy = h_f / 2.0
                    tx = float(getattr(layer, 'offset_x', 0.0)) * w_f
                    ty = float(getattr(layer, 'offset_y', 0.0)) * h_f

                    M = np.array([
                        [a * (w_f / float(w_r)), b * (h_f / float(h_r)), cx - a * cx - b * cy + tx - x1],
                        [c * (w_f / float(w_r)), d * (h_f / float(h_r)), cy - c * cx - d * cy + ty - y1]
                    ], dtype=np.float32)

                    flt_img = cv2.warpAffine(
                        flt_img, M, (img_w, img_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0.0, 0.0, 0.0)
                    )

                    if has_transform:
                        pts = np.array([[0.0, 0.0], [float(w_r), 0.0], [float(w_r), float(h_r)], [0.0, float(h_r)]], dtype=np.float32)
                        pts_dst = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
                        valid_alpha = np.zeros((img_h, img_w), dtype=np.float32)
                        cv2.fillConvexPoly(valid_alpha, np.int32(np.round(pts_dst)), 1.0)
                elif is_zoomed:
                    # Zoom 100% sin transformación espacial: recorte directo 1:1 nativo
                    rx1 = max(0, min(w_r, int(round(x1))))
                    ry1 = max(0, min(h_r, int(round(y1))))
                    rx2 = max(0, min(w_r, int(round(x2))))
                    ry2 = max(0, min(h_r, int(round(y2))))
                    sub_crop = flt_img[ry1:ry2, rx1:rx2]
                    if sub_crop.shape[:2] != (img_h, img_w):
                        flt_img = cv2.resize(sub_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        flt_img = sub_crop.copy()
                else:
                    # Vista previa / ajuste sin transformación
                    if flt_img.shape[:2] != (img_h, img_w):
                        cache_key = (id(layer.image_data), img_h, img_w)
                        cached = getattr(layer, "_cached_resized", None)
                        if cached is not None and cached[0] == cache_key:
                            flt_img = cached[1]
                        else:
                            flt_img = cv2.resize(flt_img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                            layer._cached_resized = (cache_key, flt_img)

                # Ajustes locales tonales de la capa ráster si existen
                has_adj = (abs(layer.temp) > 1e-4 or abs(layer.tint) > 1e-4 or 
                           abs(layer.exposure) > 1e-3 or abs(layer.contrast) > 1e-3 or 
                           abs(layer.saturation - 1.0) > 1e-3 or abs(layer.shadows) > 1e-3 or 
                           abs(layer.highlights) > 1e-3 or layer.has_curve())
                if has_adj:
                    if HAS_RUST and hasattr(astroloom_rs, "apply_layer_adjustments_fused"):
                        orig_c = np.ascontiguousarray(np.clip(flt_img, 0.0, 1.0), dtype=np.float32)
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
                            lut_c = np.ascontiguousarray(np.clip(lut, 0.0, 1.0), dtype=np.float32)
                        else:
                            lut_c = None

                        flt_img = np.asarray(
                            astroloom_rs.apply_layer_adjustments_fused(
                                orig_c,
                                None,
                                lut_c,
                                opacity=1.0,
                                exposure=float(layer.exposure),
                                contrast=0.0,
                                highlights=float(layer.highlights),
                                shadows=float(layer.shadows),
                                saturation=float(layer.saturation),
                                vibrance=0.0,
                                temp=float(layer.temp),
                                tint=float(layer.tint)
                            ),
                            dtype=np.float32
                        )
                    else:
                        flt_img = apply_pre_curve_adjustments(flt_img, layer, fast_draft=fast_draft)
                        if not layer.is_curve_identity():
                            if layer.curve_lut is None:
                                layer.curve_lut = compute_curve_lut_from_points(layer.curve_points)
                            u8 = (np.clip(flt_img, 0.0, 1.0) * 255.0).astype(np.uint8)
                            lut_u8 = (np.clip(layer.curve_lut, 0.0, 1.0) * 255.0).astype(np.uint8)
                            flt_img = (cv2.LUT(u8, lut_u8).astype(np.float32) / 255.0)

                # Máscara de Enfoque y Nitidez de la capa ráster si está activa
                if getattr(layer, 'sharp_amount', 0.0) > 1e-3:
                    from core.sharpening import compute_edge_mask, apply_unsharp_mask
                    edge_m = compute_edge_mask(flt_img, masking=float(layer.sharp_masking), scope="global")
                    flt_img = apply_unsharp_mask(
                        flt_img,
                        amount=float(layer.sharp_amount),
                        radius=float(layer.sharp_radius),
                        detail=float(layer.sharp_detail),
                        edge_mask=edge_m
                    )

                # Ponderación por máscara si está presente o invertida
                has_mask = (layer.brush_mask is not None or layer.inverted or layer.custom_mask is not None)
                if has_mask:
                    m_eff = compute_effective_layer_mask(
                        layer, canvas,
                        master_sky_mask=master_sky_mask,
                        starless_image=starless_image,
                        stars_image=stars_image,
                        crop_rect=crop_rect,
                        full_image_shape=full_image_shape
                    )
                    # shortcut: brush_mask is authored in canvas coordinate space, so valid_alpha clips boundaries
                    if valid_alpha is not None:
                        m_eff = m_eff * valid_alpha
                    weight = np.clip(m_eff * layer.opacity, 0.0, 1.0).astype(np.float32)
                else:
                    if valid_alpha is not None:
                        weight = np.clip(valid_alpha * layer.opacity, 0.0, 1.0).astype(np.float32)
                    else:
                        weight = float(layer.opacity)

                # Fusión con el lienzo subyacente según blend_mode
                canvas = apply_blend_mode(canvas, flt_img, getattr(layer, "blend_mode", BLEND_MODE_NORMAL), weight)

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

    # Consulta O(1) en caché de subcomposición incremental (<0.05 ms)
    ck = _compute_subcomposite_key(
        composite_image, layers_below, master_sky_mask, starless_image, stars_image, crop_rect
    )
    cached = _SUBCOMPOSITE_CACHE.get(ck)
    if cached is not None:
        return cached

    res = apply_all_mask_layers(
        composite_image,
        layers_below,
        master_sky_mask=master_sky_mask,
        starless_image=starless_image,
        stars_image=stars_image,
        crop_rect=crop_rect,
        full_image_shape=full_image_shape
    )
    if len(_SUBCOMPOSITE_CACHE) >= _MAX_SUBCOMPOSITE_CACHE:
        _SUBCOMPOSITE_CACHE.pop(next(iter(_SUBCOMPOSITE_CACHE)))
    _SUBCOMPOSITE_CACHE[ck] = res
    return res


