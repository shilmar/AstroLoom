# core/developer_presets.py
"""
core/developer_presets.py - Modelo y Gestor de Ajustes Preestablecidos (Presets) de Revelado.

Proporciona la arquitectura de datos, presets astronómicos de fábrica (Built-in),
persistencia de presets de usuario en formato JSON y aplicación selectiva (granular)
por ámbito (Todo, Solo Tono, Solo Color, Solo Estrellas/Detalles).
"""
import os
import json
import uuid
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from core.config_manager import load_config, APP_ROOT


# =============================================================================
# CONSTANTES DE ÁMBITO (SCOPES) DE APLICACIÓN GRANULAR
# =============================================================================
SCOPE_ALL = "all"
SCOPE_TONE = "tone"
SCOPE_COLOR = "color"
SCOPE_DETAILS = "details"

# Claves de estirado tonal que son únicas para cada imagen y NO deben ser alteradas por los presets
KEYS_EXCLUDED_FROM_PRESETS = {"bp_val", "mtf_val"}

PARAM_KEYS_TONE = {
    "contrast_val", "gnd_ev_val", "gnd_sh_val", "gnd_bp_val", "gnd_antihalo_val", "gnd_edge_shift_val", "curves"
}

PARAM_KEYS_COLOR = {
    "temp_val", "tint_val", "sat_sky_val", "sat_gnd_val", "vibrance_val"
}

PARAM_KEYS_DETAILS = {
    "stars_val", "contrast_starless_val", "clarity_val", "dehaze_val",
    "ond_est_val", "ond_bg_val", "denoise_val", "layer_idx", "lp_val"
}


@dataclass
class DeveloperPreset:
    """Representa un ajuste preestablecido de revelado astronómico."""
    id: str
    name: str
    description: str = ""
    category: str = "General"
    is_builtin: bool = False
    author: str = "AstroLoom"
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el preset a un diccionario compatible con JSON."""
        return {
            "version": "1.0",
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "is_builtin": self.is_builtin,
            "author": self.author,
            "created_at": self.created_at,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeveloperPreset":
        """Deserializa un preset desde un diccionario JSON."""
        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            name=str(data.get("name", "Preset sin nombre")),
            description=str(data.get("description", "")),
            category=str(data.get("category", "Usuario")),
            is_builtin=bool(data.get("is_builtin", False)),
            author=str(data.get("author", "Usuario")),
            created_at=str(data.get("created_at", datetime.datetime.now().isoformat())),
            params=dict(data.get("params", {})),
        )


# =============================================================================
# PRESETS DE FÁBRICA INTEGRADOS (BUILT-IN PRESETS)
# =============================================================================
def get_builtin_presets() -> List[DeveloperPreset]:
    """Genera la colección oficial de presets afinados para astrofotografía."""
    return [
        DeveloperPreset(
            id="builtin_neutral",
            name="Neutro / Restablecer",
            category="Básicos",
            description="Restablece todos los parámetros de color, contraste y StarNet a sus valores neutros (sin alterar el estirado MTF de la imagen).",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 0,
                "layer_idx": 0,
                "stars_val": 100,
                "contrast_starless_val": 0,
                "clarity_val": 0,
                "dehaze_val": 0,
                "ond_est_val": 0,
                "ond_bg_val": 0,
                "denoise_val": 0,
                "temp_val": 0,
                "tint_val": 0,
                "sat_sky_val": 100,
                "sat_gnd_val": 100,
                "vibrance_val": 0,
                "contrast_val": 0,
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "curves": None,
            }
        ),
        DeveloperPreset(
            id="builtin_mw_high_contrast",
            name="Vía Láctea - Alto Contraste & Polvo Oscuro",
            category="Vía Láctea",
            description="Realce agresivo de bandas oscuras y carril de polvo en el centro galáctico con microtextura de ondículas y contraste refinado.",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 10,
                "layer_idx": 0,
                "stars_val": 80,
                "contrast_starless_val": 25,
                "clarity_val": 25,
                "dehaze_val": 15,
                "ond_est_val": 20,
                "ond_bg_val": 10,
                "denoise_val": 5,
                "temp_val": -30,
                "tint_val": 25,
                "sat_sky_val": 130,
                "sat_gnd_val": 105,
                "vibrance_val": 20,
                "contrast_val": 22,
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "curves": None,
            }
        ),
        DeveloperPreset(
            id="builtin_mw_natural",
            name="Vía Láctea - Fiel y Natural",
            category="Vía Láctea",
            description="Estilo equilibrado, fiel a la visión telescópica con transición suave de estrellas y colores no saturados en exceso.",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 15,
                "layer_idx": 0,
                "stars_val": 92,
                "contrast_starless_val": 10,
                "clarity_val": 12,
                "dehaze_val": 8,
                "ond_est_val": 10,
                "ond_bg_val": 5,
                "denoise_val": 5,
                "temp_val": -15,
                "tint_val": 10,
                "sat_sky_val": 115,
                "sat_gnd_val": 100,
                "vibrance_val": 10,
                "contrast_val": 10,
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "curves": None,
            }
        ),
        DeveloperPreset(
            id="builtin_nightscape_foreground",
            name="Paisaje Nocturno - Suelo Iluminado",
            category="Paisaje Nocturno",
            description="Prioriza la visibilidad y rango dinámico del primer plano terrestre levantando sombras (+1.2 EV) sin quemar el cielo estrellado.",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 10,
                "layer_idx": 0,
                "stars_val": 95,
                "contrast_starless_val": 15,
                "clarity_val": 15,
                "dehaze_val": 10,
                "ond_est_val": 12,
                "ond_bg_val": 0,
                "denoise_val": 10,
                "temp_val": 0,
                "tint_val": 0,
                "sat_sky_val": 110,
                "sat_gnd_val": 120,
                "vibrance_val": 15,
                "contrast_val": 12,
                "gnd_ev_val": 120,
                "gnd_sh_val": 35,
                "gnd_bp_val": -8,
                "curves": None,
            }
        ),
        DeveloperPreset(
            id="builtin_dso_nebula",
            name="Cielo Profundo - Nebulosas (Deep Sky Pop)",
            category="Cielo Profundo",
            description="Optimizado para nebulosas difusas y objetos de espacio profundo; reduce estrellas de fondo y destaca estructuras gaseosas débiles.",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 15,
                "layer_idx": 0,
                "stars_val": 55,
                "contrast_starless_val": 30,
                "clarity_val": 30,
                "dehaze_val": 20,
                "ond_est_val": 35,
                "ond_bg_val": 20,
                "denoise_val": 15,
                "temp_val": -20,
                "tint_val": 35,
                "sat_sky_val": 140,
                "sat_gnd_val": 100,
                "vibrance_val": 25,
                "contrast_val": 20,
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "curves": None,
            }
        ),
        DeveloperPreset(
            id="builtin_light_pollution_fix",
            name="Reducción de Polución Lumínica Periurbana",
            category="Corrección",
            description="Atenúa cúpulas de luz amarillentas/anaranjadas de ciudades cercanas, neutraliza la dominante térmica y limpia el fondo.",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 38,
                "layer_idx": 0,
                "stars_val": 90,
                "contrast_starless_val": 15,
                "clarity_val": 15,
                "dehaze_val": 25,
                "ond_est_val": 10,
                "ond_bg_val": 15,
                "denoise_val": 12,
                "temp_val": -120,
                "tint_val": -25,
                "sat_sky_val": 105,
                "sat_gnd_val": 95,
                "vibrance_val": 5,
                "contrast_val": 15,
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "curves": None,
            }
        ),
        DeveloperPreset(
            id="builtin_star_minimizer",
            name="Reducción Drástica de Estrellas (Star Minimizer)",
            category="Estrellas",
            description="Comprime las estrellas a pequeñas cabezas de alfiler discretas (35%), despejando la visión de nebulosas y nubes galácticas.",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 0,
                "layer_idx": 0,
                "stars_val": 35,
                "contrast_starless_val": 20,
                "clarity_val": 15,
                "dehaze_val": 10,
                "ond_est_val": 15,
                "ond_bg_val": 0,
                "denoise_val": 5,
                "temp_val": 0,
                "tint_val": 0,
                "sat_sky_val": 115,
                "sat_gnd_val": 100,
                "vibrance_val": 10,
                "contrast_val": 10,
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "curves": None,
            }
        ),
        DeveloperPreset(
            id="builtin_starless_pure",
            name="Starless Puro (Solo Nebulosa / Sin Estrellas)",
            category="Estrellas",
            description="Elimina totalmente la capa de estrellas (0%), dejando visible únicamente el gas nebuloso y polvo con realce de texturas.",
            is_builtin=True,
            author="AstroLoom",
            params={
                "lp_val": 0,
                "layer_idx": 1,
                "stars_val": 0,
                "contrast_starless_val": 35,
                "clarity_val": 30,
                "dehaze_val": 25,
                "ond_est_val": 40,
                "ond_bg_val": 15,
                "denoise_val": 10,
                "temp_val": 0,
                "tint_val": 15,
                "sat_sky_val": 130,
                "sat_gnd_val": 100,
                "vibrance_val": 20,
                "contrast_val": 20,
                "gnd_ev_val": 0,
                "gnd_sh_val": 0,
                "gnd_bp_val": 0,
                "curves": None,
            }
        ),
    ]


# =============================================================================
# GESTIÓN Y PERSISTENCIA DE PRESETS DE USUARIO
# =============================================================================
def get_user_presets_dir() -> Path:
    """Retorna y asegura la existencia del directorio de presets de usuario."""
    cfg = load_config()
    raw_dir = cfg.get("presets_dir")
    if not raw_dir:
        raw_dir = os.path.join(APP_ROOT, "presets")
    presets_path = Path(raw_dir) / "developer"
    presets_path.mkdir(parents=True, exist_ok=True)
    return presets_path


def load_user_presets(custom_dir: Optional[str] = None) -> List[DeveloperPreset]:
    """Carga todos los presets de usuario guardados en disco."""
    target_dir = Path(custom_dir) if custom_dir else get_user_presets_dir()
    if not target_dir.exists():
        return []

    presets: List[DeveloperPreset] = []
    for f in target_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict):
                    preset = DeveloperPreset.from_dict(data)
                    preset.is_builtin = False
                    presets.append(preset)
        except Exception as e:
            print(f"[DeveloperPresets] Error leyendo preset '{f.name}': {e}")

    # Ordenar alfabéticamente por nombre
    presets.sort(key=lambda p: p.name.lower())
    return presets


def save_user_preset(
    name: str,
    params: Dict[str, Any],
    description: str = "",
    category: str = "Usuario",
    author: str = "Usuario",
    custom_dir: Optional[str] = None,
    preset_id: Optional[str] = None,
) -> DeveloperPreset:
    """Guarda o actualiza un preset de usuario en disco en formato JSON."""
    target_dir = Path(custom_dir) if custom_dir else get_user_presets_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_id = preset_id or f"user_{uuid.uuid4().hex[:8]}"
    # Nombre de archivo seguro
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe_name:
        safe_name = clean_id
    filename = f"{safe_name}.json"
    file_path = target_dir / filename

    # Excluir parámetros de estirado tonal específicos de cada imagen (punto negro y MTF)
    clean_params = {k: v for k, v in params.items() if k not in KEYS_EXCLUDED_FROM_PRESETS}

    preset = DeveloperPreset(
        id=clean_id,
        name=name.strip(),
        description=description.strip(),
        category=category.strip() or "Usuario",
        is_builtin=False,
        author=author.strip() or "Usuario",
        created_at=datetime.datetime.now().isoformat(),
        params=clean_params,
    )

    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(preset.to_dict(), fp, indent=4, ensure_ascii=False)

    return preset


def delete_user_preset(preset_id: str, custom_dir: Optional[str] = None) -> bool:
    """Elimina un preset de usuario buscando por su ID único."""
    target_dir = Path(custom_dir) if custom_dir else get_user_presets_dir()
    if not target_dir.exists():
        return False

    for f in target_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if data.get("id") == preset_id:
                f.unlink()
                return True
        except Exception as e:
            print(f"[DeveloperPresets] Error al eliminar preset '{f.name}': {e}")
            continue
    return False


def export_preset_file(preset: DeveloperPreset, target_path: str) -> None:
    """Exporta un preset individual a cualquier ubicación elegida por el usuario."""
    path = Path(target_path)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(preset.to_dict(), fp, indent=4, ensure_ascii=False)


def import_preset_file(source_path: str, custom_dir: Optional[str] = None) -> DeveloperPreset:
    """Importa un archivo JSON de preset externo al repositorio local de usuario."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de preset no encontrado: {source_path}")

    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, dict):
        raise ValueError("El archivo no contiene un formato de preset válido.")

    # Asegurar nuevo ID único para evitar colisiones
    data["id"] = f"user_{uuid.uuid4().hex[:8]}"
    data["is_builtin"] = False

    preset = DeveloperPreset.from_dict(data)
    save_user_preset(
        name=preset.name,
        params=preset.params,
        description=preset.description,
        category=preset.category,
        author=preset.author,
        custom_dir=custom_dir,
        preset_id=preset.id,
    )
    return preset


# =============================================================================
# FILTRADO DE PARÁMETROS POR ÁMBITO (GRANULARIDAD)
# =============================================================================
def filter_preset_params(
    preset_params: Dict[str, Any],
    current_params: Dict[str, Any],
    scope: str = SCOPE_ALL
) -> Dict[str, Any]:
    """
    Combina los parámetros del preset con los actuales según el ámbito seleccionado:
    - SCOPE_ALL: Aplica todos los parámetros del preset (sin alterar el estirado MTF ni el punto negro de la imagen).
    - SCOPE_TONE: Solo actualiza contraste general, curvas y tono del suelo.
    - SCOPE_COLOR: Solo actualiza balance de blancos, tintes y saturación.
    - SCOPE_DETAILS: Solo actualiza reducción de estrellas, claridad, dehaze y filtros.
    """
    target = current_params.copy()

    if scope == SCOPE_ALL:
        keys = set(preset_params.keys()) - KEYS_EXCLUDED_FROM_PRESETS
    elif scope == SCOPE_TONE:
        keys = PARAM_KEYS_TONE - KEYS_EXCLUDED_FROM_PRESETS
    elif scope == SCOPE_COLOR:
        keys = PARAM_KEYS_COLOR - KEYS_EXCLUDED_FROM_PRESETS
    elif scope == SCOPE_DETAILS:
        keys = PARAM_KEYS_DETAILS - KEYS_EXCLUDED_FROM_PRESETS
    else:
        keys = set(preset_params.keys()) - KEYS_EXCLUDED_FROM_PRESETS

    for k in keys:
        if k in preset_params and k not in KEYS_EXCLUDED_FROM_PRESETS:
            target[k] = preset_params[k]

    return target
