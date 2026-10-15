import os
import json

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

cpu_tot = os.cpu_count() or 4
def_workers = max(2, cpu_tot - 2) if cpu_tot > 2 else cpu_tot

DEFAULT_SUBDIRS = {
    "sessions_dir": "sesiones",
    "masks_dir": os.path.join("sesiones", "mascaras"),
    "stacked_dir": "imagenes",
    "export_dir": "exportaciones",
    "presets_dir": "presets",
}

DEFAULT_CONFIG = {
    # Rutas por defecto del espacio de trabajo
    "sessions_dir": os.path.join(APP_ROOT, DEFAULT_SUBDIRS["sessions_dir"]),
    "masks_dir": os.path.join(APP_ROOT, DEFAULT_SUBDIRS["masks_dir"]),
    "stacked_dir": os.path.join(APP_ROOT, DEFAULT_SUBDIRS["stacked_dir"]),
    "export_dir": os.path.join(APP_ROOT, DEFAULT_SUBDIRS["export_dir"]),
    "presets_dir": os.path.join(APP_ROOT, DEFAULT_SUBDIRS["presets_dir"]),

    # Parámetros del motor de apilado y hardware
    "kappa": 2.2,
    "default_kappa": 2.2,
    "storage_strategy": "auto",
    "use_gpu": True,
    "cpu_workers": def_workers,
    "lp_method": "standard",
    "lp_strength": 0.5,
    "starnet_exe": "",
    "icc_profile": "srgb",
    "graxpert_smoothing": 0.5,
    "preview_max_dim": 1500,
    "export_format_default": "TIFF 16-bit (*.tif *.tiff)",
    "language": "es",
}

CONFIG_FILE = os.path.join(APP_ROOT, "config.json")
DEFAULT_TEMPLATE_FILE = os.path.join(APP_ROOT, "config.default.json")


def _validate_and_resolve_dir(path: str | None, key: str) -> str:
    """
    Valida y resuelve una ruta de directorio del espacio de trabajo de forma tolerante a fallos.
    Si la ruta es relativa, la resuelve respecto a APP_ROOT.
    Si es absoluta pero inaccesible (ej. unidad de disco inexistente, error de permisos),
    aplica fallback automático a APP_ROOT/<subdirectorio> o al directorio de usuario.
    Garantiza que la carpeta exista físicamente en disco y nunca arroja excepciones fatales.
    """
    default_sub = DEFAULT_SUBDIRS.get(key, "sesiones")
    fallback_app_root = os.path.abspath(os.path.join(APP_ROOT, default_sub))
    fallback_home = os.path.abspath(os.path.join(os.path.expanduser("~"), "AstroLoom", default_sub))

    # 1. Si está vacía o no es cadena
    if not path or not isinstance(path, str) or not path.strip():
        resolved = fallback_app_root
    else:
        norm = os.path.normpath(path.strip())
        if not os.path.isabs(norm):
            resolved = os.path.abspath(os.path.join(APP_ROOT, norm))
        else:
            resolved = norm

    # 2. Comprobar existencia de unidad en Windows (ej. D:\ en equipo que solo tiene C:\)
    drive, _ = os.path.splitdrive(resolved)
    drive_accessible = True
    if drive:
        drive_root = drive + ("\\" if not drive.endswith("\\") else "")
        if not os.path.exists(drive_root):
            drive_accessible = False

    # 3. Intentar crear/verificar la ruta seleccionada si la unidad es accesible
    if drive_accessible:
        try:
            os.makedirs(resolved, exist_ok=True)
            if os.path.isdir(resolved):
                return resolved
        except (OSError, PermissionError, ValueError) as e:
            print(f"[ConfigManager] Aviso: No se pudo acceder a '{resolved}': {e}. Aplicando recuperación.")

    # 4. Primer Fallback: Carpeta dentro de APP_ROOT
    try:
        os.makedirs(fallback_app_root, exist_ok=True)
        if os.path.isdir(fallback_app_root):
            return fallback_app_root
    except (OSError, PermissionError, ValueError) as e:
        print(f"[ConfigManager] Aviso: Fallback APP_ROOT '{fallback_app_root}' no disponible: {e}")

    # 5. Segundo Fallback: Carpeta en el directorio del usuario (~/AstroLoom/...)
    try:
        os.makedirs(fallback_home, exist_ok=True)
        return fallback_home
    except Exception:
        pass

    # 6. Fallback final infalible: APP_ROOT raíz
    return os.path.abspath(APP_ROOT)


def ensure_workspace_directories(config: dict) -> bool:
    """
    Crea y valida todos los directorios de trabajo necesarios.
    Devuelve True si alguna ruta tuvo que ser corregida/sanitizada.
    """
    modified = False
    for key in ["sessions_dir", "masks_dir", "stacked_dir", "export_dir", "presets_dir"]:
        raw_val = config.get(key)
        resolved = _validate_and_resolve_dir(raw_val, key)
        if raw_val != resolved:
            config[key] = resolved
            modified = True
    return modified


def load_config() -> dict:
    """
    Carga la configuración de la aplicación de forma tolerante a fallos:
    1. Si config.json no existe, intenta leer config.default.json o usa DEFAULT_CONFIG.
    2. Si config.json existe pero contiene rutas inválidas o de otra máquina,
       las repara automáticamente con fallback seguro y persiste la corrección.
    """
    config = DEFAULT_CONFIG.copy()
    config_loaded_from_disk = False

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    config.update(saved)
                    config_loaded_from_disk = True
        except Exception as e:
            print(f"[ConfigManager] Error leyendo config.json: {e}. Se usarán valores por defecto.")
    elif os.path.exists(DEFAULT_TEMPLATE_FILE):
        try:
            with open(DEFAULT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    config.update(saved)
        except Exception as e:
            print(f"[ConfigManager] Error leyendo config.default.json: {e}")

    # Validar y crear directorios de forma segura (sin excepciones no controladas)
    healed = ensure_workspace_directories(config)

    # Si no existía config.json o hubo rutas corruptas/reparadas, persistir para estabilizar
    if not config_loaded_from_disk or healed:
        save_config(config)

    return config


def save_config(config: dict) -> None:
    ensure_workspace_directories(config)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ConfigManager] Error guardando config.json: {e}")


def get_config_val(key: str, default=None):
    """Retorna un valor de la configuración cargada o el valor por defecto si no existe."""
    cfg = load_config()
    return cfg.get(key, default)


def set_config_val(key: str, value) -> None:
    """Actualiza y persiste un valor individual en config.json."""
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)       
 