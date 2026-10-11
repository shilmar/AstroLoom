import os
import json

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CONFIG = {
    # Rutas por defecto del espacio de trabajo
    "sessions_dir": os.path.join(APP_ROOT, "sesiones"),
    "masks_dir": os.path.join(APP_ROOT, "sesiones", "mascaras"),
    "stacked_dir": os.path.join(APP_ROOT, "imagenes"),
    "export_dir": os.path.join(APP_ROOT, "exportaciones"),

    # Parámetros del motor de apilado y hardware
    "kappa": 2.2,
    "storage_strategy": "auto",
    "use_gpu": True,
    "cpu_workers": 4,
    "lp_method": "standard",
    "lp_strength": 0.5,
    "starnet_exe": "",
    "icc_profile": "srgb",
}

CONFIG_FILE = os.path.join(APP_ROOT, "config.json")


def ensure_workspace_directories(config: dict) -> None:
    """Crea los directorios de trabajo si no existen físicamente en disco."""
    for key in ["sessions_dir", "masks_dir", "stacked_dir", "export_dir"]:
        path = config.get(key)
        if path:
            os.makedirs(path, exist_ok=True)


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except Exception:
            pass

    ensure_workspace_directories(config)
    return config


def save_config(config: dict) -> None:
    ensure_workspace_directories(config)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error guardando config.json: {e}")

def get_config_val(key: str, default=None):
    """Retorna un valor de la configuración cargada o el valor por defecto si no existe."""
    cfg = load_config()
    return cfg.get(key, default)


def set_config_val(key: str, value) -> None:
    """Actualiza y persiste un valor individual en config.json."""
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)       
 