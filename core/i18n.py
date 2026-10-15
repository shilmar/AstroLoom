# core/i18n.py
"""
core/i18n.py - Motor centralizado de internacionalización (i18n) para AstroLoom.

Gestiona la carga de diccionarios de idiomas en formato JSON desde la carpeta locales/,
soporta fallback automático al idioma base (español), interpolación de variables
y emite señales de PySide6 al alternar entre idiomas en tiempo de ejecución.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
from PySide6.QtCore import QObject, Signal

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCALES_DIR = os.path.join(APP_ROOT, "locales")
DEFAULT_LANGUAGE = "es"


class I18nManager(QObject):
    """
    Gestor singleton de internacionalización.
    Carga y provee cadenas traducidas según el idioma activo.
    """
    language_changed = Signal(str)  # Emite el nuevo código de idioma ('es', 'en', etc.)

    def __init__(self, default_lang: str = DEFAULT_LANGUAGE, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._locales_dir = LOCALES_DIR
        self._current_lang = default_lang
        self._translations: Dict[str, Any] = {}
        self._fallback_translations: Dict[str, Any] = {}

        # 1. Cargar siempre el idioma base para fallback
        self._load_fallback()

        # 2. Cargar idioma inicial desde config.json si existe
        initial_lang = self._get_initial_language(default_lang)
        self.load_language(initial_lang)

    def _get_initial_language(self, fallback: str) -> str:
        """Determina el idioma de inicio a partir de config.json o usa el valor por defecto."""
        try:
            from core.config_manager import get_config_val
            lang = get_config_val("language", fallback)
            return lang if isinstance(lang, str) and lang else fallback
        except Exception:
            return fallback

    def _load_fallback(self) -> None:
        """Carga el diccionario en español como red de seguridad permanente."""
        path = os.path.join(self._locales_dir, f"{DEFAULT_LANGUAGE}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._fallback_translations = json.load(f)
            except Exception as e:
                print(f"[i18n] Error cargando fallback '{DEFAULT_LANGUAGE}': {e}")
                self._fallback_translations = {}

    def get_available_languages(self) -> List[Tuple[str, str]]:
        """
        Retorna la lista de idiomas disponibles escaneando la carpeta locales/.
        Devuelve tuplas de (código, nombre_para_mostrar), ej: [('es', 'Español'), ('en', 'English')].
        """
        available: List[Tuple[str, str]] = []
        if not os.path.isdir(self._locales_dir):
            return [("es", "Español"), ("en", "English")]

        for fname in os.listdir(self._locales_dir):
            if fname.endswith(".json"):
                lang_code = os.path.splitext(fname)[0]
                full_path = os.path.join(self._locales_dir, fname)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        meta = data.get("_meta", {})
                        lang_name = meta.get("language_name", lang_code.upper())
                        available.append((lang_code, lang_name))
                except Exception:
                    available.append((lang_code, lang_code.upper()))

        # Ordenar asegurando que español e inglés queden primero
        def sort_key(item):
            code = item[0]
            if code == "es":
                return (0, code)
            elif code == "en":
                return (1, code)
            return (2, code)

        available.sort(key=sort_key)
        return available

    def get_current_language(self) -> str:
        """Retorna el código del idioma activo (ej: 'es', 'en')."""
        return self._current_lang

    def load_language(self, lang_code: str) -> bool:
        """
        Carga el diccionario de un idioma determinado.
        Devuelve True si se cargó correctamente, False si falló y recurrió al fallback.
        """
        path = os.path.join(self._locales_dir, f"{lang_code}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._translations = json.load(f)
                    self._current_lang = lang_code
                    return True
            except Exception as e:
                print(f"[i18n] Error leyendo '{path}': {e}")

        # Si falló, usar fallback
        self._translations = self._fallback_translations.copy()
        self._current_lang = DEFAULT_LANGUAGE
        return False

    def set_language(self, lang_code: str, save_to_config: bool = True) -> bool:
        """
        Cambia el idioma activo de la aplicación, recarga el diccionario,
        persiste la preferencia en config.json y notifica mediante la señal language_changed.
        """
        if lang_code == self._current_lang and self._translations:
            return True

        success = self.load_language(lang_code)
        if success and save_to_config:
            try:
                from core.config_manager import set_config_val
                set_config_val("language", lang_code)
            except Exception as e:
                print(f"[i18n] No se pudo persistir el idioma en config: {e}")

        self.language_changed.emit(self._current_lang)
        return success

    def _resolve_nested_key(self, dictionary: Dict[str, Any], key: str) -> Optional[Any]:
        """Recorre claves anidadas separadas por punto (ej: 'developer.sections.starnet')."""
        parts = key.split(".")
        curr = dictionary
        for p in parts:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                return None
        return curr

    def tr(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        Traduce una clave al idioma activo.
        
        Parámetros:
            key: Identificador de la cadena, ej: 'developer.btn_export' o 'app.ready'.
            default: Valor a retornar si la clave no se encuentra en el diccionario actual ni en el fallback.
            **kwargs: Variables a interpolar en el texto (ej: nombre='Archivo.tif').
            
        Retorna:
            La cadena traducida o el valor por defecto.
        """
        val = self._resolve_nested_key(self._translations, key)
        if val is None or not isinstance(val, str):
            # Intentar en diccionario de respaldo (español)
            val = self._resolve_nested_key(self._fallback_translations, key)

        if val is None or not isinstance(val, str):
            val = default if default is not None else key

        if kwargs:
            try:
                return val.format(**kwargs)
            except Exception:
                return val

        return val


# Instancia singleton accesible globalmente
_instance: Optional[I18nManager] = None


def get_i18n() -> I18nManager:
    """Devuelve la instancia global de I18nManager."""
    global _instance
    if _instance is None:
        _instance = I18nManager()
    return _instance


def tr(key: str, default: Optional[str] = None, **kwargs) -> str:
    """Función de acceso directo global para traducir una cadena."""
    return get_i18n().tr(key, default, **kwargs)
