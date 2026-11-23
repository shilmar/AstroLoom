# gui/theme.py
"""
gui/theme.py - Sistema de Diseño y Hoja de Estilos Global Unificada de AstroLoom.

Proporciona la paleta de colores cromática 'Cosmic Dark Pro', estilos reutilizables
para botones primarios y de transferencia entre módulos, y la hoja de estilos QSS
maestra que garantiza una identidad visual unificada en todas las pestañas.
"""
import os
from typing import Optional
from PySide6.QtWidgets import QWidget, QApplication

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_CURRENT_DIR, "assets")
CHEVRON_DOWN_PATH = os.path.join(_ASSETS_DIR, "chevron_down.svg").replace("\\", "/")
CHEVRON_DOWN_HOVER_PATH = os.path.join(_ASSETS_DIR, "chevron_down_hover.svg").replace("\\", "/")
CHEVRON_DOWN_DISABLED_PATH = os.path.join(_ASSETS_DIR, "chevron_down_disabled.svg").replace("\\", "/")
CHEVRON_UP_PATH = os.path.join(_ASSETS_DIR, "chevron_up.svg").replace("\\", "/")
CHEVRON_UP_HOVER_PATH = os.path.join(_ASSETS_DIR, "chevron_up_hover.svg").replace("\\", "/")
CHEVRON_UP_DISABLED_PATH = os.path.join(_ASSETS_DIR, "chevron_up_disabled.svg").replace("\\", "/")
CHECK_WHITE_PATH = os.path.join(_ASSETS_DIR, "check_white.svg").replace("\\", "/")
CHECK_DISABLED_PATH = os.path.join(_ASSETS_DIR, "check_disabled.svg").replace("\\", "/")

# =============================================================================
# 1. PALETA DE COLORES MAESTRA - REGLA 60-30-10 (Neutral Studio & Cobalt Blue)
# =============================================================================
# 60% - Fondos Base y Visor (Gris Neutro / Carbón Profundo libre de matices cromáticos)
BG_CANVAS       = "#0a0a0b"  # Lienzo y visor principal de la imagen
BG_WINDOW       = "#121214"  # Fondo base de la ventana principal y contenedores raíz
BG_CONSOLE      = "#0e0e10"  # Consola y registros monospaciados

# 30% - Paneles, Barras y Estructura
BG_PANEL        = "#1a1a1e"  # Barras laterales, superior y dock panels
BG_CARD         = "#222226"  # Secciones modulares, bloques temáticos ("1. Sesión") y QGroupBox
BG_CARD_HEADER  = "#18181b"  # Cabeceras de secciones y acordeones (relieve DaVinci Resolve)
BG_CARD_ALT     = "#26262c"  # Sub-bloques y contenedores secundarios anidados
BG_INPUT        = "#2c2c32"  # Cajas de texto, spinboxes, tablas, listas y controles comunes

# Bordes Neutros Ultrafinos
BORDER_BASE     = "#2e2e35"  # Borde estructural neutral sutil (sin tonos de color)
BORDER_HOVER    = "#3f3f46"  # Borde interactivo al pasar el ratón
BORDER_FOCUS    = "#2f81f7"  # Borde al tener foco activo (Azul Cobalto)
BORDER_ACTIVE   = "#2f81f7"  # Indicador de estado activo

# Jerarquía Tipográfica de Alto Confort Visual Nocturno
TEXT_PRIMARY    = "#e4e4e7"  # Texto principal de alta legibilidad
TEXT_SECONDARY  = "#a1a1aa"  # Texto descriptivo, etiquetas secundarias y menor contraste
TEXT_MUTED      = "#71717a"  # Textos deshabilitados y marcadores de posición
TEXT_CYAN       = "#e4e4e7"  # Reemplazo limpio y neutro de antiguos textos cian

# 10% - Acento Único de Enfoque (Azul Cobalto Profesional)
ACCENT_COBALT        = "#2f81f7"  # ÚNICO color de enfoque activo
ACCENT_COBALT_HOVER  = "#388bfd"  # Hover acento
ACCENT_COBALT_PRESSED= "#1f6feb"  # Pressed acento
ACCENT_GHOST_BG      = "rgba(47, 129, 247, 0.12)"  # Fondo translúcido para acciones especiales/IA
ACCENT_GHOST_HOVER   = "rgba(47, 129, 247, 0.22)"
ACCENT_GHOST_PRESSED = "rgba(47, 129, 247, 0.32)"

# Estados de Cancelación / Alerta
ACCENT_RED          = "#da3633"  # Cancelación / Alerta en ejecución
ACCENT_RED_HOVER    = "#f85149"
ACCENT_RED_PRESSED  = "#b62324"

# Alias de compatibilidad retroactiva orientados a la nueva paleta neutra
ACCENT_CYAN         = "#2f81f7"
ACCENT_BLUE         = "#2f81f7"
ACCENT_GREEN        = "#2f81f7"  # Los botones de ejecución principales ahora unifican a Azul Cobalto
ACCENT_GREEN_HOVER  = "#388bfd"
ACCENT_GREEN_PRESSED= "#1f6feb"
ACCENT_GOLD         = "#e4e4e7"
ACCENT_TEAL         = "#2f81f7"
ACCENT_PURPLE       = "#2f81f7"

# =============================================================================
# 2. JERARQUÍA DE BOTONES Y ESTILOS REUTILIZABLES
# =============================================================================

# Acción Principal (Lanzar proceso / Iniciar Apilado / Generar Trazas / Exportar)
# Botón táctil con degradado sutil #3B82F6 a #2563EB, borde superior iluminado #60A5FA y texto blanco.
STYLE_RUN_IDLE = f"""
    QPushButton {{
        font-weight: bold;
        font-size: 13px;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3B82F6, stop:1 #2563EB);
        color: #ffffff;
        border: 1px solid #2563EB;
        border-top: 1px solid #60A5FA;
        border-radius: 6px;
        padding: 8px 16px;
    }}
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #60A5FA, stop:1 #3B82F6);
        border: 1px solid #3B82F6;
        border-top: 1px solid #93C5FD;
    }}
    QPushButton:pressed {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1D4ED8, stop:1 #1E40AF);
        border: 1px solid #1E3A8A;
        border-top: 1px solid #3B82F6;
    }}
    QPushButton:disabled {{
        background-color: {BG_PANEL};
        color: {TEXT_MUTED};
        border: 1px solid {BORDER_BASE};
    }}
"""

# Acción Principal Activa (Procesando / Solicitud de Cancelar)
STYLE_RUN_ACTIVE = f"""
    QPushButton {{
        font-weight: bold;
        font-size: 13px;
        background-color: {ACCENT_RED};
        color: #ffffff;
        border: 1px solid {ACCENT_RED};
        border-radius: 6px;
        padding: 8px 16px;
    }}
    QPushButton:hover {{
        background-color: {ACCENT_RED_HOVER};
        border-color: {ACCENT_RED_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_RED_PRESSED};
        border-color: {ACCENT_RED_PRESSED};
    }}
"""

STYLE_RUN_CANCELLING = f"""
    QPushButton {{
        font-weight: bold;
        font-size: 13px;
        background-color: {BG_INPUT};
        color: {TEXT_SECONDARY};
        border: 1px solid {BORDER_BASE};
        border-radius: 6px;
        padding: 8px 16px;
    }}
"""

STYLE_CANCEL_ACTIVE = STYLE_RUN_ACTIVE

# Acción Especial o de IA (ej: "Auto-Detectar Suelo", "1-Click Start", "Separar con StarNet")
# Botón estilo "Ghost/Outline" con borde fino azul #2F81F7 y fondo translúcido rgba(47, 129, 247, 0.12)
STYLE_SPECIAL_ACTION_BTN = f"""
    QPushButton {{
        font-weight: 600;
        font-size: 11px;
        background-color: {ACCENT_GHOST_BG};
        color: {ACCENT_COBALT};
        border: 1px solid {ACCENT_COBALT};
        border-radius: 4px;
        padding: 6px 12px;
    }}
    QPushButton:hover {{
        background-color: {ACCENT_GHOST_HOVER};
        border-color: {ACCENT_COBALT_HOVER};
        color: #ffffff;
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_GHOST_PRESSED};
        border-color: {ACCENT_COBALT_PRESSED};
        color: #ffffff;
    }}
    QPushButton:disabled {{
        background-color: transparent;
        color: {TEXT_MUTED};
        border-color: {BORDER_BASE};
    }}
"""

# Alias para acciones de IA y transferencia entre módulos
STYLE_AI_BTN = STYLE_SPECIAL_ACTION_BTN
STYLE_TRANSFER_BTN = STYLE_SPECIAL_ACTION_BTN

# Botones Secundarios / Ajustes (Cargar, Guardar, Limpiar, Añadir tomas)
# Botón sólido gris oscuro #2C2C32 con texto gris claro, sin bordes de color.
STYLE_SECONDARY_BTN = f"""
    QPushButton {{
        font-weight: 500;
        font-size: 11px;
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER_BASE};
        border-radius: 4px;
        padding: 5px 12px;
    }}
    QPushButton:hover {{
        background-color: #383840;
        border-color: {BORDER_HOVER};
        color: #ffffff;
    }}
    QPushButton:pressed {{
        background-color: {BG_CARD};
    }}
    QPushButton:disabled {{
        background-color: {BG_PANEL};
        color: {TEXT_MUTED};
        border-color: #242429;
    }}
"""

# Botón de peligro / eliminación secundario
STYLE_DANGER_BTN = f"""
    QPushButton {{
        font-weight: 500;
        font-size: 11px;
        background-color: {BG_INPUT};
        color: #f87171;
        border: 1px solid {BORDER_BASE};
        border-radius: 4px;
        padding: 5px 10px;
    }}
    QPushButton:hover {{
        background-color: #383840;
        border-color: {ACCENT_RED};
        color: #ffffff;
    }}
    QPushButton:pressed {{
        background-color: {BG_CARD};
    }}
"""

# Badge / Píldora de estado (Monitores de RAM, versión de motor, telemetría)
STYLE_STATUS_PILL = f"""
    background-color: {BG_CARD};
    border: 1px solid {BORDER_BASE};
    border-radius: 4px;
    padding: 2px 8px;
    color: {TEXT_SECONDARY};
    font-weight: bold;
    font-size: 10px;
"""


# =============================================================================
# 3. HOJA DE ESTILOS GLOBAL DE LA APLICACIÓN (QSS)
# =============================================================================

def get_application_stylesheet() -> str:
    """
    Devuelve la hoja de estilos QSS completa para AstroLoom.
    Garantiza uniformidad tipográfica, espacial y cromática en todas las pestañas
    bajo la regla 60-30-10 (60% Fondos Neutros #121214/#0A0A0B, 30% Paneles/Tarjetas
    #1A1A1E/#222227 con bordes #2E2E35, y 10% Acento Azul Cobalto #2F81F7).
    """
    return f"""
        /* --- VENTANA PRINCIPAL Y CONTENEDORES BASE (60%) --- */
        QMainWindow, QWidget#centralWidget {{
            background-color: {BG_WINDOW};
            color: {TEXT_PRIMARY};
        }}
        
        QWidget {{
            font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Inter", "Roboto", sans-serif;
            font-size: 11px;
            color: {TEXT_PRIMARY};
        }}

        /* --- BARRA DE PESTAÑAS PRINCIPAL DE NAVEGACIÓN (30% + 10% Acento) --- */
        QTabWidget::pane {{
            border: 1px solid {BORDER_BASE};
            background-color: {BG_PANEL};
            border-radius: 0px 0px 6px 6px;
            top: -1px;
        }}
        QTabBar::tab {{
            background-color: {BG_PANEL};
            color: {TEXT_SECONDARY};
            padding: 7px 14px;
            border: 1px solid {BORDER_BASE};
            border-bottom: 2px solid transparent;
            font-weight: bold;
            font-size: 11px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background-color: {BG_CARD};
            color: #ffffff;
            border-color: {BORDER_BASE};
            border-bottom: 2px solid {ACCENT_COBALT};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
            border-bottom: 2px solid {BORDER_HOVER};
        }}

        /* --- TARJETAS MODULARES Y SECCIONES (QGroupBox - 30%) --- */
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {BORDER_BASE};
            border-radius: 6px;
            margin-top: 14px;
            padding: 14px 10px 10px 10px;
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 4px;
            padding: 2px 8px;
            background-color: {BG_PANEL};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            color: {TEXT_PRIMARY};
            font-weight: bold;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* --- BOTONES COMUNES Y SECUNDARIOS --- */
        QPushButton {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            padding: 5px 12px;
            color: {TEXT_PRIMARY};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: #383840;
            border-color: {BORDER_HOVER};
            color: #ffffff;
        }}
        QPushButton:pressed {{
            background-color: {BG_CARD};
        }}
        QPushButton:disabled {{
            background-color: {BG_PANEL};
            color: {TEXT_MUTED};
            border-color: #242429;
        }}

        /* --- ENTRADA DE DATOS NUMÉRICOS Y TEXTO --- */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            padding: 4px 6px;
            color: {TEXT_PRIMARY};
            font-size: 11px;
            selection-background-color: {ACCENT_COBALT};
            selection-color: #ffffff;
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background-color: {BG_PANEL};
            color: {TEXT_MUTED};
            border-color: #242429;
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 16px;
            border-left: 1px solid {BORDER_BASE};
            border-bottom: 1px solid {BORDER_BASE};
            border-top-right-radius: 4px;
            background-color: {BG_CARD};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 16px;
            border-left: 1px solid {BORDER_BASE};
            border-bottom-right-radius: 4px;
            background-color: {BG_CARD};
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: #32323a;
        }}
        QSpinBox::up-button:pressed, QSpinBox::down-button:pressed,
        QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background-color: {BG_PANEL};
        }}
        QSpinBox::up-button:disabled, QSpinBox::down-button:disabled,
        QDoubleSpinBox::up-button:disabled, QDoubleSpinBox::down-button:disabled {{
            background-color: {BG_PANEL};
            border-color: #242429;
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url({CHEVRON_UP_PATH});
            width: 8px;
            height: 8px;
        }}
        QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
            image: url({CHEVRON_UP_DISABLED_PATH});
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url({CHEVRON_DOWN_PATH});
            width: 8px;
            height: 8px;
        }}
        QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
            image: url({CHEVRON_DOWN_DISABLED_PATH});
        }}

        /* --- DESPLEGABLES (QComboBox) --- */
        QComboBox {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            padding: 4px 24px 4px 8px;
            color: {TEXT_PRIMARY};
            font-size: 11px;
        }}
        QComboBox:hover {{
            border-color: {BORDER_HOVER};
        }}
        QComboBox:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
        QComboBox:disabled {{
            background-color: {BG_PANEL};
            color: {TEXT_MUTED};
            border-color: #242429;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 22px;
            border: none;
            background: transparent;
        }}
        QComboBox::drop-down:hover {{
            background-color: {ACCENT_GHOST_BG};
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
        }}
        QComboBox::down-arrow {{
            image: url({CHEVRON_DOWN_PATH});
            width: 10px;
            height: 10px;
        }}
        QComboBox::down-arrow:hover {{
            image: url({CHEVRON_DOWN_HOVER_PATH});
        }}
        QComboBox::down-arrow:on {{
            image: url({CHEVRON_DOWN_HOVER_PATH});
        }}
        QComboBox::down-arrow:disabled {{
            image: url({CHEVRON_DOWN_DISABLED_PATH});
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER_BASE};
            color: {TEXT_PRIMARY};
            selection-background-color: {ACCENT_COBALT};
            selection-color: #ffffff;
            outline: none;
            padding: 2px;
        }}

        /* --- DESLIZADORES (QSlider) - Tirador Píldora Blanco Hueso #F4F4F5 --- */
        QSlider::groove:horizontal {{
            border: none;
            height: 4px;
            background: {BG_INPUT};
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: #3B82F6;
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal:disabled {{
            background: {BG_CARD};
        }}
        QSlider::handle:horizontal {{
            background: #F4F4F5;
            width: 8px;
            height: 16px;
            margin: -6px 0;
            border-radius: 4px;
            border: 1px solid #18181B;
        }}
        QSlider::handle:horizontal:hover {{
            background: #FFFFFF;
            border: 1px solid #09090B;
        }}
        QSlider::handle:horizontal:pressed {{
            background: #E4E4E7;
            border: 1px solid #09090B;
        }}
        QSlider::handle:horizontal:disabled {{
            background: #52525B;
            border: 1px solid #27272A;
        }}
        QSlider::groove:vertical {{
            border: none;
            width: 4px;
            background: {BG_INPUT};
            border-radius: 2px;
        }}
        QSlider::sub-page:vertical {{
            background: #3B82F6;
            border-radius: 2px;
        }}
        QSlider::sub-page:vertical:disabled {{
            background: {BG_CARD};
        }}
        QSlider::handle:vertical {{
            background: #F4F4F5;
            height: 8px;
            width: 16px;
            margin: 0 -6px;
            border-radius: 4px;
            border: 1px solid #18181B;
        }}
        QSlider::handle:vertical:hover {{
            background: #FFFFFF;
            border: 1px solid #09090B;
        }}
        QSlider::handle:vertical:pressed {{
            background: #E4E4E7;
            border: 1px solid #09090B;
        }}
        QSlider::handle:vertical:disabled {{
            background: #52525B;
            border: 1px solid #27272A;
        }}

        /* --- CASILLAS DE VERIFICACIÓN (QCheckBox) --- */
        QCheckBox {{
            color: {TEXT_SECONDARY};
            font-size: 11px;
            spacing: 7px;
        }}
        QCheckBox:checked {{
            color: {TEXT_PRIMARY};
        }}
        QCheckBox:disabled {{
            color: {TEXT_MUTED};
        }}
        QCheckBox::indicator {{
            width: 14px;
            height: 14px;
            background-color: #1E1E24;
            border: 1.5px solid #55555F;
            border-radius: 3px;
        }}
        QCheckBox::indicator:hover {{
            border-color: #888896;
            background-color: #26262D;
        }}
        QCheckBox::indicator:checked {{
            image: url({CHECK_WHITE_PATH});
            background-color: {ACCENT_COBALT};
            border: 1.5px solid {ACCENT_COBALT};
        }}
        QCheckBox::indicator:checked:hover {{
            background-color: {ACCENT_COBALT_HOVER};
            border-color: {ACCENT_COBALT_HOVER};
        }}
        QCheckBox::indicator:disabled {{
            background-color: #18181B;
            border: 1.5px solid #383842;
        }}
        QCheckBox::indicator:checked:disabled {{
            image: url({CHECK_DISABLED_PATH});
            background-color: #222938;
            border: 1.5px solid #2D3748;
        }}

        /* --- SELECTORES DE OPCIÓN ÚNICA (QRadioButton) --- */
        QRadioButton {{
            color: {TEXT_SECONDARY};
            font-size: 11px;
            spacing: 7px;
        }}
        QRadioButton:checked {{
            color: {TEXT_PRIMARY};
        }}
        QRadioButton:disabled {{
            color: {TEXT_MUTED};
        }}
        QRadioButton::indicator {{
            width: 14px;
            height: 14px;
            background-color: #1E1E24;
            border: 1.5px solid #55555F;
            border-radius: 7px;
        }}
        QRadioButton::indicator:hover {{
            border-color: #888896;
            background-color: #26262D;
        }}
        QRadioButton::indicator:checked {{
            background-color: {ACCENT_COBALT};
            border: 3.5px solid #1E1E24;
        }}
        QRadioButton::indicator:checked:hover {{
            background-color: {ACCENT_COBALT_HOVER};
        }}
        QRadioButton::indicator:disabled {{
            background-color: #18181B;
            border: 1.5px solid #383842;
        }}
        QRadioButton::indicator:checked:disabled {{
            background-color: {TEXT_MUTED};
            border: 3.5px solid #18181B;
        }}

        /* --- TABLAS Y LISTAS DE ARCHIVOS (QTableWidget, QListWidget) --- */
        QTableWidget, QTableView, QListWidget {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            color: {TEXT_PRIMARY};
            gridline-color: {BG_CARD};
            font-size: 11px;
            outline: none;
        }}
        QTableWidget::item, QTableView::item, QListWidget::item {{
            padding: 3px 6px;
            border-bottom: 1px solid {BG_CARD};
        }}
        QTableWidget::item:selected, QTableView::item:selected, QListWidget::item:selected {{
            background-color: rgba(47, 129, 247, 0.25);
            color: #ffffff;
            border-left: 2px solid {ACCENT_COBALT};
        }}
        QTableWidget::item:hover:!selected, QTableView::item:hover:!selected, QListWidget::item:hover:!selected {{
            background-color: #383840;
        }}
        QHeaderView::section {{
            background-color: {BG_PANEL};
            color: {TEXT_SECONDARY};
            padding: 5px 6px;
            border: 1px solid {BORDER_BASE};
            font-weight: bold;
            font-size: 10px;
            text-transform: uppercase;
        }}

        /* --- BARRAS DE PROGRESO (QProgressBar) --- */
        QProgressBar {{
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            background-color: {BG_INPUT};
            text-align: center;
            color: #ffffff;
            font-weight: bold;
            font-size: 11px;
            height: 18px;
        }}
        QProgressBar::chunk {{
            background-color: {ACCENT_COBALT};
            border-radius: 3px;
        }}

        /* --- BARRAS DE DESPLAZAMIENTO SLIM (QScrollBar) --- */
        QScrollBar:vertical {{
            background: {BG_WINDOW};
            width: 8px;
            margin: 0;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {BG_INPUT};
            min-height: 20px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {BORDER_HOVER};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background: {BG_WINDOW};
            height: 8px;
            margin: 0;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal {{
            background: {BG_INPUT};
            min-width: 20px;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {BORDER_HOVER};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}

        /* --- CONSOLAS DE TEXTO Y LOGS (QTextEdit) --- */
        QTextEdit, QPlainTextEdit {{
            background-color: {BG_CONSOLE};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            font-family: "Consolas", "Cascadia Code", "Courier New", monospace;
            font-size: 11px;
            selection-background-color: rgba(47, 129, 247, 0.35);
        }}

        /* --- ÁREAS DE DESPLAZAMIENTO (QScrollArea) --- */
        QScrollArea {{
            border: none;
            background: transparent;
        }}

        /* --- DIVISORES AJUSTABLES (QSplitter) --- */
        QSplitter::handle {{
            background-color: {BG_PANEL};
        }}
        QSplitter::handle:horizontal {{
            background-color: {BG_PANEL};
            border-left: 1px solid {BORDER_BASE};
            border-right: 1px solid {BORDER_BASE};
        }}
        QSplitter::handle:horizontal:hover {{
            background-color: {ACCENT_COBALT};
        }}

        /* --- INFORMACIÓN FLOTANTE (QToolTip) --- */
        QToolTip {{
            background-color: {BG_PANEL};
            border: 1px solid {BORDER_BASE};
            color: {TEXT_PRIMARY};
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 11px;
        }}

        /* --- BARRA DE ESTADO (QStatusBar) --- */
        QStatusBar {{
            background-color: {BG_WINDOW};
            border-top: 1px solid {BORDER_BASE};
            color: {TEXT_SECONDARY};
            font-size: 11px;
            padding: 2px 8px;
        }}

        /* --- BARRA SUPERIOR DEL LIENZO (CANVAS TOOLBAR) --- */
        QWidget#canvas_toolbar {{
            background-color: {BG_PANEL};
            border-bottom: 1px solid {BORDER_BASE};
            min-height: 36px;
            max-height: 38px;
        }}
        QWidget#canvas_toolbar QPushButton {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            padding: 3px 9px;
            color: {TEXT_SECONDARY};
            font-size: 11px;
            font-weight: 500;
        }}
        QWidget#canvas_toolbar QPushButton:hover {{
            background-color: {BG_INPUT};
            border-color: {BORDER_HOVER};
            color: #ffffff;
        }}
        QWidget#canvas_toolbar QPushButton:pressed {{
            background-color: {BG_PANEL};
        }}
        QWidget#canvas_toolbar QPushButton:checked {{
            background-color: {ACCENT_COBALT};
            border: 1px solid {ACCENT_COBALT};
            color: #ffffff;
            font-weight: 600;
        }}
        QWidget#canvas_toolbar QPushButton:checked:hover {{
            background-color: {ACCENT_COBALT_HOVER};
            border-color: {ACCENT_COBALT_HOVER};
        }}
        QWidget#canvas_toolbar QPushButton:disabled {{
            background-color: {BG_PANEL};
            color: {TEXT_MUTED};
            border-color: {BORDER_BASE};
        }}
        QWidget#canvas_toolbar QFrame {{
            color: {BORDER_BASE};
        }}

        /* --- BARRA INFERIOR DE ESTADO DEL LIENZO (CANVAS FOOTER BAR) --- */
        QWidget#canvas_footer_bar {{
            background-color: {BG_WINDOW};
            border-top: 1px solid {BORDER_BASE};
            min-height: 24px;
            max-height: 26px;
        }}
        QWidget#canvas_footer_bar QLabel {{
            color: {TEXT_SECONDARY};
            font-size: 11px;
        }}
        QWidget#canvas_footer_bar QFrame {{
            color: {BORDER_BASE};
        }}
    """


def apply_theme(target: Optional[QWidget] = None) -> None:
    """
    Aplica la hoja de estilos global a la aplicación QApplication activa
    o a un widget contenedor (e.g. QMainWindow).
    """
    stylesheet = get_application_stylesheet()
    if target is not None:
        target.setStyleSheet(stylesheet)
    else:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)
