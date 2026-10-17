# gui/theme.py
"""
gui/theme.py - Sistema de Diseño y Hoja de Estilos Global Unificada de AstroLoom.

Proporciona la paleta de colores cromática 'Cosmic Dark Pro', estilos reutilizables
para botones primarios y de transferencia entre módulos, y la hoja de estilos QSS
maestra que garantiza una identidad visual unificada en todas las pestañas.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget, QApplication

# =============================================================================
# 1. PALETA DE COLORES MAESTRA (Cosmic Dark Pro)
# =============================================================================
BG_WINDOW       = "#0b0d14"  # Fondo base de ventanas y visores
BG_PANEL        = "#12141e"  # Paneles laterales y áreas de herramientas
BG_CARD         = "#171a26"  # Tarjetas, QGroupBox y contenedores elevados
BG_CARD_ALT     = "#1c2030"  # Elementos anidados o resaltados dentro de tarjetas
BG_INPUT        = "#0d0f17"  # Cajas de texto, spinboxes, tablas y listas
BG_CONSOLE      = "#08090f"  # Terminal y cajas de logs monospaciadas

BORDER_BASE     = "#232738"  # Bordes sutiles y divisores estructurales
BORDER_HOVER    = "#3a415a"  # Borde al pasar el ratón (hover)
BORDER_FOCUS    = "#00bcd4"  # Borde al tener foco activo (Cyan suave)
BORDER_ACTIVE   = "#00e5ff"  # Resalte vibrante activo

TEXT_PRIMARY    = "#eceff4"  # Texto principal de alta legibilidad
TEXT_SECONDARY  = "#90a4ae"  # Texto descriptivo, leyendas y encabezados de tabla
TEXT_MUTED      = "#546e7a"  # Textos deshabilitados o placeholders
TEXT_CYAN       = "#80d8ff"  # Títulos de grupos y datos celestiales destacados

ACCENT_CYAN     = "#00e5ff"  # Identidad visual de AstroLoom y pestañas activas
ACCENT_BLUE     = "#1565c0"  # Acciones secundarias principales
ACCENT_GREEN    = "#2e7d32"  # Botón primario de ejecución (Run / Apilar / Coser)
ACCENT_GREEN_HOVER = "#388e3c"
ACCENT_GREEN_PRESSED = "#1b5e20"
ACCENT_RED      = "#c62828"  # Botón de cancelación / Peligro / Eliminar
ACCENT_RED_HOVER = "#d32f2f"
ACCENT_RED_PRESSED = "#b71c1c"
ACCENT_GOLD     = "#ffd54f"  # Parámetros numéricos clave y métricas estelares
ACCENT_TEAL     = "#00838f"  # Botón de transferencia entre pestañas ("Enviar al Revelador")
ACCENT_PURPLE   = "#7b1fa2"  # Modos astronómicos especiales y máscaras

# =============================================================================
# 2. ESTILOS REUTILIZABLES PARA BOTONES DE ACCIÓN
# =============================================================================

# Botón de ejecución en reposo (Verde cósmico - Listo para iniciar tarea pesada)
STYLE_RUN_IDLE = f"""
    QPushButton {{
        font-weight: bold;
        font-size: 13px;
        background-color: {ACCENT_GREEN};
        color: #ffffff;
        border: 1px solid #43a047;
        border-radius: 6px;
        padding: 8px 16px;
    }}
    QPushButton:hover {{
        background-color: {ACCENT_GREEN_HOVER};
        border-color: #4caf50;
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_GREEN_PRESSED};
    }}
    QPushButton:disabled {{
        background-color: #1a241c;
        color: #526e57;
        border-color: #243527;
    }}
"""

# Botón de ejecución activo (Rojo carmesí - Procesando / Cancelar)
STYLE_RUN_ACTIVE = f"""
    QPushButton {{
        font-weight: bold;
        font-size: 13px;
        background-color: {ACCENT_RED};
        color: #ffffff;
        border: 1px solid #e53935;
        border-radius: 6px;
        padding: 8px 16px;
    }}
    QPushButton:hover {{
        background-color: {ACCENT_RED_HOVER};
        border-color: #ef5350;
    }}
    QPushButton:pressed {{
        background-color: {ACCENT_RED_PRESSED};
    }}
"""

# Botón en transición de parada (Gris neutro - Solicitando detención)
STYLE_RUN_CANCELLING = f"""
    QPushButton {{
        font-weight: bold;
        font-size: 13px;
        background-color: #37474f;
        color: #b0bec5;
        border: 1px solid #455a64;
        border-radius: 6px;
        padding: 8px 16px;
    }}
"""

# Alias para coherencia con módulos anteriores
STYLE_CANCEL_ACTIVE = STYLE_RUN_ACTIVE

# Botón de transferencia entre módulos ("Enviar a Revelador / Editor")
STYLE_TRANSFER_BTN = f"""
    QPushButton {{
        font-weight: bold;
        font-size: 12px;
        background-color: {ACCENT_TEAL};
        color: #ffffff;
        border: 1px solid #00acc1;
        border-radius: 6px;
        padding: 7px 14px;
    }}
    QPushButton:hover {{
        background-color: #0097a7;
        border-color: #26c6da;
    }}
    QPushButton:pressed {{
        background-color: #006064;
    }}
    QPushButton:disabled {{
        background-color: #152528;
        color: #43656b;
        border-color: #1f373b;
    }}
"""

# Botón secundario estilizado (Cargar tomas, marcar referencia, etc.)
STYLE_SECONDARY_BTN = f"""
    QPushButton {{
        font-weight: 500;
        font-size: 11px;
        background-color: #1e2230;
        color: {TEXT_PRIMARY};
        border: 1px solid #33384c;
        border-radius: 4px;
        padding: 5px 12px;
    }}
    QPushButton:hover {{
        background-color: #282e42;
        border-color: #4f5573;
    }}
    QPushButton:pressed {{
        background-color: #171924;
    }}
    QPushButton:disabled {{
        background-color: #141620;
        color: {TEXT_MUTED};
        border-color: #232738;
    }}
"""

# Botón de peligro / eliminación (Limpiar listas, resetear)
STYLE_DANGER_BTN = f"""
    QPushButton {{
        font-weight: 500;
        font-size: 11px;
        background-color: #27191e;
        color: #ff8a80;
        border: 1px solid #4d232a;
        border-radius: 4px;
        padding: 5px 10px;
    }}
    QPushButton:hover {{
        background-color: #3b2028;
        border-color: #ff5252;
        color: #ffffff;
    }}
    QPushButton:pressed {{
        background-color: #1f1216;
    }}
"""

# Badge / Píldora de estado (Monitores de hardware, versión, modo activo)
STYLE_STATUS_PILL = f"""
    background-color: #141724;
    border: 1px solid {BORDER_BASE};
    border-radius: 4px;
    padding: 2px 8px;
    color: {TEXT_CYAN};
    font-weight: bold;
    font-size: 10px;
"""


# =============================================================================
# 3. HOJA DE ESTILOS GLOBAL DE LA APLICACIÓN (QSS)
# =============================================================================

def get_application_stylesheet() -> str:
    """
    Devuelve la hoja de estilos QSS completa para AstroLoom.
    Garantiza uniformidad tipográfica, espacial y cromática en todas las pestañas.
    """
    return f"""
        /* --- VENTANA PRINCIPAL Y CONTENEDORES BASE --- */
        QMainWindow, QWidget#centralWidget {{
            background-color: {BG_WINDOW};
            color: {TEXT_PRIMARY};
        }}
        
        QWidget {{
            font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Inter", "Roboto", sans-serif;
            font-size: 11px;
            color: {TEXT_PRIMARY};
        }}

        /* --- BARRA DE PESTAÑAS PRINCIPAL DE NAVEGACIÓN --- */
        QTabWidget::pane {{
            border: 1px solid {BORDER_BASE};
            background-color: {BG_PANEL};
            border-radius: 0px 0px 6px 6px;
            top: -1px;
        }}
        QTabBar::tab {{
            background-color: #10121a;
            color: {TEXT_SECONDARY};
            padding: 8px 16px;
            border: 1px solid #1e2230;
            border-bottom: 2px solid transparent;
            font-weight: bold;
            font-size: 11px;
            margin-right: 2px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }}
        QTabBar::tab:selected {{
            background-color: {BG_CARD};
            color: {ACCENT_CYAN};
            border-color: {BORDER_BASE};
            border-bottom: 2px solid {ACCENT_CYAN};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: #151822;
            color: {TEXT_PRIMARY};
            border-bottom: 2px solid {BORDER_HOVER};
        }}

        /* --- TARJETAS MODULARES (QGroupBox) --- */
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {BORDER_BASE};
            border-radius: 8px;
            margin-top: 14px;
            padding: 16px 10px 10px 10px;
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 14px;
            top: 4px;
            padding: 2px 8px;
            background-color: {BG_CARD_ALT};
            border: 1px solid #282f44;
            border-radius: 4px;
            color: {TEXT_CYAN};
            font-weight: bold;
            font-size: 11px;
        }}

        /* --- BOTONES ESTÁNDAR --- */
        QPushButton {{
            background-color: #1f2332;
            border: 1px solid #33384c;
            border-radius: 4px;
            padding: 5px 12px;
            color: {TEXT_PRIMARY};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: #292f44;
            border-color: #4f5573;
        }}
        QPushButton:pressed {{
            background-color: #161824;
        }}
        QPushButton:disabled {{
            background-color: #141620;
            color: {TEXT_MUTED};
            border-color: #232738;
        }}

        /* --- ENTRADA DE DATOS NUMÉRICOS Y TEXTO --- */
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {BG_INPUT};
            border: 1px solid #2a2e42;
            border-radius: 4px;
            padding: 4px 6px;
            color: {TEXT_PRIMARY};
            font-size: 11px;
            selection-background-color: {ACCENT_BLUE};
            selection-color: #ffffff;
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: #181b28;
            border: none;
            width: 16px;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: #252b3d;
        }}

        /* --- DESPLEGABLES (QComboBox) --- */
        QComboBox {{
            background-color: {BG_INPUT};
            border: 1px solid #2a2e42;
            border-radius: 4px;
            padding: 4px 8px;
            color: {TEXT_PRIMARY};
            font-size: 11px;
        }}
        QComboBox:focus {{
            border: 1px solid {BORDER_FOCUS};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {BG_CARD};
            border: 1px solid #33384c;
            color: {TEXT_PRIMARY};
            selection-background-color: {ACCENT_BLUE};
            selection-color: #ffffff;
            outline: none;
            padding: 2px;
        }}

        /* --- DESLIZADORES (QSlider) - Pro Studio Styling --- */
        QSlider::groove:horizontal {{
            border: 1px solid #232738;
            height: 3px;
            background: #141722;
            border-radius: 1.5px;
        }}
        QSlider::sub-page:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
            border-radius: 1.5px;
        }}
        QSlider::sub-page:horizontal:disabled {{
            background: #232738;
        }}
        QSlider::handle:horizontal {{
            background: #e2e8f0;
            border: 1.5px solid #38bdf8;
            width: 12px;
            height: 12px;
            margin: -5px 0;
            border-radius: 6px;
        }}
        QSlider::handle:horizontal:hover {{
            background: #ffffff;
            border: 1.5px solid #00e5ff;
            width: 13px;
            height: 13px;
            margin: -5px 0;
            border-radius: 6.5px;
        }}
        QSlider::handle:horizontal:pressed {{
            background: #00e5ff;
            border: 1.5px solid #ffffff;
        }}
        QSlider::handle:horizontal:disabled {{
            background: #475569;
            border: 1px solid #334155;
        }}

        /* --- CASILLAS Y SELECTORES --- */
        QCheckBox, QRadioButton {{
            color: {TEXT_PRIMARY};
            font-size: 11px;
            spacing: 6px;
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 14px;
            height: 14px;
            background-color: {BG_INPUT};
            border: 1px solid #33384c;
            border-radius: 3px;
        }}
        QRadioButton::indicator {{
            border-radius: 7px;
        }}
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
            border-color: {ACCENT_CYAN};
        }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: #0288d1;
            border: 1px solid {ACCENT_CYAN};
        }}

        /* --- TABLAS Y LISTAS DE ARCHIVOS (QTableWidget, QListWidget) --- */
        QTableWidget, QTableView, QListWidget {{
            background-color: {BG_INPUT};
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            color: {TEXT_PRIMARY};
            gridline-color: #1b1e2c;
            font-size: 11px;
            outline: none;
        }}
        QTableWidget::item, QTableView::item, QListWidget::item {{
            padding: 3px 6px;
            border-bottom: 1px solid #141724;
        }}
        QTableWidget::item:selected, QTableView::item:selected, QListWidget::item:selected {{
            background-color: #173852;
            color: {ACCENT_CYAN};
        }}
        QTableWidget::item:hover, QTableView::item:hover, QListWidget::item:hover {{
            background-color: #161a28;
        }}
        QHeaderView::section {{
            background-color: #141724;
            color: {TEXT_SECONDARY};
            padding: 5px 6px;
            border: 1px solid {BORDER_BASE};
            font-weight: bold;
            font-size: 10px;
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
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0288d1, stop:1 {ACCENT_CYAN});
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
            background: #2b3044;
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
            background: #2b3044;
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
            color: #81c784;
            border: 1px solid {BORDER_BASE};
            border-radius: 4px;
            font-family: "Consolas", "Cascadia Code", "Courier New", monospace;
            font-size: 11px;
            selection-background-color: #1b384d;
        }}

        /* --- ÁREAS DE DESPLAZAMIENTO (QScrollArea) --- */
        QScrollArea {{
            border: none;
            background: transparent;
        }}

        /* --- DIVISORES AJUSTABLES (QSplitter) --- */
        QSplitter::handle {{
            background-color: #161a28;
        }}
        QSplitter::handle:horizontal {{
            background-color: #141824;
            border-left: 1px solid #1f2638;
            border-right: 1px solid #1f2638;
        }}
        QSplitter::handle:horizontal:hover {{
            background-color: #38bdf8;
            border-left: 1px solid #7dd3fc;
            border-right: 1px solid #7dd3fc;
        }}

        /* --- INFORMACIÓN FLOTANTE (QToolTip) --- */
        QToolTip {{
            background-color: #1a1d2b;
            border: 1px solid #3d435d;
            color: {TEXT_PRIMARY};
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 11px;
        }}

        /* --- BARRA DE ESTADO (QStatusBar) --- */
        QStatusBar {{
            background-color: #08090e;
            border-top: 1px solid {BORDER_BASE};
            color: {TEXT_SECONDARY};
            font-size: 11px;
            padding: 2px 8px;
        }}

        /* --- BARRA SUPERIOR DEL LIENZO (CANVAS TOOLBAR) --- */
        QWidget#canvas_toolbar {{
            background-color: #121520;
            border-bottom: 1px solid #1e2230;
            min-height: 36px;
            max-height: 38px;
        }}
        QWidget#canvas_toolbar QPushButton {{
            background-color: #161924;
            border: 1px solid #232738;
            border-radius: 4px;
            padding: 3px 9px;
            color: #94a3b8;
            font-size: 11px;
            font-weight: 500;
        }}
        QWidget#canvas_toolbar QPushButton:hover {{
            background-color: #1c2234;
            border-color: #38bdf8;
            color: #ffffff;
        }}
        QWidget#canvas_toolbar QPushButton:pressed {{
            background-color: #0f172a;
        }}
        QWidget#canvas_toolbar QPushButton:checked {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0369a1, stop:1 #0284c7);
            border: 1px solid #38bdf8;
            color: #ffffff;
            font-weight: 600;
        }}
        QWidget#canvas_toolbar QPushButton:checked:hover {{
            background: #0284c7;
            border-color: #7dd3fc;
        }}
        QWidget#canvas_toolbar QPushButton:disabled {{
            background-color: #12141c;
            color: #475569;
            border-color: #1a1e2a;
        }}
        QWidget#canvas_toolbar QFrame {{
            color: #232738;
        }}

        /* --- BARRA INFERIOR DE ESTADO DEL LIENZO (CANVAS FOOTER BAR) --- */
        QWidget#canvas_footer_bar {{
            background-color: #0d1017;
            border-top: 1px solid #1e2230;
            min-height: 24px;
            max-height: 26px;
        }}
        QWidget#canvas_footer_bar QLabel {{
            color: #94a3b8;
            font-size: 11px;
        }}
        QWidget#canvas_footer_bar QFrame {{
            color: #232738;
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
