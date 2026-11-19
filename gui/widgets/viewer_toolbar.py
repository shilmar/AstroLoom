# gui/widgets/viewer_toolbar.py
"""
gui/widgets/viewer_toolbar.py - Barra de herramientas superior unificada y modular
para los visores interactivos de AstroLoom (Startrails, Meteoros, Panorámicas, Eclipses).

Inspirada en la ergonomía y estética de Adobe Lightroom y el Revelador RAW de AstroLoom.
Permite colapsar el panel lateral izquierdo, regular modos de visualización (Natural/MTF/Realce),
activar indicadores de recorte, zoom 1:1, vista ajustada y lupa de inspección flotante.
"""

from typing import Optional
from PySide6.QtCore import Qt, Signal, QCoreApplication
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QComboBox, QFrame, QSizePolicy, QProgressBar
)
from gui.icons import get_icon
from gui.theme import (
    BG_PANEL, BORDER_BASE, TEXT_PRIMARY, TEXT_SECONDARY,
    TEXT_MUTED, TEXT_CYAN, ACCENT_CYAN
)


class ViewerToolbar(QWidget):
    """
    Barra superior de herramientas homogénea y estilizada para los visores de AstroLoom.
    """
    left_panel_toggled = Signal(bool)
    right_panel_toggled = Signal(bool)
    view_mode_changed = Signal(int)
    composite_toggled = Signal(bool)
    clipping_toggled = Signal(bool)
    zoom_100_toggled = Signal(bool)
    zoom_fit_clicked = Signal()
    magnifier_toggled = Signal(bool)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        show_left_toggle: bool = True,
        show_right_toggle: bool = False,
        show_view_mode: bool = False,
        show_composite_btn: bool = False,
        show_clipping: bool = False,
        show_zoom: bool = True,
        show_magnifier: bool = False,
    ):
        super().__init__(parent)
        self.setObjectName("canvas_toolbar")
        self.setFixedHeight(38)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 3, 8, 3)
        self._layout.setSpacing(6)

        # 0. Botón Plegar / Desplegar Panel Izquierdo (F7 / [)
        if show_left_toggle:
            self.btn_toggle_left_panel = QPushButton()
            self.btn_toggle_left_panel.setIcon(
                get_icon("panel_left", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=14)
            )
            self.btn_toggle_left_panel.setCheckable(True)
            self.btn_toggle_left_panel.setChecked(True)
            self.btn_toggle_left_panel.setCursor(Qt.PointingHandCursor)
            self.btn_toggle_left_panel.setToolTip("Ocultar / Mostrar panel lateral izquierdo (Atajo: F7 o [)")
            self.btn_toggle_left_panel.setFixedSize(30, 26)
            self.btn_toggle_left_panel.toggled.connect(self.left_panel_toggled.emit)
            self._layout.addWidget(self.btn_toggle_left_panel)
            self._add_separator()

        # 1. Selector de modo de visualización calibrada (Natural / Auto-MTF / Realce)
        if show_view_mode:
            lbl_v = QLabel("Visualización:")
            lbl_v.setStyleSheet(f"font-weight: bold; color: {TEXT_CYAN}; font-size: 11px;")
            self._layout.addWidget(lbl_v)

            self.combo_view_mode = QComboBox()
            self.combo_view_mode.addItems([
                "Natural / sRGB (Fiel al Revelador)",
                "Auto-MTF (Cielo Profundo)",
                "Realce de Trazas (Alto Contraste)"
            ])
            self.combo_view_mode.setToolTip(
                "Ajusta la representación visual en pantalla sin alterar los datos científicos de 32 bits ni la exportación."
            )
            self.combo_view_mode.currentIndexChanged.connect(self.view_mode_changed.emit)
            self._layout.addWidget(self.combo_view_mode)

        # 2. Botón Ver Compuesto (Meteoros)
        if show_composite_btn:
            self.btn_composite = QPushButton("Ver Compuesto")
            self.btn_composite.setIcon(
                get_icon("layers", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13)
            )
            self.btn_composite.setCheckable(True)
            self.btn_composite.setChecked(False)
            self.btn_composite.setEnabled(False)
            self.btn_composite.setCursor(Qt.PointingHandCursor)
            self.btn_composite.setToolTip(
                "Alterna la visualización entre el compuesto final con todos los meteoros y los fotogramas individuales de la serie."
            )
            self.btn_composite.toggled.connect(self.composite_toggled.emit)
            self._layout.addWidget(self.btn_composite)

        if show_view_mode or show_composite_btn:
            self._add_separator()

        # 3. Botón Aviso de Recorte (Clipping)
        if show_clipping:
            self.btn_clipping = QPushButton("Aviso recorte")
            self.btn_clipping.setIcon(
                get_icon("alert", color="#94a3b8", hover_color="#ffffff", active_color="#f87171", size=13)
            )
            self.btn_clipping.setCheckable(True)
            self.btn_clipping.setCursor(Qt.PointingHandCursor)
            self.btn_clipping.setToolTip(
                "Muestra en el lienzo los píxeles con recorte:\n"
                "• Azul para negros empastados (RGB=0)\n"
                "• Rojo para luces quemadas (RGB=255)\n"
                "Atajo de teclado: J"
            )
            self.btn_clipping.toggled.connect(self.clipping_toggled.emit)
            self._layout.addWidget(self.btn_clipping)
            self._add_separator()

        # 4. Botones Zoom (1:1 y Ajustar)
        if show_zoom:
            self.btn_zoom_100 = QPushButton("1:1")
            self.btn_zoom_100.setIcon(
                get_icon("zoom_in", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13)
            )
            self.btn_zoom_100.setCheckable(True)
            self.btn_zoom_100.setCursor(Qt.PointingHandCursor)
            self.btn_zoom_100.setToolTip(
                "Alterna ampliación 1:1 al píxel nativo del sensor.\n"
                "Atajo: Clic derecho en el lienzo."
            )
            self.btn_zoom_100.toggled.connect(self.zoom_100_toggled.emit)
            self._layout.addWidget(self.btn_zoom_100)

            self.btn_zoom_fit = QPushButton("Ajustar")
            self.btn_zoom_fit.setIcon(
                get_icon("maximize", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13)
            )
            self.btn_zoom_fit.setCursor(Qt.PointingHandCursor)
            self.btn_zoom_fit.setToolTip("Ajusta la imagen completa a la ventana del visor.")
            self.btn_zoom_fit.clicked.connect(self._on_fit_clicked)
            self._layout.addWidget(self.btn_zoom_fit)
            self._add_separator()

        # 5. Lupa de Inspección Flotante
        if show_magnifier:
            self.btn_magnifier = QPushButton()
            self.btn_magnifier.setIcon(
                get_icon("search", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13)
            )
            self.btn_magnifier.setCheckable(True)
            self.btn_magnifier.setCursor(Qt.PointingHandCursor)
            self.btn_magnifier.setFixedSize(30, 26)
            self.btn_magnifier.setToolTip(
                "Activa la lupa de inspección flotante sobre el cursor para examinar estrellas y ruido.\n"
                "Regula el aumento con la rueda del ratón.\n"
                "Atajo de teclado: Z"
            )
            self.btn_magnifier.toggled.connect(self.magnifier_toggled.emit)
            self._layout.addWidget(self.btn_magnifier)
            self._add_separator()

        # Espaciador elástico antes de los elementos de la derecha
        self._stretch_index = self._layout.count()
        self._layout.addStretch()

        # 6. Botón Plegar / Desplegar Panel Derecho (solo si show_right_toggle es True)
        if show_right_toggle:
            self._add_separator()
            self.btn_toggle_right_panel = QPushButton()
            self.btn_toggle_right_panel.setIcon(
                get_icon("panel_right", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=14)
            )
            self.btn_toggle_right_panel.setCheckable(True)
            self.btn_toggle_right_panel.setChecked(True)
            self.btn_toggle_right_panel.setCursor(Qt.PointingHandCursor)
            self.btn_toggle_right_panel.setToolTip("Ocultar / Mostrar panel lateral derecho (Atajo: F8 o ])")
            self.btn_toggle_right_panel.setFixedSize(30, 26)
            self.btn_toggle_right_panel.toggled.connect(self.right_panel_toggled.emit)
            self._layout.addWidget(self.btn_toggle_right_panel)

        self._apply_styles()

    def _add_separator(self) -> None:
        """Inserta un divisor vertical sutil."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #2b3042; margin: 4px 2px;")
        self._layout.addWidget(sep)

    def insert_custom_widget_before_stretch(self, widget: QWidget) -> None:
        """Inserta un widget personalizado en la sección izquierda de la barra (antes del stretch)."""
        idx = self._find_stretch_index()
        self._layout.insertWidget(idx, widget)

    def add_separator_before_stretch(self) -> None:
        """Inserta un divisor antes del stretch."""
        idx = self._find_stretch_index()
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #2b3042; margin: 4px 2px;")
        self._layout.insertWidget(idx, sep)

    def add_widget(self, widget: QWidget) -> None:
        """Añade un widget a la derecha (después del stretch, ej. etiquetas de estado)."""
        # Si existe botón derecho, insertar antes de él; si no, al final
        if hasattr(self, 'btn_toggle_right_panel'):
            idx = self._layout.indexOf(self.btn_toggle_right_panel)
            self._layout.insertWidget(idx, widget)
        else:
            self._layout.addWidget(widget)

    def add_separator(self) -> None:
        """Añade un divisor."""
        self._add_separator()

    def _find_stretch_index(self) -> int:
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item.spacerItem() is not None:
                return i
        return self._layout.count()

    def _on_fit_clicked(self) -> None:
        if hasattr(self, 'btn_zoom_100'):
            self.btn_zoom_100.blockSignals(True)
            self.btn_zoom_100.setChecked(False)
            self.btn_zoom_100.blockSignals(False)
        self.zoom_fit_clicked.emit()

    def set_left_panel_checked(self, checked: bool) -> None:
        if hasattr(self, 'btn_toggle_left_panel'):
            self.btn_toggle_left_panel.blockSignals(True)
            self.btn_toggle_left_panel.setChecked(checked)
            self.btn_toggle_left_panel.blockSignals(False)

    def set_right_panel_checked(self, checked: bool) -> None:
        if hasattr(self, 'btn_toggle_right_panel'):
            self.btn_toggle_right_panel.blockSignals(True)
            self.btn_toggle_right_panel.setChecked(checked)
            self.btn_toggle_right_panel.blockSignals(False)

    def set_zoom_100_checked(self, checked: bool) -> None:
        if hasattr(self, 'btn_zoom_100'):
            self.btn_zoom_100.blockSignals(True)
            self.btn_zoom_100.setChecked(checked)
            self.btn_zoom_100.blockSignals(False)

    def set_clipping_checked(self, checked: bool) -> None:
        if hasattr(self, 'btn_clipping'):
            self.btn_clipping.blockSignals(True)
            self.btn_clipping.setChecked(checked)
            self.btn_clipping.blockSignals(False)

    def set_magnifier_checked(self, checked: bool) -> None:
        if hasattr(self, 'btn_magnifier'):
            self.btn_magnifier.blockSignals(True)
            self.btn_magnifier.setChecked(checked)
            self.btn_magnifier.blockSignals(False)

    def set_composite_checked(self, checked: bool) -> None:
        if hasattr(self, 'btn_composite'):
            self.btn_composite.blockSignals(True)
            self.btn_composite.setChecked(checked)
            self.btn_composite.blockSignals(False)

    def set_composite_enabled(self, enabled: bool) -> None:
        if hasattr(self, 'btn_composite'):
            self.btn_composite.setEnabled(enabled)

    def connect_canvas(self, canvas: QWidget) -> None:
        """Conecta automáticamente los controles de la barra a un lienzo compatible."""
        if hasattr(self, 'btn_zoom_100'):
            self.zoom_100_toggled.connect(
                lambda chk: canvas.set_zoom_100(chk) if hasattr(canvas, 'set_zoom_100') else None
            )
        if hasattr(self, 'btn_zoom_fit'):
            self.zoom_fit_clicked.connect(
                lambda: canvas.fit_view() if hasattr(canvas, 'fit_view') else (
                    canvas.reset_view() if hasattr(canvas, 'reset_view') else None
                )
            )
        if hasattr(canvas, 'zoom_toggled') and hasattr(self, 'btn_zoom_100'):
            canvas.zoom_toggled.connect(lambda is_z, rx, ry: self.set_zoom_100_checked(is_z))
        if hasattr(self, 'btn_clipping') and hasattr(canvas, 'set_clipping_state'):
            self.clipping_toggled.connect(lambda chk: canvas.set_clipping_state(chk, chk))
        if hasattr(self, 'btn_magnifier') and hasattr(canvas, 'set_magnifier_active'):
            self.magnifier_toggled.connect(canvas.set_magnifier_active)
        if hasattr(canvas, 'magnifier_toggled') and hasattr(self, 'btn_magnifier'):
            canvas.magnifier_toggled.connect(self.set_magnifier_checked)

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QWidget#canvas_toolbar {{
                background-color: {BG_PANEL};
                border-bottom: 1px solid {BORDER_BASE};
            }}
            QPushButton {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                border-radius: 4px;
                padding: 3px 9px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #282e42;
                border-color: {TEXT_CYAN};
            }}
            QPushButton:checked {{
                background-color: #0d47a1;
                color: #ffffff;
                border-color: {ACCENT_CYAN};
            }}
            QPushButton:disabled {{
                background-color: #141620;
                color: {TEXT_MUTED};
                border-color: {BORDER_BASE};
            }}
            QComboBox {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                border-radius: 4px;
                padding: 2px 8px;
                color: {TEXT_PRIMARY};
                font-size: 11px;
            }}
            QComboBox:hover {{
                border-color: {TEXT_CYAN};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 18px;
            }}
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
            }}
        """)


class CanvasFooterBar(QWidget):
    """
    Barra inferior de estado, modo de herramienta, barra de progreso compacta
    (estilo Photoshop) y metadatos del lienzo reutilizable para los visores de AstroLoom.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("canvas_footer_bar")
        self.setFixedHeight(26)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)

        # 1. Indicador de Herramienta Activa (Pill badge)
        self.lbl_tool_badge = QLabel("● Modo: Vista General")
        self.lbl_tool_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        layout.addWidget(self.lbl_tool_badge)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet(f"color: {BORDER_BASE};")
        layout.addWidget(sep1)

        # 2. Información de Zoom
        self.lbl_canvas_zoom = QLabel("Zoom: Ajustar (Fit)")
        self.lbl_canvas_zoom.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_canvas_zoom)

        # Espaciador elástico
        layout.addStretch(1)

        # 3. Barra de Progreso Compacta Estilo Photoshop (oculta por defecto)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setFixedWidth(210)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #232738;
                border-radius: 4px;
                background-color: #0b0f19;
                text-align: center;
                color: #e2e8f0;
                font-size: 10px;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 4. Metadatos de la Imagen (Resolución, MP, Profundidad, Espacio de Color)
        self.lbl_canvas_info = QLabel("Sin imagen cargada")
        self.lbl_canvas_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_canvas_info)

    def set_tool_badge(self, text: str, color: str = "#38bdf8") -> None:
        """Actualiza el texto y color del indicador de modo de herramienta activo."""
        self.lbl_tool_badge.setText(text)
        self.lbl_tool_badge.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")

    def set_zoom_text(self, text: str) -> None:
        """Actualiza la etiqueta de nivel de zoom."""
        self.lbl_canvas_zoom.setText(text)

    def set_info_text(self, text: str) -> None:
        """Actualiza directamente la etiqueta de información técnica del lienzo."""
        self.lbl_canvas_info.setText(text)

    def set_image_info(self, width: int, height: int, depth_str: str = "32-bit Float", color_space: str = "sRGB") -> None:
        """Formatea y actualiza los metadatos de dimensiones, MP, profundidad y espacio de color."""
        mp = (width * height) / 1e6
        self.lbl_canvas_info.setText(f"{width} × {height} px ({mp:.1f} MP)  |  {depth_str}  |  {color_space}")

    def set_progress(self, percent: int, text: str = "") -> None:
        """Muestra y actualiza el valor y texto de la barra de progreso compacta."""
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        val = max(0, min(100, int(percent)))
        self.progress_bar.setValue(val)
        if text:
            if "%p%" in text or "%" in text:
                self.progress_bar.setFormat(text)
            else:
                self.progress_bar.setFormat(f"{text} (%p%)")
        else:
            self.progress_bar.setFormat("%p%")
        QCoreApplication.processEvents()

    def hide_progress(self) -> None:
        """Oculta y reinicia la barra de progreso compacta."""
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        QCoreApplication.processEvents()

