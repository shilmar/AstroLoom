# gui/developer/canvas_tools.py
from __future__ import annotations
import os
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QCoreApplication, QTimer, QPoint, QEvent, QRectF
from PySide6.QtGui import QPixmap, QImage, QColor, QCursor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox, QSlider, QSplitter,
    QDoubleSpinBox, QLineEdit, QComboBox, QSpinBox, QFrame,
    QButtonGroup, QProgressDialog, QCheckBox, QSizePolicy, QMenu,
    QPlainTextEdit, QTextEdit
)

from gui.icons import get_icon, get_svg_pixmap
from gui.theme import (
    STYLE_RUN_IDLE, STYLE_TRANSFER_BTN, BG_PANEL, BORDER_BASE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_CYAN, ACCENT_CYAN
)
from gui.widgets.viewer_toolbar import CanvasFooterBar
from gui.widgets.precision_slider import PrecisionSliderRow
from gui.developer.panels.panel_crop import CropToolbar
from core.log_hub import log_hub
from core.i18n import tr
from core.config_manager import load_config
from core.stacking import load_image_as_float32
from core.masking import refine_mask_guided
from core.stretch import apply_ghs_stretch, manual_stretch
from core.mask_layers import (
    MaskLayer,
    MASK_TYPE_IMAGE_BASE, MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_STARS,
    MASK_TYPE_IMAGE_GROUND, MASK_TYPE_BRUSH, MASK_TYPE_LUMA, MASK_TYPE_COLOR,
    MASK_TYPE_GROUND, MASK_TYPE_SKY, MASK_TYPE_IMAGE_RASTER,
    apply_pre_curve_adjustments, apply_local_adjustments_to_image,
    compute_effective_layer_mask, get_canvas_below_layer
)


class CanvasToolsMixin:
    """Mixin para barras de herramientas del lienzo, recorte, máscaras de pincel, pipetas y navegación."""

    def _create_canvas_toolbar(self) -> QWidget:
        """Crea la barra de herramientas superior del visor interactivo."""
        toolbar = QWidget()
        toolbar.setObjectName("canvas_toolbar")
        toolbar.setFixedHeight(38)

        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 3, 8, 3)
        tb_layout.setSpacing(6)

        # 0. Botón Ocultar/Mostrar Panel Izquierdo (Lightroom F7 / [)
        self.btn_toggle_left_panel = QPushButton()
        self.btn_toggle_left_panel.setIcon(get_icon("panel_left", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=14))
        self.btn_toggle_left_panel.setCheckable(True)
        self.btn_toggle_left_panel.setChecked(True)
        self.btn_toggle_left_panel.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_left_panel.setToolTip("Ocultar / Mostrar panel lateral izquierdo de ajustes (Atajo: F7 o [)")
        self.btn_toggle_left_panel.setFixedSize(30, 26)
        self.btn_toggle_left_panel.toggled.connect(self.toggle_left_panel)
        tb_layout.addWidget(self.btn_toggle_left_panel)

        sep_left = QFrame()
        sep_left.setFrameShape(QFrame.VLine)
        sep_left.setFrameShadow(QFrame.Sunken)
        sep_left.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep_left)

        # 0a. Botón Deshacer (Undo)
        self.btn_undo = QPushButton()
        self.btn_undo.setIcon(get_icon("undo", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_undo.setCursor(Qt.PointingHandCursor)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("Deshace el último ajuste de revelado (Atajo: Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        tb_layout.addWidget(self.btn_undo)

        # 0b. Botón Rehacer (Redo)
        self.btn_redo = QPushButton()
        self.btn_redo.setIcon(get_icon("redo", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_redo.setCursor(Qt.PointingHandCursor)
        self.btn_redo.setEnabled(False)
        self.btn_redo.setToolTip("Rehace el ajuste deshecho (Atajo: Ctrl+Y o Ctrl+Shift+Z)")
        self.btn_redo.clicked.connect(self.redo)
        tb_layout.addWidget(self.btn_redo)

        # Separador vertical de historial
        sep_undo = QFrame()
        sep_undo.setFrameShape(QFrame.VLine)
        sep_undo.setFrameShadow(QFrame.Sunken)
        sep_undo.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep_undo)

        # 1. Botón Comparador Antes / Después
        self.btn_split_view = QPushButton("Comparar")
        self.btn_split_view.setIcon(get_icon("split", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_split_view.setCheckable(True)
        self.btn_split_view.setCursor(Qt.PointingHandCursor)
        self.btn_split_view.setToolTip(
            "Activa la cortina divisoria interactiva para comparar el estado original ('Antes')\n"
            "con el revelado actual ('Después'). Puedes arrastrar la línea con el ratón o hacer doble clic para centrar.\n"
            "Atajo de teclado: Y"
        )
        self.btn_split_view.toggled.connect(self.on_split_view_toggled)
        tb_layout.addWidget(self.btn_split_view)

        # 1b. Botón Orientación Cortina (Vertical / Horizontal)
        self.btn_split_orientation = QPushButton("Vertical")
        self.btn_split_orientation.setIcon(get_icon("sliders", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_split_orientation.setCursor(Qt.PointingHandCursor)
        self.btn_split_orientation.setEnabled(False)
        self.btn_split_orientation.setToolTip(
            "Alterna la orientación de la cortina divisoria:\n"
            "• Vertical: Izquierda ('Antes') / Derecha ('Después')\n"
            "• Horizontal: Arriba ('Antes') / Abajo ('Después')\n"
            "Atajo de teclado: Shift+Y"
        )
        self.btn_split_orientation.clicked.connect(self.on_toggle_split_orientation)
        tb_layout.addWidget(self.btn_split_orientation)

        # 2. Botón Fijar como Antes (Snapshot)
        self.btn_set_before = QPushButton("Fijar ref.")
        self.btn_set_before.setIcon(get_icon("camera", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_set_before.setCursor(Qt.PointingHandCursor)
        self.btn_set_before.setToolTip(
            "Congela el procesado actual como la nueva referencia de 'Antes'\n"
            "para evaluar cambios incrementales sucesivos."
        )
        self.btn_set_before.clicked.connect(self.on_set_before_clicked)
        tb_layout.addWidget(self.btn_set_before)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        sep1.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep1)

        # 3. Botón Indicador de Recorte (Clipping)
        self.btn_toggle_clipping = QPushButton("Aviso recorte")
        self.btn_toggle_clipping.setIcon(get_icon("alert", color="#94a3b8", hover_color="#ffffff", active_color="#f87171", size=13))
        self.btn_toggle_clipping.setCheckable(True)
        self.btn_toggle_clipping.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_clipping.setToolTip(
            "Muestra en el lienzo los píxeles con recorte: Azul para negros empastados (RGB=0)\n"
            "y Rojo para luces quemadas (RGB=255).\n"
            "Atajo de teclado: J"
        )
        self.btn_toggle_clipping.toggled.connect(self.on_toolbar_clipping_toggled)
        tb_layout.addWidget(self.btn_toggle_clipping)

        # Separador vertical
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep2)

        # 4. Botón Zoom 1:1
        self.btn_toggle_zoom = QPushButton("1:1")
        self.btn_toggle_zoom.setIcon(get_icon("zoom_in", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_toggle_zoom.setCheckable(True)
        self.btn_toggle_zoom.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_zoom.setToolTip(
            "Alterna entre vista adaptada completa y ampliación 1:1 al píxel nativo del sensor.\n"
            "También puedes hacer clic derecho directamente sobre el lienzo."
        )
        self.btn_toggle_zoom.toggled.connect(self.on_toolbar_zoom_toggled)
        tb_layout.addWidget(self.btn_toggle_zoom)

        # Separador vertical
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        sep3.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep3)

        # 5. Lupa de Inspección Flotante
        self.btn_magnifier = QPushButton()
        self.btn_magnifier.setIcon(get_icon("search", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_magnifier.setCheckable(True)
        self.btn_magnifier.setCursor(Qt.PointingHandCursor)
        self.btn_magnifier.setToolTip(
            "Activa la lupa de inspección flotante sobre el cursor para examinar enfoque, estrellas y ruido a nivel de píxel.\n"
            "Puedes regular el aumento usando la rueda del ratón.\n"
            "Atajo de teclado: Z"
        )
        self.btn_magnifier.toggled.connect(self.canvas.set_magnifier_active)
        tb_layout.addWidget(self.btn_magnifier)

        # Separador vertical
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.VLine)
        sep4.setFrameShadow(QFrame.Sunken)
        sep4.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep4)

        # 7. Botón Recorte Interactivo (Crop)
        self.btn_crop = QPushButton()
        self.btn_crop.setIcon(get_icon("crop", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_crop.setCheckable(True)
        self.btn_crop.setCursor(Qt.PointingHandCursor)
        self.btn_crop.setToolTip(
            "Activa la herramienta interactiva de recorte, composición y nivelación de horizonte.\n"
            "Permite ajustar proporciones (1:1, 16:9, 3:2, etc.), rotar y recomponer con guías.\n"
            "Atajo de teclado: C"
        )
        self.btn_crop.toggled.connect(self._on_toggle_crop)
        tb_layout.addWidget(self.btn_crop)

        # Separador vertical
        sep5 = QFrame()
        sep5.setFrameShape(QFrame.VLine)
        sep5.setFrameShadow(QFrame.Sunken)
        sep5.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep5)

        # 8. Botón Máscara Suelo Interactivo
        self.btn_mask_tool = QPushButton("Máscara")
        self.btn_mask_tool.setIcon(get_icon("brush", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=13))
        self.btn_mask_tool.setCheckable(True)
        self.btn_mask_tool.setCursor(Qt.PointingHandCursor)
        self.btn_mask_tool.setToolTip(
            "Activa la herramienta interactiva de trazado y segmentación inteligente de cielo y suelo.\n"
            "Dibuja trazos verdes sobre el cielo y rojos sobre el suelo para generar la máscara guiada.\n"
            "Atajo de teclado: M"
        )
        self.btn_mask_tool.toggled.connect(self._on_toggle_mask_tool)
        tb_layout.addWidget(self.btn_mask_tool)

        # Espaciador elástico
        tb_layout.addStretch()

        # Separador vertical
        sep_right = QFrame()
        sep_right.setFrameShape(QFrame.VLine)
        sep_right.setFrameShadow(QFrame.Sunken)
        sep_right.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep_right)

        # Botón Ocultar/Mostrar Panel Derecho (Lightroom F8 / ])
        self.btn_toggle_right_panel = QPushButton()
        self.btn_toggle_right_panel.setIcon(get_icon("panel_right", color="#94a3b8", hover_color="#ffffff", active_color="#38bdf8", size=14))
        self.btn_toggle_right_panel.setCheckable(True)
        self.btn_toggle_right_panel.setChecked(True)
        self.btn_toggle_right_panel.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_right_panel.setToolTip("Ocultar / Mostrar panel lateral derecho de capas e histograma (Atajo: F8 o ])")
        self.btn_toggle_right_panel.setFixedSize(30, 26)
        self.btn_toggle_right_panel.toggled.connect(self.toggle_right_panel)
        tb_layout.addWidget(self.btn_toggle_right_panel)

        return toolbar

    def _create_canvas_footer_bar(self) -> CanvasFooterBar:
        """Crea la barra inferior de estado, modo de herramienta activa, progreso y metadatos del lienzo."""
        footer = CanvasFooterBar(self)
        self.lbl_tool_badge = footer.lbl_tool_badge
        self.lbl_canvas_zoom = footer.lbl_canvas_zoom
        self.lbl_canvas_info = footer.lbl_canvas_info
        self.canvas_progress = footer.progress_bar
        return footer

    def set_canvas_progress(self, percent: int, text: str = "") -> None:
        """Actualiza la barra de progreso compacta en el pie del lienzo estilo Photoshop."""
        if hasattr(self, 'canvas_footer') and self.canvas_footer is not None:
            self.canvas_footer.set_progress(percent, text)

    def hide_canvas_progress(self) -> None:
        """Oculta la barra de progreso compacta del lienzo."""
        if hasattr(self, 'canvas_footer') and self.canvas_footer is not None:
            self.canvas_footer.hide_progress()

    def _create_crop_toolbar(self) -> QWidget:
        """Crea la barra de herramientas interactiva de recorte y nivelación de horizonte."""
        toolbar = CropToolbar(self, resettable_label_factory=self._make_resettable_label)
        toolbar.setObjectName("crop_toolbar")
        toolbar.setFixedHeight(40)

        self.combo_crop_ratio = toolbar.combo_crop_ratio
        self.btn_crop_flip_ratio = toolbar.btn_crop_flip_ratio
        self.slider_crop_angle = toolbar.slider_crop_angle
        self.lbl_crop_angle_val = toolbar.lbl_crop_angle_val
        self.btn_crop_reset_angle = toolbar.btn_crop_reset_angle
        self.combo_crop_guides = toolbar.combo_crop_guides
        self.lbl_crop_dimensions = toolbar.lbl_crop_dimensions
        self.btn_crop_reset = toolbar.btn_crop_reset
        self.btn_crop_cancel = toolbar.btn_crop_cancel
        self.btn_crop_apply = toolbar.btn_crop_apply

        toolbar.ratio_changed.connect(self._on_crop_ratio_changed)
        toolbar.flip_ratio_clicked.connect(self._on_crop_flip_ratio)
        toolbar.angle_changed.connect(self._on_crop_angle_changed)
        toolbar.guides_changed.connect(self._on_crop_guides_changed)
        toolbar.reset_clicked.connect(self._reset_crop)
        toolbar.cancel_clicked.connect(self._cancel_crop)
        toolbar.apply_clicked.connect(self._apply_crop)

        return toolbar

    def _on_toggle_crop_shortcut(self) -> None:
        """Alterna el modo de recorte con la tecla 'C' ignorando entradas de texto activas."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_crop'):
            self.btn_crop.toggle()

    def _on_toggle_crop(self, enabled: bool) -> None:
        """Activa o desactiva la barra y modo de recorte interactivo en el lienzo."""
        if enabled and self.image_32bit is None:
            self.btn_crop.setChecked(False)
            self.log_message("[AVISO] Debe cargar una imagen antes de utilizar el recorte.")
            return

        if enabled:
            if self.is_zoomed_100:
                self.btn_toggle_zoom.setChecked(False)
            if self.canvas.eyedropper_active:
                self.btn_pipette.setChecked(False)
            if self.canvas.magnifier_active:
                self.btn_magnifier.setChecked(False)

        self.canvas.set_crop_mode(enabled)
        self.crop_toolbar.setVisible(enabled)

        if enabled:
            h, w = self.image_32bit.shape[:2]
            self.canvas.set_native_image_size((w, h))
            self.canvas._emit_crop_size()
            self.log_message("[RECORTE] Modo Recorte activado. Ajusta el marco, nivela el horizonte o presiona Intro para aplicar.")
        else:
            self.canvas.reset_crop_rect()
            if hasattr(self, 'slider_crop_angle'):
                self.slider_crop_angle.setValue(0)
        self._update_canvas_info_label()

    def _on_crop_ratio_changed(self, idx: int) -> None:
        """Ajusta la relación de aspecto forzada en el lienzo según la selección del combo."""
        ratio_map = {
            0: None,
            1: (self.image_32bit.shape[1] / float(self.image_32bit.shape[0])) if self.image_32bit is not None else None,
            2: 1.0,
            3: 16.0 / 9.0,
            4: 3.0 / 2.0,
            5: 4.0 / 3.0,
            6: 5.0 / 4.0,
            7: 2.0,
            8: 9.0 / 16.0,
            9: 2.0 / 3.0,
            10: 4.0 / 5.0,
        }
        ratio = ratio_map.get(idx, None)
        self.canvas.set_crop_aspect_ratio(ratio)

    def _on_crop_flip_ratio(self) -> None:
        """Invierte la orientación de la proporción actual (Horizontal ⇄ Vertical)."""
        idx = self.combo_crop_ratio.currentIndex()
        flip_pairs = {3: 8, 8: 3, 4: 9, 9: 4, 5: 10, 10: 5}
        if idx in flip_pairs:
            self.combo_crop_ratio.setCurrentIndex(flip_pairs[idx])
            return

        cur_ratio = self.canvas.crop_aspect_ratio
        if cur_ratio is not None and cur_ratio > 0:
            self.canvas.set_crop_aspect_ratio(1.0 / cur_ratio)
        else:
            r = self.canvas.crop_rect_norm
            cx, cy = r.center().x(), r.center().y()
            ih = float(self.image_32bit.shape[0]) if self.image_32bit is not None else 1.0
            iw = float(self.image_32bit.shape[1]) if self.image_32bit is not None else 1.0
            new_w = min(1.0, r.height() * (ih / iw))
            new_h = min(1.0, r.width() * (iw / ih))
            new_x = max(0.0, min(1.0 - new_w, cx - new_w / 2.0))
            new_y = max(0.0, min(1.0 - new_h, cy - new_h / 2.0))
            from PySide6.QtCore import QRectF
            self.canvas.crop_rect_norm = QRectF(new_x, new_y, new_w, new_h)
            self.canvas._emit_crop_size()
            self.canvas.update()

    def _on_crop_angle_changed(self, val: int) -> None:
        """Actualiza el ángulo de nivelación en tiempo real."""
        angle = val / 10.0
        self.lbl_crop_angle_val.setText(f"{angle:+.1f}°" if abs(angle) > 1e-2 else "0.0°")
        self.canvas.set_crop_angle(angle)

    def _on_crop_guides_changed(self, idx: int) -> None:
        """Alterna las guías de composición."""
        guides = ["thirds", "grid", "none"][idx] if idx < 3 else "thirds"
        self.canvas.set_crop_guides(guides)

    def _on_crop_rect_changed(self, px_w: int, px_h: int) -> None:
        """Muestra la resolución resultante en la barra de herramientas."""
        mp = (px_w * px_h) / 1e6
        self.lbl_crop_dimensions.setText(f"{px_w} × {px_h} px ({mp:.1f} MP)")

    def _reset_crop(self) -> None:
        """Restablece el marco de recorte al 100% y el ángulo a 0°."""
        if hasattr(self, 'slider_crop_angle'):
            self.slider_crop_angle.setValue(0)
        if hasattr(self, 'combo_crop_ratio'):
            self.combo_crop_ratio.setCurrentIndex(0)
        self.canvas.reset_crop_rect()

    def _cancel_crop(self) -> None:
        """Cancela el modo de recorte sin aplicar ningún cambio."""
        self.btn_crop.setChecked(False)
        self.log_message("Recorte cancelado.")

    def _apply_crop(self) -> None:
        """Aplica permanentemente el recorte y la nivelación a la imagen en memoria con Undo/Redo."""
        if self.image_32bit is None:
            self.btn_crop.setChecked(False)
            return

        orig_h, orig_w = self.image_32bit.shape[:2]
        angle = self.canvas.crop_angle
        norm = self.canvas.crop_rect_norm

        no_angle = abs(angle) < 1e-3
        full_frame = (norm.x() <= 0.002 and norm.y() <= 0.002 and 
                      norm.width() >= 0.996 and norm.height() >= 0.996)
        if no_angle and full_frame:
            self.log_message("El marco abarca el 100% de la toma sin rotación. Sin cambios.")
            self.btn_crop.setChecked(False)
            return

        # Registrar estado previo en la pila de deshacer (Undo)
        undo_item = {
            "type": "crop",
            "image_32bit": self.image_32bit.copy(),
            "image_starless": self.image_starless.copy() if self.image_starless is not None else None,
            "raw_linear_starless": self.raw_linear_starless.copy() if getattr(self, "raw_linear_starless", None) is not None else None,
            "starless_is_stretched": getattr(self, "starless_is_stretched", False),
            "image_stars": self.image_stars.copy() if self.image_stars is not None else None,
            "current_mask": self.current_mask.copy() if self.current_mask is not None else None,
            "_cache_starless_ai": self._cache_starless_ai.copy() if getattr(self, "_cache_starless_ai", None) is not None else None,
            "_cache_ground_ai": self._cache_ground_ai.copy() if getattr(self, "_cache_ground_ai", None) is not None else None,
            "mask_layers": [l.shallow_snapshot() for l in self.mask_layers_widget.layers] if hasattr(self, 'mask_layers_widget') else [],
        }
        self.undo_stack.append(undo_item)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

        # Transformación rotación y corte
        def rotate_and_crop(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
            h, w = arr.shape[:2]
            if not no_angle:
                center = (w / 2.0, h / 2.0)
                rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
                interp = cv2.INTER_LINEAR if is_mask else cv2.INTER_LANCZOS4
                arr = cv2.warpAffine(arr, rot_mat, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT)

            px_x = int(round(norm.x() * w))
            px_y = int(round(norm.y() * h))
            px_w = int(round(norm.width() * w))
            px_h = int(round(norm.height() * h))
            px_x = max(0, min(w - 1, px_x))
            px_y = max(0, min(h - 1, px_y))
            px_w = max(10, min(w - px_x, px_w))
            px_h = max(10, min(h - px_y, px_h))
            return np.ascontiguousarray(arr[px_y:px_y + px_h, px_x:px_x + px_w], dtype=np.float32)

        self.image_32bit = rotate_and_crop(self.image_32bit)
        if self.image_starless is not None:
            self.image_starless = rotate_and_crop(self.image_starless)
        if getattr(self, "raw_linear_starless", None) is not None:
            self.raw_linear_starless = rotate_and_crop(self.raw_linear_starless)
        if self.image_stars is not None:
            self.image_stars = rotate_and_crop(self.image_stars)
        if self.current_mask is not None:
            self.current_mask = rotate_and_crop(self.current_mask, is_mask=True)
            self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
        if getattr(self, "_cache_starless_ai", None) is not None:
            if self._cache_starless_ai.shape[:2] == (orig_h, orig_w):
                self._cache_starless_ai = rotate_and_crop(self._cache_starless_ai)
            else:
                self._cache_starless_ai = None
        if getattr(self, "_cache_ground_ai", None) is not None:
            if self._cache_ground_ai.shape[:2] == (orig_h, orig_w):
                self._cache_ground_ai = rotate_and_crop(self._cache_ground_ai)
            else:
                self._cache_ground_ai = None

        if hasattr(self, 'mask_layers_widget') and self.mask_layers_widget.layers:
            for l in self.mask_layers_widget.layers:
                if getattr(l, 'brush_mask', None) is not None:
                    if l.brush_mask.shape[:2] == (orig_h, orig_w):
                        l.brush_mask = rotate_and_crop(l.brush_mask, is_mask=True)
                        l.invalidate_brush_proxies()
                    else:
                        l.set_brush_mask(None)
            self.mask_layers_widget.invalidate_layer_thumbnail(None, rebuild_now=True)
            self.mask_layers_widget.update_active_layer_histogram()

        self.btn_crop.setChecked(False)
        if hasattr(self, 'slider_crop_angle'):
            self.slider_crop_angle.setValue(0)
        self.canvas.reset_crop_rect()

        new_h, new_w = self.image_32bit.shape[:2]
        self.canvas.set_native_image_size((new_w, new_h))
        self._generate_preview_proxy()
        self.snapshot_before_pixmap = None
        self.has_custom_snapshot = False
        self._generate_before_reference()
        self.update_stretch_preview()
        self._update_canvas_info_label()
        self._last_committed_state = self._capture_state()
        self._update_undo_redo_buttons()

        ang_info = f" con rotación {angle:+.1f}°" if not no_angle else ""
        self.log_message(f"[RECORTE] Recorte aplicado{ang_info}: {orig_w}x{orig_h} → {new_w}x{new_h} px. (Deshacer con Ctrl+Z)")

    def _create_mask_toolbar(self) -> QWidget:
        """Crea la barra de herramientas interactiva para dibujar semillas y refinar la máscara de suelo."""
        toolbar = QWidget()
        toolbar.setObjectName("mask_toolbar")
        toolbar.setFixedHeight(40)

        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 3, 10, 3)
        tb_layout.setSpacing(8)

        # 1. Título / Icono
        self.lbl_mask_tb_title = QLabel("Máscara:")
        self.lbl_mask_tb_title.setStyleSheet("font-weight: bold; color: #80d8ff; font-size: 11px;")
        tb_layout.addWidget(self.lbl_mask_tb_title)

        # 2. Selector de Clase de Pincel (Cielo vs Suelo)
        self.btn_brush_sky = QPushButton("● Cielo")
        self.btn_brush_sky.setCheckable(True)
        self.btn_brush_sky.setChecked(True)
        self.btn_brush_sky.setCursor(Qt.PointingHandCursor)
        self.btn_brush_sky.setStyleSheet("""
            QPushButton {
                background-color: #1a2e22;
                color: #a5d6a7;
                border: 1px solid #2e7d32;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2e7d32;
                color: #ffffff;
                border-color: #66bb6a;
            }
        """)
        self.btn_brush_sky.setToolTip("Pinta semillas de CIELO (trazo verde)")
        tb_layout.addWidget(self.btn_brush_sky)

        self.btn_brush_gnd = QPushButton("● Suelo")
        self.btn_brush_gnd.setCheckable(True)
        self.btn_brush_gnd.setCursor(Qt.PointingHandCursor)
        self.btn_brush_gnd.setStyleSheet("""
            QPushButton {
                background-color: #2e1a1a;
                color: #ef9a9a;
                border: 1px solid #c62828;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #c62828;
                color: #ffffff;
                border-color: #ef5350;
            }
        """)
        self.btn_brush_gnd.setToolTip("Pinta semillas de SUELO (trazo rojo)")
        tb_layout.addWidget(self.btn_brush_gnd)

        # Grupo exclusivo para los botones de pincel
        self.brush_group = QButtonGroup(self)
        self.brush_group.addButton(self.btn_brush_sky, 2)
        self.brush_group.addButton(self.btn_brush_gnd, 1)
        self.brush_group.idClicked.connect(self._on_brush_mode_changed)

        # Separador vertical
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        sep1.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep1)

        # 3. Radio del pincel
        lbl_r = self._make_resettable_label(QLabel("Pincel:"), lambda: self.slider_mask_brush.setValue(20), "20 px")
        lbl_r.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_r)

        self.slider_mask_brush = QSlider(Qt.Horizontal)
        self.slider_mask_brush.setRange(5, 120)
        self.slider_mask_brush.setValue(20)
        self.slider_mask_brush.setFixedWidth(90)
        self.slider_mask_brush.setToolTip("Tamaño del pincel en píxeles de pantalla (o usa la rueda del ratón)")
        self.slider_mask_brush.valueChanged.connect(self._on_slider_brush_changed)
        tb_layout.addWidget(self.slider_mask_brush)

        self.lbl_mask_brush_val = QLabel("20 px")
        self.lbl_mask_brush_val.setStyleSheet("color: #88c0d0; font-weight: bold; min-width: 38px; font-size: 11px;")
        tb_layout.addWidget(self.lbl_mask_brush_val)

        # Separador vertical
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        sep2.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        tb_layout.addWidget(sep2)

        # 4. Parámetros de refinamiento (Feather e Iterations)
        lbl_f = QLabel("Borde:")
        lbl_f.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_f)

        self.spin_mask_feather = QSpinBox()
        self.spin_mask_feather.setRange(1, 45)
        self.spin_mask_feather.setValue(7)
        self.spin_mask_feather.setSuffix(" px")
        self.spin_mask_feather.setFixedWidth(66)
        self.spin_mask_feather.setToolTip("Radio de suavizado / transición gaussiana en el perímetro del horizonte")
        tb_layout.addWidget(self.spin_mask_feather)

        lbl_it = QLabel("Pasos:")
        lbl_it.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_it)

        self.spin_mask_iters = QSpinBox()
        self.spin_mask_iters.setRange(1, 10)
        self.spin_mask_iters.setValue(3)
        self.spin_mask_iters.setFixedWidth(46)
        self.spin_mask_iters.setToolTip("Número de iteraciones del algoritmo de segmentación GrabCut")
        tb_layout.addWidget(self.spin_mask_iters)

        lbl_sh = QLabel("Desplazar:")
        lbl_sh.setStyleSheet("color: #c0caf5; font-size: 11px;")
        tb_layout.addWidget(lbl_sh)

        self.spin_mask_shift = QSpinBox()
        self.spin_mask_shift.setRange(-20, 20)
        self.spin_mask_shift.setValue(0)
        self.spin_mask_shift.setSuffix(" px")
        self.spin_mask_shift.setFixedWidth(66)
        self.spin_mask_shift.setToolTip("Desplaza la frontera de la máscara: negativo contrae el cielo hacia el suelo para tapar halos; positivo lo expande")
        tb_layout.addWidget(self.spin_mask_shift)

        # Separador vertical
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        sep3.setStyleSheet("color: #3b4252; margin: 4px 2px;")
        # 5a. Detección Automática con IA Ligera
        self.btn_ai_mask = QPushButton("Máscara IA")
        self.btn_ai_mask.setIcon(get_icon("sparkles", color="#ffffff", size=13))
        self.btn_ai_mask.setCursor(Qt.PointingHandCursor)
        self.btn_ai_mask.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6a1b9a, stop:1 #8e24aa);
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #ba68c8;
                border-radius: 4px;
                padding: 3px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b1fa2, stop:1 #9c27b0);
                border-color: #ce93d8;
            }
            QPushButton:pressed {
                background-color: #4a148c;
            }
        """)
        self.btn_ai_mask.setToolTip("Detecta automáticamente el horizonte y suelo mediante IA ligera U²-Net con 1 solo clic")
        self.btn_ai_mask.clicked.connect(self._detect_mask_ai_from_toolbar)
        tb_layout.addWidget(self.btn_ai_mask)

        # 5b. Generar Máscara desde trazos
        self.btn_generate_mask = QPushButton("Generar")
        self.btn_generate_mask.setIcon(get_icon("check", color="#ffffff", size=13))
        self.btn_generate_mask.setCursor(Qt.PointingHandCursor)
        self.btn_generate_mask.setStyleSheet("""
            QPushButton {
                background-color: #00796b;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #26a69a;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00897b;
                border-color: #4db6ac;
            }
            QPushButton:pressed {
                background-color: #004d40;
            }
        """)
        self.btn_generate_mask.setToolTip("Calcula la máscara optimizada con GrabCut asistido por luminancia y preservación de siluetas finas (Intro)")
        self.btn_generate_mask.clicked.connect(self._compute_mask_from_scribbles)
        tb_layout.addWidget(self.btn_generate_mask)

        # 5c. Cargar Máscara (del Apilado o desde Archivo)
        self.btn_load_mask = QPushButton("Cargar...")
        self.btn_load_mask.setIcon(get_icon("folder", color="#80d8ff", size=13))
        self.btn_load_mask.setCursor(Qt.PointingHandCursor)
        self.btn_load_mask.setStyleSheet("""
            QPushButton {
                background-color: #1a2736;
                color: #80d8ff;
                font-weight: bold;
                border: 1px solid #0288d1;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0288d1;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #01579b;
            }
        """)
        self.btn_load_mask.setToolTip("Carga una máscara generada previamente durante el apilado o desde un archivo (.png, .tif, .fits)")
        self.btn_load_mask.clicked.connect(self._load_mask_from_toolbar)
        tb_layout.addWidget(self.btn_load_mask)

        # 6. Ver / Ocultar Máscara Overlay
        self.btn_mask_overlay_toggle = QPushButton("Ver")
        self.btn_mask_overlay_toggle.setIcon(get_icon("eye", color="#94a3b8", active_color="#4ade80", size=13))
        self.btn_mask_overlay_toggle.setCheckable(True)
        self.btn_mask_overlay_toggle.setChecked(False)
        self.btn_mask_overlay_toggle.setEnabled(False)
        self.btn_mask_overlay_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_mask_overlay_toggle.setStyleSheet("""
            QPushButton {
                background-color: #262933;
                color: #a0a0a0;
                border: 1px solid #3b4252;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #88c0d0;
                color: #eceff4;
            }
            QPushButton:disabled {
                color: #4c566a;
                background-color: #1a1c23;
                border: 1px solid #2a2e39;
            }
        """)
        self.btn_mask_overlay_toggle.setToolTip("Muestra u oculta la superposición semitransparente (Verde=Cielo, Rojo=Suelo)")
        self.btn_mask_overlay_toggle.toggled.connect(self._on_top_mask_overlay_toggled)
        tb_layout.addWidget(self.btn_mask_overlay_toggle)

        # 7. Limpiar Trazos
        self.btn_clear_scribbles = QPushButton("Trazos")
        self.btn_clear_scribbles.setIcon(get_icon("trash", color="#94a3b8", size=13))
        self.btn_clear_scribbles.setCursor(Qt.PointingHandCursor)
        self.btn_clear_scribbles.setToolTip("Borra los trazos dibujados sin eliminar la máscara actual calculada")
        self.btn_clear_scribbles.clicked.connect(self._clear_scribbles)
        tb_layout.addWidget(self.btn_clear_scribbles)

        # 8. Quitar / Eliminar Máscara
        self.btn_delete_mask = QPushButton("Quitar")
        self.btn_delete_mask.setIcon(get_icon("x", color="#f87171", size=13))
        self.btn_delete_mask.setCursor(Qt.PointingHandCursor)
        self.btn_delete_mask.setToolTip("Elimina la máscara activa por completo de la imagen")
        self.btn_delete_mask.clicked.connect(self._delete_mask)
        tb_layout.addWidget(self.btn_delete_mask)

        tb_layout.addStretch()

        # 9. Botón Finalizar / Cerrar
        self.btn_mask_done = QPushButton("Listo")
        self.btn_mask_done.setIcon(get_icon("check", color="#ffffff", size=13))
        self.btn_mask_done.setCursor(Qt.PointingHandCursor)
        self.btn_mask_done.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #4caf50;
                border-radius: 4px;
                padding: 3px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #388e3c;
                border-color: #81c784;
            }
        """)
        self.btn_mask_done.setToolTip("Cierra el modo de trazado de máscara y vuelve al visor estándar (Esc)")
        self.btn_mask_done.clicked.connect(lambda: self.btn_mask_tool.setChecked(False))
        tb_layout.addWidget(self.btn_mask_done)

        return toolbar

    def _on_toggle_mask_shortcut(self) -> None:
        """Alterna el modo de trazado de máscara con la tecla 'M' ignorando entradas de texto activas."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_mask_tool'):
            self.btn_mask_tool.toggle()

    def _on_shortcut_invert_mask(self) -> None:
        """Invierte la máscara de la capa seleccionada mediante atajo Ctrl+I."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.invert_selected_brush_mask()

    def _on_toggle_mask_tool(self, enabled: bool) -> None:
        """Activa o desactiva la barra y modo de trazado interactivo de máscara en el lienzo."""
        if enabled and self.image_32bit is None:
            self.btn_mask_tool.setChecked(False)
            self.log_message("[AVISO] Debe cargar una imagen antes de utilizar el trazador de máscaras.")
            return

        self.mask_toolbar.setVisible(enabled)

        if enabled:
            # Desactivar herramientas excluyentes si estuvieran activas
            if hasattr(self, 'btn_crop') and self.btn_crop.isChecked():
                self.btn_crop.setChecked(False)
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_magnifier') and self.btn_magnifier.isChecked():
                self.btn_magnifier.setChecked(False)
            if self.canvas.split_view_enabled:
                self.btn_split_view.setChecked(False)

            self.canvas.set_masking_enabled(True)
            self.canvas.set_brush_mode(2 if self.btn_brush_sky.isChecked() else 1)
            self.canvas.set_brush_radius(self.slider_mask_brush.value())
            self.log_message("[MÁSCARA] Modo de trazado activo: dibuja trazos verdes en el CIELO y rojos en el SUELO.")
        else:
            self.canvas.set_masking_enabled(False)
            self.log_message("[MÁSCARA] Modo de trazado cerrado.")
        self._update_canvas_info_label()

    def _on_brush_mode_changed(self, btn_id: int) -> None:
        """Actualiza la clase de semilla seleccionada (2 = Cielo, 1 = Suelo)."""
        self.canvas.set_brush_mode(btn_id)

    def _on_slider_brush_changed(self, val: int) -> None:
        """Actualiza el radio del pincel al mover el deslizador."""
        self.lbl_mask_brush_val.setText(f"{val} px")
        self.canvas.set_brush_radius(val)

    def _on_brush_size_changed(self, val: int) -> None:
        """Actualiza el deslizador cuando el usuario cambia el tamaño con la rueda del ratón."""
        if hasattr(self, 'slider_mask_brush'):
            self.slider_mask_brush.blockSignals(True)
            self.slider_mask_brush.setValue(val)
            self.slider_mask_brush.blockSignals(False)
            self.lbl_mask_brush_val.setText(f"{val} px")

    def _open_mask_tool_from_section(self) -> None:
        """Abre y activa la herramienta de máscara desde el botón de la Sección 6."""
        if hasattr(self, 'btn_mask_tool'):
            if not self.btn_mask_tool.isChecked():
                self.btn_mask_tool.setChecked(True)
            else:
                self.canvas.set_masking_enabled(True)

    def _jump_to_mask_layers(self, target_layer_type: Optional[str] = None) -> None:
        """Asegura visibilidad del panel derecho, expande la sección de capas y selecciona la capa especificada."""
        if hasattr(self, 'right_container') and not self.right_container.isVisible():
            self.toggle_right_panel(True)
        if hasattr(self, 'sec_masks'):
            self.sec_masks.expand()
        if hasattr(self, 'mask_layers_widget') and target_layer_type:
            found = self.mask_layers_widget.select_layer_by_type(target_layer_type)
            if not found and target_layer_type == MASK_TYPE_IMAGE_GROUND:
                self.mask_layers_widget.select_layer_by_type(MASK_TYPE_GROUND)


    def _compute_mask_from_scribbles(self) -> None:
        """Calcula la segmentación continua cielo/suelo mediante GrabCut y fotometría guiada."""
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Abre primero una toma en el revelador.")
            return

        scribbles = self.canvas.get_scribbles_matrix()
        if scribbles is None or not (np.any(scribbles == 1) and np.any(scribbles == 2)):
            QMessageBox.warning(
                self, 
                "Trazos insuficientes", 
                "Debes pintar al menos un trazo verde (● Cielo) y un trazo rojo (● Suelo) sobre la imagen."
            )
            return

        self.log_message("[MÁSCARA] Iniciando segmentación inteligente de cielo y suelo...")
        self.setCursor(Qt.WaitCursor)
        QCoreApplication.processEvents()

        old_m = self.current_mask.copy() if self.current_mask is not None else None
        t0 = time.perf_counter()

        try:
            h, w = self.image_32bit.shape[:2]
            sh, sw = scribbles.shape[:2]
            if (sh, sw) != (h, w):
                scribbles_full = cv2.resize(scribbles, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                scribbles_full = scribbles

            feather = self.spin_mask_feather.value()
            iters = self.spin_mask_iters.value()
            shift_edge = self.spin_mask_shift.value() if hasattr(self, 'spin_mask_shift') else 0

            new_mask = refine_mask_guided(
                self.image_32bit,
                scribbles_full,
                feather_radius=feather,
                iterations=iters,
                shift_edge=shift_edge
            )

            self.current_mask = new_mask
            self._mask_source_name = "Trazos"
            self._generate_preview_proxy()
            self.update_stretch_preview()
            self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
            self.canvas.set_scribbles_visible(False)
            self.canvas.set_mask_visible(True)
            self._set_mask_overlay_ui_state(True)
            self._update_mask_status_ui()
            self._record_mask_action(old_m, self.current_mask, "Generar Máscara")

            elapsed = time.perf_counter() - t0
            self.log_message(f"✓ [MÁSCARA FINALIZADA] Máscara cielo/suelo calculada en {elapsed:.2f} s ({w}x{h} px, Borde: {feather}px, Pasos: {iters}).")
        except Exception as e:
            self.log_message(f"[ERROR MÁSCARA] Fallo al generar la máscara: {e}")
            QMessageBox.critical(self, "Error de Segmentación", f"No se pudo generar la máscara:\n{e}")
        finally:
            self.unsetCursor()

    def _clear_scribbles(self) -> None:
        """Borra únicamente los trazos dibujados sin eliminar la máscara actual."""
        self.canvas.clear_scribbles()
        self.log_message("[MÁSCARA] Trazos de semillas eliminados del visor.")

    def _delete_mask(self) -> None:
        """Elimina completamente la máscara de cielo/suelo y actualiza el revelador."""
        if self.current_mask is None:
            return
        old_m = self.current_mask.copy()
        self.current_mask = None
        self._mask_source_name = ""
        self.canvas.clear_scribbles()
        self.canvas.set_refined_mask(None)
        self.canvas.set_mask_visible(False)
        self._set_mask_overlay_ui_state(False)
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self._update_mask_status_ui()
        self._record_mask_action(old_m, None, "Eliminar Máscara")
        self.log_message("[MÁSCARA] Máscara de suelo eliminada.")

    def _set_mask_overlay_ui_state(self, visible: bool) -> None:
        """Actualiza el estado visual, textos y colores de los botones de alternancia de máscara."""
        has_mask = (self.current_mask is not None)
        is_active = has_mask and visible

        qss_active = """
            QPushButton {
                background-color: #2e7d32;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #81c784;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #388e3c;
                border-color: #a5d6a7;
            }
        """
        qss_inactive = """
            QPushButton {
                background-color: #262933;
                color: #a0a0a0;
                border: 1px solid #3b4252;
                font-weight: normal;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #88c0d0;
                color: #eceff4;
            }
            QPushButton:disabled {
                color: #4c566a;
                background-color: #1a1c23;
                border: 1px solid #2a2e39;
            }
        """

        if hasattr(self, 'btn_mask_overlay_toggle'):
            self.btn_mask_overlay_toggle.setEnabled(has_mask)
            self.btn_mask_overlay_toggle.blockSignals(True)
            self.btn_mask_overlay_toggle.setChecked(is_active)
            self.btn_mask_overlay_toggle.blockSignals(False)
            self.btn_mask_overlay_toggle.setStyleSheet(qss_active if is_active else qss_inactive)
            self.btn_mask_overlay_toggle.setText("Visible" if is_active else "Ver")

        if hasattr(self, 'panel_ground_sky') and hasattr(self.panel_ground_sky, 'set_overlay_state'):
            self.panel_ground_sky.set_overlay_state(is_active)

    def _on_top_mask_overlay_toggled(self, checked: bool) -> None:
        """Alterna la visibilidad del overlay de la máscara cielo/suelo en el visor interactivo."""
        if self.current_mask is None:
            self._set_mask_overlay_ui_state(False)
            return

        if checked:
            self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
        self.canvas.set_mask_visible(checked)
        self._set_mask_overlay_ui_state(checked)
        self.log_message(f"[MÁSCARA] Visualización de máscara {'activada' if checked else 'oculta'}.")

    def _load_mask_from_toolbar(self) -> None:
        """Permite cargar una máscara de cielo/suelo previamente generada en el apilado o desde archivo."""
        if self.image_32bit is None:
            QMessageBox.warning(self, "Aviso", "Abre primero una toma en el revelador.")
            return

        # Comprobar si existe una máscara activa en el Apilador (Stacker) o Startrails
        stacker_mask = None
        win = self.window()
        if win is not None:
            if hasattr(win, "tab_stacker") and getattr(win.tab_stacker, "computed_mask", None) is not None:
                stacker_mask = win.tab_stacker.computed_mask
            elif hasattr(win, "tab_startrails") and getattr(win.tab_startrails, "current_mask", None) is not None:
                stacker_mask = win.tab_startrails.current_mask

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1e2d;
                color: #e0e0e0;
                border: 1px solid #33384c;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 10px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #0288d1;
                color: #ffffff;
            }
        """)
        act_ai = menu.addAction("Auto-Detectar Suelo con IA (1-Click)")
        act_ai.setIcon(get_icon("sparkles", color="#ffffff", size=14))
        act_stacker = None
        if stacker_mask is not None:
            act_stacker = menu.addAction("Usar Máscara generada en el Apilado")
            act_stacker.setIcon(get_icon("layers", color="#80d8ff", size=14))
        act_file = menu.addAction("Cargar Archivo de Máscara (.png, .tif, .fits)...")
        act_file.setIcon(get_icon("folder", color="#80d8ff", size=14))

        pos = self.btn_load_mask.mapToGlobal(self.btn_load_mask.rect().bottomLeft())
        selected = menu.exec(pos)
        if selected == act_ai:
            self._detect_mask_ai_from_toolbar()
        elif act_stacker and selected == act_stacker:
            self._apply_loaded_mask(stacker_mask, source_desc="Apilador (Stacker)")
        elif selected == act_file:
            self._load_mask_from_file()

    def _detect_mask_ai_from_toolbar(self) -> None:
        """Calcula la máscara de suelo/cielo con IA ligera directamente desde el visor del revelador."""
        if self.image_32bit is None:
            return

        from core.ai_horizon import is_ai_model_available
        if not is_ai_model_available():
            msg = QMessageBox(self)
            msg.setWindowTitle("Modelo de IA no encontrado")
            msg.setText(
                "El modelo de IA para detección semántica precisa de suelo (skyseg.onnx, ~167 MB) no se encuentra en la carpeta 'models/'.\n\n"
                "¿Deseas descargarlo ahora automáticamente o prefieres usar el método algorítmico rápido?"
            )
            btn_download = msg.addButton("Descargar IA (~167 MB)", QMessageBox.AcceptRole)
            btn_algo = msg.addButton("Usar Algorítmico", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancelar", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_download)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_download:
                self._start_model_download_developer()
                return
            elif clicked == btn_algo:
                self._run_horizon_developer(force_algorithmic=True)
                return
            else:
                return

        self._run_horizon_developer(force_algorithmic=False)

    def _start_model_download_developer(self):
        """Descarga asistida del modelo ONNX con diálogo de progreso para el revelador."""
        progress = QProgressDialog("Conectando con el servidor de modelos...", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Descarga de Modelo IA (AstroLoom)")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setAutoClose(True)

        from gui.worker import ModelDownloadWorker
        self.download_worker = ModelDownloadWorker(parent=self)

        def on_prog(pct, d_mb, t_mb):
            progress.setValue(pct)
            progress.setLabelText(f"Descargando modelo de IA (skyseg.onnx)...\n{d_mb:.1f} MB de {t_mb:.1f} MB ({pct}%)")

        def on_finished(path):
            progress.close()
            self.log_message(f"✓ [IA HORIZONTE] Modelo descargado e instalado en: {os.path.basename(path)}")
            self._run_horizon_developer(force_algorithmic=False)

        def on_error(err):
            progress.close()
            self.log_message(f"[ERROR DESCARGA] {err}")
            QMessageBox.warning(self, "Error en Descarga", f"No se pudo descargar el modelo:\n{err}\n\nSe usará el método algorítmico.")
            self._run_horizon_developer(force_algorithmic=True)

        def on_cancel():
            self.download_worker.cancel()
            self.log_message("[IA HORIZONTE] Descarga cancelada por el usuario.")

        self.download_worker.progress_changed.connect(on_prog)
        self.download_worker.finished_success.connect(on_finished)
        self.download_worker.error_occurred.connect(on_error)
        progress.canceled.connect(on_cancel)

        self.download_worker.start()

    def _run_horizon_developer(self, force_algorithmic: bool = False):
        self.setCursor(Qt.WaitCursor)
        self.log_message("[IA HORIZONTE] Ejecutando detección de cielo/suelo...")
        QCoreApplication.processEvents()
        try:
            from core.ai_horizon import auto_detect_sky_mask
            mask, method = auto_detect_sky_mask(self.image_32bit, force_algorithmic=force_algorithmic)
            self._apply_loaded_mask(mask, source_desc=f"Auto-Detección {method}")
        except Exception as e:
            self.log_message(f"[ERROR IA HORIZONTE] Falló la detección: {e}")
            QMessageBox.critical(self, "Error Detección IA", f"No se pudo completar la segmentación automática:\n{e}")
        finally:
            self.unsetCursor()

    def _load_mask_from_file(self) -> None:
        """Abre un selector de archivo para cargar una máscara desde disco."""
        cfg = load_config()
        default_dir = cfg.get("masks_dir", os.getcwd())
        filtros = "Imágenes de Máscara (*.png *.tif *.tiff *.fits *.fit);;Todos los archivos (*.*)"
        filepath, _ = QFileDialog.getOpenFileName(self, "Cargar Máscara de Suelo / Cielo", default_dir, filtros)
        if not filepath:
            return

        try:
            raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raw = load_image_as_float32(filepath)
                if raw is not None and raw.ndim == 3:
                    raw = raw[..., 0]

            if raw is None:
                QMessageBox.warning(self, "Error", f"No se pudo decodificar el archivo de máscara:\n{filepath}")
                return

            if raw.ndim == 3:
                raw = raw[..., 0]

            max_v = 65535.0 if raw.dtype == np.uint16 else (1.0 if raw.dtype in (np.float32, np.float64) and np.max(raw) <= 1.0 else 255.0)
            mask_float = np.clip(raw.astype(np.float32) / max_v, 0.0, 1.0)
            self._apply_loaded_mask(mask_float, source_desc=os.path.basename(filepath))
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar máscara", f"Fallo al procesar la máscara:\n{e}")

    def _apply_loaded_mask(self, mask: np.ndarray, source_desc: str = "") -> None:
        """Adapta y aplica una máscara cargada a la imagen actual en el Revelador."""
        if self.image_32bit is None or mask is None:
            return

        h, w = self.image_32bit.shape[:2]
        if mask.shape[:2] != (h, w):
            self.log_message(f"[MÁSCARA] Redimensionando máscara ({mask.shape[1]}x{mask.shape[0]}) a resolución de lienzo ({w}x{h})...")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        old_m = self.current_mask.copy() if self.current_mask is not None else None
        self.current_mask = np.ascontiguousarray(np.clip(mask, 0.0, 1.0), dtype=np.float32)
        self._mask_source_name = source_desc
        self._generate_preview_proxy()
        self.update_stretch_preview()
        self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
        self.canvas.set_scribbles_visible(False)
        self.canvas.set_mask_visible(True)
        self._set_mask_overlay_ui_state(True)
        self._update_mask_status_ui()
        self._record_mask_action(old_m, self.current_mask, f"Cargar Máscara ({source_desc})")
        self.log_message(f"✓ [MÁSCARA] Máscara cargada y aplicada con éxito desde: {source_desc}.")


    def _update_mask_status_ui(self) -> None:
        """Actualiza el indicador visual de presencia de máscara en la Sección 8 y botones."""
        if hasattr(self, 'sec_gnd_tone'):
            if self.current_mask is not None:
                source_label = f"● Suelo ({self._mask_source_name})" if self._mask_source_name else "● Suelo Activo"
                self.sec_gnd_tone.set_status_badge(source_label, "#a5d6a7")
            else:
                self.sec_gnd_tone.set_status_badge("")

        if hasattr(self, 'mask_layers_widget') and self.image_starless is not None and not getattr(self, '_is_loading_project', False):
            self.mask_layers_widget.ensure_starnet_layers(has_ground=(self.current_mask is not None))

        is_mask_present = (self.current_mask is not None)
        is_visible = (getattr(self.canvas, 'show_mask_overlay', False) and is_mask_present)
        self._set_mask_overlay_ui_state(is_visible)

        if hasattr(self, 'panel_ground_sky') and hasattr(self.panel_ground_sky, 'update_mask_ui_state'):
            self.panel_ground_sky.update_mask_ui_state(is_mask_present, self._mask_source_name, is_visible)

        if not hasattr(self, 'lbl_gnd_mask_status'):
            return
        if is_mask_present:
            desc = f"Máscara de suelo activa ({self._mask_source_name})" if self._mask_source_name else "Máscara de suelo activa"
            self.lbl_gnd_mask_status.setText(desc)
            self.lbl_gnd_mask_status.setStyleSheet("color: #a5d6a7; font-size: 11px; font-weight: bold;")
            if hasattr(self, 'btn_edit_ground_mask'):
                self.btn_edit_ground_mask.setText("Editar Máscara...")
            if hasattr(self, 'btn_delete_mask'):
                self.btn_delete_mask.setEnabled(True)
        else:
            self.lbl_gnd_mask_status.setText("Sin máscara de suelo activa")
            self.lbl_gnd_mask_status.setStyleSheet("color: #ffb74d; font-size: 11px;")
            if hasattr(self, 'btn_edit_ground_mask'):
                self.btn_edit_ground_mask.setText("Crear Máscara Suelo...")
            if hasattr(self, 'btn_delete_mask'):
                self.btn_delete_mask.setEnabled(False)

    def _record_mask_action(self, old_mask: Optional[np.ndarray], new_mask: Optional[np.ndarray], action_name: str = "Máscara") -> None:
        """Registra una acción de máscara en el historial de Deshacer / Rehacer."""
        if (
            self._is_undoing_or_redoing
            or getattr(self, '_is_applying_preset', False)
            or getattr(self, '_is_previewing_preset', False)
        ):
            return
        undo_item = {
            "type": "mask",
            "name": action_name,
            "old_mask": old_mask.copy() if old_mask is not None else None,
            "new_mask": new_mask.copy() if new_mask is not None else None,
        }
        self.undo_stack.append(undo_item)
        if len(self.undo_stack) > self.MAX_UNDO_STEPS:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_undo_redo_buttons()

    def _on_layer_mask_preview_requested(self, mask_data: Any) -> None:
        """Muestra u oculta la máscara de la capa seleccionada sobre el visor interactivo (modo ruby o bw)."""
        mode = "ruby"
        if isinstance(mask_data, tuple) and len(mask_data) == 2:
            mask_array, mode = mask_data
        else:
            mask_array = mask_data

        if mask_array is not None and (mode == "bw" or np.max(mask_array) > 1e-4):
            self.canvas.set_refined_mask(mask_array, mode=mode)
            self.canvas.set_mask_visible(True)
        else:
            self.canvas.set_mask_visible(False)
            if self.canvas.enable_masking and self.current_mask is not None:
                self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
            else:
                self.canvas.set_refined_mask(None)

    def _on_layer_brush_toggled(self, active: bool, layer: Optional[MaskLayer]) -> None:
        """Activa o desactiva la interacción del pincel en el lienzo según la capa seleccionada."""
        if active and layer is not None and getattr(layer, 'enabled', True):
            self.canvas.set_layer_brush_enabled(True)
            self.canvas.set_layer_brush_params(
                layer.brush_radius, layer.brush_hardness, layer.brush_flow, layer.brush_mode
            )
            show_overlay = bool(
                hasattr(self, 'mask_layers_widget') and 
                self.mask_layers_widget.is_mask_preview_active()
            )
            if layer.brush_mask is not None and self.image_32bit is not None:
                h_f, w_f = self.image_32bit.shape[:2]
                if layer.brush_mask.shape[:2] != (h_f, w_f):
                    layer.brush_mask = cv2.resize(layer.brush_mask, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
            self.canvas.set_active_layer_mask(layer.brush_mask, show_overlay=show_overlay)
            if not show_overlay:
                self.canvas.set_refined_mask(None)
                self.canvas.set_mask_visible(False)
        else:
            self.canvas.set_layer_brush_enabled(False)
            self.canvas.set_active_layer_mask(None)

            # Limpiar overlay del lienzo si no hay previsualización válida y activa de la capa
            sel_layer = self.mask_layers_widget.get_selected_layer() if hasattr(self, 'mask_layers_widget') else None
            is_preview_valid = (
                sel_layer is not None and 
                sel_layer.enabled and 
                not sel_layer.is_system_layer and 
                hasattr(self, 'mask_layers_widget') and 
                self.mask_layers_widget.is_mask_preview_active()
            )
            if not is_preview_valid:
                if self.canvas.enable_masking and self.current_mask is not None:
                    self.canvas.set_refined_mask(self.current_mask, mode="bicolor")
                else:
                    self.canvas.set_refined_mask(None)
                    self.canvas.set_mask_visible(False)
        self._update_canvas_info_label()

    def _on_layer_brush_params_changed(self, radius: int, hardness: float, flow: float, mode: str) -> None:
        """Actualiza los parámetros interactivos de trazo en el lienzo."""
        self.canvas.set_layer_brush_params(radius, hardness, flow, mode)

    def _on_layer_brush_stroke_finished(self, stroke_mask: np.ndarray) -> None:
        """Manejador ejecutado al soltar el clic izquierdo tras pintar con el pincel de capa."""
        layer = self.mask_layers_widget.get_selected_layer()
        if layer and (layer.mask_type == "brush" or layer.mask_type == MASK_TYPE_IMAGE_RASTER):
            if hasattr(layer, "set_brush_mask"):
                layer.set_brush_mask(stroke_mask.copy())
            else:
                layer.brush_mask = stroke_mask.copy()
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.invalidate_layer_thumbnail(layer.id)
            # Si el botón de previsualización NO está activado expresamente, retirar el overlay temporal
            # del trazo para que el usuario pueda apreciar de inmediato los cambios tonales en la imagen
            if hasattr(self, 'mask_layers_widget') and not self.mask_layers_widget.is_mask_preview_active():
                self.canvas.set_refined_mask(None)
                self.canvas.set_mask_visible(False)
            self._commit_action()
            self.update_stretch_preview()

    def _get_layer_image_for_histogram(self, layer, target_size: Optional[tuple[int, int]] = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Retorna la imagen de trabajo y máscara espacial correspondiente a la capa seleccionada
        para renderizar su histograma de fondo en la Curva Tonal de Capa.
        """
        if self.image_32bit is None:
            return None, None, False

        m_type = getattr(layer, 'mask_type', '')

        # Obtener valores actuales de estirado desde panel_stretch
        st_state = self.panel_stretch.get_state() if hasattr(self, 'panel_stretch') and self.panel_stretch is not None else {}
        stretch_mode = st_state.get("stretch_mode", "mtf")
        ghs_sp = st_state.get("ghs_sp_val", 0.02)
        ghs_d = st_state.get("ghs_d_val", 5.0)
        ghs_b = st_state.get("ghs_b_val", 2.0)
        ghs_hp = st_state.get("ghs_hp_val", 0.0)
        ghs_lp = st_state.get("ghs_lp_val", 0.0)
        ghs_bp = st_state.get("ghs_bp_val", 0.0)
        ghs_color_mode = st_state.get("ghs_color_mode", "luma")
        bp_val = st_state.get("bp_val", 0.0)
        mtf_val = st_state.get("mtf_val", 0.15)

        # 1. Capa de Estrellas (StarNet AI)
        if m_type == MASK_TYPE_IMAGE_STARS:
            img = getattr(self, '_cached_stars_stretched', None)
            if img is None:
                st_src = self.proxy_stars if self.proxy_stars is not None else self.image_stars
                if st_src is not None:
                    if stretch_mode == "ghs":
                        img = np.clip(
                            apply_ghs_stretch(
                                st_src,
                                sp=ghs_sp,
                                d=ghs_d,
                                b=ghs_b,
                                hp=ghs_hp,
                                lp=ghs_lp,
                                bp=0.0,
                                color_mode="rgb"
                            ) * self.star_intensity,
                            0.0, 1.0
                        )
                    else:
                        img = np.clip(
                            manual_stretch(st_src, black_point=0.0, midtone=mtf_val) * self.star_intensity,
                            0.0, 1.0
                        )
            if img is not None:
                img = apply_pre_curve_adjustments(img, layer)
            return img, None, True

        # 2. Capa de Suelo / Terreno
        if m_type == MASK_TYPE_IMAGE_GROUND:
            img = getattr(self, '_cached_base_stretched', None)
            if img is None:
                img = getattr(self, '_cached_rendered_image', None)
            if img is None:
                b_src = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
                if b_src is not None:
                    if stretch_mode == "ghs":
                        img = apply_ghs_stretch(
                            b_src,
                            sp=ghs_sp,
                            d=ghs_d,
                            b=ghs_b,
                            hp=ghs_hp,
                            lp=ghs_lp,
                            bp=ghs_bp,
                            color_mode=ghs_color_mode
                        )
                    else:
                        img = manual_stretch(b_src, black_point=bp_val, midtone=mtf_val)
            sky_m = self.proxy_mask if self.proxy_mask is not None else self.current_mask
            gnd_m = np.clip(1.0 - sky_m, 0.0, 1.0) if sky_m is not None else None
            if img is not None:
                if gnd_m is not None:
                    m_eff_2d = gnd_m[..., 0] if gnd_m.ndim == 3 else gnd_m
                    if m_eff_2d.shape[:2] != img.shape[:2]:
                        m_eff_2d = cv2.resize(m_eff_2d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    img = apply_local_adjustments_to_image(img, m_eff_2d, layer)
                else:
                    img = apply_pre_curve_adjustments(img, layer)
            return img, gnd_m, False

        # 3. Capa de Fondo Nebular (Starless)
        if m_type == MASK_TYPE_IMAGE_STARLESS:
            img = getattr(self, '_cached_base_stretched', None)
            if img is None:
                img = getattr(self, '_cached_rendered_image', None)
            if img is None:
                sl_src = self.proxy_starless if self.proxy_starless is not None else self.image_starless
                if sl_src is not None:
                    if getattr(self, 'starless_is_stretched', False):
                        img = sl_src.copy()
                    else:
                        if stretch_mode == "ghs":
                            img = apply_ghs_stretch(
                                sl_src,
                                sp=ghs_sp,
                                d=ghs_d,
                                b=ghs_b,
                                hp=ghs_hp,
                                lp=ghs_lp,
                                bp=0.0,
                                color_mode="rgb"
                            )
                        else:
                            img = manual_stretch(sl_src, black_point=0.0, midtone=mtf_val)
            if img is not None:
                img = apply_pre_curve_adjustments(img, layer)
            has_active_ground = False
            if hasattr(self, 'mask_layers_widget'):
                has_active_ground = any(
                    getattr(l, 'mask_type', '') == MASK_TYPE_IMAGE_GROUND and getattr(l, 'enabled', True)
                    for l in self.mask_layers_widget.layers
                )
            sky_m = (self.proxy_mask if self.proxy_mask is not None else self.current_mask) if has_active_ground else None
            return img, sky_m, False

        # 4. Capa de Fondo Base
        if m_type == MASK_TYPE_IMAGE_BASE:
            img = getattr(self, '_cached_base_stretched', None)
            if img is None:
                b_src = self.preview_proxy if self.preview_proxy is not None else self.image_32bit
                if b_src is not None:
                    if stretch_mode == "ghs":
                        img = apply_ghs_stretch(
                            b_src,
                            sp=ghs_sp,
                            d=ghs_d,
                            b=ghs_b,
                            hp=ghs_hp,
                            lp=ghs_lp,
                            bp=ghs_bp,
                            color_mode=ghs_color_mode
                        )
                    else:
                        img = manual_stretch(b_src, black_point=bp_val, midtone=mtf_val)
            if img is None:
                img = getattr(self, '_cached_rendered_image', None)
            if img is not None:
                img = apply_pre_curve_adjustments(img, layer)
            return img, None, False

        # 5. Capa Ráster de Imagen / Filtros Coronales
        if m_type == MASK_TYPE_IMAGE_RASTER:
            raw_img = getattr(layer, 'image_data', None)
            if raw_img is not None:
                if raw_img.ndim == 2:
                    raw_img = np.repeat(raw_img[..., np.newaxis], 3, axis=-1)
                elif raw_img.ndim == 3 and raw_img.shape[2] == 1:
                    raw_img = np.repeat(raw_img, 3, axis=-1)

                if target_size is not None:
                    # Reducción previa ultrarrápida para miniaturas (ej. 48x48)
                    src = getattr(layer, 'proxy_draft', None)
                    if src is None:
                        src = getattr(layer, 'proxy_preview', None)
                    if src is None:
                        src = raw_img
                    raw_img = cv2.resize(src, target_size, interpolation=cv2.INTER_LINEAR)
                elif self.preview_proxy is not None:
                    h_p, w_p = self.preview_proxy.shape[:2]
                    src = getattr(layer, 'proxy_preview', None)
                    if src is None:
                        src = raw_img
                    if src.shape[:2] != (h_p, w_p):
                        raw_img = cv2.resize(src, (w_p, h_p), interpolation=cv2.INTER_LINEAR)
                    else:
                        raw_img = src
                elif getattr(layer, 'proxy_preview', None) is not None:
                    raw_img = layer.proxy_preview
                elif raw_img.shape[0] > 1024 or raw_img.shape[1] > 1024:
                    scale = 1024.0 / max(raw_img.shape[0], raw_img.shape[1])
                    nw = max(1, int(round(raw_img.shape[1] * scale)))
                    nh = max(1, int(round(raw_img.shape[0] * scale)))
                    raw_img = cv2.resize(raw_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

                img = apply_pre_curve_adjustments(raw_img, layer)
                m = getattr(layer, 'brush_mask', None)
                if m is not None and target_size is not None and (m.shape[1], m.shape[0]) != target_size:
                    m = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
                return img, m, False
            return None, None, False

        # 6. Capas de ajuste local estándar
        base_canvas = getattr(self, '_cached_base_stretched', None)
        if base_canvas is None:
            base_canvas = getattr(self, '_last_rendered_preview', None)

        if base_canvas is not None and hasattr(self, 'mask_layers_widget'):
            all_layers = getattr(self.mask_layers_widget, 'layers', [])
            sky_m = self.proxy_mask if self.proxy_mask is not None else self.current_mask
            stars_img = getattr(self, '_cached_stars_stretched', None)
            if stars_img is None and getattr(self, 'image_stars', None) is not None:
                stars_img = self.proxy_stars if self.proxy_stars is not None else self.image_stars
            starless_img = self.proxy_starless if self.proxy_starless is not None else self.image_starless

            img = get_canvas_below_layer(
                base_canvas,
                all_layers,
                layer.id,
                master_sky_mask=sky_m,
                starless_image=starless_img,
                stars_image=stars_img
            )
        else:
            img = getattr(self, '_last_rendered_preview', None) or getattr(self, '_cached_base_stretched', None)

        mask = None
        if img is not None:
            sky_m = self.proxy_mask if self.proxy_mask is not None else self.current_mask
            if sky_m is not None and sky_m.shape[:2] != img.shape[:2]:
                sky_m = cv2.resize(sky_m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = compute_effective_layer_mask(
                layer,
                img,
                master_sky_mask=sky_m
            )
            img = apply_pre_curve_adjustments(img, layer)
        elif m_type == MASK_TYPE_BRUSH:
            mask = getattr(layer, 'brush_mask', None)

        return img, mask, False

    def _on_toggle_clipping_shortcut(self) -> None:
        """Alterna los indicadores de recorte con la tecla 'J' ignorando entradas de texto activas."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'curve_widget'):
            self.curve_widget.toggle_both_clipping()

    def _on_toggle_split_shortcut(self) -> None:
        """Alterna el modo comparador Antes / Después con la tecla 'Y'."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_split_view'):
            self.btn_split_view.toggle()

    def _on_toggle_split_orientation_shortcut(self) -> None:
        """Alterna la orientación de la cortina con Shift+Y."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_split_orientation') and self.canvas.split_view_enabled:
            self.on_toggle_split_orientation()

    def _on_toggle_pipette_shortcut(self) -> None:
        """Alterna la pipeta de balance de blancos con la tecla 'W'."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_pipette'):
            self.btn_pipette.toggle()

    def _on_toggle_magnifier_shortcut(self) -> None:
        """Alterna la lupa de inspección flotante con la tecla 'Z'."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            return
        if hasattr(self, 'btn_magnifier'):
            self.btn_magnifier.toggle()

    def _on_canvas_eyedropper_toggled(self, active: bool) -> None:
        """Sincroniza el estado del cuentagotas con los botones de la interfaz."""
        if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'wb':
            self.btn_pipette.blockSignals(True)
            self.btn_pipette.setChecked(active)
            self.btn_pipette.blockSignals(False)

        if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'ghs_sp':
            self.btn_ghs_pipette.blockSignals(True)
            self.btn_ghs_pipette.setChecked(active)
            self.btn_ghs_pipette.blockSignals(False)

        if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'lp_center':
            self.btn_lp_center_pipette.blockSignals(True)
            self.btn_lp_center_pipette.setChecked(active)
            self.btn_lp_center_pipette.blockSignals(False)

        if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked() != active and getattr(self, '_eyedropper_target', 'wb') == 'lp_color':
            self.btn_lp_color_pipette.blockSignals(True)
            self.btn_lp_color_pipette.setChecked(active)
            self.btn_lp_color_pipette.blockSignals(False)

        if hasattr(self, 'mask_layers_widget') and getattr(self, '_eyedropper_target', 'wb') == 'mask_color':
            self.mask_layers_widget.set_color_picker_active(active)

        if not active:
            self._eyedropper_target = 'wb'

    def _on_toggle_wb_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas para Balance de Blancos asegurando exclusión mutua."""
        if active:
            self._eyedropper_target = 'wb'
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True, "Clic: balancear WB")
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'wb':
                self.canvas.set_eyedropper_active(False)

    def _on_toggle_ghs_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el Punto de Simetría (SP) de GHS en el lienzo."""
        if active:
            self._eyedropper_target = 'ghs_sp'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True, "Clic: punto simetría SP")
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'ghs_sp':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_toggle_lp_center_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el Centro de la Cúpula 2D en el lienzo."""
        if active:
            self._eyedropper_target = 'lp_center'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True, "Clic: centro cúpula LP")
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'lp_center':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_toggle_lp_color_pipette(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el Tinte Cromático de la Cúpula 2D en el lienzo."""
        if active:
            self._eyedropper_target = 'lp_color'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.set_color_picker_active(False)
            self.canvas.set_eyedropper_active(True, "Clic: muestrear tinte LP")
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'lp_color':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_layer_color_sample_requested(self, active: bool) -> None:
        """Alterna el cuentagotas interactivo para capturar el color de una capa de ajuste en el lienzo."""
        if active:
            self._eyedropper_target = 'mask_color'
            if hasattr(self, 'btn_pipette') and self.btn_pipette.isChecked():
                self.btn_pipette.setChecked(False)
            if hasattr(self, 'btn_ghs_pipette') and self.btn_ghs_pipette.isChecked():
                self.btn_ghs_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_center_pipette') and self.btn_lp_center_pipette.isChecked():
                self.btn_lp_center_pipette.setChecked(False)
            if hasattr(self, 'btn_lp_color_pipette') and self.btn_lp_color_pipette.isChecked():
                self.btn_lp_color_pipette.setChecked(False)
            self.canvas.set_eyedropper_active(True, "Clic: seleccionar color")
            self.log_message("[MÁSCARA COLOR] Pipeta activa: haz clic sobre cualquier zona del visor para capturar su color.")
        else:
            if getattr(self, '_eyedropper_target', 'wb') == 'mask_color':
                self.canvas.set_eyedropper_active(False)
                self._eyedropper_target = 'wb'

    def _on_eyedropper_color_sampled(self, r: float, g: float, b: float) -> None:
        """Procesa la muestra del cuentagotas para Balance de Blancos, GHS SP, Cúpula 2D o Máscara de Color."""
        if getattr(self, '_eyedropper_target', 'wb') == 'mask_color':
            disp_rgb = getattr(self.canvas, 'last_sampled_display_rgb', None)
            if disp_rgb is not None:
                dr, dg, db = disp_rgb[0] / 255.0, disp_rgb[1] / 255.0, disp_rgb[2] / 255.0
            else:
                dr, dg, db = r, g, b

            import colorsys
            h, s, v = colorsys.rgb_to_hsv(dr, dg, db)
            hue_deg = int(round(h * 360.0)) % 360
            sat_val = float(np.clip(s, 0.01, 0.90))

            if hasattr(self, 'mask_layers_widget'):
                self.mask_layers_widget.apply_sampled_color(hue_deg, sat_val)
                self.mask_layers_widget.set_color_picker_active(False)

            self._eyedropper_target = 'wb'
            self.log_message(
                f"[MÁSCARA COLOR] Color muestreado en visor: Hue={hue_deg}°, Sat={sat_val:.2f}."
            )
            return

        if getattr(self, '_eyedropper_target', 'wb') == 'ghs_sp':
            # Muestra para Punto de Simetría (SP) en GHS
            sample_lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            sp_val = float(np.clip(sample_lum, 0.00001, 0.99999))
            if hasattr(self, 'spin_ghs_sp'):
                self.spin_ghs_sp.blockSignals(True)
                self.spin_ghs_sp.setValue(sp_val)
                self.spin_ghs_sp.blockSignals(False)
            if hasattr(self, 'slider_ghs_sp'):
                slider_val = int(np.clip(sp_val * 10000.0, 0, 10000))
                self.slider_ghs_sp.blockSignals(True)
                self.slider_ghs_sp.setValue(slider_val)
                self.slider_ghs_sp.blockSignals(False)
            if hasattr(self, 'btn_ghs_pipette'):
                self.btn_ghs_pipette.setChecked(False)
            self._eyedropper_target = 'wb'
            self.log_message(
                f"[GHS] Cuentagotas SP: muestra RGB=({r:.3f}, {g:.3f}, {b:.3f}) ➜ SP fijado en {sp_val:.5f}"
            )
            self.update_stretch_preview()
            self._commit_action()
            return

        if getattr(self, '_eyedropper_target', 'wb') == 'lp_center':
            coords = getattr(self.canvas, 'last_sampled_coords', None)
            if coords is not None:
                cx, cy = coords
            else:
                cx, cy = 0.50, 0.90
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            self.lp_center_x = cx
            self.lp_center_y = cy
            if hasattr(self, 'spin_lp_cx'):
                self.spin_lp_cx.blockSignals(True)
                self.spin_lp_cx.setValue(cx)
                self.spin_lp_cx.blockSignals(False)
            if hasattr(self, 'slider_lp_cx'):
                self.slider_lp_cx.blockSignals(True)
                self.slider_lp_cx.setValue(int(round(cx * 100.0)))
                self.slider_lp_cx.blockSignals(False)
            if hasattr(self, 'spin_lp_cy'):
                self.spin_lp_cy.blockSignals(True)
                self.spin_lp_cy.setValue(cy)
                self.spin_lp_cy.blockSignals(False)
            if hasattr(self, 'slider_lp_cy'):
                self.slider_lp_cy.blockSignals(True)
                self.slider_lp_cy.setValue(int(round(cy * 100.0)))
                self.slider_lp_cy.blockSignals(False)
            if hasattr(self, 'btn_lp_center_pipette'):
                self.btn_lp_center_pipette.setChecked(False)
            self._eyedropper_target = 'wb'
            self.log_message(
                f"[Cúpula 2D] Centro seleccionado en el lienzo: X={cx:.2f}, Y={cy:.2f}"
            )
            self.update_stretch_preview()
            self._commit_action()
            return

        if getattr(self, '_eyedropper_target', 'wb') == 'lp_color':
            coords = getattr(self.canvas, 'last_sampled_coords', None)
            if coords is not None and getattr(self, '_cache_ms_pre_lp_img', None) is not None:
                h_lp, w_lp = self._cache_ms_pre_lp_img.shape[:2]
                ix = int(np.clip(coords[0] * w_lp, 0, w_lp - 1))
                iy = int(np.clip(coords[1] * h_lp, 0, h_lp - 1))
                patch = self._cache_ms_pre_lp_img[max(0, iy - 2):min(h_lp, iy + 3), max(0, ix - 2):min(w_lp, ix + 3)]
                r, g, b = np.mean(patch, axis=(0, 1))[:3]

            max_c = max(1e-4, max(r, g, b))
            tint = (float(r / max_c), float(g / max_c), float(b / max_c))
            self.lp_custom_tint = tint
            self.lp_tint_type = "custom"
            if hasattr(self, 'panel_lp') and self.panel_lp is not None:
                self.panel_lp.set_custom_tint(tint[0], tint[1], tint[2])
            elif hasattr(self, 'combo_lp_tint'):
                idx = self.combo_lp_tint.findData("custom")
                if idx >= 0:
                    self.combo_lp_tint.blockSignals(True)
                    self.combo_lp_tint.setCurrentIndex(idx)
                    self.combo_lp_tint.blockSignals(False)
            if hasattr(self, 'btn_lp_color_pipette'):
                self.btn_lp_color_pipette.setChecked(False)
            self._eyedropper_target = 'wb'
            self.log_message(
                f"[Cúpula 2D] Tinte muestreado: RGB=({r:.3f}, {g:.3f}, {b:.3f}) ➜ Normalizado=({tint[0]:.2f}, {tint[1]:.2f}, {tint[2]:.2f})"
            )
            self.update_stretch_preview()
            self._commit_action()
            return

        r = max(1e-5, float(r))
        g = max(1e-5, float(g))
        b = max(1e-5, float(b))

        w_r = 1.0 / r
        w_g = 1.0 / g
        w_b = 1.0 / b
        w_mean = (w_r + w_g + w_b) / 3.0

        gain_r = w_r / w_mean
        gain_g = w_g / w_mean
        gain_b = w_b / w_mean

        # Solución de apply_white_balance:
        # gain_r - gain_b = temp
        # tint = 2.0 * (1.0 - gain_g)
        calc_temp = gain_r - gain_b
        calc_tint = 2.0 * (1.0 - gain_g)

        # Rango del slider (-1000 a +1000 con paso 0.001)
        clamped_temp = float(np.clip(calc_temp, -1.000, 1.000))
        clamped_tint = float(np.clip(calc_tint, -1.000, 1.000))

        temp_int = int(round(clamped_temp * 1000.0))
        tint_int = int(round(clamped_tint * 1000.0))

        self.slider_temp.setValue(temp_int)
        self.slider_tint.setValue(tint_int)

        if hasattr(self, 'btn_pipette'):
            self.btn_pipette.setChecked(False)

        self.log_message(
            f"[WB] Pipeta aplicada: Muestra RGB=({r:.3f}, {g:.3f}, {b:.3f}) ➜ "
            f"Temp={self.temp_val:+.3f}, Tinte={self.tint_val:+.3f}"
        )

    def on_toggle_split_orientation(self) -> None:
        """Alterna entre cortina vertical y horizontal."""
        new_ori = self.canvas.toggle_split_orientation()
        self._update_split_orientation_ui()
        ori_nombre = "VERTICAL (Izquierda / Derecha)" if new_ori == "vertical" else "HORIZONTAL (Arriba / Abajo)"
        self.log_message(f"[CANVAS] Orientación del comparador cambiada a: {ori_nombre}")
        self._update_canvas_info_label()

    def _update_split_orientation_ui(self) -> None:
        """Actualiza el texto y estado del botón de orientación de la cortina."""
        if not hasattr(self, 'btn_split_orientation'):
            return
        is_vert = (self.canvas.split_orientation == "vertical")
        self.btn_split_orientation.setText("⇋ Vertical" if is_vert else "⥯ Horizontal")
        self.btn_split_orientation.setEnabled(self.canvas.split_view_enabled)

    def _on_live_histogram_clipping_changed(self, clip_s: bool, clip_h: bool) -> None:
        """Sincroniza el lienzo y la curva tonal cuando se conmutan los triángulos de recorte del histograma."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_clipping_state(clip_s, clip_h)
        self.on_clipping_changed_sync(clip_s, clip_h)

    def on_clipping_changed_sync(self, clip_s: bool, clip_h: bool) -> None:
        """Sincroniza el lienzo y el botón de la barra superior cuando cambia el recorte."""
        self.canvas.set_clipping_state(clip_s, clip_h)
        if hasattr(self, 'live_histogram'):
            self.live_histogram.set_clipping_states(clip_s, clip_h)
        if hasattr(self, 'btn_toggle_clipping'):
            self.btn_toggle_clipping.blockSignals(True)
            self.btn_toggle_clipping.setChecked(clip_s or clip_h)
            self.btn_toggle_clipping.blockSignals(False)

    def on_toolbar_clipping_toggled(self, checked: bool) -> None:
        """Manejador al conmutar el botón de recorte en la barra superior."""
        if hasattr(self, 'curve_widget'):
            self.curve_widget.set_clipping_state(checked, checked)
        if hasattr(self, 'live_histogram'):
            self.live_histogram.set_clipping_states(checked, checked)
        self.canvas.set_clipping_state(checked, checked)
        estado = "ACTIVADOS (Azul=Negros, Rojo=Blancos)" if checked else "DESACTIVADOS"
        self.log_message(f"[CANVAS] Indicadores de recorte: {estado}")

    def toggle_left_panel(self, visible: Optional[bool] = None) -> None:
        """Muestra u oculta el panel lateral izquierdo de ajustes (estilo Lightroom)."""
        if not hasattr(self, 'left_container'):
            return
        if visible is None:
            visible = not self.left_container.isVisible()

        # Guardar tamaño previo antes de colapsar
        if not visible and self.left_container.isVisible() and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[0] > 50:
                self._last_left_panel_width = sizes[0]

        self.left_container.setVisible(visible)
        if hasattr(self, 'btn_toggle_left_panel'):
            self.btn_toggle_left_panel.blockSignals(True)
            self.btn_toggle_left_panel.setChecked(visible)
            self.btn_toggle_left_panel.blockSignals(False)

        # Restaurar tamaño previo al desplegar
        if visible and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[0] < 50:
                sizes[0] = getattr(self, '_last_left_panel_width', 380)
                self.main_splitter.setSizes(sizes)

    def toggle_right_panel(self, visible: Optional[bool] = None) -> None:
        """Muestra u oculta el panel lateral derecho de capas e histograma (estilo Lightroom)."""
        if not hasattr(self, 'right_container'):
            return
        if visible is None:
            visible = not self.right_container.isVisible()

        # Guardar tamaño previo antes de colapsar
        if not visible and self.right_container.isVisible() and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[2] > 50:
                self._last_right_panel_width = sizes[2]

        self.right_container.setVisible(visible)
        if hasattr(self, 'btn_toggle_right_panel'):
            self.btn_toggle_right_panel.blockSignals(True)
            self.btn_toggle_right_panel.setChecked(visible)
            self.btn_toggle_right_panel.blockSignals(False)

        # Restaurar tamaño previo al desplegar
        if visible and hasattr(self, 'main_splitter'):
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[2] < 50:
                sizes[2] = getattr(self, '_last_right_panel_width', 360)
                self.main_splitter.setSizes(sizes)

    def toggle_both_panels(self) -> None:
        """Alterna simultáneamente ambos paneles laterales (Modo Inmersivo / Cine Lightroom)."""
        left_vis = self.left_container.isVisible() if hasattr(self, 'left_container') else False
        right_vis = self.right_container.isVisible() if hasattr(self, 'right_container') else False
        target_vis = not (left_vis or right_vis)
        self.toggle_left_panel(target_vis)
        self.toggle_right_panel(target_vis)

    def _on_toggle_left_wide(self) -> None:
        """Alterna el ancho del panel izquierdo entre normal (~380px) y expandido (~720px o 42% del visor)."""
        if not hasattr(self, 'main_splitter') or not hasattr(self, 'left_container'):
            return
        if not self.left_container.isVisible():
            self.toggle_left_panel(True)
        sizes = self.main_splitter.sizes()
        if len(sizes) >= 3:
            total_w = sum(sizes)
            curr_left = sizes[0]
            target_wide = min(850, max(600, int(total_w * 0.42)))
            if curr_left < 520:
                diff = target_wide - curr_left
                new_center = max(260, sizes[1] - diff)
                self.main_splitter.setSizes([target_wide, new_center, sizes[2]])
            else:
                target_norm = 380
                diff = curr_left - target_norm
                new_center = sizes[1] + diff
                self.main_splitter.setSizes([target_norm, new_center, sizes[2]])

    def _on_toggle_right_wide(self) -> None:
        """Alterna el ancho del panel derecho entre normal (~360px) y expandido (~720px o 42% del visor)."""
        if not hasattr(self, 'main_splitter') or not hasattr(self, 'right_container'):
            return
        if not self.right_container.isVisible():
            self.toggle_right_panel(True)
        sizes = self.main_splitter.sizes()
        if len(sizes) >= 3:
            total_w = sum(sizes)
            curr_right = sizes[2]
            target_wide = min(850, max(600, int(total_w * 0.42)))
            if curr_right < 520:
                diff = target_wide - curr_right
                new_center = max(260, sizes[1] - diff)
                self.main_splitter.setSizes([sizes[0], new_center, target_wide])
            else:
                target_norm = 360
                diff = curr_right - target_norm
                new_center = sizes[1] + diff
                self.main_splitter.setSizes([sizes[0], new_center, target_norm])

    def eventFilter(self, watched, event):
        """Captura eventos del sistema, como doble clic en asas del QSplitter para auto-expandir paneles."""
        if event.type() == QEvent.MouseButtonDblClick:
            if hasattr(self, 'main_splitter'):
                if watched == self.main_splitter.handle(1):
                    self._on_toggle_left_wide()
                    return True
                elif watched == self.main_splitter.handle(2):
                    self._on_toggle_right_wide()
                    return True
        return super().eventFilter(watched, event)

    def on_split_view_toggled(self, checked: bool) -> None:
        """Manejador al conmutar el modo comparador Antes / Después."""
        if checked and (self.canvas.before_pixmap is None or self.canvas.before_pixmap.isNull()):
            self._generate_before_reference()
        self.canvas.set_split_view_enabled(checked)
        self._update_split_orientation_ui()
        if checked and self.is_zoomed_100:
            self.update_stretch_preview()
        estado = "ACTIVADO (Arrastra la línea divisoria para comparar)" if checked else "DESACTIVADO"
        self.log_message(f"[CANVAS] Modo Comparador Antes / Después: {estado}")
        self._update_canvas_info_label()

    def on_set_before_clicked(self) -> None:
        """Congela el revelado actual como nueva imagen de referencia 'Antes'."""
        if self.canvas.base_pixmap and not self.canvas.base_pixmap.isNull():
            self.snapshot_before_pixmap = self.canvas.base_pixmap.copy()
            self.canvas.set_before_pixmap(self.snapshot_before_pixmap)
            self.initial_bp = self.spin_bp.value()
            self.initial_mtf = self.spin_mtf.value()
            self.initial_stretch_mode = getattr(self, "stretch_mode", "mtf")
            self.initial_ghs_sp = self.spin_ghs_sp.value() if hasattr(self, 'spin_ghs_sp') else 0.02
            self.initial_ghs_d = self.spin_ghs_d.value() if hasattr(self, 'spin_ghs_d') else 5.0
            self.initial_ghs_b = self.spin_ghs_b.value() if hasattr(self, 'spin_ghs_b') else 2.0
            self.initial_ghs_hp = self.spin_ghs_hp.value() if hasattr(self, 'spin_ghs_hp') else 0.0
            self.initial_ghs_lp = self.spin_ghs_lp.value() if hasattr(self, 'spin_ghs_lp') else 0.0
            self.initial_ghs_bp = self.spin_ghs_bp.value() if hasattr(self, 'spin_ghs_bp') else 0.0
            self.initial_ghs_color_mode = getattr(self, "ghs_color_mode", "luma")
            self.has_custom_snapshot = True
            self.log_message("[CANVAS] Estado actual fijado como nueva referencia de 'Antes'.")
            if not self.btn_split_view.isChecked():
                self.btn_split_view.setChecked(True)
        else:
            self.log_message("[AVISO] No hay ninguna imagen cargada en el lienzo para fijar.")

    def on_toolbar_zoom_toggled(self, checked: bool) -> None:
        """Manejador al conmutar el botón de zoom en la barra superior."""
        if self.is_zoomed_100 != checked:
            self.on_canvas_zoom_toggled(checked, 0.5, 0.5)

    def _generate_before_reference(self) -> None:
        """Genera el QPixmap base de referencia 'Antes' a partir de la toma cruda original."""
        if self.image_32bit is None:
            self.canvas.set_before_pixmap(None)
            return

        if self.has_custom_snapshot and self.snapshot_before_pixmap is not None and not self.is_zoomed_100:
            self.canvas.set_before_pixmap(self.snapshot_before_pixmap)
            return

        bp = self.initial_bp
        mtf = self.initial_mtf

        base_to_render, _ = self._compose_active_base(for_export=False)
        if base_to_render is not None:
            if getattr(self, "initial_stretch_mode", "mtf") == "ghs":
                stretched_init = apply_ghs_stretch(
                    base_to_render,
                    sp=getattr(self, "initial_ghs_sp", 0.02),
                    d=getattr(self, "initial_ghs_d", 5.0),
                    b=getattr(self, "initial_ghs_b", 2.0),
                    hp=getattr(self, "initial_ghs_hp", 0.0),
                    lp=getattr(self, "initial_ghs_lp", 0.0),
                    bp=getattr(self, "initial_ghs_bp", 0.0),
                    color_mode=getattr(self, "initial_ghs_color_mode", "luma")
                )
            else:
                stretched_init = manual_stretch(base_to_render, black_point=bp, midtone=mtf)
            self.canvas.set_before_image(stretched_init)

    def _update_canvas_info_label(self) -> None:
        """Actualiza la etiqueta informativa de resolución, herramienta activa y zoom en el pie del lienzo."""
        if hasattr(self, 'lbl_tool_badge'):
            if getattr(self.canvas, 'crop_mode', False):
                self.lbl_tool_badge.setText("● Modo: Recortar y Nivelar (C)")
                self.lbl_tool_badge.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'enable_masking', False):
                self.lbl_tool_badge.setText("● Modo: Máscara Suelo (M)")
                self.lbl_tool_badge.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'layer_brush_active', False):
                self.lbl_tool_badge.setText("● Modo: Pincel de Máscara")
                self.lbl_tool_badge.setStyleSheet("color: #a855f7; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'eyedropper_active', False):
                self.lbl_tool_badge.setText("● Modo: Pipeta de Muestreo (W)")
                self.lbl_tool_badge.setStyleSheet("color: #ffd54f; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'magnifier_active', False):
                self.lbl_tool_badge.setText("● Modo: Lupa de Inspección (Z)")
                self.lbl_tool_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
            elif getattr(self.canvas, 'split_view_enabled', False):
                ori_txt = "Vertical" if self.canvas.split_orientation == "vertical" else "Horizontal"
                self.lbl_tool_badge.setText(f"● Modo: Comparación Split ({ori_txt})")
                self.lbl_tool_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
            else:
                self.lbl_tool_badge.setText("● Modo: Vista General")
                self.lbl_tool_badge.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")

        if hasattr(self, 'lbl_canvas_zoom'):
            zoom_txt = "Zoom 100% (1:1)" if self.is_zoomed_100 else "Ajustar (Fit)"
            self.lbl_canvas_zoom.setText(f"Zoom: {zoom_txt}")

        if not hasattr(self, 'lbl_canvas_info'):
            return
        if self.image_32bit is None:
            self.lbl_canvas_info.setText("Sin imagen cargada")
            return
        h, w = self.image_32bit.shape[:2]
        mp = (w * h) / 1e6
        cs_labels = {
            "srgb": "sRGB",
            "adobe": "Adobe RGB (1998)",
            "prophoto": "ProPhoto RGB"
        }
        cs_name = cs_labels.get(getattr(self, 'current_color_space', 'srgb'), 'sRGB')
        self.lbl_canvas_info.setText(f"{w} × {h} px ({mp:.1f} MP)  |  32-bit Float  |  {cs_name}")

    def keyPressEvent(self, event) -> None:
        """Captura directa de teclas rápidas 'M', 'J', 'Y', 'Shift+Y', 'C' y controles interactivos."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox)):
            super().keyPressEvent(event)
            return

        if hasattr(self, 'canvas') and getattr(self.canvas, 'layer_move_active', False):
            if event.key() == Qt.Key_Escape:
                if hasattr(self, 'mask_layers_widget') and hasattr(self.mask_layers_widget, 'grp_transform'):
                    self.mask_layers_widget.grp_transform.btn_toggle_move.setChecked(False)
                else:
                    self.canvas.set_layer_move_enabled(False)
                event.accept()
                return

        if hasattr(self, 'canvas') and self.canvas.enable_masking:
            if event.key() == Qt.Key_Escape:
                if hasattr(self, 'btn_mask_tool'):
                    self.btn_mask_tool.setChecked(False)
                event.accept()
                return
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._compute_mask_from_scribbles()
                event.accept()
                return

        if self.canvas.crop_mode:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._apply_crop()
                event.accept()
                return
            elif event.key() == Qt.Key_Escape:
                self._cancel_crop()
                event.accept()
                return
            elif event.key() == Qt.Key_X:
                self._on_crop_flip_ratio()
                event.accept()
                return

        if event.key() == Qt.Key_M:
            self._on_toggle_mask_shortcut()
            event.accept()
            return
        elif event.key() == Qt.Key_C:
            self._on_toggle_crop_shortcut()
            event.accept()
            return
        elif event.key() == Qt.Key_J:
            self._on_toggle_clipping_shortcut()
            event.accept()
            return
        elif event.key() == Qt.Key_Y:
            if event.modifiers() & Qt.ShiftModifier:
                self._on_toggle_split_orientation_shortcut()
            else:
                self._on_toggle_split_shortcut()
            event.accept()
            return
        elif event.key() == Qt.Key_Tab:
            self.toggle_both_panels()
            event.accept()
            return
        elif event.key() in (Qt.Key_F7, Qt.Key_BracketLeft):
            self.toggle_left_panel()
            event.accept()
            return
        elif event.key() in (Qt.Key_F8, Qt.Key_BracketRight):
            self.toggle_right_panel()
            event.accept()
            return
        elif event.key() == Qt.Key_A:
            if hasattr(self, 'panel_dust'):
                cur = self.panel_dust.chk_visualize.isChecked()
                self.panel_dust.chk_visualize.setChecked(not cur)
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_layer_move_tool_toggled(self, active: bool) -> None:
        """Activa o desactiva la interacción directa de mover/redimensionar capa en el lienzo."""
        self.canvas.set_layer_move_enabled(active)
        if active:
            self.canvas.set_layer_brush_enabled(False)
            self.log_message("Modo Mover/Escalar Capa activo: Arrastra con el ratón para desplazar la posición, rueda para escalar.")

    def _on_canvas_layer_offset_dragged(self, dx_norm: float, dy_norm: float) -> None:
        """Aplica el desplazamiento arrastrado interactivamente desde el visor."""
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_layer_transform_from_canvas(dx_norm, dy_norm)

    def _on_canvas_layer_scale_wheel(self, factor: float) -> None:
        """Aplica el factor de escala interactivo con la rueda del ratón desde el visor."""
        if hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.set_layer_scale_from_canvas(factor)

    def _on_layer_gradient_toggled(self, active: bool, layer: Optional[Any]) -> None:
        """Activa o desactiva la interacción directa del gizmo de gradiente en el visor."""
        if active and layer is not None and getattr(layer, 'enabled', True) and getattr(layer, 'is_gradient', False):
            self.canvas.set_active_gradient_layer(layer)
        else:
            self.canvas.set_active_gradient_layer(None)

    def _on_canvas_gradient_drag_started(self) -> None:
        """Activa el modo de interacción ligera para gradientes en el visor."""
        if hasattr(self, 'async_pipeline'):
            self.async_pipeline.cancel_pending()
        self._is_dragging_gradient = True
        self._is_scrubbing_heavy_slider = True
        self._is_scrubbing_layer_slider = True

    def _on_canvas_gradient_drag_finished(self) -> None:
        """Restaura la fidelidad completa al finalizar el arrastre del gradiente y guarda en historial."""
        self._is_dragging_gradient = False
        self._is_scrubbing_heavy_slider = False
        self._is_scrubbing_layer_slider = False
        layer = getattr(self.canvas, 'active_gradient_layer', None)
        if layer is not None and hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.update_gradient_ui_from_layer(layer)
            self.mask_layers_widget.invalidate_layer_thumbnail(layer.id, rebuild_now=True)
        self.update_stretch_preview(immediate=True)
        self._commit_action()

    def _on_canvas_gradient_changed(self) -> None:
        """Se invoca en tiempo real al arrastrar un manejador del gradiente en el lienzo."""
        layer = getattr(self.canvas, 'active_gradient_layer', None)
        if layer is not None and hasattr(self, 'mask_layers_widget'):
            self.mask_layers_widget.update_gradient_ui_from_layer(layer)
        self.update_stretch_preview()

