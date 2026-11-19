# gui/dialog_presets.py
"""
gui/dialog_presets.py - Diálogos interactivos para guardar, importar, exportar y gestionar
los ajustes preestablecidos (Presets) de revelado en AstroLoom Studio Pro 2.0.
Diseño homogéneo con iconografía vectorial SVG, sin emojis y acabado Slate/Charcoal.
"""
import numpy as np
import cv2
from typing import Optional, Dict, Any, List
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QComboBox, QPushButton,
    QMessageBox, QFileDialog, QRadioButton, QButtonGroup,
    QSplitter, QWidget, QFrame, QSizePolicy, QScrollArea, QGridLayout,
    QSlider, QCheckBox
)

from core.developer_presets import (
    DeveloperPreset, get_builtin_presets, load_user_presets,
    save_user_preset, delete_user_preset, export_preset_file, import_preset_file,
    interpolate_preset_params,
    SCOPE_ALL, SCOPE_TONE, SCOPE_COLOR, SCOPE_DETAILS
)
from core.stretch import calculate_mtf_params
from gui.developer.developer_state import DeveloperState
from gui.developer.pipeline import apply_developer_pipeline
from gui.theme import (
    STYLE_RUN_IDLE, STYLE_TRANSFER_BTN, STYLE_STATUS_PILL,
    BG_WINDOW, BG_PANEL, BG_CARD, BG_INPUT, BORDER_BASE, BORDER_FOCUS,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_CYAN, ACCENT_GREEN, ACCENT_RED
)
from gui.icons import get_icon, get_svg_pixmap


class SavePresetDialog(QDialog):
    """Diálogo modal para guardar los ajustes actuales del revelador como un nuevo preset."""

    def __init__(self, current_params: Dict[str, Any], existing_categories: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_params = {k: v for k, v in current_params.items() if k not in ("mask_layers", "dust_spots")}
        self.existing_categories = existing_categories
        self.created_preset: Optional[DeveloperPreset] = None

        self.setWindowTitle("Guardar Ajustes Actuales como Preset")
        self.resize(500, 460)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f121d;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
            }
            QLineEdit, QComboBox, QTextEdit {
                background-color: #131724;
                border: 1px solid #273044;
                border-radius: 5px;
                color: #f1f5f9;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #38bdf8;
            }
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # Encabezado con icono SVG
        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        lbl_icon = QLabel()
        lbl_icon.setPixmap(get_svg_pixmap("save", color="#38bdf8", size=20))
        header_row.addWidget(lbl_icon)

        title_box = QVBoxLayout()
        lbl_header = QLabel("Guardar Nuevo Ajuste Preestablecido")
        lbl_header.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        lbl_sub = QLabel("Guarda todos los deslizadores, curvas y parámetros actuales para aplicarlos en otras imágenes.")
        lbl_sub.setWordWrap(True)
        lbl_sub.setStyleSheet("color: #94a3b8; font-size: 11px;")
        title_box.addWidget(lbl_header)
        title_box.addWidget(lbl_sub)
        header_row.addLayout(title_box)
        header_row.addStretch()
        layout.addLayout(header_row)

        # Formulario dentro de una tarjeta moderna
        form_card = QFrame()
        form_card.setStyleSheet("QFrame { background-color: #141724; border: 1px solid #222738; border-radius: 8px; }")
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(14, 14, 14, 14)
        form_layout.setSpacing(10)

        # Nombre
        lbl_n = QLabel("Nombre del Preset:*")
        lbl_n.setStyleSheet("font-weight: 600; font-size: 11px; color: #cbd5e1;")
        form_layout.addWidget(lbl_n)
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Ej: Vía Láctea - Verano Contraste Alto")
        form_layout.addWidget(self.txt_name)

        # Categoría
        lbl_c = QLabel("Categoría:")
        lbl_c.setStyleSheet("font-weight: 600; font-size: 11px; color: #cbd5e1;")
        form_layout.addWidget(lbl_c)
        self.combo_category = QComboBox()
        self.combo_category.setEditable(True)
        categories = sorted(list(set(self.existing_categories + ["Vía Láctea", "Paisaje Nocturno", "Cielo Profundo", "Usuario", "Especial"])))
        self.combo_category.addItems(categories)
        self.combo_category.setCurrentText("Usuario")
        form_layout.addWidget(self.combo_category)

        # Autor
        lbl_a = QLabel("Autor:")
        lbl_a.setStyleSheet("font-weight: 600; font-size: 11px; color: #cbd5e1;")
        form_layout.addWidget(lbl_a)
        self.txt_author = QLineEdit()
        self.txt_author.setPlaceholderText("Tu nombre o alias")
        self.txt_author.setText("Usuario")
        form_layout.addWidget(self.txt_author)

        # Descripción
        lbl_d = QLabel("Descripción / Notas (Opcional):")
        lbl_d.setStyleSheet("font-weight: 600; font-size: 11px; color: #cbd5e1;")
        form_layout.addWidget(lbl_d)
        self.txt_desc = QTextEdit()
        self.txt_desc.setPlaceholderText("Describe el tipo de lente, cámara, cielo o propósito para el que está pensado este preset...")
        self.txt_desc.setMaximumHeight(70)
        form_layout.addWidget(self.txt_desc)

        layout.addWidget(form_card)

        # Botones inferiores
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94a3b8;
                border: 1px solid #2b3245;
                border-radius: 5px;
                padding: 6px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #1e2433;
                color: #f1f5f9;
            }
        """)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Guardar Preset")
        self.btn_save.setIcon(get_icon("save", color="#ffffff", size=14))
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0ea5e9);
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 5px;
                font-weight: 600;
                padding: 6px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
            }
        """)
        self.btn_save.clicked.connect(self._on_save_clicked)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)

    def _on_save_clicked(self):
        name = self.txt_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Campo Obligatorio", "Por favor, introduce un nombre para el preset.")
            self.txt_name.setFocus()
            return

        category = self.combo_category.currentText().strip() or "Usuario"
        author = self.txt_author.text().strip() or "Usuario"
        description = self.txt_desc.toPlainText().strip()

        try:
            self.created_preset = save_user_preset(
                name=name,
                params=self.current_params,
                description=description,
                category=category,
                author=author
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar", f"No se pudo guardar el archivo del preset:\n{e}")


def numpy_to_qpixmap(arr: Optional[np.ndarray], width: int = 216, height: int = 115) -> QPixmap:
    """Convierte un array float32 [0..1] RGB a QPixmap con escalado nítido."""
    if arr is None:
        pix = QPixmap(width, height)
        pix.fill(QColor("#101420"))
        return pix

    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr_u8.ndim == 2:
        arr_u8 = np.repeat(arr_u8[..., np.newaxis], 3, axis=2)
    elif arr_u8.ndim == 3 and arr_u8.shape[2] == 1:
        arr_u8 = np.repeat(arr_u8[..., np.newaxis], 3, axis=2)

    arr_u8 = np.ascontiguousarray(arr_u8)
    h, w, ch = arr_u8.shape
    bytes_per_line = ch * w
    qimg = QImage(arr_u8.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)


def create_fallback_thumbnail(width: int = 216, height: int = 115, title: str = "") -> QPixmap:
    """Genera una miniatura gráfica decorativa cuando no hay imagen activa en el revelador."""
    pix = QPixmap(width, height)
    pix.fill(QColor("#0e121d"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    rng = np.random.RandomState(abs(hash(title)) % 10000)
    for _ in range(25):
        sx = int(rng.uniform(5, width - 5))
        sy = int(rng.uniform(5, height - 5))
        r = float(rng.uniform(0.8, 2.2))
        alpha = int(rng.uniform(100, 240))
        p.setBrush(QColor(226, 232, 240, alpha))
        p.setPen(Qt.NoPen)
        p.drawEllipse(sx, sy, int(r), int(r))
    p.end()
    return pix


class PresetCardWidget(QFrame):
    """Tarjeta interactiva para visualización de presets con miniatura gráfica y distintivos."""
    clicked = Signal(object)
    double_clicked = Signal(object)

    def __init__(self, preset: DeveloperPreset, thumb_pixmap: Optional[QPixmap] = None, parent=None):
        super().__init__(parent)
        self.preset = preset
        self._is_selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(230)
        self.setFixedHeight(215)
        self._setup_ui(thumb_pixmap)
        self._update_style()

    def _setup_ui(self, thumb_pixmap: Optional[QPixmap]):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)

        # 1. Contenedor de la miniatura
        self.lbl_thumb = QLabel()
        self.lbl_thumb.setFixedHeight(115)
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setStyleSheet("background-color: #0c0f18; border-radius: 5px;")
        if thumb_pixmap and not thumb_pixmap.isNull():
            self.lbl_thumb.setPixmap(thumb_pixmap)
        else:
            self.lbl_thumb.setPixmap(create_fallback_thumbnail(216, 115, self.preset.name))
        layout.addWidget(self.lbl_thumb)

        # 2. Badges de Requisitos y Origen
        row_badges = QHBoxLayout()
        row_badges.setSpacing(4)

        lbl_req = QLabel()
        if getattr(self.preset, "requires_starnet", False):
            lbl_req.setText("✨ StarNet")
            lbl_req.setStyleSheet("""
                background-color: rgba(251, 191, 36, 0.12);
                color: #fbbf24;
                border: 1px solid rgba(251, 191, 36, 0.35);
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 9px;
                font-weight: 700;
            """)
        elif getattr(self.preset, "requires_sky_mask", False):
            lbl_req.setText("🌄 Auto-Suelo")
            lbl_req.setStyleSheet("""
                background-color: rgba(74, 222, 128, 0.12);
                color: #4ade80;
                border: 1px solid rgba(74, 222, 128, 0.35);
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 9px;
                font-weight: 700;
            """)
        else:
            lbl_req.setText("⚡ Directo")
            lbl_req.setStyleSheet("""
                background-color: rgba(56, 189, 248, 0.12);
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.35);
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 9px;
                font-weight: 700;
            """)
        row_badges.addWidget(lbl_req)

        lbl_orig = QLabel("Fábrica" if self.preset.is_builtin else "Usuario")
        orig_color = "#94a3b8" if self.preset.is_builtin else "#e879f9"
        lbl_orig.setStyleSheet(f"""
            background-color: #111522;
            color: {orig_color};
            border: 1px solid #22293d;
            border-radius: 3px;
            padding: 1px 4px;
            font-size: 9px;
            font-weight: 600;
        """)
        row_badges.addWidget(lbl_orig)
        row_badges.addStretch()
        layout.addLayout(row_badges)

        # 3. Título del Preset
        self.lbl_title = QLabel(self.preset.name)
        self.lbl_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #f8fafc;")
        self.lbl_title.setWordWrap(True)
        self.lbl_title.setMaximumHeight(34)
        layout.addWidget(self.lbl_title)

        # 4. Categoría
        lbl_cat = QLabel(self.preset.category)
        lbl_cat.setStyleSheet("font-size: 10px; color: #64748b;")
        layout.addWidget(lbl_cat)
        layout.addStretch()

    def set_selected(self, selected: bool):
        self._is_selected = selected
        self._update_style()

    def _update_style(self):
        if self._is_selected:
            self.setStyleSheet("""
                PresetCardWidget {
                    background-color: #152238;
                    border: 2px solid #38bdf8;
                    border-radius: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                PresetCardWidget {
                    background-color: #111422;
                    border: 1px solid #20273a;
                    border-radius: 8px;
                }
                PresetCardWidget:hover {
                    background-color: #161b2c;
                    border: 1px solid #38bdf8;
                }
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.preset)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.preset)
        super().mouseDoubleClickEvent(event)


class PresetManagerDialog(QDialog):
    """Diálogo visual completo de galería, administración, importación y exportación de presets."""

    # Señal emitida cuando el usuario decide aplicar un preset: (preset, scope, intensity_pct)
    apply_preset_requested = Signal(object, str, float)
    # Señal de previsualización temporal en el lienzo principal
    preview_preset_requested = Signal(object, str, float)
    # Señal de cancelación de previsualización (restaura estado original)
    preview_cancelled = Signal()

    def __init__(self, current_params: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_params = current_params
        self.all_presets: List[DeveloperPreset] = []
        self.selected_preset: Optional[DeveloperPreset] = None
        self.card_widgets: List[PresetCardWidget] = []

        self.setWindowTitle("Galería y Gestor de Presets de Revelado")
        self.resize(1000, 660)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f121d;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
            }
        """)

        # Preparación de datos de miniatura basados en la imagen real del usuario
        self.raw_thumb_base = None
        self.thumb_mask = None
        self.thumb_starless = None
        self.thumb_stars = None
        self.thumb_bp = 0.0
        self.thumb_mtf = 0.5
        self._init_thumbnail_data()

        self._setup_ui()
        self._refresh_presets()

    def _init_thumbnail_data(self):
        parent_tab = self.parent()
        if parent_tab is None:
            return

        target_img = getattr(parent_tab, 'preview_proxy', None)
        if target_img is None:
            target_img = getattr(parent_tab, 'image_32bit', None)

        if target_img is None:
            return

        try:
            h, w = target_img.shape[:2]
            tw = 220
            th = max(60, int(h * (220.0 / float(w))))
            self.raw_thumb_base = cv2.resize(target_img, (tw, th), interpolation=cv2.INTER_AREA)

            curr_mask = getattr(parent_tab, 'current_mask', None)
            if curr_mask is not None:
                self.thumb_mask = cv2.resize(curr_mask, (tw, th), interpolation=cv2.INTER_LINEAR)

            curr_sl = getattr(parent_tab, 'image_starless', None)
            if curr_sl is not None:
                self.thumb_starless = cv2.resize(curr_sl, (tw, th), interpolation=cv2.INTER_AREA)

            curr_st = getattr(parent_tab, 'image_stars', None)
            if curr_st is not None:
                self.thumb_stars = cv2.resize(curr_st, (tw, th), interpolation=cv2.INTER_AREA)

            st_mtf = getattr(getattr(parent_tab, 'state', None), 'mtf_val', 0.5)
            st_bp = getattr(getattr(parent_tab, 'state', None), 'bp_val', 0.0)
            if st_mtf >= 0.499 and st_bp == 0.0:
                bp_calc, m_calc = calculate_mtf_params(self.raw_thumb_base, target_background=0.20)
                self.thumb_bp = bp_calc
                self.thumb_mtf = m_calc
            else:
                self.thumb_bp = st_bp
                self.thumb_mtf = st_mtf
        except Exception as e:
            print(f"[PresetManagerDialog] Error preparando base de miniaturas: {e}")

    def _generate_thumbnail_for_preset(self, preset: DeveloperPreset) -> QPixmap:
        if self.raw_thumb_base is None:
            return create_fallback_thumbnail(216, 115, preset.name)

        try:
            st = DeveloperState()
            for k, v in self.current_params.items():
                if hasattr(st, k):
                    setattr(st, k, v)
            for k, v in preset.params.items():
                if hasattr(st, k):
                    setattr(st, k, v)

            st.bp_val = self.thumb_bp
            st.mtf_val = self.thumb_mtf

            res = apply_developer_pipeline(
                self.raw_thumb_base,
                st,
                precomputed_mask=self.thumb_mask,
                image_starless=self.thumb_starless,
                image_stars=self.thumb_stars,
                is_preview=True
            )
            return numpy_to_qpixmap(res, 216, 115)
        except Exception:
            return create_fallback_thumbnail(216, 115, preset.name)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Encabezado con icono SVG
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        lbl_icon = QLabel()
        lbl_icon.setPixmap(get_svg_pixmap("sparkles", color="#38bdf8", size=24))
        header_layout.addWidget(lbl_icon)

        title_box = QVBoxLayout()
        lbl_header = QLabel("Galería Visual y Gestor de Presets")
        lbl_header.setStyleSheet("font-size: 16px; font-weight: 700; color: #f8fafc;")
        lbl_sub = QLabel("Explora miniaturas sobre tu imagen, modula la intensidad del estilo y previsualiza en tiempo real")
        lbl_sub.setStyleSheet("font-size: 11px; color: #94a3b8;")
        title_box.addWidget(lbl_header)
        title_box.addWidget(lbl_sub)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Divisor principal horizontal
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1a1e2c;
                width: 6px;
            }
            QSplitter::handle:hover {
                background-color: #38bdf8;
            }
        """)

        # --- LADO IZQUIERDO: Catálogo y Vistas (Tarjetas / Tabla) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Barra de Filtros y Conmutador de Vista
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)

        self.txt_filter = QLineEdit()
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setPlaceholderText("Buscar preset por nombre...")
        self.txt_filter.setStyleSheet("""
            QLineEdit {
                background-color: #131724;
                border: 1px solid #242c3e;
                border-radius: 6px;
                padding: 6px 10px;
                color: #f1f5f9;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #38bdf8;
            }
        """)
        self.txt_filter.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.txt_filter, 1)

        self.combo_cat_filter = QComboBox()
        self.combo_cat_filter.addItem("Todas las categorías")
        self.combo_cat_filter.setStyleSheet("""
            QComboBox {
                background-color: #131724;
                border: 1px solid #242c3e;
                border-radius: 6px;
                padding: 6px 10px;
                color: #f1f5f9;
                font-size: 12px;
            }
            QComboBox:focus {
                border-color: #38bdf8;
            }
        """)
        self.combo_cat_filter.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.combo_cat_filter)

        left_layout.addLayout(filter_layout)

        # Galería de Tarjetas con Miniaturas
        self.scroll_cards = QScrollArea()
        self.scroll_cards.setWidgetResizable(True)
        self.scroll_cards.setStyleSheet("""
            QScrollArea {
                background-color: #0b0e17;
                border: 1px solid #1e2538;
                border-radius: 6px;
            }
        """)
        self.w_cards_container = QWidget()
        self.w_cards_container.setStyleSheet("background-color: transparent;")
        self.grid_cards = QGridLayout(self.w_cards_container)
        self.grid_cards.setContentsMargins(10, 10, 10, 10)
        self.grid_cards.setSpacing(12)
        self.scroll_cards.setWidget(self.w_cards_container)
        left_layout.addWidget(self.scroll_cards, 1)

        # Botones bajo el catálogo
        tbl_btn_layout = QHBoxLayout()
        tbl_btn_layout.setSpacing(8)

        btn_style = """
            QPushButton {
                background-color: #141724;
                border: 1px solid #242c3e;
                border-radius: 5px;
                padding: 6px 12px;
                color: #cbd5e1;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1e2434;
                border-color: #38bdf8;
                color: #f8fafc;
            }
            QPushButton:disabled {
                background-color: transparent;
                border-color: #1c2230;
                color: #475569;
            }
        """

        self.btn_import = QPushButton("Importar...")
        self.btn_import.setIcon(get_icon("upload", color="#94a3b8", hover_color="#38bdf8", size=13))
        self.btn_import.setStyleSheet(btn_style)
        self.btn_import.setToolTip("Carga un archivo .json de preset desde tu disco")
        self.btn_import.clicked.connect(self._on_import_clicked)
        tbl_btn_layout.addWidget(self.btn_import)

        self.btn_export = QPushButton("Exportar...")
        self.btn_export.setIcon(get_icon("download", color="#94a3b8", hover_color="#38bdf8", size=13))
        self.btn_export.setStyleSheet(btn_style)
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("Exporta el preset seleccionado a un archivo .json para compartirlo")
        self.btn_export.clicked.connect(self._on_export_clicked)
        tbl_btn_layout.addWidget(self.btn_export)

        self.btn_delete = QPushButton("Eliminar")
        self.btn_delete.setIcon(get_icon("trash", color="#f87171", size=13))
        self.btn_delete.setEnabled(False)
        self.btn_delete.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.06);
                border: 1px solid rgba(248, 113, 113, 0.25);
                border-radius: 5px;
                padding: 6px 12px;
                color: #f87171;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(248, 113, 113, 0.16);
                border-color: #f87171;
            }
            QPushButton:disabled {
                background: transparent;
                border-color: #1c2230;
                color: #475569;
            }
        """)
        self.btn_delete.setToolTip("Elimina permanentemente el preset de usuario seleccionado")
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        tbl_btn_layout.addWidget(self.btn_delete)

        left_layout.addLayout(tbl_btn_layout)
        splitter.addWidget(left_widget)

        # --- LADO DERECHO: Detalle, Intensidad y Aplicación ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # Tarjeta 1: Detalles del Preset
        card_detail = QFrame()
        card_detail.setStyleSheet("QFrame { background-color: #141724; border: 1px solid #222738; border-radius: 8px; }")
        detail_layout = QVBoxLayout(card_detail)
        detail_layout.setContentsMargins(12, 12, 12, 12)
        detail_layout.setSpacing(8)

        lbl_sec_detail = QLabel("DETALLES DEL PRESET")
        lbl_sec_detail.setStyleSheet("font-weight: 700; font-size: 10px; color: #80d8ff; letter-spacing: 0.6px;")
        detail_layout.addWidget(lbl_sec_detail)

        self.lbl_detail_name = QLabel("Selecciona un preset")
        self.lbl_detail_name.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
        self.lbl_detail_name.setWordWrap(True)
        detail_layout.addWidget(self.lbl_detail_name)

        self.lbl_detail_badge = QLabel("")
        self.lbl_detail_badge.setStyleSheet("""
            background-color: #101420;
            border: 1px solid #242d40;
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: 600;
            color: #38bdf8;
        """)
        detail_layout.addWidget(self.lbl_detail_badge)

        self.lbl_detail_desc = QLabel("Elige un preset de la lista para ver su descripción y modular su intensidad.")
        self.lbl_detail_desc.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.3;")
        self.lbl_detail_desc.setWordWrap(True)
        detail_layout.addWidget(self.lbl_detail_desc)

        # Tarjeta 1b: Requisitos de Capas del Preset
        self.card_reqs = QFrame()
        self.card_reqs.setStyleSheet("background-color: #0d101a; border: 1px solid #1c2233; border-radius: 6px; padding: 6px;")
        reqs_layout = QVBoxLayout(self.card_reqs)
        reqs_layout.setContentsMargins(6, 6, 6, 6)
        reqs_layout.setSpacing(4)

        lbl_reqs_title = QLabel("REQUISITOS DE CAPAS")
        lbl_reqs_title.setStyleSheet("font-size: 9px; font-weight: 700; color: #64748b; letter-spacing: 0.5px;")
        reqs_layout.addWidget(lbl_reqs_title)

        self.lbl_req_sky = QLabel("● Cielo/Suelo: No requerida")
        self.lbl_req_sky.setStyleSheet("font-size: 11px; color: #94a3b8;")
        reqs_layout.addWidget(self.lbl_req_sky)

        self.lbl_req_starnet = QLabel("● StarNet AI: No requerida")
        self.lbl_req_starnet.setStyleSheet("font-size: 11px; color: #94a3b8;")
        reqs_layout.addWidget(self.lbl_req_starnet)

        detail_layout.addWidget(self.card_reqs)
        right_layout.addWidget(card_detail)

        # Tarjeta 2: Control de Intensidad del Preset
        card_intensity = QFrame()
        card_intensity.setStyleSheet("QFrame { background-color: #141724; border: 1px solid #222738; border-radius: 8px; }")
        int_layout = QVBoxLayout(card_intensity)
        int_layout.setContentsMargins(12, 10, 12, 10)
        int_layout.setSpacing(6)

        lbl_sec_int = QLabel("INTENSIDAD DEL PRESET")
        lbl_sec_int.setStyleSheet("font-weight: 700; font-size: 10px; color: #80d8ff; letter-spacing: 0.6px;")
        int_layout.addWidget(lbl_sec_int)

        row_int_hdr = QHBoxLayout()
        lbl_int_title = QLabel("Escala Global:")
        lbl_int_title.setStyleSheet("font-size: 11px; color: #cbd5e1;")
        row_int_hdr.addWidget(lbl_int_title)

        self.lbl_intensity_val = QLabel("100%")
        self.lbl_intensity_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #38bdf8;")
        row_int_hdr.addWidget(self.lbl_intensity_val)
        row_int_hdr.addStretch()

        self.btn_reset_int = QPushButton("100%")
        self.btn_reset_int.setCursor(Qt.PointingHandCursor)
        self.btn_reset_int.setStyleSheet("""
            QPushButton {
                background: #182030;
                border: 1px solid #29354d;
                border-radius: 3px;
                color: #94a3b8;
                font-size: 10px;
                padding: 2px 6px;
            }
            QPushButton:hover {
                color: #38bdf8;
                border-color: #38bdf8;
            }
        """)
        self.btn_reset_int.clicked.connect(lambda: self.slider_intensity.setValue(100))
        row_int_hdr.addWidget(self.btn_reset_int)
        int_layout.addLayout(row_int_hdr)

        self.slider_intensity = QSlider(Qt.Horizontal)
        self.slider_intensity.setRange(0, 150)
        self.slider_intensity.setValue(100)
        self.slider_intensity.valueChanged.connect(self._on_intensity_slider_changed)
        int_layout.addWidget(self.slider_intensity)

        right_layout.addWidget(card_intensity)

        # Tarjeta 3: Modo de Aplicación
        card_scope = QFrame()
        card_scope.setStyleSheet("QFrame { background-color: #141724; border: 1px solid #222738; border-radius: 8px; }")
        scope_layout = QVBoxLayout(card_scope)
        scope_layout.setContentsMargins(12, 10, 12, 10)
        scope_layout.setSpacing(5)

        lbl_sec_scope = QLabel("MODO DE APLICACIÓN")
        lbl_sec_scope.setStyleSheet("font-weight: 700; font-size: 10px; color: #80d8ff; letter-spacing: 0.6px;")
        scope_layout.addWidget(lbl_sec_scope)

        self.scope_group = QButtonGroup(self)
        self.rad_all = QRadioButton("Todo (Sin alterar estirado MTF ni punto negro)")
        self.rad_all.setChecked(True)
        self.rad_tone = QRadioButton("Solo Contraste y Suelo (Contraste, EV y sombras)")
        self.rad_color = QRadioButton("Solo Color y WB (Temp, Tinte, Sat, Vibración)")
        self.rad_details = QRadioButton("Solo Estrellas y Detalles (Claridad, Dehaze, StarNet)")

        radio_style = "QRadioButton { color: #cbd5e1; font-size: 11px; } QRadioButton:hover { color: #f8fafc; }"
        self.rad_all.setStyleSheet(radio_style)
        self.rad_tone.setStyleSheet(radio_style)
        self.rad_color.setStyleSheet(radio_style)
        self.rad_details.setStyleSheet(radio_style)

        self.scope_group.addButton(self.rad_all, 0)
        self.scope_group.addButton(self.rad_tone, 1)
        self.scope_group.addButton(self.rad_color, 2)
        self.scope_group.addButton(self.rad_details, 3)
        self.scope_group.idClicked.connect(lambda: self._emit_preview())

        scope_layout.addWidget(self.rad_all)
        scope_layout.addWidget(self.rad_tone)
        scope_layout.addWidget(self.rad_color)
        scope_layout.addWidget(self.rad_details)

        right_layout.addWidget(card_scope)

        # Checkbox de Previsualización en Vivo
        self.chk_live_preview = QCheckBox("Previsualizar en el lienzo principal en tiempo real")
        self.chk_live_preview.setChecked(True)
        self.chk_live_preview.setStyleSheet("QCheckBox { color: #7dd3fc; font-size: 11px; font-weight: 600; }")
        self.chk_live_preview.toggled.connect(self._on_live_preview_toggled)
        right_layout.addWidget(self.chk_live_preview)

        # Botón Aplicar Preset Pro
        self.btn_apply = QPushButton("Aplicar Preset a la Imagen")
        self.btn_apply.setIcon(get_icon("sparkles", color="#ffffff", size=15))
        self.btn_apply.setEnabled(False)
        self.btn_apply.setFixedHeight(38)
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0ea5e9);
                color: #ffffff;
                font-weight: 700;
                font-size: 12px;
                border: 1px solid #38bdf8;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
                border-color: #7dd3fc;
            }
            QPushButton:disabled {
                background-color: #161c28;
                color: #475569;
                border-color: #242c3b;
            }
        """)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        right_layout.addWidget(self.btn_apply)

        splitter.addWidget(right_widget)
        splitter.setSizes([600, 380])
        layout.addWidget(splitter, 1)

        # Barra inferior de cierre
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        self.btn_close = QPushButton("Cerrar / Cancelar")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #141724;
                border: 1px solid #273044;
                border-radius: 5px;
                padding: 6px 18px;
                color: #94a3b8;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1e2433;
                color: #f1f5f9;
                border-color: #38bdf8;
            }
        """)
        self.btn_close.clicked.connect(self.reject)
        close_layout.addWidget(self.btn_close)
        layout.addLayout(close_layout)

    def _refresh_presets(self):
        """Recarga la lista de presets, genera miniaturas y sincroniza las vistas."""
        builtins = get_builtin_presets()
        user_presets = load_user_presets()
        self.all_presets = builtins + user_presets

        # Limpiar tarjetas anteriores
        for card in self.card_widgets:
            card.deleteLater()
        self.card_widgets.clear()

        # Actualizar filtro de categorías
        current_cat = self.combo_cat_filter.currentText()
        self.combo_cat_filter.blockSignals(True)
        self.combo_cat_filter.clear()
        self.combo_cat_filter.addItem("Todas las categorías")
        categories = sorted(list(set(p.category for p in self.all_presets)))
        for cat in categories:
            self.combo_cat_filter.addItem(cat)
        if current_cat in categories:
            self.combo_cat_filter.setCurrentText(current_cat)
        self.combo_cat_filter.blockSignals(False)

        # Construir cuadrícula de tarjetas con miniaturas renderizadas
        cols = 2
        for idx, preset in enumerate(self.all_presets):
            thumb_pix = self._generate_thumbnail_for_preset(preset)
            card = PresetCardWidget(preset, thumb_pixmap=thumb_pix, parent=self.w_cards_container)
            card.clicked.connect(self._on_card_clicked)
            card.double_clicked.connect(self._on_card_double_clicked)
            r, c = divmod(idx, cols)
            self.grid_cards.addWidget(card, r, c)
            self.card_widgets.append(card)

        self._apply_filter()

    def _apply_filter(self):
        """Filtra las tarjetas visuales según búsqueda y categoría."""
        search = self.txt_filter.text().strip().lower()
        cat_filter = self.combo_cat_filter.currentText()

        cols = 2
        visible_cards = []
        visible_idx = 0
        for card in self.card_widgets:
            p = card.preset
            match = True
            if cat_filter != "Todas las categorías" and p.category != cat_filter:
                match = False
            elif search and search not in p.name.lower() and search not in p.description.lower():
                match = False

            card.setVisible(match)
            if match:
                self.grid_cards.removeWidget(card)
                r, c = divmod(visible_idx, cols)
                self.grid_cards.addWidget(card, r, c)
                visible_idx += 1
                visible_cards.append(card)

        # Seleccionar el preset actualmente seleccionado si sigue visible, o el primero visible
        if visible_cards:
            if self.selected_preset and any(c.preset.id == self.selected_preset.id for c in visible_cards):
                self._select_preset(self.selected_preset)
            else:
                self._select_preset(visible_cards[0].preset)
        else:
            self._clear_selection()

    def _on_card_clicked(self, preset: DeveloperPreset):
        self._select_preset(preset)

    def _on_card_double_clicked(self, preset: DeveloperPreset):
        self._select_preset(preset)
        self._on_apply_clicked()

    def _select_preset(self, preset: DeveloperPreset, sync_table: bool = True):
        self.selected_preset = preset

        # Sincronizar tarjetas
        for card in self.card_widgets:
            card.set_selected(card.preset.id == preset.id)

        # Actualizar panel de detalles
        self.lbl_detail_name.setText(preset.name)
        origin_text = "Oficial de Fábrica" if preset.is_builtin else f"Usuario ({preset.author})"
        self.lbl_detail_badge.setText(f"{preset.category}  ·  {origin_text}")
        self.lbl_detail_badge.setVisible(True)
        self.lbl_detail_desc.setText(preset.description or "Sin descripción proporcionada.")

        # Requisitos de procesamiento
        parent_tab = self.parent()
        has_ground_mask = getattr(parent_tab, 'current_mask', None) is not None if parent_tab else False
        has_starnet = getattr(parent_tab, 'image_starless', None) is not None if parent_tab else False

        if getattr(preset, "requires_sky_mask", False):
            if has_ground_mask:
                self.lbl_req_sky.setText("● Cielo/Suelo: Máscara activa detectada ✓")
                self.lbl_req_sky.setStyleSheet("font-size: 11px; color: #4ade80; font-weight: 600;")
            else:
                self.lbl_req_sky.setText("● Cielo/Suelo: Se auto-detectará con IA (<0.5s) ⚡")
                self.lbl_req_sky.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: 600;")
        else:
            self.lbl_req_sky.setText("● Cielo/Suelo: No requerida")
            self.lbl_req_sky.setStyleSheet("font-size: 11px; color: #64748b;")

        if getattr(preset, "requires_starnet", False):
            if has_starnet:
                self.lbl_req_starnet.setText("● StarNet AI: Capas listas ✓")
                self.lbl_req_starnet.setStyleSheet("font-size: 11px; color: #4ade80; font-weight: 600;")
            else:
                self.lbl_req_starnet.setText("● StarNet AI: Requiere desacoplo previo (~15s) ⚠️")
                self.lbl_req_starnet.setStyleSheet("font-size: 11px; color: #fbbf24; font-weight: 600;")
        else:
            self.lbl_req_starnet.setText("● StarNet AI: No requerida")
            self.lbl_req_starnet.setStyleSheet("font-size: 11px; color: #64748b;")

        self.btn_export.setEnabled(True)
        self.btn_delete.setEnabled(not preset.is_builtin)
        self.btn_apply.setEnabled(True)

        self._emit_preview()

    def _clear_selection(self):
        self.selected_preset = None
        self.lbl_detail_name.setText("Ningún preset seleccionado")
        self.lbl_detail_badge.setVisible(False)
        self.lbl_detail_desc.setText("Elige un preset de la lista o cuadrícula.")
        self.lbl_req_sky.setText("● Cielo/Suelo: -")
        self.lbl_req_starnet.setText("● StarNet AI: -")
        self.btn_export.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.preview_cancelled.emit()

    def _on_intensity_slider_changed(self, val: int):
        self.lbl_intensity_val.setText(f"{val}%")
        self._emit_preview()

    def _on_live_preview_toggled(self, checked: bool):
        if checked:
            self._emit_preview()
        else:
            self.preview_cancelled.emit()

    def _emit_preview(self):
        if not self.selected_preset or not getattr(self, "chk_live_preview", None):
            return
        if not self.chk_live_preview.isChecked():
            return
        scope = self._get_current_scope()
        intensity = float(self.slider_intensity.value())
        self.preview_preset_requested.emit(self.selected_preset, scope, intensity)

    def _get_current_scope(self) -> str:
        btn_id = self.scope_group.checkedId()
        if btn_id == 1:
            return SCOPE_TONE
        elif btn_id == 2:
            return SCOPE_COLOR
        elif btn_id == 3:
            return SCOPE_DETAILS
        return SCOPE_ALL

    def _on_apply_clicked(self):
        if not self.selected_preset:
            return
        scope = self._get_current_scope()
        intensity = float(self.slider_intensity.value())
        self.apply_preset_requested.emit(self.selected_preset, scope, intensity)
        self.accept()

    def reject(self):
        self.preview_cancelled.emit()
        super().reject()

    def _on_export_clicked(self):
        if not self.selected_preset:
            return
        safe_name = "".join(c for c in self.selected_preset.name if c.isalnum() or c in (" ", "_", "-")).strip()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Preset de Revelado",
            f"{safe_name}.json",
            "Archivos JSON (*.json)"
        )
        if filename:
            try:
                export_preset_file(self.selected_preset, filename)
                QMessageBox.information(self, "Exportación Exitosa", f"El preset fue exportado correctamente en:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error al Exportar", f"No se pudo exportar el preset:\n{e}")

    def _on_import_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Importar Preset de Revelado",
            "",
            "Archivos JSON (*.json)"
        )
        if filename:
            try:
                imported = import_preset_file(filename)
                QMessageBox.information(self, "Preset Importado", f"Se importó con éxito el preset '{imported.name}'.")
                self._refresh_presets()
            except Exception as e:
                QMessageBox.critical(self, "Error al Importar", f"No se pudo importar el archivo:\n{e}")

    def _on_delete_clicked(self):
        if not self.selected_preset or self.selected_preset.is_builtin:
            return
        ans = QMessageBox.question(
            self,
            "Confirmar Eliminación",
            f"¿Estás seguro de que deseas eliminar permanentemente el preset '{self.selected_preset.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if ans == QMessageBox.Yes:
            delete_user_preset(self.selected_preset.id)
            self._refresh_presets()
