# gui/dialog_presets.py
"""
gui/dialog_presets.py - Diálogos interactivos para guardar, importar, exportar y gestionar
los ajustes preestablecidos (Presets) de revelado en AstroLoom Studio Pro 2.0.
Diseño homogéneo con iconografía vectorial SVG, sin emojis y acabado Slate/Charcoal.
"""
from typing import Optional, Dict, Any, List
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QRadioButton, QButtonGroup,
    QSplitter, QWidget, QFrame, QSizePolicy
)

from core.developer_presets import (
    DeveloperPreset, get_builtin_presets, load_user_presets,
    save_user_preset, delete_user_preset, export_preset_file, import_preset_file,
    SCOPE_ALL, SCOPE_TONE, SCOPE_COLOR, SCOPE_DETAILS
)
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
        self.current_params = current_params
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


class PresetManagerDialog(QDialog):
    """Diálogo completo de administración, importación, exportación y borrado de presets."""

    # Señal emitida cuando el usuario decide aplicar un preset desde el gestor: (preset, scope)
    apply_preset_requested = Signal(object, str)

    def __init__(self, current_params: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_params = current_params
        self.all_presets: List[DeveloperPreset] = []
        self.selected_preset: Optional[DeveloperPreset] = None

        self.setWindowTitle("Gestor de Ajustes Preestablecidos (Presets)")
        self.resize(840, 560)
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

        self._setup_ui()
        self._refresh_presets()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Encabezado con icono SVG
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        lbl_icon = QLabel()
        lbl_icon.setPixmap(get_svg_pixmap("sliders", color="#38bdf8", size=22))
        header_layout.addWidget(lbl_icon)

        title_box = QVBoxLayout()
        lbl_header = QLabel("Gestor de Presets de Revelado")
        lbl_header.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        lbl_sub = QLabel("Administra, inspecciona, importa y exporta estilos de procesado astronómico")
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

        # --- LADO IZQUIERDO: Tabla de Presets ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Filtro de búsqueda / categoría
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)

        self.txt_filter = QLineEdit()
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setPlaceholderText("Buscar por nombre o descripción...")
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

        # Tabla de Presets Pro
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Nombre del Preset", "Categoría", "Origen"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #121520;
                border: 1px solid #222738;
                border-radius: 6px;
                color: #e2e8f0;
                selection-background-color: rgba(56, 189, 248, 0.15);
                selection-color: #38bdf8;
                outline: none;
            }
            QTableWidget::item {
                padding: 5px 8px;
                border-bottom: 1px solid #181d2a;
            }
            QTableWidget::item:selected {
                background-color: rgba(56, 189, 248, 0.15);
                color: #38bdf8;
            }
            QHeaderView::section {
                background-color: #161a27;
                color: #94a3b8;
                font-weight: 600;
                font-size: 11px;
                padding: 6px 8px;
                border: none;
                border-bottom: 1px solid #242c3f;
            }
        """)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.doubleClicked.connect(self._on_table_double_clicked)
        left_layout.addWidget(self.table)

        # Botones bajo la tabla
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

        # --- LADO DERECHO: Detalle del Preset y Aplicación ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # Tarjeta de Detalles del Preset
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

        self.lbl_detail_desc = QLabel("Elige un preset de la lista para ver su descripción y parámetros.")
        self.lbl_detail_desc.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.3;")
        self.lbl_detail_desc.setWordWrap(True)
        detail_layout.addWidget(self.lbl_detail_desc)

        # Resumen de Parámetros Clave
        lbl_params_hdr = QLabel("Parámetros Configurados:")
        lbl_params_hdr.setStyleSheet("font-weight: 600; font-size: 11px; color: #cbd5e1; margin-top: 4px;")
        detail_layout.addWidget(lbl_params_hdr)

        self.txt_params_summary = QTextEdit()
        self.txt_params_summary.setReadOnly(True)
        self.txt_params_summary.setStyleSheet("""
            QTextEdit {
                background-color: #0b0e15;
                border: 1px solid #1a202c;
                border-radius: 6px;
                color: #94a3b8;
                font-family: monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        detail_layout.addWidget(self.txt_params_summary)

        right_layout.addWidget(card_detail, 1)

        # Tarjeta de Modo de Aplicación
        card_scope = QFrame()
        card_scope.setStyleSheet("QFrame { background-color: #141724; border: 1px solid #222738; border-radius: 8px; }")
        scope_layout = QVBoxLayout(card_scope)
        scope_layout.setContentsMargins(12, 10, 12, 10)
        scope_layout.setSpacing(6)

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

        scope_layout.addWidget(self.rad_all)
        scope_layout.addWidget(self.rad_tone)
        scope_layout.addWidget(self.rad_color)
        scope_layout.addWidget(self.rad_details)

        right_layout.addWidget(card_scope)

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
            QPushButton:pressed {
                background-color: #075985;
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
        splitter.setSizes([470, 330])
        layout.addWidget(splitter, 1)

        # Barra inferior de cierre
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        btn_close = QPushButton("Cerrar")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("""
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
        btn_close.clicked.connect(self.accept)
        close_layout.addWidget(btn_close)
        layout.addLayout(close_layout)

    def _refresh_presets(self):
        """Recarga la lista completa combinando built-in y presets de usuario."""
        builtins = get_builtin_presets()
        user_presets = load_user_presets()
        self.all_presets = builtins + user_presets

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

        self._apply_filter()

    def _apply_filter(self):
        """Filtra las filas de la tabla según texto y categoría."""
        search = self.txt_filter.text().strip().lower()
        cat_filter = self.combo_cat_filter.currentText()

        filtered = []
        for p in self.all_presets:
            if cat_filter != "Todas las categorías" and p.category != cat_filter:
                continue
            if search and search not in p.name.lower() and search not in p.description.lower():
                continue
            filtered.append(p)

        self.table.setRowCount(len(filtered))
        for row, p in enumerate(filtered):
            # Nombre con icono vectorial sutil
            item_name = QTableWidgetItem(p.name)
            item_name.setIcon(get_icon("sliders" if p.is_builtin else "file", color="#38bdf8" if p.is_builtin else "#fbbf24", size=13))
            item_name.setData(Qt.UserRole, p)

            # Categoría
            item_cat = QTableWidgetItem(p.category)
            item_cat.setForeground(QColor("#94a3b8"))

            # Origen: Texto limpio sin emojis
            origin_str = "Fábrica" if p.is_builtin else "Usuario"
            item_origin = QTableWidgetItem(origin_str)
            item_origin.setTextAlignment(Qt.AlignCenter)
            if p.is_builtin:
                item_origin.setForeground(QColor("#38bdf8"))
            else:
                item_origin.setForeground(QColor("#fbbf24"))

            self.table.setItem(row, 0, item_name)
            self.table.setItem(row, 1, item_cat)
            self.table.setItem(row, 2, item_origin)

        if len(filtered) > 0:
            self.table.selectRow(0)
        else:
            self._on_selection_changed()

    def _on_selection_changed(self):
        """Actualiza el panel de detalles según el preset seleccionado."""
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.selected_preset = None
            self.lbl_detail_name.setText("Ningún preset seleccionado")
            self.lbl_detail_badge.setText("")
            self.lbl_detail_badge.setVisible(False)
            self.lbl_detail_desc.setText("Elige un preset de la lista para ver su descripción y parámetros.")
            self.txt_params_summary.clear()
            self.btn_export.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.btn_apply.setEnabled(False)
            return

        row = self.table.currentRow()
        item = self.table.item(row, 0)
        preset: DeveloperPreset = item.data(Qt.UserRole)
        self.selected_preset = preset

        self.lbl_detail_name.setText(preset.name)
        origin_text = "Oficial de Fábrica" if preset.is_builtin else f"Usuario ({preset.author})"
        self.lbl_detail_badge.setText(f"{preset.category}  ·  {origin_text}")
        self.lbl_detail_badge.setVisible(True)
        self.lbl_detail_desc.setText(preset.description or "Sin descripción proporcionada.")

        # Construir resumen de parámetros
        lines = []
        params = preset.params
        if "contrast_val" in params or "gnd_ev_val" in params or "gnd_sh_val" in params:
            lines.append("--- CONTRASTE & SUELO (Sin estirado) ---")
            lines.append(f"  • Contraste: {params.get('contrast_val', 0)/100.0:+.2f}")
            if params.get("gnd_ev_val", 0) != 0:
                lines.append(f"  • Suelo EV: {params.get('gnd_ev_val', 0)/100.0:+.2f} EV")
            if params.get("gnd_sh_val", 0) != 0:
                lines.append(f"  • Sombras Suelo: {params.get('gnd_sh_val', 0)}%")
            if params.get("gnd_bp_val", 0) != 0:
                lines.append(f"  • Suelo BP: {params.get('gnd_bp_val', 0)/1000.0:+.3f}")
        if "temp_val" in params or "tint_val" in params:
            lines.append("--- COLOR & BALANCE ---")
            lines.append(f"  • Temp: {params.get('temp_val', 0)/1000.0:+.3f}")
            lines.append(f"  • Tinte: {params.get('tint_val', 0)/1000.0:+.3f}")
            lines.append(f"  • Sat Cielo: {params.get('sat_sky_val', 100)/100.0:.2f}x")
            lines.append(f"  • Vibración: {params.get('vibrance_val', 0)/100.0:+.2f}")
        if "stars_val" in params or "clarity_val" in params:
            lines.append("--- ESTRELLAS & FILTROS ---")
            lines.append(f"  • Estrellas: {params.get('stars_val', 100)}%")
            lines.append(f"  • Claridad: {params.get('clarity_val', 0)/100.0:+.2f}")
            lines.append(f"  • Dehaze: {params.get('dehaze_val', 0)/100.0:+.2f}")
            lines.append(f"  • Multiescala Estructura: {params.get('ond_est_val', 0)/100.0:+.2f}")
            lines.append(f"  • Atenuar LP: {params.get('lp_val', 0)}%")

        self.txt_params_summary.setPlainText("\n".join(lines))

        self.btn_export.setEnabled(True)
        self.btn_delete.setEnabled(not preset.is_builtin)
        self.btn_apply.setEnabled(True)

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
        self.apply_preset_requested.emit(self.selected_preset, scope)
        self.accept()

    def _on_table_double_clicked(self):
        self._on_apply_clicked()

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
