# gui/dialog_presets.py
"""
gui/dialog_presets.py - Diálogos interactivos para guardar, importar, exportar y gestionar
los ajustes preestablecidos (Presets) de revelado en AstroLoom.
"""
from typing import Optional, Dict, Any, List
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
    QSplitter, QWidget, QFrame
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


class SavePresetDialog(QDialog):
    """Diálogo modal para guardar los ajustes actuales del revelador como un nuevo preset."""

    def __init__(self, current_params: Dict[str, Any], existing_categories: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_params = current_params
        self.existing_categories = existing_categories
        self.created_preset: Optional[DeveloperPreset] = None

        self.setWindowTitle("Guardar Ajustes Actuales como Preset")
        self.resize(480, 420)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Encabezado
        lbl_header = QLabel("🎨 Guardar Nuevo Ajuste Preestablecido")
        lbl_header.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT_CYAN};")
        layout.addWidget(lbl_header)

        lbl_sub = QLabel("Guarda todos los deslizadores, curvas y parámetros actuales para aplicarlos en otras imágenes.")
        lbl_sub.setWordWrap(True)
        lbl_sub.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(lbl_sub)

        # Formulario
        form_group = QGroupBox("Detalles del Preset")
        form_layout = QVBoxLayout(form_group)
        form_layout.setSpacing(8)

        # Nombre
        form_layout.addWidget(QLabel("Nombre del Preset:*"))
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Ej: Vía Láctea - Verano Contraste Alto")
        form_layout.addWidget(self.txt_name)

        # Categoría
        form_layout.addWidget(QLabel("Categoría:"))
        self.combo_category = QComboBox()
        self.combo_category.setEditable(True)
        categories = sorted(list(set(self.existing_categories + ["Vía Láctea", "Paisaje Nocturno", "Cielo Profundo", "Usuario", "Especial"])))
        self.combo_category.addItems(categories)
        self.combo_category.setCurrentText("Usuario")
        form_layout.addWidget(self.combo_category)

        # Autor
        form_layout.addWidget(QLabel("Autor:"))
        self.txt_author = QLineEdit()
        self.txt_author.setPlaceholderText("Tu nombre o alias")
        self.txt_author.setText("Usuario")
        form_layout.addWidget(self.txt_author)

        # Descripción
        form_layout.addWidget(QLabel("Descripción / Notas (Opcional):"))
        self.txt_desc = QTextEdit()
        self.txt_desc.setPlaceholderText("Describe el tipo de lente, cámara, cielo o propósito para el que está pensado este preset...")
        self.txt_desc.setMaximumHeight(70)
        form_layout.addWidget(self.txt_desc)

        layout.addWidget(form_group)

        # Botones inferiores
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("💾 Guardar Preset")
        self.btn_save.setStyleSheet(STYLE_RUN_IDLE)
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
        self.resize(780, 520)
        self.setModal(True)

        self._setup_ui()
        self._refresh_presets()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Encabezado
        lbl_header = QLabel("⚙️ Gestor de Presets de Revelado")
        lbl_header.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {TEXT_CYAN};")
        layout.addWidget(lbl_header)

        # Divisor principal horizontal
        splitter = QSplitter(Qt.Horizontal)

        # --- LADO IZQUIERDO: Tabla de Presets ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Filtro de búsqueda / categoría
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filtrar:"))
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Buscar por nombre...")
        self.txt_filter.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.txt_filter)

        self.combo_cat_filter = QComboBox()
        self.combo_cat_filter.addItem("Todas las categorías")
        self.combo_cat_filter.currentTextChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.combo_cat_filter)
        left_layout.addLayout(filter_layout)

        # Tabla
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Nombre", "Categoría", "Origen"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.doubleClicked.connect(self._on_table_double_clicked)
        left_layout.addWidget(self.table)

        # Botones bajo la tabla
        tbl_btn_layout = QHBoxLayout()
        self.btn_import = QPushButton("📂 Importar...")
        self.btn_import.setToolTip("Carga un archivo .json de preset desde tu disco")
        self.btn_import.clicked.connect(self._on_import_clicked)
        tbl_btn_layout.addWidget(self.btn_import)

        self.btn_export = QPushButton("💾 Exportar...")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("Exporta el preset seleccionado a un archivo .json para compartirlo")
        self.btn_export.clicked.connect(self._on_export_clicked)
        tbl_btn_layout.addWidget(self.btn_export)

        self.btn_delete = QPushButton("🗑️ Eliminar")
        self.btn_delete.setEnabled(False)
        self.btn_delete.setStyleSheet(f"color: #ff8a80;")
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

        grp_detail = QGroupBox("Detalles del Preset")
        detail_layout = QVBoxLayout(grp_detail)
        detail_layout.setSpacing(8)

        self.lbl_detail_name = QLabel("Selecciona un preset")
        self.lbl_detail_name.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {TEXT_PRIMARY};")
        self.lbl_detail_name.setWordWrap(True)
        detail_layout.addWidget(self.lbl_detail_name)

        self.lbl_detail_badge = QLabel("")
        self.lbl_detail_badge.setStyleSheet(STYLE_STATUS_PILL)
        detail_layout.addWidget(self.lbl_detail_badge)

        self.lbl_detail_desc = QLabel("Elige un preset de la lista para ver su descripción y parámetros.")
        self.lbl_detail_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self.lbl_detail_desc.setWordWrap(True)
        detail_layout.addWidget(self.lbl_detail_desc)

        # Resumen de Parámetros Clave
        detail_layout.addWidget(QLabel("Parámetros Configurados:"))
        self.txt_params_summary = QTextEdit()
        self.txt_params_summary.setReadOnly(True)
        self.txt_params_summary.setStyleSheet("font-family: monospace; font-size: 11px;")
        detail_layout.addWidget(self.txt_params_summary)

        right_layout.addWidget(grp_detail)

        # Grupo de Ámbito de Aplicación
        grp_scope = QGroupBox("Modo de Aplicación:")
        scope_layout = QVBoxLayout(grp_scope)
        scope_layout.setSpacing(4)

        self.scope_group = QButtonGroup(self)
        self.rad_all = QRadioButton("Completo (Todos los parámetros)")
        self.rad_all.setChecked(True)
        self.rad_tone = QRadioButton("Solo Tono y Curvas (MTF, BP, Contraste, Suelo)")
        self.rad_color = QRadioButton("Solo Color y WB (Temp, Tinte, Saturación, Vibración)")
        self.rad_details = QRadioButton("Solo Estrellas y Detalles (Claridad, Dehaze, StarNet)")

        self.scope_group.addButton(self.rad_all, 0)
        self.scope_group.addButton(self.rad_tone, 1)
        self.scope_group.addButton(self.rad_color, 2)
        self.scope_group.addButton(self.rad_details, 3)

        scope_layout.addWidget(self.rad_all)
        scope_layout.addWidget(self.rad_tone)
        scope_layout.addWidget(self.rad_color)
        scope_layout.addWidget(self.rad_details)

        right_layout.addWidget(grp_scope)

        # Botón Aplicar Preset
        self.btn_apply = QPushButton("⚡ Aplicar Preset a la Imagen")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setFixedHeight(36)
        self.btn_apply.setStyleSheet(STYLE_RUN_IDLE)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        right_layout.addWidget(self.btn_apply)

        splitter.addWidget(right_widget)
        splitter.setSizes([450, 310])
        layout.addWidget(splitter)

        # Barra inferior de cierre
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        btn_close = QPushButton("Cerrar")
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
            item_name = QTableWidgetItem(p.name)
            item_name.setData(Qt.UserRole, p)
            item_cat = QTableWidgetItem(p.category)
            origin_str = "⭐ Fábrica" if p.is_builtin else "👤 Usuario"
            item_origin = QTableWidgetItem(origin_str)

            if p.is_builtin:
                item_origin.setForeground(Qt.cyan)
            else:
                item_origin.setForeground(Qt.yellow)

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
        origin_text = "Preset Oficial de Fábrica" if preset.is_builtin else f"Preset de Usuario ({preset.author})"
        self.lbl_detail_badge.setText(f"🏷️ {preset.category}  |  {origin_text}")
        self.lbl_detail_desc.setText(preset.description or "Sin descripción proporcionada.")

        # Construir resumen de parámetros
        lines = []
        params = preset.params
        if "mtf_val" in params or "bp_val" in params:
            lines.append("--- TONALIDAD & ESTIRADO ---")
            lines.append(f"  • Estirado MTF: {params.get('mtf_val', 0.10):.4f}")
            lines.append(f"  • Punto Negro: {params.get('bp_val', 0.0):.4f}")
            lines.append(f"  • Contraste: {params.get('contrast_val', 0)/100.0:+.2f}")
            if params.get("gnd_ev_val", 0) != 0:
                lines.append(f"  • Suelo EV: {params.get('gnd_ev_val', 0)/100.0:+.2f} EV")
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
