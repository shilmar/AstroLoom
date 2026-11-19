# gui/dialog_activity_log.py
"""
gui/dialog_activity_log.py - Ventana flotante de Registro de Actividad y Telemetría.

Muestra los registros unificados de todos los módulos de AstroLoom (Apilador, Revelador,
Startrails, Eclipses, Meteoros, Panorámicas y submódulos ONNX/Rust) con filtros en vivo,
búsqueda reactiva, persistencia de posición/tamaño en QSettings y exportación.
"""
import html
from typing import Optional, List, Tuple
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QPlainTextEdit, QFileDialog,
    QApplication, QFrame
)
from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtGui import QTextCursor

from core.i18n import tr
from core.log_hub import log_hub
from gui.icons import get_icon
from gui.theme import apply_theme, BORDER_BASE, BG_CARD


class ActivityLogDialog(QDialog):
    """
    Ventana flotante no modal del Registro de Actividad y Telemetría de AstroLoom.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(tr("dialog.activity_log.title", "AstroLoom - Registro de Actividad y Telemetría"))
        self.setModal(False)
        apply_theme(self)

        self._entries: List[Tuple[str, str, str]] = []
        self._init_ui()
        self._restore_geometry()

        # Cargar historial existente y conectar señal reactiva
        self._entries = log_hub.get_history()
        self._refresh_view()
        log_hub.message_logged.connect(self._on_message_logged)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # -------------------------------------------------------------
        # BARRA SUPERIOR: Filtros y Acciones
        # -------------------------------------------------------------
        top_bar = QWidget()
        top_bar.setStyleSheet(f"background-color: {BG_CARD}; border: 1px solid {BORDER_BASE}; border-radius: 6px;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 6, 8, 6)
        top_layout.setSpacing(8)

        # Buscador reactivo por texto
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText(tr("dialog.activity_log.search_placeholder", "Buscar (texto, archivo, ONNX, error, tiempo)..."))
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.textChanged.connect(self._apply_filters)
        top_layout.addWidget(self.txt_search, stretch=2)

        # Filtro por módulo
        self.combo_module = QComboBox()
        self.combo_module.addItems([
            tr("dialog.activity_log.mod_all", "Todos los módulos"),
            "Apilador",
            "Revelador",
            "Eclipses",
            "Startrails",
            "Meteoros",
            "Panorámicas",
            "Sistema"
        ])
        self.combo_module.currentIndexChanged.connect(self._apply_filters)
        top_layout.addWidget(self.combo_module, stretch=1)

        # Filtro por nivel / criticidad
        self.combo_level = QComboBox()
        self.combo_level.addItems([
            tr("dialog.activity_log.lvl_all", "Todos los niveles"),
            tr("dialog.activity_log.lvl_errors", "Solo Errores / Avisos"),
            tr("dialog.activity_log.lvl_perf", "Solo Tiempos / Métricas"),
        ])
        self.combo_level.currentIndexChanged.connect(self._apply_filters)
        top_layout.addWidget(self.combo_level, stretch=1)

        # Divisor vertical sutil
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {BORDER_BASE};")
        top_layout.addWidget(sep)

        # Botón Copiar al Portapapeles
        self.btn_copy = QPushButton(tr("dialog.activity_log.copy", "Copiar"))
        self.btn_copy.setIcon(get_icon("copy", size=14))
        self.btn_copy.setToolTip(tr("dialog.activity_log.copy_tip", "Copia el contenido filtrado actual al portapapeles"))
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        top_layout.addWidget(self.btn_copy)

        # Botón Exportar a Archivo
        self.btn_export = QPushButton(tr("dialog.activity_log.export", "Exportar..."))
        self.btn_export.setIcon(get_icon("download", size=14))
        self.btn_export.setToolTip(tr("dialog.activity_log.export_tip", "Guarda el registro filtrado en un archivo .log o .txt"))
        self.btn_export.clicked.connect(self._export_to_file)
        top_layout.addWidget(self.btn_export)

        # Botón Limpiar
        self.btn_clear = QPushButton(tr("dialog.activity_log.clear", "Limpiar"))
        self.btn_clear.setIcon(get_icon("trash", size=14))
        self.btn_clear.setToolTip(tr("dialog.activity_log.clear_tip", "Vacía la memoria del registro"))
        self.btn_clear.clicked.connect(self._clear_logs)
        top_layout.addWidget(self.btn_clear)

        layout.addWidget(top_bar)

        # -------------------------------------------------------------
        # ÁREA CENTRAL: Visor Monospaciado de Alto Rendimiento
        # -------------------------------------------------------------
        self.txt_view = QPlainTextEdit()
        self.txt_view.setReadOnly(True)
        self.txt_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.txt_view, stretch=1)

        # -------------------------------------------------------------
        # BARRA INFERIOR DE ESTADO / CONFIGURACIÓN
        # -------------------------------------------------------------
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 0, 4, 0)

        self.lbl_stats = QLabel("0 eventos")
        self.lbl_stats.setStyleSheet("color: #78909c; font-size: 11px;")
        bottom_bar.addWidget(self.lbl_stats)

        bottom_bar.addStretch()

        self.chk_autoscroll = QCheckBox(tr("dialog.activity_log.autoscroll", "Autodesplazamiento"))
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setStyleSheet("color: #90a4ae; font-size: 11px;")
        bottom_bar.addWidget(self.chk_autoscroll)

        layout.addLayout(bottom_bar)

    def _matches_filters(self, ts: str, module: str, message: str) -> bool:
        """Comprueba si una entrada cumple con los filtros activos."""
        search_term = self.txt_search.text().strip().lower()
        if search_term:
            full_line = f"{ts} {module} {message}".lower()
            if search_term not in full_line:
                return False

        # Filtro de módulo
        mod_idx = self.combo_module.currentIndex()
        if mod_idx > 0:
            target_mod = self.combo_module.currentText().lower()
            if target_mod not in module.lower():
                return False

        # Filtro de nivel
        lvl_idx = self.combo_level.currentIndex()
        if lvl_idx == 1:
            # Solo Errores y Avisos
            msg_upper = message.upper()
            if not any(k in msg_upper for k in ("ERROR", "AVISO", "FALLO", "FAIL", "WARN", "CRÍTICO", "CRITICO")):
                return False
        elif lvl_idx == 2:
            # Solo Rendimiento / Tiempos / Métricas
            msg_lower = message.lower()
            if not any(k in msg_lower for k in (" s", " ms", "fps", "tiempo", "rendimiento", "ram", "cpu", "gpu", "vram")):
                return False

        return True

    def _format_entry_html(self, ts: str, module: str, message: str) -> str:
        """Formatea una entrada con HTML enriquecido y paleta ergonómica calibrada."""
        msg_upper = message.upper()
        if any(k in msg_upper for k in ("ERROR", "FALLO", "FAIL", "CRÍTICO", "CRITICO")):
            body_color = "#F87171"  # Coral suave
        elif any(k in msg_upper for k in ("AVISO", "WARN", "ADVERTENCIA", "FALLBACK")):
            body_color = "#FBBF24"  # Ámbar suave
        elif any(k in message for k in ("✓", "exitosamente", "éxito", "exito", "FINALIZAD", "COMPLETADO", "activa")):
            body_color = "#34D399"  # Verde esmeralda desaturado
        else:
            body_color = "#D4D4D8"  # Gris claro neutro base

        safe_msg = html.escape(message)
        safe_mod = html.escape(f"{module:<11}")
        return (
            f'<span style="color:#71717A;">[{ts}]</span> '
            f'<span style="color:#38BDF8; font-weight:bold;">[{safe_mod}]</span> '
            f'<span style="color:{body_color};">{safe_msg}</span>'
        )

    def _refresh_view(self) -> None:
        """Reconstruye el visor completo aplicando los filtros actuales con estilos HTML."""
        matching_html = [
            self._format_entry_html(ts, mod, msg)
            for ts, mod, msg in self._entries
            if self._matches_filters(ts, mod, msg)
        ]
        self.txt_view.clear()
        if matching_html:
            self.txt_view.appendHtml("<br>".join(matching_html))

        total = len(self._entries)
        visible = len(matching_html)
        self.lbl_stats.setText(f"{visible} eventos (de {total} en búfer)")

        if self.chk_autoscroll.isChecked():
            self.txt_view.moveCursor(QTextCursor.End)

    def _apply_filters(self) -> None:
        """Aplica los filtros sobre el conjunto actual de entradas."""
        self._refresh_view()

    def _on_message_logged(self, ts: str, module: str, message: str) -> None:
        """Ranura invocada reactivamente al registrarse un nuevo mensaje en cualquier hilo."""
        self._entries.append((ts, module, message))
        if self._matches_filters(ts, module, message):
            formatted_html = self._format_entry_html(ts, module, message)
            self.txt_view.appendHtml(formatted_html)

            if self.chk_autoscroll.isChecked():
                self.txt_view.moveCursor(QTextCursor.End)

        total = len(self._entries)
        visible = self.txt_view.blockCount() if self.txt_view.toPlainText() else 0
        self.lbl_stats.setText(f"{visible} eventos (de {total} en búfer)")

    def _copy_to_clipboard(self) -> None:
        """Copia el texto visible actual al portapapeles del sistema."""
        text = self.txt_view.toPlainText()
        if not text:
            return
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
            orig_text = self.btn_copy.text()
            self.btn_copy.setText(tr("dialog.activity_log.copied", "¡Copiado!"))
            QTimer.singleShot(1500, lambda: self.btn_copy.setText(orig_text))

    def _export_to_file(self) -> None:
        """Exporta el contenido actual del registro a un archivo de texto plano."""
        text = self.txt_view.toPlainText()
        if not text:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("dialog.activity_log.export_title", "Exportar Registro de Actividad"),
            "astroloom_activity_log.txt",
            "Archivos de Registro (*.log *.txt);;Todos los archivos (*.*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                log_hub.log("Sistema", f"[ERROR EXPORTACIÓN] No se pudo guardar el registro: {e}")

    def _clear_logs(self) -> None:
        """Limpia el búfer en memoria y la visualización."""
        log_hub.clear()
        self._entries.clear()
        self.txt_view.clear()
        self.lbl_stats.setText("0 eventos (de 0 en búfer)")

    def _restore_geometry(self) -> None:
        """Restaura la posición y tamaño previo desde QSettings."""
        settings = QSettings("AstroLoom", "AstroLoom")
        geom = settings.value("activity_log/geometry")
        if geom:
            self.restoreGeometry(geom)
        else:
            self.resize(780, 520)

    def closeEvent(self, event) -> None:
        """Guarda la posición y dimensiones antes de cerrar."""
        settings = QSettings("AstroLoom", "AstroLoom")
        settings.setValue("activity_log/geometry", self.saveGeometry())
        super().closeEvent(event)
