# gui/main_window.py
"""
gui/main_window.py - Ventana principal contenedora de la aplicación AstroLoom.
Coordina las pestañas principales (Apilador, Revelador, Eclipses, Startrails, Panorámicas y Ajustes).
"""
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QTabWidget, QStatusBar, QLabel, QWidget, QHBoxLayout, QComboBox

from core.i18n import tr, get_i18n
from gui.theme import apply_theme, STYLE_STATUS_PILL
from gui.tab_stacker import StackerTab
from gui.tab_developer import DeveloperTab
from gui.tab_eclipse import EclipseTab
from gui.tab_eclipse_collage import EclipseCollageTab
from gui.tab_startrails import StartrailsTab
from gui.tab_panorama import PanoramaTab
from gui.tab_settings import SettingsTab


class MainWindow(QMainWindow):
    """Ventana principal de la interfaz gráfica de AstroLoom."""

    BASE_TITLE = "AstroLoom - Suite de Astrofotografía y Paisaje Nocturno (v0.8.1)"


    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.BASE_TITLE)
        self.resize(1360, 880)

        # Aplicar el sistema de diseño global
        apply_theme(self)

        self._setup_ui()
        self._setup_status_bar()

        # Conectar cambio reactivo de idioma global
        get_i18n().language_changed.connect(lambda _: self._retranslate_ui())
        self._retranslate_ui()

    def _setup_status_bar(self) -> None:
        """Configura la barra de estado moderna con monitor de memoria RAM y motor de procesado."""
        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)

        # Mensaje de estado contextual
        self.lbl_status = QLabel("Listo para procesar")
        status_bar.addWidget(self.lbl_status, 1)

        # Píldora del Motor AstroLoom
        self.lbl_engine_badge = QLabel("⚡ AstroLoom Engine v0.8.1")
        self.lbl_engine_badge.setStyleSheet(STYLE_STATUS_PILL)
        status_bar.addPermanentWidget(self.lbl_engine_badge)

        # Píldora de Monitor de Memoria RAM en tiempo real
        self.lbl_ram_monitor = QLabel("🧠 RAM: --")
        self.lbl_ram_monitor.setStyleSheet(
            "background-color: #141724; border: 1px solid #232738; border-radius: 4px; padding: 2px 8px; color: #ffd54f; font-weight: bold; font-size: 10px;"
        )
        status_bar.addPermanentWidget(self.lbl_ram_monitor)

        # Selector de Idioma en tiempo real
        self.combo_lang_bar = QComboBox()
        self.combo_lang_bar.setStyleSheet("""
            QComboBox {
                background-color: #141724;
                border: 1px solid #232738;
                border-radius: 4px;
                padding: 1px 6px;
                color: #80d8ff;
                font-weight: bold;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
                width: 14px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1e2e;
                color: #e0e0e0;
                selection-background-color: #28334a;
            }
        """)
        avail_langs = get_i18n().get_available_languages()
        for code, name in avail_langs:
            self.combo_lang_bar.addItem(f"🌐 {code.upper()} ({name})", code)

        curr_lang = get_i18n().get_current_language()
        idx_lang = self.combo_lang_bar.findData(curr_lang)
        if idx_lang >= 0:
            self.combo_lang_bar.setCurrentIndex(idx_lang)

        self.combo_lang_bar.currentIndexChanged.connect(self._on_lang_bar_changed)
        status_bar.addPermanentWidget(self.combo_lang_bar)

        # Temporizador para actualizar métricas de hardware cada 2.5 segundos
        self._timer_ram = QTimer(self)
        self._timer_ram.timeout.connect(self._update_system_stats)
        self._timer_ram.start(2500)
        self._update_system_stats()

    def _on_lang_bar_changed(self, index: int) -> None:
        """Manejador al seleccionar un nuevo idioma desde la barra de estado."""
        code = self.combo_lang_bar.currentData()
        if code and code != get_i18n().get_current_language():
            get_i18n().set_language(code)

    def _retranslate_ui(self) -> None:
        """Actualiza todos los textos estáticos principales al cambiar de idioma."""
        self.setWindowTitle(tr("app.title", self.BASE_TITLE))
        self.lbl_status.setText(tr("app.ready", "Listo para procesar"))
        self.lbl_engine_badge.setText(tr("app.engine_badge", "⚡ AstroLoom Engine v0.8.1"))

        if hasattr(self, 'tab_widget'):
            self.tab_widget.setTabText(0, tr("app.tabs.stacker", "1. Apilador (Stacker)"))
            self.tab_widget.setTabText(1, tr("app.tabs.developer", "2. Revelador / Editor"))
            self.tab_widget.setTabText(2, tr("app.tabs.eclipse_hdr", "3. Eclipses (HDR)"))
            self.tab_widget.setTabText(3, tr("app.tabs.eclipse_collage", "4. Eclipses (Collage)"))
            self.tab_widget.setTabText(4, tr("app.tabs.startrails", "5. Trazas de Estrellas (Startrails)"))
            self.tab_widget.setTabText(5, tr("app.tabs.panorama", "6. Panorámicas (Mosaicos)"))
            self.tab_widget.setTabText(6, tr("app.tabs.settings", "7. Configuración / Ajustes"))

        if hasattr(self, 'combo_lang_bar'):
            idx = self.combo_lang_bar.findData(get_i18n().get_current_language())
            if idx >= 0 and self.combo_lang_bar.currentIndex() != idx:
                self.combo_lang_bar.blockSignals(True)
                self.combo_lang_bar.setCurrentIndex(idx)
                self.combo_lang_bar.blockSignals(False)

        if hasattr(self, 'tab_developer') and hasattr(self.tab_developer, "retranslate_ui"):
            self.tab_developer.retranslate_ui()

    def _update_system_stats(self) -> None:
        """Actualiza el consumo de memoria RAM física del sistema."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            pct = vm.percent
            used_gb = vm.used / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
            self.lbl_ram_monitor.setText(f"🧠 RAM: {pct:.0f}% ({used_gb:.1f} / {total_gb:.1f} GB)")
            if pct > 85:
                self.lbl_ram_monitor.setStyleSheet(
                    "background-color: #381a1a; border: 1px solid #c62828; border-radius: 4px; padding: 2px 8px; color: #ff8a80; font-weight: bold; font-size: 10px;"
                )
            else:
                self.lbl_ram_monitor.setStyleSheet(
                    "background-color: #141724; border: 1px solid #232738; border-radius: 4px; padding: 2px 8px; color: #ffd54f; font-weight: bold; font-size: 10px;"
                )
        except Exception:
            self.lbl_ram_monitor.setText("🧠 RAM: N/D")

    def set_status(self, text: str) -> None:
        """Actualiza el mensaje principal de la barra de estado."""
        self.lbl_status.setText(text)


    def _setup_ui(self) -> None:
        """Inicializa las pestañas de navegación y enlaza las señales entre módulos."""
        self.tab_widget = QTabWidget()

        # Instanciar pestañas del flujo de trabajo
        self.tab_stacker = StackerTab()
        self.tab_developer = DeveloperTab()
        self.tab_eclipse = EclipseTab()
        self.tab_eclipse_collage = EclipseCollageTab()
        self.tab_startrails = StartrailsTab()
        self.tab_panorama = PanoramaTab()
        self.tab_settings = SettingsTab()

        self.tab_widget.addTab(self.tab_stacker, "1. Apilador (Stacker)")
        self.tab_widget.addTab(self.tab_developer, "2. Revelador / Editor")
        self.tab_widget.addTab(self.tab_eclipse, "3. Eclipses (HDR)")
        self.tab_widget.addTab(self.tab_eclipse_collage, "4. Eclipses (Collage)")
        self.tab_widget.addTab(self.tab_startrails, "5. Trazas de Estrellas (Startrails)")
        self.tab_widget.addTab(self.tab_panorama, "6. Panorámicas (Mosaicos)")
        self.tab_widget.addTab(self.tab_settings, "7. Configuración / Ajustes")

        self.setCentralWidget(self.tab_widget)

        # 1. Transferencia automática tras finalizar apilado
        self.tab_stacker.stacking_finished.connect(self._on_stacking_finished)

        # 2. Reset del entorno completo al solicitar nueva sesión
        self.tab_stacker.new_session_requested.connect(self.tab_developer.clear_session)

        # 3. Actualización dinámica del título según la sesión activa
        self.tab_stacker.session_title_changed.connect(self._update_window_title)

        # 4. Transferencia de imagen desde el módulo de Eclipses (HDR) al Revelador
        self.tab_eclipse.export_to_developer.connect(self._on_eclipse_exported)

        # 5. Transferencia de serie desde Eclipses (HDR) al Diseñador de Collage
        self.tab_eclipse.open_collage_requested.connect(self._on_open_collage_requested)

        # 6. Transferencia de imagen desde el Collage de Eclipses al Revelador
        self.tab_eclipse_collage.export_to_developer.connect(self._on_eclipse_exported)

        # 7. Transferencia de imagen desde el módulo de Startrails al Revelador
        self.tab_startrails.export_to_developer.connect(self._on_startrail_exported)

        # 8. Proveedor de máscara de horizonte del Apilador para Startrails
        self.tab_startrails.set_stacker_mask_provider(
            lambda: getattr(self.tab_stacker, "computed_mask", None)
        )

        # 9. Transferencia de imagen desde el módulo de Panorámicas al Revelador
        self.tab_panorama.export_to_developer.connect(self._on_panorama_exported)

    def _update_window_title(self, session_name: str) -> None:
        """Actualiza el texto de la barra de título con el archivo de proyecto activo."""
        if session_name:
            self.setWindowTitle(f"{self.BASE_TITLE} — [{session_name}]")
        else:
            self.setWindowTitle(self.BASE_TITLE)

    def _on_stacking_finished(self, output_path: str, mask: Optional[object] = None) -> None:
        """
        Transfiere el archivo 32-bit generado y su máscara calculada al Revelador,
        conmutando automáticamente a la pestaña 2.
        """
        final_mask = mask if mask is not None else getattr(self.tab_stacker, "computed_mask", None)
        self.tab_developer.load_image_direct(output_path, mask=final_mask)
        self.tab_widget.setCurrentIndex(1)

    def _on_eclipse_exported(self, output_path: str) -> None:
        """
        Transfiere la imagen generada en el módulo de Eclipses (HDR o Collage) directamente
        al Revelador / Editor y conmuta automáticamente a la pestaña 2.
        """
        self.tab_developer.load_image_direct(output_path, mask=None)
        self.tab_widget.setCurrentIndex(1)

    def _on_open_collage_requested(self, filepaths: list, mode: str) -> None:
        """
        Carga las tomas en la pestaña 4 (Collage de Eclipses) y conmuta
        automáticamente a ella.
        """
        if mode:
            self.tab_eclipse_collage.set_mode(mode)
        if filepaths:
            self.tab_eclipse_collage.load_filepaths(filepaths)
        self.tab_widget.setCurrentIndex(3)

    def _on_startrail_exported(self, output_path: str) -> None:
        """
        Transfiere la imagen 32-bit generada en el módulo de Startrails directamente
        al Revelador / Editor y conmuta automáticamente a la pestaña 2.
        """
        final_mask = getattr(self.tab_startrails, "current_mask", None)
        self.tab_developer.load_image_direct(output_path, mask=final_mask)
        self.tab_widget.setCurrentIndex(1)

    def _on_panorama_exported(self, output_path: str) -> None:
        """
        Transfiere la imagen generada en el módulo de Panorámicas directamente
        al Revelador / Editor y conmuta automáticamente a la pestaña 2.
        """
        self.tab_developer.load_image_direct(output_path, mask=None)
        self.tab_widget.setCurrentIndex(1)