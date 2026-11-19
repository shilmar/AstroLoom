# gui/main_window.py
"""
gui/main_window.py - Ventana principal contenedora de la aplicación AstroLoom.
Coordina las pestañas principales (Apilador, Revelador, Eclipses, Startrails, Panorámicas y Ajustes).
"""
from typing import Optional
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QMainWindow, QTabWidget, QLabel, QWidget, QHBoxLayout, QPushButton

from core.i18n import tr, get_i18n
from gui.theme import apply_theme, STYLE_STATUS_PILL
from gui.icons import get_icon
from gui.tab_stacker import StackerTab
from gui.tab_developer import DeveloperTab
from gui.tab_eclipse import EclipseTab
from gui.tab_eclipse_collage import EclipseCollageTab
from gui.tab_startrails import StartrailsTab
from gui.tab_meteors import MeteorTab
from gui.tab_panorama import PanoramaTab
from gui.tab_settings import SettingsTab


class MainWindow(QMainWindow):
    """Ventana principal de la interfaz gráfica de AstroLoom."""

    BASE_TITLE = "AstroLoom - Suite de Astrofotografía y Paisaje Nocturno (v0.17.0)"
    gpu_fallback_signal = Signal(str)

    def __init__(self, splash: Optional[object] = None):
        super().__init__()
        self.setWindowTitle(self.BASE_TITLE)
        self.resize(1360, 880)

        # Aplicar el sistema de diseño global
        apply_theme(self)

        self._setup_ui(splash=splash)
        self._setup_tab_bar_corner()
        self.setStatusBar(None)

        # Conectar señal de degradación de hardware GPU a CPU
        self.gpu_fallback_signal.connect(self._on_gpu_fallback)
        try:
            from core.starnet_onnx import add_gpu_fallback_listener as add_sn_listener
            add_sn_listener(self.gpu_fallback_signal.emit)
        except Exception:
            pass
        try:
            from core.ai_denoise import add_gpu_fallback_listener as add_dn_listener
            add_dn_listener(self.gpu_fallback_signal.emit)
        except Exception:
            pass

        # Conectar cambio reactivo de idioma global
        get_i18n().language_changed.connect(lambda _: self._retranslate_ui())
        self._retranslate_ui()

    def _on_gpu_fallback(self, reason: str) -> None:
        """Muestra la píldora naranja de aviso en la cabecera superior al degradar de GPU a CPU."""
        if hasattr(self, 'lbl_gpu_fallback'):
            self.lbl_gpu_fallback.setVisible(True)
            self.lbl_gpu_fallback.setToolTip(f"GPU/VRAM saturada: se continuó en CPU sin interrumpir la tarea.\n{reason}\n(Clic para ocultar)")

    def _setup_tab_bar_corner(self) -> None:
        """Configura los indicadores de estado (versión de motor, RAM y alertas de hardware) en la esquina superior de las pestañas."""
        corner_widget = QWidget()
        corner_widget.setStyleSheet("background: transparent;")
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 10, 0)
        corner_layout.setSpacing(6)
        corner_layout.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        # 1. Píldora del Motor AstroLoom (versión)
        self.lbl_engine_badge = QLabel("● AstroLoom Engine v0.17.0")
        self.lbl_engine_badge.setStyleSheet(STYLE_STATUS_PILL)
        corner_layout.addWidget(self.lbl_engine_badge)

        # 2. Píldora de aviso de Fallback GPU -> CPU (naranja/amarillo, junto al monitor de RAM)
        self.lbl_gpu_fallback = QLabel("⚠️ Fallback CPU (VRAM)")
        self.lbl_gpu_fallback.setStyleSheet(
            "background-color: #3e2723; border: 1px solid #ff9800; border-radius: 4px; padding: 2px 8px; color: #ffb74d; font-weight: bold; font-size: 10px;"
        )
        self.lbl_gpu_fallback.setToolTip("La GPU se quedó sin memoria (VRAM). Se continuó en CPU de forma segura.")
        self.lbl_gpu_fallback.setVisible(False)
        self.lbl_gpu_fallback.setCursor(Qt.PointingHandCursor)
        self.lbl_gpu_fallback.mousePressEvent = lambda _: self.lbl_gpu_fallback.setVisible(False)
        corner_layout.addWidget(self.lbl_gpu_fallback)

        # 3. Píldora de Monitor de Memoria RAM en tiempo real
        self.lbl_ram_monitor = QLabel("RAM: --")
        self.lbl_ram_monitor.setStyleSheet(
            "background-color: #141724; border: 1px solid #232738; border-radius: 4px; padding: 2px 8px; color: #ffd54f; font-weight: bold; font-size: 10px;"
        )
        corner_layout.addWidget(self.lbl_ram_monitor)

        # 4. Botón de Registro de Actividad y Telemetría
        self.btn_activity_log = QPushButton(tr("app.log_btn", " Registro"))
        self.btn_activity_log.setIcon(get_icon("activity", size=13, color="#00e5ff"))
        self.btn_activity_log.setStyleSheet(
            "QPushButton {"
            "  background-color: #141724; border: 1px solid #232738; border-radius: 4px;"
            "  padding: 2px 8px; color: #80d8ff; font-weight: bold; font-size: 10px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #1c2032; border-color: #00e5ff; color: #ffffff;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #0d101a;"
            "}"
        )
        self.btn_activity_log.setCursor(Qt.PointingHandCursor)
        self.btn_activity_log.setToolTip(tr("app.log_btn_tooltip", "Abrir Registro de Actividad y Telemetría"))
        self.btn_activity_log.clicked.connect(self._open_activity_log)
        corner_layout.addWidget(self.btn_activity_log)

        self.tab_widget.setCornerWidget(corner_widget, Qt.TopRightCorner)

        # Temporizador para actualizar métricas de hardware cada 2.5 segundos
        self._timer_ram = QTimer(self)
        self._timer_ram.timeout.connect(self._update_system_stats)
        self._timer_ram.start(2500)
        self._update_system_stats()

    def _open_activity_log(self) -> None:
        """Abre o enfoca la ventana flotante del Registro de Actividad."""
        if not hasattr(self, '_dialog_activity_log') or self._dialog_activity_log is None:
            from gui.dialog_activity_log import ActivityLogDialog
            self._dialog_activity_log = ActivityLogDialog(self)
        self._dialog_activity_log.show()
        self._dialog_activity_log.raise_()
        self._dialog_activity_log.activateWindow()

    def _retranslate_ui(self) -> None:
        """Actualiza todos los textos estáticos principales al cambiar de idioma."""
        self.setWindowTitle(tr("app.title", self.BASE_TITLE))
        if hasattr(self, 'lbl_engine_badge'):
            self.lbl_engine_badge.setText(tr("app.engine_badge", "● AstroLoom Engine v0.17.0"))
        if hasattr(self, 'btn_activity_log'):
            self.btn_activity_log.setText(tr("app.log_btn", " Registro"))
            self.btn_activity_log.setToolTip(tr("app.log_btn_tooltip", "Abrir Registro de Actividad y Telemetría"))

        if hasattr(self, 'tab_widget'):
            self.tab_widget.setTabText(0, tr("app.tabs.stacker", "1. Apilador (Stacker)"))
            self.tab_widget.setTabText(1, tr("app.tabs.developer", "2. Revelador / Editor"))
            self.tab_widget.setTabText(2, tr("app.tabs.eclipse_hdr", "3. Eclipses (HDR)"))
            self.tab_widget.setTabText(3, tr("app.tabs.eclipse_collage", "4. Eclipses (Collage)"))
            self.tab_widget.setTabText(4, tr("app.tabs.startrails", "5. Trazas de Estrellas (Startrails)"))
            self.tab_widget.setTabText(5, tr("app.tabs.meteors", "6. Lluvias de Meteoros"))
            self.tab_widget.setTabText(6, tr("app.tabs.panorama", "7. Panorámicas (Mosaicos)"))
            self.tab_widget.setTabText(7, tr("app.tabs.settings", "8. Configuración / Ajustes"))

        if hasattr(self, 'tab_developer') and hasattr(self.tab_developer, "retranslate_ui"):
            self.tab_developer.retranslate_ui()

    def _update_system_stats(self) -> None:
        """Actualiza el consumo de memoria RAM física del sistema."""
        try:
            from core.system_info import get_system_memory
            vm = get_system_memory()
            pct = vm.percent
            used_gb = vm.used / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
            self.lbl_ram_monitor.setText(f"RAM: {pct:.0f}% ({used_gb:.1f} / {total_gb:.1f} GB)")
            if pct > 85:
                self.lbl_ram_monitor.setStyleSheet(
                    "background-color: #381a1a; border: 1px solid #c62828; border-radius: 4px; padding: 2px 8px; color: #ff8a80; font-weight: bold; font-size: 10px;"
                )
            else:
                self.lbl_ram_monitor.setStyleSheet(
                    "background-color: #141724; border: 1px solid #232738; border-radius: 4px; padding: 2px 8px; color: #ffd54f; font-weight: bold; font-size: 10px;"
                )
        except Exception:
            self.lbl_ram_monitor.setText("RAM: N/D")

    def set_status(self, text: str) -> None:
        """Actualiza el mensaje principal de estado (compatibilidad)."""
        pass


    def _setup_ui(self, splash: Optional[object] = None) -> None:
        """Inicializa las pestañas de navegación y enlaza las señales entre módulos."""
        self.tab_widget = QTabWidget()

        # Instanciar pestañas del flujo de trabajo
        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.stacker", "Cargando Apilador y motor CFA..."), 20)
        self.tab_stacker = StackerTab()

        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.developer", "Cargando Revelador y capas de ajuste..."), 45)
        self.tab_developer = DeveloperTab()

        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.eclipse", "Cargando módulos de Eclipses (HDR y Collage)..."), 65)
        self.tab_eclipse = EclipseTab()
        self.tab_eclipse_collage = EclipseCollageTab()

        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.star_meteor", "Cargando Trazas de Estrellas y Meteoros..."), 80)
        self.tab_startrails = StartrailsTab()
        self.tab_meteors = MeteorTab()

        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.pano_settings", "Cargando Panorámicas y Ajustes..."), 95)
        self.tab_panorama = PanoramaTab()
        self.tab_settings = SettingsTab()

        self.tab_widget.addTab(self.tab_stacker, "1. Apilador (Stacker)")
        self.tab_widget.addTab(self.tab_developer, "2. Revelador / Editor")
        self.tab_widget.addTab(self.tab_eclipse, "3. Eclipses (HDR)")
        self.tab_widget.addTab(self.tab_eclipse_collage, "4. Eclipses (Collage)")
        self.tab_widget.addTab(self.tab_startrails, "5. Trazas de Estrellas (Startrails)")
        self.tab_widget.addTab(self.tab_meteors, "6. Lluvias de Meteoros")
        self.tab_widget.addTab(self.tab_panorama, "7. Panorámicas (Mosaicos)")
        self.tab_widget.addTab(self.tab_settings, "8. Configuración / Ajustes")

        self.setCentralWidget(self.tab_widget)

        # 1. Transferencia automática tras finalizar apilado
        self.tab_stacker.stacking_finished.connect(self._on_stacking_finished)

        # 2. Reset del entorno completo al solicitar nueva sesión
        self.tab_stacker.new_session_requested.connect(self.tab_developer.clear_session)

        # 3. Actualización dinámica del título según la sesión activa
        self.tab_stacker.session_title_changed.connect(self._update_window_title)

        # 4. Transferencia de imagen desde el módulo de Eclipses (HDR) al Revelador
        self.tab_eclipse.export_to_developer.connect(self._on_eclipse_exported)
        self.tab_eclipse.export_coronal_stack_to_developer.connect(self._on_eclipse_coronal_stack_exported)

        # 5. Transferencia de imagen desde el Collage de Eclipses al Revelador
        self.tab_eclipse_collage.export_to_developer.connect(self._on_eclipse_exported)

        # 6. Transferencia de imagen desde el módulo de Startrails al Revelador
        self.tab_startrails.export_to_developer.connect(self._on_startrail_exported)

        # 7. Proveedor de máscara de horizonte del Apilador para Startrails
        self.tab_startrails.set_stacker_mask_provider(
            lambda: getattr(self.tab_stacker, "computed_mask", None)
        )

        # 8. Transferencia de imagen desde el módulo de Panorámicas al Revelador
        self.tab_panorama.export_to_developer.connect(self._on_panorama_exported)

        # 9. Transferencia de imagen desde el módulo de Meteoros al Revelador
        self.tab_meteors.export_to_developer.connect(self._on_meteor_exported)
        self.tab_meteors.set_stacker_mask_provider(
            lambda: getattr(self.tab_stacker, "computed_mask", None)
        )

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

    def _on_eclipse_coronal_stack_exported(self, base_path: str, raster_layers: list) -> None:
        """
        Transfiere la composición multicapa coronal desde el módulo de Eclipses directamente
        al Revelador con sus capas ráster y modos de fusión preconfigurados.
        """
        self.tab_developer.load_coronal_stack(base_path, raster_layers)
        self.tab_widget.setCurrentIndex(1)

    def _on_startrail_exported(self, output_path: str) -> None:
        """
        Transfiere la imagen 32-bit generada en el módulo de Startrails directamente
        al Revelador / Editor y conmuta automáticamente a la pestaña 2.
        """
        final_mask = getattr(self.tab_startrails, "current_mask", None)
        self.tab_developer.load_image_direct(output_path, mask=final_mask)
        self.tab_widget.setCurrentIndex(1)

    def _on_meteor_exported(self, output_path: str) -> None:
        """
        Transfiere el compuesto de lluvia de meteoros directamente
        al Revelador / Editor y conmuta automáticamente a la pestaña 2.
        """
        final_mask = getattr(self.tab_meteors, "current_mask", None)
        self.tab_developer.load_image_direct(output_path, mask=final_mask)
        self.tab_widget.setCurrentIndex(1)

    def _on_panorama_exported(self, output_path: str) -> None:
        """
        Transfiere la imagen generada en el módulo de Panorámicas directamente
        al Revelador / Editor y conmuta automáticamente a la pestaña 2.
        """
        self.tab_developer.load_image_direct(output_path, mask=None)
        self.tab_widget.setCurrentIndex(1)
