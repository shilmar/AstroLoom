# gui/main_window.py
"""
gui/main_window.py - Ventana principal contenedora de la aplicación AstroLoom.
Coordina las pestañas principales (Apilador, Revelador, Eclipses, Startrails, Panorámicas y Ajustes).
"""
import os
import shutil
import glob
import tempfile
from typing import Optional
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QMainWindow, QTabWidget, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton

from core import APP_VERSION
from core.i18n import tr, get_i18n
from gui.theme import apply_theme, STYLE_STATUS_PILL
from gui.icons import get_icon
from gui.tab_stacker import StackerTab


class MainWindow(QMainWindow):
    """Ventana principal de la interfaz gráfica de AstroLoom."""

    BASE_TITLE = f"AstroLoom - Suite de Astrofotografía y Paisaje Nocturno (v{APP_VERSION})"
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
        self.setAcceptDrops(True)

        # Conectar señal de degradación de hardware GPU a CPU
        self.gpu_fallback_signal.connect(self._on_gpu_fallback)
        try:
            from core.onnx_runtime import add_gpu_fallback_listener
            add_gpu_fallback_listener(self.gpu_fallback_signal.emit)
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
        self.lbl_engine_badge = QLabel(f"● AstroLoom Engine v{APP_VERSION}")
        self.lbl_engine_badge.setStyleSheet(STYLE_STATUS_PILL)
        corner_layout.addWidget(self.lbl_engine_badge)

        # 2. Píldora de aviso de Fallback GPU -> CPU
        self.lbl_gpu_fallback = QLabel("⚠️ Fallback CPU (VRAM)")
        self.lbl_gpu_fallback.setStyleSheet(
            "background-color: #2c2c32; border: 1px solid #d97706; border-radius: 4px; padding: 2px 8px; color: #fbbf24; font-weight: bold; font-size: 10px;"
        )
        self.lbl_gpu_fallback.setToolTip("La GPU se quedó sin memoria (VRAM). Se continuó en CPU de forma segura.")
        self.lbl_gpu_fallback.setVisible(False)
        self.lbl_gpu_fallback.setCursor(Qt.PointingHandCursor)
        self.lbl_gpu_fallback.mousePressEvent = lambda _: self.lbl_gpu_fallback.setVisible(False)
        corner_layout.addWidget(self.lbl_gpu_fallback)

        # 3. Píldora de Monitor de Memoria RAM en tiempo real
        self.lbl_ram_monitor = QLabel("RAM: --")
        self.lbl_ram_monitor.setStyleSheet(STYLE_STATUS_PILL)
        corner_layout.addWidget(self.lbl_ram_monitor)

        # 4. Botón de Registro de Actividad y Telemetría
        self.btn_activity_log = QPushButton(tr("app.log_btn", " Registro"))
        self.btn_activity_log.setIcon(get_icon("activity", size=13))
        self.btn_activity_log.setStyleSheet(
            "QPushButton {"
            "  background-color: #2c2c32; border: 1px solid #2e2e35; border-radius: 4px;"
            "  padding: 2px 8px; color: #e4e4e7; font-weight: bold; font-size: 10px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #383840; border-color: #2f81f7; color: #ffffff;"
            "}"
            "QPushButton:pressed {"
            "  background-color: #222227;"
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
        self.setWindowTitle(tr("app.title", self.BASE_TITLE, version=APP_VERSION))
        if hasattr(self, 'lbl_engine_badge'):
            self.lbl_engine_badge.setText(tr("app.engine_badge", f"● AstroLoom Engine v{APP_VERSION}", version=APP_VERSION))
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

        if 1 in getattr(self, '_loaded_tabs', {}) and hasattr(self._loaded_tabs[1], "retranslate_ui"):
            self._loaded_tabs[1].retranslate_ui()

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

    def _setup_ui(self, splash: Optional[object] = None) -> None:
        """Inicializa el contenedor de pestañas con carga perezosa (lazy) para arranque rápido."""
        self.tab_widget = QTabWidget()
        self._tab_containers: dict[int, QWidget] = {}
        self._loaded_tabs: dict[int, QWidget] = {}

        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.stacker", "Cargando Apilador y motor CFA..."), 30)
        self.tab_stacker = StackerTab()
        self._loaded_tabs[0] = self.tab_stacker
        self.tab_widget.addTab(self.tab_stacker, "1. Apilador (Stacker)")

        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.developer", "Cargando Revelador y capas de ajuste..."), 70)
        from gui.tab_developer import DeveloperTab
        dev_tab = DeveloperTab()
        self._loaded_tabs[1] = dev_tab
        self.tab_widget.addTab(dev_tab, "2. Revelador / Editor")

        lazy_titles = [
            (2, "3. Eclipses (HDR)"),
            (3, "4. Eclipses (Collage)"),
            (4, "5. Trazas de Estrellas (Startrails)"),
            (5, "6. Lluvias de Meteoros"),
            (6, "7. Panorámicas (Mosaicos)"),
            (7, "8. Configuración / Ajustes"),
        ]
        for idx, title in lazy_titles:
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            self._tab_containers[idx] = container
            self.tab_widget.addTab(container, title)

        self.setCentralWidget(self.tab_widget)
        self.tab_widget.currentChanged.connect(self._ensure_tab_loaded)

        # Conexiones del Apilador (pestaña 0, siempre cargada)
        self.tab_stacker.stacking_finished.connect(self._on_stacking_finished)
        self.tab_stacker.new_session_requested.connect(self._on_new_session_requested)
        self.tab_stacker.session_title_changed.connect(self._update_window_title)

        if splash and hasattr(splash, "set_status"):
            splash.set_status(tr("splash.pano_settings", "Interfaz lista."), 95)

        # Precarga escalonada en segundo plano de pestañas secundarias (2..7) tras mostrar la ventana
        self._preload_queue = [2, 3, 4, 5, 6, 7]
        QTimer.singleShot(250, self._preload_next_tab)

    def _preload_next_tab(self) -> None:
        """Precarga una pestaña secundaria por tick para fluidez instantánea sin bloquear el arranque."""
        if not getattr(self, "_preload_queue", None):
            return
        idx = self._preload_queue.pop(0)
        if idx not in self._loaded_tabs:
            self._ensure_tab_loaded(idx)
        if self._preload_queue:
            QTimer.singleShot(150, self._preload_next_tab)

    def open_developer_project(self, project_path: str) -> None:
        """Conmuta al Revelador y abre un proyecto .devloom/.loom."""
        self.tab_widget.setCurrentIndex(1)
        self.tab_developer._on_canvas_files_dropped([project_path])

    def _ensure_tab_loaded(self, idx: int) -> Optional[QWidget]:
        """Instancia bajo demanda la pestaña solicitada y conecta sus señales la primera vez."""
        if idx in self._loaded_tabs:
            return self._loaded_tabs[idx]

        tab: Optional[QWidget] = None
        if idx == 1:
            from gui.tab_developer import DeveloperTab
            tab = DeveloperTab()
        elif idx == 2:
            from gui.tab_eclipse import EclipseTab
            tab = EclipseTab()
            tab.export_to_developer.connect(self._on_eclipse_exported)
            tab.export_coronal_stack_to_developer.connect(self._on_eclipse_coronal_stack_exported)
        elif idx == 3:
            from gui.tab_eclipse_collage import EclipseCollageTab
            tab = EclipseCollageTab()
            tab.export_to_developer.connect(self._on_eclipse_exported)
        elif idx == 4:
            from gui.tab_startrails import StartrailsTab
            tab = StartrailsTab()
            tab.export_to_developer.connect(self._on_startrail_exported)
            tab.set_stacker_mask_provider(lambda: getattr(self.tab_stacker, "computed_mask", None))
        elif idx == 5:
            from gui.tab_meteors import MeteorTab
            tab = MeteorTab()
            tab.export_to_developer.connect(self._on_meteor_exported)
            tab.set_stacker_mask_provider(lambda: getattr(self.tab_stacker, "computed_mask", None))
        elif idx == 6:
            from gui.tab_panorama import PanoramaTab
            tab = PanoramaTab()
            tab.export_to_developer.connect(self._on_panorama_exported)
        elif idx == 7:
            from gui.tab_settings import SettingsTab
            tab = SettingsTab()

        if tab is not None:
            self._loaded_tabs[idx] = tab
            container = self._tab_containers.get(idx)
            if container is not None and container.layout() is not None:
                container.layout().addWidget(tab)
        return tab

    @property
    def tab_developer(self):
        return self._ensure_tab_loaded(1)

    @property
    def tab_eclipse(self):
        return self._ensure_tab_loaded(2)

    @property
    def tab_eclipse_collage(self):
        return self._ensure_tab_loaded(3)

    @property
    def tab_startrails(self):
        return self._ensure_tab_loaded(4)

    @property
    def tab_meteors(self):
        return self._ensure_tab_loaded(5)

    @property
    def tab_panorama(self):
        return self._ensure_tab_loaded(6)

    @property
    def tab_settings(self):
        return self._ensure_tab_loaded(7)

    def _on_new_session_requested(self) -> None:
        dev = self._loaded_tabs.get(1)
        if dev is not None and hasattr(dev, "clear_session"):
            dev.clear_session()

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
        layers = getattr(self.tab_panorama, "_pending_developer_layers", None)
        self.tab_panorama._pending_developer_layers = None
        if layers:
            self.tab_developer.load_coronal_stack(output_path, layers)
        else:
            self.tab_developer.load_image_direct(output_path, mask=None)
        self.tab_widget.setCurrentIndex(1)

    def closeEvent(self, event) -> None:
        """Gestiona el cierre ordenado de la aplicación, guardado de proyectos y purga de temporales."""
        # 1. Comprobar si hay cambios sin guardar en el Revelador (solo si se llegó a abrir)
        dev = self._loaded_tabs.get(1) if hasattr(self, '_loaded_tabs') else None
        if dev is not None and hasattr(dev, '_check_unsaved_changes'):
            if not dev._check_unsaved_changes():
                event.ignore()
                return

        # 2. Detener temporizadores activos
        if hasattr(self, '_timer_ram') and self._timer_ram.isActive():
            self._timer_ram.stop()

        # 3. Limpiar archivos y directorios temporales residuales
        self._cleanup_temp_files()

        event.accept()

    def _cleanup_temp_files(self) -> None:
        """Elimina carpetas temporales huérfanas creadas por procesos de AstroLoom en %TEMP%."""
        base_temp = tempfile.gettempdir()
        patterns = [
            os.path.join(base_temp, "starnet_run_*"),
            os.path.join(base_temp, "astroloom_pano_burst_*"),
            os.path.join(base_temp, "astroloom_temp_*"),
        ]
        for pattern in patterns:
            try:
                for match in glob.glob(pattern):
                    if os.path.isdir(match):
                        shutil.rmtree(match, ignore_errors=True)
                    elif os.path.isfile(match):
                        try:
                            os.remove(match)
                        except OSError:
                            pass
            except Exception:
                pass

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        from gui.widgets.drop_widgets import collect_image_files_from_mime
        paths = collect_image_files_from_mime(event.mimeData())
        if not paths:
            event.ignore()
            return

        event.acceptProposedAction()
        first_file = paths[0]

        # 1. Si se suelta un proyecto .devloom, conmutar al Revelador y cargarlo
        if first_file.lower().endswith((".devloom", ".loom")):
            self.open_developer_project(first_file)
            return

        # 2. Si no es proyecto, enrutar según la pestaña actualmente activa
        current_tab_idx = self.tab_widget.currentIndex()
        if current_tab_idx == 0:  # Apilador
            if hasattr(self, 'tab_stacker'):
                self.tab_stacker._on_canvas_files_dropped(paths)
        elif current_tab_idx == 1:  # Revelador
            if hasattr(self, 'tab_developer'):
                self.tab_developer._on_canvas_files_dropped([first_file])
        elif current_tab_idx == 2:  # Eclipses HDR
            if hasattr(self, 'tab_eclipse'):
                self.tab_eclipse.add_bracketing_paths(paths)
        elif current_tab_idx == 3:  # Eclipses Collage
            if hasattr(self, 'tab_eclipse_collage'):
                self.tab_eclipse_collage.load_filepaths(paths)
        elif current_tab_idx == 4:  # Startrails
            if hasattr(self, 'tab_startrails'):
                self.tab_startrails.add_series_files(paths)
        elif current_tab_idx == 5:  # Meteoros
            if hasattr(self, 'tab_meteors'):
                self.tab_meteors.add_series_files(paths)
        elif current_tab_idx == 6:  # Panorámicas
            if hasattr(self, 'tab_panorama'):
                self.tab_panorama.load_filepaths(paths)
        else:
            if len(paths) == 1:
                self.tab_widget.setCurrentIndex(1)
                if hasattr(self, 'tab_developer'):
                    self.tab_developer._on_canvas_files_dropped([first_file])
            else:
                self.tab_widget.setCurrentIndex(0)
                if hasattr(self, 'tab_stacker'):
                    self.tab_stacker._on_canvas_files_dropped(paths)
