# gui/tab_settings.py
"""
gui/tab_settings.py - Pestaña de configuración de hardware, rutas de trabajo, herramientas externas y parámetros generales.
Diseño moderno dividido en 2 columnas: panel de configuración técnica y panel visual de galería con resultados reales de Apilator.
"""
import os
from typing import Optional
from PySide6.QtCore import Qt, QRectF, QUrl
from PySide6.QtGui import (
    QPixmap, QPainter, QPainterPath, QColor, QPen, QFont,
    QDesktopServices
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QScrollArea, QSplitter, QSizePolicy, QFrame
)
from core.config_manager import load_config, save_config

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SHOWCASE_IMG_PATH = os.path.join(ASSETS_DIR, "showcase_milkyway.jpg")


class ShowcaseViewer(QWidget):
    """
    Visor interactivo y responsivo para mostrar la fotografía de demostración de Apilator.
    Mantiene la relación de aspecto original (KeepAspectRatio), escala suavemente
    y aplica bordes redondeados con renderizado antialias.
    """
    def __init__(self, image_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.image_path = image_path
        self._pixmap = QPixmap(image_path) if os.path.exists(image_path) else None
        self.setMinimumSize(220, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        w = self.width()
        h = self.height()

        # Fondo oscuro neutro
        painter.fillRect(0, 0, w, h, QColor("#121216"))

        if self._pixmap and not self._pixmap.isNull():
            # Escalar manteniendo aspecto
            scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2

            # Recorte con esquinas redondeadas
            path = QPainterPath()
            path.addRoundedRect(QRectF(x, y, scaled.width(), scaled.height()), 8, 8)
            painter.setClipPath(path)
            painter.drawPixmap(x, y, scaled)

            # Borde sutil acentuado
            painter.setClipping(False)
            pen = QPen(QColor("#2f2f3d"), 1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(x, y, scaled.width(), scaled.height()), 8, 8)
        else:
            painter.setPen(QColor("#78909c"))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "Imagen de demostración no disponible")


class SettingsTab(QWidget):
    """Pestaña moderna para la gestión de preferencias globales de Apilator con visor de resultados."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cfg = load_config()
        self._setup_ui()
        self._apply_styles()
        self._update_starnet_status()

    def _setup_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # Divisor horizontal de 2 columnas
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # =============================================================
        # COLUMNA IZQUIERDA: Panel de Ajustes con ScrollArea
        # =============================================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(6, 6, 12, 6)
        left_layout.setSpacing(14)

        # Encabezado del módulo
        header_box = QWidget()
        header_layout = QVBoxLayout(header_box)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(2)

        lbl_title = QLabel("⚙️ Configuración y Preferencias")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(lbl_title)

        lbl_subtitle = QLabel("Gestión de directorios del proyecto, aceleración AI y parámetros de hardware.")
        lbl_subtitle.setStyleSheet("font-size: 11px; color: #9e9e9e;")
        header_layout.addWidget(lbl_subtitle)
        left_layout.addWidget(header_box)

        # -------------------------------------------------------------
        # 1. Grupo: Directorios de Trabajo por Defecto
        # -------------------------------------------------------------
        grp_dirs = QGroupBox("1. Directorios de Trabajo del Proyecto")
        dirs_layout = QVBoxLayout(grp_dirs)
        dirs_layout.setContentsMargins(12, 14, 12, 12)
        dirs_layout.setSpacing(8)

        # Sesiones
        lbl_sess = QLabel("📁 Carpeta de Sesiones (.mwstack y máscaras):")
        dirs_layout.addWidget(lbl_sess)
        row_sess = QHBoxLayout()
        row_sess.setSpacing(8)
        self.txt_sessions = QLineEdit(self.cfg.get("sessions_dir", ""))
        self.txt_sessions.setReadOnly(True)
        row_sess.addWidget(self.txt_sessions)
        btn_browse_sess = QPushButton("Examinar...")
        btn_browse_sess.clicked.connect(lambda: self._browse_directory(self.txt_sessions, "Seleccionar Carpeta de Sesiones"))
        row_sess.addWidget(btn_browse_sess)
        dirs_layout.addLayout(row_sess)

        # Imágenes apiladas
        lbl_stack = QLabel("⚡ Carpeta de Salida de Apilados (TIFF lineales 32-bit):")
        dirs_layout.addWidget(lbl_stack)
        row_stack = QHBoxLayout()
        row_stack.setSpacing(8)
        self.txt_stacked = QLineEdit(self.cfg.get("stacked_dir", ""))
        self.txt_stacked.setReadOnly(True)
        row_stack.addWidget(self.txt_stacked)
        btn_browse_stack = QPushButton("Examinar...")
        btn_browse_stack.clicked.connect(lambda: self._browse_directory(self.txt_stacked, "Seleccionar Carpeta de Imágenes Apiladas"))
        row_stack.addWidget(btn_browse_stack)
        dirs_layout.addLayout(row_stack)

        # Exportaciones finales
        lbl_exp = QLabel("💾 Carpeta de Exportación de Revelados (TIFF 16-bit, PNG, JPG):")
        dirs_layout.addWidget(lbl_exp)
        row_exp = QHBoxLayout()
        row_exp.setSpacing(8)
        self.txt_export = QLineEdit(self.cfg.get("export_dir", ""))
        self.txt_export.setReadOnly(True)
        row_exp.addWidget(self.txt_export)
        btn_browse_exp = QPushButton("Examinar...")
        btn_browse_exp.clicked.connect(lambda: self._browse_directory(self.txt_export, "Seleccionar Carpeta de Exportaciones"))
        row_exp.addWidget(btn_browse_exp)
        dirs_layout.addLayout(row_exp)

        left_layout.addWidget(grp_dirs)

        # -------------------------------------------------------------
        # 2. Grupo: Herramientas Externas (AI)
        # -------------------------------------------------------------
        grp_paths = QGroupBox("2. Herramientas Externas (AI)")
        paths_layout = QVBoxLayout(grp_paths)
        paths_layout.setContentsMargins(12, 14, 12, 12)
        paths_layout.setSpacing(8)

        lbl_sn = QLabel("✨ Ruta al ejecutable de StarNet++ (starnet2.exe / starnet++.exe):")
        paths_layout.addWidget(lbl_sn)
        row_sn = QHBoxLayout()
        row_sn.setSpacing(8)
        self.txt_starnet = QLineEdit(self.cfg.get("starnet_exe", ""))
        self.txt_starnet.setPlaceholderText("C:\\Program Files\\StarNetv2CLI_Win\\starnet++.exe")
        self.txt_starnet.textChanged.connect(self._update_starnet_status)
        row_sn.addWidget(self.txt_starnet)

        btn_browse_sn = QPushButton("Examinar...")
        btn_browse_sn.clicked.connect(self._browse_starnet)
        row_sn.addWidget(btn_browse_sn)
        paths_layout.addLayout(row_sn)

        # Indicador de estado de StarNet
        self.lbl_sn_status = QLabel("")
        paths_layout.addWidget(self.lbl_sn_status)

        left_layout.addWidget(grp_paths)

        # -------------------------------------------------------------
        # 3. Grupo: Parámetros de Procesado por Defecto
        # -------------------------------------------------------------
        grp_defaults = QGroupBox("3. Parámetros de Procesado por Defecto")
        def_layout = QVBoxLayout(grp_defaults)
        def_layout.setContentsMargins(12, 14, 12, 12)
        def_layout.setSpacing(10)

        # Factor Kappa
        row_k = QHBoxLayout()
        lbl_kappa = QLabel("Factor Kappa por defecto (Rechazo Sigma-Clipping):")
        row_k.addWidget(lbl_kappa)
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(0.5, 5.0)
        self.spin_kappa.setValue(float(self.cfg.get("default_kappa", 2.2)))
        self.spin_kappa.setSingleStep(0.1)
        self.spin_kappa.setFixedWidth(80)
        row_k.addWidget(self.spin_kappa)
        row_k.addStretch()
        def_layout.addLayout(row_k)

        # Dimensión máxima proxy
        row_px = QHBoxLayout()
        lbl_proxy = QLabel("Dimensión máxima del proxy de previsualización:")
        row_px.addWidget(lbl_proxy)
        self.spin_proxy = QSpinBox()
        self.spin_proxy.setRange(800, 3840)
        self.spin_proxy.setSingleStep(200)
        self.spin_proxy.setSuffix(" px")
        self.spin_proxy.setValue(int(self.cfg.get("preview_max_dim", 1600)))
        self.spin_proxy.setFixedWidth(100)
        row_px.addWidget(self.spin_proxy)
        row_px.addStretch()
        def_layout.addLayout(row_px)

        # Perfil de color ICC incrustado
        row_icc = QHBoxLayout()
        lbl_icc = QLabel("Perfil de color ICC incrustado en TIFF:")
        row_icc.addWidget(lbl_icc)
        self.combo_icc = QComboBox()
        self.combo_icc.addItem("🌐 sRGB (Estándar Web y Pantallas)", "srgb")
        self.combo_icc.addItem("🎨 Adobe RGB (1998) (Gama amplia)", "adobe")
        self.combo_icc.addItem("🌌 ProPhoto RGB (Gama ultra-ancha / Astrofotografía)", "prophoto")

        saved_icc = self.cfg.get("icc_profile", "srgb")
        idx_icc = self.combo_icc.findData(saved_icc)
        if idx_icc >= 0:
            self.combo_icc.setCurrentIndex(idx_icc)

        self.combo_icc.setToolTip(
            "Define el espacio de color para la decodificación RAW y el perfil ICC incrustado en el TIFF:\n"
            "- sRGB: Máxima compatibilidad con visores web, móviles y monitores estándar.\n"
            "- Adobe RGB (1998): Mayor riqueza en cianes, verdes y fotografía de paisaje.\n"
            "- ProPhoto RGB: Gama máxima para astrofotografía a 16/32 bits, evitando cualquier\n"
            "  recorte de color (gamut clipping) en nebulosas intensas (H-Alfa, OIII)."
        )
        row_icc.addWidget(self.combo_icc)
        row_icc.addStretch()
        def_layout.addLayout(row_icc)

        left_layout.addWidget(grp_defaults)

        # -------------------------------------------------------------
        # 4. Grupo: Rendimiento de Hardware y Almacenamiento
        # -------------------------------------------------------------
        grp_perf = QGroupBox("4. Rendimiento de Hardware y Memoria")
        layout_perf = QVBoxLayout(grp_perf)
        layout_perf.setContentsMargins(12, 14, 12, 12)
        layout_perf.setSpacing(10)

        # Hilos concurrentes de CPU
        row_cpu = QHBoxLayout()
        lbl_cpu = QLabel("Hilos simultáneos de CPU (Multi-Core):")
        row_cpu.addWidget(lbl_cpu)
        self.spin_cpu_workers = QSpinBox()
        max_cpus = max(4, os.cpu_count() or 4)
        def_cpus = max(2, max_cpus - 2) if max_cpus > 2 else max_cpus
        self.spin_cpu_workers.setRange(1, max_cpus)
        self.spin_cpu_workers.setValue(int(self.cfg.get("cpu_workers", def_cpus)))
        self.spin_cpu_workers.setFixedWidth(80)
        self.spin_cpu_workers.setToolTip(
            f"Número de hilos simultáneos para decodificación, cribado de calidad y apilado en paralelo.\n"
            f"Tu equipo dispone de {max_cpus} núcleos lógicos. Se recomienda dejar 1-2 núcleos libres para el sistema operativo."
        )
        row_cpu.addWidget(self.spin_cpu_workers)
        row_cpu.addStretch()
        layout_perf.addLayout(row_cpu)

        # Estrategia de almacenamiento intermedio
        row_storage = QHBoxLayout()
        lbl_strg = QLabel("Estrategia de búfer intermedio:")
        row_storage.addWidget(lbl_strg)
        self.combo_storage = QComboBox()
        self.combo_storage.addItem("⚡ Automático (según RAM disponible)", "auto")
        self.combo_storage.addItem("🚀 Memoria RAM (Ultra-rápido)", "ram")
        self.combo_storage.addItem("💾 Caché en Disco (Bajo consumo RAM)", "disk")

        saved_storage = self.cfg.get("storage_strategy", "auto")
        idx_storage = self.combo_storage.findData(saved_storage)
        if idx_storage >= 0:
            self.combo_storage.setCurrentIndex(idx_storage)

        self.combo_storage.setToolTip(
            "Define dónde se mantienen los cuadros alineados antes del apilado final:\n"
            "- RAM: Elimina escrituras y lecturas a disco (.bin). Recomendado con 32 GB o más.\n"
            "- Disco: Escribe archivos binarios temporales. Ideal para equipos con menos memoria.\n"
            "- Automático: Evalúa la RAM física libre antes de empezar."
        )
        row_storage.addWidget(self.combo_storage)
        row_storage.addStretch()
        layout_perf.addLayout(row_storage)

        # Estado de Aceleración Numba JIT
        row_numba = QHBoxLayout()
        lbl_numba_title = QLabel("Aceleración JIT (Numba LLVM):")
        row_numba.addWidget(lbl_numba_title)

        try:
            from core.stacking import HAS_NUMBA
        except Exception:
            HAS_NUMBA = False

        status_txt = "🟢 Activa (Kernels compilados en caché L1/L2)" if HAS_NUMBA else "⚪ Inactiva (Fallback a NumPy estándar)"
        status_color = "#b9f6ca" if HAS_NUMBA else "#9e9e9e"
        lbl_numba_status = QLabel(status_txt)
        lbl_numba_status.setStyleSheet(f"color: {status_color}; font-weight: bold; font-size: 11px;")
        row_numba.addWidget(lbl_numba_status)
        row_numba.addStretch()
        layout_perf.addLayout(row_numba)

        left_layout.addWidget(grp_perf)

        # -------------------------------------------------------------
        # 5. Botón Guardar Cambios
        # -------------------------------------------------------------
        btn_save = QPushButton("💾 Guardar Cambios de Configuración")
        btn_save.setObjectName("btn_save")
        btn_save.setFixedHeight(42)
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.clicked.connect(self.save_settings)
        left_layout.addWidget(btn_save)

        left_layout.addStretch()

        scroll_area.setWidget(left_content)
        splitter.addWidget(scroll_area)

        # =============================================================
        # COLUMNA DERECHA: Galería / Muestra Visual de Resultados
        # =============================================================
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(10, 6, 10, 6)
        right_layout.setSpacing(10)

        # Tarjeta contenedora de la vitrina
        showcase_card = QWidget()
        showcase_card.setObjectName("showcase_card")
        card_layout = QVBoxLayout(showcase_card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        # Cabecera de la vitrina
        header_gallery = QHBoxLayout()
        lbl_gal_icon = QLabel("🌌")
        lbl_gal_icon.setStyleSheet("font-size: 16px;")
        header_gallery.addWidget(lbl_gal_icon)

        lbl_gal_title = QLabel("Galería de Resultados — Motor Apilator")
        lbl_gal_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #90caf9;")
        header_gallery.addWidget(lbl_gal_title)
        header_gallery.addStretch()

        btn_open_full = QPushButton("🔍 Ver imagen completa")
        btn_open_full.setObjectName("btn_open_full")
        btn_open_full.setToolTip("Abre la imagen a resolución original en el visor predeterminado del sistema")
        btn_open_full.clicked.connect(self._open_showcase_in_viewer)
        header_gallery.addWidget(btn_open_full)
        card_layout.addLayout(header_gallery)

        # Visor de la fotografía
        self.showcase_viewer = ShowcaseViewer(SHOWCASE_IMG_PATH)
        card_layout.addWidget(self.showcase_viewer, stretch=1)

        # Tarjeta de metadatos y créditos del autor
        author_box = QWidget()
        author_box.setObjectName("author_box")
        author_layout = QVBoxLayout(author_box)
        author_layout.setContentsMargins(12, 10, 12, 10)
        author_layout.setSpacing(6)

        row_author_title = QHBoxLayout()
        lbl_work_title = QLabel("Vía Láctea Estival — Centro Galáctico y Paisaje")
        lbl_work_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #ffd54f;")
        row_author_title.addWidget(lbl_work_title)
        row_author_title.addStretch()

        self.lbl_engine_badge = QLabel("Apilator v0.7.0")
        self.lbl_engine_badge.setStyleSheet("color: #80d8ff; font-size: 11px; font-weight: bold;")
        row_author_title.addWidget(self.lbl_engine_badge)
        author_layout.addLayout(row_author_title)

        lbl_credits = QLabel("📷 Fotografía y Procesado: Shilmar")
        lbl_credits.setStyleSheet("font-size: 12px; color: #e0e0e0; font-weight: 500;")
        author_layout.addWidget(lbl_credits)

        # Fila de etiquetas técnicas de procesamiento
        row_tags = QHBoxLayout()
        row_tags.setSpacing(6)

        tags = [
            ("⚡ 32-bit Float HDR", "#80d8ff", "#0d2b3e", "#1b4d6e"),
            ("✨ StarNet++ AI", "#ce93d8", "#2c1236", "#59266e"),
            ("📐 MTF Asinh", "#ffd54f", "#332a0d", "#66541a"),
            ("🎯 Alineación Subpíxel", "#a5d6a7", "#132d18", "#275931"),
        ]
        for tag_text, fg_col, bg_col, border_col in tags:
            tag_lbl = QLabel(tag_text)
            tag_lbl.setStyleSheet(f"""
                QLabel {{
                    color: {fg_col};
                    background-color: {bg_col};
                    border: 1px solid {border_col};
                    border-radius: 4px;
                    padding: 2px 6px;
                    font-size: 10px;
                    font-weight: bold;
                }}
            """)
            row_tags.addWidget(tag_lbl)
        row_tags.addStretch()
        author_layout.addLayout(row_tags)

        card_layout.addWidget(author_box)

        right_layout.addWidget(showcase_card)
        splitter.addWidget(right_container)

        # Proporción de reparto (52% izquierda, 48% derecha)
        splitter.setStretchFactor(0, 52)
        splitter.setStretchFactor(1, 48)
        splitter.setSizes([620, 580])

        root_layout.addWidget(splitter)

    def _apply_styles(self) -> None:
        """Aplica la hoja de estilos unificada consistente con el resto de módulos."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #2d2d3a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 14px;
                background-color: #1a1a22;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 4px;
                color: #90caf9;
            }
            QLineEdit {
                background-color: #121216;
                border: 1px solid #33333f;
                border-radius: 4px;
                padding: 6px 8px;
                color: #f0f0f0;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #0288d1;
            }
            QPushButton {
                background-color: #252530;
                border: 1px solid #3a3a4c;
                border-radius: 4px;
                padding: 6px 12px;
                color: #e0e0e0;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2f2f3d;
                border-color: #55556a;
            }
            QPushButton:pressed {
                background-color: #1c1c24;
            }
            QPushButton#btn_save {
                background-color: #2e7d32;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
                border: none;
                padding: 8px 16px;
            }
            QPushButton#btn_save:hover {
                background-color: #388e3c;
            }
            QPushButton#btn_save:pressed {
                background-color: #1b5e20;
            }
            QPushButton#btn_open_full {
                background-color: #1e1e28;
                border: 1px solid #3d3d4e;
                font-size: 11px;
                padding: 4px 8px;
            }
            QPushButton#btn_open_full:hover {
                background-color: #2a2a38;
                border-color: #80d8ff;
                color: #80d8ff;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #121216;
                border: 1px solid #33333f;
                border-radius: 4px;
                padding: 4px 6px;
                color: #f0f0f0;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #0288d1;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a22;
                color: #f0f0f0;
                selection-background-color: #0288d1;
            }
            QWidget#showcase_card {
                background-color: #181822;
                border: 1px solid #2d2d3a;
                border-radius: 8px;
            }
            QWidget#author_box {
                background-color: #13131c;
                border: 1px solid #252534;
                border-radius: 6px;
            }
        """)

    def _update_starnet_status(self) -> None:
        """Actualiza el indicador visual de disponibilidad del CLI de StarNet++."""
        path = self.txt_starnet.text().strip()
        if path and os.path.isfile(path):
            self.lbl_sn_status.setText("✓ StarNet++ detectado y listo para aceleración por IA")
            self.lbl_sn_status.setStyleSheet("color: #81c784; font-size: 11px; font-weight: bold;")
        elif path:
            self.lbl_sn_status.setText("⚠️ El archivo especificado no existe o la ruta no es accesible")
            self.lbl_sn_status.setStyleSheet("color: #ffb74d; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_sn_status.setText("ℹ️ Sin configurar (opcional para el desacoplo de estrellas en el Revelador)")
            self.lbl_sn_status.setStyleSheet("color: #90a4ae; font-size: 11px;")

    def _browse_directory(self, target_line_edit: QLineEdit, dialog_title: str) -> None:
        """Abre un diálogo nativo para seleccionar una carpeta física."""
        current_val = target_line_edit.text().strip()
        start_dir = current_val if (current_val and os.path.exists(current_val)) else os.getcwd()
        chosen = QFileDialog.getExistingDirectory(self, dialog_title, start_dir)
        if chosen:
            target_line_edit.setText(os.path.normpath(chosen))

    def _browse_starnet(self) -> None:
        """Abre el explorador de archivos para localizar el ejecutable de StarNet."""
        current_val = self.txt_starnet.text().strip()
        start_dir = os.path.dirname(current_val) if current_val and os.path.exists(os.path.dirname(current_val)) else "C:\\Program Files"

        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar ejecutable de StarNet++", 
            start_dir, "Ejecutables (*.exe);;Todos (*.*)"
        )
        if f:
            self.txt_starnet.setText(os.path.normpath(f))

    def _open_showcase_in_viewer(self) -> None:
        """Abre la fotografía de demostración en el visor de imágenes del sistema operativo."""
        if os.path.exists(SHOWCASE_IMG_PATH):
            QDesktopServices.openUrl(QUrl.fromLocalFile(SHOWCASE_IMG_PATH))
        else:
            QMessageBox.warning(self, "Imagen no encontrada", f"No se localizó el archivo:\n{SHOWCASE_IMG_PATH}")

    def save_settings(self) -> None:
        """Persiste las opciones seleccionadas en el archivo config.json."""
        self.cfg["sessions_dir"] = self.txt_sessions.text().strip()
        self.cfg["stacked_dir"] = self.txt_stacked.text().strip()
        self.cfg["export_dir"] = self.txt_export.text().strip()
        self.cfg["masks_dir"] = os.path.join(self.cfg["sessions_dir"], "mascaras")

        self.cfg["starnet_exe"] = self.txt_starnet.text().strip()
        self.cfg["default_kappa"] = self.spin_kappa.value()
        self.cfg["preview_max_dim"] = self.spin_proxy.value()
        self.cfg["cpu_workers"] = self.spin_cpu_workers.value()
        self.cfg["storage_strategy"] = self.combo_storage.currentData()
        self.cfg["icc_profile"] = self.combo_icc.currentData()

        save_config(self.cfg)
        QMessageBox.information(self, "Ajustes", "Configuración guardada correctamente en config.json.")