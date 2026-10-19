# gui/tab_settings.py
"""
gui/tab_settings.py - Pestaña de configuración de hardware, rutas de trabajo, herramientas externas y parámetros generales.
Diseño moderno dividido en 2 columnas: panel de configuración técnica con acordeones profesionales y panel visual de galería con resultados reales de AstroLoom.
"""
import os
from typing import Optional
from PySide6.QtCore import Qt, QRectF, QUrl
from PySide6.QtGui import (
    QPixmap, QPainter, QPainterPath, QColor, QPen, QFont,
    QDesktopServices
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QScrollArea, QSplitter, QSizePolicy, QFrame
)
from core.config_manager import load_config, save_config
from core.i18n import tr, get_i18n
from gui.theme import STYLE_RUN_IDLE, BG_CARD, BG_INPUT, BORDER_BASE
from gui.widgets.collapsible_section import CollapsibleSection
from gui.widgets.precision_slider import PrecisionSliderRow, PrecisionSpinBox
from gui.icons import (
    get_icon, get_svg_pixmap, 
    ICON_COLOR_DEFAULT, ICON_COLOR_ACCENT, ICON_COLOR_SUCCESS, 
    ICON_COLOR_WARNING, ICON_COLOR_MUTED
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SHOWCASE_IMG_PATH = os.path.join(ASSETS_DIR, "showcase_milkyway.jpg")


class ShowcaseViewer(QWidget):
    """
    Visor interactivo y responsivo para mostrar la fotografía de demostración de AstroLoom.
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
        painter.fillRect(0, 0, w, h, QColor("#12141e"))

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
            pen = QPen(QColor("#232738"), 1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(x, y, scaled.width(), scaled.height()), 8, 8)
        else:
            painter.setPen(QColor("#78909c"))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "Imagen de demostración no disponible")


class SettingsTab(QWidget):
    """Pestaña moderna para la gestión de preferencias globales de AstroLoom con visor de resultados."""

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
        left_layout.setContentsMargins(6, 6, 12, 16)
        left_layout.setSpacing(10)

        # Encabezado del módulo con icono SVG vectorial
        header_box = QWidget()
        header_layout = QHBoxLayout(header_box)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(10)

        lbl_icon = QLabel()
        lbl_icon.setPixmap(get_svg_pixmap("settings", color=ICON_COLOR_ACCENT, size=24))
        header_layout.addWidget(lbl_icon)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        lbl_title = QLabel("Configuración y Preferencias")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        title_vbox.addWidget(lbl_title)

        lbl_subtitle = QLabel("Gestión de directorios del proyecto, aceleración AI y parámetros de hardware.")
        lbl_subtitle.setStyleSheet("font-size: 11px; color: #94a3b8;")
        title_vbox.addWidget(lbl_subtitle)
        header_layout.addLayout(title_vbox)
        header_layout.addStretch()

        left_layout.addWidget(header_box)

        # -------------------------------------------------------------
        # 1. Grupo: Directorios de Trabajo por Defecto
        # -------------------------------------------------------------
        self.sec_dirs = CollapsibleSection("1. Directorios de Trabajo del Proyecto", icon="folder")
        dirs_layout = QVBoxLayout()
        dirs_layout.setContentsMargins(4, 6, 4, 6)
        dirs_layout.setSpacing(8)

        def make_dir_row(label_text: str, config_key: str, dialog_title: str):
            vbox = QVBoxLayout()
            vbox.setSpacing(3)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 500;")
            vbox.addWidget(lbl)

            row = QHBoxLayout()
            row.setSpacing(6)
            txt = QLineEdit(self.cfg.get(config_key, ""))
            txt.setReadOnly(True)
            txt.setStyleSheet(
                "background: #141722; border: 1px solid #232738; color: #e2e8f0; "
                "border-radius: 4px; padding: 4px 8px; font-size: 11px;"
            )
            row.addWidget(txt)

            btn = QPushButton("Examinar...")
            btn.setIcon(get_icon("folder", color="#cbd5e1"))
            btn.setStyleSheet("""
                QPushButton {
                    background: #1c202e;
                    border: 1px solid #2e354a;
                    color: #cbd5e1;
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background: #252b3d;
                    border-color: #38bdf8;
                    color: #ffffff;
                }
            """)
            btn.clicked.connect(lambda: self._browse_directory(txt, dialog_title))
            row.addWidget(btn)
            vbox.addLayout(row)
            return vbox, txt

        # Sesiones
        box_sess, self.txt_sessions = make_dir_row(
            "Carpeta de Sesiones (.mwstack y máscaras):", 
            "sessions_dir", 
            "Seleccionar Carpeta de Sesiones"
        )
        dirs_layout.addLayout(box_sess)

        # Imágenes apiladas
        box_stack, self.txt_stacked = make_dir_row(
            "Carpeta de Salida de Apilados (TIFF lineales 32-bit):", 
            "stacked_dir", 
            "Seleccionar Carpeta de Imágenes Apiladas"
        )
        dirs_layout.addLayout(box_stack)

        # Exportaciones finales
        box_exp, self.txt_export = make_dir_row(
            "Carpeta de Exportación de Revelados (TIFF 16-bit, PNG, JPG):", 
            "export_dir", 
            "Seleccionar Carpeta de Exportaciones"
        )
        dirs_layout.addLayout(box_exp)

        # Presets y Estilos
        box_presets, self.txt_presets = make_dir_row(
            "Carpeta de Presets de Revelado (.json):", 
            "presets_dir", 
            "Seleccionar Carpeta de Presets"
        )
        dirs_layout.addLayout(box_presets)

        self.sec_dirs.setContentLayout(dirs_layout)
        self.grp_dirs = self.sec_dirs
        left_layout.addWidget(self.sec_dirs)

        # -------------------------------------------------------------
        # 2. Grupo: Herramientas Externas (AI)
        # -------------------------------------------------------------
        self.sec_tools = CollapsibleSection("2. Herramientas Externas e Inteligencia Artificial", icon="cpu")
        tools_layout = QVBoxLayout()
        tools_layout.setContentsMargins(4, 6, 4, 6)
        tools_layout.setSpacing(8)

        lbl_sn = QLabel("Ruta al ejecutable de StarNet++ (starnet2.exe / starnet++.exe):")
        lbl_sn.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 500;")
        tools_layout.addWidget(lbl_sn)

        row_sn = QHBoxLayout()
        row_sn.setSpacing(6)
        self.txt_starnet = QLineEdit(self.cfg.get("starnet_exe", ""))
        self.txt_starnet.setPlaceholderText("C:\\Program Files\\StarNetv2CLI_Win\\starnet++.exe")
        self.txt_starnet.setStyleSheet(
            "background: #141722; border: 1px solid #232738; color: #e2e8f0; "
            "border-radius: 4px; padding: 4px 8px; font-size: 11px;"
        )
        self.txt_starnet.textChanged.connect(self._update_starnet_status)
        row_sn.addWidget(self.txt_starnet)

        btn_browse_sn = QPushButton("Examinar...")
        btn_browse_sn.setIcon(get_icon("folder", color="#cbd5e1"))
        btn_browse_sn.setStyleSheet("""
            QPushButton {
                background: #1c202e;
                border: 1px solid #2e354a;
                color: #cbd5e1;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #252b3d;
                border-color: #38bdf8;
                color: #ffffff;
            }
        """)
        btn_browse_sn.clicked.connect(self._browse_starnet)
        row_sn.addWidget(btn_browse_sn)
        tools_layout.addLayout(row_sn)

        # Indicador de estado de StarNet
        self.lbl_sn_status = QLabel("")
        self.lbl_sn_status.setStyleSheet("font-size: 11px; margin-top: 2px;")
        tools_layout.addWidget(self.lbl_sn_status)

        self.sec_tools.setContentLayout(tools_layout)
        self.grp_paths = self.sec_tools
        left_layout.addWidget(self.sec_tools)

        # -------------------------------------------------------------
        # 3. Grupo: Parámetros de Procesado por Defecto
        # -------------------------------------------------------------
        self.sec_defaults = CollapsibleSection("3. Parámetros de Procesado por Defecto", icon="sliders")
        def_layout = QVBoxLayout()
        def_layout.setContentsMargins(4, 6, 4, 6)
        def_layout.setSpacing(8)

        # Factor Kappa
        kappa_val = float(self.cfg.get("default_kappa", 2.2))
        self.spin_kappa = PrecisionSliderRow(
            title="Factor Kappa:",
            min_val=5,
            max_val=50,
            default_val=int(round(kappa_val * 10)),
            scale=10.0,
            decimals=1,
            label_width=180,
            value_width=64
        )
        # Compatibilidad: .value() retorna float para alinearse con QDoubleSpinBox
        self.spin_kappa.value = self.spin_kappa.float_value
        def_layout.addWidget(self.spin_kappa)

        # Dimensión máxima proxy
        proxy_val = int(self.cfg.get("preview_max_dim", 1600))
        self.spin_proxy = PrecisionSliderRow(
            title="Proxy de Previsualización:",
            min_val=800,
            max_val=3840,
            default_val=proxy_val,
            step=100,
            unit=" px",
            label_width=180,
            value_width=75
        )
        def_layout.addWidget(self.spin_proxy)

        # Perfil de color ICC incrustado
        row_icc = QHBoxLayout()
        lbl_icc = QLabel("Perfil de color ICC en TIFF:")
        lbl_icc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_icc.setFixedWidth(170)
        row_icc.addWidget(lbl_icc)

        self.combo_icc = QComboBox()
        self.combo_icc.addItem("sRGB (Estándar Web y Pantallas)", "srgb")
        self.combo_icc.addItem("Adobe RGB (1998) (Gama amplia)", "adobe")
        self.combo_icc.addItem("ProPhoto RGB (Gama ultra-ancha / Astrofotografía)", "prophoto")

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
        row_icc.addWidget(self.combo_icc, stretch=1)
        def_layout.addLayout(row_icc)

        # Idioma de la interfaz / Language
        row_lang = QHBoxLayout()
        lbl_lang = QLabel(tr("settings.language", "Idioma de la Interfaz:"))
        lbl_lang.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_lang.setFixedWidth(170)
        row_lang.addWidget(lbl_lang)

        self.combo_lang = QComboBox()
        avail_langs = get_i18n().get_available_languages()
        for code, name in avail_langs:
            self.combo_lang.addItem(f"{name} ({code.upper()})", code)

        curr_lang = get_i18n().get_current_language()
        idx_lang = self.combo_lang.findData(curr_lang)
        if idx_lang >= 0:
            self.combo_lang.setCurrentIndex(idx_lang)

        self.combo_lang.setToolTip("Selecciona el idioma de los menús, botones y herramientas de AstroLoom.")
        row_lang.addWidget(self.combo_lang, stretch=1)
        def_layout.addLayout(row_lang)

        self.sec_defaults.setContentLayout(def_layout)
        self.grp_defaults = self.sec_defaults
        left_layout.addWidget(self.sec_defaults)

        # -------------------------------------------------------------
        # 4. Grupo: Rendimiento de Hardware y Almacenamiento
        # -------------------------------------------------------------
        self.sec_perf = CollapsibleSection("4. Rendimiento de Hardware y Memoria", icon="zap")
        perf_layout = QVBoxLayout()
        perf_layout.setContentsMargins(4, 6, 4, 6)
        perf_layout.setSpacing(8)

        # Hilos concurrentes de CPU
        max_cpus = max(4, os.cpu_count() or 4)
        def_cpus = max(2, max_cpus - 2) if max_cpus > 2 else max_cpus
        cpu_val = int(self.cfg.get("cpu_workers", def_cpus))
        self.spin_cpu_workers = PrecisionSliderRow(
            title="Hilos concurrentes CPU:",
            min_val=1,
            max_val=max_cpus,
            default_val=cpu_val,
            unit=" hilos",
            label_width=180,
            value_width=75
        )
        self.spin_cpu_workers.setToolTip(
            f"Número de hilos simultáneos para decodificación, cribado de calidad y apilado en paralelo.\n"
            f"Tu equipo dispone de {max_cpus} núcleos lógicos. Se recomienda dejar 1-2 núcleos libres para el sistema operativo."
        )
        perf_layout.addWidget(self.spin_cpu_workers)

        # Estrategia de almacenamiento intermedio
        row_storage = QHBoxLayout()
        lbl_strg = QLabel("Búfer intermedio:")
        lbl_strg.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_strg.setFixedWidth(170)
        row_storage.addWidget(lbl_strg)

        self.combo_storage = QComboBox()
        self.combo_storage.addItem("Automático (según RAM disponible)", "auto")
        self.combo_storage.addItem("Memoria RAM (Ultra-rápido)", "ram")
        self.combo_storage.addItem("Caché en Disco (Bajo consumo RAM)", "disk")

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
        row_storage.addWidget(self.combo_storage, stretch=1)
        perf_layout.addLayout(row_storage)

        # Estado de Aceleración Numba JIT
        row_numba = QHBoxLayout()
        lbl_numba_title = QLabel("Aceleración JIT (Numba LLVM):")
        lbl_numba_title.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_numba_title.setFixedWidth(170)
        row_numba.addWidget(lbl_numba_title)

        try:
            from core.stacking import HAS_NUMBA
        except Exception:
            HAS_NUMBA = False

        status_txt = "● Activa (Kernels compilados en caché L1/L2)" if HAS_NUMBA else "● Inactiva (Fallback a NumPy estándar)"
        status_color = "#4ade80" if HAS_NUMBA else "#94a3b8"
        lbl_numba_status = QLabel(status_txt)
        lbl_numba_status.setStyleSheet(f"color: {status_color}; font-weight: 600; font-size: 11px;")
        row_numba.addWidget(lbl_numba_status)
        row_numba.addStretch()
        perf_layout.addLayout(row_numba)

        self.sec_perf.setContentLayout(perf_layout)
        self.grp_perf = self.sec_perf
        left_layout.addWidget(self.sec_perf)

        left_layout.addStretch()

        scroll_area.setWidget(left_content)

        # -------------------------------------------------------------
        # 5. Botón Guardar Cambios (anclado fijamente al pie del panel)
        # -------------------------------------------------------------
        self.btn_save = QPushButton("Guardar Cambios de Configuración")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setIcon(get_icon("save", color="#ffffff"))
        self.btn_save.setFixedHeight(40)
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton#btn_save {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
                border-radius: 6px;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 16px;
            }
            QPushButton#btn_save:hover {
                background-color: #0369a1;
                border-color: #00e5ff;
            }
            QPushButton#btn_save:pressed {
                background-color: #075985;
            }
        """)
        self.btn_save.clicked.connect(self.save_settings)

        # Contenedor columna izquierda con scroll y botón Guardar fijo
        left_column = QWidget()
        left_col_layout = QVBoxLayout(left_column)
        left_col_layout.setContentsMargins(0, 0, 0, 0)
        left_col_layout.setSpacing(10)
        left_col_layout.addWidget(scroll_area, stretch=1)
        left_col_layout.addWidget(self.btn_save)

        splitter.addWidget(left_column)

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
        lbl_gal_icon = QLabel()
        lbl_gal_icon.setPixmap(get_svg_pixmap("image", color="#38bdf8", size=18))
        header_gallery.addWidget(lbl_gal_icon)

        lbl_gal_title = QLabel("Galería de Resultados — Motor AstroLoom")
        lbl_gal_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #90caf9;")
        header_gallery.addWidget(lbl_gal_title)
        header_gallery.addStretch()

        btn_open_full = QPushButton("Ver Imagen Completa")
        btn_open_full.setObjectName("btn_open_full")
        btn_open_full.setIcon(get_icon("maximize", color="#cbd5e1"))
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

        self.lbl_engine_badge = QLabel("AstroLoom v0.10.0")
        self.lbl_engine_badge.setStyleSheet("color: #80d8ff; font-size: 11px; font-weight: bold;")
        row_author_title.addWidget(self.lbl_engine_badge)

        author_layout.addLayout(row_author_title)

        row_credits = QHBoxLayout()
        row_credits.setSpacing(6)
        lbl_cam_icon = QLabel()
        lbl_cam_icon.setPixmap(get_svg_pixmap("camera", color="#94a3b8", size=14))
        row_credits.addWidget(lbl_cam_icon)
        lbl_credits = QLabel("Fotografía y Procesado: Shilmar")
        lbl_credits.setStyleSheet("font-size: 12px; color: #e0e0e0; font-weight: 500;")
        row_credits.addWidget(lbl_credits)
        row_credits.addStretch()
        author_layout.addLayout(row_credits)

        # Fila de etiquetas técnicas de procesamiento
        row_tags = QHBoxLayout()
        row_tags.setSpacing(6)

        tags = [
            ("● 32-bit Float HDR", "#80d8ff", "#0d2b3e", "#1b4d6e"),
            ("● StarNet++ AI", "#ce93d8", "#2c1236", "#59266e"),
            ("● MTF Asinh", "#ffd54f", "#332a0d", "#66541a"),
            ("● Alineación Subpíxel", "#a5d6a7", "#132d18", "#275931"),
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
        """Aplica estilos específicos para las tarjetas de showcase de ajustes."""
        self.setStyleSheet(f"""
            QWidget#showcase_card {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_BASE};
                border-radius: 8px;
            }}
            QWidget#author_box {{
                background-color: {BG_INPUT};
                border: 1px solid {BORDER_BASE};
                border-radius: 6px;
            }}
            QPushButton#btn_open_full {{
                background-color: #1e2230;
                border: 1px solid #33384c;
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 4px;
                color: #cbd5e1;
            }}
            QPushButton#btn_open_full:hover {{
                background-color: #282e42;
                border-color: #80d8ff;
                color: #80d8ff;
            }}
        """)

    def _update_starnet_status(self) -> None:
        """Actualiza el indicador visual de disponibilidad del CLI de StarNet++."""
        path = self.txt_starnet.text().strip()
        if path and os.path.isfile(path):
            self.lbl_sn_status.setText("● StarNet++ detectado y listo para aceleración por IA")
            self.lbl_sn_status.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: 600;")
        elif path:
            self.lbl_sn_status.setText("● El archivo especificado no existe o la ruta no es accesible")
            self.lbl_sn_status.setStyleSheet("color: #fbbf24; font-size: 11px; font-weight: 600;")
        else:
            self.lbl_sn_status.setText("● Sin configurar (opcional para el desacoplo de estrellas en el Revelador)")
            self.lbl_sn_status.setStyleSheet("color: #94a3b8; font-size: 11px;")

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
        self.cfg["presets_dir"] = self.txt_presets.text().strip()
        self.cfg["masks_dir"] = os.path.join(self.cfg["sessions_dir"], "mascaras")

        self.cfg["starnet_exe"] = self.txt_starnet.text().strip()
        self.cfg["default_kappa"] = float(self.spin_kappa.value()) if hasattr(self.spin_kappa, "value") else 2.2
        self.cfg["preview_max_dim"] = int(self.spin_proxy.value())
        self.cfg["cpu_workers"] = int(self.spin_cpu_workers.value())
        self.cfg["storage_strategy"] = self.combo_storage.currentData()
        self.cfg["icc_profile"] = self.combo_icc.currentData()

        new_lang = self.combo_lang.currentData()
        if new_lang:
            self.cfg["language"] = new_lang
            get_i18n().set_language(new_lang, save_to_config=False)

        save_config(self.cfg)
        QMessageBox.information(self, "Ajustes", "Configuración guardada correctamente en config.json.")