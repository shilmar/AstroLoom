# Reglas de navegación para el agente (AstroLoom)

1. **NO leas archivos completos mayores de 1.000 líneas.** Usa siempre búsqueda de símbolos o rangos de líneas (`read_file` con `StartLine`/`EndLine` o comandos `Select-String`/`grep`) antes de volcar archivos.
2. **NO leas `CHANGELOG.md`, `MANUAL_DE_USUARIO.md` ni `INDEX.md`** salvo que la tarea pida explícitamente redactar o actualizar documentación.
3. **Arquitectura:**
   - `astroloom_rs/`: Núcleo numérico en Rust compilado con PyO3/Maturin.
   - `core/`: Pipelines matemáticos en Python (apilado, calibración, estirado MTF/GHS, inferencia ONNX).
   - `gui/`: Interfaz PySide6 (`main_window.py` es el orquestador principal de ~300 líneas; los tabs son widgets modulares).

## Mapa estructural de archivos grandes en `gui/`

### 1. `gui/tab_developer.py` (~4.380 líneas) - Tab de Revelado (Orquestador)
- **L1–364**: Imports, constantes y `DeveloperTab.__init__` (estado, pipeline, timers).
- **L365–1231**: `_setup_ui`, layout principal, docks laterales, footer y paneles.
- **L1232–1295**: `retranslate_ui` (textos y títulos reactivos).
- **L1296–1386**: `_apply_styles` (estilos visuales QSS).
- **L1387–1520**: Proxy de preview e invalidación de cachés.
- **L1521–3000**: Inferencia IA (StarNet, NAFNet denoise), deconvolución PSF, motas de polvo y curvas.
- **L3001–4383**: Sliders de ajuste, balances, estirado y pipeline reactivo.
*(Nota: Extraídos `history_presets.py`, `io_project_manager.py` y `canvas_tools.py`)*

### 2. `gui/developer/canvas_tools.py` (~2.290 líneas) - Herramientas de Lienzo (`CanvasToolsMixin`)
- Toolbars flotantes (`_create_canvas_toolbar`, `_create_crop_toolbar`, `_create_mask_toolbar`, `_create_canvas_footer_bar`).
- Recorte y rotación (`_on_toggle_crop`, `_on_crop_*`, `_apply_crop`, `_reset_crop`).
- Pincel y máscaras de suelo/cielo (`_compute_mask_from_scribbles`, `_load_mask_*`, `_apply_loaded_mask`, trazos).
- Pipetas (`_on_toggle_*_pipette`, `_on_eyedropper_color_sampled`), Split View, Zoom 1:1, navegación y `keyPressEvent`.
- Gestos de capas en lienzo (movimiento, escalado, gradientes).

### 3. `gui/developer/history_presets.py` (~1.450 líneas) - Historial y Presets (`HistoryPresetsMixin`)
- Pila Undo/Redo (`_capture_state`, `_restore_state`, `undo`, `redo`, `_commit_action`).
- Presets de revelado (`apply_preset`, `_populate_presets_combo`, diálogos de preset).
- Reseteos globales e individuales (`reset_all_parameters`, `clear_session`, `reset_*`).

### 4. `gui/developer/io_project_manager.py` (~890 líneas) - I/O y Proyectos (`ProjectIOMixin`)
- Carga de imágenes astronómicas y stacks (`load_image_direct`, `load_coronal_stack`, `open_image_file`).
- Capas ráster y drag & drop (`_on_import_raster_layer`, `_on_canvas_files_dropped`).
- Exportación multicapa (`export_image` a TIFF 16/32-bit, JPEG y PSD con capas nativas).
- Proyectos `.devloom` (`save_project`, `load_project`, `_do_save_project`).

### 5. `gui/widgets/mask_layers_widget.py` (~4.700 líneas) - Gestor de Capas y Máscaras
- **L1–121**: `LayerListResizeHandle` (frame de redimensionado interactivo).
- **L122–1816**: `MaskLayersWidget.__init__` y `_setup_ui` (árbol de capas, sliders, botones de mezcla y opacidad).
- **L1817–3200**: Conexión de sliders, sincronización de capas base/starnet/ráster y eventos de lista.
- **L3201–4724**: Lógica de combinación, exportación de máscaras y serialización de capas.

### 3. `gui/canvas.py` (~3.450 líneas) - Visor interactivo (`MaskCanvas`)
- **L1–179**: `MaskCanvas.__init__`, configuración de eventos de ratón/teclado y viewport.
- **L180–466**: Modos de pincel, trazos, radio y carga de pixmaps base.
- **L467–850**: Split-view comparativo (cortina antes/después) y visualización de clipping.
- **L851–2100**: Transformaciones de zoom, pan, navegación espacial y pipetas de color.
- **L2101–3453**: Dibujo sobre lienzo (paintEvent), overlays de máscaras y gestores de interacción.
