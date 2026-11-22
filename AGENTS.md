# Reglas de navegación para el agente (AstroLoom)

1. **NO leas archivos completos mayores de 1.000 líneas.** Usa siempre búsqueda de símbolos o rangos de líneas (`read_file` con `StartLine`/`EndLine` o comandos `Select-String`/`grep`) antes de volcar archivos.
2. **NO leas `CHANGELOG.md`, `MANUAL_DE_USUARIO.md` ni `INDEX.md`** salvo que la tarea pida explícitamente redactar o actualizar documentación.
3. **Arquitectura:**
   - `astroloom_rs/`: Núcleo numérico en Rust compilado con PyO3/Maturin.
   - `core/`: Pipelines matemáticos en Python (apilado, calibración, estirado MTF/GHS, inferencia ONNX).
   - `gui/`: Interfaz PySide6 (`main_window.py` es el orquestador principal de ~300 líneas; los tabs son widgets modulares).

## Mapa estructural de archivos grandes en `gui/`

### 1. `gui/tab_developer.py` (~1.680 líneas) - Tab de Revelado (Orquestador Central)
- **L1–364**: Imports, constantes y `DeveloperTab.__init__` (estado, pipeline, timers).
- **L365–1231**: `_setup_ui`, layout principal, docks laterales, footer y paneles modulares.
- **L1232–1295**: `retranslate_ui` (textos y títulos reactivos).
- **L1296–1386**: `_apply_styles` (estilos visuales QSS).
- **L1387–1391**: `log_message` hacia LogHub.
- **L1392–1464**: Callbacks de paneles (`_set_curve_mode`, `_on_effects_parameter_changed`, `_on_stretch_parameter_changed`, `reset_sliders`).
- **L1465–1678**: Slots ligeros de cambio de sliders directos (`on_contrast_starless_changed` ... `on_gnd_edge_shift_changed`).
*(Nota: Hereda de `RenderMixin`, `CanvasToolsMixin`, `AIWorkersMixin`, `ProjectIOMixin`, `HistoryPresetsMixin` y `LayerSyncMixin`)*

### 2. `gui/developer/render_engine.py` (~1.250 líneas) - Motor de Render y Previsualización (`RenderMixin`)
- Proxies multi-escala (`_generate_preview_proxy`, `ResolutionPyramid`) e invalidación de cachés (`_invalidate_pipeline_caches`).
- Pipeline asíncrono y composición (`_process_starless_base`, `_compose_active_base`, `_apply_pipeline_on_image`).
- Actualización de preview y LUT 1D ultrarrápido (`update_stretch_preview`, `_do_update_stretch_preview`, `_is_lut_preview_applicable`).
- Refresco de histogramas (`_refresh_histograms_now`) y zoom/pan 1:1 (`on_canvas_zoom_toggled`, `on_canvas_pan_*`, `_refine_pan_crop`).

### 3. `gui/developer/ai_workers_controller.py` (~1.220 líneas) - Orquestador de IA y Workers (`AIWorkersMixin`)
- Inferencia StarNet++ (`run_starnet`, `on_starnet_success`, `on_starnet_error`, `_start_starnet_model_download`).
- NAFNet Denoise cielo y suelo (`on_run_ai_denoise_requested`, `_run_nafnet_inference`, etc.).
- Extracción de gradientes y fondos (`on_extract_background_clicked`, `on_auto_abe_clicked`, `on_bg_extraction_success`).
- Calibración de color fotométrica PCC (`apply_pcc_calibration`).
- Estimación PSF y Deconvolución (`_get_current_deconv_psf`, `on_auto_detect_psf_clicked`).

### 4. `gui/developer/canvas_tools.py` (~2.290 líneas) - Herramientas de Lienzo (`CanvasToolsMixin`)
- Toolbars flotantes (`_create_canvas_toolbar`, `_create_crop_toolbar`, `_create_mask_toolbar`, `_create_canvas_footer_bar`).
- Recorte y rotación (`_on_toggle_crop`, `_on_crop_*`, `_apply_crop`, `_reset_crop`).
- Pincel y máscaras de suelo/cielo (`_compute_mask_from_scribbles`, `_load_mask_*`, `_apply_loaded_mask`, trazos).
- Pipetas (`_on_toggle_*_pipette`, `_on_eyedropper_color_sampled`), Split View, Zoom 1:1, navegación y `keyPressEvent`.
- Gestos de capas en lienzo (movimiento, escalado, gradientes).

### 5. `gui/developer/layer_sync_bridge.py` (~365 líneas) - Puente de Sincronización de Capas (`LayerSyncMixin`)
- Handlers `_on_mask_widget_*` (estrellas, starless, orton, difusión, suelo).
- Reseteos de estrellas y suelo (`reset_stars_intensity`, `reset_gnd_tone`).
- Detección, visualización y gestión de motas de polvo (`_on_auto_detect_dust`, `_on_canvas_dust_spots_changed`).

### 6. `gui/developer/history_presets.py` (~1.450 líneas) - Historial y Presets (`HistoryPresetsMixin`)
- Pila Undo/Redo (`_capture_state`, `_restore_state`, `undo`, `redo`, `_commit_action`).
- Presets de revelado (`apply_preset`, `_populate_presets_combo`, diálogos de preset).
- Reseteos globales e individuales (`reset_all_parameters`, `clear_session`, `reset_*`).

### 7. `gui/developer/io_project_manager.py` (~890 líneas) - I/O y Proyectos (`ProjectIOMixin`)
- Carga de imágenes astronómicas y stacks (`load_image_direct`, `load_coronal_stack`, `open_image_file`).
- Capas ráster y drag & drop (`_on_import_raster_layer`, `_on_canvas_files_dropped`).
- Exportación multicapa (`export_image` a TIFF 16/32-bit, JPEG y PSD con capas nativas).
- Proyectos `.devloom` (`save_project`, `load_project`, `_do_save_project`).

### 8. `gui/widgets/mask_layers_widget.py` (~4.700 líneas) - Gestor de Capas y Máscaras
- **L1–121**: `LayerListResizeHandle` (frame de redimensionado interactivo).
- **L122–1816**: `MaskLayersWidget.__init__` y `_setup_ui` (árbol de capas, sliders, botones de mezcla y opacidad).
- **L1817–3200**: Conexión de sliders, sincronización de capas base/starnet/ráster y eventos de lista.
- **L3201–4724**: Lógica de combinación, exportación de máscaras y serialización de capas.

### 9. `gui/canvas.py` (~3.450 líneas) - Visor interactivo (`MaskCanvas`)
- **L1–179**: `MaskCanvas.__init__`, configuración de eventos de ratón/teclado y viewport.
- **L180–466**: Modos de pincel, trazos, radio y carga de pixmaps base.
- **L467–850**: Split-view comparativo (cortina antes/después) y visualización de clipping.
- **L851–2100**: Transformaciones de zoom, pan, navegación espacial y pipetas de color.
- **L2101–3453**: Dibujo sobre lienzo (paintEvent), overlays de máscaras y gestores de interacción.
