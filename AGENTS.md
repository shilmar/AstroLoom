# Reglas de navegación para el agente (AstroLoom)

1. **NO leas archivos completos mayores de 1.000 líneas.** Usa siempre búsqueda de símbolos o rangos de líneas (`read_file` con `StartLine`/`EndLine` o comandos `Select-String`/`grep`) antes de volcar archivos.
2. **NO leas `CHANGELOG.md`, `MANUAL_DE_USUARIO.md` ni `INDEX.md`** salvo que la tarea pida explícitamente redactar o actualizar documentación.
3. **Arquitectura:**
   - `astroloom_rs/`: Núcleo numérico en Rust compilado con PyO3/Maturin.
   - `core/`: Pipelines matemáticos en Python (apilado, calibración, estirado MTF/GHS, inferencia ONNX).
   - `gui/`: Interfaz PySide6 (`main_window.py` es el orquestador principal de ~300 líneas; los tabs son widgets modulares).

## Mapa estructural de archivos grandes en `gui/`

### 1. `gui/tab_developer.py` (~8.800 líneas) - Tab de Revelado
- **L1–364**: Imports, constantes y `DeveloperTab.__init__` (estado, pipeline, timers).
- **L365–1520**: `_setup_ui`, layout principal, docks laterales, footer y `_create_canvas_toolbar`.
- **L1521–2030**: Recorte / Crop (`_create_crop_toolbar`, `_on_toggle_crop`, rotaciones y ratios).
- **L2031–2810**: Pincel y máscaras (`_create_mask_toolbar`, modos de pincel, capas y badges).
- **L2811–3550**: Accesos directos, pipetas (WB, GHS, LP), split-view (antes/después), zoom y clipping.
- **L3551–4500**: Carga/importación de imagen, proxy de preview, coronal stack, capas ráster y drag & drop.
- **L4501–6000**: Inferencia IA (StarNet, NAFNet denoise), deconvolución PSF, motas de polvo y curvas.
- **L6001–7500**: Sliders de ajuste, balances, estirado y reseteos globales / individuales.
- **L7501–8220**: Motor de Undo/Redo e historial (`_capture_state`, `_restore_state`, `undo`, `redo`).
- **L8221–8868**: Gestión de presets (`DeveloperPreset`), guardado/carga de sesión y estado dirty.

### 2. `gui/widgets/mask_layers_widget.py` (~4.700 líneas) - Gestor de Capas y Máscaras
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
