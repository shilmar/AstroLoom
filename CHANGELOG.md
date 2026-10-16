# Changelog

Todas las novedades, mejoras y correcciones notables de **AstroLoom** (anteriormente *Apilator*) se documentan en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Añadido
* **Control Rápido y Ergonomía de Superposición de Máscara de Suelo:**
  - **Botón `👁️ Ver Máscara` en la Barra Superior:** Ubicado directamente en la barra de herramientas del lienzo del Revelador junto a `🧠 Máscara IA` y `🖌️ Máscara Suelo (M)`, permitiendo encender o apagar la superposición coloreada de cielo/suelo en cualquier momento sin tener que abrir la herramienta secundaria de trazado.
  - **Respuesta Visual Dinámica de Alta Visibilidad:** Indicador de estado con cambio dinámico de color: Verde activo (`#2e7d32` / `#81c784`) cuando la máscara se está proyectando en pantalla (`👁️ Máscara Visible`), y tono oscuro/gris cuando está oculta (`👁️ Ver Máscara`), sincronizado en tiempo real con el botón `👁️ Ver` de la barra horizontal.
  - **Ocultación Automática de Trazos al Generar:** Al calcular la máscara guiada por semillas (`✨ Generar`), los trazos de pincel rojo/verde se ocultan automáticamente para mostrar de forma nítida la máscara continua refinada, y el badge de la Sección 8 pasa a indicar el origen exacto (`● Suelo (Trazos)`, `● Suelo (IA)`, etc.).
* **Pipeta Muestreadora Interactiva para Máscaras de Rango de Color (HSV):**
  - **Botón `🎯 Muestrear Color en Imagen` en Gestor de Capas:** Disponible en cualquier capa de ajuste de tipo Color (`MASK_TYPE_COLOR`).
  - **Muestreo Cromático Directo en Lienzo:** Permite hacer clic sobre cualquier zona visible de la toma (nebulosa H-Alfa, OIII, contaminación de vapor de sodio, airglow, etc.) para calcular de forma instantánea el Matiz (Hue 0°–360°) y la saturación mínima a partir de los píxeles reales mostrados.
  - **Activación Inmediata de Previsualización:** Al muestrear el color, se activa de inmediato el modo de previsualización en rubí translúcido para evaluar visualmente las áreas capturadas por la máscara.
* **Filtro Difusor Digital y Realce de Constelaciones (Star Mist Filter) [P1]:**
  - **Aislamiento Selectivo y Morfológico de Estrellas (`core/star_diffusion.py`):** Detección selectiva de picos estelares prominentes (magnitudes 1ª a 4ª) mediante análisis Top-Hat en imágenes compuestas o acoplamiento directo sobre la capa de estrellas desacoplada de StarNet++.
  - **Protección de Estrellas Débiles de Fondo:** Las millones de estrellas tenues del fondo cósmico quedan protegidas mediante una curva de umbralización suave Hermite (*smoothstep*), preservando su puntualidad y nitidez sin inflarse ni generar niebla difusa en el cielo oscuro.
  - **Recuperación Cromática Espectral (Spectral Color Recovery):** Extracción del matiz (*hue*) y saturación cromática real desde la periferia no saturada del disco de Airy estelar (clases espectrales O, B, A, F, G, K, M), infundiendo halos ricos y coloridos (el naranja cálido de Betelgeuse/Antares, el azul de Rigel/Vega).
  - **Dispersión Óptica Multi-Escala:** Modelo analítico de doble dispersión gaussiana continua con caída natural de luz difusa y semilla elíptica de micro-difracción.
  - **Modos de Fusión No Destructivos:** Soporte para *Trama (Screen - Velo Etéreo)*, *Luz Suave (Soft Light - Aterciopelado)* y *Aditivo (Additive - Luminoso)*.
  - **Protección de Paisaje Terrestre:** Integración con la máscara de cielo para anular la propagación de halos hacia el relieve, árboles o arquitectura.
  - **Integración UI Contextual y Exportación:**
    - Panel dedicado en la capa **⭐ Estrellas** del Gestor de Capas y en el revelador con controles de *Intensidad*, *Radio*, *Umbral*, *Color Espectral* y *Modo de Fusión*.
    - Integración en el pipeline principal, previsualización en vivo, histórico Deshacer/Rehacer (Ctrl+Z / Ctrl+Y) y exportación de capas nativas en Photoshop PSD.
* **Manual de Usuario Completo (`MANUAL_DE_USUARIO.md`):**
  - Documento exhaustivo en español explicando de manera accesible y visual cada módulo de AstroLoom (Apilador, Revelador, Startrails, Eclipses, Gestor de Capas).
  - Guía detallada slider por slider de todos los controles del módulo Revelador (estirado MTF/GHS, curvas, StarNet, texturas de polvo, Orton, polución lumínica, balance de blancos, máscaras IA, exportación PSD).
* **Efecto Orton Astronómico y Glow de Polvo Galáctico [P6]:**
  - Resplandor etéreo y suave enfocado selectivamente en nubes moleculares y bandas galácticas sin oscurecer estrellas.
* **Reducción Paramétrica 2D de Contaminación Lumínica Terrestre (Domo Asimétrico) [P5]:**
  - Sustracción de cúpulas lumínicas urbanas en el horizonte mediante modelo elíptico 2D con ajuste de centro, anchura, altura y tinte de vapor de sodio.
* **Estirado Hiperbólico Generalizado (GHS) [P2]:**
  - Transformación hiperbólica con punto de simetría (SP), factor de estirado (D) y protección de núcleos estelares (HP) con modos de Luminancia y RGB enlazado.

### Añadido
* **Detección Automática de Suelo y Horizonte con IA Ligera (1-Click Sky/Ground) [P8]:**
  - **Inferencia Local con Red Neuronal Profunda (`core/ai_horizon.py`):** Integración del modelo U²-Net en ONNX Runtime (`skyseg.onnx`) para segmentación semántica automática de cielo y suelo en menos de 1 segundo en CPU.
  - **Refinado Subpíxel mediante Filtro Guiado (Guided Filter):** Algoritmo de acoplamiento de bordes de alta fidelidad que proyecta la máscara probabilística sobre la luminancia nativa de la toma, preservando ramas finas, hojas, postes, antenas y contornos de rocas sin halos ni transiciones duras.
  - **Descarga Asistida e Interactiva con Diálogo Modal:** Si el modelo `.onnx` no está presente en la carpeta `models/`, la aplicación ofrece descargarlo automáticamente desde Hugging Face con barra de progreso porcentual y lectura en MB (`X.X / 167.8 MB`), asegurando descargas atómicas libres de archivos corruptos (`.tmp`).
  - **Motor de Resiliencia / Fallback Algorítmico Matemático:** Fallback instantáneo (0.01 s) basado en supresión morfológica estelar y gradiente vertical Otsu adaptativo, garantizando funcionalidad completa incluso sin conexión a internet en el campo.
  - **Integración UI en Apilador y Revelador:**
    - Botón destacado en Sección 5 del Apilador: `[ 🧠 Auto-Detectar Suelo (IA 1-Click) ]`.
    - Opción integrada en la barra horizontal del Revelador: `Cargar Máscara... -> 🧠 Auto-Detectar Suelo con IA (1-Click)`.
    - Totalmente compatible con retoque manual por pinceles (`[ ✏️ Refinar con Trazos ]`).
* **Tratamiento Anti-Ghosting de Árboles en Horizonte (Min-Horizon Boundary) [P3]:**
  - **Erradicación de Árboles Fantasma (`core/stacking.py`):** Algoritmo de proyección percentil inferior circunscrito exclusivamente a una banda de seguridad perimetral dilatada alrededor del contorno de la máscara de suelo.
  - Evita la fuga de estrellas en rotación a través de las ramas y el follaje estático durante el apilado dual (cielo móvil + suelo fijo).
  - **Controles Paramétricos en el Apilador:** Casilla activable con radio de frontera configurable (`spin_ag_radius`, 5 a 50 px) y selector de sensibilidad de follaje (*Conservadora*, *Equilibrada*, *Agresiva*).
* **Cribado Previo Inteligente de Tomas y Calidad Estelar (Frame Culling) [P4]:**
  - **Inspección Óptica Automatizada en Paralelo (`core/frame_culling.py`):** Análisis previo multihilo que mide FWHM (nitidez de foco), excentricidad estelar (alargamiento por viento o deriva) y censo de estrellas (paso de nubes o niebla).
  - **Calibración Estratificada para Grandes Angulares:** Discriminación matemática entre el centro óptico (eje principal de la lente) y la periferia/esquinas (afectadas naturalmente por aberración de coma y astigmatismo), evitando descartes falsos en focales de 14mm-24mm.
  - **Perfiles de Tolerancia Adaptativos:** Selector en la barra de herramientas con modos *Gran Angular (Tolerante)*, *Equilibrado (Recomendado)* y *Exigente (Foco Crítico)*.
  - **Badges de Puntuación Visual en Vivo:** Puntuación porcentual alineada en columna para cada archivo (`🟢 96%`, `🟡 82%`, `🔴 32% Movida` / `🔴 15% Nubes`) con ficha técnica en tooltip y botón de auto-descarte instantáneo (`⚡ Auto-Descartar`).

### Corregido
* **Alineación 1:1 de Capas y Corrección de Residuo Estelar en Zoom del Revelador:**
  - Corrección de distorsión y desplazamiento relativo en la capa de estrellas y suelo al activar el zoom al 100% en `gui/tab_developer.py` mediante recorte síncrono de coordenadas normalizadas (`crop_rect`) en `core/mask_layers.py`.
* **Ergonomía de la Barra de Herramientas de Cribado en el Apilador:**
  - Reorganización en dos filas compactas y balanceadas que erradica el scrollbar horizontal forzado en el panel izquierdo.
* **Restauración de Visibilidad y Alineación de Puntuaciones de Tomas:**
  - Corrección de visibilidad de `lbl_quality` en `FileListRow.set_quality_result()` y alineación directa tras el checkbox para un listado limpio y legible.

### Modificado
* **Salto de Versión:**
  - Actualización a **v0.8.3** en `gui/main_window.py`, `gui/tab_settings.py`, `locales/es.json` y `locales/en.json`.
  - Inclusión de dependencia `onnxruntime>=1.16.0` en `requirements.txt`.

## [0.8.2] - 2026-10-15

### Añadido
* **Gestor Avanzado de Capas y Máscaras de Ajuste Local (Estilo Photoshop / Lightroom):**
  - **Capas Especiales de Imagen:** Integración de capas dedicadas para *Fondo Base*, *Fondo Starless (Nebular)*, *Estrellas* y *Suelo Terrestre*.
  - **Orden Natural de la Pila (Bottom-Up):** Las capas y máscaras de ajuste afectan al contenido subyacente según su posición física en la lista, eliminando la ambigüedad del antiguo selector de ámbito.
  - **Desacoplamiento Estricto de Parámetros:** La intensidad y densidad de estrellas (StarNet) opera como un control estelar independiente sin alterar la opacidad de la capa ni viceversa.
  - **Paneles de Ajuste Contextuales Dedicados:**
    - Al seleccionar la capa **Suelo**, se despliegan exclusivamente sus controles tonales (sombras, exposición, contraste, punto negro).
    - Al seleccionar la capa **Estrellas**, se muestran los controles dedicados de reducción, presencia estelar y realce de color.
    - Al seleccionar la capa **Fondo Starless**, se muestran los controles de claridad, dehaze y realce nebular.
    - Para capas estándar (Pincel, Rango Luma/Color), se activan los controles locales de exposición, contraste, altas luces, sombras, saturación y balance de blancos local.
  - **Identidad Cromática Temática:** Títulos de cabecera y bordes perimetrales con colores distintivos según el tipo de capa (Verde Esmeralda para Suelo, Púrpura Cósmico para Starless, Oro Estelar para Estrellas, Rojo Carmesí para Pincel, Amarillo para Luma, Cian para Color HSV y Ámbar para Fondo Base).
  - **Visibilidad Interactiva y Memoria de Herramienta:**
    - Icono de ojo interactivo (`👁️`) en cada fila de capa para alternar la visibilidad de forma inmediata.
    - Memoria persistente de radio y opacidad/flujo del pincel durante la sesión.
    - Supresión de auto-activación del overlay rojo de máscara al seleccionar o alternar capas de pincel.
  - **Carga Directa de Máscaras en Barra Horizontal:**
    - Incorporado el botón **`📂 Cargar Máscara...`** con detección inteligente en memoria de la segmentación cielo/suelo generada en el Apilador o Circumpolares, y selector de archivos en disco (PNG 16-bit, TIFF, FITS) con remuestreo automático a la resolución del lienzo.
  - **Integración Fluida con StarNet++ AI:**
    - Auto-despliegue de la Sección 9 y scroll automático garantizando visibilidad inmediata de las capas separadas al iniciar y al completar la inferencia con IA.
    - Reorganización limpia de las Secciones 5 y 8 mediante tarjetas explicativas y botones de salto directo a la Sección 9.
* **Motor de Internacionalización (i18n) Multilingüe en Tiempo Real:**
  - Motor modular basado en diccionarios JSON en `locales/` (`es.json`, `en.json`) gestionado por `core/i18n.py`.
  - Selector de idioma en tiempo real en la barra de estado inferior (`🌐 ES / EN`) con persistencia en `config.json` y traducción reactiva de toda la interfaz.
* **Rediseño del Revelador con Acordeón Plegable Inteligente:**
  - 9 secciones colapsables e interactivas (`CollapsibleSection`) que optimizan el espacio vertical y la ergonomía de trabajo.
* **Sistema de Diseño Global "Cosmic Dark Pro" y Unificación Visual Integral:**
  - Creación del subsistema centralizado de diseño en [`gui/theme.py`](file:///D:/software/AstroLoom/gui/theme.py) con paleta de colores coherente y contrastada para astrofotografía:
    - Fondos escalonados: Fondo ventana (`#0b0d14`), Paneles laterales (`#12141e`), Tarjetas y Grupos (`#171a26`), Controles de entrada (`#0d0f17`), Consola de logs (`#08090f`).
    - Bordes refinados con estados interactivos: Base (`#232738`), Hover (`#3a415a`), Foco (`#00bcd4`), Activo (`#00e5ff`).
    - Jerarquía tipográfica consistente y legible (`#eceff4` primario, `#90a4ae` secundario, `#546e7a` atenuado).
    - Botones de acción unificados con retroalimentación visual táctil (`STYLE_RUN_IDLE`, `STYLE_RUN_ACTIVE`, `STYLE_RUN_CANCELLING`, `STYLE_TRANSFER_BTN`).
  - Unificación visual y eliminación de estilos fragmentados en los 7 módulos del sistema (Apilador, Revelador, Eclipses HDR, Eclipses Collage, Startrails, Panorámicas y Ajustes).
  - Navegación superior moderna (`QTabBar`) con resaltado de pestaña activa mediante barra inferior cian brillante (`#00e5ff`).
* **Barra de Estado Interactiva y Monitor de Hardware en Tiempo Real:**
  - Indicador permanente del motor y versión activa (`⚡ AstroLoom Engine v0.8.2`).
  - Monitor dinámico de memoria RAM del sistema con lectura porcentual y en gigabytes (`🧠 RAM: XX% (XX.X / XX.X GB)`) actualizado cada 2.5 segundos mediante `psutil`.
* **Gestor de Presets y Estilos de Revelado Astronómico (Developer Presets Manager):**
  - **Subsistema Paramétrico Centralizado ([`core/developer_presets.py`](file:///D:/software/AstroLoom/core/developer_presets.py)):**
    - Modelo de datos serializable en JSON con soporte de metadatos y 8 presets de fábrica calibrados para astrofotografía.
  - **Protección del Estirado Tonal Nativo (Histograma Protegido):**
    - El estirado MTF y el Punto Negro (`bp_val` y `mtf_val`) quedan estrictamente protegidos para preservar la calibración lumínica única del sensor.
  - **Aplicación Granular / Selectiva por Ámbitos:** Modos Todo, Solo Color/WB, Solo Estrellas/Filtros y Solo Contraste/Suelo.
  - **Diálogos de Gestión e Interfaz ([`gui/dialog_presets.py`](file:///D:/software/AstroLoom/gui/dialog_presets.py)):** Panel en cabecera del Revelador, diálogo para guardar presets y administrador completo con búsqueda y categorías.

### Modificado
* **Actualización de Versión:**
  - Salto de versión a **v0.8.2** en `gui/main_window.py`, `gui/tab_settings.py` y diccionarios de internacionalización (`locales/es.json` y `locales/en.json`).

## [0.8.1] - 2026-10-15

### Añadido
* **Herramienta Interactiva de Recorte y Nivelación de Horizonte (Crop & Leveling Tool):**
  - Incorporada en el **Revelador / Editor** una herramienta de recorte profesional estilo Lightroom / Camera Raw accesible con el botón de la barra superior `[ ✂️ Recortar (C) ]` o el atajo global de teclado `C`.
  - **Lienzo Interactivo con 8 Tiradores y Guías de Composición:**
    - Tiradores en las 4 esquinas (soportes en ángulo tipo "L") y 4 barras centrales en los bordes con cursores contextuales dinámicos.
    - Desplazamiento del marco completo arrastrando su interior o creación de nuevos encuadres haciendo clic y arrastrando fuera.
    - Superposición de zona descartada oscurecida con máscara semitransparente de alto contraste y borde perimetral dual.
    - Guías de composición integradas: *Regla de Tercios* y *Cuadrícula 6×6* con líneas semitransparentes.
    - Píldora flotante en el lienzo con lectura en tiempo real de la resolución resultante en píxeles nativos del sensor (`px_w × px_h`) y ángulo de inclinación.
  - **Relaciones de Aspecto Fijas y Asistente de Proporciones:**
    - Presets integrados: *Libre*, *Original*, *1:1 (Cuadrado)*, *16:9 (Panorámico)*, *3:2 (Fotografía)*, *4:3 (Estándar)*, *5:4 (Gran Formato)*, *2:1 (Panorámico)*, *9:16 (Vertical)*, *2:3 (Vertical 35mm)* y *4:5 (Vertical IG)*.
    - Botón `[ ⇄ Invertir (X) ]` (y atajo de teclado `X`) para alternar instantáneamente la orientación horizontal/vertical de la proporción elegida o del marco libre.
  - **Nivelación de Horizonte en Tiempo Real:**
    - Control deslizante de rotación de $-45.0^\circ$ a $+45.0^\circ$ (en pasos de $0.1^\circ$) con botón de restablecimiento rápido a `0°`.
    - Rotación en vivo en el lienzo en tiempo real a 60 FPS mediante transformaciones GPU/QPainter con la retícula de recorte horizontal fija para alinear cordilleras, horizontes marinos o el arco galáctico.
  - **Corte y Transformación a 32-bit de Alta Precisión:**
    - Rotación y recorte en espacio nativo de 32-bit float (`cv2.INTER_LANCZOS4` y `BORDER_REFLECT`) para evitar degradación de señal, pérdidas de detalle o marcos negros en bordes.
    - Si existen capas separadas de StarNet (`image_starless`, `image_stars`) o máscaras activas (`current_mask`), el recorte y rotación se aplican con congruencia milimétrica idéntica sobre todas ellas para preservar la alineación subpíxel.
  - **Integración No Destructiva con Deshacer / Rehacer (Undo / Redo):**
    - Aplicación de recorte mediante botón `[ ✅ Aplicar Recorte ]`, tecla `Intro / Return` o doble clic directo dentro del marco de recorte.
    - Cancelación sin cambios con `[ ❌ Cancelar ]` o tecla `Escape`.
    - Integrado con la pila de 50 pasos de historial (`Ctrl+Z` / `Ctrl+Y`), permitiendo restaurar la toma sin recortar en cualquier momento sin pérdida de datos.
* **Ampliación del Rango de Balance de Blancos (Temperatura y Tinte):**
  - Rango cuadruplicado de $\pm 0.250$ a $\pm 1.000$ (en 2000 pasos de $0.001$) en `slider_temp` y `slider_tint`.
  - Cuadros de entrada numérica directa (`QDoubleSpinBox`) con 3 decimales (`spin_temp` y `spin_tint`) para introducir valores exactos con el teclado o variar con las flechas paso a paso, sincronizados bidireccionalmente con los deslizadores y el historial de Deshacer/Rehacer (`editingFinished`).
  - Capacidad analítica para neutralizar dominantes cromáticas severas de polución lumínica (lámparas de vapor de sodio a baja/alta presión) o intensas emisiones de *airglow* verde/rojo en cielos oscuros.
  - Soporte de doble clic en las etiquetas de título para restablecimiento instantáneo a neutro (`0.000`).
  - Sincronización completa con la Pipeta de Balance de Blancos (`W`) adaptada al nuevo rango ensanchado.

### Modificado
* **Actualización de Versión:**
  - Salto de versión a **v0.8.1** en `gui/main_window.py` y en la tarjeta de versión de `gui/tab_settings.py`.
* **Hoja de Ruta (`README.md`):**
  - Incorporadas nuevas metas prioritarias: Rediseño y Unificación Visual (`gui/theme.py`), Histograma RGB en vivo, Sistema de Preajustes (.astropreset), Máscaras de Luminancia y Rango de Color, Anti-parpadeo (Deflicker) y Monitor de RAM/GPU en barra de estado.

## [0.8.0] - 2026-10-14

### Añadido
* **Evolución y Renombrado Oficial a AstroLoom:**
  - El proyecto evoluciona formalmente de *Apilator* a **AstroLoom** para reflejar su condición de suite completa y estación de trabajo integral de astrofotografía y paisaje nocturno (apilado con calibración rigurosa, revelador lineal no destructivo a 32-bit con IA, mosaicos panorámicos, circumpolares y secuencias de eclipses).
* **Nuevo Módulo Dedicado de Composición Panorámica y Mosaicos Nocturnos (Nightscape Stitcher):**
  - Incorporada la nueva pestaña de primer nivel **"6. Panorámicas (Mosaicos)"** en la ventana principal (`gui/tab_panorama.py` y `core/panorama.py`).
  - Desplazamiento armónico de la pestaña de Configuración a posición 7 (*1. Apilador*, *2. Revelador*, *3. Eclipses HDR*, *4. Eclipses Collage*, *5. Trazas de Estrellas*, *6. Panorámicas*, *7. Configuración*).
* **Soporte Nativo Multifila (Multi-Row 2D):**
  - **Modo Automático 2D por Grafo de Conectividad:** Descubrimiento autónomo de solapamientos 2D tanto horizontales como verticales mediante emparejamiento bidireccional SIFT y estimación robusta de homografías RANSAC.
  - **Asistente de Cuadrícula Regular $M \times N$:** Soporte opcional para definir filas y columnas con asignación de teselas para secuencias guiadas con rótula graduada.
* **Algoritmos Avanzados Especializados para Paisaje Nocturno:**
  - **Pre-realce Astro:** Aplicación de CLAHE y ecualización adaptativa sobre una copia en escala de grises para detectar miles de puntos clave fiables en terrenos en sombra profunda y estrellas débiles.
  - **Proyecciones Cartográficas:** Cilíndrica (para grandes panoramas horizontales), Esférica (para coberturas de 360° y gran angular) y Plana / Rectilínea (para mosaicos con teleobjetivo).
  - **Alineación del Eje de Trípode Normalizada:** Estimación global del vector normal al plano de barrido óptico entre tomas adyacentes para alinearlo con el eje vertical $Y$ del warper esférico/cilíndrico, eliminando singularidades polares, deformaciones en arco y ondulaciones en barridos verticales (portrait) y centrando el panel medio en el meridiano cero.
  - **Fusión Híbrida Dual-Zone Tree-Aware Blending (Cielo Continuo + Suelo Óptimo DpSeamFinder):**
    - Erradicación del 100% de árboles fantasma (*ghost trees*) y sombras residuales en el terreno causados por paralaje entre elementos cercanos (árboles a 20-50m) y estrellas en el infinito ($d=\infty$) en rótulas no nodales.
    - Dominio del cielo integrado mediante mapa de distancias cuártico ($dist^4$) para transiciones suaves y continuas de gradientes, estrellas y airglow.
    - Dominio terrestre detectado morfológicamente mediante conectividad a la base y cosido mediante corte óptimo (`cv2.detail_DpSeamFinder` o máxima proximidad al centro como fallback), conservando la nitidez de ramas y copas sin dobles siluetas ni promedios traslúcidos.
    - Transición suave (*feathering*) entre ambas zonas con dilatación y desenfoque gaussiano.
  - **Compensación Óptica de Viñeteo y Ganancia:** Modelo radial fotométrico cuadrático que homogeneiza la caída de luz en los bordes de cada panel antes del alabeo, homogeneizando el fondo del cielo estrellado.
  - **Fusión Multibanda Laplaciana (Laplacian Multi-Band Blending):** Pirámide multiescala configurable (2 a 7 niveles) con mapas de distancia ponderada, asegurando transiciones continuas sin saltos en el cielo nocturno y preservando estrellas puntuales nítidas.
  - **Fusión en Streaming $O(1)$ con Warper 3D:** Procesamiento panel a panel de bajo consumo de memoria RAM ($O(1)$) con recolección de basura periódica (`gc.collect()`), posibilitando la composición de panorámicas de decenas de tomas en alta resolución sin agotar la memoria.
  - **Auto-Recorte de Bordes Irregulares con Margen de Erosión Morfológica:** Algoritmo $O(H \times W)$ de Rectángulo Máximo Inscrito con erosión morfológica perimetral de seguridad para eliminar al 100% los bordes negros irregulares y muescas de curvatura esférica.
* **Lienzo Interactivo de Alta Precisión y Exportación Completa:**
  - Lienzo acelerado con zoom dinámico suave centrado en cursor, paneo por arrastre, botón 100% y conmutador `[ ✂️ Ver Recorte ]`.
  - Exportación en TIFF de 16-bit y 32-bit float con perfiles ICC incrustados y botón directo `[ ➡️ Enviar Panorámica al Revelador / Editor ]`.
  - Guardado y carga de sesiones de proyecto panorámicas (`.panosess`).

### Modificado
* **Actualización de Versión:**
  - Salto de versión a **v0.8.0** en `gui/main_window.py` y en la tarjeta de versión de `gui/tab_settings.py`.
* **Hoja de Ruta (`README.md`):**
  - Marcado como completado: `[x] Módulo de composición panorámica para mosaicos nocturnos (Nightscape Panorama Stitcher)`.
  - Incorporadas nuevas tareas pendientes: Herramienta interactiva de Recorte (Crop) en Revelador/Editor y ampliación del rango de ajuste de Balance de Blancos (Temperatura y Tinte).

## [0.7.1] - 2026-10-14

### Añadido
* **Detección Automática de Fenómenos de Contacto (Anillo de Diamante y Perlas de Baily):**
  - Motor de análisis fotométrico angular de 360° en torno al limbo lunar (`classify_contact_phenomenon` en `core/eclipse_collage.py`).
  - Discrimina automáticamente y clasifica:
    - 💎 **Anillo de Diamante (Diamond Ring)** en C2 (ingreso) o C3 (egreso): destello fotosférico ultrabrillante concentrado en un arco angular estrecho ($\le 65^\circ$).
    - 📿 **Perlas de Baily (Baily's Beads)** en C2 o C3: múltiples cuentas luminosas discontinuas que emergen a través de los valles del relieve lunar.
    - **Totalidad plena** y **Fases parciales** con cálculo de porcentaje de ocultación.
  - Nueva columna interactiva **"Fase Detectada"** en la tabla de cribado con tipografía cromática y badges destacados (celeste, oro, púrpura).
* **Recomendación Inteligente de Contactos y Fases Clave:**
  - Enriquecimiento del recomendador (`recommend_key_frames`) con slots dedicados y búsqueda prioritaria de los mejores fenómenos de contacto en C2 y C3.
  - Ponderación fotométrica que prioriza exposiciones balanceadas donde la corona y el limbo son visibles conjuntamente (0.003s a 0.05s) frente a subexposiciones extremas, descartando tomas trepidadas.
* **Selector Nativo de Color de Fondo de Lienzo en la GUI:**
  - Nuevo control en la Sección 3 de maquetación (`gui/tab_eclipse_collage.py`) con selector de presets (*Negro Espacio*, *Rojo Realce*, *Gris Carbón*, *Blanco Editorial*) y botón para paleta personalizada (`QColorDialog`).
  - Marcas horarias EXIF con cálculo de luminancia y contraste adaptativo automático (blanco sobre fondos oscuros, carbón sobre fondos claros).
  - Integración completa del color en previsualización, exportación plana TIFF/JPG, capa de fondo en PSD de 16-bit para Photoshop y sesiones `.ecollage`.

### Modificado
* **Unificación Proporcional de Tiles y Medallones Circulares:**
  - Eliminado el multiplicador oculto `* 1.45` en `crop_margin_factor`, garantizando que con `Escala Totalidad / Centro = 1.00x (+0%)` todos los medallones circulares mantengan un diámetro idéntico a lo largo de toda la línea de tiempo sin saltos visuales.
  - Escalado armónico y predecible de las fases centrales cuando el usuario incrementa la escala de totalidad.
* **Actualización de Versión:**
  - Salto de versión a **v0.7.1** en la ventana principal (`gui/main_window.py`) y en la insignia de versión del panel de configuración (`gui/tab_settings.py`).
* **Actualización de Hoja de Ruta (`README.md`):**
  - Marcado como completado: `[x] Detección y recomendación automática de fenómenos singulares de contacto: Perlas de Baily y Anillo de Diamante (Diamond Ring)`.

### Corregido
* **Robustez de Centrado frente a Destellos Potentes:**
  - Inmunidad de centrado subpíxel (`optimize_center_circular_flux` en `core/eclipse.py`) frente a resplandores potentes del anillo de diamante que desplazaban el centro hacia la fotosfera sobreexpuesta.
  - Incorporación de anclaje a trayectoria temporal (`expected_center`) en la segunda pasada de refinamiento.
  - Eliminación definitiva de excepciones por dimensiones negativas en recortes (`negative dimensions are not allowed`) en archivos RAW.

## [0.7.0] - 2026-10-14


### Añadido
* **Pestaña Dedicada "4. Eclipses (Collage)" en la Ventana Principal:**
  - Promoción del Diseñador de Secuencia y Collage de Eclipses a pestaña de primer nivel (`EclipseCollageTab` en `gui/tab_eclipse_collage.py`), separando con total claridad el flujo de trabajo de composición temporal del flujo de procesado HDR de corona/limbo.
  - Reorganización armónica de la barra principal en 6 pestañas nativas: *1. Apilador*, *2. Revelador*, *3. Eclipses (HDR)*, *4. Eclipses (Collage)*, *5. Trazas de Estrellas* y *6. Configuración*.
  - Enlace bidireccional entre módulos: botón `[ 🎨 Ir a Pestaña 4: Collage de Eclipses ➡️ ]` en la pestaña de Eclipses HDR para transferir la serie de fotos en un clic, y botón `[ ➡️ Enviar Collage al Revelador / Editor ]` en la pestaña de Collage para continuar el ajuste tonal en el Revelador.
  - Compatibilidad total hacia atrás mantenida en `gui/dialog_eclipse_collage.py`.

### Modificado
* **Actualización de Versión:**
  - Salto de versión a **v0.7.0** en la ventana principal (`gui/main_window.py`) y en la insignia de versión del panel de configuración (`gui/tab_settings.py`), consolidando la nueva arquitectura de 6 pestañas.

## [0.6.9] - 2026-10-14

### Añadido
* **Historial No Destructivo Deshacer / Rehacer (Undo / Redo) en el Revelador:**
  - Sistema de captura paramétrica ultraligero con capacidad de hasta **50 pasos** (`MAX_UNDO_STEPS = 50`).
  - Almacena diccionarios con el estado numérico exacto de los 19 deslizadores, spinboxes de punto negro y MTF, capa activa y nodos de control de las curvas tonales (`CurveWidget`), consumiendo menos de 1 KB por paso (< 50 KB en total para los 50 pasos frente a decenas de GB si se duplicasen las matrices de imagen).
  - Nuevos botones dedicados en la barra superior de herramientas del lienzo: `[ ↺ Deshacer ]` y `[ ↻ Rehacer ]`, con gestión dinámica de activación/desactivación visual (`:disabled`) acorde al tema oscuro.
  - Atajos de teclado universales: **`Ctrl + Z`** para deshacer, y **`Ctrl + Y`** o **`Ctrl + Shift + Z`** para rehacer.
  - Temporizador de desparasitado inteligente (*debounce* de 500 ms) para desplazamientos continuos de la rueda del ratón o teclas de flecha, evitando saturar la pila de historial.
  - Señal dedicada `curveCommitted` emitida al soltar nodos de control o restablecer curvas tonales.
  - Restauración atómica con bloqueo temporal de señales (`blockSignals(True)`) para prevenir disparos en cascada y actualizar la previsualización en una sola pasada.
* **Plantilla de Configuración Base Universal (`config.default.json`):**
  - Incorporado fichero base con rutas relativas limpias y seguras para primeras instalaciones o despliegues portables.

### Modificado
* **Conversión Cromática Rigurosa en Exportación TIFF con Perfiles ICC:**
  - Implementado motor de transformación de coordenadas de color (`convert_colorspace`) con adaptación cromática lineal von Kries / Bradford (CIE D65 a CIE D50) para sRGB, Adobe RGB (1998) y ProPhoto RGB (ROMM RGB).
  - Linearización exacta y re-gamma para cada espacio de color estándar, acelerado mediante tablas de consulta (LUT) vectorizadas de 16 y 8 bits.
  - Integrado en `save_tiff_with_icc` tanto en el Revelador como en el Apilador, garantizando que los datos de píxeles coincidan exactamente con el perfil ICC embebido (Tag 34675), eliminando la desaturación o virajes tonales al abrir las imágenes en Photoshop, Lightroom o visores calibrados.
* **Actualización de Versión:**
  - Incremento general a **v0.6.9** en la ventana principal (`gui/main_window.py`) y en la insignia de versión del panel de configuración (`gui/tab_settings.py`).

### Corregido
* **Arranque Robusto y Auto-reparación de Rutas de Espacio de Trabajo:**
  - Prevenido fallo crítico en el arranque (`FileNotFoundError: [WinError 3] El sistema no puede encontrar la ruta especificada`) al cargar configuraciones con rutas o unidades de disco no existentes en el equipo local.
  - Implementada función de validación y resolución automática (`_validate_and_resolve_dir`) con respaldo transparente hacia rutas locales válidas del usuario (`APP_ROOT` y `~/Apilator`).
* **Depuración Visual en el Módulo de Trazas de Estrellas (Startrails):**
  - Retirada la tarjeta de funciones en desarrollo de la interfaz de usuario en `gui/tab_startrails.py` para mantener un diseño limpio, profesional y consistente con el resto de módulos.

## [0.6.8] - 2026-10-13

### Añadido
* **Aceleración Multihilo y Modo Rápido en el Diseñador de Collages de Eclipses:**
  - Paralelización completa del cribado y análisis en lote (`batch_analyze_eclipse_series`) mediante `ThreadPoolExecutor`, liberando el GIL de Python y aprovechando todos los núcleos de CPU del equipo (hasta 16 hilos en CPUs de 16 núcleos).
  - Nuevo selector de rendimiento en la interfaz: `⚡ Modo Rápido (Half-size RAW 4x)` (binning 2×2 directo en hardware con LibRaw, acelerando el cribado hasta 8.7×) vs `🎯 Modo Precisión 1:1 (Full RAW)`.
  - Reescalado geométrico subpíxel automático (`×2.0`) en centros y radios, garantizando precisión absoluta al recortar tiles sobre las tomas en alta resolución.
  - Selector numérico de hilos concurrentes (`QSpinBox`) configurable por el usuario y botón interactivo de cancelación (`⏹️ Cancelar Análisis`).
* **Generación Acelerada de Composición y Sistema de Caché de Tiles:**
  - Paralelización multihilo de `compute_collage_layout` para la extracción afín (Lanczos4) y desvanecimiento suave de cada fase.
  - Modo `fast_preview` para renderizado interactivo de la composición en pantalla en menos de 0.70 s (11× más rápido que la versión secuencial de 7.33 s).
  - Sistema de caché en memoria (`_tile_cache`): remaquetado instantáneo a 60 fps (en **0.07 ms**) al cambiar parámetros espaciales (espaciado / gap, marcas horarias o modalidad de diseño) sin accesos redundantes a disco.
* **Persistencia Integral de Sesiones de Collage (`.ecollage`):**
  - Nuevas opciones `[ 📂 Abrir Sesión... ]` y `[ 💾 Guardar Sesión... ]` con serialización estructurada JSON (`.ecollage`).
  - Almacena y recupera instantáneamente (< 1 s) todos los metadatos de análisis (centros, radios, nitidez, trepidación, ocultación, fase), selecciones de fotogramas y ajustes finos de composición (layout, espaciado, escala de totalidad, margen, marcas EXIF, modo rápido y número de hilos).
* **Visor Interactivo con Zoom y Paneo (`CollageInteractiveCanvas`):**
  - Navegación fluida en el lienzo de previsualización con zoom dinámico centrado en la posición del ratón ($0.05\times$ a $8.0\times$) y paneo interactivo por arrastre.
  - Barra de herramientas con botones de zoom (`🔍 -`, `100%`, `🔍 +`, `1:1` nativo y `⛶ Ajustar` al lienzo / doble clic).
* **Nueva Modalidad de Maquetación "◇ Escala Romboide / Progresiva (`<>`)":**
  - Distribución espacial armónica basada en un taper cosenoidal suave: fase central ampliada (totalidad/pico, ej. 1.40×) y reducción gradual hacia los extremos ($0.60\times$ a $0.80\times$), todos perfectamente anclados al eje de simetría horizontal.
* **Capa Independiente de Marcas Horarias en PSD de 16 bits:**
  - Exportación con `pytoshop` que añade una capa transparente no destructiva `"Marcas Horarias EXIF"` con tipografía suavizada.

### Modificado
* **Margen de Recorte Adaptativo y Feathering Suave:**
  - Sustituido el recorte ceñido por un margen adaptativo suave con desvanecimiento circular que preserva íntegramente los rayos de difracción, destellos del anillo de diamante y filamentos de la corona solar en fases profundas ($\ge 85\%$).
* **Controles Numéricos y Desacoplamiento de Deslizadores:**
  - Incorporadas etiquetas numéricas de lectura directa en tiempo real (`px` y `%`) y tooltips explicativos en los deslizadores.
  - Desacoplado el recálculo continuo durante el arrastre, incorporando el botón explícito `[ 🔄 Actualizar Composición / Collage ]` con cursor de espera interactivo (`WaitCursor`).
* **Desbloqueo de Núcleos en la Configuración Global:**
  - `core/config_manager.py` y `gui/tab_settings.py` ahora permiten asignar hasta `os.cpu_count()` núcleos (16 en equipos compatibles), eliminando el tope previo de 4 hilos.
* **Dependencias:**
  - Añadido `pytoshop>=1.2.1` a `requirements.txt` y a las instrucciones de instalación en `README.md`.
* **Actualización de Versión:** Incremento general a **v0.6.8** en la ventana principal (`gui/main_window.py`) y panel de configuración (`gui/tab_settings.py`).

### Corregido
* **Fidelidad y Paridad Cromática Estricta en Previsualización RAW:**
  - Corregida la ausencia de `no_auto_bright=True` y curva `gamma=(2.222, 4.5)` en `_extract_single_tile_item`, erradicando la sobreexposición artificial y el halo ruidoso circular en el cielo oscuro, logrando coincidencia visual al 100% con Photoshop y exportaciones finales.
* **Manejo de Cierre y Cancelación Limpia:**
  - Implementado `closeEvent` en `EclipseCollageDialog` para interrumpir hilos de trabajo activos de forma segura al cerrar la ventana.

## [0.6.7] - 2026-10-12

### Añadido
* **Soporte Universal de Arrastrar y Soltar (Drag & Drop) en Toda la Aplicación:**
  - Nuevos componentes especializados `DropListWidget` y `DropTableWidget` (`gui/widgets/drop_widgets.py`) con feedback visual interactivo (borde cian `#80d8ff`, resalte luminoso y cursores contextuales).
  - Integrado de forma nativa en todas las listas y módulos de Apilator:
    - **Apilador (Stacker):** listas de tomas Lights, Darks, Flats, Bias y Suelo dedicado.
    - **Trazas de Estrellas (Startrails):** tabla de serie temporal de tomas y lista de Darks térmicos.
    - **Eclipses (Solar / Lunar):** tabla de tomas de bracketing multiexposición.
    - **Lienzo del Revelador (MaskCanvas):** arrastrar y soltar cualquier imagen directamente sobre el lienzo para abrirla y procesarla al instante, con overlay visual de zona de suelta.
  - Soporte inteligente para carpetas completas: exploración recursiva automática de directorios soltados con deduplicación y filtrado por extensiones de astrofotografía compatibles (`.raw`, `.cr2`, `.cr3`, `.nef`, `.arw`, `.dng`, `.fits`, `.fit`, `.tif`, `.tiff`, `.png`, `.jpg`, etc.).
* **Pipeta de Balance de Blancos (Eyedropper) en el Revelador:**
  - Nueva herramienta de calibración cromática interactiva accesible desde la barra superior del lienzo y en la Sección 3 de Balance de Blancos mediante el botón `[ 🧪 Pipeta WB (W) ]` y el atajo de teclado **`W`**.
  - Cursor de mira telescópica de precisión en cruz (`CrossCursor`) con tarjeta flotante en tiempo real que muestra una muestra de color aumentada y los valores RGB normalizados bajo el cursor.
  - Muestreo promediado en ventana $5\times 5$ píxeles para filtrar ruido residual de lectura, píxeles calientes o grano astronómico.
  - Modelo matemático de neutralización analítica exacta: resuelve en forma cerrada los multiplicadores de ganancia inversa $w = 1 / [R, G, B]$ y la fórmula de balance de Apilator ($g_r = 1 + 0.5 T + 0.25 Tint$, $g_g = 1 - 0.5 Tint$, $g_b = 1 - 0.5 T + 0.25 Tint$), calculando con precisión milimétrica los incrementos $\Delta T = g_r - g_b$ y $\Delta Tint = 2(1 - g_g)$ para neutralizar cualquier área gris o fondo neutro seleccionado en un solo clic, sincronizando deslizadores e imagen en tiempo real.
* **Lupa de Inspección Flotante (Floating Magnifier) en el Visor Interactivo:**
  - Herramienta de magnificación local accesible con el botón `[ 🔎 Lupa (Z) ]` en la barra superior del lienzo y mediante el atajo de teclado **`Z`**.
  - Lente circular flotante de radio 95 px que sigue suavemente el cursor del ratón con bisel metálico oscuro y anillo de enfoque cian.
  - Zoom de aumento ajustable interactivamente con la rueda del ratón desde **$1.5\times$ hasta $8.0\times$** (con paso de $0.25\times$), mostrando un indicador dinámico de magnificación (`2.50×`) y retícula central de precisión en pelo de camello.
  - Mapeo geométrico estricto del sensor a pantalla sin distorsión de coordenadas, permitiendo inspeccionar núcleos estelares, bordes de máscaras, ruido de fondo y textura de nebulosas sin perder la visión panorámica general ni alternar el zoom global 1:1.
  - Conmutación con exclusión mutua inteligente frente a la Pipeta WB y compatibilidad con el modo comparador Antes / Después.
* **Curvas Selectivas de Color (Hue vs Sat / Luma vs Sat) en el Revelador:**
  - Nuevo selector de modo en la Sección 2 de Curvas con botones segmentados dedicados:
    - **`[ 📈 Tonos (RGB) ]`**: curva tonal tradicional y estirado no lineal con histograma logarítmico y alertas de recorte de sombras y luces.
    - **`[ 🌈 Tono vs Sat ]`**: curva selectiva de saturación en función del tono angular (Hue vs Saturation). Incorpora franja espectral arcoíris continua (0° a 360°), línea neutra central a $1.0\times$ ($Y = 0.5$) y sincronización circular de extremos de tono rojo ($x=0.0$ y $x=1.0$). Permite saturar selectivamente canales astronómicos clave (como emisión H-alfa de nebulosas o tonos cian de OIII) o neutralizar halos cromáticos residuales.
    - **`[ 💡 Luma vs Sat ]`**: curva selectiva de saturación en función de la luminancia (Luma vs Saturation). Con franja de gradiente de escala de grises y visualización del histograma de brillo de fondo. Permite desaturar de forma suave y controlada las sombras más oscuras ($X \to 0$) erradicando el ruido de crominancia en el fondo del cielo sin desaturar nebulosas brillantes o estrellas.
  - Botones independientes `[ Resetear Curva Activa ]` y `[ Resetear Todas ]`.
  - Integración en el pipeline matemático lineal de exportación de alta precisión de 16 y 32 bits.

### Modificado
* **Actualización de Versión:** Incremento general a **v0.6.7** en la ventana principal (`gui/main_window.py`).

## [0.6.6] - 2026-10-12

### Añadido
* **Modo Comparador "Antes / Después" con Cortina Divisoria (Split-View) en el Revelador:**
  - Cortina interactiva deslizable que divide el lienzo en dos zonas para comparar la imagen original neutra y el procesado activo en tiempo real.
  - Soporte completo para orientación **Vertical** (izquierda: Antes, derecha: Después) y **Horizontal** (arriba: Antes, abajo: Después).
  - Selector de orientación dinámico `[ ⇋ Vertical ]` / `[ ⥯ Horizontal ]` en la barra superior de herramientas del lienzo.
  - Atajos de teclado: **`Y`** para activar/desactivar el comparador y **`Shift + Y`** para conmutar entre orientación horizontal y vertical.
  - Arrastre interactivo con ratón, cursores contextuales según la orientación (`↔` para vertical, `↕` para horizontal), y doble clic para recentrado rápido al 50%.
  - Etiquetas flotantes identificativas `[ ANTES ]` y `[ DESPUÉS ]` adaptadas automáticamente al eje de corte.
* **Barra Superior de Herramientas del Canvas en el Revelador:**
  - Nueva barra superior sobre el lienzo para acceso directo a herramientas visuales críticas sin saturar el panel lateral.
  - Acceso inmediato al aviso de Clipping (`[ 👁️ Clipping (J) ]`), conmutador de división (`[ ◫ Dividir Vista (Y) ]`) y orientación de cortina (`[ ⇋ Vertical / ⥯ Horizontal ]`).
* **Motor de Exportación de Vídeo Time-Lapse (.mp4) en Startrails:**
  - Integración en streaming de memoria dentro del bucle de acumulación (tanto en modo supresión de satélites como directo).
  - Codificación nativa a través de OpenCV `VideoWriter` con códec estándar `mp4v` en formato `.mp4` (reproducible en Windows Media Player, VLC, Premiere, navegadores, etc., sin requerir librerías DLL adicionales).
  - Cálculo adaptativo de dimensiones con preservación de la relación de aspecto (`compute_target_video_dims`) que garantiza dimensiones pares (múltiplos de 2) para compatibilidad con codificadores de vídeo.
  - Resoluciones configurables: **1080p Full HD**, **4K Ultra HD** y **Resolución Original del Sensor**.
  - Selector de cadencia de fotogramas: 24 FPS (cinemático), 30 FPS, 60 FPS y 12 FPS.
  - Exportación opcional de fotogramas secuenciales acumulativos en `.jpg` numerados para post-producción externa.
  - Estabilidad de suelo limpio: cuando la opción de suelo limpio está activa, fusiona cada fotograma del vídeo con el suelo de referencia para evitar parpadeos de exposición en el paisaje.
  - Botón directo `▶️ Reproducir Vídeo Time-Lapse` que abre el vídeo terminado en el reproductor del sistema.
* **Realce y Preservación de Color Estelar en Startrails (*Star Color Recovery & Boost*):**
  - Algoritmo de recuperación espectral en núcleos quemados (*Core Color Recovery*): detecta núcleos sobreexpuestos saturados a blanco ($V > 0.75, S < 0.35$) y difunde la crominancia rica de los bordes y halos no quemados mediante representación cartesiana continua $u = S \cos(H)$, $v = S \sin(H)$ (evitando discontinuidades angulares en $0^\circ/360^\circ$).
  - Las trazas de estrellas recuperan su tonalidad estelar original (azules de Rigel/Vega, naranjas de Betelgeuse/Aldebarán, amarillos de Capella) manteniendo su brillo natural.
  - Amplificación selectiva de saturación estelar (*Star Saturation Boost*) modulada por una máscara estelar suave con protección total del fondo del cielo oscuro y del suelo.
  - Botón interactivo `✨ Aplicar a la Imagen Actual (Retoque Rápido)`: permite reajustar los deslizadores de saturación y recuperación de núcleos y actualizar el visor en ~0.4 segundos sobre la imagen ya calculada sin volver a apilar las tomas.
  - Sincronización transparente con el generador de vídeo Time-Lapse.

### Modificado
* **Actualización de Versión:** Incremento general a **v0.6.6** en la ventana principal (`gui/main_window.py`).

## [0.6.5] - 2026-10-11

### Añadido
* **Arquitectura Completa de Perfiles de Color ICC (Fidelidad Cromática Profesional):**
  - Incorporación de perfiles ICC estándar en formato binario (`core/icc_profiles/`): **sRGB v2**, **Adobe RGB 1998** y **ProPhoto RGB** (ROMM RGB de amplio gamut para astrofotografía).
  - Módulo gestor `core/icc_manager.py` con utilidades de incrustación de metadatos ICC en el Tag 34675 (`InterColorProfile`) de la cabecera TIFF mediante `tifffile` y `PIL.ImageCms`.
  - Extracción y mapeo automático del espacio de color nativo de cámaras RAW (`rawpy.ColorSpace`) a perfiles ICC estandarizados.
  - Selector interactivo de perfil de color en la pestaña de Ajustes (`gui/tab_settings.py`) con persistencia en `config.json`.
  - Incrustación transversal de perfiles ICC en todas las rutinas de exportación TIFF de la aplicación: Apilador (`gui/worker.py`), Revelador (`gui/tab_developer.py`), Startrails (`gui/tab_startrails.py`) y Eclipses (`gui/tab_eclipse.py`).
  - Garantiza visualización y reproducción idéntica sin desaturación ni dominantes al abrir los resultados en Adobe Photoshop, Adobe Lightroom, Capture One o visores web.
* **Indicadores de Recorte (Clipping) de Sombras y Luces en el Canvas (Estilo Lightroom / Camera Raw):**
  - Botones interactivos triangulares en las esquinas del widget de curvas e histograma (`gui/widgets/curve_widget.py`):
    - Esquina superior izquierda (Sombras): alerta en azul si hay píxeles en negro puro (`RGB = 0`).
    - Esquina superior derecha (Luces): alerta en rojo si hay píxeles quemados al límite de rango (`RGB = 255`).
    - Detección reactiva en el histograma logarítmico con iluminación suave de advertencia y clic individual para alternar cada canal.
  - Atajo global de teclado **`J`** en el Revelador para alternar simultáneamente ambos indicadores (sombras y luces), protegido contra interferencias con campos numéricos o de texto.
  - Capa de superposición vectorial acelerada en el lienzo (`gui/canvas.py`), calculada en NumPy en menos de 3 ms a 60 fps con azul puro (`#0070ff`) para negros empastados y rojo puro (`#ff1744`) para altas luces quemadas.
  - Compatibilidad nativa e instantánea con el modo de ampliación Zoom 1:1.

### Modificado
* **Ajuste del Umbral de Almacenamiento RAM Intermedio al 70%:**
  - Optimización de la gestión dinámica de memoria en `core/stacking.py`: elevación del límite automático de almacenamiento en RAM del 60% al **70%**, aprovechando al máximo la capacidad en estaciones de trabajo y equipos con 32 GB / 64 GB de memoria sin riesgo de contención.
* **Actualización de Versión:** Incremento general a **v0.6.5** en la ventana principal (`gui/main_window.py`).

### Rendimiento
* **Aceleración del Revelado RAW con LibRaw Multihilo:**
  - Configuración optimizada de parámetros de desempaquetado y revelado de la matriz Bayer en `rawpy.RawPy.postprocess` (`demosaic_algorithm=AHD`, `fbdd_noise_reduction=Off`, `median_filter_passes=0`, `adjust_maximum_thr=0.0`) y forzado de uso de todos los núcleos lógicos del procesador vía variable de entorno `OMP_NUM_THREADS`.
  - Paralelización de franjas en la mediana de tomas maestras de calibración (`_median_stack_chunked`) mediante `ThreadPoolExecutor`.
  - **Reducción verificada en proyecto real completo** (31 Lights + 7 Darks + 12 Flats + 7 Bias + 1 Suelo RAW): el tiempo total de apilado disminuye de **1m 26s a 1m 05s** (reducción directa de 21 segundos, ~25% más rápido).

### Añadido
* **Motor Kappa-Sigma y Rechazo Estadístico Compilado con Numba (JIT LLVM):**
  - Implementación de kernels de compilación JIT en código máquina de alta eficiencia con `numba.njit(parallel=True, fastmath=True, nogil=True)`.
  - Kernel `_kappa_sigma_kernel_numba`: cálculo del MAD (Median Absolute Deviation), límites $[\mu - \kappa\sigma, \mu + \kappa\sigma]$ y suma de valores válidos en una sola pasada a nivel de registros y memoria caché L1/L2/L3 de la CPU.
  - Kernel `_percentile_kernel_numba`: cálculo optimizado de percentiles con interpolación lineal y ordenación por inserción en la pila para el rechazo asimétrico de polución lumínica (`min_rejection`).
  - Kernel `_median_kernel_numba`: cálculo de mediana acelerado para la generación por franjas de tomas maestras de calibración (Darks, Flats, Bias).
  - Eliminación total de asignaciones masivas en el heap de Python y matrices booleanas intermedias gigantes de NumPy (`valid = (channel_data >= low) & ...`), lo que reduce drásticamente el consumo de memoria RAM durante la integración.
  - Calentamiento anticipado (*pre-warming*) de los kernels JIT durante el inicio de la aplicación para evitar latencias de compilación en el primer fotograma.
  - Badge de hardware dinámico en el panel de Configuración que refleja el estado de activación del compilador JIT Numba.

### Modificado
* **Supresión de Pausas de Garbage Collector en Multi-hilo (Alineación Instantánea):**
  - Eliminación de llamadas per-frame a `gc.collect()` en `align_single_light_task` y `preprocess_subframe_lp` (que provocaban hasta 60 detenciones completas de GC en lotes de 30 fotos).
  - Erradicación de las pausas *Stop-the-World* y contención del GIL (Global Interpreter Lock) que serializaban y congelaban los hilos de trabajo paralelos en `ThreadPoolExecutor`.
  - La liberación de memoria de las matrices temporales intermedias (`warped_processed`, `raw_frame`) se delega en el conteo de referencias determinista de CPython (`free()` inmediato a nivel de C sin sobrecoste).
  - Preservación de limpiezas macro de recolección de basura exclusivamente en las transiciones de fases (`gui/worker.py`), asegurando que la memoria se mantenga limpia sin penalizar los tiempos de cómputo.
* **Actualización de Versión:** Incremento general a **v0.6.4**.

### Rendimiento
* Reducción drástica del tiempo total de apilado y alineación: el tiempo global de procesamiento para lotes de prueba ha descendido de **1m 40s a 56 segundos** (reducción cercana al 45% en tiempo de cómputo total).

## [0.6.3] - 2026-10-09

### Añadido
* **Fusión HDR por Exposición Multiescala para Eclipses Lunares (Mertens-Kautz-Van Reeth):**
  - Implementación de fusión por exposición multiescala para series de bracketing de eclipses lunares (`mode="lunar"`).
  - Descomposición en pirámides laplacianas ponderadas simultáneamente por contraste local, saturación y exposición óptima.
  - Resuelve el rango dinámico extremo (> 12–14 EV) entre el limbo intensamente iluminado por el Sol y la umbra rojiza ("Luna de Sangre"), revelando tanto el relieve fino de cráteres en las altas luces como la textura de los mares lunares en la sombra sin quemar ni empastar.
  - Erradica por completo los artefactos de corte dentado quemado en cian/blanco y la compresión al negro absoluto que producía la división de flujo lineal.
  - Conmutación automática en la GUI: título dinámico *"4. Fusión HDR (Mertens Multiescala)"* y compresión Asinh inicializada en *Lineal (1x)* al pasar a Eclipse Lunar.
* **Detección Subpíxel Robusta del Limbo Lunar:**
  - Transformada circular de Hough multiescala (`cv2.HoughCircles`) con barrido adaptativo de umbral de acumulador (`param2 = [35, 28, 22, 16, 12]`) y guiado por la región de pico de brillo lunar.
  - Sustitución del percentil global fijo (p80) que seleccionaba erróneamente nubes y halos difusos circundantes (evitando radios sobredimensionados de 450–600 px y fijando con exactitud el radio real de la Luna de ~72 px).
  - Algoritmo de reserva adaptativo local (Otsu) centrado en el foco de brillo para fases con bajo contraste.
  - Detección unificada en lote sin arrastrar radios obsoletos o desajustados en la toma de referencia.
* **Alineación de Eclipses con Réplica de Bordes:**
  - `register_frame_to_center` y `align_eclipse_bracketing` emplean ahora `cv2.BORDER_REPLICATE` para los desplazamientos entre tomas, eliminando las bandas y costuras negras en los márgenes exteriores de la composición final.
* **Relleno Continuo de Saltos entre Tomas (*Gap Filling*) en Trazas de Estrellas:**
  - Algoritmo de puenteo morfológico de luminancia entre fotogramas contiguos que elimina la apariencia de trazos punteados o discontinuos provocados por la pausa del intervalómetro y el ciclo de obturación.
  - Rango calibrado de precisión (1 a 4 px, por defecto 2 px) para evitar deformaciones laterales y preservar el grosor fino y natural de las estrellas.
  - Preservación matemática total del fondo de cielo oscuro e inmunidad a ruidos.
  - Fidelidad cromática RGB completa para mantener los colores naturales de las estrellas.
  - Exclusión automática del suelo mediante máscara de horizonte y compatibilidad nativa con el Modo Cometa.
* **Cancelación Instantánea de Procesamiento de Startrails:**
  - El botón de ejecución alterna dinámicamente a `🛑 CANCELAR PROCESAMIENTO` en color rojo durante el apilado, deteniendo el proceso de forma inmediata (< 1s) sin bloquear la GUI.

### Modificado
* **Homogeneización Visual y Rediseño de Paneles en Todos los Módulos:**
  - Paneles de control laterales unificados con ancho mínimo de 420 px, máximo de 520 px y predeterminado de 520 px en todos los módulos (`tab_stacker`, `tab_developer`, `tab_eclipse`, `tab_startrails`, `tab_settings`).
  - Armonización estilística con el esquema visual de StarTrails y Ajustes: tonos oscuros azulados (`#161622`), bordes redondeados pulidos (`#2d2d3a`), cabeceras temáticas coloreadas para cada grupo de parámetros.
  - Compactación vertical en Apilador: selectores desplegables alineados a la derecha de los títulos (algoritmo, rechazo, interpolación) y control horizontal en línea para el tamaño de pincel del cursor.
  - Compactación vertical en Revelador: fila compacta horizontal para el control de "Atenuar cúpula de luz".
* **Actualización de Versión:** Incremento general a **v0.6.3**.

### Corregido
* **Reseteo Integral del Revelador / Editor al Enviar Trazas:**
  - `load_image_direct` realiza ahora una limpieza completa del estado previo (capas StarNet++, curvas, proxies de previsualización y ajustes) para garantizar que una nueva imagen enviada desde Startrails se cargue 100% limpia sin mezclar capas residuales.
* **Eliminación de Pendiente en UI:** Se suprimió la mención a *Gap filling* de la lista de tareas pendientes en el panel de Ajustes tras su implementación completa.

## [0.6.2] - 2026-10-09

### Añadido
* **Nuevo Módulo Dedicado de Trazas de Estrellas (Startrails / Circumpolares):** Integración completa como **Pestaña 4** en la ventana principal, permitiendo la generación de trazas estelares de alta fidelidad fotométrica con streaming eficiente en memoria $O(1)$ sin importar el número ni tamaño de las tomas RAW/TIFF.
* **Algoritmo de Máximo Estándar (Lighten Clásico):** Composición progresiva de luminancia máxima para trazas estelares continuas.
* **Algoritmo de Efecto Cometa / Estela Progresiva (Comet / Meteor Fade):**
  - Factor de longitud de estela ajustable porcentualmente (5% a 100%).
  - Modulación de curva de decaimiento matemático: lineal uniforme, cosenoidal suave y exponencial rápido.
  - 3 modos de dirección temporal: hacia atrás (*backward*, cola en tomas pasadas), hacia adelante (*forward*, cola en tomas futuras) y simétrico / ambos sentidos (*bidirectional*, cabeza brillante central con colas afiladas hacia pasado y futuro).
  - Cota mínima de luminancia de fondo (*min_floor*) para evitar cielos artificialmente oscuros.
* **Fusión de Suelo Limpio (Anti-Ruido) con Máscara de Horizonte:**
  - Separación completa entre cielo de trazas y terreno estático.
  - Modos de suelo seleccionables: promedio temporal de toda la serie (reducción drástica del ruido térmico y sombras en el terreno) o toma de referencia fija.
  - Conexión e importación directa en un solo clic de la máscara de horizonte calculada en el Apilador (`🖌️ Usar Máscara del Apilador`) o carga desde archivos externos (PNG, TIFF, FITS).
  - Suavizado gaussiano de borde de máscara ajustable (*feathering*).
* **Supresión Automática de Trazas de Satélites y Aviones (Anti-Trazas Transitorias):**
  - Detección adaptativa temporal con búfer deslizante inteligente y doble referencia estadística ($\min(I_A, I_B)$) en toda la serie, incluyendo fotogramas iniciales y finales.
  - Cierre morfológico lineal que reconecta las pulsaciones discontinuas de luces estroboscópicas de aeronaves en trazas continuas.
  - Discriminación geométrica por elongación lineal (`cv2.minAreaRect`), diferenciando matemáticamente trazas rectas ($\text{elongación} \ge 2.8 - 4.0$) de estrellas en rotación o cúmulos estelares circulares ($1.0 - 1.8$).
  - Exclusión automática del suelo y vegetación en movimiento mediante la máscara de horizonte, evitando falsos positivos provocados por ramas o hierba mecidas por el viento.
  - Inpainting y sustitución acelerada por caja envolvente (*bounding box slicing* local) con tiempo de cómputo inferior a $0.15\text{ s}$ por toma de 24.5 MP.
  - Selector de sensibilidad ajustable: Baja, Media (por defecto) y Alta.
* **Exportación y Flujo de Trabajo Directo:**
  - Botones dedicados para exportación en **TIFF 16-bit** (compatible con Photoshop y Lightroom sin mapeos de tono) y **TIFF 32-bit Float** (rango dinámico HDR maestro).
  - Botón de transferencia directa al **Revelador / Editor** (`➡️ Enviar al Revelador / Editor`) con carga instantánea de imagen y máscara.

### Modificado
* **Mejoras en el Visor y Canvas Interactivo (`gui/canvas.py`):**
  - Desacoplamiento de la visualización de máscara de la herramienta de dibujo interactivo: ahora la superposición de máscara coloreada (verde cielo / rojo suelo) es visible en cualquier módulo con solo activar la casilla correspondiente.
  - Auto-reescalado dinámico de la máscara a la resolución exacta de la imagen base cargada en el visor.
* **Organización y Nomenclatura de Pestañas Principales:**
  1. `1. Apilador (Stacker)`
  2. `2. Revelador / Editor`
  3. `3. Eclipses (Solar / Lunar)`
  4. `4. Trazas de Estrellas (Startrails)`
  5. `5. Configuración / Ajustes`
* **Actualización de Versión:** Incremento general a **v0.6.2**.

---

## [0.6.1] - 2026-10-08

### Añadido
* **Vitrina de Resultados y Galería en Configuración:**
  - Nueva columna derecha en la pestaña de Configuración con visor responsivo antialias (`ShowcaseViewer`) que presenta una fotografía real de la Vía Láctea procesada íntegramente con Apilator (`showcase_milkyway.jpg`), adaptándose fluidamente al tamaño de ventana manteniendo su relación de aspecto.
  - Acreditación de autoría fotográfica integrada (**📷 Fotografía y Procesado: Shilmar**) con insignias técnicas de motor (`32-bit HDR`, `StarNet++ AI`, `MTF Asinh`, `Alineación Subpíxel`).
  - Botón de acceso directo (`🔍 Ver imagen completa`) para abrir la toma en alta resolución en el visor predeterminado del sistema operativo.
* **Monitor Dinámico de Estado para StarNet++ CLI:** En el panel de configuración, StarNet++ ahora valida en tiempo real la existencia del archivo ejecutable mostrando un distintivo dinámico en verde (`✓ StarNet++ detectado y listo`) o advertencia en ámbar.

### Modificado
* **Unificación Estética Global e Interfaz Moderna:**
  - Rediseño del panel izquierdo del **Apilador (Stacker)** con contenedor `QScrollArea`, consola de actividad inferior y barra de progreso fijas (siempre visibles sin scroll), e iconografía enriquecida (`⭐ Lights`, `🌑 Darks`, `⚪ Flats`, `⚙️ Bias`, insignia `📌 REF`).
  - Rediseño del panel izquierdo del **Revelador / Editor** con consola de actividad fija abajo, reordenación lógica de controles en 7 secciones secuenciales de revelado y emparejamiento en 2 columnas de deslizadores complementarios (Temperatura/Tinte, Saturación Cielo/Suelo, Claridad/Neblina, Ondículas/Atenuación, Contraste/Ruido, Sombras/Punto Negro).
  - Ajuste de espaciado y desahogo visual en el Revelador para una interacción cómoda y ergonómica.
  - Fijación de la barra de progreso y consola de log en la base del módulo de **Eclipses**.
  - Rediseño en división de 2 columnas de la pestaña de **Configuración** con paneles oscuros estilizados, campos de texto oscuros e interactivos y botón de guardado destacado.
* **Actualización de Versión:** Incremento general a **v0.6.1**.

---

## [0.6.0] - 2026-10-08

### Añadido
* **Nuevo Módulo Especializado de Eclipses (Solar / Lunar):** Integración completa como **Pestaña 3** en la ventana principal, permitiendo el procesado integral de series de bracketing de eclipses solares (corona, cromosfera y protuberancias) y eclipses lunares.
* **Selector de Modalidad Compacto (Boolean Toggle):** Interruptor interactivo tipo píldora booleana estilizado para alternar al instante entre Eclipse Solar (modo cromosfera/corona) y Eclipse Lunar (modo disco reflectivo), optimizando más de 50 px de espacio vertical en el panel.
* **Ingesta Inteligente de Bracketing y Parser EXIF:** Extracción automática de metadatos reales de tiempo de obturación, ISO y apertura desde formatos RAW propietarios (`.nef`, `.cr2`, `.cr3`, `.arw`, `.dng`) y formatos estándar (`.tif`, `.tiff`, `.fits`), ordenando el lote automáticamente por tiempo de exposición.
* **Detección Subpíxel de Limbo Inmune a Protuberancias (`optimize_center_circular_flux`):**
  - Detección topológica inicial del hueco lunar y escáner radial subpíxel en dos pasadas.
  - Algoritmo de optimización de flujo de gradiente radial perpendicular basado en estadística de mediana sobre 360 rayos angulares. Proporciona inmunidad matemática total frente a grandes protuberancias de hidrógeno-alfa, fulguraciones o cuentas de Baily, ubicando el centro `(cx, cy)` con precisión inferior a 0.1 px en la base real de la silueta lunar.
* **Alineación de Lote con Radio Físico Anclado:** Durante la detección en lote (*Detectar Limbo en Todo el Lote*), el radio lunar se ancla de forma exacta al diámetro físico de la toma de referencia para toda la ráfaga, eliminando cualquier encogimiento o fluctuación del radio y registrando la deriva con interpolación afín de alta fidelidad Lanczos4.
* **Fusión Fotométrica HDR Lineal de 32 bits:** Combinación de flujo lineal ponderada por tiempos de exposición reales con umbralización suave de saturación al 99.9995% y estirado no lineal Asinh para levantar la corona media y externa sin quemar protuberancias.
* **Filtro NRGF (Normalized Radial Gradient Filter) de Rango Dinámico Ampliado:**
  - Interpolación continua subpíxel (`np.interp`) sobre los perfiles radiales $\mu(r)$ y $\sigma(r)$, erradicando al 100% cualquier efecto de bandas concéntricas o escalonamientos discretos.
  - Modulación de filamentos liberada de amortiguación excesiva por luminancia, permitiendo el estiramiento y contraste completo de los filamentos coronales magnéticos ("los hilos") hasta más de 3.5 a 4.0 radios solares.
  - Compuerta radial suave (*Radial Gate*) mediante función *smoothstep* que desvanece suavemente la modulación antes del límite de la corona activa, garantizando un cielo de fondo negro aterciopelado sin elevar el grano del sensor.
* **Filtro Bilateral Tangencial de Alta Frecuencia:** Realce selectivo de filamentos finos a lo largo de las líneas de campo coronal.
* **Exportación Dual en TIFF de 16 bits y 32 bits:**
  - Botón dedicado **Guardar TIFF 16-bit**: convierte linealmente de `float32` a entero `uint16` sin pérdida con compresión `zlib` y metadatos `photometric='rgb'`, permitiendo abrir la imagen directamente en **Photoshop o Lightroom** como un TIFF estándar de 16 bits sin disparar cuadros de diálogo de mapeo tonal ni conversiones extrañas.
  - Botón dedicado **Guardar TIFF 32-bit**: exporta en coma flotante `float32` con rango dinámico lineal HDR completo para procesado científico.
* **Puente Directo con el Revelador / Editor:** Botón *Enviar al Revelador / Editor* que transfiere la imagen 32-bit procesada directamente a la Pestaña 2 y conmuta la vista de forma inmediata.

### Modificado
* **Organización y Nomenclatura de Pestañas Principales:**
  1. `1. Apilador (Stacker)`
  2. `2. Revelador / Editor`
  3. `3. Eclipses (Solar / Lunar)`
  4. `4. Configuración / Ajustes`
* **Persistencia de Selección en Tabla de Bracketing:** Selección bidireccional estable que preserva la fila activa y actualiza inmediatamente el canvas interactivo y los controles de coordenadas sin desfases.
* **Valores Predeterminados del Filtro NRGF:** Alcance radial por defecto actualizado a **3.5x** y mezcla de filamentos al **50%**.

---

## [0.5.12] - 2026-10-06
### Añadido
* Motor de gestión de memoria ultra-eficiente para apilado de lotes masivos.
* Pipeline nativo CPU multi-hilo optimizado por franjas de procesamiento (chunks).
* Eliminación de cuellos de botella en la comunicación entre procesos (IPC).

## [0.5.11] - 2026-10-05
### Añadido
* Validación preventiva de orientación de fotogramas.
* Suite unificada de StarNet++ con manejo robusto de excepciones y formatos FITS/TIFF.

## [0.5.10] - 2026-10-04
### Añadido
* Pipeline completo de calibración astronómica con pestañas independientes para Flats y Bias.
* Refinamiento guiado de bordes en segmentación con visualizador 1:1 nativo.
