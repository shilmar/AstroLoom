# Changelog

Todas las novedades, mejoras y correcciones notables de **Apilator** se documentan en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/) y este proyecto se adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

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
