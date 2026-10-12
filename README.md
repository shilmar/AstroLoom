# Apilator

**Apilator** es una herramienta especializada de posprocesado, apilado y revelado para **astrofotografía de paisaje (nightscapes)**, **circumpolares / trazas de estrellas (startrails)** y **eclipses (solares y lunares)**. Permite desacoplar el movimiento del cielo respecto al horizonte terrestre, aplicar integración estadística Kappa-Sigma en coma flotante de 32 bits, componer circumpolares con efectos cometa y supresión automática de satélites, procesar series de bracketing de eclipses con alineación subpíxel de limbo y filtro NRGF continuo, y editar la imagen final en tiempo real con asistencia de redes neuronales (GraXpert y StarNet++ v2).

---

## Características Principales

### 1. Calibración Astronómica Rigurosa
- **Pipeline Completo de Calibración:** Pestañas dedicadas y soporte nativo para **Lights**, **Darks**, **Flats** y **Bias**.
- **Sustracción de Pedestal Electrónico y Térmico:** Eliminación del ruido térmico y píxeles calientes mediante Master Dark, y del ruido de lectura/offset mediante Master Bias.
- **Aplanado de Campo y Viñeteo (Master Flat):** Corrección lineal de sombras de motas de polvo y caída de luz periférica del tren óptico, con normalización independiente por canal para preservar el balance de color.
- **Persistencia de Proyecto (.mwstack):** Guardado y restauración completa del estado de la sesión (listas de archivos de calibración, parámetros de integración y máscaras calculadas).

### 2. Apilado Diferencial Cielo / Suelo
- **Alineación Estelar Robusta:** Detección de estrellas y emparejamiento homográfico/afín de precisión subpíxel sobre datos lineales de 32 bits.
- **Integración Kappa-Sigma Pura:** Eliminación drástica y limpia de trazas de satélites, estelas de aviones y artefactos transitorios mediante rechazo por desviación absoluta respecto a la mediana (MAD) sin elevar el ruido de fondo.
- **Conservación Íntegra del Sensor (Sin Auto-Crop):** Relleno periférico reflectivo (`BORDER_REFLECT`) que neutraliza los marcos y escalones oscuros provocados por la rotación del campo estelar, conservando el 100% de la resolución nativa original.
- **Motor Optimizado por Bloques:** Procesamiento segmentado en franjas (200 filas) que previene desbordamientos de memoria RAM (`MemoryError`) y cuellos de botella de transferencia GPU en lotes pesados.
- **Motor JIT con Numba (Alta Eficiencia CPU):** Compilación en código máquina nativo LLVM multihilo para integración Kappa-Sigma y percentiles que opera directamente en la memoria caché L1/L2/L3 del procesador sin alojar matrices booleanas gigantes en RAM, acelerando el apilado entre 5x y 15x.
- **Modos de Suelo Flexibles:** Integración dual completa (cielo alineado + suelo estático), uso del suelo de la toma de referencia o composición directa desde una exposición dedicada en la pestaña Suelo.
- **Antipolución Lumínica:** Algoritmos selectivos de atenuación de gradientes de fondo (Sustracción de Domo, Rechazo Asimétrico Min-Sigma y Normalización Local).

### 3. Segmentación y Máscaras Interactivas
- **GrabCut Guiado Multiescala:** Delineación asistida del horizonte y perfiles complejos (árboles, relieve montañoso, estructuras).
- **Edición Continua no Destructiva:** Permite refinar la máscara, seguir corrigiendo con trazos de pincel en tiempo real y recalcular sin reiniciar el trabajo.
- **Control Paramétrico Fino:** Ajuste del radio de difusión/suavizado gaussiano del borde (feathering) e iteraciones del modelo.
- **Zoom 1:1 Nativo con Clic Derecho:** Inspección píxel a píxel del sensor original para retoque milimétrico de bordes.

### 4. Revelador Astrofotográfico Avanzado
- **Canal Lineal Puro:** Pipeline optimizado en coma flotante (float32), preservando la linealidad fotométrica hasta la compresión tonal final.
- **Separación de Estrellas con StarNet++:** Extracción de capas Starless y Stars-only en espacio lineal, permitiendo trabajar nebulosas y polvo galáctico sin hinchar las estrellas.
- **Estructura Multiescala por Ondículas (À Trous / B-Spline):** Realce selectivo de filamentos de gas y bandas de absorción de la Vía Láctea, junto con atenuación de la capa residual de fondo.
- **Extracción de Gradientes:** Motor dual integrado con GraXpert AI y ajuste polinómico cuadrático para corregir gradientes luminosos.
- **Reducción de Ruido Adaptativa:** Múltiples métodos para el fondo (Filtro Bilateral, Filtro Guiado y Non-Local Means) respetando los límites de las estrellas.
- **Balance de Color Fino y Tonalidad:** Curvas interactivas con histograma integrado en tiempo real, balance de temperatura/tinte calibrado ($\pm0.250$), control de vibranza y saturación diferencial cielo/suelo.
- **Curvas Selectivas de Color (Hue vs Sat / Luma vs Sat):** Modelado independiente de curvas para saturación selectiva según tonalidad cromática (espectro arcoíris continuo con sincronización circular de extremos rojos) y según luminancia (desaturación suave de sombras oscuras para erradicar ruido de crominancia en el fondo).
- **Pipeta de Balance de Blancos (Eyedropper):** Calibración cromática interactiva con botón en barra superior y atajo **`W`**, cursor de mira telescópica, tarjeta flotante RGB y cálculo analítico de ganancia inversa sobre ventana de muestreo $5\times 5$ px.
- **Lupa de Inspección Flotante (Floating Magnifier):** Lente circular flotante con botón en barra superior y atajo **`Z`**, zoom dinámico ajustable con la rueda del ratón ($1.5\times$ a $8.0\times$) y retícula central para inspeccionar núcleos estelares, bordes y ruido sin perder la vista general ni conmutar el zoom 1:1.
- **Modo Comparador "Antes / Después" con Cortina Divisoria (Split-View):** Divisor deslizable interactivo en el lienzo para contrastar la toma original neutra con el revelado activo en tiempo real. Soporta orientación horizontal y vertical con atajos `Y` y `Shift + Y`, cursores dinámicos contextuales (`↔`, `↕`) y recentrado rápido al 50% con doble clic.
- **Barra Superior de Herramientas del Canvas:** Acceso directo a funciones visuales críticas (aviso de Clipping `J`, divisor de vista `Y`, orientación de cortina, Pipeta WB `W` y Lupa flotante `Z`).
- **Soporte de Arrastrar y Soltar (Drag & Drop) Universal:** Ingesta inmediata de archivos y carpetas completas en todas las listas de Apilador, Startrails y Eclipses, así como carga directa arrastrando imágenes sobre el lienzo del Revelador.

### 5. Módulo Especializado de Eclipses (Solar / Lunar)
- **Ingesta Automática de Bracketing y Metadatos:** Carga de series de exposición completa (RAW, TIFF, FITS) con extracción de tiempos de obturación reales, ISO y apertura, ordenando automáticamente las tomas.
- **Detección Subpíxel de Limbo:**
  - *Modo Solar:* Detección topológica de hueco interior de la Luna dentro del anillo coronal brillante.
  - *Modo Lunar:* Transformada circular de Hough multiescala (`cv2.HoughCircles`) con guiado por pico de brillo y fallback adaptativo local (Otsu), inmune a nubes, halos difusos y cambios de fase.
  - Refinamiento de centro y radio subpíxel milimétrico (< 0.1 px) con escáner radial y optimización de flujo circular mediano (`optimize_center_circular_flux`).
- **Alineación de Lotes sin Costuras:** Detección en lote que ancla el radio astronómico real de la toma de referencia para toda la serie y ajusta los desplazamientos afines con interpolación Lanczos4 y replicación periférica (`BORDER_REPLICATE`) que erradica costuras y bordes negros.
- **Fusión HDR Adaptativa (Solar y Lunar):**
  - *Modo Solar:* Fusión fotométrica lineal de 32 bits ponderada por tiempo de exposición con estirado no lineal Asinh para revelar desde la cromosfera y protuberancias hasta la corona externa tenue.
  - *Modo Lunar:* Fusión por Exposición Multiescala (Mertens-Kautz-Van Reeth) que equilibra el contraste extremo (> 12–14 EV) entre la zona iluminada directamente por el Sol y la umbra rojiza/terrestre, preservando los cráteres en las altas luces y la textura de la Luna de sangre en las sombras sin quemar ni empastar.
- **Filtro NRGF (Normalized Radial Gradient Filter) Continuo:**
  - Interpolación continua subpíxel (`np.interp`) sin discretización de radios, eliminando por completo cualquier artefacto de bandas concéntricas.
  - Modulación dinámica desinhibida para estirar y contrastar filamentos coronales ("hilos") hasta 3.5x–4.0x radios solares.
  - Compuerta radial suave (*Radial Gate*) que transiciona gradualmente al cielo de fondo profundo para mantenerlo negro aterciopelado sin ruido.
  - Filtro bilateral tangencial de alta frecuencia para realce selectivo de líneas de campo magnético solar.
- **Exportación Dual 16-bit / 32-bit:** Exportación directa a TIFF de 16 bits optimizada (`uint16`, zlib, `photometric='rgb'`) compatible de forma nativa con Photoshop y Lightroom sin mapeos forzados de tono, y TIFF de 32 bits flotante para archivo maestro.

### 6. Módulo Dedicado de Trazas de Estrellas (Startrails / Circumpolares)
- **Streaming de Memoria $O(1)$:** Procesamiento secuencial sin cargar el lote completo en RAM, permitiendo componer cientos de tomas RAW o TIFF de ultra-alta resolución (24MP–60MP) sin consumo excesivo de memoria.
- **Relleno Continuo de Saltos entre Tomas (*Gap Filling*):**
  - Puenteo morfológico de luminancia calibrado (1 a 4 px, por defecto 2 px) que reconecta los trazos discontinuos o punteados originados por la pausa del intervalómetro o el ciclo de obturación.
  - Preserva el grosor fino de las estrellas, la fidelidad cromática RGB y el fondo de cielo oscuro profundo, con exclusión del suelo y compatibilidad con el Modo Cometa.
- **Cancelación Instantánea:** Botón dinámico `🛑 CANCELAR PROCESAMIENTO` que detiene el flujo de trabajo de forma inmediata (< 1s) sin bloquear la interfaz.
- **Soporte de Tomas Dark:** Calibración térmica previa mediante Master Dark para eliminar píxeles calientes (*hot pixels*) antes de la integración.
- **Algoritmo de Máximo Estándar (Lighten Clásico):** Fusión de luminancia máxima para trazas continuas.
- **Algoritmo de Efecto Cometa / Estela Progresiva (Comet / Meteor Fade):**
  - Ajuste porcentual de la longitud de la estela.
  - Curvas de decaimiento matemático: lineal uniforme, cosenoidal suave y exponencial rápido.
  - 3 modos de dirección temporal: hacia atrás (*backward*), hacia adelante (*forward*) y simétrico / ambos sentidos (*bidirectional*).
  - Cota mínima de luminancia de fondo (*min_floor*) para conservar el cielo profundo.
- **Fusión de Suelo Limpio (Separación Cielo / Suelo):**
  - Eliminación del ruido térmico y sombras en el terreno integrando el suelo mediante promedio temporal o toma fija de referencia.
  - Importación directa en un clic de la máscara de horizonte generada en el Apilador o carga de archivos de máscara externos (PNG, TIFF, FITS).
  - Suavizado gaussiano de transición de bordes (*feathering*).
- **Supresión Automática de Satélites y Aviones (Anti-Trazas Transitorias):**
  - Detección temporal diferencial basada en el mínimo de dos referencias estadísticas simultáneas ($\min(I_A, I_B)$).
  - Discriminación geométrica por elongación lineal (`cv2.minAreaRect`), reconectando destellos estroboscópicos de aviones y diferenciando satélites de estrellas en rotación.
  - Exclusión automática del suelo mediante máscara de horizonte para ignorar vegetación o ramas que se muevan con el viento.
  - Inpainting acelerado por caja envolvente local ($<0.15\text{ s}$ por toma) y sensibilidad ajustable (Baja, Media, Alta).
- **Generador de Vídeo Time-Lapse (.mp4):** Exportación en streaming de vídeo de la circumpolar durante la acumulación mediante códec nativo `mp4v` (1080p, 4K y resolución original), cadencia configurable (24, 30, 60, 12 FPS), exportación opcional de fotogramas secuenciales JPG, estabilidad de suelo limpio anti-flicker y botón de reproducción directa.
- **Realce y Preservación de Color Estelar (*Star Color Recovery & Boost*):** Algoritmo de recuperación espectral en núcleos quemados a blanco (mediante difusión cartesiana continua $(u, v)$) y saturación selectiva de trazas con protección total del fondo nocturno, junto con botón de retoque interactivo ultrarrápido (~0.4s) sobre la imagen ya calculada sin volver a apilar las tomas.
- **Exportación Dual 16-bit / 32-bit y Enlace Directo:** Guardado en TIFF 16-bit (compatible con Photoshop y Lightroom sin mapeos de tono), TIFF 32-bit float, o transferencia instantánea al Revelador / Editor.

### 7. Configuración, Rendimiento y Vitrina de Resultados
- **Interfaz Moderna y Homogénea:** Paneles laterales unificados con ancho mínimo de 420 px, máximo de 520 px y predeterminado de 520 px en todos los módulos, diseño visual armónico en tonos azulados oscuros (`#161622`) y controles compactados verticalmente (selectores en línea en Apilador y Revelador).
- **Vitrina de Resultados Integrada (Showcase):** Panel visual en la pestaña de Configuración con visor responsivo antialias que exhibe astrofotografía real de paisaje procesada de principio a fin con Apilator, acreditación de autoría (*Fotografía y Procesado: Shilmar*) e inspección a resolución nativa.
- **Aceleración por GPU y JIT Dual:** Soporte automático para NVIDIA CUDA mediante CuPy, compilación JIT de CPU multinúcleo con Numba (kernels paralelos LLVM), y conmutación transparente a CPU estándar si no se dispone de librerías aceleradoras.
- **Aceleración LibRaw Multihilo:** Desentrelazado Bayer paralelo con LibRaw/rawpy y paralelización de tomas maestras de calibración que acelera la integración entre un 20% y 25%.
- **Estrategias de Memoria Configurables:** Modos automático (umbral del 70% de RAM disponible), memoria RAM intermedia de alta velocidad o volcado temporal a disco SSD para equipos con recursos limitados.
- **Detección Automática de Binarios:** Localización y validación dinámica de ejecutables externos de StarNet++ CLI y GraXpert con indicadores en tiempo real de disponibilidad.
- **Indicadores de Recorte (Clipping) Estilo Lightroom:** Alertas visuales interactivas en las esquinas del histograma para sombras empastadas (azul) y luces quemadas (rojo), superposición vectorial acelerada en el lienzo (< 3 ms) y atajo global de teclado **`J`**.

### 8. Fidelidad Cromática y Perfiles de Color ICC
- **Incrustación de Metadatos ICC en TIFF (Tag 34675):** Soporte nativo para incrustar perfiles de color en la cabecera `InterColorProfile` de los archivos TIFF exportados en todos los módulos (Apilador, Revelador, Startrails y Eclipses), garantizando que las imágenes conserven su gama tonal y saturación de forma idéntica en Adobe Photoshop, Adobe Lightroom, Capture One o navegadores web.
- **Espacios de Color Estándar Suministrados:**
  - **sRGB:** Perfil estándar universal para publicación en internet y pantallas convencionales.
  - **Adobe RGB (1998):** Gama amplia recomendada para impresión fotográfica y pantallas de artes gráficas.
  - **ProPhoto RGB (ROMM RGB):** Gamut ultra-amplio de alta fidelidad adaptado a la astrofotografía para preservar tonos extremos de nebulosas ($H\alpha$, $O\text{III}$) y estrellas saturadas sin recorte cromático.
- **Selector Persistente en Ajustes:** Selección del perfil de salida en la pestaña de Ajustes con persistencia en `config.json`.

---

## Estructura del Proyecto

```text
apilator/
├── config.json              # Configuración persistente del usuario
├── run_app.py               # Punto de entrada de la aplicación
├── core/
│   ├── config_manager.py    # Carga y almacenamiento de ajustes JSON
│   ├── eclipse.py           # Detección de limbo, alineación subpíxel, fusión HDR y NRGF continuo
│   ├── gpu_backend.py       # Detección y gestión de aceleración NVIDIA CUDA / CuPy
│   ├── graxpert_bridge.py   # Conector CLI con GraXpert AI
│   ├── icc_manager.py       # Gestión e incrustación de perfiles ICC en TIFF (Tag 34675)
│   ├── icc_profiles/        # Perfiles de color ICC estándar embebibles
│   │   ├── AdobeRGB1998.icc # Espacio amplio para monitores profesionales e impresión
│   │   ├── ProPhotoRGB.icc  # Espacio de gamut extendido (ROMM RGB) para astrofotografía
│   │   └── sRGB.icc         # Espacio estándar universal para web y visualización
│   ├── masking.py           # Algoritmos de segmentación y refinado guiado de máscaras
│   ├── project_manager.py   # Serialización y persistencia de proyectos (.mwstack)
│   ├── stacking.py          # Motor de calibración (Dark/Flat/Bias), alineación y apilado
│   ├── starnet_bridge.py    # Conector y parser CLI con StarNet++ v2
│   ├── startrails.py        # Motor streaming de trazas, efectos cometa y supresión de satélites
│   └── stretch.py           # Algoritmos MTF, ondículas À Trous, balance y tono
└── gui/
    ├── assets/              # Recursos gráficos y fotografía de demostración
    │   └── showcase_milkyway.jpg
    ├── canvas.py            # Visor interactivo QGraphicsView acelerado con zoom 1:1 y clipping
    ├── main_window.py       # Ventana principal y gestión de pestañas maestras
    ├── tab_developer.py     # Pestaña de revelado y composición de capas
    ├── tab_eclipse.py       # Pestaña de procesado integral de eclipses solares y lunares
    ├── tab_settings.py      # Pestaña de configuración de rutas, selector ICC y parámetros
    ├── tab_stacker.py       # Pestaña de apilado dual, calibración y máscaras
    ├── tab_startrails.py    # Pestaña de generación de trazas de estrellas (Startrails)
    ├── widgets/
    │   └── curve_widget.py  # Curvas tonales, histograma logarítmico y botones de clipping
    └── worker.py            # Orquestador de tareas en segundo plano multihilo (QThread)
```

## Requisitos del Sistema

- **Python:** 3.10 o superior.
- **Dependencias Principales:**
  - `PySide6` (interfaz gráfica basada en Qt)
  - `numpy`, `scipy` (cálculo numérico y splines)
  - `numba` (compilación JIT LLVM multihilo para integración estadística ultra-rápida)
  - `opencv-python` (visión por computador y transformaciones geométricas)
  - `rawpy` (decodificación de archivos RAW de cámara)
  - `tifffile` (lectura y escritura de imágenes TIFF de alta profundidad)
  - `astropy` (gestión de archivos FITS astronómicos)
  - `Pillow` (gestión de perfiles de color ICC y metadatos)
  - `imagecodecs` (códecs extendidos de compresión de imagen)
  - `psutil` (monitorización de recursos de memoria y CPU)
  - *(Opcional)* `cupy` (aceleración por GPU NVIDIA CUDA)
- **Binarios Externos:**
  - Ejecutables de StarNet++ CLI y GraXpert configurados en el entorno.
---


### 1. Clonar el repositorio
```cmd
git clone https://github.com/shilmar/apilator.git
cd apilator
```

### 2. Instalar dependencias
```cmd
pip install -r requirements.txt
```

O instalando manualmente los paquetes requeridos:
```cmd
pip install numpy scipy opencv-python PySide6 tifffile rawpy matplotlib astropy imagecodecs psutil numba
```

### 3. Aceleracion GPU (Opcional - NVIDIA CUDA)
Si dispones de una tarjeta grafica NVIDIA, puedes habilitar el procesamiento acelerado instalando la version de CuPy adecuada a tus controladores CUDA:
* CUDA 12.x: pip install cupy-cuda12x
* CUDA 11.x: pip install cupy-cuda11x

(Si CuPy no esta presente, el motor utiliza automaticamente todos los nucleos logicos de la CPU).

---

## Uso de la Aplicacion

Inicia la herramienta ejecutando:
python run_app.py

## Flujo de Trabajo Recomendado
### 1. Flujo de Calibración y Apilado
* Abre la pestaña **1. Apilador (Stacker)**.
* Carga tus tomas de luz en Lights y haz doble clic sobre la toma que servirá como base de encuadre.
* (Opcional) Carga tomas en Darks, Flats y Bias para corrección de ruido térmico, viñeteo óptico y offset.
* Dibuja los trazos guía sobre la vista previa: Verde para el cielo y Rojo para el suelo.
* Haz clic en Refinar Automática para generar la máscara. Inspecciona los bordes manteniendo pulsado el botón derecho del ratón para ver la imagen al 100% de resolución nativa.
* Selecciona el Modo de Captura (Trípode Fijo o Star Tracker), el tratamiento del suelo deseado y el Factor Kappa.
* Pulsa INICIAR APILADO DUAL.

### 2. Flujo de Revelado
* Al concluir el apilado, la imagen lineal de 32 bits y su máscara calculada se cargarán automáticamente en la pestaña **2. Revelador / Editor**.
* Neutraliza gradientes residuales con el motor Polinómico o GraXpert AI.
* Ejecuta StarNet++ para separar el fondo galáctico de las estrellas.
* Aplica realce de gas y polvo molecular mediante los controles de Estructura Multiescala (Ondículas) sin deformar el perfil estelar.
* Ajusta curvas, temperatura de color, tinte y saturación diferencial cielo/suelo.
* Exporta el resultado final en formato TIFF 16-bit, TIFF 32-bit float o JPEG.

### 3. Flujo de Procesado de Eclipses (Solar / Lunar)
* Abre la pestaña **3. Eclipses (Solar / Lunar)**.
* Selecciona la modalidad deseada con el conmutador interactivo (**Solar** o **Lunar**).
* Pulsa **Cargar Serie de Bracketing** para importar todas las exposiciones (RAW, TIFF o FITS). El sistema leerá automáticamente tiempos de obturación e ISO y ordenará las tomas de menor a mayor exposición.
* Haz doble clic sobre una toma con buena definición del limbo para fijarla como referencia.
* Pulsa **Detectar Limbo en Todo el Lote**: el sistema detectará el centro y radio físico en la referencia y optimizará el centro subpíxel (`< 0.1 px`) en cada fotograma del lote manteniendo el radio anclado.
* Pulsa **Alinear y Fusionar Bracketing (HDR)** para generar el compuesto lineal de 32 bits con compresión Asinh.
* Ajusta los parámetros del filtro **NRGF**: alcance de la corona (por defecto 3.5x), mezcla de filamentos (50%) y filtro tangencial para estirar y contrastar los filamentos coronales sin quemar las protuberancias.
* Pulsa **Guardar TIFF 16-bit** para abrir directamente en Photoshop/Lightroom sin cuadros de diálogo de mapeo de tono, **Guardar TIFF 32-bit** para procesado HDR de alta fidelidad, o envíalo directamente al **Revelador / Editor**.

### 4. Flujo de Creación de Trazas de Estrellas (Startrails)
* Abre la pestaña **4. Trazas de Estrellas (Startrails)**.
* Pulsa **📁 Cargar Serie...** y selecciona la secuencia temporal completa de tomas fijas (RAW, TIFF, FITS o JPG).
* (Opcional) Pulsa **📌 Marcar REF** sobre la toma que servirá de encuadre o para el suelo de referencia.
* (Opcional) Pulsa **Añadir Darks...** para sustraer el ruido térmico y píxeles calientes.
* Selecciona el **Algoritmo de Trazas (Cielo)**:
  - **⭐ Máximo Estándar (Lighten Clásico)** para circumpolares de trazo continuo.
  - **☄️ Efecto Cometa / Estela Progresiva** ajustando longitud de estela (%), curva de decaimiento (lineal, cosenoidal suave o exponencial) y dirección temporal (hacia atrás, hacia adelante o simétrico/ambos sentidos).
* Si tu serie contiene satélites artificiales o destellos de aviones, activa **🛸 Suprimir satélites y aviones automáticamente** y selecciona la sensibilidad deseada (Baja, Media o Alta).
* Para corregir pequeñas pausas del intervalómetro o saltos de obturación y obtener trazas continuas sin aspecto de puntos discontinuos, activa **✨ Rellenar pausas entre tomas (Gap Filling)** y ajusta el radio del salto (1 a 4 px, por defecto 2 px).
* Para eliminar por completo el ruido del terreno estático, activa **Activar Suelo Limpio (Eliminación de Ruido)** y pulsa **🖌️ Usar Máscara del Apilador** (o importa una máscara externa). Selecciona la integración del suelo (promedio temporal o toma de referencia) y el suavizado de borde (*feathering*).
* Pulsa **⚡ Generar Trazas de Estrellas (Startrails)**.
* Guarda el resultado en **TIFF 16-bit** (para edición directa en Photoshop/Lightroom) o **TIFF 32-bit**, o pulsa **➡️ Enviar al Revelador / Editor** para seguir procesando curvas, color y contraste.

## Hoja de Ruta
[x] Apilado diferencial cielo/suelo con alineación estelar por homografía.
[x] Rechazo estadístico Kappa-Sigma (MAD) libre de trazas de satélites y aviones.
[x] Pipeline completo de calibración con Darks, Flats y Bias.
[x] Conservación íntegra de resolución nativa mediante BORDER_REFLECT.
[x] Gestión de proyectos y sesiones en disco (.mwstack).
[x] Segmentación asistida GrabCut y zoom nativo 1:1.
[x] Descomposición y realce multiescala mediante ondículas À Trous.
[x] Integración de StarNet++ v2 y GraXpert AI.
[x] Rutina específica de apilado y alineación para eclipses solares y lunares (Módulo Eclipses con NRGF continuo y detección subpíxel).
[x] Módulo dedicado de Trazas de Estrellas (Startrails) con streaming O(1), efecto cometa, suelo limpio y supresión automática de satélites/aviones.
[x] Relleno continuo de saltos entre tomas (Gap Filling) en Startrails.
[x] Exportación de perfiles de color ICC embebidos en TIFF (sRGB / AdobeRGB / ProPhoto).
[x] Generación y exportación de secuencia acumulativa de fotogramas para vídeo Time-Lapse.
[x] Soporte universal de Arrastrar y Soltar (Drag & Drop) en listas de archivos y lienzos interactivos.
[x] Pipeta de Balance de Blancos (Eyedropper) interactiva con neutralización analítica exacta.
[x] Lupa de Inspección Flotante (Floating Magnifier) con zoom dinámico de rueda de ratón.
[x] Curvas Selectivas de Color (Hue vs Sat y Luma vs Sat) con espectro arcoíris continuo.
[x] Diseñador de secuencia y collage de eclipses con cribado de trepidación, cálculo de % de ocultación y exportación PSD nativo por capas (16-bit).
[x] Indicador de progreso y estado visual explícito en la ingesta masiva de archivos ("Cargando ficheros..." -> "Carga finalizada") en el módulo de eclipses y collage.
[ ] Navegación interactiva con Zoom y Paneo en el visor de previsualización del Diseñador de Collage.
[ ] Detección y recomendación automática de fenómenos singulares de contacto: Perlas de Baily y Anillo de Diamante (Diamond Ring).
[ ] Margen y radio de recorte adaptativo / configurable para evitar cortar rayos solares, destellos del anillo de diamante y filamentos de la corona (o máscara de capa editable no destructiva en PSD).
[ ] Capa dedicada de marcas horarias EXIF y textos en la exportación de archivos PSD de Photoshop.
[ ] Aceleración multiproceso/multihilo en el cribado masivo del collage de eclipses con selector configurable de rendimiento (Velocidad half-size RAW vs Máxima resolución nativa).
[ ] Módulo de composición panorámica para mosaicos nocturnos (Nightscape Panorama Stitcher).
[ ] Modo automático de gestión de memoria RAM según carga del sistema (psutil).
[ ] Deshacer / Rehacer (Undo / Redo) con historial de pasos en el Revelador.
[ ] Anti-parpadeo (Deflicker) y morphing temporal entre fotogramas en Startrails / Time-Lapse.
[ ] Miniaturas previas (Thumbnails) en las tablas de archivos y estimación de tiempo restante (ETA) en barra de progreso.
---

## Licencia

Este proyecto esta bajo la Licencia MIT.
