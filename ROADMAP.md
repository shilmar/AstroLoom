# 🗺️ ROADMAP / TAREAS PENDIENTES — ASTROLOOM

Este documento recopila las propuestas de mejora, nuevas características y tareas pendientes planificadas para **AstroLoom**, inspiradas en los estándares de la industria y herramientas especializadas de astrofotografía de paisaje nocturno (*Nebulb*, *Sequator*, *Starry Landscape Stacker*, *Siril*, *PixInsight*).

---

## 📌 Tabla de Prioridades

| Prioridad | Tarea / Funcionalidad | Módulo | Inspiración | Dificultad (1-10) | Impacto (1-10) | Estado |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **P1** | **Realce de Estrellas / Filtro Difusor Digital** | Revelador | Sequator / Filtros físicos | 3.0 / 10 | 7.5 / 10 | ✅ Completado |
| **P2** | **Estirado Hiperbólico Generalizado (GHS)** | Revelador | Siril / PixInsight | 4.0 / 10 | 8.5 / 10 | ✅ Completado |
| **P3** | **Anti-Ghosting de Árboles en Horizonte (Min-Horizon)** | Apilador | Starry Landscape Stacker | 4.0 / 10 | 8.0 / 10 | ✅ Completado |
| **P4** | **Cribado Previo de Tomas y Calidad (Frame Culling)** | Apilador | Nebulb / Siril | 4.5 / 10 | 8.5 / 10 | ✅ Completado |
| **P5** | **Sustracción de Domo Asimétrico de Polución 2D** | Revelador | Sequator / Nebulb | 4.5 / 10 | 8.0 / 10 | ✅ Completado |
| **P6** | **Efecto Orton Cósmico y Glow de Polvo** | Revelador | Nebulb / Photoshop | 2.5 / 10 | 6.5 / 10 | ✅ Completado |
| **P7** | **Sub-stacking / Ventana Deslizante para Time-Lapse** | Startrails / Apilador | Sequator | 5.0 / 10 | 7.0 / 10 | ✅ Completado |
| **P8** | **Detección Automática de Suelo con IA Ligera** | Máscaras | Nebulb | 6.5 / 10 | 9.0 / 10 | ✅ Completado |
| **P9** | **Reducción de Distorsión Óptica en Esquinas** | Apilador | Sequator | 7.5 / 10 | 8.5 / 10 | 🔍 En estudio |
| **P10** | **Deconvolución Estelar (Richardson-Lucy / FWHM)** | Revelador | PixInsight / Siril | 7.0 / 10 | 8.0 / 10 | ✅ Completado |
| **P11** | **Persistencia Avanzada de Configuración** | Sistema | - | 3.0 / 10 | 5.0 / 10 | ⏳ Pendiente |

---

## 🚀 Desglose Detallado de Funcionalidades Pendientes

### 1. Ingesta y Calibración Previa (Pre-apilado)

#### 🔹 1.1 Cribado Inteligente de Tomas y Calidad Estelar (*Frame Culling & Star Inspector*)
* **Herramientas de referencia:** Nebulb, Siril, DeepSkyStacker.
* **Descripción técnica:** Análisis automatizado previo al apilado que evalúa cada fotograma del lote antes de iniciar el cómputo:
  - **FWHM (Full Width at Half Maximum):** Medición de la dispersión de núcleos estelares para detectar pérdida de foco o empañamiento de lente.
  - **Excentricidad / Elongación estelar:** Ajuste gaussiano 2D sobre una muestra representativa de estrellas para descartar tomas con deriva por viento o golpes de trípode.
  - **Densidad estelar y SNR:** Detección de paso de nubes densas o niebla transitoria.
  - **Interfaz de usuario:** Tabla clasificatoria con semáforo de calidad (Verde/Amarillo/Rojo) y casilla de selección automática con umbral configurable.
* **Impacto:** **8.5 / 10** *(Evita que una sola toma trepidada arruine la nitidez del compuesto final).*
* **Dificultad / Coste:** **4.5 / 10** *(Baja-Media: Aprovecha las rutinas existentes de detección de estrellas en `core/stacking.py`).*

---

### 2. Motor de Alineación y Apilado Especializado

#### 🔹 2.1 Corrección de Distorsión Óptica Periférica de Estrellas (*Reduce Star Distortion*)
* **Herramientas de referencia:** Sequator (*Complex / Distort Mode*).
* **Descripción técnica:** En objetivos gran angular (14mm–24mm a $f/1.8$–$f/2.8$), las aberraciones geométricas (coma, astigmatismo y distorsión de barril/bigote) varían radialmente. Al rotar el cielo durante sesiones de 30–60 minutos, las estrellas transitan por diferentes zonas de distorsión, impidiendo que una transformación afín u homográfica global logre coincidencia perfecta en las 4 esquinas.
  - Se implementará un registro elástico local basado en deformación por mallas de interpolación (*Thin Plate Splines* - TPS) o corrección radial por cuadrantes.
* **Impacto:** **8.5 / 10** *(Diferencia crítica de calidad entre apilados estándar y profesionales en esquinas).*
* **Dificultad / Coste:** **7.5 / 10** *(Alta: Requiere modelado matemático de deformación no rígida subpíxel).*

#### 🔹 2.2 Tratamiento Anti-Borde en Siluetas Arbóreas Complejas (*Tree Boundary Anti-Ghosting / Min-Horizon*)
* **Herramientas de referencia:** Starry Landscape Stacker, Nebulb.
* **Descripción técnica:** En horizontes con vegetación, árboles o antenas, las estrellas en rotación pasan por detrás del follaje. En un apilado clásico, esto causa "árboles fantasma" traslúcidos o estelas negras en el cielo.
  - Se implementará un algoritmo de proyección de valor mínimo (*Min-Horizon*) circunscrito exclusivamente a una banda de seguridad dilatada alrededor del contorno de la máscara de suelo, forzando la opacidad de los elementos estáticos y evitando fugas de estrellas a través de las ramas.
* **Impacto:** **8.0 / 10** *(Resuelve el escenario más habitual en astrofotografía de paisaje).*
* **Dificultad / Coste:** **4.0 / 10** *(Media: Operaciones morfológicas de dilatación de contorno combinadas con rechazo percentil inferior).*

#### 🔹 2.3 Apilado de Ventana Deslizante para Vídeo Time-Lapse (*Sub-stacking / Rolling Stacker*) [COMPLETADO]
* **Herramientas de referencia:** Sequator.
* **Descripción técnica:** Procesamiento por lotes temporales deslizantes (ej. tomas 1–5, 2–6, 3–7... o bloques de $N$ fotos) a partir de una secuencia continua de intervalómetro:
  - Genera fotogramas de vídeo time-lapse donde cada cuadro goza de la reducción de ruido ($\sqrt{W}$ de ganancia en SNR) y rango dinámico de un apilado, manteniendo a la vez el movimiento fluido del cielo.
  - Pre-cálculo lineal $O(N)$ de homografías adyacentes, buffer FIFO en RAM $O(W)$, ponderación temporal gaussiana/uniforme y streaming directo a vídeo `.mp4` estabilizado en alta resolución (1080p, 4K).
* **Impacto:** **7.0 / 10** *(Herramienta diferencial para creadores de time-lapse nocturno).*
* **Dificultad / Coste:** **5.0 / 10** *(Completado con éxito).*

---

### 3. Segmentación y Máscaras

#### 🔹 3.1 Detección Automática de Suelo / Cielo con IA Ligera (*1-Click Auto Sky/Ground*)
* **Herramientas de referencia:** Nebulb.
* **Descripción técnica:** Detección instantánea del horizonte al cargar la toma sin necesidad de dibujar trazos manuales con el ratón:
  - Inferencia local mediante red neuronal convolucional ultraligera de segmentación semántica (ej. MobileNetV3-DeepLabV3 en ONNX Runtime, $< 15\text{ MB}$) calibrada para fotografía nocturna y perfiles montañosos.
  - Los trazos de pincel verde/rojo actuales permanecerán disponibles como herramienta de retoque y refinado fino.
* **Impacto:** **9.0 / 10** *(Reduce a cero el tiempo de interacción inicial para la gran mayoría de capturas).*
* **Dificultad / Coste:** **6.5 / 10** *(Media-Alta: Integración del runtime ONNX y empaquetado del modelo pre-entrenado).*

---

### 4. Revelador y Edición Astrofotográfica

#### 🔹 4.1 Realce Espectral de Estrellas / Filtro Suave Digital (*Enhance Star Light / Digital Diffusion Filter*)
* **Herramientas de referencia:** Sequator (*Enhance Star Light*), filtros ópticos físicos (*Kenko Pro1D*, *Kase Star Diffusion*).
* **Descripción técnica:** Emulación digital de los populares filtros difusores de niebla estelar:
  - Detección de las estrellas más brillantes (percentil 95+ de luminancia estelar).
  - Aplicación de un halo gaussiano difuso proporcional a su magnitud, manteniendo su color cromático real (temperatura de color de la estrella) sin inflar los núcleos de estrellas débiles.
  - Hace destacar las figuras de las constelaciones principales (Orión, Osa Mayor, Casiopea) con un acabado etéreo y estético.
* **Impacto:** **7.5 / 10** *(Muy demandado por astrofotógrafos de paisaje que no desean adquirir o portar filtros físicos de cristal).*
* **Dificultad / Coste:** **3.0 / 10** *(Baja: Umbralización, dilatación morfológica suave y mezcla ponderada en el visor).*

#### 🔹 4.2 Estirado Hiperbólico Generalizado (*Generalized Hyperbolic Stretch - GHS*)
* **Herramientas de referencia:** Siril, PixInsight (*GHS script*).
* **Descripción técnica:** Supera las limitaciones del estirado MTF y de las curvas de contraste estándar:
  - Ecuación matemática hiperbólica con punto de simetría móvil ($SP$) que permite anclar la inflexión de estirado exactamente en la cima del histograma del gas galáctico.
  - Factor de estirado ($D$) y factor de protección de sombras ($b$) que comprimen las altas luces para evitar que las estrellas se saturen a blanco puro mientras se extrae el polvo interestelar más tenue.
* **Impacto:** **8.5 / 10** *(Aporta la máxima calidad fotométrica en el estirado de datos lineales).*
* **Dificultad / Coste:** **4.0 / 10** *(Media: Implementación matemática cerrada en `core/stretch.py` y control interactivo en GUI).*

#### 🔹 4.3 Reducción Paramétrica de Contaminación Lumínica Terrestre (*Asymmetric Light Dome Subtraction*)
* **Herramientas de referencia:** Sequator (*Reduce Light Pollution: Complex*), Nebulb.
* **Descripción técnica:** Modelado matemático específico para cúpulas de polución lumínica urbana en astropaisaje:
  - A diferencia de los modelos para cielo profundo (como GraXpert, pensados para viñeteo o gradientes suaves centrados), la polución terrestre es una cúpula asimétrica anclada en el horizonte que decae verticalmente hacia el cenit.
  - Ajuste por mínimos cuadrados de una función de decaimiento exponencial 2D ($f(x, y) = A \cdot e^{-k \cdot y} + B$) con neutralización espectral del espectro anaranjado de lámparas de vapor de sodio.
* **Impacto:** **8.0 / 10** *(Imprescindible para fotógrafos que operan en cielos Bortle 4 a 6).*
* **Dificultad / Coste:** **4.5 / 10** *(Media: Estimación robusta sobre muestras de cielo con máscara de exclusión galáctica).*

#### 🔹 4.4 Efecto Orton Astronómico y Glow Específico para Polvo Galáctico
* **Herramientas de referencia:** Nebulb, Acciones Photoshop (*MilkyWayFinisher*, *TK Panels*).
* **Descripción técnica:** Técnica estética que confiere una textura sedosa y volumétrica a las bandas de absorción y polvo de la Vía Láctea:
  - Generación de capa secundaria con estirado de alto contraste y desenfoque gaussiano amplio (radio 30–60 px).
  - Fusión en modo trama o mezcla suave modulada por máscara de luminancia de tonos medios ($L \in [0.2, 0.7]$), con supresión de halos sobre estrellas mediante la capa Starless.
* **Impacto:** **6.5 / 10** *(Acabado cosmético artístico muy cotizado).*
* **Dificultad / Coste:** **2.5 / 10** *(Baja: Capa de mezcla nativa directa en el nuevo gestor de capas de AstroLoom).*

#### 🔹 4.5 Deconvolución Estelar Regularizada (*Richardson-Lucy / PSF Estimator*) - ✅ Completado
* **Herramientas de referencia:** PixInsight, Siril.
* **Implementación:** Módulo `core/deconvolution.py` integrado en `gui/tab_developer.py`:
  - Modelado matemático de PSF sintética analítica: perfiles Moffat 2D ($\beta \approx 4.5$) y Gaussiano 2D con control subpíxel de FWHM (1.0 a 8.0 px), relación de aspecto/elipticidad (1.0 a 2.5) y ángulo de orientación de coma ($-90^\circ$ a $+90^\circ$).
  - Estimador empírico automático de PSF (`estimate_psf_from_stars`): aislamiento morfológico top-hat de estrellas de fondo, filtrado de saturadas/ruido, cálculo fraccional subpíxel del FWHM e inferencia de relación de aspecto.
  - Algoritmo iterativo de Richardson-Lucy (R-L) con regularización: desacoplo de luminancia perceptual ITU-R BT.709 para preservación espectral estricta del color de las estrellas, soporte local antianillos (*Local Deringing Support*) para erradicar halos negros ("donuts") y amortiguación bayesiana del ruido de fondo (*Bayesian Noise Damping* vía estimación MAD).
  - Ámbitos de aplicación selectivos (Toda la imagen, Solo Estrellas o Solo Fondo/Nebulosa) y protección automática del suelo en paisajes nocturnos.
* **Impacto:** **8.0 / 10** *(Máximo detalle analítico para fotógrafos avanzados).*
* **Dificultad / Coste:** **7.0 / 10** *(Alta: Algoritmo iterativo con control estricto de artefactos de anillo y estabilidad cromática).*

---

### 5. Configuración y Sistema

#### 🔹 5.1 Persistencia Avanzada y Sincronización de Preferencias
* **Descripción técnica:** Centralización en `config.json` / `QSettings` de parámetros frecuentes de herramientas (radios de difuminado, directorios de trabajo recientes, geometrías de ventana y configuraciones de rendimiento).
* **Impacto:** **5.0 / 10**
* **Dificultad / Coste:** **3.0 / 10**

---

## ✅ Tareas Recientemente Completadas
- **Extracción Analítica de Gradientes Polinómicos y Retiro de Dependencias Externas (v0.10.0):** Sustitución integral de GraXpert por el modelo matemático polinómico 2D nativo (`extract_background_polynomial` con Grado 1 lineal y Grado 2 cuadrático). Corrección inmune al relieve terrestre mediante la máscara de cielo, cálculo síncrono en ~15–25 ms y eliminación de subprocesos y archivos temporales.
- **Optimización de Rendimiento Extremo con Numba Kernel Fusion y Fast Draft Scrubbing (v0.10.0):** Fusión de kernels JIT en 32-bit float para composición de capas y máscaras, cálculo acelerado de pivote de contraste por submuestreo de mediana, supresión de histogramas durante el arrastre continuo de deslizadores y recomputación perezosa al soltar.
- **Aislamiento Estricto de Capas de Ajuste y Fidelidad de Color ICC (v0.10.0):** Corrección del sangrado de saturación en píxeles no enmascarados fuera de trazos de pincel en el kernel Numba, y respeto riguroso del perfil de color de pantalla e ICC configurado/embebido en todo el pipeline del Revelador.
- **Rediseño Integral AstroLoom Studio Pro y Layout de 3 Columnas (v0.9.0):** Nueva arquitectura visual profesional inspirada en Lightroom Classic / DaVinci Resolve con paleta `Cosmic Pro Dark` (`#0b0d13`), catálogo nativo de iconos vectoriales SVG (`gui/icons.py`), distribución en 3 columnas en el Revelador con separadores dinámicos de amplio recorrido al centro e histograma RGB en vivo interactivo en cabecera.
- **Curvas Tonales Dedicadas e Interactivas por Capa (v0.9.0):** Curva tonal spline propia para cada capa (Estrellas, Fondo Nebular Starless, Suelo, Capa Base y máscaras de ajuste) con histograma de fondo armonizado según rango tonal post-estirado y filtrado espacial.
- **Pila de Capas con Miniaturas Dinámicas y Altura Adaptativa (v0.9.0):** Previsualización fotorrealista en vivo de cada capa con selector de tamaños S / M / L (XL), adaptación automática a la relación de aspecto real de la foto (horizontal vs retrato) y redimensionador interactivo `LayerListResizeHandle` con cálculo de altura automático anti-colapso.
- **Deconvolución Óptica Regularizada Richardson-Lucy (v0.9.0 - P10):** Motor numérico en `core/deconvolution.py` con modelado de PSF (Moffat 2D y Gaussiano), estimación empírica automática de PSF desde estrellas no saturadas, deringing perimetral antianillos y amortiguación Bayesiana de ruido de fondo.
- **Modernización de Diálogos de Presets y Ergonomía del Apilador (v0.9.0):** Diálogo de presets adaptado al tema oscuro Pro Suite, prevención de desbordamientos de columna y compactación de pestañas del apilador eliminando botones de scroll.
- **Apilado de Ventana Deslizante para Vídeo Time-Lapse (*Sub-stacking / Rolling Stacker* - v0.8.5):** Pipeline completo en `core/substacking.py` y `Módulo 1: Apilador` para time-lapse nocturno. Homografías encadenadas $O(N)$, buffer FIFO en RAM $O(W)$, ponderación temporal gaussiana/uniforme/triangular, tratamiento dual de suelo (promedio temporal de bajo ruido en trípode fijo o ancla central), fusión anti-ghosting de árboles y exportación a MP4 (1080p/4K) con previsualización en vivo.
- **Filtro Difusor Digital y Realce de Constelaciones (v0.8.4 - P1):** Aislamiento selectivo de estrellas prominentes (Top-Hat), preservación de fondo estelar, recuperación cromática espectral y modos de mezcla no destructivos (*Trama, Luz Suave, Aditivo*).
- **Estirado Hiperbólico Generalizado - GHS (v0.8.4 - P2):** Transformación tonal hiperbólica con punto de simetría (SP), factor de estirado (D) y protección de núcleos estelares (HP) con modos de Luminancia y RGB enlazado.
- **Reducción Paramétrica 2D de Contaminación Lumínica Terrestre (v0.8.4 - P5):** Sustracción de cúpulas lumínicas asimétricas urbanas en el horizonte mediante modelo elíptico 2D con ajuste de centro, anchura, altura y tinte de vapor de sodio.
- **Efecto Orton Astronómico y Glow de Polvo Galáctico (v0.8.4 - P6):** Resplandor etéreo y suave enfocado selectivamente en nubes moleculares y bandas galácticas sin oscurecer estrellas.
- **Control Directo de Overlay de Máscara y Pipeta Muestreadora de Color (v0.8.4):** Botón de visualización rápida en barra superior con respuesta verde/oscura sincronizada, ocultación automática de trazos al generar y pipeta interactiva en capas de color HSV.
- **Manual de Usuario Completo (v0.8.4):** Documentación exhaustiva en `MANUAL_DE_USUARIO.md` explicando cada módulo, flujo de trabajo y parámetros del Revelador.
- **Detección Automática de Suelo con IA Ligera (v0.8.3 - P8):** Inferencia local U²-Net en ONNX Runtime con refinado subpíxel por Filtro Guiado (Guided Filter), descarga asistida interactiva y fallback algorítmico continuo sin conexión.
- **Tratamiento Anti-Ghosting de Árboles en Horizonte (v0.8.3 - P3):** Algoritmo Min-Horizon de proyección percentil inferior sobre zona de seguridad dilatada perimetral que elimina ramas transparentes en el apilado dual.
- **Cribado Previo Inteligente de Tomas y Calidad Estelar (v0.8.3 - P4):** Inspección óptica multihilo de FWHM, excentricidad y densidad estelar con calibración estratificada para grandes angulares, semáforo visual de puntuación y auto-descarte de tomas movidas o nubladas.
- **Corrección de Alineación 1:1 en Zoom del Revelador (v0.8.3):** Eliminación de desplazamiento relativo y restos estelares en el suelo al activar el zoom al 100% mediante sincronización de `crop_rect` en el pipeline de composición por capas.
- **Gestor de Capas y Máscaras de Ajuste Local (v0.8.2):** Integración de capas especiales de imagen (Fondo Base, Fondo Starless, Estrellas, Suelo), orden natural de pila (bottom-up), controles contextuales dedicados por tipo de capa, tematización cromática, memoria de herramientas, ojo de visibilidad y botón de carga de máscaras desde apilado/archivo.
- **Internacionalización i18n Multilingüe en Tiempo Real (v0.8.2):** Arquitectura JSON modular (`locales/es.json`, `locales/en.json`) con gestor `core/i18n.py` y selector de idioma en barra de estado.
- **Acordeón Plegable Inteligente en el Revelador (v0.8.2):** Rediseño modular estilo Lightroom con 9 secciones colapsables y prioridad a StarNet++ AI.
- **Sistema de Diseño Global "Cosmic Dark Pro" (v0.8.2):** Unificación visual de paletas oscuras, tipografías y botones de ejecución en los 7 módulos del sistema con monitor de hardware (RAM).
- **Gestor de Presets y Estilos de Revelado Astronómico (v0.8.2):** Biblioteca con 8 presets oficiales calibrados, protección de estirado nativo y aplicación granular por ámbitos.
- **Pincel de Máscaras y Ajustes Locales:** Pincel continuo con control de dureza y opacidad sin acumulación descontrolada.
- **Herramientas de Balance de Blancos Local y Suelo:** Ajustes con 4 decimales de precisión e interactividad con doble clic para resetear deslizadores.
