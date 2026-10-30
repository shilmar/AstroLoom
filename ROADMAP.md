# 🗺️ ROADMAP / TAREAS PENDIENTES — ASTROLOOM

Este documento recopila las propuestas de mejora, nuevas características y tareas pendientes planificadas para **AstroLoom**, inspiradas en los estándares de la industria y herramientas especializadas de astrofotografía de paisaje nocturno (*Nebulb*, *Sequator*, *Starry Landscape Stacker*, *Siril*, *PixInsight*).

> ℹ️ *Nota: El historial de todas las funcionalidades ya completadas e integradas en versiones publicadas se encuentra detallado cronológicamente en [`CHANGELOG.md`](CHANGELOG.md).*

---

## 📌 Tabla de Prioridades (Tareas Pendientes)

| Prioridad | Tarea / Funcionalidad | Módulo | Inspiración | Dificultad (1-10) | Impacto (1-10) | Estado |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **P1** | **Reducción de Ruido con IA Ligera (Astro Denoise Fase 2 - ONNX / NAFNet)** | Revelador | NoiseXTerminator / DeepSNR | 6.5 / 10 | 9.0 / 10 | ✅ Completado (v0.11.1) |
| **P2** | **Revisión y Optimización del Módulo de Panorámicas y Mosaicos** | Panorámicas | PTGui / Hugin / Lightroom | 6.0 / 10 | 8.5 / 10 | ✅ Completado (v0.11.6) |
| **P3** | **Corrección de Distorsión Óptica Periférica de Estrellas** | Apilador | Sequator (*Distort Mode*) | 7.5 / 10 | 8.5 / 10 | 🔍 En estudio |
| **P4** | **Extracción de Gradientes por Puntos de Muestreo Interactivos** | Revelador | PixInsight (*DBE*) / GraXpert | 5.0 / 10 | 8.0 / 10 | ⏳ Planificado |
| **P5** | **Procesado y Exportación en Lote (Batch Processing)** | Revelador / General | Lightroom / Adobe Camera Raw | 4.0 / 10 | 7.5 / 10 | ⏳ Pendiente |
| **P6** | **Persistencia Avanzada y Sincronización de Preferencias** | Sistema | - | 3.0 / 10 | 5.0 / 10 | ⏳ Pendiente |

---

## 🚀 Desglose Detallado de Funcionalidades Pendientes

### 1. Revelador y Reducción de Ruido

#### 🔹 1.1 Reducción de Ruido con IA Ligera para Cielo Profundo y Fondo (Astro Denoise Fase 2)
* **Herramientas de referencia:** NoiseXTerminator, DeepSNR, Lightroom Denoise AI.
* **Descripción técnica:** Modelo de red neuronal convolucional o Transformer ligero inferido localmente con ONNX Runtime (compatible con CPU y GPU/DirectML):
  - Entrenado específicamente sobre imágenes astronómicas lineales y no lineales para distinguir entre ruido estocástico de lectura/fotónico y filamentos tenues de nebulosa o polvo galáctico.
  - Aplicación directa sobre la capa *Starless* o fondo nebular para prevenir alteraciones en los perfiles de PSF de las estrellas.
  - Control de preservación de micro-contraste y textura de grano fino mediante control deslizante de porcentaje.
* **Impacto:** **9.0 / 10** *(Calidad de reducción de ruido superior sin pérdida de resolución en estructuras astronómicas).*
* **Dificultad / Coste:** **6.5 / 10** *(Media-Alta: Integración y empaquetado de pesos ONNX optimizados en ejecución local).*
* **Estado:** **✅ Completado en v0.11.1.** Implementado mediante la arquitectura convolucional NAFNet (*Nonlinear Activation Free Network*) ejecutada con ONNX Runtime multihilo (`core/ai_denoise.py`), permitiendo filtrar selectivamente el fondo nebular y terreno terrestre con máxima preservación estelar.

#### 🔹 1.2 Extracción de Gradientes de Fondo por Muestreo Interactivo (*Sample-Point Background Neutralization*)
* **Herramientas de referencia:** PixInsight (*DynamicBackgroundExtraction - DBE*), GraXpert.
* **Descripción técnica:** Complemento interactivo al modelo polinómico 2D actual:
  - Generación automática o manual de una rejilla de muestras de cielo libre de nebulosas y estrellas con exclusión del suelo terrestre.
  - Interpolación matemática multiescala 2D mediante RBF (*Radial Basis Functions*) o Splines Bivariadas para eliminar gradientes complejos de polución lumínica local irregular no modelables por polinomios de grado 2.
* **Impacto:** **8.0 / 10** *(Solución definitiva para cielos con múltiples cúpulas urbanas o gradientes atmosféricos caóticos).*
* **Dificultad / Coste:** **5.0 / 10** *(Media: UI interactiva de puntos de ancla en el lienzo y ajuste RBF con Scipy).*

---

### 2. Motor de Alineación y Apilado Especializado

#### 🔹 2.1 Corrección de Distorsión Óptica Periférica de Estrellas (*Reduce Star Distortion*)
* **Herramientas de referencia:** Sequator (*Complex / Distort Mode*).
* **Descripción técnica:** En objetivos gran angular (14mm–24mm a $f/1.8$–$f/2.8$), las aberraciones geométricas (coma, astigmatismo y distorsión de barril/bigote) varían radialmente. Al rotar el cielo durante sesiones de 30–60 minutos, las estrellas transitan por diferentes zonas de distorsión, impidiendo que una transformación afín u homográfica global logre coincidencia perfecta en las 4 esquinas:
  - Registro elástico local basado en deformación por mallas de interpolación (*Thin Plate Splines* - TPS) o corrección radial por cuadrantes.
* **Impacto:** **8.5 / 10** *(Diferencia crítica de calidad entre apilados estándar y profesionales en esquinas).*
* **Dificultad / Coste:** **7.5 / 10** *(Alta: Requiere modelado matemático de deformación no rígida subpíxel).*

---

### 3. Flujo de Trabajo y Automatización

#### 🔹 3.1 Procesado y Exportación en Lote (*Batch Processing & Preset Automation*)
* **Herramientas de referencia:** Adobe Lightroom Classic, Capture One.
* **Descripción técnica:** Aplicación masiva de ajustes de revelado y presets sobre colecciones de fotos:
  - Cola de exportación en segundo plano con selección de formato (TIFF 16-bit, JPEG alta calidad, compresión sRGB/AdobeRGB).
  - Aplicación de recetas de revelado (GHS, curvas, balance de blancos, denoise y estirado) a series de imágenes de paisaje nocturno.
* **Impacto:** **7.5 / 10** *(Ahorro drástico de tiempo para fotógrafos que regresan de expediciones con decenas de composiciones).*
* **Dificultad / Coste:** **4.0 / 10** *(Media: Motor de renderizado desacoplado de la interfaz gráfica).*

---

### 4. Configuración y Sistema

#### 🔹 4.1 Persistencia Avanzada y Sincronización de Preferencias
* **Descripción técnica:** Centralización en `config.json` / `QSettings` de parámetros frecuentes de herramientas:
  - Memorización de radios y tolerancias por defecto en máscaras, directorios de trabajo recientes, geometrías de ventana y configuraciones de rendimiento.
* **Impacto:** **5.0 / 10**
* **Dificultad / Coste:** **3.0 / 10**

---

### 5. Panorámicas y Mosaicos Gigapíxel

#### 🔹 5.1 Revisión y Optimización del Módulo de Panorámicas / Mosaicos
* **Herramientas de referencia:** PTGui Pro, Hugin, Lightroom Panorama Stitcher.
* **Descripción técnica:** Auditoría y modernización integral de `gui/tab_panorama.py` y `core/panorama.py`:
  - Fusión Dual-Zone de precisión (desacoplamiento de silueta terrestre y bóveda celeste para evitar solapamientos fantasma).
  - Optimización de proyección cilíndrica/esférica y búsqueda multiescala de puntos de unión (*control points*) con emparejamiento estelar robusto.
  - Corrección de costuras de exposición y balance fotométrico entre teselas consecutivas en 32 bits lineales.
* **Impacto:** **8.5 / 10** *(Módulo actualmente identificado como básico; su evolución equiparará AstroLoom a las mejores herramientas dedicadas de mosaico nocturno).*
* **Dificultad / Coste:** **6.0 / 10** *(Media-Alta).*
* **Estado:** **✅ Completado en v0.11.6.** Implementada la canalización de doble fase con modo ráfaga y pre-apilado multihilo por tesela (trípode y star tracker), registro guiado por asterismos y triángulos estelares con k-d tree, desacoplamiento cielo/suelo con auto-máscara IA y anti-ghosting, compensación automática de viñeteo, control de curvatura de arco y persistencia completa de sesiones `.astropano`.
