# 🌌 AstroLoom

<div align="center">

**Suite y Estación de Trabajo Integral para Astrofotografía de Paisaje, Cielo Profundo y Eventos Astronómicos**

[![Version](https://img.shields.io/badge/version-v0.17.0-blue.svg)](https://github.com/shilmar/AstroLoom/releases)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

[**📖 Manual de Usuario**](MANUAL_DE_USUARIO.md) • [**📝 Registro de Cambios**](CHANGELOG.md) • [**🤝 Contribuir**](CONTRIBUTING.md) • [**🗺️ Hoja de Ruta**](ROADMAP.md)

</div>

---

## 🔭 ¿Qué es AstroLoom?

**AstroLoom** es una suite de posprocesado astronómico no destructivo diseñada específicamente para fotógrafos nocturnos, astrofotógrafos de paisaje (*nightscapes*), cazadores de eclipses y amantes de las circumpolares y panorámicas de gran campo.

Impulsada por una arquitectura híbrida que combina una interfaz gráfica reactiva en **Python (`PySide6`)** con un potente motor de cómputo nativo en **Rust (`astroloom_rs`)**, AstroLoom ofrece máxima fidelidad matemática en coma flotante de 32 bits con renderizado a más de 160 FPS y un consumo mínimo de memoria RAM.

---

## ✨ Características Principales

### ⚡ Motor Híbrido Rust de Ultra-Alto Rendimiento (`astroloom_rs`)
- **Renderizado Directo a Memoria Qt:** Volcado en búfer contiguo plano `BGRA8888` a >160 FPS (~6 ms en 4K), sin copias intermedias en Python ni recolección de basura (*GC*).
- **Kernels Vectorizados SIMD & Rayon:** Apilado Kappa-Sigma ($2.77\times$ más rápido que Numba JIT), interpolación elástica TPS con `nalgebra`, y modos de fusión de capas a máxima velocidad de CPU.
- **Calibración RAW In-Place:** Calibración matemática pura pre-debayer directamente sobre la matriz de fotodiodos del RAW (0 bytes de memoria RAM adicional).
- **E/S y Códecs Nativos:** Lectores/escritores nativos en Rust para FITS (<15 ms en 60 MP), PSD multicapa de 16 bits (sin dependencias de `pytoshop`), e ingesta multihilo de cabeceras RAW.

### 🧩 Módulos Especializados Integrados

| Módulo | Descripción y Capacidades Clave |
| :--- | :--- |
| **1. Apilador (Stacker)** | Apilado dual cielo/suelo con alineación estelar subpíxel, corrección elástica TPS para gran angular, segmentación semántica de horizonte con IA (U²-Net) y rechazo Kappa-Sigma con relleno reflectivo `BORDER_REFLECT`. |
| **2. Revelador / Editor** | Procesado no destructivo de 32 bits: Estirado Hiperbólico (GHS), separación de estrellas por IA (StarNet++ v2 DirectML), curvas tonales por capa, supresión de contaminación lumínica (RBF 2D y modelo elíptico), balance fotométrico Gaia (PCC) y paletas Dual-Band OSC. |
| **3. Eclipses (HDR)** | Fusión HDR de series de bracketing solar/lunar con traspaso gradual anti-anillos. Arsenal coronal completo (Steerable G2, Larson-Sekanina, SWAP, WOW Bilateral, NAFE, NRGF, MGN) y exportación a PSD multicapa de 16 bits. |
| **4. Eclipses (Collage)** | Montaje cronológico automatizado de fases sobre paisajes con trayectorias geométricas (diagonal, arco, romboide), espaciado regular y fusión transparente. |
| **5. Startrails** | Circumpolares continuas o efecto cometa en streaming $O(1)$, supresión automática de satélites y aviones, relleno de pausas (*gap filling*), recuperación cromática de estrellas y exportación de vídeo 4K. |
| **6. Lluvias de Meteoros** | Detección algorítmica de bólidos, cribado en galería interactiva, filtros anti-aviones/satélites y reproyección al radiante celeste con flujo neto. |
| **7. Panorámicas (Mosaicos)** | Cosido 2D multifila de ultra-alta resolución, alineación por asterismos en Rust (<2 ms), fusión híbrida *Dual-Zone* (árboles nítidos sin efecto fantasma) y auto-recorte rectangular $O(H \times W)$. |

---

## 🚀 Novedades de la Versión v0.17.0

* **Purga Inteligente Anti-Satélites por Transformada de Hough (Startrails):** Detección geométrica lineal rápida a escala 50% con evaluación monocanal de alta velocidad; suprime con precisión estelas de satélites artificiales y aviones en circumpolares preservando las trazas estelares naturales.
* **Integración de Tecnologías Seti Astro en el Revelador:** Estirado Estadístico Adaptativo (Statistical Stretch) basado en MAD, Claridad de Textura (Texture Clarity) para realce de micro-contraste en nebulosas y polvo galáctico, Extracción Automática de Fondo (ABE) mediante superficie polinomial 2D y modulación fotométrica de capas.
* **Ondículas Multiescala (Wavelets B3-Spline):** Panel de enfoque ampliado con descomposición frecuencial multiescala: deslizador de *Estructura* para enfatizar filamentos y estructuras medias sin levantar ruido, y *Atenuar Fondo* para profundizar los negros cósmicos.
* **Pipeline de Exportación Zero-Copy en Rust e In-Place:** Grafo de capas evaluado en registros nativos de CPU en Rust (`astroloom_rs`), eliminando copias intermedias de memoria y volcando a enteros `uint16` in-place para una exportación ultrarrápida de proyectos `.devloom` y archivos PSD multicapa de 16 bits.
* **Barra de Progreso Unificada en Lienzo (`CanvasFooterBar`):** Estandarización ergonómica transversal en todos los visores y módulos (Revelador, Circumpolares, Meteoros, Panorámicas, Eclipses, Collage y Apilador), con seguimiento porcentual en tiempo real y cancelación reactiva inmediata.
* **Visor de Inspección Secuencial (*Blink Viewer*):** Inspección rápida fotograma a fotograma de series temporales con ajuste de cadencia (FPS), zoom/pan sincronizado y descarte ágil de tomas con nubes o defectos.
* **Discriminación de Balizas Aeronáuticas y Resalte Multitrazas (Meteoros):** Detección inteligente de luces de navegación y balizas intermitentes para evitar falsos positivos, junto a previsualización global sincronizada de múltiples trazas sobre la imagen compuesta.

---

## 🌟 Novedades de la Versión v0.16.0

* **Máscara de Enfoque y Nitidez (Sharpening / USM):** Nuevo motor de microcontraste y altas frecuencias espaciales en el Revelador con controles de *Cantidad*, *Radio*, *Detalle* y *Máscara* de bordes estilo Lightroom, incluyendo previsualización monocroma interactiva (`Alt`-drag) y restricción por ámbito (`Solo Suelo`, `Cielo sin Estrellas` o `Global`).
* **Purga Integral de Numba y Unificación en Rust Rayon:** Eliminación definitiva de dependencias LLVM/Numba con reducción neta de 2.36 s en el arranque en frío, LUT display mapping a >4.000 FPS en Rust y cálculo de histograma multihilo en OpenCV C++.
* **Aceleración Nativa en Rust para Eclipses:** Centrado de disco solar (`center_solar_disc_rs`), estirado Asinh Luma preservando color y fusión coronal HDR adaptativa multiescala en paralelo.
* **Resiliencia de Hardware y Caché de Búferes LRU:** Fallback en caliente GPU $\rightarrow$ CPU ante saturación de VRAM, caché persistente de grafos ONNX y presupuesto estricto de memoria RAM en el pipeline para evitar fugas al procesar archivos 4K/8K.
* **Cabecera de Telemetría y Supresión de Barra de Estado:** Monitor de RAM en tiempo real (% y GB), alerta de degradación de hardware y acceso al Registro de Actividad ubicados directamente en la esquina superior de pestañas, ganando valiosa superficie vertical de lienzo.
* **Galería Visual de Presets:** Panel rediseñado por tarjetas interactivas y guardado seguro que excluye máscaras locales de usuario para prevenir colisiones.

---

## 🎨 Compositor Multicapa, Máscaras Vectoriales y Transformación 2D (v0.15.0 / v0.15.1)

AstroLoom integra un sistema integral de composición profesional no destructiva con capas ráster y de ajuste:
* **Máscaras Vectoriales Analíticas en Rust:** Filtros de **Gradiente Lineal** (polución lumínica, viñeteo) y **Gradiente Radial / Elíptico** (centro galáctico, nebulosas) con gizmos interactivos directos sobre el lienzo a 60 FPS y restricción espacial automática a Cielo o Suelo (*Edge-Snapping*).
* **Pincel Acelerado en Rust y Dirty-Rects a 60–144 FPS (v0.15.1):** Estampado continuo interpolado mediante `stamp_brush_stroke_rs` y actualización selectiva *in-place* por rectángulos sucios (*dirty rects*) en el lienzo, eliminando lag y tirones de GPU/CPU al pintar sobre imágenes de alta resolución.
* **Curvas Selectivas Hue/Luma Fusionadas en Rust (v0.15.1):** Evaluación en registros paralelos (`apply_hue_luma_curves_fused_rs`) más de $10\times$ más rápida al prescindir de conversiones de color intermedias.
* **Transformación Espacial 2D de Capas Ráster:** Desplazamiento interactivo en el lienzo, escalado (con rueda de ratón o parámetros numéricos) y rotación libre para capas ráster importadas.
* **Miniaturas Duales Estilo Pro:** Edición independiente de la imagen o de su máscara con un solo clic.
* **Gestor Avanzado de Presets:** Modulación continua de intensidad (0% a 150%), previsualización en tiempo real y detección automática de dependencias (StarNet++ y máscara de horizonte).
* **Reducción de Ruido IA Multiescala:** NAFNet optimizado con descomposición de croma en dos escalas (1.0x luminancia, 0.5x color YCrCb), pre-estirado analítico MTF invertible para preservar fotometría lineal y caché de sesiones ONNX.
* **Modos de Fusión Profesionales y Exportación PSD:** *Normal*, *Trama (Screen)*, *Luminosidad*, *Luz Suave*, *Superponer* y *Multiplicar* en Rust Rayon; exportación completa a PSD de 16 bits preservando transformaciones espaciales y máscaras.

---

## 🚀 Instalación y Puesta en Marcha

### Arranque Rápido en Windows
Haz **doble clic en `astroloom.bat`** en la carpeta principal del proyecto. El script detectará tu entorno virtual Python o la instalación del sistema y lanzará la interfaz sin consolas secundarias.

### Instalación Manual (Todas las Plataformas)

1. **Clonar el Repositorio:**
   ```bash
   git clone https://github.com/shilmar/AstroLoom.git
   cd AstroLoom
   ```

2. **Instalar Dependencias de Python:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Compilar el Motor Nativo Rust (Recomendado para Máximo Rendimiento):**
   ```bash
   pip install maturin
   cd astroloom_rs
   maturin develop --release
   cd ..
   ```
   *(Si no se compila el módulo nativo, AstroLoom activará automáticamente su modo de reserva transparente en Python/NumPy sin interrumpir el funcionamiento).*

4. **Aceleración por GPU (Opcional - NVIDIA CUDA):**
   ```bash
   # Para CUDA 12.x:
   pip install cupy-cuda12x
   ```

5. **Iniciar la Aplicación:**
   ```bash
   python run_app.py
   ```

---

## 🗺️ Flujo de Trabajo Recomendado

```mermaid
flowchart LR
    A["1. Captura RAW"] --> B["2. Apilador (Cielo/Suelo)"]
    B --> C["3. Revelador (GHS + StarNet AI)"]
    C --> D["4. Capas y Curvas Tonales"]
    D --> E["5. Exportar (TIFF 16/32-bit o PSD)"]
```

> 📖 **Para una guía detallada paso a paso de cada módulo, parámetros y atajos de teclado, consulta el [Manual de Usuario Oficial](MANUAL_DE_USUARIO.md).**

---

## 🏛️ Estructura del Código

```text
AstroLoom/
├── astroloom_rs/            # Núcleo nativo de alto rendimiento en Rust (PyO3 + Rayon + SIMD)
│   ├── src/                 # Kernels de apilado, RBF, blend modes, deconv, FITS, PSD...
│   └── Cargo.toml           # Configuración del paquete nativo
├── core/                    # Lógica astronómica en Python (calibración, AI, máscaras, PCC...)
├── gui/                     # Interfaz gráfica moderna en PySide6 (Qt)
│   ├── developer/           # Pipeline de renderizado y estado del revelador
│   ├── widgets/             # Lienzo interactivo, gestor de capas, curvas, histograma...
│   └── tab_*.py             # Pestañas de Apilador, Revelador, Eclipses, Startrails, Panos...
├── locales/                 # Soporte multilingüe (español e inglés)
├── MANUAL_DE_USUARIO.md     # Guía detallada y completa para usuarios
├── CHANGELOG.md             # Historial de versiones y cambios
└── run_app.py               # Punto de entrada de la aplicación
```

---

## 🤝 Contribución y Código de Conduct

Las contribuciones son bienvenidas. Si deseas colaborar:
1. Revisa nuestra [Guía de Contribución](CONTRIBUTING.md).
2. Cumple con el [Código de Conducta](CODE_OF_CONDUCT.md).
3. Asegúrate de que las pruebas automatizadas pasen (`pytest`).

---

## 📜 Reconocimientos

AstroLoom integra ideas y algoritmos inspirados en investigaciones pioneras de la comunidad astrofotográfica y científica:
- **[Agentic-Eclipse-Library](https://github.com/PeterWar/Agentic-Eclipse-Library)** por Pere Guerra (PeterWar): Algoritmos de fusión coronal HDR, normalización y filtrado de eclipses solares.
- **[StarNet++](https://www.starnetastro.com/)** por Nikita Misiura: Red neuronal pionera para el desacoplo de estrellas.
- **Miloslav Druckmüller**: Fundamentos matemáticos de filtrado multiescala (MGN) y ecualización adaptativa difusa (NAFE).

---

## 📄 Licencia

Este proyecto está bajo la Licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
