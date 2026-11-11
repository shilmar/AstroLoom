# Guía de Contribución a AstroLoom 🌌

¡Gracias por tu interés en contribuir a **AstroLoom**! AstroLoom es una suite de posprocesado, apilado y revelado para astrofotografía de paisaje, circumpolares, mosaicos y eclipses, construida sobre una arquitectura híbrida de alto rendimiento (**Python/PySide6** + **motor nativo en Rust con PyO3**).

Esta guía resume el flujo de trabajo recomendado para reportar problemas, proponer mejoras y configurar tu entorno de desarrollo local.

---

## 📋 Tabla de Contenidos

- [Código de Conducta y Principios](#código-de-conducta-y-principios)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
  - [Reportar Errores (Bugs)](#reportar-errores-bugs)
  - [Proponer Nuevas Funcionalidades](#proponer-nuevas-funcionalidades)
- [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
  - [Requisitos Previos](#requisitos-previos)
  - [1. Clonar el Repositorio](#1-clonar-el-repositorio)
  - [2. Configurar el Entorno Virtual de Python](#2-configurar-el-entorno-virtual-de-python)
  - [3. Compilar el Motor Nativo en Rust (`astroloom_rs`)](#3-compilar-el-motor-nativo-en-rust-astroloom_rs)
  - [4. Ejecutar AstroLoom](#4-ejecutar-astroloom)
- [Estructura del Proyecto y Arquitectura](#estructura-del-proyecto-y-arquitectura)
- [Ejecución de Pruebas (Tests)](#ejecución-de-pruebas-tests)
- [Estilo y Buenas Prácticas](#estilo-y-buenas-prácticas)
- [Flujo de Pull Requests (PR)](#flujo-de-pull-requests-pr)

---

## Código de Conducta y Principios

Al participar en este proyecto, te comprometes a fomentar un entorno colaborativo, respetuoso e inclusivo.

Nuestros principios técnicos fundamentales son:
1. **Rigor Fotométrico y Calidad Científica:** El tratamiento de datos debe preservar la linealidad y el rango dinámico siempre que sea posible.
2. **Eficiencia y Alto Rendimiento:** AstroLoom procesa archivos RAW de alta resolución y series pesadas. Las operaciones computacionales críticas pertenecen al motor nativo en Rust (`astroloom_rs`) y deben minimizar copias en memoria RAM.
3. **UI Reactiva y Ergonomía:** La interfaz nunca debe congelarse durante cálculos pesados; los procesos intensivos se delegan a hilos de trabajo independientes (`QThread` / `Worker`).

---

## ¿Cómo Puedo Contribuir?

### Reportar Errores (Bugs)

Antes de abrir una incidencia (*Issue*):
- Revisa las [Issues existentes](https://github.com/shilmar/AstroLoom/issues) para verificar si el problema ya fue reportado.
- Asegúrate de estar probando sobre la rama principal (`main`) o la última versión publicada.

Al crear el reporte, incluye:
- **Título claro y conciso.**
- **Pasos exactos para reproducir el fallo.**
- **Comportamiento esperado vs. obtenido.**
- **Capturas de pantalla o fragmentos del registro/traza de error (traceback).**
- **Datos del entorno:** Sistema Operativo (Windows/Linux/macOS), versión de Python, modelo de GPU y tipo de archivos (ej. RAW Canon CR3, Nikon NEF, Sony ARW, TIFF 16-bit).

### Proponer Nuevas Funcionalidades

Si deseas proponer un nuevo algoritmo astronómico o una herramienta en la interfaz:
1. Abre una Issue de tipo **Feature Request** o inicia una discusión antes de escribir código grande.
2. Explica el caso de uso práctico en astrofotografía y por qué beneficia a la comunidad.
3. Si existe bibliografía científica o algoritmos de referencia (ej. papers de FWHM, sustracción de gradientes RBF, deconvolution Richardson-Lucy), menciónalos.

---

## Configuración del Entorno de Desarrollo

### Requisitos Previos

- **Git** instalado.
- **Python 3.10 o superior** (64-bit).
- **Rust Toolchain (rustc, cargo)** estable: [instalar vía rustup](https://rustup.rs/).

### 1. Clonar el Repositorio

```bash
git clone https://github.com/shilmar/AstroLoom.git
cd AstroLoom
```

### 2. Configurar el Entorno Virtual de Python

En Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install maturin pytest
```

En Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install maturin pytest
```

### 3. Compilar el Motor Nativo en Rust (`astroloom_rs`)

AstroLoom utiliza `maturin` para generar los bindings Python (`PyO3`) del núcleo en Rust:

```bash
cd astroloom_rs
maturin develop --release
cd ..
```

> 💡 **Nota:** La bandera `--release` activa las optimizaciones LTO y SIMD en Rust, indispensables para benchmarks y procesamiento en tiempo real.

### 4. Ejecutar AstroLoom

Una vez compilado `astroloom_rs` y con el entorno virtual activo:

```bash
python run_app.py
```

En Windows también puedes utilizar el lanzador local:
```powershell
.\astroloom.bat
```

---

## Estructura del Proyecto y Arquitectura

- **`core/`**: Motores de cálculo en Python, orquestación de tareas, algoritmos de fallback y pipeline de calibración/apilado.
- **`gui/`**: Interfaz gráfica construida en **PySide6 (Qt)**: lienzos de visualización, controles deslizantes, gestor de capas y temas visuales.
- **`astroloom_rs/`**: Motor nativo en Rust de alto rendimiento.
  - `src/stacking.rs`: Integración Kappa-Sigma, percentiles y medianas multihilo con Rayon.
  - `src/streaming_io.rs`: Calibración matemática *in-place* sobre mosaicos CFA sin copias intermedias.
  - `src/photometry.rs`: Calibración fotométrica estelar (PCC) contra catálogos.
  - `src/astrometry.rs`: Detección de estrellas y resolución de mallas Thin Plate Splines (TPS).
  - `src/developer.rs`: Pipeline de revelado tonal fusionado a 60 FPS.
- **`tests/`**: Suite de tests automatizados unitarios y de integración.
- **`docs/`**: Documentación técnica, manual de usuario y especificaciones.

---

## Ejecución de Pruebas (Tests)

Antes de enviar cualquier cambio, verifica que todos los tests pasen exitosamente:

```bash
# Ejecutar suite completa
pytest tests/

# Ejecutar tests de integración del motor Rust
pytest tests/test_rust_integration.py

# Ejecutar un test específico en modo detallado
pytest tests/test_cfa_calibration.py -v
```

Si realizaste cambios dentro de `astroloom_rs/`, asegúrate de verificar el formato y linting de Rust:

```bash
cd astroloom_rs
cargo fmt --check
cargo clippy -- -D warnings
cd ..
```

---

## Estilo y Buenas Prácticas

- **Python:**
  - Seguir directrices **PEP 8**.
  - Mantener nombres claros en funciones y variables astronómicas (ej. `fwhm`, `bkg_rms`, `cfa_pattern`).
  - Añadir anotaciones de tipo (*type hints*) en nuevas funciones de `core/`.
- **Rust:**
  - Código idiomático y documentado con `rustfmt`.
  - Evitar reservas innecesarias en el heap (`Vec::new()` dentro de bucles de píxeles); aprovechar buffers prealocados o slices paralelos con `rayon`.
  - Control de pánico: preferir retornar `Result<T, PyErr>` o fallbacks controlados en vez de `unwrap()` en APIs expuestas a Python.
- **Rendimiento:**
  - Si una operación matemática sobre matrices tarda más de unos pocos milisegundos en imágenes de 24+ MP, evalúa implementarla en `astroloom_rs`.

---

## Flujo de Pull Requests (PR)

1. **Crea una rama para tu cambio:**
   ```bash
   git checkout -b feature/nombre-de-tu-mejora
   # o para corregir un fallo:
   git checkout -b fix/descripcion-del-bug
   ```
2. **Realiza tus cambios y haz commits semánticos:**
   Usa mensajes descriptivos (siguiendo [Conventional Commits](https://www.conventionalcommits.org/)):
   - `feat: añadir soporte para paleta de color personalizada en revelador`
   - `fix: corregir desbordamiento en sustracción de darks en uint16`
   - `perf: optimizar detección de centroides estelares en Rust`
   - `docs: actualizar parámetros en el manual de usuario`
3. **Verifica tus cambios:**
   - Añade un test unitario si creaste una nueva funcionalidad o corregiste un bug.
   - Ejecuta `pytest tests/`.
4. **Envía el Pull Request:**
   - Explica con claridad el motivo del cambio y qué archivos toca.
   - Incluye imágenes comparativas "Antes / Después" si tu cambio afecta al resultado visual o a la interfaz gráfica.
   - Vincula la *Issue* correspondiente (ej. `Closes #42`).

---

¡Muchas gracias por ayudar a hacer de **AstroLoom** una herramienta cada vez más potente para toda la comunidad de astrofotografía! 🔭✨
