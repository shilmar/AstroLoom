# 🗺️ ROADMAP / TAREAS PENDIENTES — ASTROLOOM

Este documento recopila las propuestas de mejora, nuevas características y tareas pendientes planificadas para AstroLoom.

---

## 📌 Tabla de Prioridades

| Prioridad | Tarea / Módulo | Dificultad (1-10) | Impacto (1-10) | Estado | Descripción |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **P1** | **Internacionalización (i18n: Español / Inglés)** | 4 / 10 | 8 / 10 | ✅ Completado (v0.8.2) | Sistema de traducción modular mediante ficheros (`locales/es.json`, `locales/en.json`) con gestor `core/i18n.py`, selector de idioma en interfaz y persistencia en configuración. |
| **P2** | **Control de Calidad (Frame Culling)** | 6 / 10 | 6 / 10 | ⏳ Pendiente | Medición automática previa de FWHM, excentricidad (deriva/viento) y SNR para descartar tomas defectuosas antes del apilado. |
| **P3** | **Persistencia Avanzada de Configuración** | 3 / 10 | 5 / 10 | ⏳ Pendiente | Centralización y sincronización de preferencias de usuario y rutas recientes en `config.json` o `QSettings`. |
| **P4** | **Deconvolución Estelar (FWHM / Richardson-Lucy)** | 7 / 10 | 8 / 10 | 🔍 En estudio | Algoritmo regularizado para afinar perfiles estelares y recuperar microcontraste en nebulosas y detalles galácticos. |

---

## 🌐 Diseño Propuesto: Internacionalización (i18n)

### 1. Formato de Ficheros de Traducción
- Ficheros **JSON** ligeros y legibles organizados en el directorio `locales/`:
  - `locales/es.json` (Español - Idioma base / nativo)
  - `locales/en.json` (Inglés)
- **Ventajas frente al formato binario de Qt (`.ts` / `.qm`):**
  - Totalmente legible y editable directamente en cualquier editor sin herramientas externas (como *Qt Linguist*).
  - Integración nativa con control de versiones (Git diffs claros).
  - Permite que colaboradores añadan nuevos idiomas con solo duplicar y traducir un fichero JSON.
  - Soporte de fallback automático: si una clave falta en inglés, recurre de forma transparente al texto en español.

### 2. Arquitectura del Motor (`core/i18n.py`)
- Gestor `I18nManager` singleton con función global `tr(key, default="")`:
  ```python
  from core.i18n import tr
  
  # Ejemplo de uso en la interfaz:
  btn_export.setText(tr("developer.export_button", "💾 EXPORTAR IMAGEN REVELADA..."))
  ```
- Soporte para interpolación de variables: `tr("log.saved_file", "Guardado en {path}", path=filepath)`.

### 3. Persistencia y Selección
- Entrada en `config.json`:
  ```json
  {
      "language": "es"
  }
  ```
- Selector de idioma accesible en la barra superior o menú de configuración de la ventana principal.

### 4. Estrategia de Implementación Progresiva
1. **Fase 1 (Infraestructura base):** Módulo `core/i18n.py`, ficheros `locales/es.json` y `locales/en.json`, y parámetro en `config.json`.
2. **Fase 2 (Estructura principal):** Nombres de pestañas, barras de herramientas, títulos de acordeón y botones de acción principal.
3. **Fase 3 (Controles y Ajustes):** Deslizadores, etiquetas de balance de blancos, modos de color y menús contextuales.
4. **Fase 4 (Textos descriptivos):** Tooltips de ayuda, mensajes informativos y descripciones de presets.

---

## ✅ Tareas Recientemente Completadas
- **Gestor de Capas y Máscaras de Ajuste Local (v0.8.2):** Integración de capas especiales de imagen (Fondo Base, Fondo Starless, Estrellas, Suelo), orden natural de pila (bottom-up), controles contextuales dedicados por tipo de capa, tematización cromática, memoria de herramientas, ojo de visibilidad y botón de carga de máscaras desde apilado/archivo.
- **Internacionalización i18n Multilingüe en Tiempo Real (v0.8.2):** Arquitectura JSON modular (`locales/es.json`, `locales/en.json`) con gestor `core/i18n.py` y selector de idioma en barra de estado.
- **Acordeón Plegable Inteligente en el Revelador (v0.8.2):** Rediseño modular estilo Lightroom con 9 secciones colapsables y prioridad a StarNet++ AI.
- **Sistema de Diseño Global "Cosmic Dark Pro" (v0.8.2):** Unificación visual de paletas oscuras, tipografías y botones de ejecución en los 7 módulos del sistema con monitor de hardware (RAM).
- **Gestor de Presets y Estilos de Revelado Astronómico (v0.8.2):** Biblioteca con 8 presets oficiales calibrados, protección de estirado nativo y aplicación granular por ámbitos.
- **Pincel de Máscaras y Ajustes Locales:** Pincel continuo con control de dureza y opacidad sin acumulación descontrolada.
- **Herramientas de Balance de Blancos Local y Suelo:** Ajustes con 4 decimales de precisión e interactividad con doble clic para resetear deslizadores.
