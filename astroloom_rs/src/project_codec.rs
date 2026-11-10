// astroloom_rs/src/project_codec.rs
/*
Módulo de serialización y deserialización paralela de proyectos .devloom (Fase 14).
Permite empaquetar y desempaquetar el manifiesto JSON, máscaras 16-bit y capas
StarNet / Base en paralelo utilizando todos los núcleos del procesador con Rayon,
PNG 16-bit nativo y contenedor ZIP estándar.
*/
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Write};

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use png::{BitDepth, ColorType, Decoder, Encoder};

/// Estructura intermedia que almacena los datos decodificados de una imagen.
pub struct DecodedImage {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f32>,
}

/// Codifica un slice de float32 normalizado [0.0, 1.0] a formato PNG de 16 bits en memoria.
pub fn encode_png_16bit(
    data: &[f32],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut encoder = Encoder::new(&mut out, width, height);
    match channels {
        1 => {
            encoder.set_color(ColorType::Grayscale);
        }
        3 => {
            encoder.set_color(ColorType::Rgb);
        }
        4 => {
            encoder.set_color(ColorType::Rgba);
        }
        _ => return Err(format!("Número de canales no soportado para PNG: {}", channels)),
    }
    encoder.set_depth(BitDepth::Sixteen);
    // Compresión rápida optimizada para interfaz interactiva
    encoder.set_compression(png::Compression::Fast);

    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("Error en cabecera PNG: {}", e))?;

    let num_samples = width as usize * height as usize * channels;
    if data.len() < num_samples {
        return Err(format!(
            "Búfer de datos insuficiente: esperados {}, obtenidos {}",
            num_samples,
            data.len()
        ));
    }

    let mut raw_bytes = vec![0u8; num_samples * 2];
    for (i, &val) in data[..num_samples].iter().enumerate() {
        let u = (val.clamp(0.0, 1.0) * 65535.0).round() as u16;
        let be = u.to_be_bytes();
        raw_bytes[i * 2] = be[0];
        raw_bytes[i * 2 + 1] = be[1];
    }

    writer
        .write_image_data(&raw_bytes)
        .map_err(|e| format!("Error escribiendo datos PNG: {}", e))?;
    drop(writer);
    Ok(out)
}

/// Decodifica un flujo binario PNG a float32 normalizado en [0.0, 1.0].
pub fn decode_png_to_f32(raw_png: &[u8]) -> Result<(usize, usize, usize, Vec<f32>), String> {
    let decoder = Decoder::new(Cursor::new(raw_png));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("Error leyendo información de PNG: {}", e))?;

    let info = reader.info().clone();
    let width = info.width as usize;
    let height = info.height as usize;
    let bit_depth = info.bit_depth;
    let color_type = info.color_type;

    let channels = match color_type {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => return Err("PNG indexado no soportado".to_string()),
    };

    let bytes_per_sample = match bit_depth {
        BitDepth::Sixteen => 2,
        BitDepth::Eight => 1,
        _ => return Err(format!("Profundidad de bits no soportada: {:?}", bit_depth)),
    };

    let total_bytes = reader
        .output_buffer_size()
        .unwrap_or(width * height * channels * bytes_per_sample);
    let mut buf = vec![0u8; total_bytes];
    let _frame_info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Error decodificando fotograma PNG: {}", e))?;

    let total_samples = width * height * channels;
    let mut f32_out = Vec::with_capacity(total_samples);

    match bit_depth {
        BitDepth::Sixteen => {
            let num_samples = buf.len() / 2;
            let take = num_samples.min(total_samples);
            for i in 0..take {
                let sample = u16::from_be_bytes([buf[2 * i], buf[2 * i + 1]]);
                f32_out.push(sample as f32 / 65535.0);
            }
        }
        BitDepth::Eight => {
            let take = buf.len().min(total_samples);
            for &b in &buf[..take] {
                f32_out.push(b as f32 / 255.0);
            }
        }
        _ => return Err(format!("Profundidad de bits no soportada: {:?}", bit_depth)),
    }

    Ok((width, height, channels, f32_out))
}

/// Guarda un archivo .devloom completo en paralelo.
///
/// Comprime todas las imágenes y máscaras en paralelo usando Rayon y las escribe
/// dentro del contenedor ZIP junto con el manifiesto project.json.
#[pyfunction]
pub fn write_devloom_project_rs<'py>(
    py: Python<'py>,
    file_path: &str,
    manifest_json: &str,
    images: Vec<(String, PyReadonlyArrayDyn<'py, f32>)>,
) -> PyResult<bool> {
    // 1. Validar y preparar descriptores de imágenes en el hilo principal con el GIL
    struct PendingTask {
        name: String,
        width: u32,
        height: u32,
        channels: usize,
        data: Vec<f32>,
    }

    let mut tasks = Vec::with_capacity(images.len());
    for (name, arr) in images {
        let shape = arr.shape();
        let (height, width, channels) = match shape.len() {
            2 => (shape[0], shape[1], 1),
            3 => (shape[0], shape[1], shape[2]),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Dimensión de array no soportada para '{}': {:?}",
                    name, shape
                )));
            }
        };

        let data = match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        };

        tasks.push(PendingTask {
            name,
            width: width as u32,
            height: height as u32,
            channels,
            data,
        });
    }

    // 2. Codificación paralela en Rayon liberando el GIL
    let encoded_results: Result<Vec<(String, Vec<u8>)>, String> = py.allow_threads(|| {
        tasks
            .into_par_iter()
            .map(|t| {
                let png_bytes = encode_png_16bit(&t.data, t.width, t.height, t.channels)?;
                Ok((t.name, png_bytes))
            })
            .collect()
    });

    let encoded_images = encoded_results.map_err(|e| PyRuntimeError::new_err(e))?;

    // 3. Empaquetado en contenedor ZIP
    let file = File::create(file_path).map_err(|e| {
        PyIOError::new_err(format!(
            "No se pudo crear el archivo de proyecto '{}': {}",
            file_path, e
        ))
    })?;
    let mut zip = ZipWriter::new(BufWriter::new(file));

    // Manifiesto project.json con compresión Deflated
    let deflated_opts =
        SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file("project.json", deflated_opts)
        .map_err(|e| PyRuntimeError::new_err(format!("Error en ZIP project.json: {}", e)))?;
    zip.write_all(manifest_json.as_bytes())
        .map_err(|e| PyIOError::new_err(format!("Error escribiendo project.json: {}", e)))?;

    // Archivos PNG precomprimidos almacenados directamente (Stored) para máxima velocidad
    let stored_opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for (name, png_bytes) in encoded_images {
        zip.start_file(name.as_str(), stored_opts).map_err(|e| {
            PyRuntimeError::new_err(format!("Error en ZIP entrada '{}': {}", name, e))
        })?;
        zip.write_all(&png_bytes).map_err(|e| {
            PyIOError::new_err(format!("Error escribiendo PNG '{}': {}", name, e))
        })?;
    }

    zip.finish()
        .map_err(|e| PyRuntimeError::new_err(format!("Error finalizando contenedor ZIP: {}", e)))?;

    Ok(true)
}

/// Lee y decodifica un proyecto .devloom en paralelo.
///
/// Extrae project.json y decodifica todas las imágenes/máscaras PNG de 16 bits
/// en paralelo usando Rayon, devolviendo el texto del manifiesto y un diccionario
/// con las matrices numpy correspondientes.
#[pyfunction]
pub fn read_devloom_project_rs<'py>(
    py: Python<'py>,
    file_path: &str,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let file = File::open(file_path).map_err(|e| {
        PyIOError::new_err(format!(
            "No se pudo abrir el archivo de proyecto '{}': {}",
            file_path, e
        ))
    })?;

    let mut archive = ZipArchive::new(BufReader::new(file))
        .map_err(|e| PyRuntimeError::new_err(format!("Error abriendo archivo ZIP: {}", e)))?;

    // 1. Leer project.json
    let mut manifest_json = String::new();
    {
        let mut json_file = archive.by_name("project.json").map_err(|_| {
            PyValueError::new_err(
                "El archivo no es un contenedor .devloom válido (falta project.json)",
            )
        })?;
        json_file
            .read_to_string(&mut manifest_json)
            .map_err(|e| PyIOError::new_err(format!("Error leyendo project.json: {}", e)))?;
    }

    // 2. Extraer todos los streams de PNG en memoria
    let mut raw_pngs: Vec<(String, Vec<u8>)> = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| {
            PyRuntimeError::new_err(format!("Error accediendo a entrada ZIP {}: {}", i, e))
        })?;

        let entry_name = entry
            .name()
            .map_err(|e| PyRuntimeError::new_err(format!("Error leyendo nombre de entrada: {}", e)))?;
        let name = entry_name.to_string();
        if name.ends_with(".png") {
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry
                .read_to_end(&mut buf)
                .map_err(|e| PyIOError::new_err(format!("Error leyendo entrada '{}': {}", name, e)))?;
            raw_pngs.push((name, buf));
        }
    }

    // 3. Decodificación paralela de todos los PNG con Rayon (liberando GIL)
    let decoded_results: Result<Vec<DecodedImage>, String> = py.allow_threads(|| {
        raw_pngs
            .into_par_iter()
            .map(|(name, raw_bytes)| {
                let (width, height, channels, data) = decode_png_to_f32(&raw_bytes)?;
                Ok(DecodedImage {
                    name,
                    width,
                    height,
                    channels,
                    data,
                })
            })
            .collect()
    });

    let decoded_images = decoded_results.map_err(|e| PyRuntimeError::new_err(e))?;

    // 4. Construcción del diccionario Python con matrices NumPy
    let dict = PyDict::new(py);
    for img in decoded_images {
        let shape = if img.channels == 1 {
            vec![img.height, img.width]
        } else {
            vec![img.height, img.width, img.channels]
        };

        let arr = ArrayD::from_shape_vec(IxDyn(&shape), img.data).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Error convirtiendo array para '{}': {:?}",
                img.name, e
            ))
        })?;

        let py_arr = arr.into_pyarray(py);
        dict.set_item(img.name, py_arr)?;
    }

    Ok((manifest_json, dict))
}
