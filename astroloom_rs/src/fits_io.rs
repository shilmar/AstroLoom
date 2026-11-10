use numpy::ndarray::{Array2, Array3};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

// ============================================================================
// PARSER ULTRA-RÁPIDO DE ARCHIVOS FITS (core/stacking.py y core/frame_culling.py)
// ============================================================================

#[derive(Debug, Default, Clone)]
struct FitsHduHeader {
    bitpix: i32,
    naxis: usize,
    naxis1: usize,
    naxis2: usize,
    naxis3: usize,
    bscale: f64,
    bzero: f64,
    data_offset: u64,
}

fn parse_header_value(val_str: &str) -> String {
    let clean = if let Some(slash_idx) = val_str.find('/') {
        &val_str[..slash_idx]
    } else {
        val_str
    };
    clean.trim().replace('\'', "").trim().to_string()
}

fn read_fits_header<R: Read + Seek>(reader: &mut R) -> Result<FitsHduHeader, String> {
    let mut header = FitsHduHeader {
        bitpix: 16,
        naxis: 0,
        naxis1: 0,
        naxis2: 0,
        naxis3: 1,
        bscale: 1.0,
        bzero: 0.0,
        data_offset: 0,
    };

    let mut block = [0u8; 2880];
    let mut found_end = false;
    let mut current_offset: u64 = reader
        .stream_position()
        .map_err(|e| format!("Error de stream: {}", e))?;

    while !found_end {
        let n = reader
            .read(&mut block)
            .map_err(|e| format!("Error de lectura: {}", e))?;
        if n < 2880 {
            if n == 0 {
                return Err("Fin de archivo prematuro al leer cabecera FITS".into());
            }
            return Err("Bloque de cabecera FITS incompleto (< 2880 bytes)".into());
        }
        current_offset += 2880;

        for card_idx in 0..36 {
            let start = card_idx * 80;
            let end = start + 80;
            let card = &block[start..end];
            let key = match std::str::from_utf8(&card[0..8]) {
                Ok(s) => s.trim(),
                Err(_) => continue,
            };

            if key == "END" {
                found_end = true;
                break;
            }

            if card[8] == b'=' {
                if let Ok(raw_val) = std::str::from_utf8(&card[9..80]) {
                    let v = parse_header_value(raw_val);
                    match key {
                        "BITPIX" => {
                            if let Ok(bp) = v.parse::<i32>() {
                                header.bitpix = bp;
                            }
                        }
                        "NAXIS" => {
                            if let Ok(nx) = v.parse::<usize>() {
                                header.naxis = nx;
                            }
                        }
                        "NAXIS1" => {
                            if let Ok(nx1) = v.parse::<usize>() {
                                header.naxis1 = nx1;
                            }
                        }
                        "NAXIS2" => {
                            if let Ok(nx2) = v.parse::<usize>() {
                                header.naxis2 = nx2;
                            }
                        }
                        "NAXIS3" => {
                            if let Ok(nx3) = v.parse::<usize>() {
                                header.naxis3 = nx3;
                            }
                        }
                        "BSCALE" => {
                            let fixed = v.replace(['d', 'D'], "e");
                            if let Ok(bs) = fixed.parse::<f64>() {
                                header.bscale = bs;
                            }
                        }
                        "BZERO" => {
                            let fixed = v.replace(['d', 'D'], "e");
                            if let Ok(bz) = fixed.parse::<f64>() {
                                header.bzero = bz;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    header.data_offset = current_offset;
    Ok(header)
}

fn locate_image_hdu(filepath: &str) -> Result<(File, FitsHduHeader), String> {
    let mut file = File::open(filepath).map_err(|e| format!("No se pudo abrir '{}': {}", filepath, e))?;
    let mut hdu = read_fits_header(&mut file)?;

    // Si el HDU primario no contiene datos (NAXIS == 0), buscar el primer HDU con imagen
    if hdu.naxis == 0 {
        // En FITS con extensiones, si NAXIS == 0, no hay datos primarios, el siguiente bloque comienza de inmediato
        loop {
            match read_fits_header(&mut file) {
                Ok(next_hdu) => {
                    if next_hdu.naxis >= 2 && next_hdu.naxis1 > 0 && next_hdu.naxis2 > 0 {
                        hdu = next_hdu;
                        break;
                    }
                    // Si este HDU tampoco tiene datos válidos, saltar su bloque de datos si lo tuviera
                    if next_hdu.naxis > 0 {
                        let bytes_per_elem = (next_hdu.bitpix.unsigned_abs() as usize) / 8;
                        let count = next_hdu.naxis1
                            * next_hdu.naxis2
                            * if next_hdu.naxis >= 3 { next_hdu.naxis3 } else { 1 };
                        let raw_bytes = count * bytes_per_elem;
                        let pad = (2880 - (raw_bytes % 2880)) % 2880;
                        let skip = (raw_bytes + pad) as i64;
                        let _ = file.seek(SeekFrom::Current(skip));
                    }
                }
                Err(_) => {
                    break;
                }
            }
        }
    }

    if hdu.naxis < 2 || hdu.naxis1 == 0 || hdu.naxis2 == 0 {
        return Err(format!("El archivo FITS '{}' no contiene una matriz de imagen 2D o 3D válida.", filepath));
    }

    Ok((file, hdu))
}

#[pyfunction]
pub fn get_fits_dimensions_rs(filepath: &str) -> PyResult<(usize, usize)> {
    let (_, hdu) = locate_image_hdu(filepath)
        .map_err(|e| PyValueError::new_err(e))?;
    // En FITS: NAXIS2 es altura (filas), NAXIS1 es ancho (columnas)
    Ok((hdu.naxis2, hdu.naxis1))
}

#[pyfunction]
pub fn read_fits_f32<'py>(py: Python<'py>, filepath: &str) -> PyResult<Bound<'py, PyAny>> {
    let (mut file, hdu) = locate_image_hdu(filepath)
        .map_err(|e| PyValueError::new_err(e))?;

    let w = hdu.naxis1;
    let h = hdu.naxis2;
    let c = if hdu.naxis >= 3 { hdu.naxis3 } else { 1 };
    let total_pixels = w * h * c;

    file.seek(SeekFrom::Start(hdu.data_offset))
        .map_err(|e| PyIOError::new_err(format!("Fallo de seek: {}", e)))?;

    let bytes_per_elem = match hdu.bitpix.abs() {
        8 => 1,
        16 => 2,
        32 => 4,
        64 => 8,
        _ => return Err(PyValueError::new_err(format!("BITPIX {} no soportado", hdu.bitpix))),
    };

    let total_bytes = total_pixels * bytes_per_elem;
    let mut raw_bytes = vec![0u8; total_bytes];

    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    reader
        .read_exact(&mut raw_bytes)
        .map_err(|e| PyIOError::new_err(format!("Lectura incompleta de píxeles FITS: {}", e)))?;

    let bscale = hdu.bscale;
    let bzero = hdu.bzero;
    let bitpix = hdu.bitpix;

    // Decodificación de orden Big-Endian en paralelo con Rayon liberando el GIL
    let decoded_f32: Vec<f32> = py.allow_threads(|| match bitpix {
        8 => raw_bytes
            .par_iter()
            .map(|&b| ((b as f64) * bscale + bzero) as f32)
            .collect(),
        16 => raw_bytes
            .par_chunks_exact(2)
            .map(|chunk| {
                let v = i16::from_be_bytes([chunk[0], chunk[1]]);
                ((v as f64) * bscale + bzero) as f32
            })
            .collect(),
        32 => raw_bytes
            .par_chunks_exact(4)
            .map(|chunk| {
                let v = i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                ((v as f64) * bscale + bzero) as f32
            })
            .collect(),
        -32 => raw_bytes
            .par_chunks_exact(4)
            .map(|chunk| {
                let v = f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                ((v as f64) * bscale + bzero) as f32
            })
            .collect(),
        -64 => raw_bytes
            .par_chunks_exact(8)
            .map(|chunk| {
                let v = f64::from_be_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                (v * bscale + bzero) as f32
            })
            .collect(),
        _ => unreachable!(),
    });

    if hdu.naxis >= 3 && c > 1 {
        let arr = Array3::from_shape_vec((c, h, w), decoded_f32)
            .map_err(|e| PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(arr.into_pyarray(py).into_any())
    } else {
        let arr = Array2::from_shape_vec((h, w), decoded_f32)
            .map_err(|e| PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(arr.into_pyarray(py).into_any())
    }
}
