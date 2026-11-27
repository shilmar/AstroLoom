use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

/// 3. create_lunar_mask_rs:
/// Genera máscara booleana circular del limbo lunar (dist >= r_cutoff) en paralelo.
#[pyfunction]
pub fn create_lunar_mask_rs<'py>(
    py: Python<'py>,
    h: usize,
    w: usize,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> Bound<'py, PyArray2<bool>> {
    let out_vec = py.allow_threads(|| {
        let mut buf = vec![false; h * w];
        let r2 = radius * radius;

        buf.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_mask)| {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    row_mask[x] = (dx * dx + dy2) >= r2;
                }
            });

        buf
    });

    Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py)
}

/// 4. compute_mgn_diff_sq_rs:
/// Calcula la diferencia cuadrática normalizada d^2 = (lum - num/den)^2
/// en las regiones válidas del cielo en paralelo sin alocar arrays booleanos ni intermedios.
#[pyfunction]
pub fn compute_mgn_diff_sq_rs<'py>(
    py: Python<'py>,
    lum: PyReadonlyArray2<'py, f32>,
    m_bool: PyReadonlyArray2<'py, bool>,
    num: PyReadonlyArray2<'py, f32>,
    den: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = lum.as_array().dim();
    let lum_slice = lum.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let m_slice = m_bool.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let num_slice = num.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let den_slice = den.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

    let out_vec = py.allow_threads(|| {
        let mut buf = vec![0.0f32; h * w];

        buf.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                let offset = y * w;
                let l_row = &lum_slice[offset..offset + w];
                let m_row = &m_slice[offset..offset + w];
                let n_row = &num_slice[offset..offset + w];
                let d_row = &den_slice[offset..offset + w];

                for x in 0..w {
                    if m_row[x] && d_row[x] >= 0.15 {
                        let mu = n_row[x] / d_row[x];
                        let diff = l_row[x] - mu;
                        row_out[x] = diff * diff;
                    } else {
                        row_out[x] = 0.0;
                    }
                }
            });

        buf
    });

    Ok(Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py))
}

/// 5. combine_mgn_scales_rs:
/// Fusión matemática y acumulación in-place en Rayon de la banda gaussiana:
/// calcula varianza local, normaliza el contraste por sigma local con piso de ruido relativo,
/// aplica la transformación arctan(k * c) y acumula en acc sin crear matrices temporales.
/// Invariante de unidades (AEL 1.3.6 / PR #8): adapta nf_sq y min_sd proporcionalmente a data_scale.
#[pyfunction]
#[pyo3(signature = (acc, lum, num, den, num_v, m_bool, k=0.70, noise_floor=0.003, data_scale=None))]
pub fn combine_mgn_scales_rs<'py>(
    py: Python<'py>,
    acc: &Bound<'py, PyArray2<f32>>,
    lum: PyReadonlyArray2<'py, f32>,
    num: PyReadonlyArray2<'py, f32>,
    den: PyReadonlyArray2<'py, f32>,
    num_v: PyReadonlyArray2<'py, f32>,
    m_bool: PyReadonlyArray2<'py, bool>,
    k: f32,
    noise_floor: f32,
    data_scale: Option<f32>,
) -> PyResult<()> {
    let (_h, w) = lum.as_array().dim();
    let lum_slice = lum.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let num_slice = num.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let den_slice = den.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let num_v_slice = num_v.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let m_slice = m_bool.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

    let scale = match data_scale {
        Some(s) if s > 0.0 => s,
        _ => {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for (idx, &is_m) in m_slice.iter().enumerate() {
                if is_m {
                    let v = lum_slice[idx];
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }
            }
            let diff = max_val - min_val;
            if diff > 1e-12 {
                diff
            } else {
                1.0
            }
        }
    };

    let eff_nf = noise_floor * scale;
    let nf_sq = eff_nf * eff_nf;
    let min_sd = 1e-7 * scale;

    let acc_slice = unsafe {
        acc.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("acc must be C-contiguous: {:?}", e))
        })?
    };

    py.allow_threads(move || {
        acc_slice
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_acc)| {
                let offset = y * w;
                let l_row = &lum_slice[offset..offset + w];
                let m_row = &m_slice[offset..offset + w];
                let n_row = &num_slice[offset..offset + w];
                let d_row = &den_slice[offset..offset + w];
                let nv_row = &num_v_slice[offset..offset + w];

                for x in 0..w {
                    if m_row[x] && d_row[x] >= 0.15 {
                        let mu = n_row[x] / d_row[x];
                        let var = (nv_row[x] / d_row[x]).max(0.0);
                        let sd = (var + nf_sq).sqrt();
                        if sd > min_sd {
                            let c = (l_row[x] - mu) / sd;
                            row_acc[x] += (k * c).atan();
                        }
                    }
                }
            });
    });

    Ok(())
}

/// 6. finalize_mgn_lum_rs:
/// Recombina armónicamente la acumulación multiescala arctan normalizada [0, 1]
/// con el contexto global por compresión gamma, aplicando la máscara lunar.
#[pyfunction]
#[pyo3(signature = (acc, total_w, lum, gamma=2.8, blend_ratio=0.50, m_bool=None))]
pub fn finalize_mgn_lum_rs<'py>(
    py: Python<'py>,
    acc: PyReadonlyArray2<'py, f32>,
    total_w: f32,
    lum: PyReadonlyArray2<'py, f32>,
    gamma: f32,
    blend_ratio: f32,
    m_bool: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = acc.as_array().dim();
    let acc_slice = acc.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let lum_slice = lum.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let m_slice = m_bool
        .as_ref()
        .map(|m| m.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous")))
        .transpose()?;

    let inv_total_w = 1.0 / total_w.max(1.0);
    let inv_gamma = 1.0 / gamma.max(0.1);
    let h_weight = (1.0 - blend_ratio).clamp(0.0, 1.0);
    let pi = std::f32::consts::PI;
    let half_pi = pi / 2.0;

    let out_vec = py.allow_threads(|| {
        let mut buf = vec![0.0f32; h * w];

        buf.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                let offset = y * w;
                let a_row = &acc_slice[offset..offset + w];
                let l_row = &lum_slice[offset..offset + w];
                let m_row = m_slice.map(|s| &s[offset..offset + w]);

                for x in 0..w {
                    let is_valid = match m_row {
                        Some(m) => m[x],
                        None => true,
                    };

                    if is_valid {
                        let a_val = a_row[x] * inv_total_w;
                        let acc_norm = ((a_val + half_pi) / pi).clamp(0.0, 1.0);
                        let g = l_row[x].clamp(0.0, 1.0).powf(inv_gamma);
                        row_out[x] = (h_weight * g + (1.0 - h_weight) * acc_norm).clamp(0.0, 1.0);
                    } else {
                        row_out[x] = 0.0;
                    }
                }
            });

        buf
    });

    Ok(Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py))
}

/// 7. compute_mgn_scaled_and_chroma_rs:
/// Calcula en una sola pasada el escalado RGB radiométrico con margen adaptativo SNR
/// y extrae la matriz cromática desacoplada chroma = scaled_rgb / mgn_lum.
#[pyfunction]
pub fn compute_mgn_scaled_and_chroma_rs<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, f32>,
    mgn_lum: PyReadonlyArray2<'py, f32>,
    lum: PyReadonlyArray2<'py, f32>,
    noise_floor: f32,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
    let (h, w, _) = img.as_array().dim();
    let img_slice = img.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let mgn_slice = mgn_lum.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let lum_slice = lum.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

    let snr_floor = (noise_floor * 3.0).max(1e-3);

    let (scaled_vec, chroma_vec) = py.allow_threads(|| {
        let row_stride = w * 3;
        let mut scaled_buf = vec![0.0f32; h * row_stride];
        let mut chroma_buf = vec![0.0f32; h * row_stride];

        scaled_buf
            .par_chunks_exact_mut(row_stride)
            .zip(chroma_buf.par_chunks_exact_mut(row_stride))
            .enumerate()
            .for_each(|(y, (row_scaled, row_chroma))| {
                let offset_2d = y * w;
                let offset_3d = y * row_stride;
                let l_row = &lum_slice[offset_2d..offset_2d + w];
                let ml_row = &mgn_slice[offset_2d..offset_2d + w];
                let img_row = &img_slice[offset_3d..offset_3d + row_stride];

                for x in 0..w {
                    let base_idx = x * 3;
                    let l = l_row[x];
                    let ml = ml_row[x];

                    let raw_ratio = if l > 1e-6 { ml / l.max(1e-4) } else { 0.0 };
                    let snr = (l / snr_floor).clamp(0.0, 1.0);
                    let max_ratio = 4.0 + 20.0 * snr;
                    let ratio = raw_ratio.min(max_ratio);

                    for c in 0..3 {
                        let s = (img_row[base_idx + c] * ratio).clamp(0.0, 1.0);
                        row_scaled[base_idx + c] = s;
                        row_chroma[base_idx + c] = if ml > 1e-5 { s / ml.max(1e-4) } else { 0.0 };
                    }
                }
            });

        (scaled_buf, chroma_buf)
    });

    let s_arr = Array3::from_shape_vec((h, w, 3), scaled_vec).unwrap();
    let c_arr = Array3::from_shape_vec((h, w, 3), chroma_vec).unwrap();

    Ok((s_arr.into_pyarray(py), c_arr.into_pyarray(py)))
}

/// 8. compute_mgn_prominence_mask_rs:
/// Genera la máscara binaria de protuberancias solares (H-alfa nativa: dist < mr * 1.35 y R/G > 1.15)
/// en paralelo sin instanciar matrices 2D de coordenadas.
#[pyfunction]
pub fn compute_mgn_prominence_mask_rs<'py>(
    py: Python<'py>,
    scaled_rgb: PyReadonlyArray3<'py, f32>,
    mcx: f32,
    mcy: f32,
    max_r: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w, _) = scaled_rgb.as_array().dim();
    let s_slice = scaled_rgb.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let max_r2 = max_r * max_r;

    let out_vec = py.allow_threads(|| {
        let row_stride = w * 3;
        let mut buf = vec![0.0f32; h * w];

        buf.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                let dy = y as f32 - mcy;
                let dy2 = dy * dy;
                let s_row = &s_slice[y * row_stride..(y + 1) * row_stride];

                for x in 0..w {
                    let dx = x as f32 - mcx;
                    let r2 = dx * dx + dy2;
                    if r2 < max_r2 {
                        let r = s_row[x * 3];
                        let g = s_row[x * 3 + 1];
                        if (r / g.max(1e-4)) > 1.15 {
                            row_out[x] = 1.0;
                        }
                    }
                }
            });

        buf
    });

    Ok(Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py))
}

/// 9. blend_mgn_prominences_and_mask_rs:
/// Fusión final de croma filtrado, preservación de protuberancias H-alfa
/// y recorte estricto del disco lunar en una sola pasada.
#[pyfunction]
#[pyo3(signature = (scaled_rgb, denoised_rgb, prom_w=None, m_bool=None))]
pub fn blend_mgn_prominences_and_mask_rs<'py>(
    py: Python<'py>,
    scaled_rgb: PyReadonlyArray3<'py, f32>,
    denoised_rgb: PyReadonlyArray3<'py, f32>,
    prom_w: Option<PyReadonlyArray2<'py, f32>>,
    m_bool: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (h, w, _) = scaled_rgb.as_array().dim();
    let scaled_slice = scaled_rgb.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let denoised_slice = denoised_rgb.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let prom_slice = prom_w
        .as_ref()
        .map(|p| p.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous")))
        .transpose()?;
    let m_slice = m_bool
        .as_ref()
        .map(|m| m.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous")))
        .transpose()?;

    let out_vec = py.allow_threads(|| {
        let row_stride = w * 3;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_exact_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_out)| {
                let offset_2d = y * w;
                let offset_3d = y * row_stride;
                let s_row = &scaled_slice[offset_3d..offset_3d + row_stride];
                let d_row = &denoised_slice[offset_3d..offset_3d + row_stride];
                let p_row = prom_slice.map(|p| &p[offset_2d..offset_2d + w]);
                let m_row = m_slice.map(|m| &m[offset_2d..offset_2d + w]);

                for x in 0..w {
                    let base_idx = x * 3;
                    let is_valid = match m_row {
                        Some(m) => m[x],
                        None => true,
                    };

                    if !is_valid {
                        row_out[base_idx] = 0.0;
                        row_out[base_idx + 1] = 0.0;
                        row_out[base_idx + 2] = 0.0;
                        continue;
                    }

                    let pw = match p_row {
                        Some(p) => p[x],
                        None => 0.0,
                    };

                    for c in 0..3 {
                        let s = s_row[base_idx + c];
                        let d = d_row[base_idx + c];
                        row_out[base_idx + c] = (pw * s + (1.0 - pw) * d).clamp(0.0, 1.0);
                    }
                }
            });

        buf
    });

    Ok(Array3::from_shape_vec((h, w, 3), out_vec).unwrap().into_pyarray(py))
}
