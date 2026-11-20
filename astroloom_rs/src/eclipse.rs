use numpy::ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// FILTROS CORONALES DE ECLIPSES HDR (core/eclipse.py) — PRIORIDAD 5
// ============================================================================

#[inline(always)]
fn median_f32(data: &mut [f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len();
    let mid = n / 2;
    data.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        data[mid]
    } else {
        let val1 = data[mid];
        let val0 = *data[..mid].iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&val1);
        0.5 * (val0 + val1)
    }
}

#[inline(always)]
fn reflect101(mut idx: isize, len: isize) -> usize {
    if len <= 1 {
        return 0;
    }
    while idx < 0 || idx >= len {
        if idx < 0 {
            idx = -idx;
        } else if idx >= len {
            idx = 2 * len - 2 - idx;
        }
    }
    idx as usize
}

fn gaussian_blur_1d(src: &[f32], dst: &mut [f32], ksize: usize, sigma: f32) {
    let n = src.len();
    if n == 0 {
        return;
    }
    if n == 1 || ksize <= 1 || sigma <= 0.0 {
        dst.copy_from_slice(src);
        return;
    }

    let r = (ksize / 2) as isize;
    let two_sig_sq = 2.0 * sigma * sigma;

    // Precomputar pesos gaussianos normalizados
    let mut kernel = vec![0.0f32; ksize];
    let mut k_sum = 0.0f32;
    for i in 0..ksize {
        let d = (i as isize - r) as f32;
        let w = (-d * d / two_sig_sq).exp();
        kernel[i] = w;
        k_sum += w;
    }
    let inv_sum = 1.0 / k_sum;
    for w in kernel.iter_mut() {
        *w *= inv_sum;
    }

    let len_isize = n as isize;
    for i in 0..n {
        let mut sum = 0.0f32;
        for (k_idx, &w) in kernel.iter().enumerate() {
            let offset = k_idx as isize - r;
            let pos = reflect101(i as isize + offset, len_isize);
            sum += src[pos] * w;
        }
        dst[i] = sum;
    }
}

/// 1. compute_nrgf_profiles_and_detail_rs:
/// Cálculo en paralelo con Rayon de las tablas radiales mu(r) y sigma(r),
/// estimación del ruido de fondo de cielo sky_noise, y cálculo continuo
/// de raw_detail = (lum - mu) / sigma sin instanciar matrices 2D de coordenadas.
#[pyfunction]
#[pyo3(signature = (lum, center_x, center_y, radius_min=10.0, radius_max_factor=3.5, robust=true))]
pub fn compute_nrgf_profiles_and_detail_rs<'py>(
    py: Python<'py>,
    lum: PyReadonlyArray2<'py, f32>,
    center_x: f32,
    center_y: f32,
    radius_min: f32,
    radius_max_factor: f32,
    robust: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, f32)> {
    let lum_view = lum.as_array();
    let (h, w) = lum_view.dim();
    let lum_slice = lum.as_slice().unwrap();

    let r_max = radius_min * (1.2f32).max(radius_max_factor);

    let (raw_detail_vec, sigma_prof_vec, sky_noise) = py.allow_threads(|| {
        let corners = [
            (0.0f32, 0.0f32),
            (0.0f32, (h - 1) as f32),
            ((w - 1) as f32, 0.0f32),
            ((w - 1) as f32, (h - 1) as f32),
        ];
        let max_r = corners
            .iter()
            .map(|&(x, y)| {
                let dx = x - center_x;
                let dy = y - center_y;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0f32, f32::max);

        let max_bin = (max_r as usize) + 2;

        // Histograma radial por hilos
        let counts = (0..h)
            .into_par_iter()
            .fold(
                || vec![0usize; max_bin],
                |mut acc, y| {
                    let dy = y as f32 - center_y;
                    let dy2 = dy * dy;
                    let row = &lum_slice[y * w..(y + 1) * w];
                    for x in 0..w {
                        let dx = x as f32 - center_x;
                        let r = (dx * dx + dy2).sqrt();
                        let b = (r as usize).min(max_bin - 1);
                        acc[b] += 1;
                        let _ = row;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0usize; max_bin],
                |mut acc, thread_hist| {
                    for (a, t) in acc.iter_mut().zip(thread_hist.iter()) {
                        *a += *t;
                    }
                    acc
                },
            );

        // Desplazamientos acumulados para cubos de píxeles
        let mut starts = vec![0usize; max_bin + 1];
        for b in 0..max_bin {
            starts[b + 1] = starts[b] + counts[b];
        }
        let total_pixels = starts[max_bin];

        let mut samples = vec![0.0f32; total_pixels];
        let mut write_pos = starts[..max_bin].to_vec();

        // Dispersión a cubos radiales y muestra de cielo exterior
        let mut sky_samples = Vec::new();
        for y in 0..h {
            let dy = y as f32 - center_y;
            let dy2 = dy * dy;
            let row = &lum_slice[y * w..(y + 1) * w];
            for x in 0..w {
                let dx = x as f32 - center_x;
                let r = (dx * dx + dy2).sqrt();
                let b = (r as usize).min(max_bin - 1);
                let val = row[x];
                samples[write_pos[b]] = val;
                write_pos[b] += 1;

                if r >= r_max {
                    sky_samples.push(val);
                }
            }
        }

        // Estimación robusta de ruido de fondo de cielo
        let sky_noise = if sky_samples.len() > 100 {
            let med = median_f32(&mut sky_samples);
            for v in sky_samples.iter_mut() {
                *v = (*v - med).abs();
            }
            (1.4826 * median_f32(&mut sky_samples)).max(1e-4)
        } else {
            // Muestreo del percentil 90 exterior
            let far_bin = (max_bin as f32 * 0.90) as usize;
            let mut far_samples = Vec::new();
            for b in far_bin..max_bin {
                far_samples.extend_from_slice(&samples[starts[b]..starts[b] + counts[b]]);
            }
            if far_samples.len() > 20 {
                let med = median_f32(&mut far_samples);
                for v in far_samples.iter_mut() {
                    *v = (*v - med).abs();
                }
                (1.4826 * median_f32(&mut far_samples)).max(1e-4)
            } else {
                1e-4
            }
        };

        // Cálculo de perfiles radiales mu(r) y sigma(r)
        let mut mu_prof = vec![0.0f32; max_bin];
        let mut sigma_prof = vec![1.0f32; max_bin];
        let r_cutoff = radius_min * 0.95;

        if robust {
            let stats: Vec<(f32, f32)> = (0..max_bin)
                .into_par_iter()
                .map(|b| {
                    let cnt = counts[b];
                    if cnt >= 30 && (b as f32) >= r_cutoff {
                        let mut seg = samples[starts[b]..starts[b] + cnt].to_vec();
                        let med = median_f32(&mut seg);
                        for v in seg.iter_mut() {
                            *v = (*v - med).abs();
                        }
                        let mad = 1.4826 * median_f32(&mut seg);
                        (med, mad)
                    } else {
                        (0.0, 1.0)
                    }
                })
                .collect();

            for (b, &(med, mad)) in stats.iter().enumerate() {
                if counts[b] >= 30 && (b as f32) >= r_cutoff {
                    mu_prof[b] = med;
                    sigma_prof[b] = mad;
                }
            }
        } else {
            for b in 0..max_bin {
                let cnt = counts[b];
                if cnt >= 30 && (b as f32) >= r_cutoff {
                    let seg = &samples[starts[b]..starts[b] + cnt];
                    let sum: f64 = seg.iter().map(|&v| v as f64).sum();
                    let mean = (sum / cnt as f64) as f32;
                    let sum_sq: f64 = seg.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let var =
                        ((sum_sq / cnt as f64) - (mean as f64 * mean as f64)).max(1e-8) as f32;
                    mu_prof[b] = mean;
                    sigma_prof[b] = var.sqrt();
                }
            }
        }

        // Suavizado gaussiano 1D de perfiles
        if max_bin > 15 {
            let mut blurred_mu = vec![0.0f32; max_bin];
            let mut blurred_sigma = vec![0.0f32; max_bin];
            gaussian_blur_1d(&mu_prof, &mut blurred_mu, 25, 6.0);
            gaussian_blur_1d(&sigma_prof, &mut blurred_sigma, 25, 6.0);
            mu_prof = blurred_mu;
            sigma_prof = blurred_sigma;
        }

        let min_sigma = (sky_noise * 2.0).max(0.0025);
        for s in sigma_prof.iter_mut() {
            *s = (*s).max(min_sigma);
        }

        // Cálculo continuo de raw_detail en una sola pasada de Rayon
        let mut raw_detail = vec![0.0f32; h * w];
        raw_detail
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_detail)| {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                let row_lum = &lum_slice[y * w..(y + 1) * w];
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    let r = (dx * dx + dy2).sqrt();
                    if r < radius_min {
                        row_detail[x] = 0.0;
                    } else {
                        let i = (r.floor() as usize).min(max_bin - 2);
                        let frac = (r - i as f32).clamp(0.0, 1.0);
                        let mu = mu_prof[i] * (1.0 - frac) + mu_prof[i + 1] * frac;
                        let sigma = sigma_prof[i] * (1.0 - frac) + sigma_prof[i + 1] * frac;
                        row_detail[x] = (row_lum[x] - mu) / sigma;
                    }
                }
            });

        (raw_detail, sigma_prof, sky_noise)
    });

    let raw_arr = Array2::from_shape_vec((h, w), raw_detail_vec).unwrap();
    let sig_arr = Array1::from_shape_vec(sigma_prof_vec.len(), sigma_prof_vec).unwrap();

    Ok((raw_arr.into_pyarray(py), sig_arr.into_pyarray(py), sky_noise))
}

/// 2. apply_nrgf_continuous_rs:
/// Evaluación paralela por filas de la distancia al centro solar (cx, cy),
/// compuerta radial suave Hermite (smoothstep), modulación continua de SNR y
/// recomposición fotométrica RGB sin asignar matrices intermedias de coordenadas ni máscaras.
#[pyfunction]
#[pyo3(signature = (image, clean_detail, sigma_prof, sky_noise, center_x, center_y, radius_min=10.0, radius_max_factor=3.5, blend_ratio=0.55))]
pub fn apply_nrgf_continuous_rs<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    clean_detail: PyReadonlyArray2<'py, f32>,
    sigma_prof: PyReadonlyArray1<'py, f32>,
    sky_noise: f32,
    center_x: f32,
    center_y: f32,
    radius_min: f32,
    radius_max_factor: f32,
    blend_ratio: f32,
) -> PyResult<Bound<'py, PyAny>> {
    let clean_slice = clean_detail.as_slice().unwrap();
    let (h, w) = clean_detail.as_array().dim();
    let sig_slice = sigma_prof.as_slice().unwrap();
    let sig_len = sig_slice.len();

    let r_max = radius_min * (1.2f32).max(radius_max_factor);
    let gate_start = radius_min * (1.0 + (radius_max_factor - 1.0) * 0.75);
    let gate_span = (r_max - gate_start).max(1.0);
    let blend = blend_ratio.clamp(0.0, 1.0);

    let is_3d = if let Ok(arr3) = image.extract::<PyReadonlyArray3<f32>>() {
        let (ih, iw, c) = arr3.as_array().dim();
        if ih != h || iw != w || c != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dimensiones de imagen 3D no coinciden con clean_detail",
            ));
        }
        true
    } else if let Ok(arr2) = image.extract::<PyReadonlyArray2<f32>>() {
        let (ih, iw) = arr2.as_array().dim();
        if ih != h || iw != w {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dimensiones de imagen 2D no coinciden con clean_detail",
            ));
        }
        false
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "image debe ser un ndarray float32 de 2 o 3 dimensiones",
        ));
    };

    if is_3d {
        let img_arr = image.extract::<PyReadonlyArray3<f32>>()?;
        let img_slice = img_arr.as_slice().unwrap();

        let out_vec = py.allow_threads(|| {
            let row_stride = w * 3;
            let mut buf = vec![0.0f32; h * row_stride];

            buf.par_chunks_exact_mut(row_stride)
                .enumerate()
                .for_each(|(y, row_out)| {
                    let dy = y as f32 - center_y;
                    let dy2 = dy * dy;
                    let clean_row = &clean_slice[y * w..(y + 1) * w];
                    let img_row = &img_slice[y * row_stride..(y + 1) * row_stride];

                    for x in 0..w {
                        let dx = x as f32 - center_x;
                        let r = (dx * dx + dy2).sqrt();
                        let base_idx = x * 3;

                        if r < radius_min {
                            row_out[base_idx] = 0.0;
                            row_out[base_idx + 1] = 0.0;
                            row_out[base_idx + 2] = 0.0;
                            continue;
                        }

                        let r0 = img_row[base_idx];
                        let g0 = img_row[base_idx + 1];
                        let b0 = img_row[base_idx + 2];
                        let lum = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;

                        let i = (r.floor() as usize).min(sig_len - 2);
                        let frac = (r - i as f32).clamp(0.0, 1.0);
                        let sig = sig_slice[i] * (1.0 - frac) + sig_slice[i + 1] * frac;

                        let t = ((r - gate_start) / gate_span).clamp(0.0, 1.0);
                        let radial_gate = if r > r_max {
                            0.0
                        } else {
                            1.0 - (t * t * (3.0 - 2.0 * t))
                        };

                        let snr_factor =
                            ((sig - 1.2 * sky_noise) / (sig + 1e-6)).clamp(0.0, 1.0);
                        let clipped_detail = (clean_row[x].clamp(-2.5, 2.5)) / 2.5;

                        let streamer_mod = clipped_detail * radial_gate * snr_factor * blend;
                        let upper_headroom = if streamer_mod > 0.0 {
                            (1.0 - lum).max(0.0)
                        } else {
                            1.0
                        };
                        let final_mod = streamer_mod * (0.4 + 0.6 * upper_headroom.sqrt());
                        let new_lum = (lum + final_mod).clamp(0.0, 1.0);

                        let ratio = if lum > 1e-6 {
                            (new_lum / lum.max(1e-4)).clamp(0.0, 4.0)
                        } else {
                            0.0
                        };

                        row_out[base_idx] = (r0 * ratio).clamp(0.0, 1.0);
                        row_out[base_idx + 1] = (g0 * ratio).clamp(0.0, 1.0);
                        row_out[base_idx + 2] = (b0 * ratio).clamp(0.0, 1.0);
                    }
                });

            buf
        });

        let out_arr = Array3::from_shape_vec((h, w, 3), out_vec).unwrap();
        Ok(out_arr.into_pyarray(py).as_any().clone())
    } else {
        let img_arr = image.extract::<PyReadonlyArray2<f32>>()?;
        let img_slice = img_arr.as_slice().unwrap();

        let out_vec = py.allow_threads(|| {
            let mut buf = vec![0.0f32; h * w];

            buf.par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(y, row_out)| {
                    let dy = y as f32 - center_y;
                    let dy2 = dy * dy;
                    let clean_row = &clean_slice[y * w..(y + 1) * w];
                    let img_row = &img_slice[y * w..(y + 1) * w];

                    for x in 0..w {
                        let dx = x as f32 - center_x;
                        let r = (dx * dx + dy2).sqrt();

                        if r < radius_min {
                            row_out[x] = 0.0;
                            continue;
                        }

                        let lum = img_row[x];
                        let i = (r.floor() as usize).min(sig_len - 2);
                        let frac = (r - i as f32).clamp(0.0, 1.0);
                        let sig = sig_slice[i] * (1.0 - frac) + sig_slice[i + 1] * frac;

                        let t = ((r - gate_start) / gate_span).clamp(0.0, 1.0);
                        let radial_gate = if r > r_max {
                            0.0
                        } else {
                            1.0 - (t * t * (3.0 - 2.0 * t))
                        };

                        let snr_factor =
                            ((sig - 1.2 * sky_noise) / (sig + 1e-6)).clamp(0.0, 1.0);
                        let clipped_detail = (clean_row[x].clamp(-2.5, 2.5)) / 2.5;

                        let streamer_mod = clipped_detail * radial_gate * snr_factor * blend;
                        let upper_headroom = if streamer_mod > 0.0 {
                            (1.0 - lum).max(0.0)
                        } else {
                            1.0
                        };
                        let final_mod = streamer_mod * (0.4 + 0.6 * upper_headroom.sqrt());
                        row_out[x] = (lum + final_mod).clamp(0.0, 1.0);
                    }
                });

            buf
        });

        let out_arr = Array2::from_shape_vec((h, w), out_vec).unwrap();
        Ok(out_arr.into_pyarray(py).as_any().clone())
    }
}

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
) -> Bound<'py, PyArray2<f32>> {
    let (h, w) = lum.as_array().dim();
    let lum_slice = lum.as_slice().unwrap();
    let m_slice = m_bool.as_slice().unwrap();
    let num_slice = num.as_slice().unwrap();
    let den_slice = den.as_slice().unwrap();

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

    Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py)
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
    let lum_slice = lum.as_slice().unwrap();
    let num_slice = num.as_slice().unwrap();
    let den_slice = den.as_slice().unwrap();
    let num_v_slice = num_v.as_slice().unwrap();
    let m_slice = m_bool.as_slice().unwrap();

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
) -> Bound<'py, PyArray2<f32>> {
    let (h, w) = acc.as_array().dim();
    let acc_slice = acc.as_slice().unwrap();
    let lum_slice = lum.as_slice().unwrap();
    let m_slice = m_bool.as_ref().map(|m| m.as_slice().unwrap());

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

    Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py)
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
) -> (Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>) {
    let (h, w, _) = img.as_array().dim();
    let img_slice = img.as_slice().unwrap();
    let mgn_slice = mgn_lum.as_slice().unwrap();
    let lum_slice = lum.as_slice().unwrap();

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

    (s_arr.into_pyarray(py), c_arr.into_pyarray(py))
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
) -> Bound<'py, PyArray2<f32>> {
    let (h, w, _) = scaled_rgb.as_array().dim();
    let s_slice = scaled_rgb.as_slice().unwrap();
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

    Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py)
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
) -> Bound<'py, PyArray3<f32>> {
    let (h, w, _) = scaled_rgb.as_array().dim();
    let scaled_slice = scaled_rgb.as_slice().unwrap();
    let denoised_slice = denoised_rgb.as_slice().unwrap();
    let prom_slice = prom_w.as_ref().map(|p| p.as_slice().unwrap());
    let m_slice = m_bool.as_ref().map(|m| m.as_slice().unwrap());

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

    Array3::from_shape_vec((h, w, 3), out_vec).unwrap().into_pyarray(py)
}

/// 10. display_detail_rings_rs:
/// Mapeo de detalle con nivel y dispersión medidos anillo por anillo en torno al centro solar (AEL 1.3.4 / Issue #9).
/// Calcula mediana y 1.4826*MAD en anillos de ancho width_px con Rayon, suaviza en 1D e interpola
/// de forma continua sin efecto de escalonamiento. Mapeo: 0.5 + 0.5 * tanh((detail - level) / (2.5 * spread)).
pub(crate) fn display_detail_rings_raw(
    detail_slice: &[f32],
    mask_slice: Option<&[bool]>,
    h: usize,
    w: usize,
    center_x: f32,
    center_y: f32,
    width_px: f32,
    min_count: usize,
    smooth_bins: usize,
    noise_floor: f32,
) -> Vec<f32> {
    let w_px = width_px.max(0.1);
    let inv_w = 1.0 / w_px;

    let corners = [
        (0.0f32, 0.0f32),
        (0.0f32, (h - 1) as f32),
        ((w - 1) as f32, 0.0f32),
        ((w - 1) as f32, (h - 1) as f32),
    ];
    let max_r = corners
        .iter()
        .map(|&(x, y)| {
            let dx = x - center_x;
            let dy = y - center_y;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0f32, f32::max);

    let max_bin = (max_r * inv_w) as usize + 2;

    // 1. Conteo de píxeles y suma radial por bin
    let (counts, rsums, total_good) = (0..h)
        .into_par_iter()
        .fold(
            || (vec![0usize; max_bin], vec![0.0f64; max_bin], 0usize),
            |(mut acc_cnt, mut acc_r, mut acc_good), y| {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                let offset = y * w;
                let d_row = &detail_slice[offset..offset + w];
                let m_row = mask_slice.map(|m| &m[offset..offset + w]);

                for x in 0..w {
                    let is_valid = match m_row {
                        Some(m) => m[x] && d_row[x].is_finite(),
                        None => d_row[x].is_finite(),
                    };
                    if is_valid {
                        let dx = x as f32 - center_x;
                        let r = (dx * dx + dy2).sqrt();
                        let b = ((r * inv_w) as usize).min(max_bin - 1);
                        acc_cnt[b] += 1;
                        acc_r[b] += r as f64;
                        acc_good += 1;
                    }
                }
                (acc_cnt, acc_r, acc_good)
            },
        )
        .reduce(
            || (vec![0usize; max_bin], vec![0.0f64; max_bin], 0usize),
            |(mut a_c, mut a_r, a_g), (b_c, b_r, b_g)| {
                for (ac, bc) in a_c.iter_mut().zip(b_c.iter()) {
                    *ac += *bc;
                }
                for (ar, br) in a_r.iter_mut().zip(b_r.iter()) {
                    *ar += *br;
                }
                (a_c, a_r, a_g + b_g)
            },
        );

    if total_good == 0 {
        return vec![0.0f32; h * w];
    }

    let sel: Vec<usize> = (0..max_bin)
        .filter(|&b| counts[b] >= min_count)
        .collect();

    // Caso fallback si hay menos de 3 anillos poblados
    if sel.len() < 3 {
        let mut all_vals: Vec<f32> = Vec::with_capacity(total_good);
        for y in 0..h {
            let offset = y * w;
            let d_row = &detail_slice[offset..offset + w];
            let m_row = mask_slice.map(|m| &m[offset..offset + w]);
            for x in 0..w {
                let is_valid = match m_row {
                    Some(m) => m[x] && d_row[x].is_finite(),
                    None => d_row[x].is_finite(),
                };
                if is_valid {
                    all_vals.push(d_row[x]);
                }
            }
        }
        let med = median_f32(&mut all_vals);
        let mut diffs: Vec<f32> = all_vals.iter().map(|&v| (v - med).abs()).collect();
        let mad = (1.4826f32 * median_f32(&mut diffs)).max(noise_floor);
        let denom = 2.5f32 * mad;

        let mut out = vec![0.0f32; h * w];
        out.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                let offset = y * w;
                let d_row = &detail_slice[offset..offset + w];
                let m_row = mask_slice.map(|m| &m[offset..offset + w]);
                for x in 0..w {
                    let is_valid = match m_row {
                        Some(m) => m[x] && d_row[x].is_finite(),
                        None => d_row[x].is_finite(),
                    };
                    if is_valid {
                        row_out[x] = 0.5f32 + 0.5f32 * ((d_row[x] - med) / denom).tanh();
                    } else {
                        row_out[x] = 0.0f32;
                    }
                }
            });
        return out;
    }

    let n_sel = sel.len();
    let mut bin_to_sel = vec![usize::MAX; max_bin];
    for (sel_idx, &b) in sel.iter().enumerate() {
        bin_to_sel[b] = sel_idx;
    }

    let mut offsets = vec![0usize; n_sel + 1];
    for (i, &b) in sel.iter().enumerate() {
        offsets[i + 1] = offsets[i] + counts[b];
    }
    let total_sel_pixels = offsets[n_sel];
    let mut bin_values = vec![0.0f32; total_sel_pixels];

    let write_indices: Vec<AtomicUsize> = offsets[..n_sel]
        .iter()
        .map(|&off| AtomicUsize::new(off))
        .collect();

    let raw_addr = bin_values.as_mut_ptr() as usize;

    (0..h).into_par_iter().for_each(|y| {
        let dy = y as f32 - center_y;
        let dy2 = dy * dy;
        let offset = y * w;
        let d_row = &detail_slice[offset..offset + w];
        let m_row = mask_slice.map(|m| &m[offset..offset + w]);

        for x in 0..w {
            let is_valid = match m_row {
                Some(m) => m[x] && d_row[x].is_finite(),
                None => d_row[x].is_finite(),
            };
            if is_valid {
                let dx = x as f32 - center_x;
                let r = (dx * dx + dy2).sqrt();
                let b = ((r * inv_w) as usize).min(max_bin - 1);
                let sel_idx = bin_to_sel[b];
                if sel_idx != usize::MAX {
                    let pos = write_indices[sel_idx].fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        let ptr = raw_addr as *mut f32;
                        *ptr.add(pos) = d_row[x];
                    }
                }
            }
        }
    });

    // 2. Mediana y MAD en paralelo para cada anillo seleccionado
    let mut med_vec = vec![0.0f32; n_sel];
    let mut mad_vec = vec![0.0f32; n_sel];
    let mut rm_vec = vec![0.0f32; n_sel];

    let results: Vec<(f32, f32, f32)> = (0..n_sel)
        .into_par_iter()
        .map(|i| {
            let b = sel[i];
            let start = offsets[i];
            let cnt = counts[b];
            let ptr = raw_addr as *mut f32;
            let seg = unsafe {
                std::slice::from_raw_parts_mut(ptr.add(start), cnt)
            };
            let med = median_f32(seg);
            let mut diffs: Vec<f32> = seg.iter().map(|&v| (v - med).abs()).collect();
            let mad = 1.4826f32 * median_f32(&mut diffs);
            let rm = (rsums[b] / (cnt as f64)) as f32;
            (rm, med, mad)
        })
        .collect();

    for (i, (rm, med, mad)) in results.into_iter().enumerate() {
        rm_vec[i] = rm;
        med_vec[i] = med;
        mad_vec[i] = mad;
    }

    // 3. Suavizado 1D a lo largo de los anillos si smooth_bins > 1
    let (smooth_med, smooth_mad) = if smooth_bins > 1 {
        let half = (smooth_bins / 2) as isize;
        let mut s_med = vec![0.0f32; n_sel];
        let mut s_mad = vec![0.0f32; n_sel];
        for i in 0..n_sel {
            let start = (i as isize - half).max(0) as usize;
            let end = (i as isize + half + (smooth_bins % 2) as isize).min(n_sel as isize) as usize;
            let k_count = (end - start) as f32;
            let sum_m: f32 = med_vec[start..end].iter().sum();
            let sum_d: f32 = mad_vec[start..end].iter().sum();
            s_med[i] = sum_m / k_count;
            s_mad[i] = sum_d / k_count;
        }
        (s_med, s_mad)
    } else {
        (med_vec, mad_vec)
    };

    // 4. Mapeo final en paralelo con Rayon e interpolación continua
    let mut out = vec![0.0f32; h * w];
    out.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row_out)| {
            let dy = y as f32 - center_y;
            let dy2 = dy * dy;
            let offset = y * w;
            let d_row = &detail_slice[offset..offset + w];
            let m_row = mask_slice.map(|m| &m[offset..offset + w]);

            for x in 0..w {
                let is_valid = match m_row {
                    Some(m) => m[x] && d_row[x].is_finite(),
                    None => d_row[x].is_finite(),
                };

                if is_valid {
                    let dx = x as f32 - center_x;
                    let r = (dx * dx + dy2).sqrt();

                    // Interpolación lineal continua en rm_vec
                    let (level, spread) = if r <= rm_vec[0] {
                        (smooth_med[0], smooth_mad[0])
                    } else if r >= rm_vec[n_sel - 1] {
                        (smooth_med[n_sel - 1], smooth_mad[n_sel - 1])
                    } else {
                        let idx = rm_vec.partition_point(|&v| v <= r).saturating_sub(1).min(n_sel - 2);
                        let r0 = rm_vec[idx];
                        let r1 = rm_vec[idx + 1];
                        let t = ((r - r0) / (r1 - r0).max(1e-6)).clamp(0.0, 1.0);
                        let lev = smooth_med[idx] + t * (smooth_med[idx + 1] - smooth_med[idx]);
                        let sp = smooth_mad[idx] + t * (smooth_mad[idx + 1] - smooth_mad[idx]);
                        (lev, sp)
                    };

                    let effective_spread = spread.max(noise_floor);
                    let norm_val = (d_row[x] - level) / (2.5f32 * effective_spread);
                    row_out[x] = 0.5f32 + 0.5f32 * norm_val.tanh();
                } else {
                    row_out[x] = 0.0f32;
                }
            }
        });

    out
}

/// 10. display_detail_rings_rs:
/// Mapeo de detalle con nivel y dispersión medidos anillo por anillo en torno al centro solar (AEL 1.3.4 / Issue #9).
/// Calcula mediana y 1.4826*MAD en anillos de ancho width_px con Rayon, suaviza en 1D e interpola
/// de forma continua sin efecto de escalonamiento. Mapeo: 0.5 + 0.5 * tanh((detail - level) / (2.5 * spread)).
#[pyfunction]
#[pyo3(signature = (detail, valid_mask=None, center_x=0.0, center_y=0.0, width_px=2.0, min_count=50, smooth_bins=3, noise_floor=1e-5))]
pub fn display_detail_rings_rs<'py>(
    py: Python<'py>,
    detail: PyReadonlyArray2<'py, f32>,
    valid_mask: Option<PyReadonlyArray2<'py, bool>>,
    center_x: f32,
    center_y: f32,
    width_px: f32,
    min_count: usize,
    smooth_bins: usize,
    noise_floor: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = detail.as_array().dim();
    let detail_slice = detail.as_slice().unwrap();
    let mask_slice = valid_mask.as_ref().map(|m| m.as_slice().unwrap());

    let out_vec = py.allow_threads(|| {
        display_detail_rings_raw(
            detail_slice,
            mask_slice,
            h,
            w,
            center_x,
            center_y,
            width_px,
            min_count,
            smooth_bins,
            noise_floor,
        )
    });

    Ok(Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py))
}

/// 11. accumulate_steerable_g2_rs:
/// Fusión matemática y acumulación de Steerable G2 (Freeman & Adelson 1991 / Weickert 1999).
/// Evalúa el tensor de estructura y la respuesta orientada mediante la identidad algebraica exacta
/// G2 = 0.5*(Hxx + Hyy) + 0.5*( (J11 - J22)*(Hxx - Hyy) + 4*J12*Hxy ) / sqrt((J11-J22)^2 + 4*J12^2)
/// sin llamadas trigonométricas ni asignación de memoria dinámica, acumulando -G2 * scale_weight.
#[pyfunction]
#[pyo3(signature = (acc, hxx, hyy, hxy, j11, j22, j12, scale_weight=1.0))]
pub fn accumulate_steerable_g2_rs<'py>(
    py: Python<'py>,
    acc: &Bound<'py, PyArray2<f32>>,
    hxx: PyReadonlyArray2<'py, f32>,
    hyy: PyReadonlyArray2<'py, f32>,
    hxy: PyReadonlyArray2<'py, f32>,
    j11: PyReadonlyArray2<'py, f32>,
    j22: PyReadonlyArray2<'py, f32>,
    j12: PyReadonlyArray2<'py, f32>,
    scale_weight: f32,
) -> PyResult<()> {
    let (_h, w) = hxx.as_array().dim();
    let hxx_slice = hxx.as_slice().unwrap();
    let hyy_slice = hyy.as_slice().unwrap();
    let hxy_slice = hxy.as_slice().unwrap();
    let j11_slice = j11.as_slice().unwrap();
    let j22_slice = j22.as_slice().unwrap();
    let j12_slice = j12.as_slice().unwrap();

    let acc_slice = unsafe {
        acc.as_slice_mut().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Error buffer acc: {}", e))
        })?
    };

    py.allow_threads(|| {
        acc_slice
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_acc)| {
                let offset = y * w;
                let hxx_row = &hxx_slice[offset..offset + w];
                let hyy_row = &hyy_slice[offset..offset + w];
                let hxy_row = &hxy_slice[offset..offset + w];
                let j11_row = &j11_slice[offset..offset + w];
                let j22_row = &j22_slice[offset..offset + w];
                let j12_row = &j12_slice[offset..offset + w];

                for x in 0..w {
                    let hxx_val = hxx_row[x];
                    let hyy_val = hyy_row[x];
                    let hxy_val = hxy_row[x];
                    let j11_val = j11_row[x];
                    let j22_val = j22_row[x];
                    let j12_val = j12_row[x];

                    let big_x = j11_val - j22_val;
                    let big_y = 2.0 * j12_val;
                    let r_sq = big_x * big_x + big_y * big_y;

                    let g2 = if r_sq > 1e-12 {
                        let r = r_sq.sqrt();
                        0.5 * (hxx_val + hyy_val) + 0.5 * (big_x * (hxx_val - hyy_val) + 2.0 * big_y * hxy_val) / r
                    } else {
                        0.5 * (hxx_val + hyy_val)
                    };

                    // Cresta positiva para filamentos brillantes: -G2
                    row_acc[x] += -g2 * scale_weight;
                }
            });
    });

    Ok(())
}

#[inline(always)]
fn keys_cubic_weight(t: f32) -> f32 {
    let at = t.abs();
    if at <= 1.0 {
        1.5 * at * at * at - 2.5 * at * at + 1.0
    } else if at < 2.0 {
        -0.5 * at * at * at + 2.5 * at * at - 4.0 * at + 2.0
    } else {
        0.0
    }
}

#[inline(always)]
fn sample_bicubic_keys(src: &[f32], h: usize, w: usize, x: f32, y: f32) -> f32 {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let fx = x - (ix as f32);
    let fy = y - (iy as f32);

    let wx = [
        keys_cubic_weight(1.0 + fx),
        keys_cubic_weight(fx),
        keys_cubic_weight(1.0 - fx),
        keys_cubic_weight(2.0 - fx),
    ];
    let wy = [
        keys_cubic_weight(1.0 + fy),
        keys_cubic_weight(fy),
        keys_cubic_weight(1.0 - fy),
        keys_cubic_weight(2.0 - fy),
    ];

    let h_isize = h as isize;
    let w_isize = w as isize;
    let mut sum = 0.0f32;

    for m in 0..4 {
        let py = reflect101(iy - 1 + (m as isize), h_isize);
        let row_offset = py * w;
        let mut row_sum = 0.0f32;
        for n in 0..4 {
            let px = reflect101(ix - 1 + (n as isize), w_isize);
            row_sum += src[row_offset + px] * wx[n];
        }
        sum += row_sum * wy[m];
    }
    sum
}

/// 12. compute_larson_sekanina_rs:
/// Gradiente rotacional de Larson-Sekanina (1984) evaluado con interpolación cúbica de Keys exacta (a = -0.5).
/// LS = 2*ln(L) - ln(L(+d_theta)) - ln(L(-d_theta)).
/// Excluye el interior del disco lunar y zonas enmascaradas (ej. protuberancias H-alfa).
#[pyfunction]
#[pyo3(signature = (lum, center_x, center_y, delta_theta_deg=1.0, radius_min=0.0, exclusion_mask=None, epsilon=1e-4))]
pub fn compute_larson_sekanina_rs<'py>(
    py: Python<'py>,
    lum: PyReadonlyArray2<'py, f32>,
    center_x: f32,
    center_y: f32,
    delta_theta_deg: f32,
    radius_min: f32,
    exclusion_mask: Option<PyReadonlyArray2<'py, bool>>,
    epsilon: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = lum.as_array().dim();
    let lum_slice = lum.as_slice().unwrap();
    let mask_slice = exclusion_mask.as_ref().map(|m| m.as_slice().unwrap());

    let rad = delta_theta_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();
    let eps = epsilon.max(1e-8);
    let r_min_sq = (radius_min.max(0.0) - 2.5).max(0.0).powi(2);

    let mut out = vec![0.0f32; h * w];

    py.allow_threads(|| {
        out.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                let offset = y * w;
                let lum_row = &lum_slice[offset..offset + w];
                let m_row = mask_slice.map(|m| &m[offset..offset + w]);

                for x in 0..w {
                    let dx = x as f32 - center_x;
                    let r_sq = dx * dx + dy2;

                    let is_excluded = match m_row {
                        Some(m) => m[x] || r_sq < r_min_sq,
                        None => r_sq < r_min_sq,
                    };

                    if is_excluded {
                        row_out[x] = 0.0;
                        continue;
                    }

                    let xp = center_x + dx * cos_a - dy * sin_a;
                    let yp = center_y + dx * sin_a + dy * cos_a;

                    let xm = center_x + dx * cos_a + dy * sin_a;
                    let ym = center_y - dx * sin_a + dy * cos_a;

                    let val_c = lum_row[x].max(eps);
                    let val_p = sample_bicubic_keys(lum_slice, h, w, xp, yp).max(eps);
                    let val_m = sample_bicubic_keys(lum_slice, h, w, xm, ym).max(eps);

                    let ls = 2.0 * val_c.ln() - val_p.ln() - val_m.ln();
                    row_out[x] = ls;
                }
            });
    });

    Ok(Array2::from_shape_vec((h, w), out).unwrap().into_pyarray(py))
}

/// 13. polar_circular_median_filter_rs:
/// Filtro de mediana móvil circular a lo largo de las columnas (eje azimutal theta)
/// para cada radio r en paralelo con Rayon, utilizado por el filtro SWAP (Seaton et al. 2023).
#[pyfunction]
#[pyo3(signature = (polar, window_size=121))]
pub fn polar_circular_median_filter_rs<'py>(
    py: Python<'py>,
    polar: PyReadonlyArray2<'py, f32>,
    window_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h_r, w_theta) = polar.as_array().dim();
    let polar_slice = polar.as_slice().unwrap();

    let win = if window_size % 2 == 0 { window_size + 1 } else { window_size };
    let half = (win / 2) as isize;

    let mut out = vec![0.0f32; h_r * w_theta];

    py.allow_threads(|| {
        out.par_chunks_exact_mut(w_theta)
            .enumerate()
            .for_each(|(r, row_out)| {
                let offset = r * w_theta;
                let row_in = &polar_slice[offset..offset + w_theta];
                let mut buf = vec![0.0f32; win];

                let w_isize = w_theta as isize;

                for c in 0..w_theta {
                    let c_isize = c as isize;
                    for k in 0..win {
                        let src_idx = (c_isize - half + (k as isize)).rem_euclid(w_isize) as usize;
                        buf[k] = row_in[src_idx];
                    }
                    row_out[c] = median_f32(&mut buf);
                }
            });
    });

    Ok(Array2::from_shape_vec((h_r, w_theta), out).unwrap().into_pyarray(py))
}

/// 14. compute_red_excess_mask_rs:
/// Detecta protuberancias H-alfa midiendo el exceso de cociente R/G anillo por anillo (AEL 1.4.2).
/// Marca píxeles en [r_min, r_max] con R/G > threshold_factor * median_ring(R/G).
#[pyfunction]
#[pyo3(signature = (rgb, center_x, center_y, radius_min, radius_max, threshold_factor=1.12))]
pub fn compute_red_excess_mask_rs<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    center_x: f32,
    center_y: f32,
    radius_min: f32,
    radius_max: f32,
    threshold_factor: f32,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let (h, w, c) = rgb.as_array().dim();
    if c < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Requiere imagen RGB"));
    }
    let rgb_slice = rgb.as_slice().unwrap();

    let width_px = 2.0f32;
    let inv_w = 1.0 / width_px;
    let r_min = radius_min.max(1.0);
    let r_max = radius_max.max(r_min + 1.0);
    let n_bins = (((r_max - r_min) * inv_w) as usize) + 2;

    let out_mask = py.allow_threads(|| {
        let mut ratios_by_bin: Vec<Vec<f32>> = vec![Vec::new(); n_bins];

        for y in 0..h {
            let dy = y as f32 - center_y;
            let dy2 = dy * dy;
            let row_offset = y * w * 3;
            for x in 0..w {
                let dx = x as f32 - center_x;
                let r = (dx * dx + dy2).sqrt();
                if r >= r_min && r <= r_max {
                    let b = (((r - r_min) * inv_w) as usize).min(n_bins - 1);
                    let p_idx = row_offset + x * 3;
                    let red = rgb_slice[p_idx];
                    let green = rgb_slice[p_idx + 1];
                    let ratio = red / green.max(1e-4);
                    ratios_by_bin[b].push(ratio);
                }
            }
        }

        let ring_medians: Vec<f32> = ratios_by_bin
            .par_iter_mut()
            .map(|vals| {
                if vals.len() >= 10 {
                    median_f32(vals)
                } else {
                    1.0
                }
            })
            .collect();

        let mut mask = vec![false; h * w];
        mask.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_mask)| {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                let row_offset = y * w * 3;
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    let r = (dx * dx + dy2).sqrt();
                    if r >= r_min && r <= r_max {
                        let b = (((r - r_min) * inv_w) as usize).min(n_bins - 1);
                        let p_idx = row_offset + x * 3;
                        let red = rgb_slice[p_idx];
                        let green = rgb_slice[p_idx + 1];
                        let ratio = red / green.max(1e-4);
                        let med = ring_medians[b];
                        if ratio > threshold_factor * med {
                            row_mask[x] = true;
                        }
                    }
                }
            });

        mask
    });

    Ok(Array2::from_shape_vec((h, w), out_mask).unwrap().into_pyarray(py))
}

const B3K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// 15. paired_b3_step_rs:
/// Paso à-trous B3 con soporte bilateral de pares simétricos (AEL 1.3.4 / 1.3.5 / Issue #4).
/// Garantiza que los taps opuestos estén presentes para evitar sesgo en los bordes de datos
/// y aplica pesos bilaterales a partir de la guía en paralelo con Rayon.
pub(crate) fn paired_b3_step_raw(
    c_slice: &[f32],
    m_slice: &[bool],
    h: usize,
    w: usize,
    step: usize,
    bilateral: bool,
    guide_slice: Option<&[f32]>,
    out: &mut [f32],
    compl: &mut [f32],
    pair: &mut [bool],
) {
    let step_i = step as isize;
    let h_i = h as isize;
    let w_i = w as isize;

    out.par_chunks_exact_mut(w)
        .zip(compl.par_chunks_exact_mut(w))
        .zip(pair.par_chunks_exact_mut(w))
        .enumerate()
        .for_each(|(y, ((out_row, compl_row), pair_row))| {
            let y_i = y as isize;
            let row_offset = y * w;

            for x in 0..w {
                let idx = row_offset + x;
                if !m_slice[idx] {
                    continue;
                }
                let x_i = x as isize;
                let ctr = c_slice[idx];
                let ctg = match guide_slice {
                    Some(g) => g[idx],
                    None => ctr,
                };

                let mut num = 0.0f64;
                let mut den = 0.0f64;
                let mut ks = 0.0f32;
                let mut kn = 0.0f32;

                if bilateral {
                    let mut valid_taps = [(0.0f32, 0.0f32, 0.0f32); 25];
                    let mut n_taps = 0usize;
                    let mut mo = 0.0f64;
                    let mut mo2 = 0.0f64;
                    let mut ma = 0.0f64;

                    for dy in -2..=2 {
                        let ky = B3K[(dy + 2) as usize];
                        let y1 = y_i + dy * step_i;
                        let y2 = y_i - dy * step_i;
                        for dx in -2..=2 {
                            let kx = B3K[(dx + 2) as usize];
                            let k = ky * kx;
                            let x1 = x_i + dx * step_i;
                            let x2 = x_i - dx * step_i;

                            let g = if dy == 0 && dx == 0 {
                                true
                            } else {
                                y1 >= 0
                                    && y1 < h_i
                                    && x1 >= 0
                                    && x1 < w_i
                                    && m_slice[(y1 as usize) * w + (x1 as usize)]
                                    && y2 >= 0
                                    && y2 < h_i
                                    && x2 >= 0
                                    && x2 < w_i
                                    && m_slice[(y2 as usize) * w + (x2 as usize)]
                            };

                            if g {
                                let tap_idx = (y1 as usize) * w + (x1 as usize);
                                let v = c_slice[tap_idx];
                                let vg = match guide_slice {
                                    Some(g_s) => g_s[tap_idx],
                                    None => v,
                                };
                                valid_taps[n_taps] = (k, v, vg);
                                n_taps += 1;
                                ks += k;
                                if dy != 0 || dx != 0 {
                                    kn += k;
                                }
                                let kd = k as f64;
                                let dl = (vg as f64) - (ctg as f64);
                                mo += kd * dl;
                                mo2 += kd * dl * dl;
                                ma += kd;
                            }
                        }
                    }

                    let ma_safe = ma.max(1e-20);
                    let mean = mo / ma_safe;
                    let vv = (mo2 / ma_safe - mean * mean).max(1e-20);
                    let inv_two_vv = 0.5 / vv;

                    for &(k, v, vg) in &valid_taps[..n_taps] {
                        let diff = (ctg as f64) - (vg as f64);
                        let wt = (k as f64) * (-diff * diff * inv_two_vv).exp();
                        num += wt * ((v - ctr) as f64);
                        den += wt;
                    }
                } else {
                    for dy in -2..=2 {
                        let ky = B3K[(dy + 2) as usize];
                        let y1 = y_i + dy * step_i;
                        let y2 = y_i - dy * step_i;
                        for dx in -2..=2 {
                            let kx = B3K[(dx + 2) as usize];
                            let k = ky * kx;
                            let x1 = x_i + dx * step_i;
                            let x2 = x_i - dx * step_i;

                            let g = if dy == 0 && dx == 0 {
                                true
                            } else {
                                y1 >= 0
                                    && y1 < h_i
                                    && x1 >= 0
                                    && x1 < w_i
                                    && m_slice[(y1 as usize) * w + (x1 as usize)]
                                    && y2 >= 0
                                    && y2 < h_i
                                    && x2 >= 0
                                    && x2 < w_i
                                    && m_slice[(y2 as usize) * w + (x2 as usize)]
                            };

                            if g {
                                let tap_idx = (y1 as usize) * w + (x1 as usize);
                                let v = c_slice[tap_idx];
                                if dy != 0 || dx != 0 {
                                    kn += k;
                                }
                                let wt = k as f64;
                                num += wt * ((v - ctr) as f64);
                                den += wt;
                                ks += k;
                            }
                        }
                    }
                }

                out_row[x] = (ctr as f64 + num / den.max(1e-20)) as f32;
                compl_row[x] = ks;
                pair_row[x] = kn > 0.0;
            }
        });
}

/// 15. paired_b3_step_rs:
/// Paso à-trous B3 con soporte bilateral de pares simétricos (AEL 1.3.4 / 1.3.5 / Issue #4).
/// Garantiza que los taps opuestos estén presentes para evitar sesgo en los bordes de datos
/// y aplica pesos bilaterales a partir de la guía en paralelo con Rayon.
#[pyfunction]
#[pyo3(signature = (c, m, step, bilateral=false, guide=None))]
pub fn paired_b3_step_rs<'py>(
    py: Python<'py>,
    c: PyReadonlyArray2<'py, f32>,
    m: PyReadonlyArray2<'py, bool>,
    step: usize,
    bilateral: bool,
    guide: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<bool>>,
)> {
    let (h, w) = c.as_array().dim();
    let c_slice = c.as_slice().unwrap();
    let m_slice = m.as_slice().unwrap();
    let guide_slice = guide.as_ref().map(|g| g.as_slice().unwrap());

    let (out_vec, compl_vec, pair_vec) = py.allow_threads(|| {
        let mut out = vec![0.0f32; h * w];
        let mut compl = vec![0.0f32; h * w];
        let mut pair = vec![false; h * w];
        paired_b3_step_raw(
            c_slice,
            m_slice,
            h,
            w,
            step,
            bilateral,
            guide_slice,
            &mut out,
            &mut compl,
            &mut pair,
        );
        (out, compl, pair)
    });

    Ok((
        Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py),
        Array2::from_shape_vec((h, w), compl_vec).unwrap().into_pyarray(py),
        Array2::from_shape_vec((h, w), pair_vec).unwrap().into_pyarray(py),
    ))
}

pub(crate) fn b3_normalized_raw(
    a_slice: &[f32],
    m_slice: &[bool],
    h: usize,
    w: usize,
    step: usize,
    out: &mut [f32],
) {
    let step_i = step as isize;
    let h_i = h as isize;
    let w_i = w as isize;

    let conv_b3 = |src: &[f32]| -> Vec<f32> {
        let mut tmp = vec![0.0f32; h * w];
        tmp.par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, tmp_row)| {
                let y_i = y as isize;
                for dy in -2..=2 {
                    let ky = B3K[(dy + 2) as usize];
                    let yy = y_i + dy * step_i;
                    if yy >= 0 && yy < h_i {
                        let src_row = &src[(yy as usize) * w..(yy as usize + 1) * w];
                        for x in 0..w {
                            tmp_row[x] += ky * src_row[x];
                        }
                    }
                }
            });

        let mut res = vec![0.0f32; h * w];
        res.par_chunks_exact_mut(w)
            .zip(tmp.par_chunks_exact(w))
            .for_each(|(out_row, tmp_row)| {
                for x in 0..w {
                    let x_i = x as isize;
                    let mut sum = 0.0f32;
                    for dx in -2..=2 {
                        let kx = B3K[(dx + 2) as usize];
                        let xx = x_i + dx * step_i;
                        if xx >= 0 && xx < w_i {
                            sum += kx * tmp_row[xx as usize];
                        }
                    }
                    out_row[x] = sum;
                }
            });

        res
    };

    let mut masked_a = vec![0.0f32; h * w];
    let mut m_f32 = vec![0.0f32; h * w];
    for i in 0..(h * w) {
        if m_slice[i] {
            masked_a[i] = a_slice[i];
            m_f32[i] = 1.0;
        }
    }

    let conv_a = conv_b3(&masked_a);
    let conv_m = conv_b3(&m_f32);

    out.par_chunks_exact_mut(w)
        .zip(conv_a.par_chunks_exact(w))
        .zip(conv_m.par_chunks_exact(w))
        .for_each(|((res_row, ca_row), cm_row)| {
            for x in 0..w {
                res_row[x] = ca_row[x] / cm_row[x].max(1e-20);
            }
        });
}

/// 16. b3_normalized_rs:
/// Convolución separable B3 à-trous 2D normalizada por la máscara de soporte observado:
/// conv(masked_a, step) / max(conv(m, step), 1e-20).
#[pyfunction]
#[pyo3(signature = (a, m, step))]
pub fn b3_normalized_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    m: PyReadonlyArray2<'py, bool>,
    step: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (h, w) = a.as_array().dim();
    let a_slice = a.as_slice().unwrap();
    let m_slice = m.as_slice().unwrap();

    let out_vec = py.allow_threads(|| {
        let mut out = vec![0.0f32; h * w];
        b3_normalized_raw(a_slice, m_slice, h, w, step, &mut out);
        out
    });

    Ok(Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py))
}

fn yvv_gaussian_blur_1d(src: &[f32], dst: &mut [f32], n: usize, sigma: f32) {
    if sigma <= 0.5 {
        dst.copy_from_slice(src);
        return;
    }
    let q = if sigma >= 2.5 {
        0.98711 * (sigma as f64) - 0.96330
    } else {
        3.97156 - 4.14554 * (1.0 - 0.26891 * (sigma as f64)).max(0.0).sqrt()
    };

    let b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    let b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    let b2 = -(1.4281 * q * q + 1.26661 * q * q * q);
    let b3 = 0.422205 * q * q * q;

    let b = 1.0 - (b1 + b2 + b3) / b0;
    let b1_norm = b1 / b0;
    let b2_norm = b2 / b0;
    let b3_norm = b3 / b0;

    let mut w1 = src[0] as f64;
    let mut w2 = src[0] as f64;
    let mut w3 = src[0] as f64;
    let mut fwd = vec![0.0f64; n];

    for i in 0..n {
        let val = b * (src[i] as f64) + b1_norm * w1 + b2_norm * w2 + b3_norm * w3;
        fwd[i] = val;
        w3 = w2;
        w2 = w1;
        w1 = val;
    }

    let mut w1_b = fwd[n - 1];
    let mut w2_b = fwd[n - 1];
    let mut w3_b = fwd[n - 1];

    for i in (0..n).rev() {
        let val = b * fwd[i] + b1_norm * w1_b + b2_norm * w2_b + b3_norm * w3_b;
        dst[i] = val as f32;
        w3_b = w2_b;
        w2_b = w1_b;
        w1_b = val;
    }
}

fn transpose_rayon(
    in_mat: &[f32],
    out_t: &mut [f32],
    rows: usize,
    cols: usize,
) {
    out_t
        .par_chunks_exact_mut(rows)
        .enumerate()
        .for_each(|(c, col_out)| {
            for r in 0..rows {
                col_out[r] = in_mat[r * cols + c];
            }
        });
}

fn gaussian_blur_2d_separable(
    src: &[f32],
    dst: &mut [f32],
    h: usize,
    w: usize,
    sigma: f32,
    tmp_buf: &mut [f32],
    trans_in: &mut [f32],
    trans_out: &mut [f32],
) {
    if sigma <= 0.0 {
        dst.copy_from_slice(src);
        return;
    }

    // Pass 1: Horizontal IIR blur (h x w) -> tmp_buf
    tmp_buf
        .par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(out_row, in_row)| {
            yvv_gaussian_blur_1d(in_row, out_row, w, sigma);
        });

    // Pass 2: Transpose tmp_buf (h x w) -> trans_in (w x h)
    transpose_rayon(tmp_buf, trans_in, h, w);

    // Pass 3: Horizontal IIR blur on trans_in (w rows of length h) -> trans_out
    trans_out
        .par_chunks_exact_mut(h)
        .zip(trans_in.par_chunks_exact(h))
        .for_each(|(out_row, in_row)| {
            yvv_gaussian_blur_1d(in_row, out_row, h, sigma);
        });

    // Pass 4: Transpose trans_out (w x h) -> dst (h x w)
    transpose_rayon(trans_out, dst, w, h);
}

/// 17. apply_wow_edge_aware_rs:
/// Filtro WOW bilateral guiado completo ejecutado 100% en Rust con paralelismo Rayon.
/// Descompone la corona en n_scales à-trous B3 con pares simétricos, potencia local b3_normalized,
/// desenfoque gaussiano separable exacto para sustracción de fondo, normalización radial y overlay blend.
#[pyfunction]
#[pyo3(signature = (
    image,
    center_x,
    center_y,
    radius_min = 10.0,
    n_scales = 7,
    guided = true,
    completeness_lo = 0.6,
    completeness_hi = 0.995,
    blend_ratio = 0.50,
    use_detail_rings = true
))]
pub fn apply_wow_edge_aware_rs<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, f32>,
    center_x: f32,
    center_y: f32,
    radius_min: f32,
    n_scales: usize,
    guided: bool,
    completeness_lo: f32,
    completeness_hi: f32,
    blend_ratio: f32,
    use_detail_rings: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let shape = image.shape().to_vec();
    let is_3d = shape.len() == 3;
    let (h, w) = if is_3d {
        (shape[0], shape[1])
    } else if shape.len() == 2 {
        (shape[0], shape[1])
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "image must have 2 or 3 dimensions",
        ));
    };

    let img_slice = image.as_slice().unwrap();

    let out_vec = py.allow_threads(|| {
        let n_pixels = h * w;
        let mut lum = vec![0.0f32; n_pixels];
        if is_3d {
            lum.par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(y, lum_row)| {
                    let base_idx = y * w * 3;
                    for x in 0..w {
                        let idx3 = base_idx + x * 3;
                        lum_row[x] = 0.2126 * img_slice[idx3]
                            + 0.7152 * img_slice[idx3 + 1]
                            + 0.0722 * img_slice[idx3 + 2];
                    }
                });
        } else {
            lum.copy_from_slice(img_slice);
        }

        let min_r = (radius_min - 2.5).max(0.0);
        let min_r_sq = min_r * min_r;
        let mut valid_mask = vec![false; n_pixels];
        valid_mask
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, mask_row)| {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    mask_row[x] = (dx * dx + dy2) >= min_r_sq;
                }
            });

        let mut c = vec![0.0f32; n_pixels];
        for i in 0..n_pixels {
            if valid_mask[i] {
                c[i] = lum[i];
            }
        }
        let mut out = vec![0.0f32; n_pixels];

        let mut nxt = vec![0.0f32; n_pixels];
        let mut compl = vec![0.0f32; n_pixels];
        let mut pair = vec![false; n_pixels];
        let mut guide_buf = vec![0.0f32; n_pixels];
        let mut guide_compl = vec![0.0f32; n_pixels];
        let mut guide_pair = vec![false; n_pixels];
        let mut wave_sq = vec![0.0f32; n_pixels];
        let mut wave = vec![0.0f32; n_pixels];
        let mut local_power = vec![0.0f32; n_pixels];
        let mut g = vec![0.0f32; n_pixels];
        let mut weight = vec![0.0f32; n_pixels];

        let mut inp_num = vec![0.0f32; n_pixels];
        let mut inp_den = vec![0.0f32; n_pixels];
        let mut bg_num = vec![0.0f32; n_pixels];
        let mut bg_den = vec![0.0f32; n_pixels];

        let mut tmp_buf = vec![0.0f32; n_pixels];
        let mut trans_in = vec![0.0f32; n_pixels];
        let mut trans_out = vec![0.0f32; n_pixels];

        for s in 0..n_scales {
            let step = 1 << s;

            let guide_slice_opt = if guided {
                paired_b3_step_raw(
                    &c,
                    &valid_mask,
                    h,
                    w,
                    step,
                    false,
                    None,
                    &mut guide_buf,
                    &mut guide_compl,
                    &mut guide_pair,
                );
                Some(guide_buf.as_slice())
            } else {
                None
            };

            paired_b3_step_raw(
                &c,
                &valid_mask,
                h,
                w,
                step,
                true,
                guide_slice_opt,
                &mut nxt,
                &mut compl,
                &mut pair,
            );

            nxt.par_iter_mut()
                .zip(c.par_iter())
                .zip(valid_mask.par_iter())
                .zip(wave.par_iter_mut())
                .zip(wave_sq.par_iter_mut())
                .for_each(|((((nxt_val, &c_val), &m_val), w_val), w_sq_val)| {
                    if !m_val {
                        *nxt_val = 0.0;
                        *w_val = 0.0;
                        *w_sq_val = 0.0;
                    } else {
                        let diff = c_val - *nxt_val;
                        let eps_thresh = 8.0 * f32::EPSILON * c_val.abs().max(nxt_val.abs());
                        let filtered_wave = if diff.abs() <= eps_thresh { 0.0 } else { diff };
                        *w_val = filtered_wave;
                        *w_sq_val = filtered_wave * filtered_wave;
                    }
                });

            b3_normalized_raw(&wave_sq, &valid_mask, h, w, step, &mut local_power);

            let lo = completeness_lo;
            let hi = completeness_hi;
            let inv_range = 1.0 / (hi - lo).max(1e-6);

            g.par_iter_mut()
                .zip(weight.par_iter_mut())
                .zip(wave.par_iter())
                .zip(local_power.par_iter())
                .zip(compl.par_iter())
                .zip(valid_mask.par_iter())
                .for_each(|(((((g_val, wt_val), &w_val), &lp_val), &cp_val), &m_val)| {
                    if m_val && cp_val > 0.0 {
                        let denom = lp_val.max(1e-20).sqrt();
                        *g_val = w_val / denom;
                        if s >= 3 {
                            let q = ((cp_val - lo) * inv_range).clamp(0.0, 1.0);
                            *wt_val = q * q * (3.0 - 2.0 * q);
                        } else {
                            *wt_val = 1.0;
                        }
                    } else {
                        *g_val = 0.0;
                        *wt_val = 0.0;
                    }
                });

            let (sum_whole, cnt_whole) = g
                .par_iter()
                .zip(compl.par_iter())
                .zip(valid_mask.par_iter())
                .fold(
                    || (0.0f64, 0usize),
                    |(acc_sum, acc_cnt), ((&g_val, &cp_val), &m_val)| {
                        if m_val && cp_val >= 0.999 {
                            (acc_sum + g_val as f64, acc_cnt + 1)
                        } else {
                            (acc_sum, acc_cnt)
                        }
                    },
                )
                .reduce(|| (0.0f64, 0usize), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));

            if cnt_whole > 0 {
                let mean_whole = (sum_whole / cnt_whole as f64) as f32;
                g.par_iter_mut()
                    .zip(valid_mask.par_iter())
                    .for_each(|(g_val, &m_val)| {
                        if m_val {
                            *g_val -= mean_whole;
                        }
                    });
            }

            inp_num
                .par_iter_mut()
                .zip(inp_den.par_iter_mut())
                .zip(g.par_iter())
                .zip(weight.par_iter())
                .zip(pair.par_iter())
                .for_each(|((((num, den), &g_val), &wt_val), &p_val)| {
                    let pf = if p_val { 1.0f32 } else { 0.0f32 };
                    *num = g_val * wt_val * pf;
                    *den = wt_val * pf;
                });

            let sigma_bg = (4.0 * step as f32).clamp(32.0, 150.0);
            gaussian_blur_2d_separable(
                &inp_num,
                &mut bg_num,
                h,
                w,
                sigma_bg,
                &mut tmp_buf,
                &mut trans_in,
                &mut trans_out,
            );
            gaussian_blur_2d_separable(
                &inp_den,
                &mut bg_den,
                h,
                w,
                sigma_bg,
                &mut tmp_buf,
                &mut trans_in,
                &mut trans_out,
            );

            out.par_iter_mut()
                .zip(c.par_iter_mut())
                .zip(nxt.par_iter())
                .zip(g.par_iter())
                .zip(weight.par_iter())
                .zip(pair.par_iter())
                .zip(bg_num.par_iter())
                .zip(bg_den.par_iter())
                .zip(valid_mask.par_iter())
                .for_each(|((((((((out_val, c_val), &nxt_val), &g_val), &wt_val), &p_val), &num_val), &den_val), &m_val)| {
                    if m_val {
                        let pf = if p_val { 1.0f32 } else { 0.0f32 };
                        let local_bg = num_val / den_val.max(1e-6);
                        let g_corrected = g_val - pf * local_bg;
                        *out_val += wt_val * g_corrected;
                        *c_val = nxt_val;
                    } else {
                        *c_val = 0.0;
                    }
                });
        }

        let mut raw_detail = out;
        for i in 0..n_pixels {
            if !valid_mask[i] {
                raw_detail[i] = 0.0;
            }
        }

        let norm_detail = if use_detail_rings {
            display_detail_rings_raw(
                &raw_detail,
                Some(&valid_mask),
                h,
                w,
                center_x,
                center_y,
                2.0,
                50,
                3,
                1e-5,
            )
        } else {
            let mut valid_vals: Vec<f32> = raw_detail
                .iter()
                .zip(valid_mask.iter())
                .filter_map(|(&v, &m)| if m { Some(v) } else { None })
                .collect();
            let med = median_f32(&mut valid_vals);
            let mut diffs: Vec<f32> = valid_vals.iter().map(|&v| (v - med).abs()).collect();
            let mad = 1.4826f32 * median_f32(&mut diffs);
            let denom = (2.5f32 * mad).max(1e-5);
            let mut norm = vec![0.0f32; n_pixels];
            norm.par_iter_mut()
                .zip(raw_detail.par_iter())
                .zip(valid_mask.par_iter())
                .for_each(|((n_val, &d_val), &m_val)| {
                    if m_val {
                        *n_val = 0.5f32 + 0.5f32 * ((d_val - med) / denom).tanh();
                    }
                });
            norm
        };

        let blend = blend_ratio.clamp(0.0, 1.0);
        let total_out_size = if is_3d { n_pixels * 3 } else { n_pixels };
        let mut final_out = vec![0.0f32; total_out_size];

        if is_3d {
            final_out
                .par_chunks_exact_mut(w * 3)
                .enumerate()
                .for_each(|(y, row_out)| {
                    let base_pix = y * w;
                    let base_src = y * w * 3;
                    for x in 0..w {
                        let pix_idx = base_pix + x;
                        let src_idx = base_src + x * 3;
                        let nd = norm_detail[pix_idx];
                        let is_valid = valid_mask[pix_idx];

                        for c in 0..3 {
                            if !is_valid {
                                row_out[x * 3 + c] = 0.0;
                            } else {
                                let orig = img_slice[src_idx + c].clamp(0.0, 1.0);
                                let ov = if orig < 0.5 {
                                    2.0 * orig * nd
                                } else {
                                    1.0 - 2.0 * (1.0 - orig) * (1.0 - nd)
                                };
                                let fin = (1.0 - blend) * orig + blend * ov;
                                row_out[x * 3 + c] = fin.clamp(0.0, 1.0);
                            }
                        }
                    }
                });
        } else {
            final_out
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(y, row_out)| {
                    let base_pix = y * w;
                    for x in 0..w {
                        let pix_idx = base_pix + x;
                        let nd = norm_detail[pix_idx];
                        let is_valid = valid_mask[pix_idx];
                        if !is_valid {
                            row_out[x] = 0.0;
                        } else {
                            let orig = img_slice[pix_idx].clamp(0.0, 1.0);
                            let ov = if orig < 0.5 {
                                2.0 * orig * nd
                            } else {
                                1.0 - 2.0 * (1.0 - orig) * (1.0 - nd)
                            };
                            let fin = (1.0 - blend) * orig + blend * ov;
                            row_out[x] = fin.clamp(0.0, 1.0);
                        }
                    }
                });
        }

        final_out
    });

    let arr = ArrayD::from_shape_vec(IxDyn(&shape), out_vec).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create ArrayD: {}", e))
    })?;
    Ok(arr.into_pyarray(py))
}

#[inline(always)]
fn soft_shoulder_val(peak: f32, threshold: f32, ceiling: f32) -> f32 {
    if peak > threshold {
        let p = peak - threshold;
        let span = (ceiling - threshold).max(1e-4);
        threshold + span * (1.0 - (-p / span).exp())
    } else {
        peak
    }
}

/// 17. apply_asinh_stretch_rs:
/// Estirado de arco seno hiperbólico (asinh) con hombro suave de altas luces.
/// Preserva ratios cromáticos exactos R:G:B en 3D (H, W, 3) y soporta imágenes monocromas 2D (H, W).
#[pyfunction]
#[pyo3(signature = (image, stretch_factor=30.0, black_point=0.0, shoulder_threshold=0.70, highlight_ceiling=1.0))]
pub fn apply_asinh_stretch_rs<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    stretch_factor: f32,
    black_point: f32,
    shoulder_threshold: f32,
    highlight_ceiling: f32,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(arr3) = image.extract::<PyReadonlyArray3<f32>>() {
        let (h, w, c) = arr3.as_array().dim();
        let img_slice = arr3.as_slice().unwrap();

        let out_vec = py.allow_threads(|| {
            let total = h * w * c;
            let mut out = vec![0.0f32; total];

            if stretch_factor <= 1.0 {
                out.par_iter_mut()
                    .zip(img_slice.par_iter())
                    .for_each(|(o, &v)| {
                        *o = (v - black_point).clamp(0.0, 1.0);
                    });
            } else {
                let beta = stretch_factor;
                let inv_denom = 1.0 / beta.asinh();

                if c == 3 {
                    let row_stride = w * 3;
                    out.par_chunks_exact_mut(row_stride)
                        .zip(img_slice.par_chunks_exact(row_stride))
                        .for_each(|(out_row, in_row)| {
                            for x in 0..w {
                                let idx = x * 3;
                                let r = (in_row[idx] - black_point).max(0.0);
                                let g = (in_row[idx + 1] - black_point).max(0.0);
                                let b = (in_row[idx + 2] - black_point).max(0.0);

                                let ur = (beta * r).asinh() * inv_denom;
                                let ug = (beta * g).asinh() * inv_denom;
                                let ub = (beta * b).asinh() * inv_denom;

                                let peak = ur.max(ug).max(ub);
                                let factor = if peak > 1e-6 {
                                    soft_shoulder_val(peak, shoulder_threshold, highlight_ceiling) / peak
                                } else {
                                    1.0
                                };

                                out_row[idx] = (ur * factor).clamp(0.0, 1.0);
                                out_row[idx + 1] = (ug * factor).clamp(0.0, 1.0);
                                out_row[idx + 2] = (ub * factor).clamp(0.0, 1.0);
                            }
                        });
                } else {
                    let row_stride = w * c;
                    out.par_chunks_exact_mut(row_stride)
                        .zip(img_slice.par_chunks_exact(row_stride))
                        .for_each(|(out_row, in_row)| {
                            for x in 0..w {
                                let base = x * c;
                                let mut peak = 0.0f32;
                                for ch in 0..c {
                                    let v = (in_row[base + ch] - black_point).max(0.0);
                                    let u = (beta * v).asinh() * inv_denom;
                                    out_row[base + ch] = u;
                                    if u > peak {
                                        peak = u;
                                    }
                                }
                                let factor = if peak > 1e-6 {
                                    soft_shoulder_val(peak, shoulder_threshold, highlight_ceiling) / peak
                                } else {
                                    1.0
                                };
                                for ch in 0..c {
                                    out_row[base + ch] = (out_row[base + ch] * factor).clamp(0.0, 1.0);
                                }
                            }
                        });
                }
            }

            out
        });

        let py_arr = Array3::from_shape_vec((h, w, c), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(py_arr.into_pyarray(py).into_any())
    } else if let Ok(arr2) = image.extract::<PyReadonlyArray2<f32>>() {
        let (h, w) = arr2.as_array().dim();
        let img_slice = arr2.as_slice().unwrap();

        let out_vec = py.allow_threads(|| {
            let total = h * w;
            let mut out = vec![0.0f32; total];

            if stretch_factor <= 1.0 {
                out.par_iter_mut()
                    .zip(img_slice.par_iter())
                    .for_each(|(o, &v)| {
                        *o = (v - black_point).clamp(0.0, 1.0);
                    });
            } else {
                let beta = stretch_factor;
                let inv_denom = 1.0 / beta.asinh();

                out.par_chunks_exact_mut(w)
                    .zip(img_slice.par_chunks_exact(w))
                    .for_each(|(out_row, in_row)| {
                        for x in 0..w {
                            let v = (in_row[x] - black_point).max(0.0);
                            let u = (beta * v).asinh() * inv_denom;
                            let sh = soft_shoulder_val(u, shoulder_threshold, highlight_ceiling);
                            out_row[x] = sh.clamp(0.0, 1.0);
                        }
                    });
            }

            out
        });

        let py_arr = Array2::from_shape_vec((h, w), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(py_arr.into_pyarray(py).into_any())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "image debe ser un ndarray float32 de 2 o 3 dimensiones",
        ))
    }
}

/// 18. balance_coronal_white_balance_rs:
/// Balance de blancos coronal neutro sobre el anillo coronal medio [ring_lo * R, ring_hi * R].
/// Devuelve (imagen_balanceada, (ganancia_r, ganancia_g, ganancia_b)).
#[pyfunction]
#[pyo3(signature = (image, center_x, center_y, radius, ring_lo=1.2, ring_hi=1.8))]
pub fn balance_coronal_white_balance_rs<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    center_x: f32,
    center_y: f32,
    radius: f32,
    ring_lo: f32,
    ring_hi: f32,
) -> PyResult<(Bound<'py, PyAny>, (f32, f32, f32))> {
    if let Ok(arr3) = image.extract::<PyReadonlyArray3<f32>>() {
        let (h, w, c) = arr3.as_array().dim();
        let img_slice = arr3.as_slice().unwrap();

        if c != 3 {
            let out_arr = Array3::from_shape_vec((h, w, c), img_slice.to_vec())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
            return Ok((out_arr.into_pyarray(py).into_any(), (1.0, 1.0, 1.0)));
        }

        let r_lo = radius * ring_lo;
        let r_hi = radius * ring_hi;
        let r_lo_sq = r_lo * r_lo;
        let r_hi_sq = r_hi * r_hi;

        let (balanced_vec, gains) = py.allow_threads(|| {
            let mut r_vals = Vec::new();
            let mut g_vals = Vec::new();
            let mut b_vals = Vec::new();

            for y in 0..h {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                let row_offset = y * w * 3;
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    let d2 = dx * dx + dy2;
                    if d2 >= r_lo_sq && d2 <= r_hi_sq {
                        let idx = row_offset + x * 3;
                        r_vals.push(img_slice[idx]);
                        g_vals.push(img_slice[idx + 1]);
                        b_vals.push(img_slice[idx + 2]);
                    }
                }
            }

            if r_vals.len() < 100 {
                return (img_slice.to_vec(), (1.0f32, 1.0f32, 1.0f32));
            }

            let med_r = median_f32(&mut r_vals);
            let med_g = median_f32(&mut g_vals);
            let med_b = median_f32(&mut b_vals);

            let g_val = if med_g > 1e-6 {
                med_g
            } else {
                (med_r + med_g + med_b) / 3.0
            };

            let f_r = (g_val / med_r.max(1e-6)).clamp(0.4, 2.5);
            let f_g = (g_val / med_g.max(1e-6)).clamp(0.4, 2.5);
            let f_b = (g_val / med_b.max(1e-6)).clamp(0.4, 2.5);

            let mut out = vec![0.0f32; h * w * 3];
            let row_stride = w * 3;
            out.par_chunks_exact_mut(row_stride)
                .zip(img_slice.par_chunks_exact(row_stride))
                .for_each(|(out_row, in_row)| {
                    for x in 0..w {
                        let idx = x * 3;
                        out_row[idx] = (in_row[idx] * f_r).clamp(0.0, 1.0);
                        out_row[idx + 1] = (in_row[idx + 1] * f_g).clamp(0.0, 1.0);
                        out_row[idx + 2] = (in_row[idx + 2] * f_b).clamp(0.0, 1.0);
                    }
                });

            (out, (f_r, f_g, f_b))
        });

        let out_arr = Array3::from_shape_vec((h, w, 3), balanced_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok((out_arr.into_pyarray(py).into_any(), gains))
    } else if let Ok(arr2) = image.extract::<PyReadonlyArray2<f32>>() {
        let (h, w) = arr2.as_array().dim();
        let img_slice = arr2.as_slice().unwrap();
        let out_arr = Array2::from_shape_vec((h, w), img_slice.to_vec())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok((out_arr.into_pyarray(py).into_any(), (1.0, 1.0, 1.0)))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "image debe ser un ndarray float32 de 2 o 3 dimensiones",
        ))
    }
}

// ----------------------------------------------------------------------------
// Tarea 3: Optimización subpíxel de centro por flujo circular sin OpenCV
// ----------------------------------------------------------------------------

#[inline(always)]
fn border_reflect(p: isize, l: usize) -> usize {
    if l <= 1 {
        return 0;
    }
    let m = (2 * l) as isize;
    let r = p.rem_euclid(m) as usize;
    if r >= l {
        2 * l - 1 - r
    } else {
        r
    }
}

#[inline(always)]
fn interp_bilinear_reflect(gray: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let wx = x - (x0 as f32);
    let wy = y - (y0 as f32);

    let rx0 = border_reflect(x0, w);
    let rx1 = border_reflect(x1, w);
    let ry0 = border_reflect(y0, h);
    let ry1 = border_reflect(y1, h);

    let v00 = gray[ry0 * w + rx0];
    let v01 = gray[ry0 * w + rx1];
    let v10 = gray[ry1 * w + rx0];
    let v11 = gray[ry1 * w + rx1];

    (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)
}

#[inline(always)]
fn percentile_linear_180(sorted: &[f32; 180], q: f32) -> f32 {
    let idx_f = (q / 100.0) * 179.0;
    let i0 = (idx_f.floor() as usize).min(178);
    let i1 = i0 + 1;
    let frac = idx_f - (i0 as f32);
    (1.0 - frac) * sorted[i0] + frac * sorted[i1]
}

#[pyfunction]
#[pyo3(signature = (gray, init_cx, init_cy, radius, mode="solar", max_search_dist=None))]
pub fn optimize_center_circular_flux_rs<'py>(
    py: Python<'py>,
    gray: PyReadonlyArray2<'py, f32>,
    init_cx: f32,
    init_cy: f32,
    radius: f32,
    mode: &str,
    max_search_dist: Option<f32>,
) -> PyResult<(f32, f32)> {
    let (h, w) = gray.as_array().dim();
    let gray_slice = gray.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array no contiguo: {}", e))
    })?;

    let mode_solar = mode == "solar";
    let r_in = (1.0f32).max(radius - 2.5);
    let r_out = radius + 2.5;
    let r_int = radius * 0.50;

    let max_dev = match max_search_dist {
        Some(d) if d > 0.0 => d,
        _ => (35.0f32).min((12.0f32).max(radius * 0.12)),
    };

    let result = py.allow_threads(|| {
        let mut cos_a = [0.0f32; 180];
        let mut sin_a = [0.0f32; 180];
        for i in 0..180 {
            let a = (2.0 * std::f32::consts::PI * (i as f32)) / 180.0;
            cos_a[i] = a.cos();
            sin_a[i] = a.sin();
        }

        let loss_fn = |cx: f32, cy: f32| -> f32 {
            let dist_dev = (cx - init_cx).hypot(cy - init_cy);
            if dist_dev > max_dev || cx < 0.0 || cx >= (w as f32) || cy < 0.0 || cy >= (h as f32) {
                return 1e6 + dist_dev * 1000.0;
            }

            let mut diff = [0.0f32; 180];
            let mut sum_v_int = 0.0f32;

            if mode_solar {
                for i in 0..180 {
                    let ca = cos_a[i];
                    let sa = sin_a[i];
                    let v_in = interp_bilinear_reflect(gray_slice, w, h, cx + r_in * ca, cy + r_in * sa);
                    let v_out = interp_bilinear_reflect(gray_slice, w, h, cx + r_out * ca, cy + r_out * sa);
                    diff[i] = v_out - v_in;
                    let v_int = interp_bilinear_reflect(gray_slice, w, h, cx + r_int * ca, cy + r_int * sa);
                    sum_v_int += v_int;
                }
                diff.sort_unstable_by(|a, b| a.total_cmp(b));
                let grad = percentile_linear_180(&diff, 75.0);
                let pen_in = sum_v_int / 180.0;
                let eff_pen = (pen_in * 0.5).min((0.0f32).max(grad * 0.4));
                -(grad - eff_pen)
            } else {
                for i in 0..180 {
                    let ca = cos_a[i];
                    let sa = sin_a[i];
                    let v_in = interp_bilinear_reflect(gray_slice, w, h, cx + r_in * ca, cy + r_in * sa);
                    let v_out = interp_bilinear_reflect(gray_slice, w, h, cx + r_out * ca, cy + r_out * sa);
                    diff[i] = v_in - v_out;
                }
                diff.sort_unstable_by(|a, b| a.total_cmp(b));
                -percentile_linear_180(&diff, 60.0)
            }
        };

        let mut best_cx = init_cx;
        let mut best_cy = init_cy;
        let mut best_loss = loss_fn(best_cx, best_cy);

        let search_range = max_dev.min(24.0) as i32;
        let step = if search_range <= 16 { 2 } else { 3 };

        let mut dx = -search_range;
        while dx <= search_range {
            let mut dy = -search_range;
            while dy <= search_range {
                let cx = init_cx + (dx as f32);
                let cy = init_cy + (dy as f32);
                let l = loss_fn(cx, cy);
                if l < best_loss {
                    best_loss = l;
                    best_cx = cx;
                    best_cy = cy;
                }
                dy += step;
            }
            dx += step;
        }

        let nonzdelt = 0.05f32;
        let zdelt = 0.00025f32;
        let x0 = [best_cx, best_cy];
        let mut sim = [
            [x0[0], x0[1]],
            [if x0[0] != 0.0 { x0[0] * (1.0 + nonzdelt) } else { zdelt }, x0[1]],
            [x0[0], if x0[1] != 0.0 { x0[1] * (1.0 + nonzdelt) } else { zdelt }],
        ];
        let mut fsim = [
            loss_fn(sim[0][0], sim[0][1]),
            loss_fn(sim[1][0], sim[1][1]),
            loss_fn(sim[2][0], sim[2][1]),
        ];

        let sort_simplex = |s: &mut [[f32; 2]; 3], f: &mut [f32; 3]| {
            if f[0] > f[1] {
                f.swap(0, 1);
                s.swap(0, 1);
            }
            if f[1] > f[2] {
                f.swap(1, 2);
                s.swap(1, 2);
            }
            if f[0] > f[1] {
                f.swap(0, 1);
                s.swap(0, 1);
            }
        };

        sort_simplex(&mut sim, &mut fsim);

        let xatol = 0.05f32;
        let fatol = 1e-4f32;
        let maxiter = 50usize;

        for _ in 1..maxiter {
            let max_x_diff = (sim[1][0] - sim[0][0]).abs()
                .max((sim[2][0] - sim[0][0]).abs())
                .max((sim[1][1] - sim[0][1]).abs())
                .max((sim[2][1] - sim[0][1]).abs());
            let max_f_diff = (fsim[0] - fsim[1]).abs().max((fsim[0] - fsim[2]).abs());
            if max_x_diff <= xatol && max_f_diff <= fatol {
                break;
            }

            let xbar = [0.5 * (sim[0][0] + sim[1][0]), 0.5 * (sim[0][1] + sim[1][1])];
            let xr = [2.0 * xbar[0] - sim[2][0], 2.0 * xbar[1] - sim[2][1]];
            let fxr = loss_fn(xr[0], xr[1]);

            let mut doshrink = false;
            if fxr < fsim[0] {
                let xe = [3.0 * xbar[0] - 2.0 * sim[2][0], 3.0 * xbar[1] - 2.0 * sim[2][1]];
                let fxe = loss_fn(xe[0], xe[1]);
                if fxe < fxr {
                    sim[2] = xe;
                    fsim[2] = fxe;
                } else {
                    sim[2] = xr;
                    fsim[2] = fxr;
                }
            } else if fxr < fsim[1] {
                sim[2] = xr;
                fsim[2] = fxr;
            } else {
                if fxr < fsim[2] {
                    let xc = [1.5 * xbar[0] - 0.5 * sim[2][0], 1.5 * xbar[1] - 0.5 * sim[2][1]];
                    let fxc = loss_fn(xc[0], xc[1]);
                    if fxc <= fxr {
                        sim[2] = xc;
                        fsim[2] = fxc;
                    } else {
                        doshrink = true;
                    }
                } else {
                    let xcc = [0.5 * xbar[0] + 0.5 * sim[2][0], 0.5 * xbar[1] + 0.5 * sim[2][1]];
                    let fxcc = loss_fn(xcc[0], xcc[1]);
                    if fxcc < fsim[2] {
                        sim[2] = xcc;
                        fsim[2] = fxcc;
                    } else {
                        doshrink = true;
                    }
                }
                if doshrink {
                    sim[1] = [sim[0][0] + 0.5 * (sim[1][0] - sim[0][0]), sim[0][1] + 0.5 * (sim[1][1] - sim[0][1])];
                    sim[2] = [sim[0][0] + 0.5 * (sim[2][0] - sim[0][0]), sim[0][1] + 0.5 * (sim[2][1] - sim[0][1])];
                    fsim[1] = loss_fn(sim[1][0], sim[1][1]);
                    fsim[2] = loss_fn(sim[2][0], sim[2][1]);
                }
            }
            sort_simplex(&mut sim, &mut fsim);
        }

        let final_cx = sim[0][0];
        let final_cy = sim[0][1];
        let final_loss = fsim[0];
        let final_dist = (final_cx - init_cx).hypot(final_cy - init_cy);

        if final_dist > max_dev
            || final_cx < 0.0
            || final_cx >= (w as f32)
            || final_cy < 0.0
            || final_cy >= (h as f32)
            || final_loss > best_loss
        {
            (best_cx, best_cy)
        } else {
            (final_cx, final_cy)
        }
    });

    Ok(result)
}

// ----------------------------------------------------------------------------
// Tarea 2: Fusión HDR Solar en una sola pasada paralela (core/eclipse.py)
// ----------------------------------------------------------------------------

#[inline(always)]
fn smoothstep_f32(x: f32, edge0: f32, edge1: f32) -> f32 {
    let denom = (edge1 - edge0).max(1e-6);
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
fn percentile_exact_f32(data: &mut [f32], q: f32) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let idx_f = (q / 100.0) * ((n - 1) as f32);
    let i0 = (idx_f.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    let frac = idx_f - (i0 as f32);
    data.select_nth_unstable_by(i0, |a, b| a.total_cmp(b));
    let v0 = data[i0];
    let v1 = if i1 > i0 {
        *data[i0 + 1..].iter().min_by(|a, b| a.total_cmp(b)).unwrap_or(&v0)
    } else {
        v0
    };
    (1.0 - frac) * v0 + frac * v1
}

#[pyfunction]
#[pyo3(signature = (
    images,
    exposures_s,
    saturation_threshold=0.90,
    noise_floor=0.015,
    lunar_mask_params=None,
    gradual_handover=true,
    bg_pedestals=None
))]
pub fn fuse_hdr_bracketing_solar_rs<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyAny>,
    exposures_s: Vec<f32>,
    saturation_threshold: f32,
    noise_floor: f32,
    lunar_mask_params: Option<(f32, f32, f32)>,
    gradual_handover: bool,
    bg_pedestals: Option<Vec<Vec<f32>>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py_list = images.downcast::<pyo3::types::PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("images debe ser una lista de ndarrays"))?;
    let n = py_list.len();
    if n == 0 || n != exposures_s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "images y exposures_s deben tener la misma longitud no vacía",
        ));
    }

    let is_3d = py_list.get_item(0)?.extract::<PyReadonlyArray3<f32>>().is_ok();
    let sat_lo = (0.40f32).max(saturation_threshold - 0.20);
    let sat_hi = saturation_threshold;

    if is_3d {
        let mut arrs = Vec::with_capacity(n);
        let mut h = 0;
        let mut w = 0;
        for i in 0..n {
            let item = py_list.get_item(i)?;
            let arr = item.extract::<PyReadonlyArray3<f32>>()?;
            let (img_h, img_w, c) = arr.as_array().dim();
            if c != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Imágenes 3D deben tener 3 canales",
                ));
            }
            if i == 0 {
                h = img_h;
                w = img_w;
            } else if img_h != h || img_w != w {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Todas las imágenes deben tener las mismas dimensiones",
                ));
            }
            arrs.push(arr);
        }
        let img_slices: Vec<&[f32]> = arrs
            .iter()
            .map(|a| a.as_slice().unwrap())
            .collect();

        let bg: Vec<[f32; 3]> = match bg_pedestals {
            Some(ref peds) if peds.len() == n => peds
                .iter()
                .map(|p| {
                    if p.len() >= 3 {
                        [p[0], p[1], p[2]]
                    } else if !p.is_empty() {
                        [p[0], p[0], p[0]]
                    } else {
                        [0.0, 0.0, 0.0]
                    }
                })
                .collect(),
            _ => vec![[0.0, 0.0, 0.0]; n],
        };

        let mut out_vec = vec![0.0f32; h * w * 3];

        py.allow_threads(|| {
            out_vec
                .par_chunks_exact_mut(w * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let moon_w = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                            let dist = (x_f - mcx).hypot(y_f - mcy);
                            smoothstep_f32(dist, mr - 1.5, mr + 1.0) as f64
                        } else {
                            1.0f64
                        };

                        let mut wf_r = 0.0f64;
                        let mut wf_g = 0.0f64;
                        let mut wf_b = 0.0f64;
                        let mut sw = 0.0f64;

                        let pix_idx = (y * w + x) * 3;

                        for i in 0..n {
                            let t = (exposures_s[i] as f64).max(1e-6);
                            let shorter_t = if i > 0 {
                                Some(exposures_s[i - 1] as f64)
                            } else {
                                None
                            };

                            let in_r = img_slices[i][pix_idx];
                            let in_g = img_slices[i][pix_idx + 1];
                            let in_b = img_slices[i][pix_idx + 2];

                            let clean_r = ((in_r - bg[i][0]).max(0.0)) as f64;
                            let clean_g = ((in_g - bg[i][1]).max(0.0)) as f64;
                            let clean_b = ((in_b - bg[i][2]).max(0.0)) as f64;

                            let flux_r = clean_r / t;
                            let flux_g = clean_g / t;
                            let flux_b = clean_b / t;

                            let lum = 0.2126 * (in_r as f64) + 0.7152 * (in_g as f64) + 0.0722 * (in_b as f64);
                            let max_c = (in_r.max(in_g).max(in_b)) as f64;

                            let w_low = smoothstep_f32(lum as f32, 0.0, noise_floor * 2.0) as f64;
                            let w_high = 1.0 - (smoothstep_f32(max_c as f32, sat_lo, sat_hi) as f64);
                            let mut w = w_low * w_high;

                            if gradual_handover {
                                if let Some(st) = shorter_t {
                                    if st > 0.0 {
                                        let taper = w_high;
                                        let weight_factor = taper + (1.0 - taper) * (t / st);
                                        w /= weight_factor.max(1e-4);
                                    }
                                }
                            }

                            if lunar_mask_params.is_some() {
                                w *= moon_w;
                            }

                            wf_r += flux_r * w;
                            wf_g += flux_g * w;
                            wf_b += flux_b * w;
                            sw += w;
                        }

                        let (mut merged_r, mut merged_g, mut merged_b) = if sw < 1e-6 {
                            let in_r0 = img_slices[0][pix_idx];
                            let in_g0 = img_slices[0][pix_idx + 1];
                            let in_b0 = img_slices[0][pix_idx + 2];
                            let shortest_max = in_r0.max(in_g0).max(in_b0);

                            let shortest_t = (exposures_s[0] as f64).max(1e-6);
                            let longest_t = (exposures_s[n - 1] as f64).max(1e-6);

                            if shortest_max >= sat_lo {
                                (
                                    (in_r0 as f64) / shortest_t,
                                    (in_g0 as f64) / shortest_t,
                                    (in_b0 as f64) / shortest_t,
                                )
                            } else {
                                let in_rn = img_slices[n - 1][pix_idx];
                                let in_gn = img_slices[n - 1][pix_idx + 1];
                                let in_bn = img_slices[n - 1][pix_idx + 2];
                                (
                                    (in_rn as f64) / longest_t,
                                    (in_gn as f64) / longest_t,
                                    (in_bn as f64) / longest_t,
                                )
                            }
                        } else {
                            (wf_r / sw, wf_g / sw, wf_b / sw)
                        };

                        if lunar_mask_params.is_some() {
                            merged_r *= moon_w;
                            merged_g *= moon_w;
                            merged_b *= moon_w;
                        }

                        row[x * 3] = merged_r as f32;
                        row[x * 3 + 1] = merged_g as f32;
                        row[x * 3 + 2] = merged_b as f32;
                    }
                });

            let mut valid_vals = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                let mut v = Vec::with_capacity(h * w * 3);
                for y in 0..h {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let dist = (x_f - mcx).hypot(y_f - mcy);
                        let mw = smoothstep_f32(dist, mr - 1.5, mr + 1.0);
                        if mw > 0.5 {
                            let idx = (y * w + x) * 3;
                            v.push(out_vec[idx]);
                            v.push(out_vec[idx + 1]);
                            v.push(out_vec[idx + 2]);
                        }
                    }
                }
                v
            } else {
                out_vec.clone()
            };

            let p_high = percentile_exact_f32(&mut valid_vals, 99.999);
            let p_low = percentile_exact_f32(&mut valid_vals, 0.05);
            let denom = (p_high - p_low).max(1e-8);

            out_vec.par_iter_mut().for_each(|v| {
                *v = ((*v - p_low) / denom).clamp(0.0, 1.0);
            });
        });

        let out_arr = Array3::from_shape_vec((h, w, 3), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(out_arr.into_pyarray(py).into_any())
    } else {
        // 2D Monochrome branch
        let mut arrs = Vec::with_capacity(n);
        let mut h = 0;
        let mut w = 0;
        for i in 0..n {
            let item = py_list.get_item(i)?;
            let arr = item.extract::<PyReadonlyArray2<f32>>()?;
            let (img_h, img_w) = arr.as_array().dim();
            if i == 0 {
                h = img_h;
                w = img_w;
            } else if img_h != h || img_w != w {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Todas las imágenes deben tener las mismas dimensiones",
                ));
            }
            arrs.push(arr);
        }
        let img_slices: Vec<&[f32]> = arrs
            .iter()
            .map(|a| a.as_slice().unwrap())
            .collect();

        let bg: Vec<f32> = match bg_pedestals {
            Some(ref peds) if peds.len() == n => peds
                .iter()
                .map(|p| if !p.is_empty() { p[0] } else { 0.0 })
                .collect(),
            _ => vec![0.0; n],
        };

        let mut out_vec = vec![0.0f32; h * w];

        py.allow_threads(|| {
            out_vec
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(y, row)| {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let moon_w = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                            let dist = (x_f - mcx).hypot(y_f - mcy);
                            smoothstep_f32(dist, mr - 1.5, mr + 1.0) as f64
                        } else {
                            1.0f64
                        };

                        let mut wf = 0.0f64;
                        let mut sw = 0.0f64;
                        let pix_idx = y * w + x;

                        for i in 0..n {
                            let t = (exposures_s[i] as f64).max(1e-6);
                            let shorter_t = if i > 0 {
                                Some(exposures_s[i - 1] as f64)
                            } else {
                                None
                            };

                            let in_v = img_slices[i][pix_idx];
                            let clean_v = ((in_v - bg[i]).max(0.0)) as f64;
                            let flux = clean_v / t;

                            let w_low = smoothstep_f32(in_v, 0.0, noise_floor * 2.0) as f64;
                            let w_high = 1.0 - (smoothstep_f32(in_v, sat_lo, sat_hi) as f64);
                            let mut w = w_low * w_high;

                            if gradual_handover {
                                if let Some(st) = shorter_t {
                                    if st > 0.0 {
                                        let taper = w_high;
                                        let weight_factor = taper + (1.0 - taper) * (t / st);
                                        w /= weight_factor.max(1e-4);
                                    }
                                }
                            }

                            if lunar_mask_params.is_some() {
                                w *= moon_w;
                            }

                            wf += flux * w;
                            sw += w;
                        }

                        let mut merged = if sw < 1e-6 {
                            let in_v0 = img_slices[0][pix_idx];
                            let shortest_t = (exposures_s[0] as f64).max(1e-6);
                            let longest_t = (exposures_s[n - 1] as f64).max(1e-6);

                            if in_v0 >= sat_lo {
                                (in_v0 as f64) / shortest_t
                            } else {
                                (img_slices[n - 1][pix_idx] as f64) / longest_t
                            }
                        } else {
                            wf / sw
                        };

                        if lunar_mask_params.is_some() {
                            merged *= moon_w;
                        }

                        row[x] = merged as f32;
                    }
                });

            let mut valid_vals = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                let mut v = Vec::with_capacity(h * w);
                for y in 0..h {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let dist = (x_f - mcx).hypot(y_f - mcy);
                        let mw = smoothstep_f32(dist, mr - 1.5, mr + 1.0);
                        if mw > 0.5 {
                            v.push(out_vec[y * w + x]);
                        }
                    }
                }
                v
            } else {
                out_vec.clone()
            };

            let p_high = percentile_exact_f32(&mut valid_vals, 99.999);
            let p_low = percentile_exact_f32(&mut valid_vals, 0.05);
            let denom = (p_high - p_low).max(1e-8);

            out_vec.par_iter_mut().for_each(|v| {
                *v = ((*v - p_low) / denom).clamp(0.0, 1.0);
            });
        });

        let out_arr = Array2::from_shape_vec((h, w), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(out_arr.into_pyarray(py).into_any())
    }
}
