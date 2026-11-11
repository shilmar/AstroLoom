use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
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

    let out_vec = py.allow_threads(|| {
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

                    let mut vv = 1e-20f64;
                    if bilateral {
                        let mut mo = 0.0f64;
                        let mut mo2 = 0.0f64;
                        let mut ma = 0.0f64;

                        for dy in -2..=2 {
                            let ky = B3K[(dy + 2) as usize] as f64;
                            let y1 = y_i + dy * step_i;
                            let y2 = y_i - dy * step_i;
                            for dx in -2..=2 {
                                let kx = B3K[(dx + 2) as usize] as f64;
                                let k = ky * kx;
                                let x1 = x_i + dx * step_i;
                                let x2 = x_i - dx * step_i;

                                let g = if dy == 0 && dx == 0 {
                                    true
                                } else {
                                    let ok1 = y1 >= 0
                                        && y1 < h_i
                                        && x1 >= 0
                                        && x1 < w_i
                                        && m_slice[(y1 as usize) * w + (x1 as usize)];
                                    let ok2 = y2 >= 0
                                        && y2 < h_i
                                        && x2 >= 0
                                        && x2 < w_i
                                        && m_slice[(y2 as usize) * w + (x2 as usize)];
                                    ok1 && ok2
                                };

                                if g {
                                    let vg = match guide_slice {
                                        Some(g_s) => g_s[(y1 as usize) * w + (x1 as usize)],
                                        None => c_slice[(y1 as usize) * w + (x1 as usize)],
                                    } as f64;
                                    let dl = vg - (ctg as f64);
                                    mo += k * dl;
                                    mo2 += k * dl * dl;
                                    ma += k;
                                }
                            }
                        }
                        let ma_safe = ma.max(1e-20);
                        let mean = mo / ma_safe;
                        vv = (mo2 / ma_safe - mean * mean).max(1e-20);
                    }

                    let mut num = 0.0f64;
                    let mut den = 0.0f64;
                    let mut ks = 0.0f32;
                    let mut kn = 0.0f32;

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
                                let ok1 = y1 >= 0
                                    && y1 < h_i
                                    && x1 >= 0
                                    && x1 < w_i
                                    && m_slice[(y1 as usize) * w + (x1 as usize)];
                                let ok2 = y2 >= 0
                                    && y2 < h_i
                                    && x2 >= 0
                                    && x2 < w_i
                                    && m_slice[(y2 as usize) * w + (x2 as usize)];
                                ok1 && ok2
                            };

                            if g {
                                let tap_idx = (y1 as usize) * w + (x1 as usize);
                                let v = c_slice[tap_idx];
                                if dy != 0 || dx != 0 {
                                    kn += k;
                                }
                                let wt = if bilateral {
                                    let vg = match guide_slice {
                                        Some(g_s) => g_s[tap_idx],
                                        None => v,
                                    } as f64;
                                    let diff = (ctg as f64) - vg;
                                    (k as f64) * (-0.5 * diff * diff / vv).exp()
                                } else {
                                    k as f64
                                };
                                num += wt * ((v - ctr) as f64);
                                den += wt;
                                ks += k;
                            }
                        }
                    }

                    out_row[x] = (ctr as f64 + num / den.max(1e-20)) as f32;
                    compl_row[x] = ks;
                    pair_row[x] = kn > 0.0;
                }
            });

        (out, compl, pair)
    });

    Ok((
        Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py),
        Array2::from_shape_vec((h, w), compl_vec).unwrap().into_pyarray(py),
        Array2::from_shape_vec((h, w), pair_vec).unwrap().into_pyarray(py),
    ))
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
        let step_i = step as isize;
        let h_i = h as isize;
        let w_i = w as isize;

        // Convolución separable B3 para una imagen f32
        let conv_b3 = |src: &[f32]| -> Vec<f32> {
            let mut tmp = vec![0.0f32; h * w];
            // Paso vertical
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

            let mut out = vec![0.0f32; h * w];
            // Paso horizontal
            out.par_chunks_exact_mut(w)
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

            out
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

        let mut res = vec![0.0f32; h * w];
        res.par_chunks_exact_mut(w)
            .zip(conv_a.par_chunks_exact(w))
            .zip(conv_m.par_chunks_exact(w))
            .for_each(|((res_row, ca_row), cm_row)| {
                for x in 0..w {
                    res_row[x] = ca_row[x] / cm_row[x].max(1e-20);
                }
            });

        res
    });

    Ok(Array2::from_shape_vec((h, w), out_vec).unwrap().into_pyarray(py))
}
