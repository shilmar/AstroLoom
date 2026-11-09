use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;

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
/// calcula varianza local, normaliza el contraste por sigma local con piso de ruido,
/// aplica la transformación arctan(k * c) y acumula en acc sin crear matrices temporales.
#[pyfunction]
#[pyo3(signature = (acc, lum, num, den, num_v, m_bool, k=0.70, noise_floor=0.003))]
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
) -> PyResult<()> {
    let (_h, w) = lum.as_array().dim();
    let lum_slice = lum.as_slice().unwrap();
    let num_slice = num.as_slice().unwrap();
    let den_slice = den.as_slice().unwrap();
    let num_v_slice = num_v.as_slice().unwrap();
    let m_slice = m_bool.as_slice().unwrap();

    let nf_sq = noise_floor * noise_floor;

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
                        if sd > 1e-5 {
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
