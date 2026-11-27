use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::{gaussian_blur_1d, median_f32};

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
    let lum_slice = lum.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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
    let clean_slice = clean_detail.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let (h, w) = clean_detail.as_array().dim();
    let sig_slice = sigma_prof.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
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
        let img_slice = img_arr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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
                        let final_mod = if streamer_mod > 0.0 {
                            let upper = (1.0 - lum).max(0.0);
                            streamer_mod * (0.4 + 0.6 * upper.sqrt())
                        } else {
                            let lower = lum.max(0.0);
                            streamer_mod * lower * (0.3 + 0.7 * lower.sqrt())
                        };
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
        let img_slice = img_arr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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
                        let final_mod = if streamer_mod > 0.0 {
                            let upper = (1.0 - lum).max(0.0);
                            streamer_mod * (0.4 + 0.6 * upper.sqrt())
                        } else {
                            let lower = lum.max(0.0);
                            streamer_mod * lower * (0.3 + 0.7 * lower.sqrt())
                        };
                        row_out[x] = (lum + final_mod).clamp(0.0, 1.0);
                    }
                });

            buf
        });

        let out_arr = Array2::from_shape_vec((h, w), out_vec).unwrap();
        Ok(out_arr.into_pyarray(py).as_any().clone())
    }
}
