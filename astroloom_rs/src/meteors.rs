use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// DETECCIÓN Y FUSIÓN FOTOMÉTRICA DE LLUVIAS DE METEOROS (core/meteors.py)
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
fn sample_bilinear_2d(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let v00 = img[y0 * w + x0];
    let v10 = img[y0 * w + x1];
    let v01 = img[y1 * w + x0];
    let v11 = img[y1 * w + x1];

    let top = v00 + fx * (v10 - v00);
    let bot = v01 + fx * (v11 - v01);
    top + fy * (bot - top)
}

#[inline(always)]
fn sample_bilinear_3d(img: &[f32], w: usize, h: usize, x: f32, y: f32, c: usize) -> f32 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let stride = w * 3;
    let v00 = img[y0 * stride + x0 * 3 + c];
    let v10 = img[y0 * stride + x1 * 3 + c];
    let v01 = img[y1 * stride + x0 * 3 + c];
    let v11 = img[y1 * stride + x1 * 3 + c];

    let top = v00 + fx * (v10 - v00);
    let bot = v01 + fx * (v11 - v01);
    top + fy * (bot - top)
}

/// 1. meteor_positive_diff_and_stats_rs:
/// Diferencia positiva max(0, curr - bg), media, desviación típica y
/// estimación robusta de ruido de fondo (mediana y MAD) en una sola pasada.
#[pyfunction]
pub fn meteor_positive_diff_and_stats_rs<'py>(
    py: Python<'py>,
    arr_curr: PyReadonlyArray2<'py, f32>,
    arr_bg: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32, f32, f32, f32)> {
    let curr_view = arr_curr.as_array();
    let bg_view = arr_bg.as_array();
    let (h, w) = curr_view.dim();

    if bg_view.dim() != (h, w) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: curr ({}, {}) vs bg ({}, {})",
            h, w, bg_view.dim().0, bg_view.dim().1
        )));
    }

    let mut out_diff = Array2::<f32>::zeros((h, w));

    let (mean_val, std_val, med_val, mad_val) = py.allow_threads(|| {
        let curr_slice = if let Some(s) = curr_view.as_slice() {
            s
        } else {
            return (0.0, 0.0, 0.0, 0.0);
        };
        let bg_slice = if let Some(s) = bg_view.as_slice() {
            s
        } else {
            return (0.0, 0.0, 0.0, 0.0);
        };
        let out_slice = out_diff.as_slice_mut().unwrap();

        let total_pixels = (h * w) as f64;
        let (sum_diff, sum_sq): (f64, f64) = out_slice
            .par_chunks_mut(w)
            .zip(curr_slice.par_chunks(w))
            .zip(bg_slice.par_chunks(w))
            .map(|((out_row, c_row), b_row)| {
                let mut s = 0.0f64;
                let mut sq = 0.0f64;
                for i in 0..w {
                    let d = c_row[i] - b_row[i];
                    if d > 0.0 {
                        out_row[i] = d;
                        let df = d as f64;
                        s += df;
                        sq += df * df;
                    }
                }
                (s, sq)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

        let mean = if total_pixels > 0.0 {
            (sum_diff / total_pixels) as f32
        } else {
            0.0
        };
        let var = if total_pixels > 0.0 {
            ((sum_sq / total_pixels) - (mean as f64 * mean as f64)).max(0.0)
        } else {
            0.0
        };
        let std = var.sqrt() as f32;

        // Muestreo paso 4 para estimación robusta del ruido (mediana y MAD)
        let sample_cap = ((h + 3) / 4) * ((w + 3) / 4);
        let mut sample = Vec::with_capacity(sample_cap);
        for r in (0..h).step_by(4) {
            let row_offset = r * w;
            for c in (0..w).step_by(4) {
                sample.push(curr_slice[row_offset + c] - bg_slice[row_offset + c]);
            }
        }

        let med = median_f32(&mut sample);
        for x in sample.iter_mut() {
            *x = (*x - med).abs();
        }
        let mad = median_f32(&mut sample);

        (mean, std, med, mad)
    });

    Ok((out_diff.into_pyarray(py), mean_val, std_val, med_val, mad_val))
}

/// 2. compute_meteor_capsule_mask_rs:
/// Generación continua subpíxel de máscara de cápsula con desvanecimiento suave Hermite (smoothstep).
#[pyfunction]
#[pyo3(signature = (height, width, p_start, p_end, thick, feather_px=10.0))]
pub fn compute_meteor_capsule_mask_rs<'py>(
    py: Python<'py>,
    height: usize,
    width: usize,
    p_start: (f32, f32),
    p_end: (f32, f32),
    thick: f32,
    feather_px: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut mask = Array2::<f32>::zeros((height, width));

    py.allow_threads(|| {
        let mask_slice = mask.as_slice_mut().unwrap();
        let (x1, y1) = p_start;
        let (x2, y2) = p_end;
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;

        let r0 = (thick * 0.5).max(1.0);
        let r1 = r0 + feather_px.max(1.0);
        let r_diff = r1 - r0;

        mask_slice
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y_idx, row)| {
                let y = y_idx as f32;
                for x_idx in 0..width {
                    let x = x_idx as f32;
                    let dist = if len_sq < 1e-6 {
                        let px = x - x1;
                        let py = y - y1;
                        (px * px + py * py).sqrt()
                    } else {
                        let t = (((x - x1) * dx + (y - y1) * dy) / len_sq).clamp(0.0, 1.0);
                        let px = x - (x1 + t * dx);
                        let py = y - (y1 + t * dy);
                        (px * px + py * py).sqrt()
                    };

                    if dist <= r0 {
                        row[x_idx] = 1.0;
                    } else if dist >= r1 {
                        row[x_idx] = 0.0;
                    } else {
                        let u = (dist - r0) / r_diff;
                        row[x_idx] = 1.0 - u * u * (3.0 - 2.0 * u);
                    }
                }
            });
    });

    Ok(mask.into_pyarray(py))
}

/// 3. estimate_channel_percentile_rs:
/// Cálculo del percentil especificado (ej. 30%) por canal RGB sobre píxeles no enmascarados.
#[pyfunction]
#[pyo3(signature = (patch, mask=None, percentile=30.0))]
pub fn estimate_channel_percentile_rs<'py>(
    py: Python<'py>,
    patch: PyReadonlyArray3<'py, f32>,
    mask: Option<PyReadonlyArray2<'py, u8>>,
    percentile: f32,
) -> PyResult<Vec<f32>> {
    let patch_view = patch.as_array();
    let (h, w, c) = patch_view.dim();

    let mask_opt = mask.as_ref().map(|m| m.as_array());

    let res = py.allow_threads(|| {
        let mut out = vec![0.0f32; c];
        for ch in 0..c {
            let mut vals = Vec::with_capacity(h * w);
            for y in 0..h {
                for x in 0..w {
                    let is_valid = if let Some(ref m) = mask_opt {
                        m[(y, x)] == 0
                    } else {
                        true
                    };
                    if is_valid {
                        vals.push(patch_view[(y, x, ch)]);
                    }
                }
            }
            if vals.is_empty() {
                // Fallback a todos los píxeles
                for y in 0..h {
                    for x in 0..w {
                        vals.push(patch_view[(y, x, ch)]);
                    }
                }
            }
            if !vals.is_empty() {
                let n = vals.len();
                let rank = (percentile / 100.0) * ((n - 1) as f32);
                let i0 = rank as usize;
                let i1 = (i0 + 1).min(n - 1);
                let frac = rank - (i0 as f32);
                if i0 == i1 {
                    vals.select_nth_unstable_by(i0, |a, b| a.total_cmp(b));
                    out[ch] = vals[i0];
                } else {
                    vals.select_nth_unstable_by(i1, |a, b| a.total_cmp(b));
                    let v1 = vals[i1];
                    let v0 = *vals[..i1].iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&v1);
                    out[ch] = v0 + frac * (v1 - v0);
                }
            }
        }
        out
    });

    Ok(res)
}

/// 4. blend_meteor_capsule_lighten_rs:
/// Fusión en modo Aclarar (Lighten) ponderada por máscara adaptativa.
/// Soporta:
/// - Mapeo homográfico inverso (Zero-Allocation de matrices full-frame).
/// - Fusión directa con desplazamiento local (x_offset, y_offset).
/// - Fusión en paralelo sobre imágenes pre-deformadas.
#[pyfunction]
#[pyo3(signature = (out_img, patch, mask, homography=None, x_offset=0, y_offset=0))]
pub fn blend_meteor_capsule_lighten_rs<'py>(
    py: Python<'py>,
    out_img: Bound<'py, PyArray3<f32>>,
    patch: PyReadonlyArray3<'py, f32>,
    mask: PyReadonlyArray2<'py, f32>,
    homography: Option<PyReadonlyArray2<'py, f32>>,
    x_offset: usize,
    y_offset: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let patch_view = patch.as_array();
    let mask_view = mask.as_array();
    let (h_p, w_p, c_p) = patch_view.dim();
    let (h_m, w_m) = mask_view.dim();

    if (h_p, w_p) != (h_m, w_m) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch between patch ({}, {}) and mask ({}, {})",
            h_p, w_p, h_m, w_m
        )));
    }

    let out_shape = unsafe { out_img.as_array().dim() };
    let (h_out, w_out, c_out) = out_shape;

    if c_p != c_out {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Channel mismatch: patch {} vs out_img {}",
            c_p, c_out
        )));
    }

    let patch_slice = patch_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("patch must be C-contiguous")
    })?;
    let mask_slice = mask_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("mask must be C-contiguous")
    })?;

    // Inversión y preparación de matriz de homografía si está presente
    let h_inv_opt = if let Some(ref h_arr) = homography {
        let h_view = h_arr.as_array();
        if h_view.dim() == (3, 3) {
            let h_mat = nalgebra::Matrix3::new(
                h_view[(0, 0)] as f64, h_view[(0, 1)] as f64, h_view[(0, 2)] as f64,
                h_view[(1, 0)] as f64, h_view[(1, 1)] as f64, h_view[(1, 2)] as f64,
                h_view[(2, 0)] as f64, h_view[(2, 1)] as f64, h_view[(2, 2)] as f64,
            );
            let det = h_mat.determinant();
            // Graceful Fallback: si det es nulo, NaN o degenerado, cae en fallback directo
            if det.is_finite() && det.abs() > 1e-5 {
                h_mat.try_inverse().map(|inv| (h_mat, inv))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let out_slice = unsafe {
        out_img.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("out_img must be C-contiguous: {:?}", e))
        })?
    };

    py.allow_threads(move || {
        if let Some((h_mat, h_inv)) = h_inv_opt {
            // Caso 1: Proyección Homográfica Inversa sobre Bounding Box Acotada
            let corners = [
                nalgebra::Vector3::new(0.0f64, 0.0f64, 1.0f64),
                nalgebra::Vector3::new((w_p - 1) as f64, 0.0f64, 1.0f64),
                nalgebra::Vector3::new((w_p - 1) as f64, (h_p - 1) as f64, 1.0f64),
                nalgebra::Vector3::new(0.0f64, (h_p - 1) as f64, 1.0f64),
            ];

            let mut min_x = f64::MAX;
            let mut max_x = f64::MIN;
            let mut min_y = f64::MAX;
            let mut max_y = f64::MIN;

            for c in &corners {
                let p = h_mat * c;
                if p.z.abs() > 1e-9 {
                    let px = p.x / p.z;
                    let py = p.y / p.z;
                    if px < min_x { min_x = px; }
                    if px > max_x { max_x = px; }
                    if py < min_y { min_y = py; }
                    if py > max_y { max_y = py; }
                }
            }

            let bx1 = (min_x.floor() as isize - 2).max(0) as usize;
            let bx2 = ((max_x.ceil() as isize + 3).max(0) as usize).min(w_out);
            let by1 = (min_y.floor() as isize - 2).max(0) as usize;
            let by2 = ((max_y.ceil() as isize + 3).max(0) as usize).min(h_out);

            if bx1 < bx2 && by1 < by2 {
                out_slice
                    .par_chunks_mut(w_out * 3)
                    .enumerate()
                    .for_each(|(y_dst, row_dst)| {
                        if y_dst >= by1 && y_dst < by2 {
                            for x_dst in bx1..bx2 {
                                let dst_pt = nalgebra::Vector3::new(x_dst as f64, y_dst as f64, 1.0f64);
                                let src_pt = h_inv * dst_pt;
                                if src_pt.z.abs() > 1e-9 {
                                    let sx = (src_pt.x / src_pt.z) as f32;
                                    let sy = (src_pt.y / src_pt.z) as f32;

                                    if sx >= 0.0 && sx < (w_p - 1) as f32 && sy >= 0.0 && sy < (h_p - 1) as f32 {
                                        let m = sample_bilinear_2d(mask_slice, w_p, h_p, sx, sy);
                                        if m > 1e-5 {
                                            let px = x_dst * 3;
                                            for ch in 0..3 {
                                                let p_val = sample_bilinear_3d(patch_slice, w_p, h_p, sx, sy, ch);
                                                let diff = p_val - row_dst[px + ch];
                                                if diff > 0.0 {
                                                    row_dst[px + ch] += m * diff;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
            }
        } else if x_offset == 0 && y_offset == 0 && (h_p, w_p) == (h_out, w_out) {
            // Caso 2: Imágenes del mismo tamaño (ej. parche pre-proyectado)
            out_slice
                .par_chunks_mut(w_out * 3)
                .zip(patch_slice.par_chunks(w_out * 3))
                .zip(mask_slice.par_chunks(w_out))
                .for_each(|((out_row, patch_row), mask_row)| {
                    for x in 0..w_out {
                        let m = mask_row[x];
                        if m > 1e-5 {
                            let base_idx = x * 3;
                            for ch in 0..3 {
                                let diff = patch_row[base_idx + ch] - out_row[base_idx + ch];
                                if diff > 0.0 {
                                    out_row[base_idx + ch] += m * diff;
                                }
                            }
                        }
                    }
                });
        } else {
            // Caso 3: Inserción directa con offset (x_offset, y_offset)
            let max_y = (y_offset + h_p).min(h_out);
            let max_x = (x_offset + w_p).min(w_out);
            let valid_w = if max_x > x_offset { max_x - x_offset } else { 0 };

            out_slice
                .par_chunks_mut(w_out * 3)
                .enumerate()
                .for_each(|(y_dst, row_dst)| {
                    if y_dst >= y_offset && y_dst < max_y {
                        let y_local = y_dst - y_offset;
                        let row_p_offset = y_local * w_p * 3;
                        let row_m_offset = y_local * w_p;

                        for x_local in 0..valid_w {
                            let m = mask_slice[row_m_offset + x_local];
                            if m > 1e-5 {
                                let x_dst = x_offset + x_local;
                                let px_dst = x_dst * 3;
                                let px_p = row_p_offset + x_local * 3;

                                for ch in 0..3 {
                                    let p_val = patch_slice[px_p + ch];
                                    let diff = p_val - row_dst[px_dst + ch];
                                    if diff > 0.0 {
                                        row_dst[px_dst + ch] += m * diff;
                                    }
                                }
                            }
                        }
                    }
                });
        }
    });

    Ok(out_img)
}
