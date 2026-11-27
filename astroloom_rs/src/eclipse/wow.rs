use numpy::ndarray::{Array2, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::detail::display_detail_rings_raw;
use super::median_f32;

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
    let a_slice = a.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let m_slice = m.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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

    let img_slice = image.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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
