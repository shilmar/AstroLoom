use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{median_f32, reflect101};

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
    let detail_slice = detail.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let mask_slice = valid_mask.as_ref().map(|m| m.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))).transpose()?;

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
    let hxx_slice = hxx.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let hyy_slice = hyy.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let hxy_slice = hxy.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let j11_slice = j11.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let j22_slice = j22.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let j12_slice = j12.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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
    let lum_slice = lum.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
    let mask_slice = exclusion_mask.as_ref().map(|m| m.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))).transpose()?;

    let rad = delta_theta_deg.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();
    let eps = epsilon.max(1e-8);
    let r_min_sq = radius_min.max(0.0).powi(2);

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

                    if let Some(mask) = mask_slice {
                        let ix_p = xp.round() as isize;
                        let iy_p = yp.round() as isize;
                        if ix_p >= 0 && ix_p < w as isize && iy_p >= 0 && iy_p < h as isize {
                            if mask[iy_p as usize * w + ix_p as usize] {
                                row_out[x] = 0.0;
                                continue;
                            }
                        }
                        let ix_m = xm.round() as isize;
                        let iy_m = ym.round() as isize;
                        if ix_m >= 0 && ix_m < w as isize && iy_m >= 0 && iy_m < h as isize {
                            if mask[iy_m as usize * w + ix_m as usize] {
                                row_out[x] = 0.0;
                                continue;
                            }
                        }
                    }

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
    let polar_slice = polar.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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
    let rgb_slice = rgb.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

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
            let row_offset = y * w * c;
            for x in 0..w {
                let dx = x as f32 - center_x;
                let r = (dx * dx + dy2).sqrt();
                if r >= r_min && r <= r_max {
                    let b = (((r - r_min) * inv_w) as usize).min(n_bins - 1);
                    let p_idx = row_offset + x * c;
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
                let row_offset = y * w * c;
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    let r = (dx * dx + dy2).sqrt();
                    if r >= r_min && r <= r_max {
                        let b = (((r - r_min) * inv_w) as usize).min(n_bins - 1);
                        let p_idx = row_offset + x * c;
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
