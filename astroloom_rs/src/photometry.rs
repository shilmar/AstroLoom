use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// FASE 7: FOTOMETRÍA GAIA (PCC), PALETAS DUAL-BAND Y REPARACIÓN ESTELAR
// (core/dualband.py y core/pcc.py)
// ============================================================================

#[inline(always)]
fn median_f32_vec(mut data: Vec<f32>) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
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
fn percentile_f32_sampled(channel: &[f32], p: f32) -> f32 {
    let n = channel.len();
    if n == 0 {
        return 0.0;
    }
    // Subsample up to 50,000 points for ultra-fast percentile computation
    let step = (n / 50_000).max(1);
    let mut sample: Vec<f32> = channel.iter().step_by(step).copied().collect();
    let sample_n = sample.len();
    if sample_n == 0 {
        return 0.0;
    }
    let idx = ((sample_n - 1) as f32 * (p / 100.0)).round() as usize;
    let idx = idx.min(sample_n - 1);
    sample.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    sample[idx]
}

#[inline(always)]
fn median_f64(mut data: Vec<f64>) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
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

// ----------------------------------------------------------------------------
// Theil-Sen Robust Linear Regressor: slope = median(S_ij), intercept = median(y) - slope * median(x)
// ----------------------------------------------------------------------------
fn theil_sen_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let num_pairs = n * (n - 1) / 2;
    let mut slopes = Vec::with_capacity(num_pairs);

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx.abs() > 1e-12 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }

    if slopes.is_empty() {
        return (0.0, median_f64(y.to_vec()));
    }

    let slope = median_f64(slopes);
    let med_y = median_f64(y.to_vec());
    let med_x = median_f64(x.to_vec());
    let intercept = med_y - slope * med_x;

    (slope, intercept)
}

#[pyfunction]
#[pyo3(signature = (flux_r, flux_g, flux_b, bp_rp_catalog=None))]
pub fn fit_theil_sen_pcc_rs<'py>(
    py: Python<'py>,
    flux_r: PyReadonlyArray1<'py, f64>,
    flux_g: PyReadonlyArray1<'py, f64>,
    flux_b: PyReadonlyArray1<'py, f64>,
    bp_rp_catalog: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(f64, f64, f64)> {
    let fr = flux_r.as_slice()?;
    let fg = flux_g.as_slice()?;
    let fb = flux_b.as_slice()?;
    let cat_vec = bp_rp_catalog.as_ref().and_then(|c| c.as_slice().ok());

    let (kr, kg, kb) = py.allow_threads(|| {
        let n = fr.len().min(fg.len()).min(fb.len());
        if n < 6 {
            return (1.0, 1.0, 1.0);
        }

        let eps = 1e-7f64;
        let mut log_rg = Vec::with_capacity(n);
        let mut log_bg = Vec::with_capacity(n);
        let mut cat_valid = Vec::with_capacity(n);

        let has_cat = cat_vec.is_some() && cat_vec.unwrap().len() == n;

        for i in 0..n {
            let r = fr[i];
            let g = fg[i];
            let b = fb[i];
            let l_rg = ((r + eps) / (g + eps)).log10();
            let l_bg = ((b + eps) / (g + eps)).log10();

            if l_rg.is_finite() && l_bg.is_finite() {
                log_rg.push(l_rg);
                log_bg.push(l_bg);
                if has_cat {
                    cat_valid.push(cat_vec.unwrap()[i]);
                }
            }
        }

        if log_rg.len() < 6 {
            return (1.0, 1.0, 1.0);
        }

        let solar_bp_rp = 0.82f64;
        let (expected_log_rg, expected_log_bg) = if has_cat && cat_valid.len() == log_rg.len() {
            // Count unique rounded values
            let mut unique_vals: Vec<i64> = cat_valid.iter().map(|&v| (v * 100.0).round() as i64).collect();
            unique_vals.sort_unstable();
            unique_vals.dedup();

            if unique_vals.len() >= 2 {
                let (slope_r, icept_r) = theil_sen_fit(&cat_valid, &log_rg);
                let (slope_b, icept_b) = theil_sen_fit(&cat_valid, &log_bg);
                (
                    slope_r * solar_bp_rp + icept_r,
                    slope_b * solar_bp_rp + icept_b,
                )
            } else {
                (median_f64(log_rg), median_f64(log_bg))
            }
        } else {
            (median_f64(log_rg), median_f64(log_bg))
        };

        let kr = (10.0f64.powf(-expected_log_rg)).clamp(0.25, 4.0);
        let kb = (10.0f64.powf(-expected_log_bg)).clamp(0.25, 4.0);
        (kr, 1.0, kb)
    });

    Ok((kr, kg, kb))
}

#[pyfunction]
#[pyo3(signature = (img_rgb, k_r, k_b, preserve_neutral_bg=true))]
pub fn apply_pcc_gains_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    k_r: f32,
    k_b: f32,
    preserve_neutral_bg: bool,
) -> Bound<'py, PyArray3<f32>> {
    let img = img_rgb.as_array();
    let (h, w, c) = img.dim();

    let out_vec = py.allow_threads(|| {
        let total_pixels = h * w;
        let mut buf = vec![0.0f32; total_pixels * 3];

        if preserve_neutral_bg {
            let mut r_all = Vec::with_capacity(total_pixels);
            let mut g_all = Vec::with_capacity(total_pixels);
            let mut b_all = Vec::with_capacity(total_pixels);

            for y in 0..h {
                for x in 0..w {
                    r_all.push(unsafe { *img.uget((y, x, 0)) });
                    g_all.push(unsafe { *img.uget((y, x, 1)) });
                    b_all.push(unsafe { *img.uget((y, x, 2)) });
                }
            }

            let bg_r = percentile_f32_sampled(&r_all, 10.0);
            let bg_g = percentile_f32_sampled(&g_all, 10.0);
            let bg_b = percentile_f32_sampled(&b_all, 10.0);
            let bg_target = (bg_r + bg_g + bg_b) / 3.0;

            buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
                for x in 0..w {
                    let r = unsafe { *img.uget((y, x, 0)) };
                    let g = unsafe { *img.uget((y, x, 1)) };
                    let b = unsafe { *img.uget((y, x, 2)) };

                    let r_sig = r - bg_r;
                    let g_sig = g - bg_g;
                    let b_sig = b - bg_b;

                    let idx = x * 3;
                    row_slice[idx] = r_sig * k_r + bg_target;
                    row_slice[idx + 1] = g_sig + bg_target;
                    row_slice[idx + 2] = b_sig * k_b + bg_target;
                }
            });
        } else {
            buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
                for x in 0..w {
                    let r = unsafe { *img.uget((y, x, 0)) };
                    let g = unsafe { *img.uget((y, x, 1)) };
                    let b = unsafe { *img.uget((y, x, 2)) };

                    let idx = x * 3;
                    row_slice[idx] = r * k_r;
                    row_slice[idx + 1] = g;
                    row_slice[idx + 2] = b * k_b;
                }
            });
        }

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (img_rgb, amount=1.0, protect_ha=true))]
pub fn unpurple_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    amount: f32,
    protect_ha: bool,
) -> Bound<'py, PyArray3<f32>> {
    let img = img_rgb.as_array();
    let (h, w, c) = img.dim();

    if amount <= 1e-4 {
        return img.to_owned().into_pyarray(py);
    }

    let amt = amount.clamp(0.0, 1.0);

    let out_vec = py.allow_threads(|| {
        let mut buf = vec![0.0f32; h * w * 3];

        buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
            for x in 0..w {
                let r = unsafe { *img.uget((y, x, 0)) };
                let g = unsafe { *img.uget((y, x, 1)) };
                let b = unsafe { *img.uget((y, x, 2)) };

                let mut fringe = (r - g).min(b - g).max(0.0);
                if protect_ha && fringe > 0.0 {
                    let ha_dom = ((r - b) / (r + 1e-4)).clamp(0.0, 1.0);
                    fringe *= 1.0 - ha_dom;
                }

                let idx = x * 3;
                row_slice[idx] = r - amt * fringe * 0.75;
                row_slice[idx + 1] = g;
                row_slice[idx + 2] = b - amt * fringe;
            }
        });

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}

// ----------------------------------------------------------------------------
// Desaturate Star Cores
// ----------------------------------------------------------------------------
#[inline(always)]
fn desaturate_cores_in_place(buf: &mut [f32], _h: usize, w: usize, threshold: f32, feather: f32) {
    let inv_feather = 1.0 / feather.max(1e-4);
    buf.par_chunks_mut(w * 3).for_each(|row_slice| {
        for x in 0..w {
            let idx = x * 3;
            let r = row_slice[idx];
            let g = row_slice[idx + 1];
            let b = row_slice[idx + 2];

            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let mask = ((luma - threshold) * inv_feather).clamp(0.0, 1.0);
            if mask > 1e-4 {
                let inv_m = 1.0 - mask;
                row_slice[idx] = r * inv_m + luma * mask;
                row_slice[idx + 1] = g * inv_m + luma * mask;
                row_slice[idx + 2] = b * inv_m + luma * mask;
            }
        }
    });
}

#[pyfunction]
#[pyo3(signature = (img_rgb, palette="HOO", boost_oiii=1.0, desat_cores=true))]
pub fn apply_dualband_palette_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    palette: &str,
    boost_oiii: f32,
    desat_cores: bool,
) -> Bound<'py, PyArray3<f32>> {
    let img = img_rgb.as_array();
    let (h, w, c) = img.dim();

    let palette_upper = palette.trim().to_uppercase();
    if palette_upper == "NONE" || palette_upper == "OFF" || palette_upper == "RGB" {
        return img.to_owned().into_pyarray(py);
    }

    let out_vec = py.allow_threads(|| {
        let total_pixels = h * w;
        let mut r_all = Vec::with_capacity(total_pixels);
        let mut g_all = Vec::with_capacity(total_pixels);
        let mut b_all = Vec::with_capacity(total_pixels);

        for y in 0..h {
            for x in 0..w {
                r_all.push(unsafe { *img.uget((y, x, 0)) });
                g_all.push(unsafe { *img.uget((y, x, 1)) });
                b_all.push(unsafe { *img.uget((y, x, 2)) });
            }
        }

        let bg_r = percentile_f32_sampled(&r_all, 5.0);
        let bg_g = percentile_f32_sampled(&g_all, 5.0);
        let bg_b = percentile_f32_sampled(&b_all, 5.0);
        let med_bg = (bg_r + bg_g + bg_b) / 3.0;

        let o3_scale = boost_oiii.max(0.1);

        // Precompute Ha and OIII
        let mut ha_vec = vec![0.0f32; total_pixels];
        let mut oiii_vec = vec![0.0f32; total_pixels];

        ha_vec.par_chunks_mut(w).zip(oiii_vec.par_chunks_mut(w)).enumerate().for_each(|(y, (ha_row, o3_row))| {
            for x in 0..w {
                let r = unsafe { *img.uget((y, x, 0)) };
                let g = unsafe { *img.uget((y, x, 1)) };
                let b = unsafe { *img.uget((y, x, 2)) };

                let ha_raw = (r - bg_r).max(0.0);
                let oiii_raw = (0.5 * (g - bg_g) + 0.5 * (b - bg_b)).max(0.0);
                let oiii = (oiii_raw - 0.04 * ha_raw).max(0.0);

                ha_row[x] = ha_raw;
                o3_row[x] = oiii * o3_scale;
            }
        });

        let mut buf = vec![0.0f32; total_pixels * 3];

        if palette_upper == "SHO" {
            buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
                let offset = y * w;
                for x in 0..w {
                    let ha = ha_vec[offset + x];
                    let o3 = oiii_vec[offset + x];
                    let sii = 0.45 * ha;
                    let idx = x * 3;
                    row_slice[idx] = sii + med_bg;
                    row_slice[idx + 1] = (0.80 * ha + 0.20 * o3) + med_bg;
                    row_slice[idx + 2] = o3 + med_bg;
                }
            });
        } else if palette_upper == "FORAXX" {
            let max_ha = ha_vec.par_iter().cloned().reduce(|| 0.0f32, f32::max) + 1e-6;
            let max_o3 = oiii_vec.par_iter().cloned().reduce(|| 0.0f32, f32::max) + 1e-6;

            buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
                let offset = y * w;
                for x in 0..w {
                    let ha = ha_vec[offset + x];
                    let o3 = oiii_vec[offset + x];

                    let norm_ha = (ha / max_ha).clamp(0.0, 1.0);
                    let norm_o3 = (o3 / max_o3).clamp(0.0, 1.0);
                    let prod = norm_ha * norm_o3;
                    let exp = (1.0 - prod).max(0.01);
                    let f = prod.powf(exp);

                    let g_synth = f * ha + (1.0 - f) * o3;
                    let idx = x * 3;
                    row_slice[idx] = ha + med_bg;
                    row_slice[idx + 1] = g_synth + med_bg;
                    row_slice[idx + 2] = o3 + med_bg;
                }
            });
        } else if palette_upper == "CANNISTRA" {
            buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
                let offset = y * w;
                for x in 0..w {
                    let ha = ha_vec[offset + x];
                    let o3 = oiii_vec[offset + x];
                    let idx = x * 3;
                    row_slice[idx] = ha + med_bg;
                    row_slice[idx + 1] = o3.max(0.50 * ha) + med_bg;
                    row_slice[idx + 2] = o3 + med_bg;
                }
            });
        } else {
            // Default HOO
            buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
                let offset = y * w;
                for x in 0..w {
                    let ha = ha_vec[offset + x];
                    let o3 = oiii_vec[offset + x];
                    let idx = x * 3;
                    row_slice[idx] = ha + med_bg;
                    row_slice[idx + 1] = o3 + med_bg;
                    row_slice[idx + 2] = o3 + med_bg;
                }
            });
        }

        if desat_cores {
            desaturate_cores_in_place(&mut buf, h, w, 0.75, 0.20);
        }

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}

// ----------------------------------------------------------------------------
// Star Unclip: Rescues saturated cores using outer perimeter PSF chromatic ratios
// ----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (img_rgb, sat_threshold=0.92, ring_width=3))]
pub fn unclip_stars_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    sat_threshold: f32,
    ring_width: i32,
) -> Bound<'py, PyArray3<f32>> {
    let img = img_rgb.as_array();
    let (h, w, c) = img.dim();

    let out_vec = py.allow_threads(|| {
        let total = h * w;
        let mut buf = vec![0.0f32; total * 3];

        // Copy img to buf
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                buf[idx] = unsafe { *img.uget((y, x, 0)) };
                buf[idx + 1] = unsafe { *img.uget((y, x, 1)) };
                buf[idx + 2] = unsafe { *img.uget((y, x, 2)) };
            }
        }

        // Build sat_mask (1 if any channel >= sat_threshold)
        let mut sat_mask = vec![false; total];
        let mut has_sat = false;
        for i in 0..total {
            let r = buf[i * 3];
            let g = buf[i * 3 + 1];
            let b = buf[i * 3 + 2];
            if r >= sat_threshold || g >= sat_threshold || b >= sat_threshold {
                sat_mask[i] = true;
                has_sat = true;
            }
        }

        if !has_sat {
            return buf;
        }

        // Connected components on sat_mask
        let mut labels = vec![0u32; total];
        let mut current_label = 0u32;

        let mut queue = std::collections::VecDeque::new();
        let eff_r = ring_width.max(2);

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if sat_mask[idx] && labels[idx] == 0 {
                    current_label += 1;
                    labels[idx] = current_label;
                    queue.push_back((y, x));

                    let mut min_y = y;
                    let mut max_y = y;
                    let mut min_x = x;
                    let mut max_x = x;
                    let mut comp_pixels = Vec::new();

                    while let Some((cy, cx)) = queue.pop_front() {
                        comp_pixels.push((cy, cx));
                        if cy < min_y { min_y = cy; }
                        if cy > max_y { max_y = cy; }
                        if cx < min_x { min_x = cx; }
                        if cx > max_x { max_x = cx; }

                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dy == 0 && dx == 0 { continue; }
                                let ny = cy as i32 + dy;
                                let nx = cx as i32 + dx;
                                if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                                    let nidx = ny as usize * w + nx as usize;
                                    if sat_mask[nidx] && labels[nidx] == 0 {
                                        labels[nidx] = current_label;
                                        queue.push_back((ny as usize, nx as usize));
                                    }
                                }
                            }
                        }
                    }

                    // Identify perimeter ring around this component using exact OpenCV ellipse dilation
                    let pad = eff_r as usize;
                    let ry1 = min_y.saturating_sub(pad);
                    let ry2 = (max_y + pad + 1).min(h);
                    let rx1 = min_x.saturating_sub(pad);
                    let rx2 = (max_x + pad + 1).min(w);

                    let sub_h = ry2 - ry1;
                    let sub_w = rx2 - rx1;
                    let mut is_dilated = vec![false; sub_h * sub_w];

                    for &(cy, cx) in &comp_pixels {
                        for i in 0..=(2 * eff_r) {
                            let dy = i - eff_r;
                            let dx = ((eff_r as f64) * (((eff_r * eff_r - dy * dy) as f64) / ((eff_r * eff_r) as f64)).sqrt() + 0.5) as i32;
                            let ny = cy as i32 + dy;
                            if ny >= ry1 as i32 && ny < ry2 as i32 {
                                let local_y = (ny - ry1 as i32) as usize;
                                let min_nx = (cx as i32 - dx).max(rx1 as i32);
                                let max_nx = (cx as i32 + dx).min(rx2 as i32 - 1);
                                for nx in min_nx..=max_nx {
                                    let local_x = (nx - rx1 as i32) as usize;
                                    is_dilated[local_y * sub_w + local_x] = true;
                                }
                            }
                        }
                    }

                    let mut rg_ratios = Vec::new();
                    let mut bg_ratios = Vec::new();
                    let eps = 1e-6f32;

                    for py_cur in ry1..ry2 {
                        let local_y = py_cur - ry1;
                        for px_cur in rx1..rx2 {
                            let local_x = px_cur - rx1;
                            let pidx = py_cur * w + px_cur;
                            if !sat_mask[pidx] && is_dilated[local_y * sub_w + local_x] {
                                let r_val = buf[pidx * 3];
                                let g_val = buf[pidx * 3 + 1];
                                let b_val = buf[pidx * 3 + 2];
                                rg_ratios.push((r_val + eps) / (g_val + eps));
                                bg_ratios.push((b_val + eps) / (g_val + eps));
                            }
                        }
                    }

                    if !rg_ratios.is_empty() {
                        let ratio_rg = median_f32_vec(rg_ratios);
                        let ratio_bg = median_f32_vec(bg_ratios);
                        let denom = 0.2126 * ratio_rg + 0.7152 + 0.0722 * ratio_bg + eps;

                        for (cy, cx) in comp_pixels {
                            let cidx = (cy * w + cx) * 3;
                            let r = buf[cidx];
                            let g = buf[cidx + 1];
                            let b = buf[cidx + 2];
                            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                            let synth_g = luma / denom;
                            let synth_r = synth_g * ratio_rg;
                            let synth_b = synth_g * ratio_bg;

                            buf[cidx] = synth_r;
                            buf[cidx + 1] = synth_g;
                            buf[cidx + 2] = synth_b;
                        }
                    }
                }
            }
        }

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}

// ----------------------------------------------------------------------------
// Fused Dual-Band Pipeline: Palette + Unclip + Unpurple in a single optimized pass
// ----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (img_rgb, palette="HOO", boost_oiii=1.0, desat_cores=true, unclip_amount=0.0, unpurple_amount=0.0, sky_mask=None))]
pub fn apply_dualband_pipeline_fused<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    palette: &str,
    boost_oiii: f32,
    desat_cores: bool,
    unclip_amount: f32,
    unpurple_amount: f32,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
) -> Bound<'py, PyArray3<f32>> {
    let _ = sky_mask;
    // 1. Dual-band palette
    let p_img = apply_dualband_palette_rs(py, img_rgb, palette, boost_oiii, desat_cores);

    // 2. Unclip stars if requested
    let u_img = if unclip_amount > 1e-4 {
        unclip_stars_rs(py, p_img.readonly(), 0.92, 3)
    } else {
        p_img
    };

    // 3. Unpurple if requested
    if unpurple_amount > 1e-4 {
        unpurple_rs(py, u_img.readonly(), unpurple_amount, true)
    } else {
        u_img
    }
}
