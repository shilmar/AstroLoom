use numpy::ndarray::{Array3, ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArray3, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// FASE 8: SUPRESIÓN DE BANDING CMOS Y FRONTERA CIELO/SUELO SDF
// (core/banding.py, core/masking.py y core/stacking.py)
// ============================================================================

#[inline(always)]
fn median_f32_slice(data: &mut [f32]) -> f32 {
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
// 1D Gaussian smoothing with nearest-neighbor boundary condition
// ----------------------------------------------------------------------------
fn gaussian_filter_1d(profile: &[f32], sigma: f32) -> Vec<f32> {
    let n = profile.len();
    if n == 0 {
        return Vec::new();
    }
    let s = sigma.max(1.0);
    let radius = (4.0 * s + 0.5) as usize;
    let k_size = 2 * radius + 1;

    let mut kernel = Vec::with_capacity(k_size);
    let mut sum = 0.0f32;
    for i in 0..k_size {
        let x = (i as i32 - radius as i32) as f32;
        let w = (-0.5 * (x / s).powi(2)).exp();
        kernel.push(w);
        sum += w;
    }
    let inv_sum = 1.0 / sum;
    for w in &mut kernel {
        *w *= inv_sum;
    }

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut acc = 0.0f32;
        for (k_idx, &w) in kernel.iter().enumerate() {
            let src_idx = (i as i32 + (k_idx as i32 - radius as i32)).clamp(0, (n - 1) as i32) as usize;
            acc += profile[src_idx] * w;
        }
        out[i] = acc;
    }

    out
}

// ----------------------------------------------------------------------------
// Core 1D Banding Suppression on a 2D float32 slice (rows)
// ----------------------------------------------------------------------------
fn suppress_1d_banding_rows(
    data: &mut [f32],
    h: usize,
    w: usize,
    amount: f32,
    protect_sigma: f32,
    smooth_sigma: f32,
    sky_mask: Option<&[f32]>,
    subsample_step: usize,
) {
    if amount <= 1e-4 {
        return;
    }

    // 1. Sky background estimation (Median and MAD)
    let mut sky_pts = Vec::new();
    if let Some(mask) = sky_mask {
        for i in 0..(h * w) {
            if mask[i] > 0.5 {
                sky_pts.push(data[i]);
            }
        }
    }
    if sky_pts.len() < 100 {
        sky_pts = data.to_vec();
    }

    let n_pts = sky_pts.len();
    let step = (n_pts / 50_000).max(1);
    let mut sample: Vec<f32> = sky_pts.iter().step_by(step).copied().collect();
    let med_bg = median_f32_slice(&mut sample);

    let mut dev: Vec<f32> = sample.iter().map(|&v| (v - med_bg).abs()).collect();
    let mad_bg = median_f32_slice(&mut dev);
    let sigma_bg = (1.4826 * mad_bg).clamp(1e-4, 0.020);

    // 2. Bounds
    let lo_bound = med_bg - 2.5 * sigma_bg;
    let hi_bound = med_bg + protect_sigma * sigma_bg;

    let sub_step = subsample_step.clamp(1, (w / 50).max(1));
    let sub_cols = (w + sub_step - 1) / sub_step;
    let min_valid_cols = ((sub_cols as f32 * 0.15).round() as usize).max(10);

    // 3. Row medians
    let mut row_medians = vec![f32::NAN; h];

    row_medians.par_iter_mut().enumerate().for_each(|(y, row_med)| {
        let row_offset = y * w;
        let mut row_valid = Vec::with_capacity(sub_cols);

        for x in (0..w).step_by(sub_step) {
            let idx = row_offset + x;
            let val = data[idx];
            let in_mask = if let Some(mask) = sky_mask {
                mask[idx] > 0.5
            } else {
                true
            };

            if in_mask && val >= lo_bound && val <= hi_bound {
                row_valid.push(val);
            }
        }

        if row_valid.len() >= min_valid_cols {
            *row_med = median_f32_slice(&mut row_valid);
        }
    });

    // 4. Interpolate NaN rows
    let valid_indices: Vec<usize> = (0..h).filter(|&i| !row_medians[i].is_nan()).collect();
    if valid_indices.is_empty() {
        for v in &mut row_medians {
            *v = med_bg;
        }
    } else if valid_indices.len() < h {
        // Nearest or linear interpolation for NaN entries
        for i in 0..h {
            if row_medians[i].is_nan() {
                // Find previous and next valid
                let prev = valid_indices.iter().filter(|&&idx| idx < i).max().copied();
                let next = valid_indices.iter().filter(|&&idx| idx > i).min().copied();

                match (prev, next) {
                    (Some(p), Some(n)) => {
                        let t = (i - p) as f32 / (n - p) as f32;
                        row_medians[i] = row_medians[p] * (1.0 - t) + row_medians[n] * t;
                    }
                    (Some(p), None) => row_medians[i] = row_medians[p],
                    (None, Some(n)) => row_medians[i] = row_medians[n],
                    (None, None) => row_medians[i] = med_bg,
                }
            }
        }
    }

    // 5. Frequency separation: Gaussian low-pass vs high frequency banding
    let smooth_profile = gaussian_filter_1d(&row_medians, smooth_sigma.max(3.0));

    // 6. Subtract banding pattern along rows
    let mut banding_pattern = vec![0.0f32; h];
    for i in 0..h {
        let raw_diff = row_medians[i] - smooth_profile[i];
        banding_pattern[i] = raw_diff.clamp(-0.015, 0.015) * amount;
    }

    data.par_chunks_mut(w).enumerate().for_each(|(y, row_slice)| {
        let pat = banding_pattern[y];
        for x in 0..w {
            row_slice[x] -= pat;
        }
    });
}

#[pyfunction]
#[pyo3(signature = (image, direction="horizontal", amount=1.0, protect_sigma=2.5, smooth_sigma=35.0, sky_mask=None))]
pub fn suppress_cmos_banding_rs<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, f32>,
    direction: &str,
    amount: f32,
    protect_sigma: f32,
    smooth_sigma: f32,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
) -> Bound<'py, PyArrayDyn<f32>> {
    let img_view = image.as_array();
    let shape = img_view.shape();
    let dir_lower = direction.trim().to_lowercase();

    let mask_opt = sky_mask.as_ref().map(|m| m.as_array());

    let (h, w, c, is_2d) = if shape.len() == 2 {
        (shape[0], shape[1], 1, true)
    } else if shape.len() == 3 {
        (shape[0], shape[1], shape[2], false)
    } else {
        panic!("Image must be 2D or 3D");
    };

    let out_vec = py.allow_threads(|| {
        let total = h * w * c;
        let mut out = vec![0.0f32; total];

        let mask_flat: Option<Vec<f32>> = mask_opt.map(|m| {
            let mut mf = Vec::with_capacity(h * w);
            for y in 0..h {
                for x in 0..w {
                    mf.push(unsafe { *m.uget((y, x)) });
                }
            }
            mf
        });

        // Copy channel by channel and process
        for ch in 0..c {
            let mut ch_data = vec![0.0f32; h * w];
            for y in 0..h {
                for x in 0..w {
                    let val = if is_2d {
                        unsafe { *img_view.uget([y, x]) }
                    } else {
                        unsafe { *img_view.uget([y, x, ch]) }
                    };
                    ch_data[y * w + x] = val;
                }
            }

            if dir_lower == "horizontal" || dir_lower == "both" {
                suppress_1d_banding_rows(
                    &mut ch_data,
                    h,
                    w,
                    amount,
                    protect_sigma,
                    smooth_sigma,
                    mask_flat.as_deref(),
                    4,
                );
            }

            if dir_lower == "vertical" || dir_lower == "both" {
                // Transpose to apply row suppression on columns
                let mut transposed = vec![0.0f32; w * h];
                for y in 0..h {
                    for x in 0..w {
                        transposed[x * h + y] = ch_data[y * w + x];
                    }
                }
                let mask_transposed: Option<Vec<f32>> = mask_flat.as_ref().map(|mf| {
                    let mut mt = vec![0.0f32; w * h];
                    for y in 0..h {
                        for x in 0..w {
                            mt[x * h + y] = mf[y * w + x];
                        }
                    }
                    mt
                });

                suppress_1d_banding_rows(
                    &mut transposed,
                    w,
                    h,
                    amount,
                    protect_sigma,
                    smooth_sigma,
                    mask_transposed.as_deref(),
                    4,
                );

                // Transpose back
                for y in 0..h {
                    for x in 0..w {
                        ch_data[y * w + x] = transposed[x * h + y];
                    }
                }
            }

            // Put into out
            if is_2d {
                out.copy_from_slice(&ch_data);
            } else {
                for y in 0..h {
                    for x in 0..w {
                        out[(y * w + x) * c + ch] = ch_data[y * w + x];
                    }
                }
            }
        }

        out
    });

    let shape_dyn = if is_2d {
        vec![h, w]
    } else {
        vec![h, w, c]
    };
    let out_arr = ArrayD::from_shape_vec(IxDyn(&shape_dyn), out_vec).unwrap();
    out_arr.into_pyarray(py)
}

// ----------------------------------------------------------------------------
// 2D Gaussian Blur (5x5, sigma=1.0)
// ----------------------------------------------------------------------------
fn gaussian_blur_5x5(data: &[f32], h: usize, w: usize) -> Vec<f32> {
    let k = [0.05448868f32, 0.24420134, 0.40261995, 0.24420134, 0.05448868];
    let mut temp = vec![0.0f32; h * w];

    // Horizontal pass
    temp.par_chunks_mut(w).enumerate().for_each(|(y, row_slice)| {
        let row_offset = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kw) in k.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - 2).clamp(0, (w - 1) as i32) as usize;
                acc += data[row_offset + sx] * kw;
            }
            row_slice[x] = acc;
        }
    });

    let mut out = vec![0.0f32; h * w];

    // Vertical pass
    out.par_chunks_mut(w).enumerate().for_each(|(y, row_slice)| {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kw) in k.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - 2).clamp(0, (h - 1) as i32) as usize;
                acc += temp[sy * w + x] * kw;
            }
            row_slice[x] = acc;
        }
    });

    out
}

// ----------------------------------------------------------------------------
// SDF Sky/Ground Anti-Ghosting Composite (core/stacking.py)
// ----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (sky_stacked, ground_layer, sdf_map, raw_mask, boundary_radius=25.0, tree_sensitivity=0.5))]
pub fn compose_sky_ground_sdf_rs<'py>(
    py: Python<'py>,
    sky_stacked: PyReadonlyArray3<'py, f32>,
    ground_layer: PyReadonlyArray3<'py, f32>,
    sdf_map: PyReadonlyArray2<'py, f32>,
    raw_mask: PyReadonlyArray2<'py, f32>,
    boundary_radius: f32,
    tree_sensitivity: f32,
) -> Bound<'py, PyArray3<f32>> {
    let sky = sky_stacked.as_array();
    let gnd = ground_layer.as_array();
    let sdf = sdf_map.as_array();
    let mask = raw_mask.as_array();

    let (h, w, c) = sky.dim();

    let out_vec = py.allow_threads(|| {
        let total = h * w;

        // 1. Estimate reference luminances
        let mut sky_lum_samples = Vec::new();
        let mut gnd_lum_samples = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let m = unsafe { *mask.uget((y, x)) };
                let r_s = unsafe { *sky.uget((y, x, 0)) };
                let g_s = unsafe { *sky.uget((y, x, 1)) };
                let b_s = unsafe { *sky.uget((y, x, 2)) };
                let lum_s = 0.299 * r_s + 0.587 * g_s + 0.114 * b_s;

                let r_g = unsafe { *gnd.uget((y, x, 0)) };
                let g_g = unsafe { *gnd.uget((y, x, 1)) };
                let b_g = unsafe { *gnd.uget((y, x, 2)) };
                let lum_g = 0.299 * r_g + 0.587 * g_g + 0.114 * b_g;

                if m > 0.6 {
                    sky_lum_samples.push(lum_s);
                } else if m < 0.4 {
                    gnd_lum_samples.push(lum_g);
                }
            }
        }

        let sky_ref_val = if sky_lum_samples.is_empty() {
            0.15f32
        } else {
            median_f32_slice(&mut sky_lum_samples)
        };
        let gnd_ref_val = if gnd_lum_samples.is_empty() {
            0.02f32
        } else {
            median_f32_slice(&mut gnd_lum_samples)
        };

        let sens = tree_sensitivity.clamp(0.1, 1.0);
        let thresh = gnd_ref_val + (sky_ref_val - gnd_ref_val) * (0.20 + 0.30 * sens);
        let rad = boundary_radius.max(4.0);

        let mut w_smooth = vec![0.0f32; total];
        let mut dark_branch = vec![0.0f32; total];
        let mut has_branch = false;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let s_val = unsafe { *sdf.uget((y, x)) };

                // Optical cosine feather
                let w_lin = ((s_val / 5.0) + 0.5).clamp(0.0, 1.0);
                w_smooth[idx] = 0.5 - 0.5 * (std::f32::consts::PI * w_lin).cos();

                if s_val > 0.0 && s_val <= rad && tree_sensitivity > 0.05 {
                    let r_s = unsafe { *sky.uget((y, x, 0)) };
                    let g_s = unsafe { *sky.uget((y, x, 1)) };
                    let b_s = unsafe { *sky.uget((y, x, 2)) };
                    let lum_s = 0.299 * r_s + 0.587 * g_s + 0.114 * b_s;

                    let r_g = unsafe { *gnd.uget((y, x, 0)) };
                    let g_g = unsafe { *gnd.uget((y, x, 1)) };
                    let b_g = unsafe { *gnd.uget((y, x, 2)) };
                    let lum_g = 0.299 * r_g + 0.587 * g_g + 0.114 * b_g;

                    if lum_g < thresh && lum_g < (lum_s - 0.005) {
                        dark_branch[idx] = 1.0;
                        has_branch = true;
                    }
                }
            }
        }

        if has_branch {
            let branch_smooth = gaussian_blur_5x5(&dark_branch, h, w);
            for i in 0..total {
                w_smooth[i] = w_smooth[i].min(1.0 - branch_smooth[i]);
            }
        }

        // Final blend
        let mut buf = vec![0.0f32; total * 3];
        buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
            let offset = y * w;
            for x in 0..w {
                let w_val = w_smooth[offset + x].clamp(0.0, 1.0);
                let inv_w = 1.0 - w_val;
                let idx = x * 3;

                for ch in 0..3 {
                    let s_px = unsafe { *sky.uget((y, x, ch)) };
                    let g_px = unsafe { *gnd.uget((y, x, ch)) };
                    row_slice[idx + ch] = s_px * w_val + g_px * inv_w;
                }
            }
        });

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}

// ----------------------------------------------------------------------------
// Horizon Anti-Halo Suppression (core/masking.py)
// ----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (image_rgb, sky_mask, radius=3, strength=0.8, suppress_dark_halos=false))]
pub fn suppress_horizon_halos_rs<'py>(
    py: Python<'py>,
    image_rgb: PyReadonlyArray3<'py, f32>,
    sky_mask: PyReadonlyArray2<'py, f32>,
    radius: i32,
    strength: f32,
    suppress_dark_halos: bool,
) -> Bound<'py, PyArray3<f32>> {
    let img = image_rgb.as_array();
    let mask = sky_mask.as_array();
    let (h, w, c) = img.dim();

    if strength <= 1e-4 || radius < 1 {
        return img.to_owned().into_pyarray(py);
    }

    let out_vec = py.allow_threads(|| {
        let total = h * w;
        let mut buf = vec![0.0f32; total * 3];

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                buf[idx] = unsafe { *img.uget((y, x, 0)) };
                buf[idx + 1] = unsafe { *img.uget((y, x, 1)) };
                buf[idx + 2] = unsafe { *img.uget((y, x, 2)) };
            }
        }

        let r = radius.max(1);
        let r_sq = (r * r) as f32;

        // Detect boundary: pixels where dilation - erosion of binary mask > 0
        let mut min_y = h;
        let mut max_y = 0;
        let mut min_x = w;
        let mut max_x = 0;
        let mut has_boundary = false;

        let mut boundary = vec![0.0f32; total];

        for y in 0..h {
            for x in 0..w {
                let mut has_sky = false;
                let mut has_gnd = false;

                for dy in -r..=r {
                    for dx in -r..=r {
                        if (dy * dy + dx * dx) as f32 <= r_sq {
                            let ny = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;
                            let nx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;
                            let m = unsafe { *mask.uget((ny, nx)) } > 0.5;
                            if m { has_sky = true; } else { has_gnd = true; }
                        }
                    }
                }

                if has_sky && has_gnd {
                    boundary[y * w + x] = 1.0;
                    has_boundary = true;
                    if y < min_y { min_y = y; }
                    if y > max_y { max_y = y; }
                    if x < min_x { min_x = x; }
                    if x > max_x { max_x = x; }
                }
            }
        }

        if !has_boundary {
            return buf;
        }

        let pad = (r * 3) as usize;
        let y1 = min_y.saturating_sub(pad);
        let y2 = (max_y + pad + 1).min(h);
        let x1 = min_x.saturating_sub(pad);
        let x2 = (max_x + pad + 1).min(w);

        let sub_h = y2 - y1;
        let sub_w = x2 - x1;

        // Compute inner ground in sub-region (ground eroded by r)
        let mut inner_gnd = vec![false; sub_h * sub_w];
        let mut any_inner_gnd = false;

        for sy in 0..sub_h {
            let y = y1 + sy;
            for sx in 0..sub_w {
                let x = x1 + sx;
                let mut is_pure_gnd = true;
                for dy in -r..=r {
                    for dx in -r..=r {
                        if (dy * dy + dx * dx) as f32 <= r_sq {
                            let ny = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;
                            let nx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;
                            if unsafe { *mask.uget((ny, nx)) } > 0.5 {
                                is_pure_gnd = false;
                                break;
                            }
                        }
                    }
                    if !is_pure_gnd { break; }
                }
                if is_pure_gnd {
                    inner_gnd[sy * sub_w + sx] = true;
                    any_inner_gnd = true;
                }
            }
        }

        // Bounding-box minimum filter for ground halo clamping
        for sy in 0..sub_h {
            let y = y1 + sy;
            for sx in 0..sub_w {
                let x = x1 + sx;
                let idx = y * w + x;
                let b_val = boundary[idx];
                let m_val = unsafe { *mask.uget((y, x)) };

                if b_val > 0.0 && m_val < 0.5 {
                    // Search local ground minimum over inner_gnd within radius r
                    let mut min_r = 1.0f32;
                    let mut min_g = 1.0f32;
                    let mut min_b = 1.0f32;

                    for dy in -r..=r {
                        for dx in -r..=r {
                            if (dy * dy + dx * dx) as f32 <= r_sq {
                                let ny = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;
                                let nx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;
                                let is_gnd_candidate = if any_inner_gnd {
                                    if ny >= y1 && ny < y2 && nx >= x1 && nx < x2 {
                                        inner_gnd[(ny - y1) * sub_w + (nx - x1)]
                                    } else {
                                        let val = unsafe { *mask.uget((ny, nx)) };
                                        val <= 0.5
                                    }
                                } else {
                                    let val = unsafe { *mask.uget((ny, nx)) };
                                    val <= 0.5
                                };

                                if is_gnd_candidate {
                                    let cidx = (ny * w + nx) * 3;
                                    min_r = min_r.min(buf[cidx]);
                                    min_g = min_g.min(buf[cidx + 1]);
                                    min_b = min_b.min(buf[cidx + 2]);
                                }
                            }
                        }
                    }

                    let w_blend = ((1.0 - m_val) * strength).clamp(0.0, 1.0);
                    let cidx = idx * 3;
                    buf[cidx] = buf[cidx] * (1.0 - w_blend) + min_r.min(buf[cidx]) * w_blend;
                    buf[cidx + 1] = buf[cidx + 1] * (1.0 - w_blend) + min_g.min(buf[cidx + 1]) * w_blend;
                    buf[cidx + 2] = buf[cidx + 2] * (1.0 - w_blend) + min_b.min(buf[cidx + 2]) * w_blend;
                } else if suppress_dark_halos && b_val > 0.0 && m_val >= 0.5 {
                    // Sky maximum filter
                    let mut max_r = 0.0f32;
                    let mut max_g = 0.0f32;
                    let mut max_b = 0.0f32;

                    for dy in -r..=r {
                        for dx in -r..=r {
                            if (dy * dy + dx * dx) as f32 <= r_sq {
                                let ny = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;
                                let nx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;
                                let val = unsafe { *mask.uget((ny, nx)) };
                                if val >= 0.5 {
                                    let cidx = (ny * w + nx) * 3;
                                    max_r = max_r.max(buf[cidx]);
                                    max_g = max_g.max(buf[cidx + 1]);
                                    max_b = max_b.max(buf[cidx + 2]);
                                }
                            }
                        }
                    }

                    let w_blend = (m_val * strength).clamp(0.0, 1.0);
                    let cidx = idx * 3;
                    buf[cidx] = buf[cidx] * (1.0 - w_blend) + max_r.max(buf[cidx]) * w_blend;
                    buf[cidx + 1] = buf[cidx + 1] * (1.0 - w_blend) + max_g.max(buf[cidx + 1]) * w_blend;
                    buf[cidx + 2] = buf[cidx + 2] * (1.0 - w_blend) + max_b.max(buf[cidx + 2]) * w_blend;
                }
            }
        }

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}
