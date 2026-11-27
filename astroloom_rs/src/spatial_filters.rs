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
        let img_owned;
        let img_slice: &[f32] = match img.as_slice() {
            Some(s) => s,
            None => {
                img_owned = img.to_owned();
                img_owned.as_slice().unwrap()
            }
        };
        let mask_owned;
        let mask_slice: &[f32] = match mask.as_slice() {
            Some(s) => s,
            None => {
                mask_owned = mask.to_owned();
                mask_owned.as_slice().unwrap()
            }
        };

        let mut buf = img_slice.to_vec();

        let r = radius.max(1);
        let r_sq = r * r;
        let mut offsets: Vec<(i32, i32)> = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                if (dy != 0 || dx != 0) && (dy * dy + dx * dx) <= r_sq {
                    offsets.push((dy, dx));
                }
            }
        }

        // Parallel boundary detection with early exit per pixel
        let mut boundary = vec![0u8; total];
        boundary
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row_b)| {
                let yi = y as i32;
                let h_max = (h - 1) as i32;
                let w_max = (w - 1) as i32;
                for x in 0..w {
                    let center_sky = mask_slice[y * w + x] > 0.5;
                    let xi = x as i32;
                    for &(dy, dx) in &offsets {
                        let ny = (yi + dy).clamp(0, h_max) as usize;
                        let nx = (xi + dx).clamp(0, w_max) as usize;
                        if (mask_slice[ny * w + nx] > 0.5) != center_sky {
                            row_b[x] = 1;
                            break;
                        }
                    }
                }
            });

        let mut min_y = h;
        let mut max_y = 0usize;
        let mut min_x = w;
        let mut max_x = 0usize;
        let mut has_boundary = false;
        for y in 0..h {
            let row_off = y * w;
            for x in 0..w {
                if boundary[row_off + x] != 0 {
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

        // Compute inner ground in sub-region (ground eroded by r) in parallel
        let mut inner_gnd = vec![false; sub_h * sub_w];
        inner_gnd
            .par_chunks_mut(sub_w)
            .enumerate()
            .for_each(|(sy, row_ig)| {
                let y = y1 + sy;
                let yi = y as i32;
                let h_max = (h - 1) as i32;
                let w_max = (w - 1) as i32;
                for sx in 0..sub_w {
                    let x = x1 + sx;
                    if mask_slice[y * w + x] > 0.5 {
                        continue;
                    }
                    let xi = x as i32;
                    let mut is_pure_gnd = true;
                    for &(dy, dx) in &offsets {
                        let ny = (yi + dy).clamp(0, h_max) as usize;
                        let nx = (xi + dx).clamp(0, w_max) as usize;
                        if mask_slice[ny * w + nx] > 0.5 {
                            is_pure_gnd = false;
                            break;
                        }
                    }
                    row_ig[sx] = is_pure_gnd;
                }
            });
        let any_inner_gnd = inner_gnd.iter().any(|&v| v);

        // Parallel bounding-box clamping reading from immutable img_slice
        let h_max = (h - 1) as i32;
        let w_max = (w - 1) as i32;
        buf[y1 * w * 3..y2 * w * 3]
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(sy, row_out)| {
                let y = y1 + sy;
                let yi = y as i32;
                for sx in 0..sub_w {
                    let x = x1 + sx;
                    let idx = y * w + x;
                    if boundary[idx] == 0 {
                        continue;
                    }
                    let m_val = mask_slice[idx];
                    let xi = x as i32;
                    let c_row = x * 3;
                    let src_r = img_slice[idx * 3];
                    let src_g = img_slice[idx * 3 + 1];
                    let src_b = img_slice[idx * 3 + 2];

                    if m_val < 0.5 {
                        let mut min_r = 1.0f32;
                        let mut min_g = 1.0f32;
                        let mut min_b = 1.0f32;
                        for &(dy, dx) in &offsets {
                            let ny = (yi + dy).clamp(0, h_max) as usize;
                            let nx = (xi + dx).clamp(0, w_max) as usize;
                            let is_gnd_candidate = if any_inner_gnd {
                                if ny >= y1 && ny < y2 && nx >= x1 && nx < x2 {
                                    inner_gnd[(ny - y1) * sub_w + (nx - x1)]
                                } else {
                                    mask_slice[ny * w + nx] <= 0.5
                                }
                            } else {
                                mask_slice[ny * w + nx] <= 0.5
                            };
                            if is_gnd_candidate {
                                let cidx = (ny * w + nx) * 3;
                                min_r = min_r.min(img_slice[cidx]);
                                min_g = min_g.min(img_slice[cidx + 1]);
                                min_b = min_b.min(img_slice[cidx + 2]);
                            }
                        }
                        let w_blend = ((1.0 - m_val) * strength).clamp(0.0, 1.0);
                        let inv_w = 1.0 - w_blend;
                        row_out[c_row] = src_r * inv_w + min_r.min(src_r) * w_blend;
                        row_out[c_row + 1] = src_g * inv_w + min_g.min(src_g) * w_blend;
                        row_out[c_row + 2] = src_b * inv_w + min_b.min(src_b) * w_blend;
                    } else if suppress_dark_halos {
                        let mut max_r = 0.0f32;
                        let mut max_g = 0.0f32;
                        let mut max_b = 0.0f32;
                        for &(dy, dx) in &offsets {
                            let ny = (yi + dy).clamp(0, h_max) as usize;
                            let nx = (xi + dx).clamp(0, w_max) as usize;
                            if mask_slice[ny * w + nx] >= 0.5 {
                                let cidx = (ny * w + nx) * 3;
                                max_r = max_r.max(img_slice[cidx]);
                                max_g = max_g.max(img_slice[cidx + 1]);
                                max_b = max_b.max(img_slice[cidx + 2]);
                            }
                        }
                        let w_blend = (m_val * strength).clamp(0.0, 1.0);
                        let inv_w = 1.0 - w_blend;
                        row_out[c_row] = src_r * inv_w + max_r.max(src_r) * w_blend;
                        row_out[c_row + 1] = src_g * inv_w + max_g.max(src_g) * w_blend;
                        row_out[c_row + 2] = src_b * inv_w + max_b.max(src_b) * w_blend;
                    }
                }
            });

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}


// ============================================================================
// FASE 2 (ZOOM 1:1): ONDÍCULAS À-TROUS DISPERSAS, CLARITY/DEHAZE O(1) Y CONTRASTE LUT
// ============================================================================

#[inline(always)]
fn reflect_idx(pos: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let m = (len - 1) as isize;
    let mut v = pos;
    if v < 0 {
        v = -v;
    }
    if v > m {
        let cycle = 2 * m;
        v %= cycle;
        if v > m {
            v = cycle - v;
        }
    }
    v as usize
}

/// 1D horizontal sliding-window box blur (O(1) per pixel regardless of radius), Rayon parallelized.
fn box_blur_h(src: &[f32], dst: &mut [f32], _h: usize, w: usize, r: usize) {
    if r == 0 || w <= 1 {
        dst.copy_from_slice(src);
        return;
    }
    let inv_k = 1.0f32 / (2 * r + 1) as f32;
    let ri = r as isize;
    dst.par_chunks_exact_mut(w).enumerate().for_each(|(y, out_row)| {
        let in_row = &src[y * w..(y + 1) * w];
        let mut acc = 0.0f64;
        for dx in -ri..=ri {
            acc += in_row[reflect_idx(dx, w)] as f64;
        }
        out_row[0] = (acc as f32) * inv_k;
        for x in 1..w {
            let add_i = reflect_idx(x as isize + ri, w);
            let sub_i = reflect_idx(x as isize - ri - 1, w);
            acc += in_row[add_i] as f64 - in_row[sub_i] as f64;
            out_row[x] = (acc as f32) * inv_k;
        }
    });
}

/// 1D vertical sliding-window box blur (O(1) per pixel regardless of radius), Rayon parallelized over column strips.
fn box_blur_v(src: &[f32], dst: &mut [f32], h: usize, w: usize, r: usize) {
    if r == 0 || h <= 1 {
        dst.copy_from_slice(src);
        return;
    }
    let inv_k = 1.0f32 / (2 * r + 1) as f32;
    let ri = r as isize;
    let strip = 64usize;
    let n_strips = (w + strip - 1) / strip;
    let dst_ptr = dst.as_mut_ptr() as usize;

    (0..n_strips).into_par_iter().for_each(|s| {
        let x0 = s * strip;
        let x1 = (x0 + strip).min(w);
        let sw = x1 - x0;
        let mut acc = vec![0.0f64; sw];
        for dy in -ri..=ri {
            let ry = reflect_idx(dy, h) * w + x0;
            for j in 0..sw {
                acc[j] += src[ry + j] as f64;
            }
        }
        let ptr = dst_ptr as *mut f32;
        unsafe {
            for j in 0..sw {
                *ptr.add(x0 + j) = (acc[j] as f32) * inv_k;
            }
            for y in 1..h {
                let add_row = reflect_idx(y as isize + ri, h) * w + x0;
                let sub_row = reflect_idx(y as isize - ri - 1, h) * w + x0;
                let out_row = y * w + x0;
                for j in 0..sw {
                    acc[j] += src[add_row + j] as f64 - src[sub_row + j] as f64;
                    *ptr.add(out_row + j) = (acc[j] as f32) * inv_k;
                }
            }
        }
    });
}

/// Fast O(1) Gaussian approximation via 3-pass separable box blur.
fn fast_gaussian_blur_2d(src: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.5 || h <= 1 || w <= 1 {
        return src.to_vec();
    }
    let n = 3.0f32;
    let w_ideal = ((12.0 * sigma * sigma / n) + 1.0).sqrt();
    let mut wl = w_ideal.floor() as usize;
    if wl % 2 == 0 {
        wl = wl.saturating_sub(1);
    }
    wl = wl.max(1);
    let wu = wl + 2;
    let m_ideal = (12.0 * sigma * sigma - n * (wl * wl) as f32 - 4.0 * n * wl as f32 - 3.0 * n)
        / (-4.0 * wl as f32 - 4.0);
    let m = m_ideal.round().clamp(0.0, n) as usize;

    let mut cur = src.to_vec();
    let mut tmp = vec![0.0f32; h * w];
    for pass in 0..3 {
        let box_w = if pass < m { wl } else { wu };
        let r = box_w / 2;
        if r > 0 {
            box_blur_h(&cur, &mut tmp, h, w, r);
            box_blur_v(&tmp, &mut cur, h, w, r);
        }
    }
    cur
}

/// Realce multiescala por ondículas À-Trous B3-Spline dispersas (5 taps reales por eje).
#[pyfunction]
#[pyo3(signature = (img_rgb, fuerza_estructura=0.0, reduccion_fondo=0.0, sky_mask=None, global_pedestal=None))]
pub fn atrous_wavelets_enhance_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    fuerza_estructura: f32,
    reduccion_fondo: f32,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
    global_pedestal: Option<f32>,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = img_rgb.as_array();
    let (h, w, c) = img_view.dim();
    let img_slice = img_view.as_slice().expect("img_rgb must be C-contiguous float32");
    let mask_view = sky_mask.as_ref().map(|m| m.as_array());
    let mask_slice_opt: Option<&[f32]> = mask_view.as_ref().and_then(|v| v.as_slice());
    let mask_owned: Option<Vec<f32>> = if mask_slice_opt.is_none() {
        mask_view.as_ref().map(|v| v.iter().copied().collect())
    } else {
        None
    };
    let mask_ref: Option<&[f32]> = mask_slice_opt.or(mask_owned.as_deref());

    let out_vec = py.allow_threads(|| {
        let npix = h * w;
        let mut lum = vec![0.0f32; npix];
        lum.par_chunks_exact_mut(w).enumerate().for_each(|(y, row)| {
            let base = y * w * c;
            for x in 0..w {
                let i = base + x * c;
                let l = 0.2126 * img_slice[i] + 0.7152 * img_slice[i + 1] + 0.0722 * img_slice[i + 2];
                row[x] = l.clamp(0.0, 1.0);
            }
        });

        let pesos = [
            0.0f32,
            0.0f32,
            fuerza_estructura * 0.4,
            fuerza_estructura * 0.8,
            fuerza_estructura * 0.6,
        ];

        let mut cur = lum.clone();
        let mut tmp = vec![0.0f32; npix];
        let mut next = vec![0.0f32; npix];
        let mut delta = vec![0.0f32; npix];

        for nivel in 0..5usize {
            let s = (1usize << nivel) as isize;
            let s2 = s * 2;

            tmp.par_chunks_exact_mut(w).enumerate().for_each(|(y, out_row)| {
                let in_row = &cur[y * w..(y + 1) * w];
                for x in 0..w {
                    let xi = x as isize;
                    let v0 = in_row[reflect_idx(xi - s2, w)];
                    let v1 = in_row[reflect_idx(xi - s, w)];
                    let v2 = in_row[x];
                    let v3 = in_row[reflect_idx(xi + s, w)];
                    let v4 = in_row[reflect_idx(xi + s2, w)];
                    out_row[x] = 0.0625 * (v0 + v4) + 0.25 * (v1 + v3) + 0.375 * v2;
                }
            });

            let w_delta = pesos[nivel];
            let acc_detail = w_delta.abs() > 1e-5;
            next.par_chunks_exact_mut(w)
                .zip(delta.par_chunks_exact_mut(w))
                .enumerate()
                .for_each(|(y, (out_row, d_row))| {
                    let yi = y as isize;
                    let r0 = reflect_idx(yi - s2, h) * w;
                    let r1 = reflect_idx(yi - s, h) * w;
                    let r2 = y * w;
                    let r3 = reflect_idx(yi + s, h) * w;
                    let r4 = reflect_idx(yi + s2, h) * w;
                    let cur_row = &cur[r2..r2 + w];
                    for x in 0..w {
                        let lp = 0.0625 * (tmp[r0 + x] + tmp[r4 + x])
                            + 0.25 * (tmp[r1 + x] + tmp[r3 + x])
                            + 0.375 * tmp[r2 + x];
                        out_row[x] = lp;
                        if acc_detail {
                            d_row[x] += (cur_row[x] - lp) * w_delta;
                        }
                    }
                });

            std::mem::swap(&mut cur, &mut next);
        }

        if reduccion_fondo > 1e-4 {
            let pedestal = if let Some(gp) = global_pedestal {
                gp
            } else {
                let mut sample = Vec::with_capacity((h / 4 + 1) * (w / 4 + 1));
                for y in (0..h).step_by(4) {
                    let row = &cur[y * w..(y + 1) * w];
                    for x in (0..w).step_by(4) {
                        sample.push(row[x]);
                    }
                }
                if sample.is_empty() {
                    0.0
                } else {
                    let k = (((sample.len() as f32) * 0.05).round() as usize).min(sample.len() - 1);
                    sample.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
                    sample[k]
                }
            };
            let bg_scale = reduccion_fondo * 0.5;
            delta.par_iter_mut().zip(cur.par_iter()).for_each(|(d, &res)| {
                *d -= (res - pedestal) * bg_scale;
            });
        }

        let mut out = vec![0.0f32; npix * c];
        out.par_chunks_exact_mut(w * c).enumerate().for_each(|(y, out_row)| {
            let row_pix = y * w;
            let row_rgb = row_pix * c;
            for x in 0..w {
                let p = row_pix + x;
                let mut d = delta[p];
                if let Some(m) = mask_ref {
                    d *= m[p];
                }
                let l = lum[p];
                let ratio = (l + d) / (l + 1e-6);
                let i = row_rgb + x * c;
                let o = x * c;
                out_row[o] = (img_slice[i] * ratio).clamp(0.0, 1.0);
                out_row[o + 1] = (img_slice[i + 1] * ratio).clamp(0.0, 1.0);
                out_row[o + 2] = (img_slice[i + 2] * ratio).clamp(0.0, 1.0);
            }
        });
        out
    });

    Array3::from_shape_vec((h, w, c), out_vec).unwrap().into_pyarray(py)
}

/// Contraste local (Claridad) en Rust con desenfoque Gaussiano O(1) y reconstrucción RGB fusionada.
#[pyfunction]
#[pyo3(signature = (img_rgb, strength=0.0, ref_min_dim=None))]
pub fn apply_clarity_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    strength: f32,
    ref_min_dim: Option<usize>,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = img_rgb.as_array();
    let (h, w, c) = img_view.dim();
    let img_slice = img_view.as_slice().expect("img_rgb must be C-contiguous float32");

    let out_vec = py.allow_threads(|| {
        let npix = h * w;
        let mut lum = vec![0.0f32; npix];
        lum.par_chunks_exact_mut(w).enumerate().for_each(|(y, row)| {
            let base = y * w * c;
            for x in 0..w {
                let i = base + x * c;
                row[x] = 0.2126 * img_slice[i] + 0.7152 * img_slice[i + 1] + 0.0722 * img_slice[i + 2];
            }
        });

        let base_dim = ref_min_dim.unwrap_or_else(|| h.min(w));
        let mut r = ((base_dim / 35) | 1).max(9);
        if r % 2 == 0 {
            r += 1;
        }
        let sigma = (r as f32) / 3.0;
        let lum_blur = fast_gaussian_blur_2d(&lum, h, w, sigma);

        let factor = strength * 0.85;
        let mut out = vec![0.0f32; npix * c];
        out.par_chunks_exact_mut(w * c).enumerate().for_each(|(y, out_row)| {
            let row_pix = y * w;
            let row_rgb = row_pix * c;
            for x in 0..w {
                let p = row_pix + x;
                let l = lum[p];
                let new_l = (l + (l - lum_blur[p]) * factor).clamp(1e-6, 1.0);
                let ratio = new_l / l.max(1e-6);
                let i = row_rgb + x * c;
                let o = x * c;
                out_row[o] = (img_slice[i] * ratio).clamp(0.0, 1.0);
                out_row[o + 1] = (img_slice[i + 1] * ratio).clamp(0.0, 1.0);
                out_row[o + 2] = (img_slice[i + 2] * ratio).clamp(0.0, 1.0);
            }
        });
        out
    });

    Array3::from_shape_vec((h, w, c), out_vec).unwrap().into_pyarray(py)
}

/// Borrar Neblina (Dehaze) en Rust con desenfoque O(1) del canal oscuro y compensación fusionada.
#[pyfunction]
#[pyo3(signature = (img_rgb, strength=0.0, ref_min_dim=None, global_airlight_p95=None))]
pub fn apply_dehaze_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    strength: f32,
    ref_min_dim: Option<usize>,
    global_airlight_p95: Option<f32>,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = img_rgb.as_array();
    let (h, w, c) = img_view.dim();
    let img_slice = img_view.as_slice().expect("img_rgb must be C-contiguous float32");

    let out_vec = py.allow_threads(|| {
        let npix = h * w;
        let mut dark = vec![0.0f32; npix];
        dark.par_chunks_exact_mut(w).enumerate().for_each(|(y, row)| {
            let base = y * w * c;
            for x in 0..w {
                let i = base + x * c;
                row[x] = img_slice[i].min(img_slice[i + 1]).min(img_slice[i + 2]);
            }
        });

        let base_dim = ref_min_dim.unwrap_or_else(|| h.min(w));
        let mut r = ((base_dim / 25) | 1).max(15);
        if r % 2 == 0 {
            r += 1;
        }
        let sigma = (r as f32) / 2.5;
        let haze_map = fast_gaussian_blur_2d(&dark, h, w, sigma);

        let mut out = vec![0.0f32; npix * c];
        if strength > 0.0 {
            let p95 = if let Some(gp) = global_airlight_p95 {
                gp
            } else {
                let mut sample = Vec::with_capacity((h / 4 + 1) * (w / 4 + 1));
                for y in (0..h).step_by(4) {
                    let row = &haze_map[y * w..(y + 1) * w];
                    for x in (0..w).step_by(4) {
                        sample.push(row[x]);
                    }
                }
                if sample.is_empty() {
                    0.5
                } else {
                    let k = (((sample.len() as f32) * 0.95).round() as usize).min(sample.len() - 1);
                    sample.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
                    sample[k]
                }
            };
            let airlight = p95 * (0.35 * strength).min(0.65);
            let s_k = strength * 0.45;
            let sat_boost = 1.0 + strength * 0.25;

            out.par_chunks_exact_mut(w * c).enumerate().for_each(|(y, out_row)| {
                let row_pix = y * w;
                let row_rgb = row_pix * c;
                for x in 0..w {
                    let p = row_pix + x;
                    let t = (1.0 - s_k * haze_map[p]).clamp(0.12, 1.0);
                    let inv_t = 1.0 / t;
                    let air_term = airlight * (1.0 - t);
                    let i = row_rgb + x * c;
                    let r0 = (img_slice[i] - air_term) * inv_t;
                    let g0 = (img_slice[i + 1] - air_term) * inv_t;
                    let b0 = (img_slice[i + 2] - air_term) * inv_t;
                    let l = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;
                    let o = x * c;
                    out_row[o] = (l + (r0 - l) * sat_boost).clamp(0.0, 1.0);
                    out_row[o + 1] = (l + (g0 - l) * sat_boost).clamp(0.0, 1.0);
                    out_row[o + 2] = (l + (b0 - l) * sat_boost).clamp(0.0, 1.0);
                }
            });
        } else {
            let factor = (strength.abs() * 0.35).min(0.85);
            let inv_f = 1.0 - factor;
            out.par_chunks_exact_mut(w * c).enumerate().for_each(|(y, out_row)| {
                let row_pix = y * w;
                let row_rgb = row_pix * c;
                for x in 0..w {
                    let h_val = haze_map[row_pix + x] * factor;
                    let i = row_rgb + x * c;
                    let o = x * c;
                    out_row[o] = (img_slice[i] * inv_f + h_val).clamp(0.0, 1.0);
                    out_row[o + 1] = (img_slice[i + 1] * inv_f + h_val).clamp(0.0, 1.0);
                    out_row[o + 2] = (img_slice[i + 2] * inv_f + h_val).clamp(0.0, 1.0);
                }
            });
        }
        out
    });

    Array3::from_shape_vec((h, w, c), out_vec).unwrap().into_pyarray(py)
}

/// Contraste sigmoidal suave con LUT de 4096 niveles (interpolación lineal) y mezcla opcional por máscara.
#[pyfunction]
#[pyo3(signature = (img_rgb, contrast=0.0, pivot=0.35, sky_mask=None))]
pub fn adjust_contrast_masked_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    contrast: f32,
    pivot: f32,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = img_rgb.as_array();
    let (h, w, c) = img_view.dim();
    let img_slice = img_view.as_slice().expect("img_rgb must be C-contiguous float32");
    let mask_view = sky_mask.as_ref().map(|m| m.as_array());
    let mask_slice_opt: Option<&[f32]> = mask_view.as_ref().and_then(|v| v.as_slice());
    let mask_owned: Option<Vec<f32>> = if mask_slice_opt.is_none() {
        mask_view.as_ref().map(|v| v.iter().copied().collect())
    } else {
        None
    };
    let mask_ref: Option<&[f32]> = mask_slice_opt.or(mask_owned.as_deref());

    let out_vec = py.allow_threads(|| {
        const LUT_N: usize = 4096;
        let piv = pivot.clamp(0.10, 0.70);
        let mut lut = [0.0f32; LUT_N];
        if contrast > 0.0 {
            let k = 1.0 + contrast * 9.0;
            let sig_0 = 1.0 / (1.0 + (k * piv).exp());
            let sig_1 = 1.0 / (1.0 + (-k * (1.0 - piv)).exp());
            let inv_span = 1.0 / (sig_1 - sig_0).max(1e-7);
            let inv_c = 1.0 - contrast;
            for i in 0..LUT_N {
                let x = (i as f32) / ((LUT_N - 1) as f32);
                let sig = 1.0 / (1.0 + (-k * (x - piv)).exp());
                let norm_s = (sig - sig_0) * inv_span;
                lut[i] = (inv_c * x + contrast * norm_s).clamp(0.0, 1.0);
            }
        } else {
            let factor = 1.0 + contrast * 0.5;
            for i in 0..LUT_N {
                let x = (i as f32) / ((LUT_N - 1) as f32);
                lut[i] = (piv + (x - piv) * factor).clamp(0.0, 1.0);
            }
        }

        let scale = (LUT_N - 1) as f32;
        let max_idx = LUT_N - 2;
        #[inline(always)]
        fn eval_lut(lut: &[f32; LUT_N], v: f32, scale: f32, max_idx: usize) -> f32 {
            let f = v.clamp(0.0, 1.0) * scale;
            let idx = (f as usize).min(max_idx);
            let frac = f - (idx as f32);
            lut[idx] + (lut[idx + 1] - lut[idx]) * frac
        }

        let mut out = vec![0.0f32; h * w * c];
        out.par_chunks_exact_mut(w * c).enumerate().for_each(|(y, out_row)| {
            let row_pix = y * w;
            let row_rgb = row_pix * c;
            for x in 0..w {
                let i = row_rgb + x * c;
                let o = x * c;
                let r0 = img_slice[i];
                let g0 = img_slice[i + 1];
                let b0 = img_slice[i + 2];
                let r1 = eval_lut(&lut, r0, scale, max_idx);
                let g1 = eval_lut(&lut, g0, scale, max_idx);
                let b1 = eval_lut(&lut, b0, scale, max_idx);
                if let Some(m) = mask_ref {
                    let mw = m[row_pix + x].clamp(0.0, 1.0);
                    let inv_mw = 1.0 - mw;
                    out_row[o] = r0 * inv_mw + r1 * mw;
                    out_row[o + 1] = g0 * inv_mw + g1 * mw;
                    out_row[o + 2] = b0 * inv_mw + b1 * mw;
                } else {
                    out_row[o] = r1;
                    out_row[o + 1] = g1;
                    out_row[o + 2] = b1;
                }
            }
        });
        out
    });

    Array3::from_shape_vec((h, w, c), out_vec).unwrap().into_pyarray(py)
}

/// Filtro auto-guiado O(1) 100% paralelo con Rayon usando box_blur_h/box_blur_v (reemplaza SAT secuencial).
#[pyfunction]
#[pyo3(signature = (img_2d, radius=4, eps=0.01))]
pub fn self_guided_filter_rs<'py>(
    py: Python<'py>,
    img_2d: PyReadonlyArray2<'py, f32>,
    radius: usize,
    eps: f32,
) -> Bound<'py, numpy::PyArray2<f32>> {
    let view = img_2d.as_array();
    let (h, w) = view.dim();
    let src_opt = view.as_slice();
    let src_owned: Option<Vec<f32>> = if src_opt.is_none() {
        Some(view.iter().copied().collect())
    } else {
        None
    };
    let src: &[f32] = src_opt.unwrap_or_else(|| src_owned.as_deref().unwrap());

    let out_vec = py.allow_threads(|| {
        let npix = h * w;
        let r = radius.max(1);
        let epsilon = eps.max(1e-8);
        let mut tmp = vec![0.0f32; npix];

        // 1. mean_I = box(I)
        let mut mean_i = vec![0.0f32; npix];
        box_blur_h(src, &mut tmp, h, w, r);
        box_blur_v(&tmp, &mut mean_i, h, w, r);

        // 2. mean_II = box(I * I)
        let mut ii = vec![0.0f32; npix];
        ii.par_iter_mut().zip(src.par_iter()).for_each(|(d, &v)| {
            *d = v * v;
        });
        let mut mean_ii = vec![0.0f32; npix];
        box_blur_h(&ii, &mut tmp, h, w, r);
        box_blur_v(&tmp, &mut mean_ii, h, w, r);

        // 3. a = var_I / (var_I + eps), b = mean_I * (1 - a)
        let mut a = ii; // reuse buffer
        let mut b = mean_ii; // reuse buffer
        a.par_iter_mut()
            .zip(b.par_iter_mut())
            .zip(mean_i.par_iter())
            .for_each(|((a_out, b_val), &mi)| {
                let var = (*b_val - mi * mi).max(0.0);
                let av = var / (var + epsilon);
                *a_out = av;
                *b_val = mi * (1.0 - av);
            });

        // 4. mean_a = box(a), mean_b = box(b)
        let mut mean_a = mean_i; // reuse buffer
        box_blur_h(&a, &mut tmp, h, w, r);
        box_blur_v(&tmp, &mut mean_a, h, w, r);

        let mut mean_b = a; // reuse buffer
        box_blur_h(&b, &mut tmp, h, w, r);
        box_blur_v(&tmp, &mut mean_b, h, w, r);

        // 5. q = mean_a * I + mean_b
        let mut out = b; // reuse buffer
        out.par_iter_mut()
            .zip(src.par_iter())
            .zip(mean_a.par_iter())
            .zip(mean_b.par_iter())
            .for_each(|(((q, &v), &ma), &mb)| {
                *q = (ma * v + mb).clamp(0.0, 1.0);
            });
        out
    });

    numpy::ndarray::Array2::from_shape_vec((h, w), out_vec)
        .unwrap()
        .into_pyarray(py)
}
