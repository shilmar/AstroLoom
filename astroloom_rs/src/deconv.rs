use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// DECONVOLUCIÓN ÓPTICA REGULARIZADA Y MODELADO PSF (core/deconvolution.py)
// ============================================================================

#[inline(always)]
fn reflect_index(mut idx: isize, max_len: usize) -> usize {
    let m = max_len as isize;
    if m <= 1 {
        return 0;
    }
    while idx < 0 || idx >= m {
        if idx < 0 {
            idx = -idx - 1;
        } else if idx >= m {
            idx = 2 * m - 1 - idx;
        }
    }
    idx as usize
}

fn convolve2d_reflect(
    src: &[f32],
    dst: &mut [f32],
    h: usize,
    w: usize,
    kernel: &[f32],
    ksize: usize,
) {
    let rad = (ksize / 2) as isize;
    let interior_y_start = (rad as usize).min(h);
    let interior_y_end = if h >= (rad as usize) { h - (rad as usize) } else { 0 };
    let interior_x_start = (rad as usize).min(w);
    let interior_x_end = if w >= (rad as usize) { w - (rad as usize) } else { 0 };

    dst.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row_dst)| {
            if y >= interior_y_start && y < interior_y_end {
                // Fila interior
                // 1. Frontera izquierda
                for x in 0..interior_x_start {
                    let mut sum = 0.0f32;
                    let mut k_idx = 0;
                    for ky in -rad..=rad {
                        let ry = (y as isize + ky) as usize;
                        let row_offset = ry * w;
                        for kx in -rad..=rad {
                            let rx = reflect_index(x as isize + kx, w);
                            sum += src[row_offset + rx] * kernel[k_idx];
                            k_idx += 1;
                        }
                    }
                    row_dst[x] = sum;
                }

                // 2. Interior vectorizado SIMD (SAXPY / FMA directo)
                if interior_x_end > interior_x_start {
                    let dst_interior = &mut row_dst[interior_x_start..interior_x_end];
                    dst_interior.fill(0.0);

                    for ky in -rad..=rad {
                        let src_row = ((y as isize + ky) as usize) * w;
                        let k_row = ((ky + rad) as usize) * ksize;
                        for kx in -rad..=rad {
                            let k_val = unsafe { *kernel.get_unchecked(k_row + (kx + rad) as usize) };
                            let start_src = (interior_x_start as isize + kx) as usize;
                            let src_slice = unsafe {
                                std::slice::from_raw_parts(src.as_ptr().add(src_row + start_src), dst_interior.len())
                            };
                            for (d, &s) in dst_interior.iter_mut().zip(src_slice.iter()) {
                                *d += s * k_val;
                            }
                        }
                    }
                }

                // 3. Frontera derecha
                for x in interior_x_end..w {
                    let mut sum = 0.0f32;
                    let mut k_idx = 0;
                    for ky in -rad..=rad {
                        let ry = (y as isize + ky) as usize;
                        let row_offset = ry * w;
                        for kx in -rad..=rad {
                            let rx = reflect_index(x as isize + kx, w);
                            sum += src[row_offset + rx] * kernel[k_idx];
                            k_idx += 1;
                        }
                    }
                    row_dst[x] = sum;
                }
            } else {
                // Fila completa de frontera superior / inferior
                for x in 0..w {
                    let mut sum = 0.0f32;
                    let mut k_idx = 0;
                    for ky in -rad..=rad {
                        let ry = reflect_index(y as isize + ky, h);
                        let row_offset = ry * w;
                        for kx in -rad..=rad {
                            let rx = reflect_index(x as isize + kx, w);
                            sum += src[row_offset + rx] * kernel[k_idx];
                            k_idx += 1;
                        }
                    }
                    row_dst[x] = sum;
                }
            }
        });
}

#[pyfunction]
#[pyo3(signature = (fwhm=3.0, beta=4.5, aspect_ratio=1.0, angle_deg=0.0, kernel_size=None))]
pub fn generate_moffat_psf_rs<'py>(
    py: Python<'py>,
    fwhm: f32,
    beta: f32,
    aspect_ratio: f32,
    angle_deg: f32,
    kernel_size: Option<usize>,
) -> Bound<'py, PyArray2<f32>> {
    let fwhm_d = (fwhm as f64).max(1.0);
    let beta_d = (beta as f64).max(1.1);
    let aspect_d = (aspect_ratio as f64).clamp(1.0, 3.0);

    let alpha = fwhm_d / (2.0 * (2.0f64.powf(1.0 / beta_d) - 1.0).sqrt());

    let rad = match kernel_size {
        Some(ks) => {
            let k = if ks % 2 == 1 { ks } else { ks + 1 };
            (k / 2) as isize
        }
        None => {
            let r = (fwhm_d * 3.5).ceil() as isize;
            r.clamp(7, 25)
        }
    };
    let ksize = (2 * rad + 1) as usize;

    let (cos_t, sin_t) = if angle_deg != 0.0 {
        let theta = (angle_deg as f64).to_radians();
        (theta.cos(), theta.sin())
    } else {
        (1.0, 0.0)
    };

    let ax = alpha * aspect_d.sqrt();
    let ay = alpha / aspect_d.sqrt();

    let mut buf = vec![0.0f32; ksize * ksize];
    let mut sum = 0.0f64;

    for y_idx in 0..ksize {
        let y = (y_idx as isize - rad) as f64;
        for x_idx in 0..ksize {
            let x = (x_idx as isize - rad) as f64;
            let (xr, yr) = if angle_deg != 0.0 {
                (x * cos_t + y * sin_t, -x * sin_t + y * cos_t)
            } else {
                (x, y)
            };
            let r_sq = (xr / ax).powi(2) + (yr / ay).powi(2);
            let val = (1.0 + r_sq).powf(-beta_d);
            buf[y_idx * ksize + x_idx] = val as f32;
            sum += val;
        }
    }

    if sum > 0.0 {
        let inv_sum = (1.0 / sum) as f32;
        for v in buf.iter_mut() {
            *v *= inv_sum;
        }
    } else {
        buf.fill(0.0);
        let center = (rad as usize) * ksize + (rad as usize);
        buf[center] = 1.0;
    }

    let arr = Array2::from_shape_vec((ksize, ksize), buf).unwrap();
    arr.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (fwhm=3.0, aspect_ratio=1.0, angle_deg=0.0, kernel_size=None))]
pub fn generate_gaussian_psf_rs<'py>(
    py: Python<'py>,
    fwhm: f32,
    aspect_ratio: f32,
    angle_deg: f32,
    kernel_size: Option<usize>,
) -> Bound<'py, PyArray2<f32>> {
    let fwhm_d = (fwhm as f64).max(1.0);
    let sigma = fwhm_d / (2.0 * (2.0 * 2.0f64.ln()).sqrt());
    let aspect_d = (aspect_ratio as f64).clamp(1.0, 3.0);

    let rad = match kernel_size {
        Some(ks) => {
            let k = if ks % 2 == 1 { ks } else { ks + 1 };
            (k / 2) as isize
        }
        None => {
            let r = (sigma * 3.5).ceil() as isize;
            r.clamp(7, 25)
        }
    };
    let ksize = (2 * rad + 1) as usize;

    let (cos_t, sin_t) = if angle_deg != 0.0 {
        let theta = (angle_deg as f64).to_radians();
        (theta.cos(), theta.sin())
    } else {
        (1.0, 0.0)
    };

    let sx = sigma * aspect_d.sqrt();
    let sy = sigma / aspect_d.sqrt();

    let mut buf = vec![0.0f32; ksize * ksize];
    let mut sum = 0.0f64;

    for y_idx in 0..ksize {
        let y = (y_idx as isize - rad) as f64;
        for x_idx in 0..ksize {
            let x = (x_idx as isize - rad) as f64;
            let (xr, yr) = if angle_deg != 0.0 {
                (x * cos_t + y * sin_t, -x * sin_t + y * cos_t)
            } else {
                (x, y)
            };
            let val = (-0.5 * ((xr / sx).powi(2) + (yr / sy).powi(2))).exp();
            buf[y_idx * ksize + x_idx] = val as f32;
            sum += val;
        }
    }

    if sum > 0.0 {
        let inv_sum = (1.0 / sum) as f32;
        for v in buf.iter_mut() {
            *v *= inv_sum;
        }
    } else {
        buf.fill(0.0);
        let center = (rad as usize) * ksize + (rad as usize);
        buf[center] = 1.0;
    }

    let arr = Array2::from_shape_vec((ksize, ksize), buf).unwrap();
    arr.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (
    work_channel,
    psf_kernel,
    iterations=15,
    noise_threshold_sq=0.0,
    bright_star_mask=None,
    deringing_floor=None,
    est_initial=None
))]
pub fn richardson_lucy_deconv_rs<'py>(
    py: Python<'py>,
    work_channel: PyReadonlyArray2<'py, f32>,
    psf_kernel: PyReadonlyArray2<'py, f32>,
    iterations: usize,
    noise_threshold_sq: f32,
    bright_star_mask: Option<PyReadonlyArray2<'py, f32>>,
    deringing_floor: Option<PyReadonlyArray2<'py, f32>>,
    est_initial: Option<PyReadonlyArray2<'py, f32>>,
) -> Bound<'py, PyArray2<f32>> {
    let work_view = work_channel.as_array();
    let (h, w) = work_view.dim();
    let total_pixels = h * w;

    let psf_view = psf_kernel.as_array();
    let (kh, kw) = psf_view.dim();
    assert_eq!(kh, kw, "PSF kernel must be square");
    let ksize = kh;

    let mut psf_flat = Vec::with_capacity(ksize * ksize);
    for y in 0..ksize {
        for x in 0..ksize {
            psf_flat.push(unsafe { *psf_view.uget((y, x)) });
        }
    }
    let mut psf_flipped = psf_flat.clone();
    psf_flipped.reverse();

    let star_mask_flat = bright_star_mask.as_ref().map(|sm| {
        let v = sm.as_array();
        let mut buf = vec![0.0f32; total_pixels];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = unsafe { *v.uget((y, x)) };
            }
        }
        buf
    });

    let floor_flat = deringing_floor.as_ref().map(|df| {
        let v = df.as_array();
        let mut buf = vec![0.0f32; total_pixels];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = unsafe { *v.uget((y, x)) };
            }
        }
        buf
    });

    let mut work_flat = vec![0.0f32; total_pixels];
    for y in 0..h {
        for x in 0..w {
            work_flat[y * w + x] = unsafe { *work_view.uget((y, x)) };
        }
    }

    let initial_est = if let Some(ref init) = est_initial {
        let iv = init.as_array();
        let mut buf = vec![0.0f32; total_pixels];
        for y in 0..h {
            for x in 0..w {
                buf[y * w + x] = unsafe { *iv.uget((y, x)) };
            }
        }
        buf
    } else {
        work_flat.clone()
    };

    let out_vec = py.allow_threads(|| {
        let mut est_buf = initial_est;
        let mut conv_buf = vec![0.0f32; total_pixels];
        let mut b_proj_buf = vec![0.0f32; total_pixels];

        let has_deringing = star_mask_flat.is_some() && floor_flat.is_some();
        let sm_slice = star_mask_flat.as_deref();
        let df_slice = floor_flat.as_deref();

        for _it in 0..iterations {
            // 1. Convolución de est_buf con la PSF -> conv_buf
            convolve2d_reflect(&est_buf, &mut conv_buf, h, w, &psf_flat, ksize);

            // 2. Fusión de ratio: conv_buf = work / (conv + 1e-6)
            conv_buf
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, row_conv)| {
                    let row_offset = y * w;
                    for x in 0..w {
                        let w_val = unsafe { *work_flat.get_unchecked(row_offset + x) };
                        let c_val = row_conv[x];
                        row_conv[x] = w_val / (c_val + 1e-6);
                    }
                });

            // 3. Convolución del ratio con la PSF invertida -> b_proj_buf
            convolve2d_reflect(&conv_buf, &mut b_proj_buf, h, w, &psf_flipped, ksize);

            // 4. Actualización fusionada: amortiguación bayesiana + clamping + deringing
            est_buf
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, row_est)| {
                    let row_offset = y * w;
                    for x in 0..w {
                        let idx = row_offset + x;
                        let b_proj = unsafe { *b_proj_buf.get_unchecked(idx) };
                        let delta = b_proj - 1.0;

                        let update_factor = if noise_threshold_sq > 0.0 {
                            let delta_sq = delta * delta;
                            let damp_weight = delta_sq / (delta_sq + noise_threshold_sq + 1e-7);
                            1.0 + delta * damp_weight
                        } else {
                            b_proj
                        };

                        let mut new_est = (row_est[x] * update_factor).clamp(0.0, 1.0);

                        if has_deringing {
                            let mask_val = unsafe { *sm_slice.unwrap().get_unchecked(idx) };
                            if mask_val > 0.1 {
                                let floor_val = unsafe { *df_slice.unwrap().get_unchecked(idx) };
                                if new_est < floor_val {
                                    new_est = floor_val;
                                }
                            }
                        }

                        row_est[x] = new_est;
                    }
                });
        }

        est_buf
    });

    let arr = Array2::from_shape_vec((h, w), out_vec).unwrap();
    arr.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (
    image_rgb,
    est_luma,
    work_luma,
    star_mask=None,
    target_scope="all"
))]
pub fn reconstruct_deconv_color_rs<'py>(
    py: Python<'py>,
    image_rgb: PyReadonlyArray3<'py, f32>,
    est_luma: PyReadonlyArray2<'py, f32>,
    work_luma: PyReadonlyArray2<'py, f32>,
    star_mask: Option<PyReadonlyArray2<'py, f32>>,
    target_scope: &str,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = image_rgb.as_array();
    let est_view = est_luma.as_array();
    let work_view = work_luma.as_array();
    let mask_view = star_mask.as_ref().map(|sm| sm.as_array());
    let (h, w, c) = img_view.dim();

    let scope_mode = match target_scope {
        "stars_only" => 1,
        "background_only" => 2,
        _ => 0,
    };

    let out_vec = py.allow_threads(|| {
        let row_stride = w * c;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let base_idx = x * c;
                    let est_val = unsafe { *est_view.uget((y, x)) };
                    let orig_lum = unsafe { *work_view.uget((y, x)) };

                    let final_lum = if scope_mode > 0 && mask_view.is_some() {
                        let sm = unsafe { *mask_view.as_ref().unwrap().uget((y, x)) };
                        if scope_mode == 1 {
                            est_val * sm + orig_lum * (1.0 - sm)
                        } else {
                            est_val * (1.0 - sm) + orig_lum * sm
                        }
                    } else {
                        est_val
                    };

                    let ratio = final_lum / (orig_lum + 1e-6);

                    for ch in 0..c {
                        let orig_val = unsafe { *img_view.uget((y, x, ch)) };
                        row_slice[base_idx + ch] = (orig_val * ratio).clamp(0.0, 1.0);
                    }
                }
            });

        buf
    });

    let arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    arr.into_pyarray(py)
}
