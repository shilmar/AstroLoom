use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// MOTOR RBF Y NEUTRALIZACIÓN DE FONDO (core/background_extraction.py)
// ============================================================================

#[inline(always)]
pub fn rbf_basis(r: f64, kernel: &str, epsilon: f64) -> f64 {
    let er = epsilon * r;
    match kernel {
        "multiquadric" => -(1.0 + er * er).sqrt(),
        "inverse_multiquadric" => 1.0 / (1.0 + er * er).sqrt(),
        "gaussian" => (-er * er).exp(),
        "thin_plate_spline" => {
            if r < 1e-12 {
                0.0
            } else {
                r * r * r.ln()
            }
        }
        "cubic" => r * r * r,
        _ => -r, // "linear"
    }
}

#[pyfunction]
#[pyo3(signature = (pts, vals, eval_h, eval_w, kernel="multiquadric", smoothing=0.15, epsilon=1.0))]
pub fn fit_and_eval_rbf_proxy<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f64>,
    vals: PyReadonlyArray2<'py, f64>,
    eval_h: usize,
    eval_w: usize,
    kernel: &str,
    smoothing: f64,
    epsilon: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let pts_view = pts.as_array();
    let vals_view = vals.as_array();
    let n = pts_view.nrows();

    let proxy_vec = py.allow_threads(|| {
        let sys_size = n + 3;
        let mut mat = DMatrix::<f64>::zeros(sys_size, sys_size);

        for i in 0..n {
            let xi = pts_view[[i, 0]];
            let yi = pts_view[[i, 1]];
            for j in i..n {
                let dx = xi - pts_view[[j, 0]];
                let dy = yi - pts_view[[j, 1]];
                let r = (dx * dx + dy * dy).sqrt();
                let mut val = rbf_basis(r, kernel, epsilon);
                if i == j {
                    val -= smoothing;
                }
                mat[(i, j)] = val;
                mat[(j, i)] = val;
            }
            mat[(i, n)] = 1.0;
            mat[(i, n + 1)] = xi;
            mat[(i, n + 2)] = yi;

            mat[(n, i)] = 1.0;
            mat[(n + 1, i)] = xi;
            mat[(n + 2, i)] = yi;
        }

        let lu = mat.lu();
        let mut weights = Array2::<f64>::zeros((sys_size, 3));

        for ch in 0..3 {
            let mut rhs = DVector::<f64>::zeros(sys_size);
            for i in 0..n {
                rhs[i] = vals_view[[i, ch]];
            }
            if let Some(sol) = lu.solve(&rhs) {
                for i in 0..sys_size {
                    weights[[i, ch]] = sol[i];
                }
            }
        }

        let mut min_val = [f32::INFINITY; 3];
        let mut max_val = [f32::NEG_INFINITY; 3];
        for ch in 0..3 {
            for i in 0..n {
                let v = vals_view[[i, ch]] as f32;
                if v < min_val[ch] {
                    min_val[ch] = v;
                }
                if v > max_val[ch] {
                    max_val[ch] = v;
                }
            }
            min_val[ch] = (min_val[ch] * 0.90).max(0.0);
            max_val[ch] *= 1.05;
        }

        let row_stride = eval_w * 3;
        let mut buf = vec![0.0f32; eval_h * row_stride];
        let inv_h = if eval_h > 1 { 1.0 / ((eval_h - 1) as f64) } else { 0.0 };
        let inv_w = if eval_w > 1 { 1.0 / ((eval_w - 1) as f64) } else { 0.0 };

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(iy, row_slice)| {
                let gy = (iy as f64) * inv_h;
                for ix in 0..eval_w {
                    let gx = (ix as f64) * inv_w;
                    let mut sum_rgb = [0.0f64; 3];

                    for i in 0..n {
                        let dx = gx - pts_view[[i, 0]];
                        let dy = gy - pts_view[[i, 1]];
                        let r = (dx * dx + dy * dy).sqrt();
                        let phi = rbf_basis(r, kernel, epsilon);
                        sum_rgb[0] += phi * weights[[i, 0]];
                        sum_rgb[1] += phi * weights[[i, 1]];
                        sum_rgb[2] += phi * weights[[i, 2]];
                    }

                    let base_idx = ix * 3;
                    for ch in 0..3 {
                        let poly = weights[[n, ch]]
                            + gx * weights[[n + 1, ch]]
                            + gy * weights[[n + 2, ch]];
                        let v = (sum_rgb[ch] + poly) as f32;
                        row_slice[base_idx + ch] = v.clamp(min_val[ch], max_val[ch]);
                    }
                }
            });

        buf
    });

    let out_array = Array3::from_shape_vec((eval_h, eval_w, 3), proxy_vec).unwrap();
    Ok(out_array.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (img_rgb, bg_full, pedestal, is_division, sky_mask=None))]
pub fn apply_bg_model_fused<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    bg_full: PyReadonlyArray3<'py, f32>,
    pedestal: PyReadonlyArray1<'py, f32>,
    is_division: bool,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
) -> Bound<'py, PyArray3<f32>> {
    let img = img_rgb.as_array();
    let bg = bg_full.as_array();
    let ped = pedestal.as_slice().unwrap();
    let mask_opt = sky_mask.as_ref().map(|m| m.as_array());

    let (h, w, c) = img.dim();
    let ped_rgb = [ped[0], ped[1], ped[2]];

    let out_vec = py.allow_threads(|| {
        let row_stride = w * c;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let m_val = if let Some(ref m) = mask_opt {
                        unsafe { (*m.uget((y, x))).clamp(0.0, 1.0) }
                    } else {
                        1.0
                    };

                    let base_idx = x * 3;
                    for ch in 0..3 {
                        let orig = unsafe { *img.uget((y, x, ch)) };
                        let bg_val = unsafe { *bg.uget((y, x, ch)) };
                        let p_val = ped_rgb[ch];

                        let corr = if is_division {
                            (orig / bg_val.max(1e-4)) * p_val
                        } else {
                            orig - bg_val + p_val
                        };

                        let blended = if mask_opt.is_some() {
                            corr * m_val + orig * (1.0 - m_val)
                        } else {
                            corr
                        };

                        row_slice[base_idx + ch] = blended.clamp(0.0, 1.0);
                    }
                }
            });

        buf
    });

    let out_array = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_array.into_pyarray(py)
}
