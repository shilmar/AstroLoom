use nalgebra::{DMatrix, DVector};
use numpy::ndarray::{Array1, Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Tamaño máximo de sub-tomas en pila (Stack L1) sin asignar memoria en Heap.
pub const STACK_BUF_CAP: usize = 128;

#[inline(always)]
pub fn compute_median_in_place(slice: &mut [f32]) -> f32 {
    let n = slice.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    let (_, &mut med_hi, _) = slice.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        med_hi
    } else {
        let max_lo = slice[..mid]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        0.5 * (max_lo + med_hi)
    }
}

// ============================================================================
// KERNELS DE APILADO (core/stacking.py)
// ============================================================================

#[pyfunction]
pub fn kappa_sigma_kernel<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    kappa: f32,
    weights: PyReadonlyArray1<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    let data_view: ArrayView2<f32> = data.as_array();
    let weights_slice = weights.as_slice().unwrap();
    let (n_frames, total_items) = data_view.dim();

    let out_vec: Vec<f32> = py.allow_threads(|| {
        let mut result = vec![0.0f32; total_items];

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(p, out_pixel)| {
                let mut stack_v = [0.0f32; STACK_BUF_CAP];
                let mut stack_d = [0.0f32; STACK_BUF_CAP];
                let mut heap_v: Vec<f32>;
                let mut heap_d: Vec<f32>;

                let (v_buf, d_buf): (&mut [f32], &mut [f32]) = if n_frames <= STACK_BUF_CAP {
                    (&mut stack_v[..n_frames], &mut stack_d[..n_frames])
                } else {
                    heap_v = vec![0.0f32; n_frames];
                    heap_d = vec![0.0f32; n_frames];
                    (&mut heap_v[..], &mut heap_d[..])
                };

                let mut cnt_val = 0usize;
                for i in 0..n_frames {
                    let val = unsafe { *data_view.uget((i, p)) };
                    if val > -0.5 {
                        v_buf[cnt_val] = val;
                        cnt_val += 1;
                    }
                }

                if cnt_val == 0 {
                    *out_pixel = 0.0;
                    return;
                }

                let valid_v = &mut v_buf[..cnt_val];
                let med = compute_median_in_place(valid_v);

                let valid_d = &mut d_buf[..cnt_val];
                for i in 0..cnt_val {
                    valid_d[i] = (valid_v[i] - med).abs();
                }

                let mad = compute_median_in_place(valid_d);
                let sigma = 1.4826 * mad + 1e-6;
                let low = med - kappa * sigma;
                let high = med + kappa * sigma;

                let mut s = 0.0f32;
                let mut w_sum = 0.0f32;
                for i in 0..n_frames {
                    let val = unsafe { *data_view.uget((i, p)) };
                    if val > -0.5 && val >= low && val <= high {
                        let w = unsafe { *weights_slice.get_unchecked(i) };
                        s += val * w;
                        w_sum += w;
                    }
                }

                *out_pixel = if w_sum > 0.0 { s / w_sum } else { med };
            });

        result
    });

    Array1::from_vec(out_vec).into_pyarray(py)
}

#[pyfunction]
pub fn percentile_kernel<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    target_p: f32,
) -> Bound<'py, PyArray1<f32>> {
    let data_view = data.as_array();
    let (n_frames, total_items) = data_view.dim();

    let out_vec: Vec<f32> = py.allow_threads(|| {
        let mut result = vec![0.0f32; total_items];

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(p, out_pixel)| {
                let mut stack_v = [0.0f32; STACK_BUF_CAP];
                let mut heap_v: Vec<f32>;
                let v_buf: &mut [f32] = if n_frames <= STACK_BUF_CAP {
                    &mut stack_v[..n_frames]
                } else {
                    heap_v = vec![0.0f32; n_frames];
                    &mut heap_v[..]
                };

                let mut cnt_val = 0usize;
                for i in 0..n_frames {
                    let val = unsafe { *data_view.uget((i, p)) };
                    if val > -0.5 {
                        v_buf[cnt_val] = val;
                        cnt_val += 1;
                    }
                }

                if cnt_val == 0 {
                    *out_pixel = 0.0;
                    return;
                }

                let valid_v = &mut v_buf[..cnt_val];
                valid_v.sort_unstable_by(|a, b| a.total_cmp(b));

                let rank = (target_p / 100.0) * ((cnt_val - 1) as f32);
                let i0 = rank as usize;
                let i1 = (i0 + 1).min(cnt_val - 1);
                let frac = rank - (i0 as f32);
                *out_pixel = valid_v[i0] + frac * (valid_v[i1] - valid_v[i0]);
            });

        result
    });

    Array1::from_vec(out_vec).into_pyarray(py)
}

#[pyfunction]
pub fn median_kernel<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
) -> Bound<'py, PyArray1<f32>> {
    let data_view = data.as_array();
    let (n_frames, total_items) = data_view.dim();

    let out_vec: Vec<f32> = py.allow_threads(|| {
        let mut result = vec![0.0f32; total_items];

        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(p, out_pixel)| {
                let mut stack_v = [0.0f32; STACK_BUF_CAP];
                let mut heap_v: Vec<f32>;
                let v_buf: &mut [f32] = if n_frames <= STACK_BUF_CAP {
                    &mut stack_v[..n_frames]
                } else {
                    heap_v = vec![0.0f32; n_frames];
                    &mut heap_v[..]
                };

                let mut cnt_val = 0usize;
                for i in 0..n_frames {
                    let val = unsafe { *data_view.uget((i, p)) };
                    if val > -0.5 {
                        v_buf[cnt_val] = val;
                        cnt_val += 1;
                    }
                }

                if cnt_val == 0 {
                    *out_pixel = 0.0;
                    return;
                }

                *out_pixel = compute_median_in_place(&mut v_buf[..cnt_val]);
            });

        result
    });

    Array1::from_vec(out_vec).into_pyarray(py)
}

// ============================================================================
// DEFORMACIÓN DE MALLA TPS (THIN-PLATE SPLINE) (core/stacking.py)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (bin_centers, bin_residuals, grid_w, grid_h, w, h, smoothing=2.0))]
pub fn compute_tps_grid_rs<'py>(
    py: Python<'py>,
    bin_centers: PyReadonlyArray2<'py, f32>,
    bin_residuals: PyReadonlyArray2<'py, f32>,
    grid_w: usize,
    grid_h: usize,
    w: f64,
    h: f64,
    smoothing: f64,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let pts_view = bin_centers.as_array();
    let res_view = bin_residuals.as_array();
    let n = pts_view.nrows();

    if n < 3 {
        let empty = Array3::<f32>::zeros((grid_h, grid_w, 2));
        return Ok(empty.into_pyarray(py));
    }

    let out_vec: Vec<f32> = py.allow_threads(|| {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for i in 0..n {
            let x = pts_view[[i, 0]] as f64;
            let y = pts_view[[i, 1]] as f64;
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        }

        let shift_x = (max_x + min_x) * 0.5;
        let shift_y = (max_y + min_y) * 0.5;
        let mut scale_x = (max_x - min_x) * 0.5;
        let mut scale_y = (max_y - min_y) * 0.5;
        if scale_x < 1e-12 { scale_x = 1.0; }
        if scale_y < 1e-12 { scale_y = 1.0; }

        let sys_size = n + 3;
        let mut mat = DMatrix::<f64>::zeros(sys_size, sys_size);

        for i in 0..n {
            let xi = pts_view[[i, 0]] as f64;
            let yi = pts_view[[i, 1]] as f64;
            for j in i..n {
                let dx = xi - pts_view[[j, 0]] as f64;
                let dy = yi - pts_view[[j, 1]] as f64;
                let r_sq = dx * dx + dy * dy;
                let mut val = if r_sq > 1e-24 {
                    0.5 * r_sq * r_sq.ln()
                } else {
                    0.0
                };
                if i == j {
                    val += smoothing;
                }
                mat[(i, j)] = val;
                mat[(j, i)] = val;
            }
            let norm_x = (xi - shift_x) / scale_x;
            let norm_y = (yi - shift_y) / scale_y;
            mat[(i, n)] = 1.0;
            mat[(i, n + 1)] = norm_x;
            mat[(i, n + 2)] = norm_y;

            mat[(n, i)] = 1.0;
            mat[(n + 1, i)] = norm_x;
            mat[(n + 2, i)] = norm_y;
        }

        let lu = mat.full_piv_lu();
        let mut weights_x = vec![0.0f64; n];
        let mut weights_y = vec![0.0f64; n];
        let mut poly_x = [0.0f64; 3];
        let mut poly_y = [0.0f64; 3];

        let mut rhs_x = DVector::<f64>::zeros(sys_size);
        let mut rhs_y = DVector::<f64>::zeros(sys_size);
        for i in 0..n {
            rhs_x[i] = res_view[[i, 0]] as f64;
            rhs_y[i] = res_view[[i, 1]] as f64;
        }

        if let Some(sol_x) = lu.solve(&rhs_x) {
            for i in 0..n {
                weights_x[i] = sol_x[i];
            }
            poly_x[0] = sol_x[n];
            poly_x[1] = sol_x[n + 1];
            poly_x[2] = sol_x[n + 2];
        }

        if let Some(sol_y) = lu.solve(&rhs_y) {
            for i in 0..n {
                weights_y[i] = sol_y[i];
            }
            poly_y[0] = sol_y[n];
            poly_y[1] = sol_y[n + 1];
            poly_y[2] = sol_y[n + 2];
        }

        let pts_x: Vec<f64> = (0..n).map(|i| pts_view[[i, 0]] as f64).collect();
        let pts_y: Vec<f64> = (0..n).map(|i| pts_view[[i, 1]] as f64).collect();

        let inv_gw = if grid_w > 1 { 1.0 / ((grid_w - 1) as f64) } else { 0.0 };
        let inv_gh = if grid_h > 1 { 1.0 / ((grid_h - 1) as f64) } else { 0.0 };
        let row_stride = grid_w * 2;
        let mut buf = vec![0.0f32; grid_h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(iy, row)| {
                let gy = (iy as f64) * inv_gh * (h - 1.0);
                let norm_y = (gy - shift_y) / scale_y;
                for ix in 0..grid_w {
                    let gx = (ix as f64) * inv_gw * (w - 1.0);
                    let norm_x = (gx - shift_x) / scale_x;

                    let mut val_x = poly_x[0] + poly_x[1] * norm_x + poly_x[2] * norm_y;
                    let mut val_y = poly_y[0] + poly_y[1] * norm_x + poly_y[2] * norm_y;

                    for i in 0..n {
                        let dx = gx - pts_x[i];
                        let dy = gy - pts_y[i];
                        let r_sq = dx * dx + dy * dy;
                        if r_sq > 1e-24 {
                            let phi = 0.5 * r_sq * r_sq.ln();
                            val_x += phi * weights_x[i];
                            val_y += phi * weights_y[i];
                        }
                    }

                    row[ix * 2] = val_x as f32;
                    row[ix * 2 + 1] = val_y as f32;
                }
            });

        buf
    });

    let out_array = Array3::from_shape_vec((grid_h, grid_w, 2), out_vec).unwrap();
    Ok(out_array.into_pyarray(py))
}
