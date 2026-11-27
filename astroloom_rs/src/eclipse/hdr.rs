use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::median_f32;

#[inline(always)]
fn soft_shoulder_val(peak: f32, threshold: f32, ceiling: f32) -> f32 {
    if peak > threshold {
        let p = peak - threshold;
        let span = (ceiling - threshold).max(1e-4);
        threshold + span * (1.0 - (-p / span).exp())
    } else {
        peak
    }
}

/// 17. apply_asinh_stretch_rs:
/// Estirado de arco seno hiperbólico (asinh) con hombro suave de altas luces.
/// Preserva ratios cromáticos exactos R:G:B en 3D (H, W, 3) y soporta imágenes monocromas 2D (H, W).
#[pyfunction]
#[pyo3(signature = (image, stretch_factor=30.0, black_point=0.0, shoulder_threshold=0.70, highlight_ceiling=1.0, white_point=1.0, color_preserve=0.0))]
pub fn apply_asinh_stretch_rs<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    stretch_factor: f32,
    black_point: f32,
    shoulder_threshold: f32,
    highlight_ceiling: f32,
    white_point: f32,
    color_preserve: f32,
) -> PyResult<Bound<'py, PyAny>> {
    let wp = white_point.max(1.0);
    let cp = color_preserve.clamp(0.0, 1.0);

    if let Ok(arr3) = image.extract::<PyReadonlyArray3<f32>>() {
        let (h, w, c) = arr3.as_array().dim();
        let img_slice = arr3.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

        let out_vec = py.allow_threads(|| {
            let total = h * w * c;
            let mut out = vec![0.0f32; total];

            if stretch_factor <= 1.0 {
                if wp <= 1.0 + 1e-6 {
                    out.par_iter_mut()
                        .zip(img_slice.par_iter())
                        .for_each(|(o, &v)| {
                            *o = (v - black_point).clamp(0.0, 1.0);
                        });
                } else {
                    let row_stride = w * c;
                    let inv_wp = 1.0 / wp;
                    out.par_chunks_exact_mut(row_stride)
                        .zip(img_slice.par_chunks_exact(row_stride))
                        .for_each(|(out_row, in_row)| {
                            for x in 0..w {
                                let base = x * c;
                                let mut peak = 0.0f32;
                                for ch in 0..c {
                                    let u = (in_row[base + ch] - black_point).max(0.0) * inv_wp;
                                    out_row[base + ch] = u;
                                    if u > peak {
                                        peak = u;
                                    }
                                }
                                let factor = if peak > 1e-6 {
                                    soft_shoulder_val(peak, shoulder_threshold, highlight_ceiling) / peak
                                } else {
                                    1.0
                                };
                                for ch in 0..c {
                                    out_row[base + ch] = (out_row[base + ch] * factor).clamp(0.0, 1.0);
                                }
                            }
                        });
                }
            } else {
                let beta = stretch_factor;
                let inv_denom = 1.0 / (beta * wp).asinh();

                if c == 3 {
                    let row_stride = w * 3;
                    out.par_chunks_exact_mut(row_stride)
                        .zip(img_slice.par_chunks_exact(row_stride))
                        .for_each(|(out_row, in_row)| {
                            for x in 0..w {
                                let idx = x * 3;
                                let r = (in_row[idx] - black_point).max(0.0);
                                let g = (in_row[idx + 1] - black_point).max(0.0);
                                let b = (in_row[idx + 2] - black_point).max(0.0);

                                let (ur, ug, ub) = if cp > 1e-4 {
                                    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                                    let gain = if lum > 1e-6 {
                                        ((beta * lum).asinh() * inv_denom) / lum
                                    } else {
                                        0.0
                                    };
                                    let lr = r * gain;
                                    let lg = g * gain;
                                    let lb = b * gain;
                                    if cp >= 0.999 {
                                        (lr, lg, lb)
                                    } else {
                                        let cr = (beta * r).asinh() * inv_denom;
                                        let cg = (beta * g).asinh() * inv_denom;
                                        let cb = (beta * b).asinh() * inv_denom;
                                        (
                                            cp * lr + (1.0 - cp) * cr,
                                            cp * lg + (1.0 - cp) * cg,
                                            cp * lb + (1.0 - cp) * cb,
                                        )
                                    }
                                } else {
                                    (
                                        (beta * r).asinh() * inv_denom,
                                        (beta * g).asinh() * inv_denom,
                                        (beta * b).asinh() * inv_denom,
                                    )
                                };

                                let peak = ur.max(ug).max(ub);
                                let factor = if peak > 1e-6 {
                                    soft_shoulder_val(peak, shoulder_threshold, highlight_ceiling) / peak
                                } else {
                                    1.0
                                };

                                out_row[idx] = (ur * factor).clamp(0.0, 1.0);
                                out_row[idx + 1] = (ug * factor).clamp(0.0, 1.0);
                                out_row[idx + 2] = (ub * factor).clamp(0.0, 1.0);
                            }
                        });
                } else {
                    let row_stride = w * c;
                    out.par_chunks_exact_mut(row_stride)
                        .zip(img_slice.par_chunks_exact(row_stride))
                        .for_each(|(out_row, in_row)| {
                            for x in 0..w {
                                let base = x * c;
                                let mut peak = 0.0f32;
                                for ch in 0..c {
                                    let v = (in_row[base + ch] - black_point).max(0.0);
                                    let u = (beta * v).asinh() * inv_denom;
                                    out_row[base + ch] = u;
                                    if u > peak {
                                        peak = u;
                                    }
                                }
                                let factor = if peak > 1e-6 {
                                    soft_shoulder_val(peak, shoulder_threshold, highlight_ceiling) / peak
                                } else {
                                    1.0
                                };
                                for ch in 0..c {
                                    out_row[base + ch] = (out_row[base + ch] * factor).clamp(0.0, 1.0);
                                }
                            }
                        });
                }
            }

            out
        });

        let py_arr = Array3::from_shape_vec((h, w, c), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(py_arr.into_pyarray(py).into_any())
    } else if let Ok(arr2) = image.extract::<PyReadonlyArray2<f32>>() {
        let (h, w) = arr2.as_array().dim();
        let img_slice = arr2.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

        let out_vec = py.allow_threads(|| {
            let total = h * w;
            let mut out = vec![0.0f32; total];

            if stretch_factor <= 1.0 {
                if wp <= 1.0 + 1e-6 {
                    out.par_iter_mut()
                        .zip(img_slice.par_iter())
                        .for_each(|(o, &v)| {
                            *o = (v - black_point).clamp(0.0, 1.0);
                        });
                } else {
                    let inv_wp = 1.0 / wp;
                    out.par_iter_mut()
                        .zip(img_slice.par_iter())
                        .for_each(|(o, &v)| {
                            let u = (v - black_point).max(0.0) * inv_wp;
                            *o = soft_shoulder_val(u, shoulder_threshold, highlight_ceiling).clamp(0.0, 1.0);
                        });
                }
            } else {
                let beta = stretch_factor;
                let inv_denom = 1.0 / (beta * wp).asinh();

                out.par_chunks_exact_mut(w)
                    .zip(img_slice.par_chunks_exact(w))
                    .for_each(|(out_row, in_row)| {
                        for x in 0..w {
                            let v = (in_row[x] - black_point).max(0.0);
                            let u = (beta * v).asinh() * inv_denom;
                            let sh = soft_shoulder_val(u, shoulder_threshold, highlight_ceiling);
                            out_row[x] = sh.clamp(0.0, 1.0);
                        }
                    });
            }

            out
        });

        let py_arr = Array2::from_shape_vec((h, w), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(py_arr.into_pyarray(py).into_any())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "image debe ser un ndarray float32 de 2 o 3 dimensiones",
        ))
    }
}

/// 18. balance_coronal_white_balance_rs:
/// Balance de blancos coronal neutro sobre el anillo coronal medio [ring_lo * R, ring_hi * R].
/// Devuelve (imagen_balanceada, (ganancia_r, ganancia_g, ganancia_b)).
#[pyfunction]
#[pyo3(signature = (image, center_x, center_y, radius, ring_lo=1.2, ring_hi=1.8, clip_upper=true))]
pub fn balance_coronal_white_balance_rs<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    center_x: f32,
    center_y: f32,
    radius: f32,
    ring_lo: f32,
    ring_hi: f32,
    clip_upper: bool,
) -> PyResult<(Bound<'py, PyAny>, (f32, f32, f32))> {
    if let Ok(arr3) = image.extract::<PyReadonlyArray3<f32>>() {
        let (h, w, c) = arr3.as_array().dim();
        let img_slice = arr3.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

        if c != 3 {
            let out_arr = Array3::from_shape_vec((h, w, c), img_slice.to_vec())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
            return Ok((out_arr.into_pyarray(py).into_any(), (1.0, 1.0, 1.0)));
        }

        let r_lo = radius * ring_lo;
        let r_hi = radius * ring_hi;
        let r_lo_sq = r_lo * r_lo;
        let r_hi_sq = r_hi * r_hi;

        let (balanced_vec, gains) = py.allow_threads(|| {
            let mut r_vals = Vec::new();
            let mut g_vals = Vec::new();
            let mut b_vals = Vec::new();

            for y in 0..h {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                let row_offset = y * w * 3;
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    let d2 = dx * dx + dy2;
                    if d2 >= r_lo_sq && d2 <= r_hi_sq {
                        let idx = row_offset + x * 3;
                        r_vals.push(img_slice[idx]);
                        g_vals.push(img_slice[idx + 1]);
                        b_vals.push(img_slice[idx + 2]);
                    }
                }
            }

            if r_vals.len() < 100 {
                return (img_slice.to_vec(), (1.0f32, 1.0f32, 1.0f32));
            }

            let med_r = median_f32(&mut r_vals);
            let med_g = median_f32(&mut g_vals);
            let med_b = median_f32(&mut b_vals);

            let g_val = if med_g > 1e-6 {
                med_g
            } else {
                (med_r + med_g + med_b) / 3.0
            };

            let f_r = (g_val / med_r.max(1e-6)).clamp(0.4, 2.5);
            let f_g = (g_val / med_g.max(1e-6)).clamp(0.4, 2.5);
            let f_b = (g_val / med_b.max(1e-6)).clamp(0.4, 2.5);

            let mut out = vec![0.0f32; h * w * 3];
            let row_stride = w * 3;
            out.par_chunks_exact_mut(row_stride)
                .zip(img_slice.par_chunks_exact(row_stride))
                .for_each(|(out_row, in_row)| {
                    for x in 0..w {
                        let idx = x * 3;
                        if clip_upper {
                            out_row[idx] = (in_row[idx] * f_r).clamp(0.0, 1.0);
                            out_row[idx + 1] = (in_row[idx + 1] * f_g).clamp(0.0, 1.0);
                            out_row[idx + 2] = (in_row[idx + 2] * f_b).clamp(0.0, 1.0);
                        } else {
                            out_row[idx] = (in_row[idx] * f_r).max(0.0);
                            out_row[idx + 1] = (in_row[idx + 1] * f_g).max(0.0);
                            out_row[idx + 2] = (in_row[idx + 2] * f_b).max(0.0);
                        }
                    }
                });

            (out, (f_r, f_g, f_b))
        });

        let out_arr = Array3::from_shape_vec((h, w, 3), balanced_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok((out_arr.into_pyarray(py).into_any(), gains))
    } else if let Ok(arr2) = image.extract::<PyReadonlyArray2<f32>>() {
        let (h, w) = arr2.as_array().dim();
        let img_slice = arr2.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;
        let out_arr = Array2::from_shape_vec((h, w), img_slice.to_vec())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok((out_arr.into_pyarray(py).into_any(), (1.0, 1.0, 1.0)))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "image debe ser un ndarray float32 de 2 o 3 dimensiones",
        ))
    }
}

// ----------------------------------------------------------------------------
// Tarea 3: Optimización subpíxel de centro por flujo circular sin OpenCV
// ----------------------------------------------------------------------------

#[inline(always)]
fn border_reflect(p: isize, l: usize) -> usize {
    if l <= 1 {
        return 0;
    }
    let m = (2 * l) as isize;
    let r = p.rem_euclid(m) as usize;
    if r >= l {
        2 * l - 1 - r
    } else {
        r
    }
}

#[inline(always)]
fn interp_bilinear_reflect(gray: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let wx = x - (x0 as f32);
    let wy = y - (y0 as f32);

    let rx0 = border_reflect(x0, w);
    let rx1 = border_reflect(x1, w);
    let ry0 = border_reflect(y0, h);
    let ry1 = border_reflect(y1, h);

    let v00 = gray[ry0 * w + rx0];
    let v01 = gray[ry0 * w + rx1];
    let v10 = gray[ry1 * w + rx0];
    let v11 = gray[ry1 * w + rx1];

    (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)
}

#[inline(always)]
fn percentile_linear_180(sorted: &[f32; 180], q: f32) -> f32 {
    let idx_f = (q / 100.0) * 179.0;
    let i0 = (idx_f.floor() as usize).min(178);
    let i1 = i0 + 1;
    let frac = idx_f - (i0 as f32);
    (1.0 - frac) * sorted[i0] + frac * sorted[i1]
}

#[pyfunction]
#[pyo3(signature = (gray, init_cx, init_cy, radius, mode="solar", max_search_dist=None))]
pub fn optimize_center_circular_flux_rs<'py>(
    py: Python<'py>,
    gray: PyReadonlyArray2<'py, f32>,
    init_cx: f32,
    init_cy: f32,
    radius: f32,
    mode: &str,
    max_search_dist: Option<f32>,
) -> PyResult<(f32, f32)> {
    let (h, w) = gray.as_array().dim();
    let gray_slice = gray.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array no contiguo: {}", e))
    })?;

    let mode_solar = mode == "solar";
    let r_in = (1.0f32).max(radius - 2.5);
    let r_out = radius + 2.5;
    let r_int = radius * 0.50;

    let max_dev = match max_search_dist {
        Some(d) if d > 0.0 => d,
        _ => (35.0f32).min((12.0f32).max(radius * 0.12)),
    };

    let result = py.allow_threads(|| {
        let mut cos_a = [0.0f32; 180];
        let mut sin_a = [0.0f32; 180];
        for i in 0..180 {
            let a = (2.0 * std::f32::consts::PI * (i as f32)) / 180.0;
            cos_a[i] = a.cos();
            sin_a[i] = a.sin();
        }

        let loss_fn = |cx: f32, cy: f32| -> f32 {
            let dist_dev = (cx - init_cx).hypot(cy - init_cy);
            if dist_dev > max_dev || cx < 0.0 || cx >= (w as f32) || cy < 0.0 || cy >= (h as f32) {
                return 1e6 + dist_dev * 1000.0;
            }

            let mut diff = [0.0f32; 180];
            let mut sum_v_int = 0.0f32;

            if mode_solar {
                for i in 0..180 {
                    let ca = cos_a[i];
                    let sa = sin_a[i];
                    let v_in = interp_bilinear_reflect(gray_slice, w, h, cx + r_in * ca, cy + r_in * sa);
                    let v_out = interp_bilinear_reflect(gray_slice, w, h, cx + r_out * ca, cy + r_out * sa);
                    diff[i] = v_out - v_in;
                    let v_int = interp_bilinear_reflect(gray_slice, w, h, cx + r_int * ca, cy + r_int * sa);
                    sum_v_int += v_int;
                }
                diff.sort_unstable_by(|a, b| a.total_cmp(b));
                let grad = percentile_linear_180(&diff, 75.0);
                let pen_in = sum_v_int / 180.0;
                let eff_pen = (pen_in * 0.5).min((0.0f32).max(grad * 0.4));
                -(grad - eff_pen)
            } else {
                for i in 0..180 {
                    let ca = cos_a[i];
                    let sa = sin_a[i];
                    let v_in = interp_bilinear_reflect(gray_slice, w, h, cx + r_in * ca, cy + r_in * sa);
                    let v_out = interp_bilinear_reflect(gray_slice, w, h, cx + r_out * ca, cy + r_out * sa);
                    diff[i] = v_in - v_out;
                }
                diff.sort_unstable_by(|a, b| a.total_cmp(b));
                -percentile_linear_180(&diff, 60.0)
            }
        };

        let mut best_cx = init_cx;
        let mut best_cy = init_cy;
        let mut best_loss = loss_fn(best_cx, best_cy);

        let search_range = max_dev.min(24.0) as i32;
        let step = if search_range <= 16 { 2 } else { 3 };

        let mut dx = -search_range;
        while dx <= search_range {
            let mut dy = -search_range;
            while dy <= search_range {
                let cx = init_cx + (dx as f32);
                let cy = init_cy + (dy as f32);
                let l = loss_fn(cx, cy);
                if l < best_loss {
                    best_loss = l;
                    best_cx = cx;
                    best_cy = cy;
                }
                dy += step;
            }
            dx += step;
        }

        let nonzdelt = 0.05f32;
        let zdelt = 0.00025f32;
        let x0 = [best_cx, best_cy];
        let mut sim = [
            [x0[0], x0[1]],
            [if x0[0] != 0.0 { x0[0] * (1.0 + nonzdelt) } else { zdelt }, x0[1]],
            [x0[0], if x0[1] != 0.0 { x0[1] * (1.0 + nonzdelt) } else { zdelt }],
        ];
        let mut fsim = [
            loss_fn(sim[0][0], sim[0][1]),
            loss_fn(sim[1][0], sim[1][1]),
            loss_fn(sim[2][0], sim[2][1]),
        ];

        let sort_simplex = |s: &mut [[f32; 2]; 3], f: &mut [f32; 3]| {
            if f[0] > f[1] {
                f.swap(0, 1);
                s.swap(0, 1);
            }
            if f[1] > f[2] {
                f.swap(1, 2);
                s.swap(1, 2);
            }
            if f[0] > f[1] {
                f.swap(0, 1);
                s.swap(0, 1);
            }
        };

        sort_simplex(&mut sim, &mut fsim);

        let xatol = 0.05f32;
        let fatol = 1e-4f32;
        let maxiter = 50usize;

        for _ in 1..maxiter {
            let max_x_diff = (sim[1][0] - sim[0][0]).abs()
                .max((sim[2][0] - sim[0][0]).abs())
                .max((sim[1][1] - sim[0][1]).abs())
                .max((sim[2][1] - sim[0][1]).abs());
            let max_f_diff = (fsim[0] - fsim[1]).abs().max((fsim[0] - fsim[2]).abs());
            if max_x_diff <= xatol && max_f_diff <= fatol {
                break;
            }

            let xbar = [0.5 * (sim[0][0] + sim[1][0]), 0.5 * (sim[0][1] + sim[1][1])];
            let xr = [2.0 * xbar[0] - sim[2][0], 2.0 * xbar[1] - sim[2][1]];
            let fxr = loss_fn(xr[0], xr[1]);

            let mut doshrink = false;
            if fxr < fsim[0] {
                let xe = [3.0 * xbar[0] - 2.0 * sim[2][0], 3.0 * xbar[1] - 2.0 * sim[2][1]];
                let fxe = loss_fn(xe[0], xe[1]);
                if fxe < fxr {
                    sim[2] = xe;
                    fsim[2] = fxe;
                } else {
                    sim[2] = xr;
                    fsim[2] = fxr;
                }
            } else if fxr < fsim[1] {
                sim[2] = xr;
                fsim[2] = fxr;
            } else {
                if fxr < fsim[2] {
                    let xc = [1.5 * xbar[0] - 0.5 * sim[2][0], 1.5 * xbar[1] - 0.5 * sim[2][1]];
                    let fxc = loss_fn(xc[0], xc[1]);
                    if fxc <= fxr {
                        sim[2] = xc;
                        fsim[2] = fxc;
                    } else {
                        doshrink = true;
                    }
                } else {
                    let xcc = [0.5 * xbar[0] + 0.5 * sim[2][0], 0.5 * xbar[1] + 0.5 * sim[2][1]];
                    let fxcc = loss_fn(xcc[0], xcc[1]);
                    if fxcc < fsim[2] {
                        sim[2] = xcc;
                        fsim[2] = fxcc;
                    } else {
                        doshrink = true;
                    }
                }
                if doshrink {
                    sim[1] = [sim[0][0] + 0.5 * (sim[1][0] - sim[0][0]), sim[0][1] + 0.5 * (sim[1][1] - sim[0][1])];
                    sim[2] = [sim[0][0] + 0.5 * (sim[2][0] - sim[0][0]), sim[0][1] + 0.5 * (sim[2][1] - sim[0][1])];
                    fsim[1] = loss_fn(sim[1][0], sim[1][1]);
                    fsim[2] = loss_fn(sim[2][0], sim[2][1]);
                }
            }
            sort_simplex(&mut sim, &mut fsim);
        }

        let final_cx = sim[0][0];
        let final_cy = sim[0][1];
        let final_loss = fsim[0];
        let final_dist = (final_cx - init_cx).hypot(final_cy - init_cy);

        if final_dist > max_dev
            || final_cx < 0.0
            || final_cx >= (w as f32)
            || final_cy < 0.0
            || final_cy >= (h as f32)
            || final_loss > best_loss
        {
            (best_cx, best_cy)
        } else {
            (final_cx, final_cy)
        }
    });

    Ok(result)
}

// ----------------------------------------------------------------------------
// Tarea 2: Fusión HDR Solar en una sola pasada paralela (core/eclipse.py)
// ----------------------------------------------------------------------------

#[inline(always)]
fn smoothstep_f32(x: f32, edge0: f32, edge1: f32) -> f32 {
    let denom = (edge1 - edge0).max(1e-6);
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
pub(super) fn percentile_exact_f32(data: &mut [f32], q: f32) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let idx_f = (q / 100.0) * ((n - 1) as f32);
    let i0 = (idx_f.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    let frac = idx_f - (i0 as f32);
    data.select_nth_unstable_by(i0, |a, b| a.total_cmp(b));
    let v0 = data[i0];
    let v1 = if i1 > i0 {
        *data[i0 + 1..].iter().min_by(|a, b| a.total_cmp(b)).unwrap_or(&v0)
    } else {
        v0
    };
    (1.0 - frac) * v0 + frac * v1
}

#[pyfunction]
#[pyo3(signature = (
    images,
    exposures_s,
    saturation_threshold=0.90,
    noise_floor=0.015,
    lunar_mask_params=None,
    gradual_handover=true,
    bg_pedestals=None,
    clip_highlights=true
))]
pub fn fuse_hdr_bracketing_solar_rs<'py>(
    py: Python<'py>,
    images: &Bound<'py, PyAny>,
    exposures_s: Vec<f32>,
    saturation_threshold: f32,
    noise_floor: f32,
    lunar_mask_params: Option<(f32, f32, f32)>,
    gradual_handover: bool,
    bg_pedestals: Option<Vec<Vec<f32>>>,
    clip_highlights: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py_list = images.downcast::<pyo3::types::PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("images debe ser una lista de ndarrays"))?;
    let n = py_list.len();
    if n == 0 || n != exposures_s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "images y exposures_s deben tener la misma longitud no vacía",
        ));
    }

    let is_3d = py_list.get_item(0)?.extract::<PyReadonlyArray3<f32>>().is_ok();
    let sat_lo = (0.40f32).max(saturation_threshold - 0.20);
    let sat_hi = saturation_threshold;

    if is_3d {
        let mut arrs = Vec::with_capacity(n);
        let mut h = 0;
        let mut w = 0;
        for i in 0..n {
            let item = py_list.get_item(i)?;
            let arr = item.extract::<PyReadonlyArray3<f32>>()?;
            let (img_h, img_w, c) = arr.as_array().dim();
            if c != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Imágenes 3D deben tener 3 canales",
                ));
            }
            if i == 0 {
                h = img_h;
                w = img_w;
            } else if img_h != h || img_w != w {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Todas las imágenes deben tener las mismas dimensiones",
                ));
            }
            arrs.push(arr);
        }
        let img_slices: Vec<&[f32]> = arrs
            .iter()
            .map(|a| a.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous")))
            .collect::<Result<Vec<_>, _>>()?;

        let bg: Vec<[f32; 3]> = match bg_pedestals {
            Some(ref peds) if peds.len() == n => peds
                .iter()
                .map(|p| {
                    if p.len() >= 3 {
                        [p[0], p[1], p[2]]
                    } else if !p.is_empty() {
                        [p[0], p[0], p[0]]
                    } else {
                        [0.0, 0.0, 0.0]
                    }
                })
                .collect(),
            _ => vec![[0.0, 0.0, 0.0]; n],
        };

        let mut out_vec = vec![0.0f32; h * w * 3];

        py.allow_threads(|| {
            out_vec
                .par_chunks_exact_mut(w * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let moon_w = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                            let dist = (x_f - mcx).hypot(y_f - mcy);
                            smoothstep_f32(dist, mr - 2.0, mr) as f64
                        } else {
                            1.0f64
                        };

                        let mut wf_r = 0.0f64;
                        let mut wf_g = 0.0f64;
                        let mut wf_b = 0.0f64;
                        let mut sw = 0.0f64;

                        let pix_idx = (y * w + x) * 3;

                        for i in 0..n {
                            let t = (exposures_s[i] as f64).max(1e-6);
                            let shorter_t = if i > 0 {
                                Some(exposures_s[i - 1] as f64)
                            } else {
                                None
                            };

                            let in_r = img_slices[i][pix_idx];
                            let in_g = img_slices[i][pix_idx + 1];
                            let in_b = img_slices[i][pix_idx + 2];

                            let clean_r = ((in_r - bg[i][0]).max(0.0)) as f64;
                            let clean_g = ((in_g - bg[i][1]).max(0.0)) as f64;
                            let clean_b = ((in_b - bg[i][2]).max(0.0)) as f64;

                            let flux_r = clean_r / t;
                            let flux_g = clean_g / t;
                            let flux_b = clean_b / t;

                            let lum = 0.2126 * (in_r as f64) + 0.7152 * (in_g as f64) + 0.0722 * (in_b as f64);
                            let max_c = (in_r.max(in_g).max(in_b)) as f64;

                            let w_low = smoothstep_f32(lum as f32, 0.0, noise_floor * 2.0) as f64;
                            let w_high = 1.0 - (smoothstep_f32(max_c as f32, sat_lo, sat_hi) as f64);
                            let mut w = w_low * w_high;

                            if gradual_handover {
                                if let Some(st) = shorter_t {
                                    if st > 0.0 {
                                        let taper = w_high;
                                        let weight_factor = taper + (1.0 - taper) * (t / st);
                                        w /= weight_factor.max(1e-4);
                                    }
                                }
                            }

                            if lunar_mask_params.is_some() {
                                w *= moon_w;
                            }

                            wf_r += flux_r * w;
                            wf_g += flux_g * w;
                            wf_b += flux_b * w;
                            sw += w;
                        }

                        let (mut merged_r, mut merged_g, mut merged_b) = if sw < 1e-6 {
                            let in_r0 = img_slices[0][pix_idx];
                            let in_g0 = img_slices[0][pix_idx + 1];
                            let in_b0 = img_slices[0][pix_idx + 2];
                            let shortest_max = in_r0.max(in_g0).max(in_b0);

                            let shortest_t = (exposures_s[0] as f64).max(1e-6);
                            let longest_t = (exposures_s[n - 1] as f64).max(1e-6);

                            if shortest_max >= sat_lo {
                                (
                                    (in_r0 as f64) / shortest_t,
                                    (in_g0 as f64) / shortest_t,
                                    (in_b0 as f64) / shortest_t,
                                )
                            } else {
                                let in_rn = img_slices[n - 1][pix_idx];
                                let in_gn = img_slices[n - 1][pix_idx + 1];
                                let in_bn = img_slices[n - 1][pix_idx + 2];
                                (
                                    (in_rn as f64) / longest_t,
                                    (in_gn as f64) / longest_t,
                                    (in_bn as f64) / longest_t,
                                )
                            }
                        } else {
                            (wf_r / sw, wf_g / sw, wf_b / sw)
                        };

                        if lunar_mask_params.is_some() {
                            merged_r *= moon_w;
                            merged_g *= moon_w;
                            merged_b *= moon_w;
                        }

                        row[x * 3] = merged_r as f32;
                        row[x * 3 + 1] = merged_g as f32;
                        row[x * 3 + 2] = merged_b as f32;
                    }
                });

            let mut valid_vals = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                let mut v = Vec::with_capacity(h * w * 3);
                for y in 0..h {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let dist = (x_f - mcx).hypot(y_f - mcy);
                        let mw = smoothstep_f32(dist, mr - 2.0, mr);
                        if mw > 0.5 {
                            let idx = (y * w + x) * 3;
                            v.push(out_vec[idx]);
                            v.push(out_vec[idx + 1]);
                            v.push(out_vec[idx + 2]);
                        }
                    }
                }
                v
            } else {
                out_vec.clone()
            };

            let hi_pct = if clip_highlights { 99.999 } else { 99.98 };
            let p_high = percentile_exact_f32(&mut valid_vals, hi_pct);
            let p_low = percentile_exact_f32(&mut valid_vals, 0.05);
            let denom = (p_high - p_low).max(1e-8);

            out_vec.par_iter_mut().for_each(|v| {
                let norm = (*v - p_low) / denom;
                *v = if clip_highlights { norm.clamp(0.0, 1.0) } else { norm.max(0.0) };
            });
        });

        let out_arr = Array3::from_shape_vec((h, w, 3), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(out_arr.into_pyarray(py).into_any())
    } else {
        // 2D Monochrome branch
        let mut arrs = Vec::with_capacity(n);
        let mut h = 0;
        let mut w = 0;
        for i in 0..n {
            let item = py_list.get_item(i)?;
            let arr = item.extract::<PyReadonlyArray2<f32>>()?;
            let (img_h, img_w) = arr.as_array().dim();
            if i == 0 {
                h = img_h;
                w = img_w;
            } else if img_h != h || img_w != w {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Todas las imágenes deben tener las mismas dimensiones",
                ));
            }
            arrs.push(arr);
        }
        let img_slices: Vec<&[f32]> = arrs
            .iter()
            .map(|a| a.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous")))
            .collect::<Result<Vec<_>, _>>()?;

        let bg: Vec<f32> = match bg_pedestals {
            Some(ref peds) if peds.len() == n => peds
                .iter()
                .map(|p| if !p.is_empty() { p[0] } else { 0.0 })
                .collect(),
            _ => vec![0.0; n],
        };

        let mut out_vec = vec![0.0f32; h * w];

        py.allow_threads(|| {
            out_vec
                .par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(y, row)| {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let moon_w = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                            let dist = (x_f - mcx).hypot(y_f - mcy);
                            smoothstep_f32(dist, mr - 2.0, mr) as f64
                        } else {
                            1.0f64
                        };

                        let mut wf = 0.0f64;
                        let mut sw = 0.0f64;
                        let pix_idx = y * w + x;

                        for i in 0..n {
                            let t = (exposures_s[i] as f64).max(1e-6);
                            let shorter_t = if i > 0 {
                                Some(exposures_s[i - 1] as f64)
                            } else {
                                None
                            };

                            let in_v = img_slices[i][pix_idx];
                            let clean_v = ((in_v - bg[i]).max(0.0)) as f64;
                            let flux = clean_v / t;

                            let w_low = smoothstep_f32(in_v, 0.0, noise_floor * 2.0) as f64;
                            let w_high = 1.0 - (smoothstep_f32(in_v, sat_lo, sat_hi) as f64);
                            let mut w = w_low * w_high;

                            if gradual_handover {
                                if let Some(st) = shorter_t {
                                    if st > 0.0 {
                                        let taper = w_high;
                                        let weight_factor = taper + (1.0 - taper) * (t / st);
                                        w /= weight_factor.max(1e-4);
                                    }
                                }
                            }

                            if lunar_mask_params.is_some() {
                                w *= moon_w;
                            }

                            wf += flux * w;
                            sw += w;
                        }

                        let mut merged = if sw < 1e-6 {
                            let in_v0 = img_slices[0][pix_idx];
                            let shortest_t = (exposures_s[0] as f64).max(1e-6);
                            let longest_t = (exposures_s[n - 1] as f64).max(1e-6);

                            if in_v0 >= sat_lo {
                                (in_v0 as f64) / shortest_t
                            } else {
                                (img_slices[n - 1][pix_idx] as f64) / longest_t
                            }
                        } else {
                            wf / sw
                        };

                        if lunar_mask_params.is_some() {
                            merged *= moon_w;
                        }

                        row[x] = merged as f32;
                    }
                });

            let mut valid_vals = if let Some((mcx, mcy, mr)) = lunar_mask_params {
                let mut v = Vec::with_capacity(h * w);
                for y in 0..h {
                    let y_f = y as f32;
                    for x in 0..w {
                        let x_f = x as f32;
                        let dist = (x_f - mcx).hypot(y_f - mcy);
                        let mw = smoothstep_f32(dist, mr - 2.0, mr);
                        if mw > 0.5 {
                            v.push(out_vec[y * w + x]);
                        }
                    }
                }
                v
            } else {
                out_vec.clone()
            };

            let hi_pct = if clip_highlights { 99.999 } else { 99.98 };
            let p_high = percentile_exact_f32(&mut valid_vals, hi_pct);
            let p_low = percentile_exact_f32(&mut valid_vals, 0.05);
            let denom = (p_high - p_low).max(1e-8);

            out_vec.par_iter_mut().for_each(|v| {
                let norm = (*v - p_low) / denom;
                *v = if clip_highlights { norm.clamp(0.0, 1.0) } else { norm.max(0.0) };
            });
        });

        let out_arr = Array2::from_shape_vec((h, w), out_vec)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape inválido: {}", e)))?;
        Ok(out_arr.into_pyarray(py).into_any())
    }
}
