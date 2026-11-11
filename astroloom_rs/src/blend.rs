use numpy::ndarray::{Array3, ArrayView2, ArrayView3};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// KERNELS PARALELOS DE FUSIÓN DE CAPAS (BLEND MODES) - RAYON SIMD
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Screen,
    Luminosity,
    SoftLight,
    Overlay,
    Multiply,
}

impl BlendMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "screen" | "scrn" => BlendMode::Screen,
            "luminosity" | "lum" => BlendMode::Luminosity,
            "soft_light" | "softlight" | "sftl" => BlendMode::SoftLight,
            "overlay" | "over" => BlendMode::Overlay,
            "multiply" | "mul" => BlendMode::Multiply,
            _ => BlendMode::Normal,
        }
    }
}

#[inline(always)]
fn calc_weight(mask_view: Option<&ArrayView2<'_, f32>>, y: usize, x: usize, opacity: f32) -> f32 {
    match mask_view {
        Some(m) => {
            let mv = unsafe { *m.uget((y, x)) };
            (mv * opacity).clamp(0.0, 1.0)
        }
        None => opacity.clamp(0.0, 1.0),
    }
}

#[inline(always)]
fn blend_pixel_normal(cv: &[f32; 3], fv: &[f32; 3], out: &mut [f32; 3], w: f32) {
    let inv_w = 1.0 - w;
    out[0] = (inv_w * cv[0] + w * fv[0]).clamp(0.0, 1.0);
    out[1] = (inv_w * cv[1] + w * fv[1]).clamp(0.0, 1.0);
    out[2] = (inv_w * cv[2] + w * fv[2]).clamp(0.0, 1.0);
}

#[inline(always)]
fn blend_pixel_screen(cv: &[f32; 3], fv: &[f32; 3], out: &mut [f32; 3], w: f32) {
    for ch in 0..3 {
        let s = (fv[ch] * w).clamp(0.0, 1.0);
        let c = cv[ch];
        out[ch] = (c + s - c * s).clamp(0.0, 1.0);
    }
}

#[inline(always)]
fn blend_pixel_luminosity(cv: &[f32; 3], fv: &[f32; 3], out: &mut [f32; 3], w: f32) {
    let l_base = 0.2126 * cv[0] + 0.7152 * cv[1] + 0.0722 * cv[2];
    let l_top = 0.2126 * fv[0] + 0.7152 * fv[1] + 0.0722 * fv[2];
    let d = l_top - l_base;

    let mut r = cv[0] + d;
    let mut g = cv[1] + d;
    let mut b = cv[2] + d;

    let l = l_top;
    let n = r.min(g).min(b);
    let x = r.max(g).max(b);

    if n < 0.0 && l != n {
        let factor = l / (l - n).max(1e-6);
        r = l + (r - l) * factor;
        g = l + (g - l) * factor;
        b = l + (b - l) * factor;
    }
    if x > 1.0 && x != l {
        let factor = (1.0 - l) / (x - l).max(1e-6);
        r = l + (r - l) * factor;
        g = l + (g - l) * factor;
        b = l + (b - l) * factor;
    }

    let inv_w = 1.0 - w;
    out[0] = (inv_w * cv[0] + w * r.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    out[1] = (inv_w * cv[1] + w * g.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    out[2] = (inv_w * cv[2] + w * b.clamp(0.0, 1.0)).clamp(0.0, 1.0);
}

#[inline(always)]
fn blend_pixel_soft_light(cv: &[f32; 3], fv: &[f32; 3], out: &mut [f32; 3], w: f32) {
    let inv_w = 1.0 - w;
    for ch in 0..3 {
        let c = cv[ch];
        let f = fv[ch];
        let d = if f <= 0.5 {
            c - (1.0 - 2.0 * f) * c * (1.0 - c)
        } else {
            let sqrt_c = c.max(0.0).sqrt();
            c + (2.0 * f - 1.0) * (sqrt_c - c)
        };
        out[ch] = (inv_w * c + w * d.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    }
}

#[inline(always)]
fn blend_pixel_overlay(cv: &[f32; 3], fv: &[f32; 3], out: &mut [f32; 3], w: f32) {
    let inv_w = 1.0 - w;
    for ch in 0..3 {
        let c = cv[ch];
        let f = fv[ch];
        let d = if c <= 0.5 {
            2.0 * c * f
        } else {
            1.0 - 2.0 * (1.0 - c) * (1.0 - f)
        };
        out[ch] = (inv_w * c + w * d.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    }
}

#[inline(always)]
fn blend_pixel_multiply(cv: &[f32; 3], fv: &[f32; 3], out: &mut [f32; 3], w: f32) {
    let inv_w = 1.0 - w;
    out[0] = (inv_w * cv[0] + w * (cv[0] * fv[0])).clamp(0.0, 1.0);
    out[1] = (inv_w * cv[1] + w * (cv[1] * fv[1])).clamp(0.0, 1.0);
    out[2] = (inv_w * cv[2] + w * (cv[2] * fv[2])).clamp(0.0, 1.0);
}

pub fn blend_images_parallel(
    canvas_view: &ArrayView3<'_, f32>,
    top_view: &ArrayView3<'_, f32>,
    mask_view: Option<&ArrayView2<'_, f32>>,
    mode: BlendMode,
    opacity: f32,
) -> Vec<f32> {
    let (h, w, c) = canvas_view.dim();
    let row_stride = w * c;
    let mut buf = vec![0.0f32; h * row_stride];

    buf.par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, row_slice)| {
            let mut cv_pixel = [0.0f32; 3];
            let mut fv_pixel = [0.0f32; 3];
            let mut out_pixel = [0.0f32; 3];

            for x in 0..w {
                let base_idx = x * c;
                let weight = calc_weight(mask_view, y, x, opacity);
                if weight <= 1e-6 {
                    for ch in 0..c {
                        row_slice[base_idx + ch] = unsafe { *canvas_view.uget((y, x, ch)) };
                    }
                    continue;
                }

                if c == 3 {
                    cv_pixel[0] = unsafe { *canvas_view.uget((y, x, 0)) };
                    cv_pixel[1] = unsafe { *canvas_view.uget((y, x, 1)) };
                    cv_pixel[2] = unsafe { *canvas_view.uget((y, x, 2)) };

                    fv_pixel[0] = unsafe { *top_view.uget((y, x, 0)) };
                    fv_pixel[1] = unsafe { *top_view.uget((y, x, 1)) };
                    fv_pixel[2] = unsafe { *top_view.uget((y, x, 2)) };

                    match mode {
                        BlendMode::Normal => blend_pixel_normal(&cv_pixel, &fv_pixel, &mut out_pixel, weight),
                        BlendMode::Screen => blend_pixel_screen(&cv_pixel, &fv_pixel, &mut out_pixel, weight),
                        BlendMode::Luminosity => blend_pixel_luminosity(&cv_pixel, &fv_pixel, &mut out_pixel, weight),
                        BlendMode::SoftLight => blend_pixel_soft_light(&cv_pixel, &fv_pixel, &mut out_pixel, weight),
                        BlendMode::Overlay => blend_pixel_overlay(&cv_pixel, &fv_pixel, &mut out_pixel, weight),
                        BlendMode::Multiply => blend_pixel_multiply(&cv_pixel, &fv_pixel, &mut out_pixel, weight),
                    }

                    row_slice[base_idx + 0] = out_pixel[0];
                    row_slice[base_idx + 1] = out_pixel[1];
                    row_slice[base_idx + 2] = out_pixel[2];
                } else {
                    for ch in 0..c {
                        let cv = unsafe { *canvas_view.uget((y, x, ch)) };
                        let fv = unsafe { *top_view.uget((y, x, ch)) };
                        row_slice[base_idx + ch] = (1.0 - weight) * cv + weight * fv;
                    }
                }
            }
        });

    buf
}

#[pyfunction]
#[pyo3(signature = (canvas, top, blend_mode="normal", opacity=1.0, mask=None))]
pub fn apply_blend_mode_rs<'py>(
    py: Python<'py>,
    canvas: PyReadonlyArray3<'py, f32>,
    top: PyReadonlyArray3<'py, f32>,
    blend_mode: &str,
    opacity: f32,
    mask: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let c_view = canvas.as_array();
    let t_view = top.as_array();
    let (h, w, c) = c_view.dim();
    let (th, tw, tc) = t_view.dim();

    if h != th || w != tw || c != tc {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimensiones incompatibles: canvas es ({}, {}, {}), top es ({}, {}, {})",
            h, w, c, th, tw, tc
        )));
    }

    let mask_arr = mask.as_ref().map(|m| m.as_array());
    if let Some(ref m) = mask_arr {
        let (mh, mw) = m.dim();
        if mh != h || mw != w {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimensiones de máscara incompatibles: esperada ({}, {}), obtenida ({}, {})",
                h, w, mh, mw
            )));
        }
    }

    let mode = BlendMode::from_str(blend_mode);
    let mask_ref = mask_arr.as_ref();

    let out_vec = py.allow_threads(|| {
        blend_images_parallel(&c_view, &t_view, mask_ref, mode, opacity)
    });

    let out_array = Array3::from_shape_vec((h, w, c), out_vec)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out_array.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (canvas, top, opacity=1.0, mask=None))]
pub fn luminosity_blend_rs<'py>(
    py: Python<'py>,
    canvas: PyReadonlyArray3<'py, f32>,
    top: PyReadonlyArray3<'py, f32>,
    opacity: f32,
    mask: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    apply_blend_mode_rs(py, canvas, top, "luminosity", opacity, mask)
}

#[pyfunction]
#[pyo3(signature = (canvas, top, opacity=1.0, mask=None))]
pub fn soft_light_blend_rs<'py>(
    py: Python<'py>,
    canvas: PyReadonlyArray3<'py, f32>,
    top: PyReadonlyArray3<'py, f32>,
    opacity: f32,
    mask: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    apply_blend_mode_rs(py, canvas, top, "soft_light", opacity, mask)
}
