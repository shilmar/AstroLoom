use crate::stacking::compute_median_in_place;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// AJUSTES LOCALES Y MOTOR DEL REVELADOR (core/mask_layers.py y core/stretch.py)
// ============================================================================

#[inline(always)]
fn adjust_contrast_val(x: f32, contrast: f32, pivot: f32) -> f32 {
    if contrast.abs() < 1e-4 {
        return x;
    }
    if contrast > 0.0 {
        let k = 1.0 + (contrast * 9.0);
        let sig = 1.0 / (1.0 + (-k * (x - pivot)).exp());
        let sig_0 = 1.0 / (1.0 + (k * pivot).exp());
        let sig_1 = 1.0 / (1.0 + (-k * (1.0 - pivot)).exp());
        let norm_s = (sig - sig_0) / (sig_1 - sig_0).max(1e-7);
        (1.0 - contrast) * x + contrast * norm_s
    } else {
        let factor = 1.0 + (contrast * 0.5);
        pivot + (x - pivot) * factor
    }
}

#[inline(always)]
fn adjust_sat_and_vibrance(r: &mut f32, g: &mut f32, b: &mut f32, saturation: f32, vibrance: f32) {
    let v = (*r).max((*g).max(*b));
    let min_c = (*r).min((*g).min(*b));
    let c = v - min_c;
    if v > 1e-7 && c > 1e-7 {
        let s = c / v;
        let mut sat_mult = saturation;
        if vibrance.abs() > 1e-4 {
            if vibrance > 0.0 {
                sat_mult *= 1.0 + vibrance * (1.0 - s);
            } else {
                sat_mult *= 1.0 + vibrance;
            }
        }
        let c_new = (c * sat_mult).min(v);
        let ratio = c_new / c;
        *r = v - (v - *r) * ratio;
        *g = v - (v - *g) * ratio;
        *b = v - (v - *b) * ratio;
    }
}

#[pyfunction]
#[pyo3(signature = (
    orig_img,
    mask=None,
    curve_lut=None,
    opacity=1.0,
    exposure=0.0,
    contrast=0.0,
    highlights=0.0,
    shadows=0.0,
    saturation=1.0,
    vibrance=0.0,
    temp=0.0,
    tint=0.0,
    gain_r=None,
    gain_g=None,
    gain_b=None
))]
pub fn apply_layer_adjustments_fused<'py>(
    py: Python<'py>,
    orig_img: PyReadonlyArray3<'py, f32>,
    mask: Option<PyReadonlyArray2<'py, f32>>,
    curve_lut: Option<PyReadonlyArray1<'py, f32>>,
    opacity: f32,
    exposure: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    saturation: f32,
    vibrance: f32,
    temp: f32,
    tint: f32,
    gain_r: Option<f32>,
    gain_g: Option<f32>,
    gain_b: Option<f32>,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = orig_img.as_array();
    let mask_view = mask.as_ref().map(|m| m.as_array());
    let (h, w, c) = img_view.dim();

    let (g_r, g_g, g_b) = if let (Some(gr), Some(gg), Some(gb)) = (gain_r, gain_g, gain_b) {
        (gr, gg, gb)
    } else {
        let (mut r, mut g, mut b) = if temp.abs() > 1e-4 || tint.abs() > 1e-4 {
            let rg = 1.0 + (temp * 0.5) + (tint * 0.25);
            let gg = 1.0 - (tint * 0.5);
            let bg = 1.0 - (temp * 0.5) + (tint * 0.25);
            let mean = (rg + gg + bg) / 3.0;
            (rg / mean, gg / mean, bg / mean)
        } else {
            (1.0, 1.0, 1.0)
        };
        if exposure.abs() > 1e-3 {
            let ev = 2.0f32.powf(exposure);
            r *= ev;
            g *= ev;
            b *= ev;
        }
        (r, g, b)
    };

    let effective_lut: Option<Vec<f32>> = if let Some(lut) = curve_lut {
        Some(lut.as_slice().unwrap().to_vec())
    } else if contrast.abs() > 1e-3 {
        let mut v = Vec::with_capacity(256);
        for i in 0..256 {
            let x = (i as f32) / 255.0;
            v.push(adjust_contrast_val(x, contrast, 0.35));
        }
        Some(v)
    } else {
        None
    };

    let has_sat_or_vib = (saturation - 1.0).abs() > 1e-3 || vibrance.abs() > 1e-3;
    let has_shadows = shadows.abs() > 1e-3;
    let has_highlights = highlights.abs() > 1e-3;

    let out_vec = py.allow_threads(|| {
        let row_stride = w * c;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let base_idx = x * c;
                    let r0 = unsafe { *img_view.uget((y, x, 0)) };
                    let g0 = unsafe { *img_view.uget((y, x, 1)) };
                    let b0 = unsafe { *img_view.uget((y, x, 2)) };

                    let m_val = if let Some(ref m) = mask_view {
                        unsafe { *m.uget((y, x)) }
                    } else {
                        1.0
                    };

                    let weight = (m_val * opacity).clamp(0.0, 1.0);
                    if weight <= 1e-4 {
                        row_slice[base_idx] = r0;
                        row_slice[base_idx + 1] = g0;
                        row_slice[base_idx + 2] = b0;
                        continue;
                    }

                    let mut r = r0;
                    let mut g = g0;
                    let mut b = b0;

                    // 1. Saturación y vibrance en HSV sin copias
                    if has_sat_or_vib {
                        adjust_sat_and_vibrance(&mut r, &mut g, &mut b, saturation, vibrance);
                    }

                    // 2. Ganancias de balance de blancos y exposición
                    r = (r * g_r).clamp(0.0, 1.0);
                    g = (g * g_g).clamp(0.0, 1.0);
                    b = (b * g_b).clamp(0.0, 1.0);

                    // 3. Sombras
                    if has_shadows {
                        let lift_r = (1.0 - r) * (1.0 - r);
                        r *= 1.0 + shadows * lift_r;
                        let lift_g = (1.0 - g) * (1.0 - g);
                        g *= 1.0 + shadows * lift_g;
                        let lift_b = (1.0 - b) * (1.0 - b);
                        b *= 1.0 + shadows * lift_b;
                    }

                    // 4. Luces (Highlights)
                    if has_highlights {
                        r += highlights * (r * r) * 0.4;
                        g += highlights * (g * g) * 0.4;
                        b += highlights * (b * b) * 0.4;
                    }

                    r = r.clamp(0.0, 1.0);
                    g = g.clamp(0.0, 1.0);
                    b = b.clamp(0.0, 1.0);

                    // 5. Curva tonal / Contraste por LUT
                    if let Some(ref lut) = effective_lut {
                        let lut_max = (lut.len() - 1) as f32;
                        let ir = ((r * lut_max) as usize).min(lut.len() - 1);
                        let ig = ((g * lut_max) as usize).min(lut.len() - 1);
                        let ib = ((b * lut_max) as usize).min(lut.len() - 1);
                        r = unsafe { *lut.get_unchecked(ir) };
                        g = unsafe { *lut.get_unchecked(ig) };
                        b = unsafe { *lut.get_unchecked(ib) };
                    }

                    // 6. Mezcla ponderada por máscara y opacidad
                    row_slice[base_idx] = r * weight + r0 * (1.0 - weight);
                    row_slice[base_idx + 1] = g * weight + g0 * (1.0 - weight);
                    row_slice[base_idx + 2] = b * weight + b0 * (1.0 - weight);
                }
            });

        buf
    });

    let out_array = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_array.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (canvas, stars, opacity=1.0))]
pub fn screen_blend<'py>(
    py: Python<'py>,
    canvas: PyReadonlyArray3<'py, f32>,
    stars: PyReadonlyArray3<'py, f32>,
    opacity: f32,
) -> Bound<'py, PyArray3<f32>> {
    let c_view = canvas.as_array();
    let s_view = stars.as_array();
    let (h, w, c) = c_view.dim();

    let out_vec = py.allow_threads(|| {
        let row_stride = w * c;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let base_idx = x * c;
                    for ch in 0..c {
                        let cv = unsafe { *c_view.uget((y, x, ch)) };
                        let st = unsafe { *s_view.uget((y, x, ch)) };
                        let s = (st * opacity).clamp(0.0, 1.0);
                        row_slice[base_idx + ch] = cv + s - cv * s;
                    }
                }
            });

        buf
    });

    let out_array = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_array.into_pyarray(py)
}

struct MtfParams {
    c0: f32,
    denom_norm: f32,
    m: f32,
}

fn compute_channel_params(
    samples: &mut [f32],
    target_bg: f32,
    shadow_clip_sigma: f32,
) -> MtfParams {
    if samples.is_empty() {
        return MtfParams {
            c0: 0.0,
            denom_norm: 1.0,
            m: 0.5,
        };
    }
    let m_c = compute_median_in_place(samples);
    let mut diffs: Vec<f32> = samples.iter().map(|&v| (v - m_c).abs()).collect();
    let mad = compute_median_in_place(&mut diffs);
    let mut sigma_c = 1.4826 * mad;
    if sigma_c < 1e-6 {
        let sum_sq: f32 = samples.iter().map(|&v| (v - m_c) * (v - m_c)).sum();
        sigma_c = (sum_sq / samples.len().max(1) as f32).sqrt();
    }
    let c0 = if sigma_c < 1e-6 {
        0.0
    } else {
        (m_c - shadow_clip_sigma * sigma_c).clamp(0.0, (m_c - 1e-4).max(0.0))
    };
    let denom_norm = (1.0 - c0).max(1e-7);
    let delta = ((m_c - c0) / denom_norm).clamp(1e-5, 1.0 - 1e-5);
    let denom_m = delta * (1.0 - 2.0 * target_bg) + target_bg;
    let m = if denom_m.abs() > 1e-7 {
        (delta * (1.0 - target_bg)) / denom_m
    } else {
        0.5
    }
    .clamp(1e-5, 1.0 - 1e-5);

    MtfParams { c0, denom_norm, m }
}

#[pyfunction]
#[pyo3(signature = (
    img_starless,
    sky_mask=None,
    target_bg=0.15,
    shadow_clip_sigma=2.0,
    saturation_boost=2.5,
    unlinked=true
))]
pub fn auto_stretch_adaptive_starless_rs<'py>(
    py: Python<'py>,
    img_starless: PyReadonlyArray3<'py, f32>,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
    target_bg: f32,
    shadow_clip_sigma: f32,
    saturation_boost: f32,
    unlinked: bool,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = img_starless.as_array();
    let mask_opt = sky_mask.as_ref().map(|m| m.as_array());
    let (h, w, c) = img_view.dim();

    let (is_all_sky, n_sky) = if let Some(ref m) = mask_opt {
        let mut count = 0usize;
        for y in 0..h {
            for x in 0..w {
                if unsafe { *m.uget((y, x)) } > 0.5 {
                    count += 1;
                }
            }
        }
        if count < 100 {
            (true, h * w)
        } else {
            (false, count)
        }
    } else {
        (true, h * w)
    };

    let stride = (n_sky / 500_000).max(1);

    let params: [MtfParams; 3] = if unlinked {
        let mut p_arr = Vec::with_capacity(3);
        for ch in 0..3 {
            let mut samples = Vec::with_capacity((n_sky / stride) + 1);
            let mut sky_counter = 0usize;
            for y in 0..h {
                for x in 0..w {
                    let is_sky =
                        is_all_sky || unsafe { *mask_opt.as_ref().unwrap().uget((y, x)) } > 0.5;
                    if is_sky {
                        if sky_counter % stride == 0 {
                            samples.push(unsafe { *img_view.uget((y, x, ch)) });
                        }
                        sky_counter += 1;
                    }
                }
            }
            p_arr.push(compute_channel_params(
                &mut samples,
                target_bg,
                shadow_clip_sigma,
            ));
        }
        [
            p_arr.remove(0),
            p_arr.remove(0),
            p_arr.remove(0),
        ]
    } else {
        let total_sky_items = n_sky * 3;
        let linked_stride = (total_sky_items / 500_000).max(1);
        let mut samples = Vec::with_capacity((total_sky_items / linked_stride) + 1);
        let mut sky_counter = 0usize;
        for ch in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let is_sky =
                        is_all_sky || unsafe { *mask_opt.as_ref().unwrap().uget((y, x)) } > 0.5;
                    if is_sky {
                        if sky_counter % linked_stride == 0 {
                            samples.push(unsafe { *img_view.uget((y, x, ch)) });
                        }
                        sky_counter += 1;
                    }
                }
            }
        }
        let common = compute_channel_params(&mut samples, target_bg, shadow_clip_sigma);
        [
            MtfParams { c0: common.c0, denom_norm: common.denom_norm, m: common.m },
            MtfParams { c0: common.c0, denom_norm: common.denom_norm, m: common.m },
            MtfParams { c0: common.c0, denom_norm: common.denom_norm, m: common.m },
        ]
    };

    let out_vec = py.allow_threads(|| {
        let row_stride = w * c;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let base_idx = x * c;
                    let mut stretched = [0.0f32; 3];

                    for ch in 0..3 {
                        let p = &params[ch];
                        let val = unsafe { *img_view.uget((y, x, ch)) };
                        let x_norm = ((val - p.c0) / p.denom_norm).clamp(0.0, 1.0);
                        let num = (p.m - 1.0) * x_norm;
                        let den = (2.0 * p.m - 1.0) * x_norm - p.m;
                        stretched[ch] = if den.abs() > 1e-7 {
                            (num / den).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                    }

                    let mut r = stretched[0];
                    let mut g = stretched[1];
                    let mut b = stretched[2];

                    // Realce de saturación HSV en float32
                    if (saturation_boost - 1.0).abs() > 1e-4 {
                        let v = r.max(g.max(b));
                        let min_c = r.min(g.min(b));
                        let ch_chroma = v - min_c;
                        if v > 1e-7 && ch_chroma > 1e-7 {
                            let c_new = (ch_chroma * saturation_boost).min(v);
                            let ratio = c_new / ch_chroma;
                            r = v - (v - r) * ratio;
                            g = v - (v - g) * ratio;
                            b = v - (v - b) * ratio;
                        }
                    }

                    // Preservación del terreno con sky_mask si se suministró
                    if let Some(ref m) = mask_opt {
                        let mask_val = unsafe { (*m.uget((y, x))).clamp(0.0, 1.0) };
                        let orig_r = unsafe { *img_view.uget((y, x, 0)) };
                        let orig_g = unsafe { *img_view.uget((y, x, 1)) };
                        let orig_b = unsafe { *img_view.uget((y, x, 2)) };
                        r = (r * mask_val + orig_r * (1.0 - mask_val)).clamp(0.0, 1.0);
                        g = (g * mask_val + orig_g * (1.0 - mask_val)).clamp(0.0, 1.0);
                        b = (b * mask_val + orig_b * (1.0 - mask_val)).clamp(0.0, 1.0);
                    }

                    row_slice[base_idx] = r;
                    row_slice[base_idx + 1] = g;
                    row_slice[base_idx + 2] = b;
                }
            });

        buf
    });

    let out_array = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_array.into_pyarray(py)
}

// ============================================================================
// ESTIRADO HIPERBÓLICO GENERALIZADO (GHS) (core/ghs.py) - PRIORIDAD 2
// ============================================================================

#[inline(always)]
fn eval_ghs_q_point(
    v: f64,
    sp_eff: f64,
    d: f64,
    b: f64,
    ds: f64,
    bs: f64,
    hp_thresh: f64,
    lp_thresh: f64,
    q_hp: f64,
    s_hp: f64,
    q_lp: f64,
    s_lp: f64,
) -> f64 {
    let q_raw = |val: f64| -> f64 {
        let diff_lower = (sp_eff - val).max(0.0);
        let diff_upper = (val - sp_eff).max(0.0);
        if b <= 1e-6 {
            if val < sp_eff {
                (-d * diff_lower).exp()
            } else {
                2.0 - (-d * diff_upper).exp()
            }
        } else {
            if val < sp_eff {
                (1.0 + ds * diff_lower).powf(-1.0 / bs)
            } else {
                2.0 - (1.0 + ds * diff_upper).powf(-1.0 / bs)
            }
        }
    };

    let base = q_raw(v);
    let mut res = if v > hp_thresh {
        q_hp + s_hp * (v - hp_thresh)
    } else {
        base
    };
    if v < lp_thresh {
        res = q_lp - s_lp * (lp_thresh - v);
    }
    res
}

#[pyfunction]
#[pyo3(signature = (sp, d, b, hp=0.0, lp=0.0, bp=0.0, lut_size=65536))]
pub fn compute_ghs_lut_rs<'py>(
    py: Python<'py>,
    sp: f32,
    d: f32,
    b: f32,
    hp: f32,
    lp: f32,
    bp: f32,
    lut_size: usize,
) -> Bound<'py, PyArray1<f32>> {
    let mut lut = vec![0.0f32; lut_size];
    let max_idx = (lut_size - 1) as f64;

    let bp_d = bp as f64;
    let sp_d = sp as f64;
    let d_d = (d as f64).max(0.0);
    let b_d = (b as f64).max(0.0);
    let hp_d = (hp as f64).clamp(0.0, 0.99);
    let lp_d = (lp as f64).clamp(0.0, 0.99);

    let sp_eff = if bp_d > 0.0 {
        ((sp_d - bp_d) / (1.0 - bp_d).max(1e-6)).clamp(1e-5, 1.0 - 1e-5)
    } else if bp_d < 0.0 {
        (sp_d - bp_d).clamp(1e-5, 1.0 - 1e-5)
    } else {
        sp_d.clamp(1e-5, 1.0 - 1e-5)
    };

    if d_d <= 1e-6 {
        for i in 0..lut_size {
            let x = i as f64 / max_idx;
            let x_adj = if bp_d > 0.0 {
                ((x - bp_d) / (1.0 - bp_d).max(1e-6)).clamp(0.0, 1.0)
            } else if bp_d < 0.0 {
                (x - bp_d).clamp(0.0, 1.0)
            } else {
                x
            };
            lut[i] = x_adj as f32;
        }
        let arr = Array1::from_vec(lut);
        return arr.into_pyarray(py);
    }

    let hp_thresh = 1.0 - hp_d * (1.0 - sp_eff);
    let lp_thresh = lp_d * sp_eff;
    let ds = if b_d > 0.0 { d_d * b_d.max(1e-6) } else { d_d };
    let bs = b_d.max(1e-6);

    let q_raw_pt = |val: f64| -> f64 {
        let diff_lower = (sp_eff - val).max(0.0);
        let diff_upper = (val - sp_eff).max(0.0);
        if b_d <= 1e-6 {
            if val < sp_eff {
                (-d_d * diff_lower).exp()
            } else {
                2.0 - (-d_d * diff_upper).exp()
            }
        } else {
            if val < sp_eff {
                (1.0 + ds * diff_lower).powf(-1.0 / bs)
            } else {
                2.0 - (1.0 + ds * diff_upper).powf(-1.0 / bs)
            }
        }
    };

    let q_deriv_pt = |val: f64| -> f64 {
        let diff_lower = (sp_eff - val).max(0.0);
        let diff_upper = (val - sp_eff).max(0.0);
        if b_d <= 1e-6 {
            if val < sp_eff {
                d_d * (-d_d * diff_lower).exp()
            } else {
                d_d * (-d_d * diff_upper).exp()
            }
        } else {
            if val < sp_eff {
                d_d * (1.0 + ds * diff_lower).powf(-1.0 / bs - 1.0)
            } else {
                d_d * (1.0 + ds * diff_upper).powf(-1.0 / bs - 1.0)
            }
        }
    };

    let q_hp = q_raw_pt(hp_thresh);
    let s_hp = q_deriv_pt(hp_thresh);
    let q_lp = q_raw_pt(lp_thresh);
    let s_lp = q_deriv_pt(lp_thresh);

    let q0 = eval_ghs_q_point(0.0, sp_eff, d_d, b_d, ds, bs, hp_thresh, lp_thresh, q_hp, s_hp, q_lp, s_lp);
    let q1 = eval_ghs_q_point(1.0, sp_eff, d_d, b_d, ds, bs, hp_thresh, lp_thresh, q_hp, s_hp, q_lp, s_lp);
    let denom = (q1 - q0).max(1e-7);

    for i in 0..lut_size {
        let x = i as f64 / max_idx;
        let x_adj = if bp_d > 0.0 {
            ((x - bp_d) / (1.0 - bp_d).max(1e-6)).clamp(0.0, 1.0)
        } else if bp_d < 0.0 {
            (x - bp_d).clamp(0.0, 1.0)
        } else {
            x
        };
        let qv = eval_ghs_q_point(x_adj, sp_eff, d_d, b_d, ds, bs, hp_thresh, lp_thresh, q_hp, s_hp, q_lp, s_lp);
        let val = ((qv - q0) / denom).clamp(0.0, 1.0);
        lut[i] = val as f32;
    }

    let arr = Array1::from_vec(lut);
    arr.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (img, lut, color_mode="luma"))]
pub fn apply_ghs_stretch_rs<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    lut: PyReadonlyArray1<'py, f32>,
    color_mode: &str,
) -> PyResult<PyObject> {
    let lut_slice = lut.as_slice()?;
    let lut_len = lut_slice.len();
    if lut_len < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("LUT size must be at least 2"));
    }
    let max_idx = (lut_len - 1) as f32;

    if let Ok(img3) = img.extract::<PyReadonlyArray3<f32>>() {
        let view = img3.as_array();
        let (h, w, c) = view.dim();
        let is_luma = color_mode == "luma" && c == 3;

        let out_vec = py.allow_threads(|| {
            let row_stride = w * c;
            let mut buf = vec![0.0f32; h * row_stride];

            buf.par_chunks_mut(row_stride)
                .enumerate()
                .for_each(|(y, row_slice)| {
                    if is_luma {
                        for x in 0..w {
                            let base_idx = x * 3;
                            let r = unsafe { *view.uget((y, x, 0)) };
                            let g = unsafe { *view.uget((y, x, 1)) };
                            let b = unsafe { *view.uget((y, x, 2)) };

                            let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
                            let idx = ((lum * max_idx).clamp(0.0, max_idx)) as usize;
                            let lum_str = unsafe { *lut_slice.get_unchecked(idx.min(lut_len - 1)) };
                            let ratio = (lum_str + 1e-6) / (lum + 1e-6);

                            row_slice[base_idx] = (r * ratio).clamp(0.0, 1.0);
                            row_slice[base_idx + 1] = (g * ratio).clamp(0.0, 1.0);
                            row_slice[base_idx + 2] = (b * ratio).clamp(0.0, 1.0);
                        }
                    } else {
                        for x in 0..w {
                            let base_idx = x * c;
                            for ch in 0..c {
                                let val = unsafe { *view.uget((y, x, ch)) };
                                let idx = ((val * max_idx).clamp(0.0, max_idx)) as usize;
                                let str_val = unsafe { *lut_slice.get_unchecked(idx.min(lut_len - 1)) };
                                row_slice[base_idx + ch] = str_val;
                            }
                        }
                    }
                });

            buf
        });

        let arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
        Ok(arr.into_pyarray(py).into_any().unbind())
    } else if let Ok(img2) = img.extract::<PyReadonlyArray2<f32>>() {
        let view = img2.as_array();
        let (h, w) = view.dim();

        let out_vec = py.allow_threads(|| {
            let mut buf = vec![0.0f32; h * w];

            buf.par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, row_slice)| {
                    for x in 0..w {
                        let val = unsafe { *view.uget((y, x)) };
                        let idx = ((val * max_idx).clamp(0.0, max_idx)) as usize;
                        row_slice[x] = unsafe { *lut_slice.get_unchecked(idx.min(lut_len - 1)) };
                    }
                });

            buf
        });

        let arr = Array2::from_shape_vec((h, w), out_vec).unwrap();
        Ok(arr.into_pyarray(py).into_any().unbind())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected 2D or 3D numpy float32 array",
        ))
    }
}

// ============================================================================
// EFECTOS ÓPTICOS: ORTON Y DIFUSOR ESTELAR (core/orton.py, core/star_diffusion.py)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (glow_base, diffuse_luma, threshold=0.05, exponent=1.0))]
pub fn compute_orton_glow_source_rs<'py>(
    py: Python<'py>,
    glow_base: PyReadonlyArray3<'py, f32>,
    diffuse_luma: PyReadonlyArray2<'py, f32>,
    threshold: f32,
    exponent: f32,
) -> Bound<'py, PyArray3<f32>> {
    let base_view = glow_base.as_array();
    let luma_view = diffuse_luma.as_array();
    let (h, w, c) = base_view.dim();

    let thr = threshold.max(0.0);
    let knee = (thr * 0.5).max(0.04);
    let exp_active = (exponent - 1.0).abs() > 1e-3;
    let exp_val = exponent.max(0.2);

    let out_vec = py.allow_threads(|| {
        let row_stride = w * c;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let base_idx = x * c;
                    let lum = unsafe { *luma_view.uget((y, x)) };
                    let t = ((lum - thr) / knee).clamp(0.0, 1.0);
                    let mut em = t * t * (3.0 - 2.0 * t);
                    if exp_active {
                        em = em.powf(exp_val);
                    }

                    for ch in 0..c {
                        let bv = unsafe { *base_view.uget((y, x, ch)) };
                        row_slice[base_idx + ch] = bv * em;
                    }
                }
            });

        buf
    });

    let arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    arr.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (
    img_rgb,
    bloom,
    tint=None,
    sky_mask=None,
    blend_mode="soft_light",
    strength=1.0
))]
pub fn blend_orton_and_diffusion_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    bloom: PyReadonlyArray3<'py, f32>,
    tint: Option<PyReadonlyArray1<'py, f32>>,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
    blend_mode: &str,
    strength: f32,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = img_rgb.as_array();
    let b_view = bloom.as_array();
    let mask_view = sky_mask.as_ref().map(|m| m.as_array());
    let (h, w, c) = img_view.dim();

    let t_factors: [f32; 3] = if let Some(ref t) = tint {
        let ts = t.as_slice().unwrap();
        if ts.len() >= 3 {
            [ts[0], ts[1], ts[2]]
        } else {
            [1.0, 1.0, 1.0]
        }
    } else {
        [1.0, 1.0, 1.0]
    };

    let mode = match blend_mode {
        "screen" => 1,
        "additive" => 2,
        _ => 0, // soft_light
    };

    let out_vec = py.allow_threads(|| {
        let row_stride = w * c;
        let mut buf = vec![0.0f32; h * row_stride];

        buf.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let base_idx = x * c;
                    let m_val = if let Some(ref m) = mask_view {
                        unsafe { *m.uget((y, x)) }
                    } else {
                        1.0
                    };

                    for ch in 0..3 {
                        let i_val = unsafe { *img_view.uget((y, x, ch)) };
                        let b_raw = unsafe { *b_view.uget((y, x, ch)) };
                        let b_val = b_raw * t_factors[ch] * m_val;

                        let blended = match mode {
                            1 => 1.0 - (1.0 - i_val) * (1.0 - b_val.clamp(0.0, 1.0)),
                            2 => i_val + b_val,
                            _ => i_val + 2.0 * b_val * i_val * (1.0 - i_val),
                        };

                        let final_val = (1.0 - strength) * i_val + strength * blended;
                        row_slice[base_idx + ch] = final_val.clamp(0.0, 1.0);
                    }
                }
            });

        buf
    });

    let arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    arr.into_pyarray(py)
}
