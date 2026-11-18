use crate::stacking::compute_median_in_place;
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn, PyReadwriteArray2,
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

// ============================================================================
// MÁSCARAS VECTORIALES ANALÍTICAS (GRADIENTE LINEAL Y RADIAL)
// ============================================================================

#[inline(always)]
fn hermite_smoothstep(t: f32) -> f32 {
    let t_c = t.clamp(0.0, 1.0);
    1.0 - (3.0 * t_c * t_c - 2.0 * t_c * t_c * t_c)
}

#[pyfunction]
#[pyo3(signature = (
    h,
    w,
    x0,
    y0,
    x1,
    y1,
    midpoint=0.5,
    profile="smoothstep",
    span_b=1.0,
    crop_rect=None,
    full_shape=None
))]
pub fn compute_linear_gradient_rs<'py>(
    py: Python<'py>,
    h: usize,
    w: usize,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    midpoint: f32,
    profile: &str,
    span_b: f32,
    crop_rect: Option<(usize, usize, usize, usize)>,
    full_shape: Option<(usize, usize)>,
) -> Bound<'py, PyArray2<f32>> {
    let (x_offset, y_offset, fw, fh) = match (crop_rect, full_shape) {
        (Some((cx1, cy1, _, _)), Some((fh_val, fw_val))) if fw_val > 0 && fh_val > 0 => {
            (cx1 as f32, cy1 as f32, fw_val as f32, fh_val as f32)
        }
        _ => (0.0, 0.0, w as f32, h as f32),
    };

    let aspect = if fh > 0.0 { fw / fh } else { 1.0 };
    let vx = (x1 - x0) * aspect;
    let vy = y1 - y0;
    let len_sq = vx * vx + vy * vy;
    let mid = midpoint.clamp(0.01, 0.99);

    let is_linear = profile == "linear";
    let is_exponential = profile == "exponential";
    let is_reflected = profile == "reflected";
    let span_b_safe = span_b.max(0.01);
    let inv_span_b = 1.0 / span_b_safe;

    let out_vec = py.allow_threads(|| {
        let mut buf = vec![0.0f32; h * w];

        if len_sq < 1e-7 {
            // Vector nulo: gradiente degenerado
            return buf;
        }

        let inv_len_sq = 1.0 / len_sq;

        buf.par_chunks_mut(w)
            .enumerate()
            .for_each(|(y_idx, row_slice)| {
                let py_coord = (y_idx as f32 + y_offset) / fh.max(1.0);
                let dy = py_coord - y0;
                let dy_vy = dy * vy;

                for x_idx in 0..w {
                    let px_coord = (x_idx as f32 + x_offset) / fw.max(1.0);
                    let dx = (px_coord - x0) * aspect;
                    let proj = dx * vx + dy_vy;

                    let val = if is_reflected {
                        let t = if proj >= 0.0 {
                            (proj * inv_len_sq).clamp(0.0, 1.0)
                        } else {
                            (proj.abs() * inv_len_sq * inv_span_b).clamp(0.0, 1.0)
                        };
                        hermite_smoothstep(t)
                    } else {
                        let mut t = (proj * inv_len_sq).clamp(0.0, 1.0);

                        if (mid - 0.5).abs() > 1e-4 {
                            if t < mid {
                                t = 0.5 * (t / mid);
                            } else {
                                t = 0.5 + 0.5 * ((t - mid) / (1.0 - mid));
                            }
                        }

                        if is_linear {
                            1.0 - t
                        } else if is_exponential {
                            (1.0 - t) * (1.0 - t)
                        } else {
                            hermite_smoothstep(t)
                        }
                    };

                    row_slice[x_idx] = val.clamp(0.0, 1.0);
                }
            });

        buf
    });

    let arr = Array2::from_shape_vec((h, w), out_vec).unwrap();
    arr.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (
    h,
    w,
    cx,
    cy,
    rx,
    ry,
    angle_deg=0.0,
    feather=0.5,
    crop_rect=None,
    full_shape=None
))]
pub fn compute_radial_gradient_rs<'py>(
    py: Python<'py>,
    h: usize,
    w: usize,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    angle_deg: f32,
    feather: f32,
    crop_rect: Option<(usize, usize, usize, usize)>,
    full_shape: Option<(usize, usize)>,
) -> Bound<'py, PyArray2<f32>> {
    let (x_offset, y_offset, fw, fh) = match (crop_rect, full_shape) {
        (Some((cx1, cy1, _, _)), Some((fh_val, fw_val))) if fw_val > 0 && fh_val > 0 => {
            (cx1 as f32, cy1 as f32, fw_val as f32, fh_val as f32)
        }
        _ => (0.0, 0.0, w as f32, h as f32),
    };

    let aspect = if fh > 0.0 { fw / fh } else { 1.0 };
    let inv_aspect = 1.0 / aspect.max(1e-5);
    let rx_safe = rx.abs().max(1e-5);
    let ry_safe = ry.abs().max(1e-5);
    let feather_safe = feather.clamp(0.01, 1.0);
    let d_in = (1.0 - feather_safe).max(0.0);
    let inv_feather = 1.0 / feather_safe;

    let rad = angle_deg.to_radians();
    let cos_t = rad.cos();
    let sin_t = rad.sin();

    let out_vec = py.allow_threads(|| {
        let mut buf = vec![0.0f32; h * w];

        buf.par_chunks_mut(w)
            .enumerate()
            .for_each(|(y_idx, row_slice)| {
                let py_coord = (y_idx as f32 + y_offset) / fh.max(1.0);
                let dy = py_coord - cy;

                for x_idx in 0..w {
                    let px_coord = (x_idx as f32 + x_offset) / fw.max(1.0);
                    let dx = px_coord - cx;

                    // Rotación métrica en el plano de píxeles (aspect-ratio aware)
                    let x_rot = dx * cos_t + (dy * inv_aspect) * sin_t;
                    let y_rot = -(dx * aspect) * sin_t + dy * cos_t;

                    let nx = x_rot / rx_safe;
                    let ny = y_rot / ry_safe;
                    let d = (nx * nx + ny * ny).sqrt();

                    let val = if d <= d_in {
                        1.0
                    } else if d >= 1.0 {
                        0.0
                    } else {
                        let t = (d - d_in) * inv_feather;
                        hermite_smoothstep(t)
                    };

                    row_slice[x_idx] = val.clamp(0.0, 1.0);
                }
            });

        buf
    });

    let arr = Array2::from_shape_vec((h, w), out_vec).unwrap();
    arr.into_pyarray(py)
}

// ============================================================================
// GENERADORES NATIVOS DE MÁSCARAS ANALÍTICAS (Fase 3: Luminancia y Rango Color)
// ============================================================================

#[inline(always)]
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let diff = (edge1 - edge0).max(1e-6);
    let t = ((x - edge0) / diff).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
fn rgb_to_hue_sat(r: f32, g: f32, b: f32) -> (f32, f32) {
    let v = r.max(g.max(b));
    let min_c = r.min(g.min(b));
    let c = v - min_c;
    if v <= 1e-7 || c <= 1e-7 {
        return (0.0, 0.0);
    }
    let s = c / v;
    let mut h = if (v - r).abs() < 1e-7 {
        60.0 * (g - b) / c
    } else if (v - g).abs() < 1e-7 {
        120.0 + 60.0 * (b - r) / c
    } else {
        240.0 + 60.0 * (r - g) / c
    };
    if h < 0.0 {
        h += 360.0;
    } else if h >= 360.0 {
        h -= 360.0;
    }
    (h, s)
}

#[pyfunction]
#[pyo3(signature = (
    image,
    luma_min=0.0,
    luma_max=1.0,
    feather=0.08,
    invert=false
))]
pub fn generate_luminance_mask_rs<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, f32>,
    luma_min: f32,
    luma_max: f32,
    feather: f32,
    invert: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let img_view = image.as_array();
    let f = feather.max(1e-4);
    let l_min = luma_min.min(luma_max);
    let l_max = luma_min.max(luma_max);
    let ndim = img_view.ndim();

    if ndim == 3 {
        let view3 = img_view.into_dimensionality::<numpy::ndarray::Ix3>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid 3D image shape: {}", e))
        })?;
        let (h, w, c) = view3.dim();
        let out_vec = py.allow_threads(|| {
            let mut buf = vec![0.0f32; h * w];

            buf.par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, row_slice)| {
                    for x in 0..w {
                        let luma = if c >= 3 {
                            let r = unsafe { *view3.uget((y, x, 0)) };
                            let g = unsafe { *view3.uget((y, x, 1)) };
                            let b = unsafe { *view3.uget((y, x, 2)) };
                            0.2126 * r + 0.7152 * g + 0.0722 * b
                        } else {
                            unsafe { *view3.uget((y, x, 0)) }.clamp(0.0, 1.0)
                        };

                        let low_ramp = if l_min > 0.0 {
                            smoothstep_f32(l_min - f, l_min, luma)
                        } else {
                            1.0
                        };
                        let high_ramp = if l_max < 1.0 {
                            1.0 - smoothstep_f32(l_max, l_max + f, luma)
                        } else {
                            1.0
                        };

                        let mut val = (low_ramp * high_ramp).clamp(0.0, 1.0);
                        if invert {
                            val = 1.0 - val;
                        }
                        row_slice[x] = val;
                    }
                });

            buf
        });

        let arr = Array2::from_shape_vec((h, w), out_vec).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to reshape output mask: {}", e))
        })?;
        Ok(arr.into_pyarray(py))
    } else if ndim == 2 {
        let view2 = img_view.into_dimensionality::<numpy::ndarray::Ix2>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid 2D image shape: {}", e))
        })?;
        let (h, w) = view2.dim();
        let out_vec = py.allow_threads(|| {
            let mut buf = vec![0.0f32; h * w];

            buf.par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, row_slice)| {
                    for x in 0..w {
                        let luma = unsafe { *view2.uget((y, x)) }.clamp(0.0, 1.0);

                        let low_ramp = if l_min > 0.0 {
                            smoothstep_f32(l_min - f, l_min, luma)
                        } else {
                            1.0
                        };
                        let high_ramp = if l_max < 1.0 {
                            1.0 - smoothstep_f32(l_max, l_max + f, luma)
                        } else {
                            1.0
                        };

                        let mut val = (low_ramp * high_ramp).clamp(0.0, 1.0);
                        if invert {
                            val = 1.0 - val;
                        }
                        row_slice[x] = val;
                    }
                });

            buf
        });

        let arr = Array2::from_shape_vec((h, w), out_vec).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to reshape output mask: {}", e))
        })?;
        Ok(arr.into_pyarray(py))
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Image must be 2D or 3D float32 array",
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (
    image,
    target_hue_deg,
    hue_tolerance_deg=30.0,
    min_saturation=0.08,
    feather=0.08,
    invert=false
))]
pub fn generate_color_range_mask_rs<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, f32>,
    target_hue_deg: f32,
    hue_tolerance_deg: f32,
    min_saturation: f32,
    feather: f32,
    invert: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let img_view = image.as_array();
    let (h, w, c) = img_view.dim();
    if c < 3 {
        let arr = Array2::ones((h, w));
        return Ok(arr.into_pyarray(py));
    }

    let target_h = target_hue_deg.rem_euclid(360.0);
    let tol = hue_tolerance_deg.max(1.0);
    let f = feather.max(0.01);
    let fade_deg = f * 45.0;
    let s_min = min_saturation.max(0.01);
    let s_edge0 = (s_min - f * 0.15).max(0.0);

    let out_vec = py.allow_threads(|| {
        let mut buf = vec![0.0f32; h * w];

        buf.par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..w {
                    let r = unsafe { *img_view.uget((y, x, 0)) }.clamp(0.0, 1.0);
                    let g = unsafe { *img_view.uget((y, x, 1)) }.clamp(0.0, 1.0);
                    let b = unsafe { *img_view.uget((y, x, 2)) }.clamp(0.0, 1.0);

                    let (hue, sat) = rgb_to_hue_sat(r, g, b);

                    let diff = (hue - target_h).abs();
                    let angular_dist = diff.min(360.0 - diff);

                    let hue_mask = 1.0 - smoothstep_f32(tol, tol + fade_deg, angular_dist);
                    let sat_ramp = smoothstep_f32(s_edge0, s_min, sat);

                    let mut val = (hue_mask * sat_ramp).clamp(0.0, 1.0);
                    if invert {
                        val = 1.0 - val;
                    }
                    row_slice[x] = val;
                }
            });

        buf
    });

    let arr = Array2::from_shape_vec((h, w), out_vec).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to reshape output mask: {}", e))
    })?;
    Ok(arr.into_pyarray(py))
}

// ============================================================================
// CURVAS SELECTIVAS HUE/LUMA FUSIONADAS EN REGISTROS (core/stretch.py)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    img_rgb,
    lut_hue_sat=None,
    lut_luma_sat=None,
    sky_mask=None
))]
pub fn apply_hue_luma_curves_fused_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    lut_hue_sat: Option<PyReadonlyArray1<'py, f32>>,
    lut_luma_sat: Option<PyReadonlyArray1<'py, f32>>,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
) -> Bound<'py, PyArray3<f32>> {
    let img_view = img_rgb.as_array();
    let (h, w, c) = img_view.dim();

    let hue_lut: Option<Vec<f32>> = lut_hue_sat.as_ref().and_then(|l| {
        let s = l.as_slice().ok()?;
        if s.len() == 256 && s.iter().any(|&v| (v - 0.5).abs() > 1e-3) {
            Some(s.to_vec())
        } else {
            None
        }
    });

    let luma_lut: Option<Vec<f32>> = lut_luma_sat.as_ref().and_then(|l| {
        let s = l.as_slice().ok()?;
        if s.len() == 256 && s.iter().any(|&v| (v - 0.5).abs() > 1e-3) {
            Some(s.to_vec())
        } else {
            None
        }
    });

    if hue_lut.is_none() && luma_lut.is_none() {
        return img_view.to_owned().into_pyarray(py);
    }

    let mask_view = sky_mask.as_ref().map(|m| m.as_array());

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

                    let v = r0.max(g0.max(b0));
                    let min_c = r0.min(g0.min(b0));
                    let chroma = v - min_c;

                    if v <= 1e-7 || chroma <= 1e-7 {
                        row_slice[base_idx] = r0;
                        row_slice[base_idx + 1] = g0;
                        row_slice[base_idx + 2] = b0;
                        continue;
                    }

                    let (hue, _) = rgb_to_hue_sat(r0, g0, b0);
                    let mut mult = 1.0f32;
                    if let Some(ref hl) = hue_lut {
                        let h_idx = ((hue / 360.0) * 255.0).clamp(0.0, 255.0) as usize;
                        mult *= hl[h_idx] * 2.0;
                    }
                    if let Some(ref ll) = luma_lut {
                        let luma = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;
                        let l_idx = (luma * 255.0).clamp(0.0, 255.0) as usize;
                        mult *= ll[l_idx] * 2.0;
                    }

                    let c_new = (chroma * mult).min(v);
                    let ratio = c_new / chroma;
                    let mut r = (v - (v - r0) * ratio).clamp(0.0, 1.0);
                    let mut g = (v - (v - g0) * ratio).clamp(0.0, 1.0);
                    let mut b = (v - (v - b0) * ratio).clamp(0.0, 1.0);

                    if let Some(ref m_view) = mask_view {
                        let m = unsafe { *m_view.uget((y, x)) }.clamp(0.0, 1.0);
                        r = r * m + r0 * (1.0 - m);
                        g = g * m + g0 * (1.0 - m);
                        b = b * m + b0 * (1.0 - m);
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
// ESTAMPADO CONTINUO DE PINCEL POR ROI (gui/canvas.py)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    mask,
    p0,
    p1,
    radius,
    hardness=0.5,
    flow=0.5,
    erase=false,
    stroke_buf=None,
    stroke_orig=None
))]
pub fn stamp_brush_stroke_rs<'py>(
    py: Python<'py>,
    mut mask: PyReadwriteArray2<'py, f32>,
    p0: (f32, f32),
    p1: (f32, f32),
    radius: f32,
    hardness: f32,
    flow: f32,
    erase: bool,
    mut stroke_buf: Option<PyReadwriteArray2<'py, f32>>,
    stroke_orig: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Option<(usize, usize, usize, usize)>> {
    let (h, w) = mask.as_array().dim();
    if h == 0 || w == 0 {
        return Ok(None);
    }

    let r = radius.max(1.0);
    let bx0 = ((p0.0.min(p1.0) - r).floor() as isize).clamp(0, w as isize) as usize;
    let bx1 = ((p0.0.max(p1.0) + r + 1.0).ceil() as isize).clamp(0, w as isize) as usize;
    let by0 = ((p0.1.min(p1.1) - r).floor() as isize).clamp(0, h as isize) as usize;
    let by1 = ((p0.1.max(p1.1) + r + 1.0).ceil() as isize).clamp(0, h as isize) as usize;

    if bx0 >= bx1 || by0 >= by1 {
        return Ok(None);
    }

    let mask_slice = mask.as_slice_mut().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("mask must be C-contiguous: {:?}", e))
    })?;

    let mut buf_slice_opt = if let Some(ref mut sb) = stroke_buf {
        Some(sb.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("stroke_buf must be C-contiguous: {:?}", e))
        })?)
    } else {
        None
    };

    let orig_slice_opt = if let Some(ref so) = stroke_orig {
        Some(so.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("stroke_orig must be C-contiguous: {:?}", e))
        })?)
    } else {
        None
    };

    py.allow_threads(|| {
        let vx = p1.0 - p0.0;
        let vy = p1.1 - p0.1;
        let v_sq = vx * vx + vy * vy;
        let h_val = hardness.clamp(0.0, 0.999);
        let r_sq = r * r;
        let inv_r = 1.0 / r;
        let inv_feather = 1.0 / (1.0 - h_val).max(1e-4);
        let f = flow.clamp(0.01, 1.0);

        if let (Some(buf_sl), Some(orig_sl)) = (buf_slice_opt.as_deref_mut(), orig_slice_opt) {
            for y in by0..by1 {
                let row_offset = y * w;
                let py_val = y as f32 - p0.1;
                for x in bx0..bx1 {
                    let px = x as f32 - p0.0;
                    let t = if v_sq > 1e-6 {
                        ((px * vx + py_val * vy) / v_sq).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let dx = px - t * vx;
                    let dy = py_val - t * vy;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq <= r_sq {
                        let dist_norm = dist_sq.sqrt() * inv_r;
                        let weight = if dist_norm <= h_val {
                            1.0
                        } else {
                            let t_f = (dist_norm - h_val) * inv_feather;
                            1.0 - (3.0 * t_f * t_f - 2.0 * t_f * t_f * t_f)
                        };

                        let idx = row_offset + x;
                        let b_val = &mut buf_sl[idx];
                        if weight > *b_val {
                            *b_val = weight;
                        }
                        let eff = *b_val * f;
                        let orig = orig_sl[idx];
                        mask_slice[idx] = if erase {
                            (orig - eff).clamp(0.0, 1.0)
                        } else {
                            (orig + eff * (1.0 - orig)).clamp(0.0, 1.0)
                        };
                    }
                }
            }
        } else {
            for y in by0..by1 {
                let row_offset = y * w;
                let py_val = y as f32 - p0.1;
                for x in bx0..bx1 {
                    let px = x as f32 - p0.0;
                    let t = if v_sq > 1e-6 {
                        ((px * vx + py_val * vy) / v_sq).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let dx = px - t * vx;
                    let dy = py_val - t * vy;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq <= r_sq {
                        let dist_norm = dist_sq.sqrt() * inv_r;
                        let weight = if dist_norm <= h_val {
                            1.0
                        } else {
                            let t_f = (dist_norm - h_val) * inv_feather;
                            1.0 - (3.0 * t_f * t_f - 2.0 * t_f * t_f * t_f)
                        };

                        let idx = row_offset + x;
                        let eff = weight * f;
                        let m = mask_slice[idx];
                        mask_slice[idx] = if erase {
                            (m - eff).clamp(0.0, 1.0)
                        } else {
                            (m + eff * (1.0 - m)).clamp(0.0, 1.0)
                        };
                    }
                }
            }
        }
    });

    Ok(Some((bx0, by0, bx1, by1)))
}

// ============================================================================
// EVALUACIÓN FUSIONADA DEL GRAFO PROXY EN TIEMPO REAL (60 FPS FIJOS)
// Fusión de WB + Unpurple + SCNR + Saturación/Vibranza + MTF + Contraste
// ============================================================================
#[pyfunction]
#[pyo3(signature = (
    img_rgb,
    temp_val=0.0,
    tint_val=0.0,
    scnr_amount=0.0,
    unpurple_amount=0.0,
    sat_sky=1.0,
    sat_gnd=1.0,
    vibrance=0.0,
    contrast=0.0,
    contrast_pivot=0.20,
    mtf_mid=None,
    mtf_bp=0.0,
    sky_mask=None
))]
pub fn eval_developer_proxy_fused_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    temp_val: f32,
    tint_val: f32,
    scnr_amount: f32,
    unpurple_amount: f32,
    sat_sky: f32,
    sat_gnd: f32,
    vibrance: f32,
    contrast: f32,
    contrast_pivot: f32,
    mtf_mid: Option<f32>,
    mtf_bp: f32,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
) -> Bound<'py, PyArray3<f32>> {
    let img = img_rgb.as_array();
    let (h, w, c) = img.dim();

    let mask_arr = sky_mask.as_ref().map(|m| m.as_array());

    let out_vec = py.allow_threads(|| {
        let mut buf = vec![0.0f32; h * w * 3];

        let r_gain = 1.0 + (temp_val * 0.5) + (tint_val * 0.25);
        let g_gain = 1.0 - (tint_val * 0.5);
        let b_gain = 1.0 - (temp_val * 0.5) + (tint_val * 0.25);

        let amt_scnr = scnr_amount.clamp(0.0, 1.0);
        let amt_unpurple = unpurple_amount.clamp(0.0, 1.0);

        buf.par_chunks_mut(w * 3).enumerate().for_each(|(y, row_slice)| {
            for x in 0..w {
                let mut r = unsafe { *img.uget((y, x, 0)) }.clamp(0.0, 1.0);
                let mut g = unsafe { *img.uget((y, x, 1)) }.clamp(0.0, 1.0);
                let mut b = unsafe { *img.uget((y, x, 2)) }.clamp(0.0, 1.0);

                // 1. Balance de blancos
                if (temp_val.abs() > 1e-5) || (tint_val.abs() > 1e-5) {
                    r = (r * r_gain).clamp(0.0, 1.0);
                    g = (g * g_gain).clamp(0.0, 1.0);
                    b = (b * b_gain).clamp(0.0, 1.0);
                }

                // 2. Unpurple / Atenuación de halos magenta
                if amt_unpurple > 1e-4 {
                    let mut fringe = (r - g).min(b - g).max(0.0);
                    if fringe > 0.0 {
                        let ha_dom = ((r - b) / (r + 1e-4)).clamp(0.0, 1.0);
                        fringe *= 1.0 - ha_dom;
                        r = (r - amt_unpurple * fringe * 0.75).clamp(0.0, 1.0);
                        b = (b - amt_unpurple * fringe).clamp(0.0, 1.0);
                    }
                }

                // 3. SCNR Verde
                if amt_scnr > 1e-4 {
                    let neutral = 0.5 * (r + b);
                    let excess = (g - neutral).max(0.0);
                    if excess > 0.0 {
                        let g_corr = (g - amt_scnr * excess).max(0.0);
                        let y_orig = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                        let y_corr = 0.2126 * r + 0.7152 * g_corr + 0.0722 * b;
                        let s = if y_corr > 1e-6 { (y_orig / y_corr).min(1.5) } else { 1.0 };
                        r = (r * s).clamp(0.0, 1.0);
                        g = (g_corr * s).clamp(0.0, 1.0);
                        b = (b * s).clamp(0.0, 1.0);
                    }
                }

                // 4. Saturación diferencial cielo / suelo y Vibranza
                let effective_sat = if let Some(ref m) = mask_arr {
                    let sky_w = unsafe { *m.uget((y, x)) }.clamp(0.0, 1.0);
                    sat_sky * sky_w + sat_gnd * (1.0 - sky_w)
                } else {
                    sat_sky
                };

                if (effective_sat - 1.0).abs() > 1e-4 || vibrance.abs() > 1e-4 {
                    adjust_sat_and_vibrance(&mut r, &mut g, &mut b, effective_sat, vibrance);
                }

                // 5. Estirado MTF (si está activo)
                if let Some(m_param) = mtf_mid {
                    let m_clamped = m_param.clamp(0.0001, 0.9999);
                    let denom_norm = (1.0 - mtf_bp).max(1e-6);
                    let r_norm = ((r - mtf_bp) / denom_norm).clamp(0.0, 1.0);
                    let g_norm = ((g - mtf_bp) / denom_norm).clamp(0.0, 1.0);
                    let b_norm = ((b - mtf_bp) / denom_norm).clamp(0.0, 1.0);

                    #[inline(always)]
                    fn mtf(v: f32, m: f32) -> f32 {
                        let denom = (2.0 * m - 1.0) * v - m;
                        if denom.abs() < 1e-7 { v } else { ((m - 1.0) * v) / denom }
                    }

                    r = mtf(r_norm, m_clamped).clamp(0.0, 1.0);
                    g = mtf(g_norm, m_clamped).clamp(0.0, 1.0);
                    b = mtf(b_norm, m_clamped).clamp(0.0, 1.0);
                }

                // 6. Contraste
                if contrast.abs() > 1e-4 {
                    r = adjust_contrast_val(r, contrast, contrast_pivot).clamp(0.0, 1.0);
                    g = adjust_contrast_val(g, contrast, contrast_pivot).clamp(0.0, 1.0);
                    b = adjust_contrast_val(b, contrast, contrast_pivot).clamp(0.0, 1.0);
                }

                let idx = x * 3;
                row_slice[idx] = r;
                row_slice[idx + 1] = g;
                row_slice[idx + 2] = b;
            }
        });

        buf
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).unwrap();
    out_arr.into_pyarray(py)
}




