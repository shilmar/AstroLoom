use std::fs::File;
use std::io::Read;
use numpy::ndarray::{Array3, ArrayD, IxDyn};
use numpy::{
    IntoPyArray, PyArray3, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
    PyReadwriteArray2, PyReadwriteArray3,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

// ============================================================================
// FASE 10: CALIBRACIÓN PURA CFA uint16, STARTRAILS E ICC BRADFORD
// (core/stacking.py, core/startrails.py y core/icc_manager.py)
// ============================================================================

// ----------------------------------------------------------------------------
// 1. Calibración RAW CFA uint16 In-Place (0 bytes de asignación en RAM)
// ----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (raw_cfa_u16, master_dark=None, master_flat=None, master_bias=None, black_level=0.0, dark_pedestal=0.0))]
pub fn calibrate_raw_cfa_inplace_rs(
    py: Python<'_>,
    mut raw_cfa_u16: PyReadwriteArray2<u16>,
    master_dark: Option<PyReadonlyArray2<f32>>,
    master_flat: Option<PyReadonlyArray2<f32>>,
    master_bias: Option<PyReadonlyArray2<f32>>,
    black_level: f32,
    dark_pedestal: f32,
) -> PyResult<()> {
    let (_h, w) = raw_cfa_u16.as_array().dim();
    let cfa_slice = raw_cfa_u16.as_slice_mut().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("raw_cfa_u16 must be C-contiguous: {:?}", e))
    })?;

    let dark_slice = master_dark.as_ref().and_then(|d| d.as_slice().ok());
    let flat_slice = master_flat.as_ref().and_then(|f| f.as_slice().ok());
    let bias_slice = master_bias.as_ref().and_then(|b| b.as_slice().ok());

    py.allow_threads(|| {
        cfa_slice
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(r, row)| {
                let row_offset = r * w;
                for c in 0..w {
                    let idx = row_offset + c;
                    let mut val = row[c] as f32;

                    // 1. Sustracción térmica / píxeles calientes
                    if let Some(dark) = dark_slice {
                        let d = dark[idx];
                        val -= d - dark_pedestal;
                    } else if let Some(bias) = bias_slice {
                        let b = bias[idx];
                        val -= b - dark_pedestal;
                    }

                    // 2. Corrección de Flat en espacio de fotositos CFA
                    if let Some(flat) = flat_slice {
                        let f = flat[idx];
                        let safe_flat = if f > 1e-4 { f } else { 1.0 };
                        let sig = val - black_level;
                        val = sig / safe_flat + black_level;
                    }

                    row[c] = val.round().clamp(0.0, 65535.0) as u16;
                }
            });
    });

    Ok(())
}

// ----------------------------------------------------------------------------
// 2. Acumulación Progresiva de Startrails y Efecto Cometa (core/startrails.py)
// ----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (sky_acc, frame_img, weight=1.0, prev_dilated=None, curr_dilated=None, w_bridge=1.0))]
pub fn accumulate_startrail_step_rs(
    py: Python<'_>,
    mut sky_acc: PyReadwriteArray3<f32>,
    frame_img: PyReadonlyArray3<f32>,
    weight: f32,
    prev_dilated: Option<PyReadonlyArray3<f32>>,
    curr_dilated: Option<PyReadonlyArray3<f32>>,
    w_bridge: f32,
) -> PyResult<()> {
    let acc_slice = sky_acc.as_slice_mut().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("sky_acc must be C-contiguous: {:?}", e))
    })?;
    let frame_slice = frame_img.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("frame_img must be C-contiguous: {:?}", e))
    })?;

    let pd_slice = prev_dilated.as_ref().and_then(|p| p.as_slice().ok());
    let cd_slice = curr_dilated.as_ref().and_then(|c| c.as_slice().ok());

    py.allow_threads(|| {
        // Bloque de trabajo óptimo para saturar la caché L1/L2 (16.384 floats = 64 KB).
        // Mantiene los streams de datos simultáneos dentro de la caché rápida de la CPU
        // eliminando el cache-thrashing contra la memoria RAM principal.
        const CHUNK_SIZE: usize = 16384;

        if let (Some(pd), Some(cd)) = (pd_slice, cd_slice) {
            if pd.len() == acc_slice.len() && cd.len() == acc_slice.len() {
                acc_slice
                    .par_chunks_mut(CHUNK_SIZE)
                    .zip(frame_slice.par_chunks(CHUNK_SIZE))
                    .zip(pd.par_chunks(CHUNK_SIZE))
                    .zip(cd.par_chunks(CHUNK_SIZE))
                    .for_each(|(((acc_c, frame_c), pd_c), cd_c)| {
                        // Bucle completamente libre de saltos (branchless) para auto-vectorización SIMD
                        // (emisión directa de vmulps, vminps, vcmpgeps, vmaxps en AVX2/AVX-512)
                        for i in 0..acc_c.len() {
                            let f_val = frame_c[i] * weight;
                            let m_val = pd_c[i].min(cd_c[i]);
                            let b_val = if m_val >= 0.005 { m_val * w_bridge } else { 0.0 };
                            acc_c[i] = acc_c[i].max(f_val).max(b_val);
                        }
                    });
                return;
            }
        }

        // Vía rápida sin puente de estrellas (solo acumulación directa ponderada)
        acc_slice
            .par_chunks_mut(CHUNK_SIZE)
            .zip(frame_slice.par_chunks(CHUNK_SIZE))
            .for_each(|(acc_c, frame_c)| {
                for i in 0..acc_c.len() {
                    let f_val = frame_c[i] * weight;
                    acc_c[i] = acc_c[i].max(f_val);
                }
            });
    });

    Ok(())
}

// ----------------------------------------------------------------------------
// 3. Recuperación y Realce de Color Estelar (Star Color Recovery & Boost)
// ----------------------------------------------------------------------------

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

#[inline(always)]
fn reflect_101(mut idx: isize, len: usize) -> usize {
    let n = len as isize;
    if n <= 1 {
        return 0;
    }
    loop {
        if idx < 0 {
            idx = -idx;
        } else if idx >= n {
            idx = 2 * n - 2 - idx;
        } else {
            break;
        }
    }
    idx as usize
}

fn gaussian_blur_separable_2d(
    src: &[f32],
    w: usize,
    h: usize,
    k_size: usize,
    sigma: f32,
) -> Vec<f32> {
    let radius = (k_size / 2) as isize;
    let mut kernel = Vec::with_capacity(k_size);
    let mut sum = 0.0f32;
    for i in 0..k_size {
        let x = (i as isize - radius) as f32;
        let weight = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(weight);
        sum += weight;
    }
    let inv_sum = 1.0 / sum;
    for kw in &mut kernel {
        *kw *= inv_sum;
    }

    // Pase Horizontal
    let mut temp = vec![0.0f32; w * h];
    temp.par_chunks_mut(w)
        .enumerate()
        .for_each(|(r, row_out)| {
            let row_in_offset = r * w;
            for c in 0..w {
                let mut acc = 0.0f32;
                for (k, &kw) in kernel.iter().enumerate() {
                    let sc = reflect_101(c as isize + (k as isize - radius), w);
                    acc += src[row_in_offset + sc] * kw;
                }
                row_out[c] = acc;
            }
        });

    // Pase Vertical
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w)
        .enumerate()
        .for_each(|(r, row_out)| {
            for c in 0..w {
                let mut acc = 0.0f32;
                for (k, &kw) in kernel.iter().enumerate() {
                    let sr = reflect_101(r as isize + (k as isize - radius), h);
                    acc += temp[sr * w + c] * kw;
                }
                row_out[c] = acc;
            }
        });

    out
}

#[pyfunction]
#[pyo3(signature = (img_rgb, saturation_boost=1.5, core_recovery=0.5, sky_mask=None, bg_protect=true))]
pub fn recover_star_trail_colors_rs<'py>(
    py: Python<'py>,
    img_rgb: PyReadonlyArray3<'py, f32>,
    saturation_boost: f32,
    core_recovery: f32,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
    bg_protect: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let img = img_rgb.as_array();
    let shape = img.shape();
    let (h, w, _) = (shape[0], shape[1], shape[2]);

    if saturation_boost <= 1.0 && core_recovery <= 0.0 {
        let mut out = Array3::<f32>::zeros((h, w, 3));
        out.assign(&img);
        return Ok(out.into_pyarray(py));
    }

    let mask_view = sky_mask.as_ref().map(|m| m.as_array());

    // 1. Luminancia y cálculo de fondo estelar
    let mut lum = vec![0.0f32; h * w];
    for r in 0..h {
        let row_offset = r * w;
        for c in 0..w {
            let r_val = img[[r, c, 0]];
            let g_val = img[[r, c, 1]];
            let b_val = img[[r, c, 2]];
            lum[row_offset + c] = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val;
        }
    }

    // Muestreo subpixel (cada 8 píxeles) para estimar mediana y MAD
    let mut sample = Vec::with_capacity((h / 8 + 1) * (w / 8 + 1));
    for r in (0..h).step_by(8) {
        let row_offset = r * w;
        for c in (0..w).step_by(8) {
            sample.push(lum[row_offset + c]);
        }
    }

    let med_bg = median_f32_slice(&mut sample);
    let mut dev: Vec<f32> = sample.iter().map(|&v| (v - med_bg).abs()).collect();
    let mad_bg = 1.4826 * median_f32_slice(&mut dev);

    let star_thresh = med_bg + (mad_bg * 2.5).max(0.025);

    // 2. Máscara de peso estelar suave y conversión HSV
    let mut star_weight = vec![0.0f32; h * w];
    let mut h_chan = vec![0.0f32; h * w];
    let mut s_chan = vec![0.0f32; h * w];
    let mut v_chan = vec![0.0f32; h * w];

    let mut has_core = false;
    let mut has_valid_star = false;

    for r in 0..h {
        let row_offset = r * w;
        for c in 0..w {
            let idx = row_offset + c;
            let l = lum[idx];

            let mut sw = if bg_protect {
                let w_norm = ((l - star_thresh) / star_thresh.max(0.04)).clamp(0.0, 1.0);
                w_norm * w_norm * (3.0 - 2.0 * w_norm)
            } else {
                1.0
            };

            if let Some(ref m) = mask_view {
                let mut sm_val = m[[r, c]];
                if sm_val > 1.05 {
                    sm_val /= 255.0;
                }
                sw *= sm_val.clamp(0.0, 1.0);
            }
            star_weight[idx] = sw;

            // RGB to HSV
            let red = img[[r, c, 0]].clamp(0.0, 1.0);
            let grn = img[[r, c, 1]].clamp(0.0, 1.0);
            let blu = img[[r, c, 2]].clamp(0.0, 1.0);

            let max_val = red.max(grn).max(blu);
            let min_val = red.min(grn).min(blu);
            let delta = max_val - min_val;

            let v = max_val;
            let s = if max_val > 1e-6 { delta / max_val } else { 0.0 };
            let mut h_deg = if delta < 1e-6 {
                0.0
            } else if (max_val - red).abs() < 1e-6 {
                60.0 * ((grn - blu) / delta)
            } else if (max_val - grn).abs() < 1e-6 {
                60.0 * ((blu - red) / delta + 2.0)
            } else {
                60.0 * ((red - grn) / delta + 4.0)
            };
            if h_deg < 0.0 {
                h_deg += 360.0;
            }

            h_chan[idx] = h_deg;
            s_chan[idx] = s;
            v_chan[idx] = v;

            if core_recovery > 0.0 {
                if v > 0.75 && s < 0.35 && sw > 0.15 {
                    has_core = true;
                }
                if sw > 0.20 && s > 0.15 && v > star_thresh {
                    has_valid_star = true;
                }
            }
        }
    }

    // 3. Recuperación de color en núcleos quemados (Core Recovery)
    if core_recovery > 0.0 && has_core && has_valid_star {
        let mut u_src = vec![0.0f32; h * w];
        let mut v_src = vec![0.0f32; h * w];
        let mut norm_src = vec![0.0f32; h * w];

        for i in 0..(h * w) {
            let sw = star_weight[i];
            let s = s_chan[i];
            let v = v_chan[i];
            if sw > 0.20 && s > 0.15 && v > star_thresh {
                let rad = h_chan[i].to_radians();
                u_src[i] = s * rad.cos();
                v_src[i] = s * rad.sin();
                norm_src[i] = 1.0;
            }
        }

        let mut k_size = (h.min(w) / 100).max(11);
        if k_size % 2 == 0 {
            k_size += 1;
        }
        let sigma = k_size as f32 / 2.5;

        let u_diffused = gaussian_blur_separable_2d(&u_src, w, h, k_size, sigma);
        let v_diffused = gaussian_blur_separable_2d(&v_src, w, h, k_size, sigma);
        let norm_diffused = gaussian_blur_separable_2d(&norm_src, w, h, k_size, sigma);

        for i in 0..(h * w) {
            let v = v_chan[i];
            let s = s_chan[i];
            let sw = star_weight[i];
            let is_core = v > 0.75 && s < 0.35 && sw > 0.15;

            if is_core {
                let nd = norm_diffused[i].max(1e-5);
                let u_rec = u_diffused[i] / nd;
                let v_rec = v_diffused[i] / nd;
                let rec_s = (u_rec * u_rec + v_rec * v_rec).sqrt().clamp(0.0, 1.0);
                let mut rec_h = v_rec.atan2(u_rec).to_degrees();
                if rec_h < 0.0 {
                    rec_h += 360.0;
                }

                let confidence = (nd * 2.0).clamp(0.0, 1.0);
                let infuse = (core_recovery * confidence).clamp(0.0, 1.0);

                if infuse > 0.0 {
                    h_chan[i] = (1.0 - infuse) * h_chan[i] + infuse * rec_h;
                    let target_core_s = (rec_s * 1.25).clamp(0.15, 0.75);
                    s_chan[i] = s_chan[i].max(target_core_s * infuse);
                }
            }
        }
    }

    // 4. Amplificación de Saturación y Reconstrucción RGB
    let mut out = Array3::<f32>::zeros((h, w, 3));
    let boost_factor = saturation_boost - 1.0;

    for r in 0..h {
        let row_offset = r * w;
        for c in 0..w {
            let idx = row_offset + c;
            let mut s = s_chan[idx];
            if saturation_boost > 1.0 {
                let boost_mult = 1.0 + boost_factor * star_weight[idx];
                s = (s * boost_mult).clamp(0.0, 1.0);
            }

            let v = v_chan[idx];
            let h_deg = h_chan[idx];

            // HSV to RGB
            let chroma = v * s;
            let h_prime = h_deg / 60.0;
            let x = chroma * (1.0 - ((h_prime % 2.0) - 1.0).abs());
            let m = v - chroma;

            let (r1, g1, b1) = if h_prime < 1.0 {
                (chroma, x, 0.0)
            } else if h_prime < 2.0 {
                (x, chroma, 0.0)
            } else if h_prime < 3.0 {
                (0.0, chroma, x)
            } else if h_prime < 4.0 {
                (0.0, x, chroma)
            } else if h_prime < 5.0 {
                (x, 0.0, chroma)
            } else {
                (chroma, 0.0, x)
            };

            let mut final_r = (r1 + m).clamp(0.0, 1.0);
            let mut final_g = (g1 + m).clamp(0.0, 1.0);
            let mut final_b = (b1 + m).clamp(0.0, 1.0);

            if let Some(ref mask) = mask_view {
                let mut sm_val = mask[[r, c]];
                if sm_val > 1.05 {
                    sm_val /= 255.0;
                }
                sm_val = sm_val.clamp(0.0, 1.0);
                let orig_r = img[[r, c, 0]];
                let orig_g = img[[r, c, 1]];
                let orig_b = img[[r, c, 2]];
                final_r = final_r * sm_val + orig_r * (1.0 - sm_val);
                final_g = final_g * sm_val + orig_g * (1.0 - sm_val);
                final_b = final_b * sm_val + orig_b * (1.0 - sm_val);
            }

            out[[r, c, 0]] = final_r;
            out[[r, c, 1]] = final_g;
            out[[r, c, 2]] = final_b;
        }
    }

    Ok(out.into_pyarray(py))
}

// ----------------------------------------------------------------------------
// 4. Conversión Cromática Bradford LUT (core/icc_manager.py)
// ----------------------------------------------------------------------------

#[inline(always)]
fn srgb_to_lin_val(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[inline(always)]
fn lin_to_srgb_val(lin: f32) -> f32 {
    let lin = lin.clamp(0.0, 1.0);
    if lin <= 0.0031308 {
        lin * 12.92
    } else {
        (1.055 * lin.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0)
    }
}

#[inline(always)]
fn adobe_to_lin_val(a: f32) -> f32 {
    a.clamp(0.0, 1.0).powf(2.19921875)
}

#[inline(always)]
fn lin_to_adobe_val(lin: f32) -> f32 {
    lin.clamp(0.0, 1.0).powf(1.0 / 2.19921875)
}

#[inline(always)]
fn prophoto_to_lin_val(p: f32) -> f32 {
    let p = p.clamp(0.0, 1.0);
    if p < 0.031248 {
        p / 16.0
    } else {
        p.powf(1.8)
    }
}

#[inline(always)]
fn lin_to_prophoto_val(lin: f32) -> f32 {
    let lin = lin.clamp(0.0, 1.0);
    if lin < 0.001953 {
        16.0 * lin
    } else {
        lin.powf(1.0 / 1.8).clamp(0.0, 1.0)
    }
}

fn get_trc_fns(space: &str) -> (fn(f32) -> f32, fn(f32) -> f32) {
    match space {
        "adobe" => (adobe_to_lin_val, lin_to_adobe_val),
        "prophoto" => (prophoto_to_lin_val, lin_to_prophoto_val),
        _ => (srgb_to_lin_val, lin_to_srgb_val),
    }
}

fn get_default_matrix(src_space: &str, dst_space: &str) -> [[f32; 3]; 3] {
    match (src_space, dst_space) {
        ("srgb", "adobe") => [
            [0.7151626, 0.2848374, 0.0000000],
            [0.0000000, 1.0000000, 0.0000000],
            [0.0000000, 0.0411705, 0.9588295],
        ],
        ("adobe", "srgb") => [
            [1.3982835, -0.39828342, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.04293829, 1.0429382],
        ],
        ("srgb", "prophoto") => [
            [0.529346, 0.33007282, 0.14058122],
            [0.09837437, 0.87346095, 0.02816469],
            [0.0168832, 0.1176725, 0.8654443],
        ],
        ("prophoto", "srgb") => [
            [2.0340757, -0.72733426, -0.30674154],
            [-0.2288134, 1.2317303, -0.00291693],
            [-0.00856978, -0.15328665, 1.1618564],
        ],
        ("adobe", "prophoto") => [
            [0.7401757, 0.11320677, 0.14661753],
            [0.13755526, 0.83307076, 0.02937404],
            [0.0236075, 0.07378752, 0.902605],
        ],
        ("prophoto", "adobe") => [
            [1.3895203, -0.16931938, -0.22020091],
            [-0.22881338, 1.2317302, -0.00291693],
            [-0.01763732, -0.09626482, 1.1139021],
        ],
        _ => [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
}

#[pyfunction]
#[pyo3(signature = (img_data, src_space="srgb", dst_space="srgb", matrix=None))]
pub fn convert_colorspace_bradford_rs<'py>(
    py: Python<'py>,
    img_data: PyReadonlyArrayDyn<'py, f32>,
    src_space: &str,
    dst_space: &str,
    matrix: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let arr = img_data.as_array();
    let shape = arr.shape();
    let ndim = shape.len();

    if src_space == dst_space || arr.is_empty() {
        let mut out = ArrayD::<f32>::zeros(IxDyn(shape));
        out.assign(&arr);
        return Ok(out.into_pyarray(py));
    }

    let (to_lin, _) = get_trc_fns(src_space);
    let (_, from_lin) = get_trc_fns(dst_space);

    let mat = if let Some(ref m) = matrix {
        let ma = m.as_array();
        [
            [ma[[0, 0]], ma[[0, 1]], ma[[0, 2]]],
            [ma[[1, 0]], ma[[1, 1]], ma[[1, 2]]],
            [ma[[2, 0]], ma[[2, 1]], ma[[2, 2]]],
        ]
    } else {
        get_default_matrix(src_space, dst_space)
    };

    let mut out = ArrayD::<f32>::zeros(IxDyn(shape));

    if ndim == 3 {
        let c = shape[2];
        let in_slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("img_data must be C-contiguous")
        })?;
        let out_slice = out.as_slice_mut().unwrap();

        py.allow_threads(|| {
            out_slice
                .par_chunks_mut(c)
                .zip(in_slice.par_chunks(c))
                .for_each(|(out_pix, in_pix)| {
                    let r_in = in_pix[0];
                    let g_in = in_pix[1];
                    let b_in = in_pix[2];

                    let lin_r = to_lin(r_in);
                    let lin_g = to_lin(g_in);
                    let lin_b = to_lin(b_in);

                    let c_r = mat[0][0] * lin_r + mat[0][1] * lin_g + mat[0][2] * lin_b;
                    let c_g = mat[1][0] * lin_r + mat[1][1] * lin_g + mat[1][2] * lin_b;
                    let c_b = mat[2][0] * lin_r + mat[2][1] * lin_g + mat[2][2] * lin_b;

                    out_pix[0] = from_lin(c_r).clamp(0.0, 1.0);
                    out_pix[1] = from_lin(c_g).clamp(0.0, 1.0);
                    out_pix[2] = from_lin(c_b).clamp(0.0, 1.0);

                    if c == 4 {
                        out_pix[3] = in_pix[3];
                    }
                });
        });

        Ok(out.into_pyarray(py))
    } else {
        out.assign(&arr);
        Ok(out.into_pyarray(py))
    }
}

#[pyfunction]
#[pyo3(signature = (img_data, src_space="srgb", dst_space="srgb", matrix=None))]
pub fn convert_colorspace_u16_bradford_rs<'py>(
    py: Python<'py>,
    img_data: PyReadonlyArrayDyn<'py, u16>,
    src_space: &str,
    dst_space: &str,
    matrix: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, PyArrayDyn<u16>>> {
    let arr = img_data.as_array();
    let shape = arr.shape();
    let ndim = shape.len();

    if src_space == dst_space || arr.is_empty() {
        let mut out = ArrayD::<u16>::zeros(IxDyn(shape));
        out.assign(&arr);
        return Ok(out.into_pyarray(py));
    }

    let (to_lin, _) = get_trc_fns(src_space);
    let (_, from_lin) = get_trc_fns(dst_space);

    let mat = if let Some(ref m) = matrix {
        let ma = m.as_array();
        [
            [ma[[0, 0]], ma[[0, 1]], ma[[0, 2]]],
            [ma[[1, 0]], ma[[1, 1]], ma[[1, 2]]],
            [ma[[2, 0]], ma[[2, 1]], ma[[2, 2]]],
        ]
    } else {
        get_default_matrix(src_space, dst_space)
    };

    // Precalcular tabla LUT de linearización uint16
    let lut: Vec<f32> = (0..65536)
        .map(|i| to_lin(i as f32 / 65535.0))
        .collect();

    let mut out = ArrayD::<u16>::zeros(IxDyn(shape));

    if ndim == 3 {
        let c = shape[2];
        let in_slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("img_data must be C-contiguous")
        })?;
        let out_slice = out.as_slice_mut().unwrap();

        py.allow_threads(|| {
            out_slice
                .par_chunks_mut(c)
                .zip(in_slice.par_chunks(c))
                .for_each(|(out_pix, in_pix)| {
                    let r_in = in_pix[0] as usize;
                    let g_in = in_pix[1] as usize;
                    let b_in = in_pix[2] as usize;

                    let lin_r = lut[r_in];
                    let lin_g = lut[g_in];
                    let lin_b = lut[b_in];

                    let c_r = mat[0][0] * lin_r + mat[0][1] * lin_g + mat[0][2] * lin_b;
                    let c_g = mat[1][0] * lin_r + mat[1][1] * lin_g + mat[1][2] * lin_b;
                    let c_b = mat[2][0] * lin_r + mat[2][1] * lin_g + mat[2][2] * lin_b;

                    let out_r = (from_lin(c_r) * 65535.0 + 0.5).round().clamp(0.0, 65535.0) as u16;
                    let out_g = (from_lin(c_g) * 65535.0 + 0.5).round().clamp(0.0, 65535.0) as u16;
                    let out_b = (from_lin(c_b) * 65535.0 + 0.5).round().clamp(0.0, 65535.0) as u16;

                    out_pix[0] = out_r;
                    out_pix[1] = out_g;
                    out_pix[2] = out_b;

                    if c == 4 {
                        out_pix[3] = in_pix[3];
                    }
                });
        });

        Ok(out.into_pyarray(py))
    } else {
        out.assign(&arr);
        Ok(out.into_pyarray(py))
    }
}

// ============================================================================
// FASE 17: INGESTA CONCURRENTE DE RAWS Y METADATOS (Rayon Batch Scanner)
// (core/stacking.py y core/startrails.py)
// ============================================================================

#[derive(Debug, Clone)]
pub struct RawFileMetadata {
    pub filepath: String,
    pub is_valid: bool,
    pub is_raw: bool,
    pub width: usize,
    pub height: usize,
    pub raw_width: usize,
    pub raw_height: usize,
    pub black_level: f64,
    pub white_balance: [f64; 3],
    pub color_temp: u32,
    pub make: String,
    pub model: String,
    pub iso: Option<u32>,
    pub exposure_time: Option<f64>,
    pub orientation: u16,
    pub error: Option<String>,
}

fn is_raw_extension(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".nef")
        || lower.ends_with(".cr2")
        || lower.ends_with(".cr3")
        || lower.ends_with(".arw")
        || lower.ends_with(".dng")
        || lower.ends_with(".raw")
        || lower.ends_with(".rw2")
        || lower.ends_with(".pef")
        || lower.ends_with(".orf")
        || lower.ends_with(".raf")
        || lower.ends_with(".kdc")
        || lower.ends_with(".mrw")
        || lower.ends_with(".sr2")
}

struct TiffReader<'a> {
    data: &'a [u8],
    is_le: bool,
}

impl<'a> TiffReader<'a> {
    #[inline(always)]
    fn new(data: &'a [u8], is_le: bool) -> Self {
        Self { data, is_le }
    }

    #[inline(always)]
    fn u16(&self, off: usize) -> Option<u16> {
        if off + 2 > self.data.len() {
            return None;
        }
        if self.is_le {
            Some(u16::from_le_bytes([self.data[off], self.data[off + 1]]))
        } else {
            Some(u16::from_be_bytes([self.data[off], self.data[off + 1]]))
        }
    }

    #[inline(always)]
    fn u32(&self, off: usize) -> Option<u32> {
        if off + 4 > self.data.len() {
            return None;
        }
        if self.is_le {
            Some(u32::from_le_bytes([
                self.data[off],
                self.data[off + 1],
                self.data[off + 2],
                self.data[off + 3],
            ]))
        } else {
            Some(u32::from_be_bytes([
                self.data[off],
                self.data[off + 1],
                self.data[off + 2],
                self.data[off + 3],
            ]))
        }
    }

    #[inline(always)]
    fn rational(&self, off: usize) -> Option<f64> {
        let num = self.u32(off)?;
        let den = self.u32(off + 4)?;
        if den == 0 {
            return None;
        }
        Some(num as f64 / den as f64)
    }

    fn ascii(&self, off: usize, count: usize) -> String {
        if off >= self.data.len() {
            return String::new();
        }
        let max_len = count.min(self.data.len() - off);
        let slice = &self.data[off..off + max_len];
        let end = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
        String::from_utf8_lossy(&slice[..end]).trim().to_string()
    }
}

fn parse_tiff_metadata(buf: &[u8], is_raw_ext: bool) -> Option<RawFileMetadata> {
    if buf.len() < 16 {
        return None;
    }
    let is_le = match &buf[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => {
            if &buf[0..4] == b"IIRO" {
                true
            } else if &buf[0..4] == b"MMOR" {
                false
            } else {
                return None;
            }
        }
    };

    let reader = TiffReader::new(buf, is_le);
    let version = reader.u16(2).unwrap_or(0);
    if version != 42 && version != 0x55 && version != 0x4F52 && &buf[0..4] != b"IIRO" && &buf[0..4] != b"MMOR" {
        return None;
    }

    let ifd0_off = reader.u32(4)? as usize;
    if ifd0_off + 2 > buf.len() {
        return None;
    }

    let n_entries = reader.u16(ifd0_off)? as usize;
    let mut w0 = 0usize;
    let mut h0 = 0usize;
    let mut make = String::new();
    let mut model = String::new();
    let mut orientation = 1u16;
    let mut subifd_offsets = Vec::new();
    let mut exif_offset: Option<usize> = None;
    let mut dng_black_level: Option<f64> = None;
    let mut dng_wb: Option<[f64; 3]> = None;

    for i in 0..n_entries {
        let entry_off = ifd0_off + 2 + i * 12;
        if entry_off + 12 > buf.len() {
            break;
        }
        let tag = reader.u16(entry_off).unwrap_or(0);
        let typ = reader.u16(entry_off + 2).unwrap_or(0);
        let cnt = reader.u32(entry_off + 4).unwrap_or(0) as usize;
        let val_or_off = reader.u32(entry_off + 8).unwrap_or(0);

        match tag {
            0x0100 => {
                w0 = if typ == 3 {
                    reader.u16(entry_off + 8).unwrap_or(0) as usize
                } else {
                    val_or_off as usize
                };
            }
            0x0101 => {
                h0 = if typ == 3 {
                    reader.u16(entry_off + 8).unwrap_or(0) as usize
                } else {
                    val_or_off as usize
                };
            }
            0x010F => {
                make = reader.ascii(val_or_off as usize, cnt);
            }
            0x0110 => {
                model = reader.ascii(val_or_off as usize, cnt);
            }
            0x0112 => {
                orientation = reader.u16(entry_off + 8).unwrap_or(1);
            }
            0x014A => {
                if cnt == 1 {
                    subifd_offsets.push(val_or_off as usize);
                } else if cnt > 1 {
                    let sub_base = val_or_off as usize;
                    for k in 0..cnt {
                        if let Some(off) = reader.u32(sub_base + k * 4) {
                            subifd_offsets.push(off as usize);
                        }
                    }
                }
            }
            0x8769 => {
                exif_offset = Some(val_or_off as usize);
            }
            0xC61A => {
                if typ == 3 {
                    dng_black_level = Some(reader.u16(entry_off + 8).unwrap_or(0) as f64);
                } else if typ == 4 {
                    dng_black_level = Some(val_or_off as f64);
                } else if typ == 5 {
                    dng_black_level = reader.rational(val_or_off as usize);
                }
            }
            0xC628 => {
                if cnt >= 3 {
                    let off_wb = val_or_off as usize;
                    if let (Some(r), Some(g), Some(b)) = (
                        reader.rational(off_wb),
                        reader.rational(off_wb + 8),
                        reader.rational(off_wb + 16),
                    ) {
                        if r > 1e-6 && g > 1e-6 && b > 1e-6 {
                            let r_gain = (1.0 / r) / (1.0 / g);
                            let b_gain = (1.0 / b) / (1.0 / g);
                            dng_wb = Some([r_gain, 1.0, b_gain]);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut raw_w = w0;
    let mut raw_h = h0;

    // Check Canon CR2 RAW IFD pointer at offset 12
    if buf.len() >= 16 && &buf[8..12] == b"CR\x02\x00" {
        if let Some(cr2_raw_off) = reader.u32(12) {
            let cr2_off = cr2_raw_off as usize;
            if cr2_off + 2 <= buf.len() {
                if let Some(n_cr2) = reader.u16(cr2_off) {
                    for i in 0..(n_cr2 as usize) {
                        let off = cr2_off + 2 + i * 12;
                        if off + 12 > buf.len() {
                            break;
                        }
                        let tag = reader.u16(off).unwrap_or(0);
                        let typ = reader.u16(off + 2).unwrap_or(0);
                        let val_or_off = reader.u32(off + 8).unwrap_or(0);
                        if tag == 0x0100 {
                            raw_w = if typ == 3 {
                                reader.u16(off + 8).unwrap_or(0) as usize
                            } else {
                                val_or_off as usize
                            };
                        } else if tag == 0x0101 {
                            raw_h = if typ == 3 {
                                reader.u16(off + 8).unwrap_or(0) as usize
                            } else {
                                val_or_off as usize
                            };
                        }
                    }
                }
            }
        }
    }

    // Check SubIFDs for full sensor dimensions (Nikon NEF, Sony ARW, DNG)
    for sub_off in subifd_offsets.iter() {
        let off_val = *sub_off;
        if off_val + 2 > buf.len() {
            continue;
        }
        if let Some(n_sub) = reader.u16(off_val) {
            let mut sw = 0usize;
            let mut sh = 0usize;
            for i in 0..(n_sub as usize) {
                let off = off_val + 2 + i * 12;
                if off + 12 > buf.len() {
                    break;
                }
                let tag = reader.u16(off).unwrap_or(0);
                let typ = reader.u16(off + 2).unwrap_or(0);
                let val_or_off = reader.u32(off + 8).unwrap_or(0);
                if tag == 0x0100 {
                    sw = if typ == 3 {
                        reader.u16(off + 8).unwrap_or(0) as usize
                    } else {
                        val_or_off as usize
                    };
                } else if tag == 0x0101 {
                    sh = if typ == 3 {
                        reader.u16(off + 8).unwrap_or(0) as usize
                    } else {
                        val_or_off as usize
                    };
                } else if tag == 0xC61A && dng_black_level.is_none() {
                    if typ == 3 {
                        dng_black_level = Some(reader.u16(off + 8).unwrap_or(0) as f64);
                    } else if typ == 4 {
                        dng_black_level = Some(val_or_off as f64);
                    } else if typ == 5 {
                        dng_black_level = reader.rational(val_or_off as usize);
                    }
                }
            }
            if sw * sh > raw_w * raw_h {
                raw_w = sw;
                raw_h = sh;
            }
        }
    }

    // Exif tags: ISO, ExposureTime, MakerNote
    let mut iso: Option<u32> = None;
    let mut exposure_time: Option<f64> = None;
    let mut wb = dng_wb.unwrap_or([2.0, 1.0, 1.5]);

    if let Some(exif_off) = exif_offset {
        if exif_off + 2 <= buf.len() {
            if let Some(n_exif) = reader.u16(exif_off) {
                let mut mkn_offset: Option<usize> = None;
                for i in 0..(n_exif as usize) {
                    let off = exif_off + 2 + i * 12;
                    if off + 12 > buf.len() {
                        break;
                    }
                    let tag = reader.u16(off).unwrap_or(0);
                    let val_or_off = reader.u32(off + 8).unwrap_or(0);
                    match tag {
                        0x829A => {
                            exposure_time = reader.rational(val_or_off as usize);
                        }
                        0x8827 => {
                            iso = Some(reader.u16(off + 8).unwrap_or(val_or_off as u16) as u32);
                        }
                        0x927C => {
                            mkn_offset = Some(val_or_off as usize);
                        }
                        _ => {}
                    }
                }

                if let Some(mkn_off) = mkn_offset {
                    if mkn_off + 20 <= buf.len() && &buf[mkn_off..mkn_off + 6] == b"Nikon\0" {
                        let mkn_base = mkn_off + 10;
                        if mkn_base + 10 <= buf.len() {
                            let n_mkn = reader.u16(mkn_base + 8).unwrap_or(0) as usize;
                            for i in 0..n_mkn {
                                let off = mkn_base + 10 + i * 12;
                                if off + 12 > buf.len() {
                                    break;
                                }
                                let tag = reader.u16(off).unwrap_or(0);
                                let val_or_off = reader.u32(off + 8).unwrap_or(0);
                                if tag == 0x000C {
                                    let wb_off = mkn_base + val_or_off as usize;
                                    if wb_off + 32 <= buf.len() {
                                        if let (Some(r), Some(b), Some(g)) = (
                                            reader.rational(wb_off),
                                            reader.rational(wb_off + 8),
                                            reader.rational(wb_off + 16),
                                        ) {
                                            if g > 1e-6 {
                                                wb = [r / g, 1.0, b / g];
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let make_upper = make.to_uppercase();
    let black_level = if let Some(bl) = dng_black_level {
        bl
    } else if make_upper.contains("NIKON") {
        if model.contains('Z') || model.contains("D850") || model.contains("D750") {
            1008.0
        } else if model.contains("D7000") || model.contains("D5100") {
            600.0
        } else {
            1008.0
        }
    } else if make_upper.contains("CANON") {
        1024.0
    } else if make_upper.contains("SONY") {
        512.0
    } else {
        0.0
    };

    let r_gain = wb[0].max(0.1);
    let b_gain = wb[2].max(0.1);
    let ratio = (r_gain / b_gain) / 1.44;
    let color_temp = (5500.0 * ratio.powf(0.85)).round().clamp(2000.0, 15000.0) as u32;

    let (width, height) = if matches!(orientation, 5 | 6 | 7 | 8) {
        (raw_h, raw_w)
    } else {
        (raw_w, raw_h)
    };

    let is_raw = is_raw_ext
        || !subifd_offsets.is_empty()
        || make_upper.contains("NIKON")
        || make_upper.contains("CANON")
        || make_upper.contains("SONY");

    Some(RawFileMetadata {
        filepath: String::new(),
        is_valid: raw_w > 0 && raw_h > 0,
        is_raw,
        width,
        height,
        raw_width: raw_w,
        raw_height: raw_h,
        black_level,
        white_balance: wb,
        color_temp,
        make,
        model,
        iso,
        exposure_time,
        orientation,
        error: None,
    })
}

fn parse_fits_metadata(buf: &[u8]) -> Option<RawFileMetadata> {
    if buf.len() < 80 {
        return None;
    }
    let header_str = String::from_utf8_lossy(buf);
    if !header_str.starts_with("SIMPLE  ") && !header_str.starts_with("XTENSION") {
        return None;
    }

    let mut w = 0usize;
    let mut h = 0usize;
    let mut exp_time: Option<f64> = None;
    let mut instr = String::new();
    let mut tele = String::new();

    for chunk in buf.chunks_exact(80) {
        let card = String::from_utf8_lossy(chunk);
        if card.starts_with("END ") {
            break;
        }
        if card.starts_with("NAXIS1  =") {
            if let Some(val_str) = card.split('=').nth(1) {
                let clean = val_str.split('/').next().unwrap_or("").trim();
                w = clean.parse().unwrap_or(0);
            }
        } else if card.starts_with("NAXIS2  =") {
            if let Some(val_str) = card.split('=').nth(1) {
                let clean = val_str.split('/').next().unwrap_or("").trim();
                h = clean.parse().unwrap_or(0);
            }
        } else if card.starts_with("EXPTIME =") {
            if let Some(val_str) = card.split('=').nth(1) {
                let clean = val_str.split('/').next().unwrap_or("").trim();
                exp_time = clean.parse().ok();
            }
        } else if card.starts_with("INSTRUME=") {
            if let Some(val_str) = card.split('=').nth(1) {
                instr = val_str.replace('\'', "").trim().to_string();
            }
        } else if card.starts_with("TELESCOP=") {
            if let Some(val_str) = card.split('=').nth(1) {
                tele = val_str.replace('\'', "").trim().to_string();
            }
        }
    }

    if w == 0 || h == 0 {
        return None;
    }

    Some(RawFileMetadata {
        filepath: String::new(),
        is_valid: true,
        is_raw: false,
        width: w,
        height: h,
        raw_width: w,
        raw_height: h,
        black_level: 0.0,
        white_balance: [1.0, 1.0, 1.0],
        color_temp: 5500,
        make: tele,
        model: instr,
        iso: None,
        exposure_time: exp_time,
        orientation: 1,
        error: None,
    })
}

fn parse_png_metadata(buf: &[u8]) -> Option<RawFileMetadata> {
    if buf.len() < 24 || &buf[0..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    if &buf[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes([buf[16], buf[17], buf[18], buf[19]]) as usize;
    let h = u32::from_be_bytes([buf[20], buf[21], buf[22], buf[23]]) as usize;

    Some(RawFileMetadata {
        filepath: String::new(),
        is_valid: w > 0 && h > 0,
        is_raw: false,
        width: w,
        height: h,
        raw_width: w,
        raw_height: h,
        black_level: 0.0,
        white_balance: [1.0, 1.0, 1.0],
        color_temp: 5500,
        make: String::new(),
        model: String::new(),
        iso: None,
        exposure_time: None,
        orientation: 1,
        error: None,
    })
}

fn parse_jpeg_metadata(buf: &[u8]) -> Option<RawFileMetadata> {
    if buf.len() < 4 || buf[0] != 0xFF || buf[1] != 0xD8 {
        return None;
    }
    let mut off = 2usize;
    let mut w = 0usize;
    let mut h = 0usize;

    while off + 4 < buf.len() {
        if buf[off] != 0xFF {
            off += 1;
            continue;
        }
        let marker = buf[off + 1];
        if marker == 0xC0 || marker == 0xC2 {
            if off + 9 <= buf.len() {
                h = u16::from_be_bytes([buf[off + 5], buf[off + 6]]) as usize;
                w = u16::from_be_bytes([buf[off + 7], buf[off + 8]]) as usize;
                break;
            }
        }
        if marker == 0xD9 || marker == 0xDA {
            break;
        }
        let len = u16::from_be_bytes([buf[off + 2], buf[off + 3]]) as usize;
        off += 2 + len;
    }

    if w == 0 || h == 0 {
        return None;
    }

    Some(RawFileMetadata {
        filepath: String::new(),
        is_valid: true,
        is_raw: false,
        width: w,
        height: h,
        raw_width: w,
        raw_height: h,
        black_level: 0.0,
        white_balance: [1.0, 1.0, 1.0],
        color_temp: 5500,
        make: String::new(),
        model: String::new(),
        iso: None,
        exposure_time: None,
        orientation: 1,
        error: None,
    })
}

pub fn parse_file_metadata(path_str: &str) -> RawFileMetadata {
    let is_raw_ext = is_raw_extension(path_str);
    let mut file = match File::open(path_str) {
        Ok(f) => f,
        Err(e) => {
            return RawFileMetadata {
                filepath: path_str.to_string(),
                is_valid: false,
                is_raw: is_raw_ext,
                width: 0,
                height: 0,
                raw_width: 0,
                raw_height: 0,
                black_level: 0.0,
                white_balance: [1.0, 1.0, 1.0],
                color_temp: 5500,
                make: String::new(),
                model: String::new(),
                iso: None,
                exposure_time: None,
                orientation: 1,
                error: Some(format!("No se pudo abrir el archivo: {}", e)),
            };
        }
    };

    let mut buf = vec![0u8; 262144];
    let n = match file.read(&mut buf) {
        Ok(n) => n,
        Err(e) => {
            return RawFileMetadata {
                filepath: path_str.to_string(),
                is_valid: false,
                is_raw: is_raw_ext,
                width: 0,
                height: 0,
                raw_width: 0,
                raw_height: 0,
                black_level: 0.0,
                white_balance: [1.0, 1.0, 1.0],
                color_temp: 5500,
                make: String::new(),
                model: String::new(),
                iso: None,
                exposure_time: None,
                orientation: 1,
                error: Some(format!("Error leyendo cabecera: {}", e)),
            };
        }
    };
    buf.truncate(n);

    let mut meta = parse_tiff_metadata(&buf, is_raw_ext)
        .or_else(|| parse_fits_metadata(&buf))
        .or_else(|| parse_png_metadata(&buf))
        .or_else(|| parse_jpeg_metadata(&buf))
        .unwrap_or_else(|| RawFileMetadata {
            filepath: path_str.to_string(),
            is_valid: is_raw_ext,
            is_raw: is_raw_ext,
            width: 0,
            height: 0,
            raw_width: 0,
            raw_height: 0,
            black_level: 0.0,
            white_balance: [1.0, 1.0, 1.0],
            color_temp: 5500,
            make: String::new(),
            model: String::new(),
            iso: None,
            exposure_time: None,
            orientation: 1,
            error: if is_raw_ext {
                None
            } else {
                Some("Formato de archivo no soportado o desconocido".to_string())
            },
        });

    meta.filepath = path_str.to_string();
    meta
}

#[pyfunction]
#[pyo3(signature = (file_paths))]
pub fn scan_raw_batch_metadata_rs(
    py: Python<'_>,
    file_paths: Vec<String>,
) -> PyResult<PyObject> {
    let results: Vec<(String, RawFileMetadata)> = py.allow_threads(|| {
        file_paths
            .par_iter()
            .map(|p| {
                let meta = parse_file_metadata(p);
                (p.clone(), meta)
            })
            .collect()
    });

    let py_dict = PyDict::new(py);
    for (path, meta) in results {
        let item = PyDict::new(py);
        item.set_item("filepath", &meta.filepath)?;
        item.set_item("is_valid", meta.is_valid)?;
        item.set_item("is_raw", meta.is_raw)?;
        item.set_item("width", meta.width)?;
        item.set_item("height", meta.height)?;
        item.set_item("raw_width", meta.raw_width)?;
        item.set_item("raw_height", meta.raw_height)?;
        item.set_item("black_level", meta.black_level)?;
        item.set_item("white_balance", meta.white_balance.to_vec())?;
        item.set_item("color_temp", meta.color_temp)?;
        item.set_item("make", &meta.make)?;
        item.set_item("model", &meta.model)?;
        item.set_item("iso", meta.iso)?;
        item.set_item("exposure_time", meta.exposure_time)?;
        item.set_item("orientation", meta.orientation)?;
        item.set_item("error", meta.error)?;
        py_dict.set_item(path, item)?;
    }

    Ok(py_dict.into())
}

#[pyfunction]
#[pyo3(signature = (file_path))]
pub fn scan_raw_metadata_rs(
    py: Python<'_>,
    file_path: String,
) -> PyResult<PyObject> {
    let meta = py.allow_threads(|| parse_file_metadata(&file_path));
    let item = PyDict::new(py);
    item.set_item("filepath", &meta.filepath)?;
    item.set_item("is_valid", meta.is_valid)?;
    item.set_item("is_raw", meta.is_raw)?;
    item.set_item("width", meta.width)?;
    item.set_item("height", meta.height)?;
    item.set_item("raw_width", meta.raw_width)?;
    item.set_item("raw_height", meta.raw_height)?;
    item.set_item("black_level", meta.black_level)?;
    item.set_item("white_balance", meta.white_balance.to_vec())?;
    item.set_item("color_temp", meta.color_temp)?;
    item.set_item("make", &meta.make)?;
    item.set_item("model", &meta.model)?;
    item.set_item("iso", meta.iso)?;
    item.set_item("exposure_time", meta.exposure_time)?;
    item.set_item("orientation", meta.orientation)?;
    item.set_item("error", meta.error)?;
    Ok(item.into())
}
