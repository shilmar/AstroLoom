use numpy::ndarray::{Array3, ArrayD, IxDyn};
use numpy::{
    IntoPyArray, PyArray3, PyArrayDyn, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
    PyReadwriteArray2, PyReadwriteArray3,
};
use pyo3::prelude::*;
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
    let (h, w) = raw_cfa_u16.as_array().dim();
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
