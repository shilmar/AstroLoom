use numpy::ndarray::{Array3, Array4, ArrayD, IxDyn};
use numpy::{
    IntoPyArray, PyArray3, PyArray4, PyArrayDyn, PyArrayMethods, PyReadonlyArray3,
    PyReadonlyArray4, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// FASE 13: TILING ONNX Y FILTRO GUIADO (core/starnet_onnx.py, core/ai_horizon.py)
// Pre y post-procesado acelerado para redes neuronales (StarNet, NAFNet, U2-Net)
// 1. Tiling Hann 2D: Partición, ventana Hann simétrica y recomposición ponderada con Rayon.
// 2. Fast Guided Filter: Filtrado guiado O(1) por píxel mediante imágenes integrales.
// ============================================================================

/// Calcula el mapeo de coordenadas con reflejo simétrico (BORDER_REFLECT_101).
#[inline(always)]
fn reflect_101(pos: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let m = (len - 1) as isize;
    let mut v = pos.abs();
    let cycle = 2 * m;
    v %= cycle;
    if v > m {
        v = cycle - v;
    }
    v as usize
}

/// Construye una imagen de área sumada (SAT / Integral Image) con extensión BORDER_REFLECT_101.
fn build_sat_reflected(
    input: &[f32],
    h: usize,
    w: usize,
    r: usize,
    h_pad: usize,
    w_pad: usize,
    sat: &mut [f64],
) {
    let sat_stride = w_pad + 1;
    sat[..sat_stride].fill(0.0);

    for yp in 0..h_pad {
        let yo = reflect_101(yp as isize - r as isize, h);
        let in_offset = yo * w;
        let prev_offset = yp * sat_stride;
        let curr_offset = (yp + 1) * sat_stride;

        sat[curr_offset] = 0.0;
        let mut row_sum = 0.0f64;

        for xp in 0..w_pad {
            let xo = reflect_101(xp as isize - r as isize, w);
            row_sum += input[in_offset + xo] as f64;
            sat[curr_offset + xp + 1] = sat[prev_offset + xp + 1] + row_sum;
        }
    }
}

/// Evalúa en paralelo el promedio de cada píxel en un radio r usando la SAT.
fn query_box_filter(
    sat: &[f64],
    w_pad: usize,
    _h: usize,
    w: usize,
    r: usize,
    inv_area: f32,
    out: &mut [f32],
) {
    let sat_stride = w_pad + 1;
    let r_step = 2 * r + 1;

    out.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let y0 = y;
            let y1 = y + r_step;
            let r0_offset = y0 * sat_stride;
            let r1_offset = y1 * sat_stride;

            for (x, out_val) in row.iter_mut().enumerate() {
                let x0 = x;
                let x1 = x + r_step;
                let sum = sat[r1_offset + x1]
                    - sat[r0_offset + x1]
                    - sat[r1_offset + x0]
                    + sat[r0_offset + x0];
                *out_val = (sum as f32) * inv_area;
            }
        });
}

/// Ejecuta el Filtro Guiado (He et al., ECCV 2010) ultrarrápido en tiempo O(1) con Rayon.
/// Soporta guía 2D y fuente 2D o 3D (procesando cada canal con el modelo afín).
#[pyfunction]
#[pyo3(signature = (guide, src, radius=15, eps=1e-3))]
pub fn fast_guided_filter_rs<'py>(
    py: Python<'py>,
    guide: PyReadonlyArrayDyn<'py, f32>,
    src: PyReadonlyArrayDyn<'py, f32>,
    radius: usize,
    eps: f32,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let g_view = guide.as_array();
    let s_view = src.as_array();

    let (h, w) = match g_view.ndim() {
        2 => (g_view.shape()[0], g_view.shape()[1]),
        3 if g_view.shape()[2] == 1 => (g_view.shape()[0], g_view.shape()[1]),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "guide debe ser 2D (H, W) o 3D con 1 canal (H, W, 1)",
            ));
        }
    };

    let (s_h, s_w, c, is_2d) = match s_view.ndim() {
        2 => (s_view.shape()[0], s_view.shape()[1], 1, true),
        3 => (s_view.shape()[0], s_view.shape()[1], s_view.shape()[2], false),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "src debe ser 2D (H, W) o 3D (H, W, C)",
            ));
        }
    };

    if s_h != h || s_w != w {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimensiones incompatibles: guide ({}, {}) != src ({}, {})",
            h, w, s_h, s_w
        )));
    }

    let r = radius.max(1);
    let epsilon = eps.max(1e-8);
    let h_pad = h + 2 * r;
    let w_pad = w + 2 * r;
    let inv_area = 1.0 / ((2 * r + 1) * (2 * r + 1)) as f32;

    // Aplanar guía contigua
    let guide_flat: Vec<f32> = if let Some(slice) = g_view.as_slice() {
        slice.to_vec()
    } else {
        g_view.iter().copied().collect()
    };

    let total_pixels = h * w;
    let mut out_vec = vec![0.0f32; total_pixels * c];

    py.allow_threads(|| {
        let mut sat = vec![0.0f64; (h_pad + 1) * (w_pad + 1)];
        let mut mean_i = vec![0.0f32; total_pixels];
        let mut mean_ii = vec![0.0f32; total_pixels];
        let mut var_i = vec![0.0f32; total_pixels];

        // 1. mean_I = box_filter(I)
        build_sat_reflected(&guide_flat, h, w, r, h_pad, w_pad, &mut sat);
        query_box_filter(&sat, w_pad, h, w, r, inv_area, &mut mean_i);

        // 2. mean_II = box_filter(I * I)
        let mut ii = vec![0.0f32; total_pixels];
        ii.par_iter_mut()
            .zip(guide_flat.par_iter())
            .for_each(|(val, &g)| *val = g * g);
        build_sat_reflected(&ii, h, w, r, h_pad, w_pad, &mut sat);
        query_box_filter(&sat, w_pad, h, w, r, inv_area, &mut mean_ii);

        // 3. var_I = mean_II - mean_I * mean_I
        var_i
            .par_iter_mut()
            .zip(mean_ii.par_iter())
            .zip(mean_i.par_iter())
            .for_each(|((v, &mii), &mi)| *v = mii - mi * mi);

        // Búferes temporales reutilizables por canal
        let mut p_ch = vec![0.0f32; total_pixels];
        let mut ip_ch = vec![0.0f32; total_pixels];
        let mut mean_p = vec![0.0f32; total_pixels];
        let mut mean_ip = vec![0.0f32; total_pixels];
        let mut a = vec![0.0f32; total_pixels];
        let mut b = vec![0.0f32; total_pixels];
        let mut mean_a = vec![0.0f32; total_pixels];
        let mut mean_b = vec![0.0f32; total_pixels];

        // 4. Procesar cada canal
        for ch in 0..c {
            // Extraer canal ch
            if is_2d {
                if let Some(slice) = s_view.as_slice() {
                    p_ch.copy_from_slice(slice);
                } else {
                    for (idx, &v) in s_view.iter().enumerate() {
                        p_ch[idx] = v;
                    }
                }
            } else {
                for y in 0..h {
                    for x in 0..w {
                        p_ch[y * w + x] = unsafe { *s_view.uget([y, x, ch]) };
                    }
                }
            }

            // mean_p = box_filter(p)
            build_sat_reflected(&p_ch, h, w, r, h_pad, w_pad, &mut sat);
            query_box_filter(&sat, w_pad, h, w, r, inv_area, &mut mean_p);

            // mean_Ip = box_filter(I * p)
            ip_ch
                .par_iter_mut()
                .zip(guide_flat.par_iter())
                .zip(p_ch.par_iter())
                .for_each(|((val, &g), &p)| *val = g * p);
            build_sat_reflected(&ip_ch, h, w, r, h_pad, w_pad, &mut sat);
            query_box_filter(&sat, w_pad, h, w, r, inv_area, &mut mean_ip);

            // a = (mean_Ip - mean_I * mean_p) / (var_I + eps)
            // b = mean_p - a * mean_I
            a.par_iter_mut()
                .zip(b.par_iter_mut())
                .zip(mean_ip.par_iter())
                .zip(mean_i.par_iter())
                .zip(mean_p.par_iter())
                .zip(var_i.par_iter())
                .for_each(|(((((out_a, out_b), &mip), &mi), &mp), &vi)| {
                    let cov = mip - mi * mp;
                    let a_val = cov / (vi + epsilon);
                    *out_a = a_val;
                    *out_b = mp - a_val * mi;
                });

            // mean_a = box_filter(a)
            build_sat_reflected(&a, h, w, r, h_pad, w_pad, &mut sat);
            query_box_filter(&sat, w_pad, h, w, r, inv_area, &mut mean_a);

            // mean_b = box_filter(b)
            build_sat_reflected(&b, h, w, r, h_pad, w_pad, &mut sat);
            query_box_filter(&sat, w_pad, h, w, r, inv_area, &mut mean_b);

            // q = mean_a * I + mean_b
            if is_2d {
                out_vec
                    .par_iter_mut()
                    .zip(mean_a.par_iter())
                    .zip(mean_b.par_iter())
                    .zip(guide_flat.par_iter())
                    .for_each(|(((q, &ma), &mb), &g)| {
                        *q = (ma * g + mb).clamp(0.0, 1.0);
                    });
            } else {
                for y in 0..h {
                    for x in 0..w {
                        let idx = y * w + x;
                        let ma = mean_a[idx];
                        let mb = mean_b[idx];
                        let g = guide_flat[idx];
                        out_vec[idx * c + ch] = (ma * g + mb).clamp(0.0, 1.0);
                    }
                }
            }
        }
    });

    let shape_dyn = if is_2d { vec![h, w] } else { vec![h, w, c] };
    let out_arr = ArrayD::from_shape_vec(IxDyn(&shape_dyn), out_vec).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error construyendo ArrayD: {:?}", e))
    })?;
    Ok(out_arr.into_pyarray(py))
}

/// Genera una ventana separable 2D de Hann:
/// w(y, x) = sin²(pi * (y + 0.5) / H) * sin²(pi * (x + 0.5) / W)
/// Devuelve un array de forma (height, width, 1).
#[pyfunction]
pub fn compute_hann_2d_window_rs<'py>(
    py: Python<'py>,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if height == 0 || width == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Las dimensiones deben ser mayores a 0",
        ));
    }

    let y_w: Vec<f32> = (0..height)
        .map(|y| {
            let s = ((std::f32::consts::PI * (y as f32 + 0.5)) / (height as f32)).sin();
            s * s
        })
        .collect();
    let x_w: Vec<f32> = (0..width)
        .map(|x| {
            let s = ((std::f32::consts::PI * (x as f32 + 0.5)) / (width as f32)).sin();
            s * s
        })
        .collect();

    let mut data = vec![0.0f32; height * width];
    data.par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let yw = y_w[y];
            for (x, val) in row.iter_mut().enumerate() {
                *val = yw * x_w[x];
            }
        });

    let arr = Array3::from_shape_vec((height, width, 1), data).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error construyendo Array3: {:?}", e))
    })?;
    Ok(arr.into_pyarray(py))
}

/// Genera coordenadas uniformes de partición en baldosas cubriendo toda la longitud.
#[pyfunction]
pub fn get_tile_coords_rs(length: usize, tile_size: usize, stride: usize) -> Vec<usize> {
    if length <= tile_size {
        return vec![0];
    }
    let s = stride.max(1);
    let mut coords = Vec::new();
    let mut pos = 0;
    while pos + tile_size < length {
        coords.push(pos);
        pos += s;
    }
    coords.push(length - tile_size);
    coords.sort_unstable();
    coords.dedup();
    coords
}

/// Acumula una baldosa ponderada con ventana Hann in-place directamente sobre los búferes de trabajo.
#[pyfunction]
pub fn accumulate_tile_hann_inplace_rs<'py>(
    _py: Python<'py>,
    acc_img: &Bound<'py, PyArray3<f32>>,
    acc_weights: &Bound<'py, PyArray3<f32>>,
    tile: PyReadonlyArray3<'py, f32>,
    hann_win: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
) -> PyResult<()> {
    let tile_view = tile.as_array();
    let (th, tw, tc) = (tile_view.shape()[0], tile_view.shape()[1], tile_view.shape()[2]);
    let (ih, iw, ic) = (acc_img.shape()[0], acc_img.shape()[1], acc_img.shape()[2]);

    if y + th > ih || x + tw > iw {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "La baldosa en ({}, {}) de tamaño ({}, {}) desborda el acumulador ({}, {})",
            y, x, th, tw, ih, iw
        )));
    }
    if tc != ic {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "El número de canales de la baldosa debe coincidir con el acumulador",
        ));
    }

    let win_view = hann_win.as_array();

    let acc_img_slice = unsafe {
        acc_img.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("acc_img debe ser C-contiguo: {:?}", e))
        })?
    };
    let acc_weights_slice = unsafe {
        acc_weights.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "acc_weights debe ser C-contiguo: {:?}",
                e
            ))
        })?
    };

    let weight_channels = acc_weights.shape()[2];

    for ty in 0..th {
        let iy = y + ty;
        for tx in 0..tw {
            let ix = x + tx;
            let w = unsafe { *win_view.uget([ty, tx, 0]) };
            let img_idx = (iy * iw + ix) * ic;
            for c_idx in 0..ic {
                let t_val = unsafe { *tile_view.uget([ty, tx, c_idx]) };
                acc_img_slice[img_idx + c_idx] += t_val * w;
            }
            let weight_idx = (iy * iw + ix) * weight_channels;
            acc_weights_slice[weight_idx] += w;
        }
    }
    Ok(())
}

/// Normaliza el búfer acumulador por la suma de pesos Hann y recorta al tamaño original.
#[pyfunction]
pub fn reconstruct_hann_tiling_rs<'py>(
    py: Python<'py>,
    acc_img: PyReadonlyArray3<'py, f32>,
    acc_weights: PyReadonlyArray3<'py, f32>,
    orig_h: usize,
    orig_w: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let img_view = acc_img.as_array();
    let w_view = acc_weights.as_array();
    let img_shape = img_view.shape();
    let (ih, iw, ic) = (img_shape[0], img_shape[1], img_shape[2]);

    let target_h = orig_h.min(ih);
    let target_w = orig_w.min(iw);

    let mut out = vec![0.0f32; target_h * target_w * ic];

    py.allow_threads(|| {
        out.par_chunks_exact_mut(target_w * ic)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..target_w {
                    let weight = unsafe { *w_view.uget([y, x, 0]) };
                    let inv_w = if weight > 1e-12 { 1.0 / weight } else { 0.0 };
                    for c_idx in 0..ic {
                        let val = unsafe { *img_view.uget([y, x, c_idx]) };
                        row[x * ic + c_idx] = (val * inv_w).clamp(0.0, 1.0);
                    }
                }
            });
    });

    let arr = Array3::from_shape_vec((target_h, target_w, ic), out).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error creando Array3: {:?}", e))
    })?;
    Ok(arr.into_pyarray(py))
}

/// Recompone en paralelo una lista completa de baldosas con ventana Hann 2D en una sola llamada.
#[pyfunction]
#[pyo3(signature = (tiles, coords, h_pad, w_pad, orig_h, orig_w, tile_size))]
pub fn recompose_hann_tiles_rs<'py>(
    py: Python<'py>,
    tiles: Vec<PyReadonlyArray3<'py, f32>>,
    coords: Vec<(usize, usize)>,
    h_pad: usize,
    w_pad: usize,
    orig_h: usize,
    orig_w: usize,
    tile_size: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if tiles.is_empty() || tiles.len() != coords.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "La lista de baldosas y coordenadas debe ser no vacía y de igual longitud",
        ));
    }

    let ic = tiles[0].as_array().shape()[2];
    let mut acc_img = vec![0.0f32; h_pad * w_pad * ic];
    let mut acc_weights = vec![0.0f32; h_pad * w_pad];

    // Precalcular ventana Hann
    let y_w: Vec<f32> = (0..tile_size)
        .map(|y| {
            let s = ((std::f32::consts::PI * (y as f32 + 0.5)) / (tile_size as f32)).sin();
            s * s
        })
        .collect();
    let x_w: Vec<f32> = (0..tile_size)
        .map(|x| {
            let s = ((std::f32::consts::PI * (x as f32 + 0.5)) / (tile_size as f32)).sin();
            s * s
        })
        .collect();

    // Acumular baldosas secuencialmente en memoria (sub-milisegundo)
    for (tile_ro, &(y, x)) in tiles.iter().zip(coords.iter()) {
        let t_view = tile_ro.as_array();
        let th = t_view.shape()[0].min(tile_size);
        let tw = t_view.shape()[1].min(tile_size);

        for ty in 0..th {
            let iy = y + ty;
            if iy >= h_pad {
                continue;
            }
            let yw = y_w[ty];
            for tx in 0..tw {
                let ix = x + tx;
                if ix >= w_pad {
                    continue;
                }
                let w = yw * x_w[tx];
                let img_idx = (iy * w_pad + ix) * ic;
                for c_idx in 0..ic {
                    let t_val = unsafe { *t_view.uget([ty, tx, c_idx]) };
                    acc_img[img_idx + c_idx] += t_val * w;
                }
                acc_weights[iy * w_pad + ix] += w;
            }
        }
    }

    let target_h = orig_h.min(h_pad);
    let target_w = orig_w.min(w_pad);
    let mut out = vec![0.0f32; target_h * target_w * ic];

    // Normalización multihilo con Rayon sin GIL
    py.allow_threads(|| {
        out.par_chunks_exact_mut(target_w * ic)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..target_w {
                    let weight = acc_weights[y * w_pad + x];
                    let inv_w = if weight > 1e-12 { 1.0 / weight } else { 0.0 };
                    let img_offset = (y * w_pad + x) * ic;
                    for c_idx in 0..ic {
                        row[x * ic + c_idx] = (acc_img[img_offset + c_idx] * inv_w).clamp(0.0, 1.0);
                    }
                }
            });
    });

    let arr = Array3::from_shape_vec((target_h, target_w, ic), out).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error creando Array3: {:?}", e))
    })?;
    Ok(arr.into_pyarray(py))
}

/// Inserción rápida de baldosa recortada por el centro (estrategia de StarNet v2) directamente in-place.
#[pyfunction]
pub fn place_tile_cropped_inplace_rs<'py>(
    _py: Python<'py>,
    dest_img: &Bound<'py, PyArray3<f32>>,
    tile_out: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
    h_chunk: usize,
    w_chunk: usize,
    border: usize,
) -> PyResult<()> {
    let (dh, dw, dc) = (dest_img.shape()[0], dest_img.shape()[1], dest_img.shape()[2]);

    if y + h_chunk > dh || x + w_chunk > dw {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "El fragmento excede los límites de la imagen destino",
        ));
    }

    let t_view = tile_out.as_array();
    let dest_slice = unsafe {
        dest_img.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("dest_img debe ser C-contiguo: {:?}", e))
        })?
    };

    for dy in 0..h_chunk {
        let iy = y + dy;
        let ty = border + dy;
        let d_row_offset = (iy * dw + x) * dc;
        for dx in 0..w_chunk {
            let tx = border + dx;
            let d_idx = d_row_offset + dx * dc;
            for c_idx in 0..dc {
                dest_slice[d_idx + c_idx] = unsafe { *t_view.uget([ty, tx, c_idx]) };
            }
        }
    }
    Ok(())
}

/// Recompone en paralelo un conjunto de baldosas con la estrategia Center-Crop de StarNet.
#[pyfunction]
pub fn recompose_cropped_tiles_rs<'py>(
    py: Python<'py>,
    tiles: Vec<PyReadonlyArray3<'py, f32>>,
    coords: Vec<(usize, usize)>,
    stride: usize,
    border: usize,
    orig_h: usize,
    orig_w: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if tiles.is_empty() || tiles.len() != coords.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "La lista de baldosas y coordenadas debe ser no vacía y coincidente",
        ));
    }

    let c = tiles[0].as_array().shape()[2];
    let mut out = vec![0.0f32; orig_h * orig_w * c];

    for (tile_ro, &(y, x)) in tiles.iter().zip(coords.iter()) {
        let t_view = tile_ro.as_array();
        let h_chunk = stride.min(orig_h.saturating_sub(y));
        let w_chunk = stride.min(orig_w.saturating_sub(x));

        for dy in 0..h_chunk {
            let iy = y + dy;
            let ty = border + dy;
            let out_row_offset = (iy * orig_w + x) * c;
            for dx in 0..w_chunk {
                let tx = border + dx;
                let out_idx = out_row_offset + dx * c;
                for c_idx in 0..c {
                    out[out_idx + c_idx] = unsafe { *t_view.uget([ty, tx, c_idx]) };
                }
            }
        }
    }

    let arr = Array3::from_shape_vec((orig_h, orig_w, c), out).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error creando Array3: {:?}", e))
    })?;
    Ok(arr.into_pyarray(py))
}

// ============================================================================
// FASE 16: TILING STARNET ONNX Y FACTOR MTF (core/starnet_onnx.py)
// Preparación paralela de baldosas con padding reflectivo y estirado MTF.
// Recomposición con Center-Crop o Hann y reversión analítica MTF con Rayon.
// ============================================================================

/// Calcula el mapeo de coordenadas con reflejo de bordes idéntico a OpenCV BORDER_REFLECT.
#[inline(always)]
fn reflect_border(mut pos: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let m = len as isize;
    let cycle = 2 * m;
    pos = pos.rem_euclid(cycle);
    if pos >= m {
        pos = cycle - 1 - pos;
    }
    pos as usize
}

/// Calcula la mediana de un slice mutable mediante selección in-place O(N).
fn compute_median_slice(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len();
    let mid = n / 2;
    v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let val_mid = v[mid];
    if n % 2 == 0 {
        let val_prev = v[..mid]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        0.5 * (val_mid + val_prev)
    } else {
        val_mid
    }
}

/// Calcula los parámetros de estirado analítico MTF (bp, m) para un canal.
fn calc_channel_mtf(sample: &mut [f32], target_bg: f32, shadows_factor: f32) -> (f32, f32) {
    if sample.is_empty() {
        return (0.0, 0.5);
    }
    let med = compute_median_slice(sample);
    let mut abs_diff: Vec<f32> = sample.iter().map(|&x| (x - med).abs()).collect();
    let mad = compute_median_slice(&mut abs_diff);

    let bp = (med + shadows_factor * 1.4826 * mad).clamp(0.0, 0.95);
    let norm_med = ((med - bp) / (1.0 - bp).max(1e-6)).max(1e-6);
    let denom = norm_med * (2.0 * target_bg - 1.0) - target_bg;
    let m = if denom.abs() > 1e-7 {
        (norm_med * (target_bg - 1.0)) / denom
    } else {
        0.5
    }
    .clamp(0.0050, 0.9999);

    (bp, m)
}

/// Función analítica de curva MTF: s(x) = (m - 1) * x / ((2m - 1) * x - m).
#[inline(always)]
fn mtf_curve(x: f32, m: f32) -> f32 {
    let num = (m - 1.0) * x;
    let den = (2.0 * m - 1.0) * x - m;
    if den.abs() > 1e-7 {
        num / den
    } else {
        0.0
    }
}

/// Aplica el estirado analítico MTF a un píxel.
#[inline(always)]
fn mtf_stretch_pixel(x: f32, bp: f32, m: f32) -> f32 {
    let c_norm = ((x - bp) / (1.0 - bp).max(1e-6)).clamp(0.0, 1.0);
    mtf_curve(c_norm, m).clamp(0.0, 1.0)
}

/// Revierte el estirado analítico MTF a espacio lineal restaurando sombras recortadas.
#[inline(always)]
fn mtf_inverse_pixel(val: f32, ch_orig: f32, bp: f32, m: f32) -> f32 {
    let m_inv = 1.0 - m;
    let s_norm = mtf_curve(val, m_inv);
    let s_unnorm = s_norm * (1.0 - bp) + bp;
    let s_restored = if ch_orig <= bp { ch_orig } else { s_unnorm };
    s_restored.clamp(0.0, 1.0)
}

/// Prepara las baldosas y parámetros MTF de una imagen para inferencia con StarNet v2.
/// Devuelve (tiles_4d, coords, bps, ms).
#[pyfunction]
#[pyo3(signature = (img, tile_size=512, stride=384, is_linear=true, target_bg=0.25, shadows_factor=-2.80))]
pub fn prepare_starnet_tiles_rs<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, f32>,
    tile_size: usize,
    stride: usize,
    is_linear: bool,
    target_bg: f32,
    shadows_factor: f32,
) -> PyResult<(
    Bound<'py, PyArray4<f32>>,
    Vec<(usize, usize, usize, usize)>,
    Vec<f32>,
    Vec<f32>,
)> {
    let img_view = img.as_array();
    let shape = img_view.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "img debe tener forma (H, W, 3)",
        ));
    }
    let (h, w) = (shape[0], shape[1]);
    if h == 0 || w == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Las dimensiones de la imagen deben ser mayores que 0",
        ));
    }

    let actual_tile_size = tile_size.max(32);
    let actual_stride = stride.clamp(16, actual_tile_size.saturating_sub(16)).max(1);
    let border = (actual_tile_size - actual_stride) / 2;

    let mut y_steps = Vec::new();
    let mut y = 0;
    while y < h {
        y_steps.push(y);
        y += actual_stride;
    }
    let mut x_steps = Vec::new();
    let mut x = 0;
    while x < w {
        x_steps.push(x);
        x += actual_stride;
    }

    let mut coords = Vec::with_capacity(y_steps.len() * x_steps.len());
    for &cy in &y_steps {
        let h_chunk = actual_stride.min(h - cy);
        for &cx in &x_steps {
            let w_chunk = actual_stride.min(w - cx);
            coords.push((cy, cx, h_chunk, w_chunk));
        }
    }

    let total_tiles = coords.len();

    // 1. Calcular parámetros MTF por canal si es lineal
    let mut bps = vec![0.0f32; 3];
    let mut ms = vec![0.5f32; 3];

    if is_linear {
        for c in 0..3 {
            let mut sample = Vec::with_capacity((h / 4 + 1) * (w / 4 + 1));
            for sy in (0..h).step_by(4) {
                for sx in (0..w).step_by(4) {
                    sample.push(unsafe { *img_view.uget([sy, sx, c]) });
                }
            }
            let (bp, m) = calc_channel_mtf(&mut sample, target_bg, shadows_factor);
            bps[c] = bp;
            ms[c] = m;
        }
    }

    // 2. Extraer todos los tiles en un buffer plano en paralelo sin GIL
    let mut tiles_data = vec![0.0f32; total_tiles * actual_tile_size * actual_tile_size * 3];

    py.allow_threads(|| {
        tiles_data
            .par_chunks_exact_mut(actual_tile_size * actual_tile_size * 3)
            .enumerate()
            .for_each(|(t_idx, patch_buf)| {
                let (cy, cx, _, _) = coords[t_idx];
                for ty in 0..actual_tile_size {
                    let py_pos = (cy as isize) - (border as isize) + (ty as isize);
                    let src_y = reflect_border(py_pos, h);
                    for tx in 0..actual_tile_size {
                        let px_pos = (cx as isize) - (border as isize) + (tx as isize);
                        let src_x = reflect_border(px_pos, w);
                        let patch_offset = (ty * actual_tile_size + tx) * 3;
                        for c in 0..3 {
                            let mut val = unsafe { *img_view.uget([src_y, src_x, c]) };
                            if is_linear {
                                val = mtf_stretch_pixel(val, bps[c], ms[c]);
                            }
                            patch_buf[patch_offset + c] = val;
                        }
                    }
                }
            });
    });

    let arr = Array4::from_shape_vec(
        (total_tiles, actual_tile_size, actual_tile_size, 3),
        tiles_data,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error creando Array4: {:?}", e))
    })?;

    Ok((arr.into_pyarray(py), coords, bps, ms))
}

enum TileAccessor<'a> {
    Array4D(numpy::ndarray::ArrayView4<'a, f32>),
    List3D(Vec<numpy::ndarray::ArrayView3<'a, f32>>),
}

impl<'a> TileAccessor<'a> {
    #[inline(always)]
    fn get(&self, t_idx: usize, ty: usize, tx: usize, c: usize) -> f32 {
        match self {
            TileAccessor::Array4D(arr) => unsafe { *arr.uget([t_idx, ty, tx, c]) },
            TileAccessor::List3D(list) => unsafe { *list[t_idx].uget([ty, tx, c]) },
        }
    }

    fn len(&self) -> usize {
        match self {
            TileAccessor::Array4D(arr) => arr.shape()[0],
            TileAccessor::List3D(list) => list.len(),
        }
    }
}

/// Recompone la imagen a partir de las baldosas inferidas por StarNet, revirtiendo el estirado MTF si es lineal.
/// Admite tanto reconstrucción Center-Crop (use_hann=false) como mezcla suave Hann (use_hann=true).
#[pyfunction]
#[pyo3(signature = (tiles, coords, orig_h, orig_w, stride=384, tile_size=512, is_linear=true, bps=None, ms=None, orig_img=None, use_hann=false))]
pub fn assemble_starnet_tiles_rs<'py>(
    py: Python<'py>,
    tiles: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    orig_h: usize,
    orig_w: usize,
    stride: usize,
    tile_size: usize,
    is_linear: bool,
    bps: Option<Vec<f32>>,
    ms: Option<Vec<f32>>,
    orig_img: Option<PyReadonlyArray3<'py, f32>>,
    use_hann: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if orig_h == 0 || orig_w == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "orig_h y orig_w deben ser mayores que 0",
        ));
    }

    // Parsear coordenadas (soporta tuplas de 2 o 4 elementos)
    let mut parsed_coords = Vec::new();
    for item in coords.try_iter()? {
        let item = item?;
        if let Ok((y, x, hc, wc)) = item.extract::<(usize, usize, usize, usize)>() {
            parsed_coords.push((y, x, hc, wc));
        } else if let Ok((y, x)) = item.extract::<(usize, usize)>() {
            let hc = stride.min(orig_h.saturating_sub(y));
            let wc = stride.min(orig_w.saturating_sub(x));
            parsed_coords.push((y, x, hc, wc));
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cada coordenada debe ser una tupla de 2 o 4 elementos",
            ));
        }
    }

    // Extraer tiles (soporta 4D o lista de 3D)
    let maybe_arr4 = tiles.extract::<PyReadonlyArray4<'py, f32>>().ok();
    let maybe_list3 = if maybe_arr4.is_none() {
        tiles.extract::<Vec<PyReadonlyArray3<'py, f32>>>().ok()
    } else {
        None
    };

    let tiles_accessor = if let Some(ref arr4) = maybe_arr4 {
        TileAccessor::Array4D(arr4.as_array())
    } else if let Some(ref list3) = maybe_list3 {
        let views: Vec<_> = list3.iter().map(|t| t.as_array()).collect();
        TileAccessor::List3D(views)
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "tiles debe ser un ndarray 4D (N, H, W, 3) o una lista de ndarrays 3D (H, W, 3)",
        ));
    };

    if tiles_accessor.len() != parsed_coords.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Número de baldosas ({}) no coincide con el número de coordenadas ({})",
            tiles_accessor.len(),
            parsed_coords.len()
        )));
    }

    let actual_tile_size = tile_size.max(32);
    let actual_stride = stride.max(1);
    let border = (actual_tile_size.saturating_sub(actual_stride)) / 2;

    let have_mtf = is_linear && bps.is_some() && ms.is_some();
    let bps_vec = bps.unwrap_or_else(|| vec![0.0; 3]);
    let ms_vec = ms.unwrap_or_else(|| vec![0.5; 3]);

    let orig_vec: Option<Vec<f32>> = orig_img.as_ref().map(|o| {
        if let Ok(slice) = o.as_slice() {
            slice.to_vec()
        } else {
            o.as_array().iter().copied().collect()
        }
    });
    let orig_slice = orig_vec.as_deref();

    let mut out_img = vec![0.0f32; orig_h * orig_w * 3];

    if use_hann {
        // Modo Hann 2D: Acumulación con pesos suaves
        let y_w: Vec<f32> = (0..actual_tile_size)
            .map(|y| {
                let s = ((std::f32::consts::PI * (y as f32 + 0.5)) / (actual_tile_size as f32)).sin();
                s * s
            })
            .collect();
        let x_w: Vec<f32> = (0..actual_tile_size)
            .map(|x| {
                let s = ((std::f32::consts::PI * (x as f32 + 0.5)) / (actual_tile_size as f32)).sin();
                s * s
            })
            .collect();

        let mut acc_img = vec![0.0f32; orig_h * orig_w * 3];
        let mut acc_weights = vec![0.0f32; orig_h * orig_w];

        for (t_idx, &(y, x, _, _)) in parsed_coords.iter().enumerate() {
            for ty in 0..actual_tile_size {
                let iy = y + ty;
                if iy >= orig_h {
                    continue;
                }
                let yw = y_w[ty];
                for tx in 0..actual_tile_size {
                    let ix = x + tx;
                    if ix >= orig_w {
                        continue;
                    }
                    let w = yw * x_w[tx];
                    let img_idx = (iy * orig_w + ix) * 3;
                    for c in 0..3 {
                        let val = tiles_accessor.get(t_idx, ty, tx, c);
                        acc_img[img_idx + c] += val * w;
                    }
                    acc_weights[iy * orig_w + ix] += w;
                }
            }
        }

        py.allow_threads(|| {
            out_img
                .par_chunks_exact_mut(orig_w * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..orig_w {
                        let weight = acc_weights[y * orig_w + x];
                        let inv_w = if weight > 1e-12 { 1.0 / weight } else { 0.0 };
                        let px_offset = x * 3;
                        for c in 0..3 {
                            let mut val = acc_img[(y * orig_w + x) * 3 + c] * inv_w;
                            if have_mtf {
                                let ch_orig = if let Some(slice) = orig_slice {
                                    slice[(y * orig_w + x) * 3 + c]
                                } else {
                                    f32::INFINITY
                                };
                                val = mtf_inverse_pixel(val, ch_orig, bps_vec[c], ms_vec[c]);
                            } else {
                                val = val.clamp(0.0, 1.0);
                            }
                            row[px_offset + c] = val;
                        }
                    }
                });
        });
    } else {
        // Modo Center-Crop: Descarte de márgenes inestables
        for (t_idx, &(y, x, h_chunk, w_chunk)) in parsed_coords.iter().enumerate() {
            for dy in 0..h_chunk {
                let iy = y + dy;
                if iy >= orig_h {
                    continue;
                }
                let ty = border + dy;
                let row_offset = (iy * orig_w + x) * 3;
                for dx in 0..w_chunk {
                    let ix = x + dx;
                    if ix >= orig_w {
                        continue;
                    }
                    let tx = border + dx;
                    let px_offset = row_offset + dx * 3;
                    for c in 0..3 {
                        out_img[px_offset + c] = tiles_accessor.get(t_idx, ty, tx, c);
                    }
                }
            }
        }

        py.allow_threads(|| {
            out_img
                .par_chunks_exact_mut(orig_w * 3)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..orig_w {
                        let px_offset = x * 3;
                        for c in 0..3 {
                            let mut val = row[px_offset + c];
                            if have_mtf {
                                let ch_orig = if let Some(slice) = orig_slice {
                                    slice[(y * orig_w + x) * 3 + c]
                                } else {
                                    f32::INFINITY
                                };
                                val = mtf_inverse_pixel(val, ch_orig, bps_vec[c], ms_vec[c]);
                            } else {
                                val = val.clamp(0.0, 1.0);
                            }
                            row[px_offset + c] = val;
                        }
                    }
                });
        });
    }

    let arr = Array3::from_shape_vec((orig_h, orig_w, 3), out_img).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error creando Array3: {:?}", e))
    })?;
    Ok(arr.into_pyarray(py))
}
