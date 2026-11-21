use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ============================================================================
// ÁRBOL K-D 4D NATIVO EN RUST (Búsqueda de Asterismos en Microsegundos)
// ============================================================================

struct KdNode4D {
    point: [f32; 4],
    index: usize,
    left: Option<Box<KdNode4D>>,
    right: Option<Box<KdNode4D>>,
}

struct KdTree4D {
    root: Option<Box<KdNode4D>>,
}

impl KdTree4D {
    fn new(points: &[[f32; 4]]) -> Self {
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build(&mut indices, points, 0);
        Self { root }
    }

    fn build(indices: &mut [usize], points: &[[f32; 4]], depth: usize) -> Option<Box<KdNode4D>> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % 4;
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            points[a][axis].total_cmp(&points[b][axis])
        });
        let idx = indices[mid];
        let (left_indices, right_with_mid) = indices.split_at_mut(mid);
        let right_indices = &mut right_with_mid[1..];

        Some(Box::new(KdNode4D {
            point: points[idx],
            index: idx,
            left: Self::build(left_indices, points, depth + 1),
            right: Self::build(right_indices, points, depth + 1),
        }))
    }

    fn query_ball(&self, target: &[f32; 4], r_sq: f32, depth: usize, results: &mut Vec<usize>) {
        Self::search_ball(&self.root, target, r_sq, depth, results);
    }

    fn search_ball(
        node: &Option<Box<KdNode4D>>,
        target: &[f32; 4],
        r_sq: f32,
        depth: usize,
        results: &mut Vec<usize>,
    ) {
        if let Some(n) = node {
            let axis = depth % 4;
            let mut d_sq = 0.0f32;
            for i in 0..4 {
                let diff = n.point[i] - target[i];
                d_sq += diff * diff;
            }
            if d_sq <= r_sq {
                results.push(n.index);
            }
            let diff_axis = target[axis] - n.point[axis];
            let (first, second) = if diff_axis < 0.0 {
                (&n.left, &n.right)
            } else {
                (&n.right, &n.left)
            };
            Self::search_ball(first, target, r_sq, depth + 1, results);
            if diff_axis * diff_axis <= r_sq {
                Self::search_ball(second, target, r_sq, depth + 1, results);
            }
        }
    }
}

// ============================================================================
// CONSTRUCCIÓN Y EMPAREJAMIENTO DE ASTERISMOS ESTELARES (core/panorama.py)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (pts, k_neighbors=6))]
pub fn build_star_asterisms_rs<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f32>,
    k_neighbors: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
    let pts_view = pts.as_array();
    let n = pts_view.nrows();

    if n < 4 {
        let empty_feats = Array2::<f32>::zeros((0, 4));
        let empty_quads = Array2::<i32>::zeros((0, 4));
        return Ok((empty_feats.into_pyarray(py), empty_quads.into_pyarray(py)));
    }

    let (feats_vec, quads_vec) = py.allow_threads(|| {
        let pts_coords: Vec<[f32; 2]> = (0..n).map(|i| [pts_view[[i, 0]], pts_view[[i, 1]]]).collect();

        // Para cada estrella i, encontrar sus k_neighbors vecinos más cercanos por distancia euclídea
        let mut all_nbrs: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut dist_buf: Vec<(f32, usize)> = Vec::with_capacity(n);

        for i in 0..n {
            dist_buf.clear();
            let pi = pts_coords[i];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = pts_coords[j][0] - pi[0];
                let dy = pts_coords[j][1] - pi[1];
                dist_buf.push((dx * dx + dy * dy, j));
            }
            dist_buf.sort_by(|a, b| a.0.total_cmp(&b.0));
            let k_actual = k_neighbors.min(dist_buf.len());
            let nbrs: Vec<usize> = dist_buf[..k_actual].iter().map(|&(_, j)| j).collect();
            all_nbrs.push(nbrs);
        }

        let mut seen = HashSet::<[usize; 4]>::new();
        let mut feats = Vec::new();
        let mut quads = Vec::new();

        for i in 0..n {
            let nbrs = &all_nbrs[i];
            let len_nbrs = nbrs.len();
            for a in 0..len_nbrs {
                for b in (a + 1)..len_nbrs {
                    for c in (b + 1)..len_nbrs {
                        let mut q_indices = [i, nbrs[a], nbrs[b], nbrs[c]];
                        q_indices.sort_unstable();

                        if !seen.insert(q_indices) {
                            continue;
                        }

                        let q_pts = [
                            pts_coords[q_indices[0]],
                            pts_coords[q_indices[1]],
                            pts_coords[q_indices[2]],
                            pts_coords[q_indices[3]],
                        ];

                        let mut max_d = -1.0f32;
                        let mut best_pair = (0usize, 1usize);
                        for p_i in 0..4 {
                            for p_j in (p_i + 1)..4 {
                                let dx = q_pts[p_i][0] - q_pts[p_j][0];
                                let dy = q_pts[p_i][1] - q_pts[p_j][1];
                                let d = (dx * dx + dy * dy).sqrt();
                                if d > max_d {
                                    max_d = d;
                                    best_pair = (p_i, p_j);
                                }
                            }
                        }

                        let (mut i_a, mut i_b) = best_pair;
                        let other: Vec<usize> = (0..4).filter(|&idx| idx != i_a && idx != i_b).collect();
                        let (mut i_c, mut i_d) = (other[0], other[1]);

                        let mut v_a = q_pts[i_a];
                        let mut v_b = q_pts[i_b];
                        let v_c = q_pts[i_c];
                        let v_d = q_pts[i_d];

                        // Orientación positiva de C respecto a la línea AB
                        let cross = (v_b[0] - v_a[0]) * (v_c[1] - v_a[1]) - (v_b[1] - v_a[1]) * (v_c[0] - v_a[0]);
                        if cross < 0.0 {
                            std::mem::swap(&mut v_a, &mut v_b);
                            std::mem::swap(&mut i_a, &mut i_b);
                        }

                        let dx = v_b[0] - v_a[0];
                        let dy = v_b[1] - v_a[1];
                        let l = dx.hypot(dy);
                        if l < 5.0 {
                            continue;
                        }

                        let cos_th = dx / l;
                        let sin_th = dy / l;

                        let rx_c = (v_c[0] - v_a[0]) / l;
                        let ry_c = (v_c[1] - v_a[1]) / l;
                        let mut x_c = rx_c * cos_th + ry_c * sin_th;
                        let mut y_c = -rx_c * sin_th + ry_c * cos_th;

                        let rx_d = (v_d[0] - v_a[0]) / l;
                        let ry_d = (v_d[1] - v_a[1]) / l;
                        let mut x_d = rx_d * cos_th + ry_d * sin_th;
                        let mut y_d = -rx_d * sin_th + ry_d * cos_th;

                        if x_c > x_d || ((x_c - x_d).abs() < 1e-4 && y_c > y_d) {
                            std::mem::swap(&mut x_c, &mut x_d);
                            std::mem::swap(&mut y_c, &mut y_d);
                            std::mem::swap(&mut i_c, &mut i_d);
                        }

                        feats.push([x_c, y_c, x_d, y_d]);
                        quads.push([
                            q_indices[i_a] as i32,
                            q_indices[i_b] as i32,
                            q_indices[i_c] as i32,
                            q_indices[i_d] as i32,
                        ]);
                    }
                }
            }
        }

        (feats, quads)
    });

    let num_quads = feats_vec.len();
    let mut flat_feats = Vec::with_capacity(num_quads * 4);
    let mut flat_quads = Vec::with_capacity(num_quads * 4);

    for f in &feats_vec {
        flat_feats.extend_from_slice(f);
    }
    for q in &quads_vec {
        flat_quads.extend_from_slice(q);
    }

    let feats_arr = Array2::from_shape_vec((num_quads, 4), flat_feats).unwrap();
    let quads_arr = Array2::from_shape_vec((num_quads, 4), flat_quads).unwrap();

    Ok((feats_arr.into_pyarray(py), quads_arr.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (feats1, quads1, feats2, quads2, tolerance=0.020, min_votes=2))]
pub fn match_star_asterisms_rs<'py>(
    py: Python<'py>,
    feats1: PyReadonlyArray2<'py, f32>,
    quads1: PyReadonlyArray2<'py, i32>,
    feats2: PyReadonlyArray2<'py, f32>,
    quads2: PyReadonlyArray2<'py, i32>,
    tolerance: f32,
    min_votes: u32,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>)> {
    let f1_view = feats1.as_array();
    let q1_view = quads1.as_array();
    let f2_view = feats2.as_array();
    let q2_view = quads2.as_array();

    let m1 = f1_view.nrows();
    let m2 = f2_view.nrows();

    if m1 == 0 || m2 == 0 {
        let empty_m1 = Array1::<i32>::zeros(0);
        let empty_m2 = Array1::<i32>::zeros(0);
        return Ok((empty_m1.into_pyarray(py), empty_m2.into_pyarray(py)));
    }

    let (out_m1, out_m2) = py.allow_threads(|| {
        let pts_feats2: Vec<[f32; 4]> = (0..m2)
            .map(|i| [f2_view[[i, 0]], f2_view[[i, 1]], f2_view[[i, 2]], f2_view[[i, 3]]])
            .collect();

        let kdtree = KdTree4D::new(&pts_feats2);
        let tol_sq = tolerance * tolerance;

        let mut votes: HashMap<(i32, i32), u32> = HashMap::new();
        let mut search_buf = Vec::new();

        for i1 in 0..m1 {
            let f1 = [f1_view[[i1, 0]], f1_view[[i1, 1]], f1_view[[i1, 2]], f1_view[[i1, 3]]];
            search_buf.clear();
            kdtree.query_ball(&f1, tol_sq, 0, &mut search_buf);

            for &i2 in &search_buf {
                for k in 0..4 {
                    let pa = q1_view[[i1, k]];
                    let pb = q2_view[[i2, k]];
                    *votes.entry((pa, pb)).or_insert(0) += 1;
                }
            }
        }

        let mut best_1: HashMap<i32, (i32, u32)> = HashMap::new();
        let mut best_2: HashMap<i32, (i32, u32)> = HashMap::new();

        for (&(s1, s2), &c) in votes.iter() {
            if c >= min_votes {
                match best_1.get(&s1) {
                    Some(&(_, best_c)) if c <= best_c => {}
                    _ => { best_1.insert(s1, (s2, c)); }
                }
                match best_2.get(&s2) {
                    Some(&(_, best_c)) if c <= best_c => {}
                    _ => { best_2.insert(s2, (s1, c)); }
                }
            }
        }

        let mut pairs: Vec<(i32, i32)> = Vec::new();
        for (&s1, &(s2, _)) in best_1.iter() {
            if let Some(&(target_s1, _)) = best_2.get(&s2) {
                if target_s1 == s1 {
                    pairs.push((s1, s2));
                }
            }
        }

        pairs.sort_by_key(|p| (p.0, p.1));
        let (res_m1, res_m2): (Vec<i32>, Vec<i32>) = pairs.into_iter().unzip();
        (res_m1, res_m2)
    });

    let arr_m1 = Array1::from_vec(out_m1);
    let arr_m2 = Array1::from_vec(out_m2);

    Ok((arr_m1.into_pyarray(py), arr_m2.into_pyarray(py)))
}

// ============================================================================
// COMPENSACIÓN RADIAL DE VIÑETEO Y GANANCIA FOTOMÉTRICA (IN-PLACE)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (img_rgb, vignette_amount=0.5, panel_gain=1.0))]
pub fn apply_vignette_and_gain_rs<'py>(
    py: Python<'py>,
    mut img_rgb: PyReadwriteArray3<'py, f32>,
    vignette_amount: f32,
    panel_gain: f32,
) -> PyResult<()> {
    let (h, w, c) = img_rgb.as_array().dim();
    if h == 0 || w == 0 || c == 0 {
        return Ok(());
    }

    if vignette_amount.abs() <= 0.01 && (panel_gain - 1.0).abs() <= 1e-4 {
        return Ok(());
    }

    let slice = img_rgb.as_slice_mut().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("img_rgb must be C-contiguous: {:?}", e))
    })?;

    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let r_max_sq = (cx * cx + cy * cy).max(1.0);
    let inv_r_max_sq = 1.0 / r_max_sq;
    let apply_vig = vignette_amount.abs() > 0.01;
    let vig_factor = vignette_amount;
    let gain = panel_gain;

    py.allow_threads(|| {
        slice
            .par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row)| {
                let dy = y as f32 - cy;
                let dy_sq = dy * dy;

                for x in 0..w {
                    let factor = if apply_vig {
                        let dx = x as f32 - cx;
                        let r_sq = (dx * dx + dy_sq) * inv_r_max_sq;
                        let corr = 1.0 + vig_factor * r_sq + 0.05 * (r_sq * r_sq);
                        gain * corr
                    } else {
                        gain
                    };

                    let base = x * c;
                    for k in 0..c {
                        let val = row[base + k] * factor;
                        row[base + k] = val.clamp(0.0, 1.0);
                    }
                }
            });
    });

    Ok(())
}

// ============================================================================
// DEFORMACIÓN PARABÓLICA DE ARCO / HORIZONTE SIN MALLAS DE COORDENADAS
// ============================================================================

#[inline(always)]
fn cubic_keys_weights(t: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    let w0 = -0.5 * t3 + t2 - 0.5 * t;
    let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    let w3 = 0.5 * t3 - 0.5 * t2;
    (w0, w1, w2, w3)
}

#[pyfunction]
pub fn apply_arch_curvature_rgb_rs<'py>(
    py: Python<'py>,
    image: PyReadonlyArray3<'py, f32>,
    curvature: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (h, w, c) = image.as_array().dim();
    if h == 0 || w == 0 || c == 0 {
        let empty = Array3::<f32>::zeros((0, 0, 0));
        return Ok(empty.into_pyarray(py));
    }

    let in_slice = image.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("image must be C-contiguous: {:?}", e))
    })?;

    if curvature.abs() < 1e-5 {
        let out_arr = Array3::from_shape_vec((h, w, c), in_slice.to_vec()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Shape error: {:?}", e))
        })?;
        return Ok(out_arr.into_pyarray(py));
    }

    // Precalcular el desplazamiento vertical dy por columna (vector 1D de tamaño W)
    let half_w = w as f32 / 2.0;
    let h_f32 = h as f32;
    let dy_cols: Vec<f32> = (0..w)
        .map(|x| {
            let nx = (x as f32 - half_w) / half_w;
            curvature * h_f32 * (1.0 - nx * nx)
        })
        .collect();

    let mut out_vec = vec![0.0f32; h * w * c];

    py.allow_threads(|| {
        out_vec
            .par_chunks_exact_mut(w * c)
            .enumerate()
            .for_each(|(y, row_out)| {
                let y_f = y as f32;

                for x in 0..w {
                    let y_src = y_f - dy_cols[x];
                    if y_src < -2.0 || y_src >= h_f32 + 1.0 {
                        continue;
                    }

                    let y0 = y_src.floor() as isize;
                    let t = y_src - y0 as f32;
                    let (w0, w1, w2, w3) = cubic_keys_weights(t);
                    let ys = [y0 - 1, y0, y0 + 1, y0 + 2];
                    let ws = [w0, w1, w2, w3];

                    let base_out = x * c;
                    for k in 0..c {
                        let mut sum = 0.0f32;
                        for i in 0..4 {
                            let yi = ys[i];
                            if yi >= 0 && yi < h as isize {
                                let idx = (yi as usize * w + x) * c + k;
                                sum += ws[i] * in_slice[idx];
                            }
                        }
                        row_out[base_out + k] = sum.clamp(0.0, 1.0);
                    }
                }
            });
    });

    let out_arr = Array3::from_shape_vec((h, w, c), out_vec).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {:?}", e))
    })?;
    Ok(out_arr.into_pyarray(py))
}

#[pyfunction]
pub fn apply_arch_curvature_mask_rs<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'py, u8>,
    curvature: f32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (h, w) = mask.as_array().dim();
    if h == 0 || w == 0 {
        let empty = Array2::<u8>::zeros((0, 0));
        return Ok(empty.into_pyarray(py));
    }

    let in_slice = mask.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("mask must be C-contiguous: {:?}", e))
    })?;

    if curvature.abs() < 1e-5 {
        let out_arr = Array2::from_shape_vec((h, w), in_slice.to_vec()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Shape error: {:?}", e))
        })?;
        return Ok(out_arr.into_pyarray(py));
    }

    let half_w = w as f32 / 2.0;
    let h_f32 = h as f32;
    let dy_cols: Vec<f32> = (0..w)
        .map(|x| {
            let nx = (x as f32 - half_w) / half_w;
            curvature * h_f32 * (1.0 - nx * nx)
        })
        .collect();

    let mut out_vec = vec![0u8; h * w];

    py.allow_threads(|| {
        out_vec
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                let y_f = y as f32;

                for x in 0..w {
                    let y_src = y_f - dy_cols[x];
                    if y_src < -2.0 || y_src >= h_f32 + 1.0 {
                        continue;
                    }

                    let y0 = y_src.floor() as isize;
                    let t = y_src - y0 as f32;
                    let (w0, w1, w2, w3) = cubic_keys_weights(t);
                    let ys = [y0 - 1, y0, y0 + 1, y0 + 2];
                    let ws = [w0, w1, w2, w3];

                    let mut sum = 0.0f32;
                    for i in 0..4 {
                        let yi = ys[i];
                        if yi >= 0 && yi < h as isize {
                            let idx = yi as usize * w + x;
                            sum += ws[i] * in_slice[idx] as f32;
                        }
                    }
                    row_out[x] = sum.round().clamp(0.0, 255.0) as u8;
                }
            });
    });

    let out_arr = Array2::from_shape_vec((h, w), out_vec).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {:?}", e))
    })?;
    Ok(out_arr.into_pyarray(py))
}
