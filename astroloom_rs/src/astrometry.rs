use nalgebra::{DMatrix, DVector, Matrix3};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

// ============================================================================
// 1. ÁRBOL K-D 2D NATIVO EN RUST (Búsqueda Espacial de Razón de Triángulos)
// ============================================================================

struct KdNode2D {
    point: [f32; 2],
    index: usize,
    left: Option<Box<KdNode2D>>,
    right: Option<Box<KdNode2D>>,
}

struct KdTree2D {
    root: Option<Box<KdNode2D>>,
}

impl KdTree2D {
    fn new(points: &[[f32; 2]]) -> Self {
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build(&mut indices, points, 0);
        Self { root }
    }

    fn build(indices: &mut [usize], points: &[[f32; 2]], depth: usize) -> Option<Box<KdNode2D>> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % 2;
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            points[a][axis].total_cmp(&points[b][axis])
        });
        let idx = indices[mid];
        let (left_indices, right_with_mid) = indices.split_at_mut(mid);
        let right_indices = &mut right_with_mid[1..];

        Some(Box::new(KdNode2D {
            point: points[idx],
            index: idx,
            left: Self::build(left_indices, points, depth + 1),
            right: Self::build(right_indices, points, depth + 1),
        }))
    }

    fn query_ball(&self, target: &[f32; 2], r_sq: f32, depth: usize, results: &mut Vec<usize>) {
        Self::search_ball(&self.root, target, r_sq, depth, results);
    }

    fn search_ball(
        node: &Option<Box<KdNode2D>>,
        target: &[f32; 2],
        r_sq: f32,
        depth: usize,
        results: &mut Vec<usize>,
    ) {
        if let Some(n) = node {
            let axis = depth % 2;
            let dx = n.point[0] - target[0];
            let dy = n.point[1] - target[1];
            let d_sq = dx * dx + dy * dy;
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
// 2. RECORTE DE RECTÁNGULO MÁXIMO INSCRITO O(H x W) (core/panorama.py)
// ============================================================================

#[pyfunction]
pub fn find_max_inscribed_rect_rs<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'py, u8>,
) -> PyResult<(i32, i32, i32, i32)> {
    let mask_view = mask.as_array();
    let (h, w) = mask_view.dim();

    if h == 0 || w == 0 {
        return Ok((0, 0, 0, 0));
    }

    let rect = py.allow_threads(|| {
        let mut heights = vec![0u32; w];
        let mut max_area = 0u64;
        let mut best_rect = (0i32, 0i32, w as i32, h as i32);
        let mut stack = Vec::with_capacity(w + 1);

        for r in 0..h {
            for c in 0..w {
                if unsafe { *mask_view.uget((r, c)) } > 127 {
                    heights[c] += 1;
                } else {
                    heights[c] = 0;
                }
            }

            stack.clear();
            for c in 0..=w {
                let curr_h = if c < w { heights[c] } else { 0 };
                while let Some(&top_idx) = stack.last() {
                    let bar_h = heights[top_idx];
                    if bar_h >= curr_h {
                        stack.pop();
                        let width = if stack.is_empty() {
                            c
                        } else {
                            c - stack.last().unwrap() - 1
                        };
                        let area = (bar_h as u64) * (width as u64);
                        if area > max_area {
                            max_area = area;
                            let top_y = (r + 1 - (bar_h as usize)) as i32;
                            let left_x = if stack.is_empty() {
                                0
                            } else {
                                (stack.last().unwrap() + 1) as i32
                            };
                            best_rect = (left_x, top_y, width as i32, bar_h as i32);
                        }
                    } else {
                        break;
                    }
                }
                stack.push(c);
            }
        }

        best_rect
    });

    Ok(rect)
}

// ============================================================================
// 3. EMPAREJAMIENTO DE TRIÁNGULOS ESTELARES + RANSAC (core/panorama.py)
// ============================================================================

struct StarTriangle {
    feat: [f32; 2],       // [b/a, c/a] invariantes de escala y rotación
    vertices: [usize; 3], // Índices de estrellas ordenadas: opuestas a [a, b, c]
}

#[pyfunction]
#[pyo3(signature = (pts1, pts2, tolerance=0.025, max_ransac_iters=5000, inlier_thresh=5.0))]
pub fn match_star_triangles_ransac_rs<'py>(
    py: Python<'py>,
    pts1: PyReadonlyArray2<'py, f32>,
    pts2: PyReadonlyArray2<'py, f32>,
    tolerance: f32,
    max_ransac_iters: usize,
    inlier_thresh: f32,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<bool>>)> {
    let p1_view = pts1.as_array();
    let p2_view = pts2.as_array();

    let n1 = p1_view.nrows();
    let n2 = p2_view.nrows();

    if n1 < 4 || n2 < 4 {
        let empty_h = Array2::<f64>::eye(3);
        let empty_inl = numpy::ndarray::Array1::<bool>::from_elem(n2, false);
        return Ok((empty_h.into_pyarray(py), empty_inl.into_pyarray(py)));
    }

    let (best_h_mat, inliers_vec) = py.allow_threads(|| {
        let coords1: Vec<[f32; 2]> = (0..n1).map(|i| [p1_view[[i, 0]], p1_view[[i, 1]]]).collect();
        let coords2: Vec<[f32; 2]> = (0..n2).map(|i| [p2_view[[i, 0]], p2_view[[i, 1]]]).collect();

        // 1. Construir triángulos para cada panel utilizando los 7 vecinos más cercanos
        let build_triangles = |pts: &[[f32; 2]]| -> Vec<StarTriangle> {
            let n = pts.len();
            let k = 7.min(n.saturating_sub(1));
            let mut tri_list = Vec::new();
            let mut seen = HashSet::<[usize; 3]>::new();
            let mut dists = Vec::with_capacity(n);

            for i in 0..n {
                dists.clear();
                let pi = pts[i];
                for j in 0..n {
                    if i != j {
                        let dx = pts[j][0] - pi[0];
                        let dy = pts[j][1] - pi[1];
                        dists.push((dx * dx + dy * dy, j));
                    }
                }
                dists.sort_by(|a, b| a.0.total_cmp(&b.0));
                let nbrs_len = k.min(dists.len());
                let nbrs: Vec<usize> = dists[..nbrs_len].iter().map(|&(_, j)| j).collect();

                for a_idx in 0..nbrs_len {
                    for b_idx in (a_idx + 1)..nbrs_len {
                        let j_a = nbrs[a_idx];
                        let j_b = nbrs[b_idx];

                        let mut triplet = [i, j_a, j_b];
                        triplet.sort_unstable();
                        if !seen.insert(triplet) {
                            continue;
                        }

                        let p_i = pts[i];
                        let p_a = pts[j_a];
                        let p_b = pts[j_b];

                        let s_ab = (p_a[0] - p_b[0]).hypot(p_a[1] - p_b[1]); // opuesto a i
                        let s_bi = (p_b[0] - p_i[0]).hypot(p_b[1] - p_i[1]); // opuesto a j_a
                        let s_ia = (p_i[0] - p_a[0]).hypot(p_i[1] - p_a[1]); // opuesto a j_b

                        let mut sides = [
                            (s_ab, i),
                            (s_bi, j_a),
                            (s_ia, j_b),
                        ];
                        // Ordenar lados en orden descendente: a >= b >= c
                        sides.sort_by(|a, b| b.0.total_cmp(&a.0));

                        let side_a = sides[0].0;
                        let side_b = sides[1].0;
                        let side_c = sides[2].0;

                        if side_a < 5.0 || side_c < 1.0 {
                            continue;
                        }

                        let r1 = side_b / side_a;
                        let r2 = side_c / side_a;

                        tri_list.push(StarTriangle {
                            feat: [r1, r2],
                            vertices: [sides[0].1, sides[1].1, sides[2].1],
                        });
                    }
                }
            }
            tri_list
        };

        let tris1 = build_triangles(&coords1);
        let tris2 = build_triangles(&coords2);

        if tris1.is_empty() || tris2.is_empty() {
            return (Matrix3::<f64>::identity(), vec![false; n2]);
        }

        // 2. Indexar triángulos de panel 2 en KD-Tree 2D
        let feats2: Vec<[f32; 2]> = tris2.iter().map(|t| t.feat).collect();
        let kdtree2 = KdTree2D::new(&feats2);
        let tol_sq = tolerance * tolerance;

        // 3. Votación de correspondencias de vértices
        let mut votes: HashMap<(usize, usize), u32> = HashMap::new();
        let mut search_buf = Vec::new();

        for t1 in &tris1 {
            search_buf.clear();
            kdtree2.query_ball(&t1.feat, tol_sq, 0, &mut search_buf);

            for &idx2 in &search_buf {
                let t2 = &tris2[idx2];
                for k in 0..3 {
                    let v1 = t1.vertices[k];
                    let v2 = t2.vertices[k];
                    *votes.entry((v1, v2)).or_insert(0) += 1;
                }
            }
        }

        // 4. Filtrar correspondencias mutuas con al menos 2 votos
        let mut best_1: HashMap<usize, (usize, u32)> = HashMap::new();
        let mut best_2: HashMap<usize, (usize, u32)> = HashMap::new();

        for (&(s1, s2), &count) in &votes {
            if count >= 2 {
                match best_1.get(&s1) {
                    Some(&(_, best_c)) if count <= best_c => {}
                    _ => { best_1.insert(s1, (s2, count)); }
                }
                match best_2.get(&s2) {
                    Some(&(_, best_c)) if count <= best_c => {}
                    _ => { best_2.insert(s2, (s1, count)); }
                }
            }
        }

        let mut candidate_pairs: Vec<(usize, usize)> = Vec::new();
        for (&s1, &(s2, _)) in &best_1 {
            if let Some(&(target_s1, _)) = best_2.get(&s2) {
                if target_s1 == s1 {
                    candidate_pairs.push((s1, s2));
                }
            }
        }

        let num_pairs = candidate_pairs.len();
        if num_pairs < 4 {
            return (Matrix3::<f64>::identity(), vec![false; n2]);
        }

        // 5. Estimador afín / similitud por pares (2 puntos) para RANSAC ultra-rápido y estable
        let solve_similarity_2pts = |p0: usize, p1: usize| -> Option<Matrix3<f64>> {
            let (i1_a, i2_a) = candidate_pairs[p0];
            let (i1_b, i2_b) = candidate_pairs[p1];
            let (x1, y1) = (coords2[i2_a][0] as f64, coords2[i2_a][1] as f64);
            let (x2, y2) = (coords2[i2_b][0] as f64, coords2[i2_b][1] as f64);
            let (u1, v1) = (coords1[i1_a][0] as f64, coords1[i1_a][1] as f64);
            let (u2, v2) = (coords1[i1_b][0] as f64, coords1[i1_b][1] as f64);

            let dx = x2 - x1;
            let dy = y2 - y1;
            let d_sq = dx * dx + dy * dy;
            if d_sq < 10.0 {
                return None;
            }

            let du = u2 - u1;
            let dv = v2 - v1;

            let a = (dx * du + dy * dv) / d_sq;
            let b = (dx * dv - dy * du) / d_sq;
            let tx = u1 - a * x1 + b * y1;
            let ty = v1 - b * x1 - a * y1;

            Some(Matrix3::new(
                a, -b, tx,
                b,  a, ty,
                0.0, 0.0, 1.0,
            ))
        };

        // Generador PRNG xorshift64 determinista y rápido
        let mut rng_state: u64 = 0x853c49e6748fea9b;
        let mut next_rand = || -> usize {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as usize) % num_pairs
        };

        let inlier_thresh_sq = (inlier_thresh * inlier_thresh) as f64;
        let mut best_inlier_count = 0usize;
        let mut best_h = Matrix3::<f64>::identity();
        let mut best_inliers_mask = vec![false; n2];

        let iters = max_ransac_iters.min(5000);

        for _ in 0..iters {
            let s0 = next_rand();
            let s1 = next_rand();
            if s0 == s1 {
                continue;
            }

            if let Some(h_cand) = solve_similarity_2pts(s0, s1) {
                let mut current_inliers = 0usize;
                let mut current_mask = vec![false; n2];

                for &(i1, i2) in &candidate_pairs {
                    let x = coords2[i2][0] as f64;
                    let y = coords2[i2][1] as f64;
                    let u_proj = h_cand[(0, 0)] * x + h_cand[(0, 1)] * y + h_cand[(0, 2)];
                    let v_proj = h_cand[(1, 0)] * x + h_cand[(1, 1)] * y + h_cand[(1, 2)];

                    let du = u_proj - (coords1[i1][0] as f64);
                    let dv = v_proj - (coords1[i1][1] as f64);

                    if du * du + dv * dv <= inlier_thresh_sq {
                        current_inliers += 1;
                        current_mask[i2] = true;
                    }
                }

                if current_inliers > best_inlier_count {
                    best_inlier_count = current_inliers;
                    best_h = h_cand;
                    best_inliers_mask = current_mask;

                    if best_inlier_count as f32 / num_pairs as f32 > 0.90 && best_inlier_count >= 6 {
                        break;
                    }
                }
            }
        }

        // 6. Refinamiento proyectivo por mínimos cuadrados (DLT SVD / Autovalores) sobre todos los inliers
        if best_inlier_count >= 4 {
            let inlier_pairs: Vec<(usize, usize)> = candidate_pairs
                .iter()
                .copied()
                .filter(|&(_, i2)| best_inliers_mask[i2])
                .collect();

            let k_inl = inlier_pairs.len();
            if k_inl >= 4 {
                let mut a_data = vec![0.0f64; 2 * k_inl * 9];
                for (row_i, &(i1, i2)) in inlier_pairs.iter().enumerate() {
                    let (u, v) = (coords1[i1][0] as f64, coords1[i1][1] as f64);
                    let (x, y) = (coords2[i2][0] as f64, coords2[i2][1] as f64);

                    let base0 = (row_i * 2) * 9;
                    a_data[base0 + 0] = x;
                    a_data[base0 + 1] = y;
                    a_data[base0 + 2] = 1.0;
                    a_data[base0 + 6] = -u * x;
                    a_data[base0 + 7] = -u * y;
                    a_data[base0 + 8] = -u;

                    let base1 = (row_i * 2 + 1) * 9;
                    a_data[base1 + 3] = x;
                    a_data[base1 + 4] = y;
                    a_data[base1 + 5] = 1.0;
                    a_data[base1 + 6] = -v * x;
                    a_data[base1 + 7] = -v * y;
                    a_data[base1 + 8] = -v;
                }

                let mat = DMatrix::from_row_slice(2 * k_inl, 9, &a_data);
                let ata = mat.tr_mul(&mat);
                let eig = ata.symmetric_eigen();
                let (min_col, _) = eig.eigenvalues.iter().enumerate().min_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap();
                let h8 = eig.eigenvectors[(8, min_col)];
                if h8.abs() > 1e-9 {
                    let inv_h8 = 1.0 / h8;
                    best_h = Matrix3::new(
                        eig.eigenvectors[(0, min_col)] * inv_h8, eig.eigenvectors[(1, min_col)] * inv_h8, eig.eigenvectors[(2, min_col)] * inv_h8,
                        eig.eigenvectors[(3, min_col)] * inv_h8, eig.eigenvectors[(4, min_col)] * inv_h8, eig.eigenvectors[(5, min_col)] * inv_h8,
                        eig.eigenvectors[(6, min_col)] * inv_h8, eig.eigenvectors[(7, min_col)] * inv_h8, 1.0,
                    );
                }
            }
        }

        (best_h, best_inliers_mask)
    });

    let mut h_arr = Array2::<f64>::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            h_arr[[r, c]] = best_h_mat[(r, c)];
        }
    }
    let inl_arr = numpy::ndarray::Array1::<bool>::from_vec(inliers_vec);

    Ok((h_arr.into_pyarray(py), inl_arr.into_pyarray(py)))
}

// ============================================================================
// 4. DEFORMACIÓN TPS FUSIONADA CON PROYECCIÓN H^-1 (core/stacking.py)
// ============================================================================

#[inline(always)]
fn compute_median_f64(slice: &mut [f64]) -> f64 {
    let n = slice.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, &mut med_hi, _) = slice.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        med_hi
    } else {
        let max_lo = slice[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (max_lo + med_hi)
    }
}

#[inline(always)]
fn compute_median_f32(slice: &mut [f32]) -> f32 {
    let n = slice.len();
    if n == 0 {
        return f32::NAN;
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

#[inline(always)]
fn interpolate_cubic(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    let c2 = ((A + 2.0) * (1.0 - t) - (A + 3.0)) * (1.0 - t) * (1.0 - t) + 1.0;
    let c3 = 1.0 - c0 - c1 - c2;
    [c0, c1, c2, c3]
}

#[pyfunction]
#[pyo3(signature = (src_pts, dst_pts, h_matrix, w, h, grid_size=24, min_inliers=30))]
pub fn compute_tps_remap_grid_rs<'py>(
    py: Python<'py>,
    src_pts: PyReadonlyArray2<'py, f32>,
    dst_pts: PyReadonlyArray2<'py, f32>,
    h_matrix: PyReadonlyArray2<'py, f64>,
    w: usize,
    h: usize,
    grid_size: usize,
    min_inliers: usize,
) -> PyResult<Option<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)>> {
    let src_view = src_pts.as_array();
    let dst_view = dst_pts.as_array();
    let h_view = h_matrix.as_array();

    let n = src_view.nrows();
    if n < min_inliers {
        return Ok(None);
    }

    let h_mat = Matrix3::new(
        h_view[[0, 0]], h_view[[0, 1]], h_view[[0, 2]],
        h_view[[1, 0]], h_view[[1, 1]], h_view[[1, 2]],
        h_view[[2, 0]], h_view[[2, 1]], h_view[[2, 2]],
    );

    let h_inv = match h_mat.try_inverse() {
        Some(inv) => inv,
        None => return Ok(None),
    };

    let result = py.allow_threads(|| -> Option<(Vec<f32>, Vec<f32>)> {
        // 1. Calcular residuos entre puntos destino proyectados y puntos origen
        let mut residuals = Vec::with_capacity(n);
        let mut dst_coords = Vec::with_capacity(n);
        let mut max_res = 0.0f32;

        for i in 0..n {
            let x_d = dst_view[[i, 0]] as f64;
            let y_d = dst_view[[i, 1]] as f64;
            let x_s = src_view[[i, 0]];
            let y_s = src_view[[i, 1]];

            let denom = (h_inv[(2, 0)] * x_d + h_inv[(2, 1)] * y_d + h_inv[(2, 2)]).max(1e-6);
            let x_proj = ((h_inv[(0, 0)] * x_d + h_inv[(0, 1)] * y_d + h_inv[(0, 2)]) / denom) as f32;
            let y_proj = ((h_inv[(1, 0)] * x_d + h_inv[(1, 1)] * y_d + h_inv[(1, 2)]) / denom) as f32;

            let rx = x_s - x_proj;
            let ry = y_s - y_proj;
            let norm = rx.hypot(ry);
            if norm > max_res {
                max_res = norm;
            }
            residuals.push([rx, ry]);
            dst_coords.push([x_d as f32, y_d as f32]);
        }

        // Filtro estricto: solo aplicar TPS si los residuos están entre 0.35 y 10.0 px
        if max_res < 0.35 || max_res > 10.0 {
            return None;
        }

        // 2. Binning espacial 6x6 regular
        let grid_bins = 6;
        let mut bin_centers = Vec::new();
        let mut bin_residuals = Vec::new();

        let step_x = (w as f32) / (grid_bins as f32);
        let step_y = (h as f32) / (grid_bins as f32);

        for bi in 0..grid_bins {
            let y0 = (bi as f32) * step_y;
            let y1 = ((bi + 1) as f32) * step_y;
            for bj in 0..grid_bins {
                let x0 = (bj as f32) * step_x;
                let x1 = ((bj + 1) as f32) * step_x;

                let mut bin_rx = Vec::new();
                let mut bin_ry = Vec::new();

                for k in 0..n {
                    let x = dst_coords[k][0];
                    let y = dst_coords[k][1];
                    if x >= x0 && x < x1 && y >= y0 && y < y1 {
                        bin_rx.push(residuals[k][0]);
                        bin_ry.push(residuals[k][1]);
                    }
                }

                if !bin_rx.is_empty() {
                    let med_x = compute_median_f32(&mut bin_rx);
                    let med_y = compute_median_f32(&mut bin_ry);

                    bin_centers.push([0.5 * (x0 + x1), 0.5 * (y0 + y1)]);
                    bin_residuals.push([med_x, med_y]);
                }
            }
        }

        if bin_centers.len() < 12 {
            return None;
        }

        // 3. Resolver sistema TPS Thin Plate Spline con regularización
        let m = bin_centers.len();
        let sys_size = m + 3;
        let shift_x = 0.5 * (w as f64);
        let shift_y = 0.5 * (h as f64);
        let scale_x = 0.5 * (w as f64);
        let scale_y = 0.5 * (h as f64);

        let mut mat = DMatrix::<f64>::zeros(sys_size, sys_size);
        for i in 0..m {
            let xi = bin_centers[i][0] as f64;
            let yi = bin_centers[i][1] as f64;

            for j in 0..m {
                if i != j {
                    let xj = bin_centers[j][0] as f64;
                    let yj = bin_centers[j][1] as f64;
                    let dx = xi - xj;
                    let dy = yi - yj;
                    let r_sq = dx * dx + dy * dy;
                    if r_sq > 1e-24 {
                        mat[(i, j)] = 0.5 * r_sq * r_sq.ln();
                    }
                } else {
                    mat[(i, j)] = 2.0; // Smoothing factor = 2.0
                }
            }

            let norm_x = (xi - shift_x) / scale_x;
            let norm_y = (yi - shift_y) / scale_y;
            mat[(i, m)] = 1.0;
            mat[(i, m + 1)] = norm_x;
            mat[(i, m + 2)] = norm_y;

            mat[(m, i)] = 1.0;
            mat[(m + 1, i)] = norm_x;
            mat[(m + 2, i)] = norm_y;
        }

        let lu = mat.full_piv_lu();
        let mut rhs_x = DVector::<f64>::zeros(sys_size);
        let mut rhs_y = DVector::<f64>::zeros(sys_size);
        for i in 0..m {
            rhs_x[i] = bin_residuals[i][0] as f64;
            rhs_y[i] = bin_residuals[i][1] as f64;
        }

        let sol_x = lu.solve(&rhs_x)?;
        let sol_y = lu.solve(&rhs_y)?;

        let mut weights_x = vec![0.0f64; m];
        let mut weights_y = vec![0.0f64; m];
        for i in 0..m {
            weights_x[i] = sol_x[i];
            weights_y[i] = sol_y[i];
        }
        let poly_x = [sol_x[m], sol_x[m + 1], sol_x[m + 2]];
        let poly_y = [sol_y[m], sol_y[m + 1], sol_y[m + 2]];

        let pts_x: Vec<f64> = (0..m).map(|i| bin_centers[i][0] as f64).collect();
        let pts_y: Vec<f64> = (0..m).map(|i| bin_centers[i][1] as f64).collect();

        // 4. Evaluar rejilla de desplazamiento TPS de grid_size x grid_size
        let gw = grid_size;
        let gh = grid_size;
        let inv_gw = if gw > 1 { 1.0 / ((gw - 1) as f64) } else { 0.0 };
        let inv_gh = if gh > 1 { 1.0 / ((gh - 1) as f64) } else { 0.0 };

        let mut disp_grid_x = vec![0.0f32; gw * gh];
        let mut disp_grid_y = vec![0.0f32; gw * gh];

        for iy in 0..gh {
            let gy = (iy as f64) * inv_gh * ((h - 1) as f64);
            let norm_y = (gy - shift_y) / scale_y;
            for ix in 0..gw {
                let gx = (ix as f64) * inv_gw * ((w - 1) as f64);
                let norm_x = (gx - shift_x) / scale_x;

                let mut val_x = poly_x[0] + poly_x[1] * norm_x + poly_x[2] * norm_y;
                let mut val_y = poly_y[0] + poly_y[1] * norm_x + poly_y[2] * norm_y;

                for i in 0..m {
                    let dx = gx - pts_x[i];
                    let dy = gy - pts_y[i];
                    let r_sq = dx * dx + dy * dy;
                    if r_sq > 1e-24 {
                        let phi = 0.5 * r_sq * r_sq.ln();
                        val_x += phi * weights_x[i];
                        val_y += phi * weights_y[i];
                    }
                }

                if val_x.abs() > 10.0 || val_y.abs() > 10.0 || val_x.is_nan() || val_y.is_nan() {
                    return None;
                }

                disp_grid_x[iy * gw + ix] = val_x as f32;
                disp_grid_y[iy * gw + ix] = val_y as f32;
            }
        }

        // 5. Interpolar bicúbico en paralelo y fusionar directamente con proyección H^-1
        let mut map_x = vec![0.0f32; w * h];
        let mut map_y = vec![0.0f32; w * h];

        let scale_grid_x = (gw as f32) / (w as f32);
        let scale_grid_y = (gh as f32) / (h as f32);

        map_x.par_chunks_mut(w)
            .zip(map_y.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (row_mx, row_my))| {
                let y_f = y as f32;
                let y_real = (y_f + 0.5) * scale_grid_y - 0.5;
                let y0 = y_real.floor() as isize;
                let ty = y_real - (y0 as f32);
                let wy = interpolate_cubic(ty);

                for x in 0..w {
                    let x_f = x as f32;
                    let x_real = (x_f + 0.5) * scale_grid_x - 0.5;
                    let x0 = x_real.floor() as isize;
                    let tx = x_real - (x0 as f32);
                    let wx = interpolate_cubic(tx);

                    // Muestreo bicúbico 4x4
                    let mut dx_val = 0.0f32;
                    let mut dy_val = 0.0f32;

                    for i in 0..4 {
                        let py = (y0 - 1 + (i as isize)).clamp(0, (gh - 1) as isize) as usize;
                        let row_offset = py * gw;
                        let wy_i = wy[i];

                        for j in 0..4 {
                            let px = (x0 - 1 + (j as isize)).clamp(0, (gw - 1) as isize) as usize;
                            let w_ij = wy_i * wx[j];
                            dx_val += disp_grid_x[row_offset + px] * w_ij;
                            dy_val += disp_grid_y[row_offset + px] * w_ij;
                        }
                    }

                    // Proyección proyectiva inversa H^-1 * [x, y, 1]
                    let x_d = x as f64;
                    let y_d = y as f64;
                    let denom = (h_inv[(2, 0)] * x_d + h_inv[(2, 1)] * y_d + h_inv[(2, 2)]).max(1e-6);
                    let inv_denom = 1.0 / denom;
                    let proj_x = (h_inv[(0, 0)] * x_d + h_inv[(0, 1)] * y_d + h_inv[(0, 2)]) * inv_denom;
                    let proj_y = (h_inv[(1, 0)] * x_d + h_inv[(1, 1)] * y_d + h_inv[(1, 2)]) * inv_denom;

                    row_mx[x] = (proj_x as f32) + dx_val;
                    row_my[x] = (proj_y as f32) + dy_val;
                }
            });

        Some((map_x, map_y))
    });

    match result {
        Some((mx_vec, my_vec)) => {
            let mx_arr = Array2::from_shape_vec((h, w), mx_vec).unwrap();
            let my_arr = Array2::from_shape_vec((h, w), my_vec).unwrap();
            Ok(Some((mx_arr.into_pyarray(py), my_arr.into_pyarray(py))))
        }
        None => Ok(None),
    }
}

// ============================================================================
// 5. ANÁLISIS DE CALIDAD ÓPTICA 3x3 Y MOMENTOS ESTELARES (core/frame_culling.py)
// ============================================================================

fn evaluate_optical_diagnosis_from_data<'py>(
    py: Python<'py>,
    star_profiles: &[[f32; 4]], // [cx, cy, fwhm, ecc]
    h: usize,
    w: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mut fwhm_grid_vals: [[Vec<f32>; 3]; 3] = Default::default();
    let mut ecc_grid_vals: [[Vec<f32>; 3]; 3] = Default::default();

    let cell_h = (h as f32 / 3.0).max(1.0);
    let cell_w = (w as f32 / 3.0).max(1.0);

    for prof in star_profiles {
        let cx = prof[0];
        let cy = prof[1];
        let fwhm = prof[2];
        let ecc = prof[3];

        let r = ((cy / cell_h) as usize).clamp(0, 2);
        let c = ((cx / cell_w) as usize).clamp(0, 2);

        fwhm_grid_vals[r][c].push(fwhm);
        ecc_grid_vals[r][c].push(ecc);
    }

    let mut grid_fwhm = [[f64::NAN; 3]; 3];
    let mut grid_ecc = [[f64::NAN; 3]; 3];
    let mut grid_counts = [[0usize; 3]; 3];

    let mut valid_cells = 0usize;
    for r in 0..3 {
        for c in 0..3 {
            let cnt = fwhm_grid_vals[r][c].len();
            grid_counts[r][c] = cnt;
            if cnt >= 2 {
                let med_fwhm = compute_median_f32(&mut fwhm_grid_vals[r][c]) as f64;
                let med_ecc = compute_median_f32(&mut ecc_grid_vals[r][c]) as f64;

                grid_fwhm[r][c] = (med_fwhm * 100.0).round() / 100.0;
                grid_ecc[r][c] = (med_ecc * 1000.0).round() / 1000.0;
                valid_cells += 1;
            }
        }
    }

    let dict = PyDict::new(py);

    let build_float_grid = |arr: &[[f64; 3]; 3]| -> PyResult<Bound<'py, PyList>> {
        let outer = PyList::empty(py);
        for r in 0..3 {
            let row = PyList::empty(py);
            for c in 0..3 {
                row.append(arr[r][c])?;
            }
            outer.append(row)?;
        }
        Ok(outer)
    };

    let build_int_grid = |arr: &[[usize; 3]; 3]| -> PyResult<Bound<'py, PyList>> {
        let outer = PyList::empty(py);
        for r in 0..3 {
            let row = PyList::empty(py);
            for c in 0..3 {
                row.append(arr[r][c])?;
            }
            outer.append(row)?;
        }
        Ok(outer)
    };

    dict.set_item("grid_fwhm", build_float_grid(&grid_fwhm)?)?;
    dict.set_item("grid_ecc", build_float_grid(&grid_ecc)?)?;
    dict.set_item("grid_counts", build_int_grid(&grid_counts)?)?;

    if star_profiles.len() < 6 || valid_cells < 3 {
        dict.set_item("curvature_ratio", 0.0f64)?;
        dict.set_item("tilt_x", 0.0f64)?;
        dict.set_item("tilt_y", 0.0f64)?;
        dict.set_item("total_tilt", 0.0f64)?;
        dict.set_item("defect_type", "INSUFFICIENT_DATA")?;
        dict.set_item("diagnosis_msg", "Estrellas insuficientes para diagnóstico de campo")?;
        return Ok(dict);
    }

    // 1. Curvatura de campo (Centro vs 4 Esquinas)
    let mut f_center = grid_fwhm[1][1];
    if f_center.is_nan() {
        let mut adj = Vec::new();
        for &(r, c) in &[(0, 1), (1, 0), (1, 2), (2, 1)] {
            if !grid_fwhm[r][c].is_nan() {
                adj.push(grid_fwhm[r][c]);
            }
        }
        if !adj.is_empty() {
            f_center = compute_median_f64(&mut adj);
        }
    }

    let mut corner_fwhms = Vec::new();
    for &(r, c) in &[(0, 0), (0, 2), (2, 0), (2, 2)] {
        if !grid_fwhm[r][c].is_nan() {
            corner_fwhms.push(grid_fwhm[r][c]);
        }
    }

    let (curvature_ratio, corner_spread) = if corner_fwhms.len() >= 2 && !f_center.is_nan() && f_center > 0.5 {
        let med_corner = compute_median_f64(&mut corner_fwhms);
        let curv = ((med_corner - f_center) / f_center).max(0.0);

        let max_c = corner_fwhms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_c = corner_fwhms.iter().copied().fold(f64::INFINITY, f64::min);
        let spread = (max_c - min_c) / med_corner.max(1e-3);
        (curv, spread)
    } else {
        (0.0, 0.0)
    };

    // 2. Inclinación de sensor (Tilt horizontal y vertical)
    let mut left_vals = Vec::new();
    let mut right_vals = Vec::new();
    for r in 0..3 {
        if !grid_fwhm[r][0].is_nan() {
            left_vals.push(grid_fwhm[r][0]);
        }
        if !grid_fwhm[r][2].is_nan() {
            right_vals.push(grid_fwhm[r][2]);
        }
    }

    let tilt_x = if !left_vals.is_empty() && !right_vals.is_empty() {
        let med_l = compute_median_f64(&mut left_vals);
        let med_r = compute_median_f64(&mut right_vals);

        (med_l - med_r).abs() / med_l.min(med_r).max(0.5)
    } else {
        0.0
    };

    let mut top_vals = Vec::new();
    let mut bot_vals = Vec::new();
    for c in 0..3 {
        if !grid_fwhm[0][c].is_nan() {
            top_vals.push(grid_fwhm[0][c]);
        }
        if !grid_fwhm[2][c].is_nan() {
            bot_vals.push(grid_fwhm[2][c]);
        }
    }

    let tilt_y = if !top_vals.is_empty() && !bot_vals.is_empty() {
        let med_t = compute_median_f64(&mut top_vals);
        let med_b = compute_median_f64(&mut bot_vals);

        (med_t - med_b).abs() / med_t.min(med_b).max(0.5)
    } else {
        0.0
    };

    let total_tilt = tilt_x.hypot(tilt_y);

    // 3. Clasificación diagnóstica
    const TILT_THRESH: f64 = 0.18;
    const CURV_THRESH: f64 = 0.20;

    let (defect_type, diagnosis_msg) = if total_tilt >= TILT_THRESH && curvature_ratio >= CURV_THRESH {
        (
            "COMBINED",
            format!("Curvatura (+{:.0}%) e Inclinación/Tilt ({:.0}%)", curvature_ratio * 100.0, total_tilt * 100.0),
        )
    } else if total_tilt >= TILT_THRESH {
        let axis = if tilt_x >= tilt_y { "horizontal (izq/der)" } else { "vertical (sup/inf)" };
        (
            "SENSOR_TILT",
            format!("Inclinación/Tilt mecánico {} ({:.0}%)", axis, total_tilt * 100.0),
        )
    } else if curvature_ratio >= CURV_THRESH && corner_spread <= 0.25 {
        (
            "FIELD_CURVATURE",
            format!("Curvatura de campo simétrica (+{:.0}% FWHM esquinas)", curvature_ratio * 100.0),
        )
    } else if curvature_ratio >= 0.25 && corner_spread > 0.25 {
        (
            "SENSOR_TILT",
            "Degradación asimétrica de esquinas (posible flexión/tilt)".to_string(),
        )
    } else {
        (
            "OK",
            "Óptica bien corregida (campo plano y simétrico)".to_string(),
        )
    };

    dict.set_item("curvature_ratio", (curvature_ratio * 1000.0).round() / 1000.0)?;
    dict.set_item("tilt_x", (tilt_x * 1000.0).round() / 1000.0)?;
    dict.set_item("tilt_y", (tilt_y * 1000.0).round() / 1000.0)?;
    dict.set_item("total_tilt", (total_tilt * 1000.0).round() / 1000.0)?;
    dict.set_item("defect_type", defect_type)?;
    dict.set_item("diagnosis_msg", diagnosis_msg)?;

    Ok(dict)
}

#[pyfunction]
pub fn compute_3x3_optical_diagnosis_rs<'py>(
    py: Python<'py>,
    star_profiles: PyReadonlyArray2<'py, f32>,
    h: usize,
    w: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let prof_view = star_profiles.as_array();
    let n = prof_view.nrows();

    let mut profiles_vec = Vec::with_capacity(n);
    for i in 0..n {
        profiles_vec.push([
            prof_view[[i, 0]],
            prof_view[[i, 1]],
            prof_view[[i, 2]],
            prof_view[[i, 3]],
        ]);
    }

    evaluate_optical_diagnosis_from_data(py, &profiles_vec, h, w)
}

#[pyfunction]
#[pyo3(signature = (gray, centroids, bg_median, h, w, patch_radius=6))]
pub fn analyze_frame_quality_3x3_rs<'py>(
    py: Python<'py>,
    gray: PyReadonlyArray2<'py, f32>,
    centroids: PyReadonlyArray2<'py, f32>,
    bg_median: f32,
    h: usize,
    w: usize,
    patch_radius: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyDict>)> {
    let gray_view = gray.as_array();
    let cent_view = centroids.as_array();
    let n = cent_view.nrows();
    let r = patch_radius as isize;

    let valid_profiles: Vec<[f32; 4]> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let cx = cent_view[[i, 0]];
                let cy = cent_view[[i, 1]];
                let ix = cx.round() as isize;
                let iy = cy.round() as isize;

                if iy < r || iy + r >= (h as isize) || ix < r || ix + r >= (w as isize) {
                    return None;
                }

                let mut m00 = 0.0f32;
                let mut m10 = 0.0f32;
                let mut m01 = 0.0f32;

                for dy in -r..=r {
                    let py_idx = (iy + dy) as usize;
                    for dx in -r..=r {
                        let px_idx = (ix + dx) as usize;
                        let val = (unsafe { *gray_view.uget((py_idx, px_idx)) } - bg_median).max(0.0);
                        m00 += val;
                        m10 += (dx as f32) * val;
                        m01 += (dy as f32) * val;
                    }
                }

                if m00 <= 1e-4 {
                    return None;
                }

                let inv_m00 = 1.0 / m00;
                let cx_sub = m10 * inv_m00;
                let cy_sub = m01 * inv_m00;

                let mut mu20 = 0.0f32;
                let mut mu02 = 0.0f32;
                let mut mu11 = 0.0f32;

                for dy in -r..=r {
                    let py_idx = (iy + dy) as usize;
                    let y_rel = (dy as f32) - cy_sub;
                    for dx in -r..=r {
                        let px_idx = (ix + dx) as usize;
                        let x_rel = (dx as f32) - cx_sub;
                        let val = (unsafe { *gray_view.uget((py_idx, px_idx)) } - bg_median).max(0.0);
                        mu20 += x_rel * x_rel * val;
                        mu02 += y_rel * y_rel * val;
                        mu11 += x_rel * y_rel * val;
                    }
                }

                mu20 *= inv_m00;
                mu02 *= inv_m00;
                mu11 *= inv_m00;

                let delta = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).max(0.0).sqrt();
                let lambda1 = (((mu20 + mu02) + delta) * 0.5).max(1e-4);
                let lambda2 = (((mu20 + mu02) - delta) * 0.5).max(1e-4);

                let ratio = (lambda2 / lambda1).clamp(0.0, 1.0);
                let ecc = (1.0 - ratio).sqrt();
                let sigma_mean = (((lambda1 + lambda2) * 0.5)).sqrt();
                let fwhm = 2.35482 * sigma_mean;

                if fwhm >= 0.5 && fwhm <= 25.0 {
                    Some([cx, cy, fwhm, ecc])
                } else {
                    None
                }
            })
            .collect()
    });

    let dict = evaluate_optical_diagnosis_from_data(py, &valid_profiles, h, w)?;

    let num_valid = valid_profiles.len();
    let mut flat_prof = Vec::with_capacity(num_valid * 4);
    for p in &valid_profiles {
        flat_prof.extend_from_slice(p);
    }

    let prof_arr = Array2::from_shape_vec((num_valid, 4), flat_prof).unwrap();
    Ok((prof_arr.into_pyarray(py), dict))
}
