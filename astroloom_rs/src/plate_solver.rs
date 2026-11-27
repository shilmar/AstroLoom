use nalgebra::{Matrix3, Vector3};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::{Arc, Mutex, OnceLock};

const MAGIC: &[u8; 8] = b"ALCAT001";
const X_MIN: f32 = -0.5;
const X_MAX: f32 = 0.5;
const Y_MIN: f32 = -0.9;
const Y_MAX: f32 = 0.9;

#[derive(Clone, Copy)]
struct StarRec {
    id: u32,
    u: [f32; 3],
    mag_milli: i16,
    bp_rp_milli: i16,
    #[allow(dead_code)]
    cell_id: u32,
}

#[derive(Clone, Copy)]
struct QuadRec {
    star_idx: [u32; 4],
    code: [i16; 4],
    scale_mrad: u16,
}

struct PlateCatalog {
    profile_id: u8,
    hash_bins: usize,
    stars: Vec<StarRec>,
    grid_offsets: Vec<u32>,
    quads: Vec<QuadRec>,
    hash_offsets: Vec<u32>,
    cell_centers: Vec<[f32; 3]>,
    cell_max_cos_radius: f32,
}

static CATALOG_CACHE: OnceLock<Mutex<HashMap<String, Arc<PlateCatalog>>>> = OnceLock::new();

fn read_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn read_i16_le(buf: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([buf[off], buf[off + 1]])
}

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_u64_le(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off],
        buf[off + 1],
        buf[off + 2],
        buf[off + 3],
        buf[off + 4],
        buf[off + 5],
        buf[off + 6],
        buf[off + 7],
    ])
}

fn read_f32_le(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn cell_center_unit(cell: usize, nside: usize) -> [f32; 3] {
    let face_size = nside * nside;
    let face = cell / face_size;
    let rem = cell % face_size;
    let iv = rem / nside;
    let iu = rem % nside;
    let u = ((iu as f32 + 0.5) / nside as f32) * 2.0 - 1.0;
    let v = ((iv as f32 + 0.5) / nside as f32) * 2.0 - 1.0;
    let (x, y, z) = match face {
        0 => (1.0f32, u, v),
        1 => (-1.0f32, -u, v),
        2 => (-u, 1.0f32, v),
        3 => (u, -1.0f32, v),
        4 => (u, v, 1.0f32),
        _ => (u, -v, -1.0f32),
    };
    let inv = 1.0 / (x * x + y * y + z * z).sqrt();
    [x * inv, y * inv, z * inv]
}

impl PlateCatalog {
    fn load(path: &str) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("No se pudo leer catálogo '{}': {}", path, e))?;
        if data.len() < 64 {
            return Err("Archivo ALCAT001 truncado (<64 bytes)".to_string());
        }
        if &data[0..8] != MAGIC {
            return Err("Cabecera inválida: se esperaba ALCAT001".to_string());
        }
        let version = read_u32_le(&data, 8);
        if version != 1 {
            return Err(format!("Versión ALCAT no soportada: {}", version));
        }
        let profile_id = data[12];
        let grid_nside = read_u16_le(&data, 14) as usize;
        let hash_bins = read_u16_le(&data, 16) as usize;
        let star_count = read_u32_le(&data, 20) as usize;
        let quad_count = read_u32_le(&data, 24) as usize;
        let stars_offset = read_u64_le(&data, 32) as usize;
        let grid_offsets_offset = read_u64_le(&data, 40) as usize;
        let quads_offset = read_u64_le(&data, 48) as usize;
        let hash_offsets_offset = read_u64_le(&data, 56) as usize;

        let n_cells = 6 * grid_nside * grid_nside;
        let n_buckets = hash_bins.pow(4);

        let min_len = hash_offsets_offset
            .checked_add((n_buckets + 1) * 4)
            .ok_or_else(|| "Overflow en tamaño de catálogo".to_string())?;
        if data.len() < min_len {
            return Err(format!(
                "Catálogo truncado: {} bytes, esperados al menos {}",
                data.len(),
                min_len
            ));
        }

        let mut stars = Vec::with_capacity(star_count);
        for i in 0..star_count {
            let o = stars_offset + i * 24;
            stars.push(StarRec {
                id: read_u32_le(&data, o),
                u: [
                    read_f32_le(&data, o + 4),
                    read_f32_le(&data, o + 8),
                    read_f32_le(&data, o + 12),
                ],
                mag_milli: read_i16_le(&data, o + 16),
                bp_rp_milli: read_i16_le(&data, o + 18),
                cell_id: read_u32_le(&data, o + 20),
            });
        }

        let mut grid_offsets = Vec::with_capacity(n_cells + 1);
        for i in 0..=n_cells {
            grid_offsets.push(read_u32_le(&data, grid_offsets_offset + i * 4));
        }

        let mut quads = Vec::with_capacity(quad_count);
        for i in 0..quad_count {
            let o = quads_offset + i * 28;
            quads.push(QuadRec {
                star_idx: [
                    read_u32_le(&data, o),
                    read_u32_le(&data, o + 4),
                    read_u32_le(&data, o + 8),
                    read_u32_le(&data, o + 12),
                ],
                code: [
                    read_i16_le(&data, o + 16),
                    read_i16_le(&data, o + 18),
                    read_i16_le(&data, o + 20),
                    read_i16_le(&data, o + 22),
                ],
                scale_mrad: read_u16_le(&data, o + 24),
            });
        }

        let mut hash_offsets = Vec::with_capacity(n_buckets + 1);
        for i in 0..=n_buckets {
            hash_offsets.push(read_u32_le(&data, hash_offsets_offset + i * 4));
        }

        let cell_centers: Vec<[f32; 3]> = (0..n_cells)
            .map(|c| cell_center_unit(c, grid_nside))
            .collect();
        // Radio angular máximo de una celda cube-sphere (~sqrt(2)/nside rad)
        let cell_radius_rad = (2.0f32.sqrt() / (grid_nside as f32)) * 1.15;

        Ok(Self {
            profile_id,
            hash_bins,
            stars,
            grid_offsets,
            quads,
            hash_offsets,
            cell_centers,
            cell_max_cos_radius: cell_radius_rad,
        })
    }

    fn stars_in_cone(&self, center: &[f64; 3], radius_rad: f64, max_mag_milli: i16) -> Vec<usize> {
        let cos_cell = (radius_rad + self.cell_max_cos_radius as f64).min(std::f64::consts::PI).cos();
        let cos_star = radius_rad.min(std::f64::consts::PI).cos();
        let mut out = Vec::with_capacity(512);
        let n_cells = self.cell_centers.len();
        for c in 0..n_cells {
            let cc = self.cell_centers[c];
            let dot_c = cc[0] as f64 * center[0] + cc[1] as f64 * center[1] + cc[2] as f64 * center[2];
            if dot_c < cos_cell {
                continue;
            }
            let s0 = self.grid_offsets[c] as usize;
            let s1 = self.grid_offsets[c + 1] as usize;
            for idx in s0..s1 {
                let st = &self.stars[idx];
                if st.mag_milli > max_mag_milli {
                    // Estrellas están ordenadas por magnitud ascendente dentro de cada celda
                    break;
                }
                let dot = st.u[0] as f64 * center[0]
                    + st.u[1] as f64 * center[1]
                    + st.u[2] as f64 * center[2];
                if dot >= cos_star {
                    out.push(idx);
                }
            }
        }
        out
    }
}

fn get_or_load_catalog(path: &str) -> Result<Arc<PlateCatalog>, String> {
    let cache = CATALOG_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().unwrap();
        if let Some(cat) = guard.get(path) {
            return Ok(Arc::clone(cat));
        }
    }
    let loaded = Arc::new(PlateCatalog::load(path)?);
    let mut guard = cache.lock().unwrap();
    guard.insert(path.to_string(), Arc::clone(&loaded));
    Ok(loaded)
}

// ============================================================================
// DETECCIÓN PARALELA DE CENTROIDES SUBPÍXEL (float32 + máscara de cielo)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (gray, sky_mask=None, max_stars=180, min_separation=10, sigma_thresh=4.5))]
pub fn detect_plate_stars_rs<'py>(
    py: Python<'py>,
    gray: PyReadonlyArray2<'py, f32>,
    sky_mask: Option<PyReadonlyArray2<'py, f32>>,
    max_stars: usize,
    min_separation: usize,
    sigma_thresh: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let g = gray.as_array();
    let (h, w) = g.dim();
    if h < 16 || w < 16 {
        return Ok(Array2::<f32>::zeros((0, 4)).into_pyarray(py));
    }

    let mask_view = sky_mask.as_ref().map(|m| m.as_array());
    if let Some(ref mv) = mask_view {
        if mv.dim() != (h, w) {
            return Err(PyValueError::new_err(
                "sky_mask debe tener las mismas dimensiones que gray",
            ));
        }
    }

    let stars_vec: Vec<[f32; 4]> = py.allow_threads(|| {
        // Fondo y sigma por bloques (32x32) para robustez frente a gradientes/Vía Láctea
        let tile = 32usize;
        let ty_n = (h + tile - 1) / tile;
        let tx_n = (w + tile - 1) / tile;

        let tile_stats: Vec<(f32, f32)> = (0..(ty_n * tx_n))
            .into_par_iter()
            .map(|tidx| {
                let ty = tidx / tx_n;
                let tx = tidx % tx_n;
                let y0 = ty * tile;
                let y1 = (y0 + tile).min(h);
                let x0 = tx * tile;
                let x1 = (x0 + tile).min(w);

                let mut vals = Vec::with_capacity(((y1 - y0) * (x1 - x0)) / 4 + 1);
                for y in (y0..y1).step_by(2) {
                    for x in (x0..x1).step_by(2) {
                        if let Some(ref mv) = mask_view {
                            if mv[[y, x]] <= 0.5 {
                                continue;
                            }
                        }
                        let v = g[[y, x]];
                        if v.is_finite() {
                            vals.push(v);
                        }
                    }
                }
                if vals.len() < 8 {
                    return (0.0f32, 1e9f32);
                }
                let mid = vals.len() / 2;
                let (_, &mut med, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                let mut devs: Vec<f32> = vals.iter().map(|&v| (v - med).abs()).collect();
                let (_, &mut mad, _) = devs.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                let sigma = (1.4826 * mad).max(1e-4);
                (med, sigma)
            })
            .collect();

        // Buscar máximos locales 5x5 en paralelo por filas
        let margin = 5usize;
        let candidates: Vec<[f32; 4]> = (margin..(h - margin))
            .into_par_iter()
            .flat_map_iter(|y| {
                let mut row_cands = Vec::new();
                let ty = y / tile;
                for x in margin..(w - margin) {
                    if let Some(ref mv) = mask_view {
                        if mv[[y, x]] <= 0.5 {
                            continue;
                        }
                    }
                    let (bg, sig) = tile_stats[ty * tx_n + (x / tile)];
                    let peak = g[[y, x]];
                    let thresh = bg + sigma_thresh * sig;
                    if peak <= thresh {
                        continue;
                    }

                    // Máximo estricto en vecindario 5x5
                    let mut is_max = true;
                    'nbr: for dy in -2isize..=2 {
                        let yy = (y as isize + dy) as usize;
                        for dx in -2isize..=2 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let xx = (x as isize + dx) as usize;
                            let nv = g[[yy, xx]];
                            if nv > peak || (nv == peak && (dy < 0 || (dy == 0 && dx < 0))) {
                                is_max = false;
                                break 'nbr;
                            }
                        }
                    }
                    if !is_max {
                        continue;
                    }

                    // Rechazo de hot pixels de 1 píxel: al menos 2 vecinos-4 deben destacar sobre fondo
                    let n_thr = bg + 1.2 * sig;
                    let mut strong_nbrs = 0u8;
                    if g[[y - 1, x]] > n_thr { strong_nbrs += 1; }
                    if g[[y + 1, x]] > n_thr { strong_nbrs += 1; }
                    if g[[y, x - 1]] > n_thr { strong_nbrs += 1; }
                    if g[[y, x + 1]] > n_thr { strong_nbrs += 1; }
                    if strong_nbrs < 2 {
                        continue;
                    }

                    // Centroide subpíxel + flujo neto + excentricidad en ventana 7x7 (r=3)
                    let mut m00 = 0.0f32;
                    let mut m10 = 0.0f32;
                    let mut m01 = 0.0f32;
                    for dy in -3isize..=3 {
                        let yy = (y as isize + dy) as usize;
                        for dx in -3isize..=3 {
                            let xx = (x as isize + dx) as usize;
                            let val = (g[[yy, xx]] - bg).max(0.0);
                            m00 += val;
                            m10 += (dx as f32) * val;
                            m01 += (dy as f32) * val;
                        }
                    }
                    if m00 <= 1e-4 {
                        continue;
                    }
                    let dx_c = (m10 / m00).clamp(-1.5, 1.5);
                    let dy_c = (m01 / m00).clamp(-1.5, 1.5);

                    // Momentos de segundo orden para descartar trazos de satélites/aviones
                    let mut mu20 = 0.0f32;
                    let mut mu02 = 0.0f32;
                    let mut mu11 = 0.0f32;
                    for dy in -3isize..=3 {
                        let yy = (y as isize + dy) as usize;
                        let ry = dy as f32 - dy_c;
                        for dx in -3isize..=3 {
                            let xx = (x as isize + dx) as usize;
                            let rx = dx as f32 - dx_c;
                            let val = (g[[yy, xx]] - bg).max(0.0);
                            mu20 += rx * rx * val;
                            mu02 += ry * ry * val;
                            mu11 += rx * ry * val;
                        }
                    }
                    mu20 /= m00;
                    mu02 /= m00;
                    mu11 /= m00;
                    let delta = ((mu20 - mu02).powi(2) + 4.0 * mu11 * mu11).max(0.0).sqrt();
                    let l1 = ((mu20 + mu02 + delta) * 0.5).max(1e-4);
                    let l2 = ((mu20 + mu02 - delta) * 0.5).max(1e-4);
                    let ecc = (1.0 - (l2 / l1).clamp(0.0, 1.0)).sqrt();
                    if ecc > 0.88 {
                        continue;
                    }

                    row_cands.push([x as f32 + dx_c, y as f32 + dy_c, m00, peak]);
                }
                row_cands
            })
            .collect();

        // Ordenar por flujo neto descendente y aplicar separación mínima
        let mut sorted = candidates;
        sorted.sort_by(|a, b| b[2].total_cmp(&a[2]));

        let min_sep_sq = (min_separation as f32) * (min_separation as f32);
        let mut selected: Vec<[f32; 4]> = Vec::with_capacity(max_stars);
        for cand in sorted {
            let mut too_close = false;
            for s in &selected {
                let dx = cand[0] - s[0];
                let dy = cand[1] - s[1];
                if dx * dx + dy * dy < min_sep_sq {
                    too_close = true;
                    break;
                }
            }
            if !too_close {
                selected.push(cand);
                if selected.len() >= max_stars {
                    break;
                }
            }
        }
        selected
    });

    let n = stars_vec.len();
    let mut flat = Vec::with_capacity(n * 4);
    for s in &stars_vec {
        flat.extend_from_slice(s);
    }
    let arr = Array2::from_shape_vec((n, 4), flat).unwrap();
    Ok(arr.into_pyarray(py))
}

// ============================================================================
// INVARIANTE QUAD 4D + KABSCH SVD + REFINAMIENTO (R, f, k1)
// ============================================================================

struct ImageQuad {
    img_idx: [usize; 4],
    code: [f32; 4],
    chord_l: f32,
}

fn compute_quad_invariant_3d(
    vs: &[[f64; 3]; 4],
    ids: [usize; 4],
) -> Option<([usize; 4], [f32; 4], f32)> {
    let mut max_d2 = -1.0f64;
    let mut best_pair = (0usize, 1usize);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let dx = vs[i][0] - vs[j][0];
            let dy = vs[i][1] - vs[j][1];
            let dz = vs[i][2] - vs[j][2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > max_d2 {
                max_d2 = d2;
                best_pair = (i, j);
            }
        }
    }
    let l = max_d2.sqrt();
    if l < 1e-4 {
        return None;
    }

    let (mut ia, mut ib) = best_pair;
    let mut other = [0usize; 2];
    let mut oi = 0;
    for k in 0..4 {
        if k != ia && k != ib {
            other[oi] = k;
            oi += 1;
        }
    }
    let (mut ic, mut id) = (other[0], other[1]);

    let va = Vector3::new(vs[ia][0], vs[ia][1], vs[ia][2]);
    let vb = Vector3::new(vs[ib][0], vs[ib][1], vs[ib][2]);
    let vc = Vector3::new(vs[ic][0], vs[ic][1], vs[ic][2]);
    let vd = Vector3::new(vs[id][0], vs[id][1], vs[id][2]);

    let mid = va + vb;
    let mid_n = mid.norm();
    if mid_n < 1e-6 {
        return None;
    }
    let mid = mid / mid_n;

    let ex = (vb - va) / l;
    let ey = mid.cross(&ex);
    let ey_n = ey.norm();
    if ey_n < 1e-6 {
        return None;
    }
    let ey = ey / ey_n;

    let dc = vc - mid;
    let dd = vd - mid;
    let mut xc = (dc.dot(&ex) / l) as f32;
    let mut yc = (dc.dot(&ey) / l) as f32;
    let mut xd = (dd.dot(&ex) / l) as f32;
    let mut yd = (dd.dot(&ey) / l) as f32;

    let sx = xc + xd;
    let sy = yc + yd;
    if sx < -1e-5 || (sx.abs() <= 1e-5 && sy < 0.0) {
        std::mem::swap(&mut ia, &mut ib);
        xc = -xc;
        yc = -yc;
        xd = -xd;
        yd = -yd;
    }

    if xc > xd || ((xc - xd).abs() <= 1e-5 && yc > yd) {
        std::mem::swap(&mut ic, &mut id);
        std::mem::swap(&mut xc, &mut xd);
        std::mem::swap(&mut yc, &mut yd);
    }

    if xc < X_MIN || xc > X_MAX || xd < X_MIN || xd > X_MAX || yc < Y_MIN || yc > Y_MAX || yd < Y_MIN || yd > Y_MAX {
        return None;
    }

    Some(([ids[ia], ids[ib], ids[ic], ids[id]], [xc, yc, xd, yd], l as f32))
}

fn kabsch_so3(cam_vecs: &[[f64; 3]], cat_vecs: &[[f64; 3]]) -> Matrix3<f64> {
    // Resuelve R tal que cat_vec ≈ R * cam_vec (det(R) = +1)
    let mut h = Matrix3::<f64>::zeros();
    for (c, u) in cam_vecs.iter().zip(cat_vecs.iter()) {
        let vc = Vector3::new(c[0], c[1], c[2]);
        let vu = Vector3::new(u[0], u[1], u[2]);
        h += vu * vc.transpose();
    }
    let svd = h.svd(true, true);
    let u_mat = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let mut r = u_mat * v_t;
    if r.determinant() < 0.0 {
        let mut u_fix = u_mat;
        u_fix[(0, 2)] = -u_fix[(0, 2)];
        u_fix[(1, 2)] = -u_fix[(1, 2)];
        u_fix[(2, 2)] = -u_fix[(2, 2)];
        r = u_fix * v_t;
    }
    r
}

fn unproject_pixel(px: f64, py: f64, cx: f64, cy: f64, focal: f64, k1: f64) -> [f64; 3] {
    let dx = (px - cx) / focal;
    let dy = (py - cy) / focal;
    // Inversión de 2 iteraciones de Newton para r_d = r_u * (1 + k1 * r_u^2)
    let (tx, ty) = if k1.abs() > 1e-6 {
        let rd = dx.hypot(dy);
        if rd > 1e-9 {
            let mut ru = rd;
            for _ in 0..3 {
                let f_val = ru * (1.0 + k1 * ru * ru) - rd;
                let df = 1.0 + 3.0 * k1 * ru * ru;
                if df.abs() > 1e-6 {
                    ru -= f_val / df;
                }
            }
            let scale = (ru / rd).clamp(0.5, 2.0);
            (dx * scale, dy * scale)
        } else {
            (dx, dy)
        }
    } else {
        (dx, dy)
    };
    let inv = 1.0 / (tx * tx + ty * ty + 1.0).sqrt();
    [tx * inv, ty * inv, inv]
}

fn project_icrf_to_pixel(
    u: &[f32; 3],
    r_cam_to_icrf: &Matrix3<f64>,
    cx: f64,
    cy: f64,
    focal: f64,
    k1: f64,
) -> Option<(f64, f64)> {
    // d_cam = R^T * u
    let ux = u[0] as f64;
    let uy = u[1] as f64;
    let uz = u[2] as f64;
    let dz = r_cam_to_icrf[(0, 2)] * ux + r_cam_to_icrf[(1, 2)] * uy + r_cam_to_icrf[(2, 2)] * uz;
    if dz <= 0.12 {
        return None;
    }
    let dx = r_cam_to_icrf[(0, 0)] * ux + r_cam_to_icrf[(1, 0)] * uy + r_cam_to_icrf[(2, 0)] * uz;
    let dy = r_cam_to_icrf[(0, 1)] * ux + r_cam_to_icrf[(1, 1)] * uy + r_cam_to_icrf[(2, 1)] * uz;
    let tx = dx / dz;
    let ty = dy / dz;
    let r2 = tx * tx + ty * ty;
    let dist = 1.0 + k1 * r2;
    Some((cx + focal * tx * dist, cy + focal * ty * dist))
}

struct SolveSolution {
    r_cam_to_icrf: Matrix3<f64>,
    focal_px: f64,
    k1: f64,
    rms_px: f64,
    expected_random: f64,
    reliable: bool,
    matches: Vec<(usize, usize, f32)>, // (img_idx, cat_star_idx, residual_px)
}

fn match_and_refine(
    cat: &PlateCatalog,
    stars_xy: &[[f32; 4]],
    w: f64,
    h: f64,
    mut r: Matrix3<f64>,
    mut focal: f64,
    init_thresh_px: f64,
) -> Option<SolveSolution> {
    let cx = w * 0.5;
    let cy = h * 0.5;
    let half_diag_rad = (0.5 * w.hypot(h) / focal).atan() * 1.18;
    let mut k1 = 0.0f64;

    let center = [r[(0, 2)], r[(1, 2)], r[(2, 2)]];
    let max_mag_milli: i16 = match cat.profile_id {
        1 => 8500,
        2 => 11500,
        _ => 13500,
    };
    let mut fov_stars = cat.stars_in_cone(&center, half_diag_rad, max_mag_milli);
    if fov_stars.len() < 8 {
        return None;
    }
    if fov_stars.len() > 600 {
        fov_stars.select_nth_unstable_by_key(600, |&idx| cat.stars[idx].mag_milli);
        fov_stars.truncate(600);
        fov_stars.sort_unstable_by_key(|&idx| cat.stars[idx].mag_milli);
    }

    let mut last_matches: Vec<(usize, usize, f32)> = Vec::new();
    let thresholds = [init_thresh_px, init_thresh_px * 0.55, 5.5, 4.0];

    for (iter, &thr_px) in thresholds.iter().enumerate() {
        let thr_sq = thr_px * thr_px;
        let mut used_img = vec![false; stars_xy.len()];
        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();

        // Estrellas del catálogo ya vienen ordenadas por brillo dentro de cada celda;
        // emparejamos por cercanía en píxeles
        for &cat_idx in &fov_stars {
            let st = &cat.stars[cat_idx];
            let Some((px, py)) = project_icrf_to_pixel(&st.u, &r, cx, cy, focal, k1) else {
                continue;
            };
            if px < -10.0 || px > w + 10.0 || py < -10.0 || py > h + 10.0 {
                continue;
            }
            let mut best_i = usize::MAX;
            let mut best_d2 = thr_sq;
            for (i, s) in stars_xy.iter().enumerate() {
                if used_img[i] {
                    continue;
                }
                let ddx = s[0] as f64 - px;
                let ddy = s[1] as f64 - py;
                let d2 = ddx * ddx + ddy * ddy;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_i = i;
                }
            }
            if best_i != usize::MAX {
                used_img[best_i] = true;
                pairs.push((best_i, cat_idx, best_d2.sqrt() as f32));
            }
        }

        if pairs.len() < 8 {
            return None;
        }

        // Refinar R vía Kabsch SVD
        let cam_v: Vec<[f64; 3]> = pairs
            .iter()
            .map(|&(i, _, _)| unproject_pixel(stars_xy[i][0] as f64, stars_xy[i][1] as f64, cx, cy, focal, k1))
            .collect();
        let cat_v: Vec<[f64; 3]> = pairs
            .iter()
            .map(|&(_, c, _)| {
                let u = cat.stars[c].u;
                [u[0] as f64, u[1] as f64, u[2] as f64]
            })
            .collect();
        r = kabsch_so3(&cam_v, &cat_v);

        // Refinar (focal, focal * k1) por mínimos cuadrados lineales 2x2: r_px = a * r_t + b * r_t^3
        if pairs.len() >= 10 && iter >= 1 {
            let mut s11 = 0.0f64;
            let mut s12 = 0.0f64;
            let mut s22 = 0.0f64;
            let mut t1 = 0.0f64;
            let mut t2 = 0.0f64;

            for &(i, c, _) in &pairs {
                let u = cat.stars[c].u;
                let ux = u[0] as f64;
                let uy = u[1] as f64;
                let uz = u[2] as f64;
                let dz = r[(0, 2)] * ux + r[(1, 2)] * uy + r[(2, 2)] * uz;
                if dz <= 0.15 {
                    continue;
                }
                let dx = r[(0, 0)] * ux + r[(1, 0)] * uy + r[(2, 0)] * uz;
                let dy = r[(0, 1)] * ux + r[(1, 1)] * uy + r[(2, 1)] * uz;
                let rt = (dx / dz).hypot(dy / dz);
                if rt < 0.03 {
                    continue;
                }
                let rt3 = rt * rt * rt;
                let rpx = (stars_xy[i][0] as f64 - cx).hypot(stars_xy[i][1] as f64 - cy);

                s11 += rt * rt;
                s12 += rt * rt3;
                s22 += rt3 * rt3;
                t1 += rt * rpx;
                t2 += rt3 * rpx;
            }
            let det = s11 * s22 - s12 * s12;
            if det.abs() > 1e-9 {
                let new_f = (s22 * t1 - s12 * t2) / det;
                let new_fk1 = (-s12 * t1 + s11 * t2) / det;
                if new_f > focal * 0.6 && new_f < focal * 1.6 && new_f.is_finite() {
                    let new_k1 = (new_fk1 / new_f).clamp(-0.18, 0.12);
                    focal = new_f;
                    k1 = new_k1;
                }
            }
        }

        last_matches = pairs;
    }

    // Recalcular residuos finales exactos y estimar coincidencias esperadas por azar
    let mut sum_sq = 0.0f64;
    let mut final_matches = Vec::with_capacity(last_matches.len());
    for &(i, c, _) in &last_matches {
        if let Some((px, py)) = project_icrf_to_pixel(&cat.stars[c].u, &r, cx, cy, focal, k1) {
            let res = (stars_xy[i][0] as f64 - px).hypot(stars_xy[i][1] as f64 - py);
            if res <= 4.5 {
                sum_sq += res * res;
                final_matches.push((i, c, res as f32));
            }
        }
    }
    let in_frame_cat = fov_stars
        .iter()
        .filter(|&&c| {
            project_icrf_to_pixel(&cat.stars[c].u, &r, cx, cy, focal, k1)
                .map_or(false, |(px, py)| px >= 0.0 && px < w && py >= 0.0 && py < h)
        })
        .count();
    let area_frac = ((stars_xy.len() as f64 * std::f64::consts::PI * 4.5 * 4.5) / (w * h).max(1.0)).min(0.5);
    let expected_random = (in_frame_cat as f64) * area_frac;
    let min_required = (expected_random * 3.2 + 8.0).ceil().max(16.0) as usize;
    if final_matches.len() < min_required {
        return None;
    }
    let rms_px = (sum_sq / final_matches.len() as f64).sqrt();
    if rms_px > 3.2 {
        return None;
    }
    let reliable = final_matches.len() >= 24
        && rms_px < 2.6
        && (final_matches.len() as f64) >= (expected_random * 4.5 + 10.0);
    Some(SolveSolution {
        r_cam_to_icrf: r,
        focal_px: focal,
        k1,
        rms_px,
        expected_random,
        reliable,
        matches: final_matches,
    })
}

fn solve_with_focal_guess(
    cat: &PlateCatalog,
    stars_xy: &[[f32; 4]],
    w: f64,
    h: f64,
    f_guess: f64,
    scale_tol: f32,
    quad_tol: f32,
) -> Option<SolveSolution> {
    let n_pat = stars_xy.len().min(55);
    if n_pat < 6 {
        return None;
    }
    let cx = w * 0.5;
    let cy = h * 0.5;

    let unit_rays: Vec<[f64; 3]> = (0..n_pat)
        .map(|i| unproject_pixel(stars_xy[i][0] as f64, stars_xy[i][1] as f64, cx, cy, f_guess, 0.0))
        .collect();

    // Construir quads locales sobre los rayos unitarios (k=6 vecinos más cercanos)
    let k_nbrs = 6usize.min(n_pat - 1);
    let mut seen = HashSet::<[usize; 4]>::new();
    let mut img_quads: Vec<ImageQuad> = Vec::with_capacity(n_pat * 10);
    let mut dists = Vec::with_capacity(n_pat);

    for i in 0..n_pat {
        dists.clear();
        let vi = unit_rays[i];
        for j in 0..n_pat {
            if i == j {
                continue;
            }
            let vj = unit_rays[j];
            let d2 = (vi[0] - vj[0]).powi(2) + (vi[1] - vj[1]).powi(2) + (vi[2] - vj[2]).powi(2);
            dists.push((d2, j));
        }
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        let nbrs: Vec<usize> = dists[..k_nbrs].iter().map(|&(_, j)| j).collect();

        for a in 0..k_nbrs {
            for b in (a + 1)..k_nbrs {
                for c in (b + 1)..k_nbrs {
                    let mut key = [i, nbrs[a], nbrs[b], nbrs[c]];
                    key.sort_unstable();
                    if !seen.insert(key) {
                        continue;
                    }
                    let vs = [unit_rays[key[0]], unit_rays[key[1]], unit_rays[key[2]], unit_rays[key[3]]];
                    if let Some((canon_idx, code, chord_l)) = compute_quad_invariant_3d(&vs, key) {
                        if chord_l >= 0.025 && chord_l <= 0.42 {
                            img_quads.push(ImageQuad {
                                img_idx: canon_idx,
                                code,
                                chord_l,
                            });
                        }
                    }
                }
            }
        }
    }

    if img_quads.is_empty() {
        return None;
    }

    let b_axis = cat.hash_bins;
    let tol_sq = quad_tol * quad_tol;
    let init_thresh_px = (w.max(h) * 0.014).clamp(10.0, 28.0);

    // Evaluar quads en paralelo con Rayon, devolviendo la mejor solución
    let bin_range = |val: f32, vmin: f32, vmax: f32| -> (usize, usize) {
        let lo = (((val - quad_tol - vmin) / (vmax - vmin)) * b_axis as f32).floor() as isize;
        let hi = (((val + quad_tol - vmin) / (vmax - vmin)) * b_axis as f32).floor() as isize;
        (
            lo.clamp(0, (b_axis - 1) as isize) as usize,
            hi.clamp(0, (b_axis - 1) as isize) as usize,
        )
    };

    // Limitar a los primeros 400 quads (estrellas más brillantes primero) para máxima velocidad
    let query_slice = &img_quads[..img_quads.len().min(400)];

    let best: Option<SolveSolution> = query_slice
        .par_iter()
        .filter_map(|iq| {
            let (ix0, ix1) = bin_range(iq.code[0], X_MIN, X_MAX);
            let (iy0, iy1) = bin_range(iq.code[1], Y_MIN, Y_MAX);
            let (jx0, jx1) = bin_range(iq.code[2], X_MIN, X_MAX);
            let (jy0, jy1) = bin_range(iq.code[3], Y_MIN, Y_MAX);

            let cam4 = [
                unit_rays[iq.img_idx[0]],
                unit_rays[iq.img_idx[1]],
                unit_rays[iq.img_idx[2]],
                unit_rays[iq.img_idx[3]],
            ];

            let mut local_best: Option<SolveSolution> = None;

            for ix in ix0..=ix1 {
                for iy in iy0..=iy1 {
                    for jx in jx0..=jx1 {
                        for jy in jy0..=jy1 {
                            let bucket = ((ix * b_axis + iy) * b_axis + jx) * b_axis + jy;
                            let q0 = cat.hash_offsets[bucket] as usize;
                            let q1 = cat.hash_offsets[bucket + 1] as usize;
                            for qi in q0..q1 {
                                let cq = &cat.quads[qi];
                                let l_cat = (cq.scale_mrad as f32) * 1e-4;
                                if (iq.chord_l - l_cat).abs() > l_cat * scale_tol {
                                    continue;
                                }
                                let d0 = iq.code[0] - (cq.code[0] as f32 / 30000.0);
                                let d1 = iq.code[1] - (cq.code[1] as f32 / 30000.0);
                                let d2 = iq.code[2] - (cq.code[2] as f32 / 30000.0);
                                let d3 = iq.code[3] - (cq.code[3] as f32 / 30000.0);
                                if d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 > tol_sq {
                                    continue;
                                }

                                let cat4 = [
                                    {
                                        let u = cat.stars[cq.star_idx[0] as usize].u;
                                        [u[0] as f64, u[1] as f64, u[2] as f64]
                                    },
                                    {
                                        let u = cat.stars[cq.star_idx[1] as usize].u;
                                        [u[0] as f64, u[1] as f64, u[2] as f64]
                                    },
                                    {
                                        let u = cat.stars[cq.star_idx[2] as usize].u;
                                        [u[0] as f64, u[1] as f64, u[2] as f64]
                                    },
                                    {
                                        let u = cat.stars[cq.star_idx[3] as usize].u;
                                        [u[0] as f64, u[1] as f64, u[2] as f64]
                                    },
                                ];

                                // Estimar escala refinada desde el ratio de cuerdas
                                let f_hyp = f_guess * (iq.chord_l as f64 / l_cat as f64);
                                let r_hyp = kabsch_so3(&cam4, &cat4);

                                // Comprobación rápida de los 4 vértices (< 1.5° de error)
                                let mut quad_ok = true;
                                for k in 0..4 {
                                    let vc = Vector3::new(cam4[k][0], cam4[k][1], cam4[k][2]);
                                    let vu = Vector3::new(cat4[k][0], cat4[k][1], cat4[k][2]);
                                    let dot = (r_hyp * vc).dot(&vu);
                                    if dot < 0.9994 {
                                        quad_ok = false;
                                        break;
                                    }
                                }
                                if !quad_ok {
                                    continue;
                                }

                                if let Some(sol) = match_and_refine(
                                    cat,
                                    stars_xy,
                                    w,
                                    h,
                                    r_hyp,
                                    f_hyp,
                                    init_thresh_px,
                                ) {
                                    let is_better = match &local_best {
                                        None => true,
                                        Some(prev) => {
                                            sol.matches.len() > prev.matches.len()
                                                || (sol.matches.len() == prev.matches.len()
                                                    && sol.rms_px < prev.rms_px)
                                        }
                                    };
                                    if is_better {
                                        local_best = Some(sol);
                                        if let Some(ref lb) = local_best {
                                            if lb.matches.len() >= 40 && lb.rms_px < 2.0 {
                                                return local_best;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            local_best
        })
        .reduce_with(|a, b| {
            if a.matches.len() > b.matches.len()
                || (a.matches.len() == b.matches.len() && a.rms_px < b.rms_px)
            {
                a
            } else {
                b
            }
        });

    best
}

#[pyfunction]
#[pyo3(signature = (stars_xy, w, h, catalog_path, focal_px_hint=None, scale_tol=0.40, quad_tol=0.025))]
pub fn plate_solve_stars_rs<'py>(
    py: Python<'py>,
    stars_xy: PyReadonlyArray2<'py, f32>,
    w: usize,
    h: usize,
    catalog_path: &str,
    focal_px_hint: Option<f64>,
    scale_tol: f32,
    quad_tol: f32,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let view = stars_xy.as_array();
    let n = view.nrows();
    if n < 6 || view.ncols() < 2 {
        return Ok(None);
    }

    let cat = get_or_load_catalog(catalog_path).map_err(PyValueError::new_err)?;
    let stars_vec: Vec<[f32; 4]> = (0..n)
        .map(|i| {
            [
                view[[i, 0]],
                view[[i, 1]],
                if view.ncols() > 2 { view[[i, 2]] } else { 1.0 },
                if view.ncols() > 3 { view[[i, 3]] } else { 1.0 },
            ]
        })
        .collect();

    let wf = w as f64;
    let hf = h as f64;
    let max_dim = wf.max(hf);

    let guesses: Vec<f64> = match focal_px_hint {
        Some(f) if f > 50.0 => vec![f, f * 1.48, f * 0.68],
        _ => vec![
            max_dim * 0.42, // ~15mm FF
            max_dim * 0.62, // ~22mm FF
            max_dim * 0.95, // ~34mm FF
            max_dim * 1.45, // ~52mm FF
        ],
    };

    let sol_opt = py.allow_threads(|| {
        let mut best: Option<SolveSolution> = None;
        for f_guess in guesses {
            if let Some(sol) = solve_with_focal_guess(&cat, &stars_vec, wf, hf, f_guess, scale_tol, quad_tol) {
                let good_enough = sol.reliable && sol.matches.len() >= 28 && sol.rms_px < 2.4;
                let better = match &best {
                    None => true,
                    Some(prev) => sol.matches.len() > prev.matches.len(),
                };
                if better {
                    best = Some(sol);
                }
                if good_enough {
                    break;
                }
            }
        }
        best
    });

    let Some(sol) = sol_opt else {
        return Ok(None);
    };

    let dict = PyDict::new(py);
    let r = sol.r_cam_to_icrf;
    let center_u = [r[(0, 2)], r[(1, 2)], r[(2, 2)]];
    let mut ra_deg = center_u[1].atan2(center_u[0]).to_degrees();
    if ra_deg < 0.0 {
        ra_deg += 360.0;
    }
    let dec_deg = center_u[2].clamp(-1.0, 1.0).asin().to_degrees();
    let scale_arcsec_px = (1.0 / sol.focal_px).to_degrees() * 3600.0;

    let mut r_arr = Array2::<f64>::zeros((3, 3));
    for rr in 0..3 {
        for cc in 0..3 {
            r_arr[[rr, cc]] = r[(rr, cc)];
        }
    }

    dict.set_item("solved", true)?;
    dict.set_item("profile_id", cat.profile_id)?;
    dict.set_item("ra_deg", ra_deg)?;
    dict.set_item("dec_deg", dec_deg)?;
    dict.set_item("focal_px", sol.focal_px)?;
    dict.set_item("scale_arcsec_px", scale_arcsec_px)?;
    dict.set_item("k1", sol.k1)?;
    dict.set_item("rms_px", sol.rms_px)?;
    dict.set_item("num_inliers", sol.matches.len())?;
    dict.set_item("expected_random", sol.expected_random)?;
    dict.set_item("reliable", sol.reliable)?;
    dict.set_item("rotation_icrf", r_arr.into_pyarray(py))?;

    let matched_list = PyList::empty(py);
    let max_r = 0.5 * wf.hypot(hf);
    for (img_i, cat_i, res_px) in sol.matches {
        let st = &cat.stars[cat_i];
        let x = stars_vec[img_i][0];
        let y = stars_vec[img_i][1];
        let peak = stars_vec[img_i][3];

        // Peso de confianza [0.0, 1.0]: penaliza residuo alto, bordes y saturación
        let edge_frac = ((x as f64 - wf * 0.5).hypot(y as f64 - hf * 0.5) / max_r).clamp(0.0, 1.0);
        let w_res = (1.0 - (res_px as f64 / 5.0)).clamp(0.1, 1.0);
        let w_edge = (1.0 - 0.35 * edge_frac * edge_frac).clamp(0.5, 1.0);
        let w_sat = if peak >= 0.96 { 0.65 } else { 1.0 };
        let weight = (w_res * w_edge * w_sat).clamp(0.05, 1.0);

        let entry = PyDict::new(py);
        entry.set_item("star_id", st.id)?;
        entry.set_item("x", x)?;
        entry.set_item("y", y)?;
        entry.set_item("mag", st.mag_milli as f32 / 1000.0)?;
        if st.bp_rp_milli != i16::MIN {
            entry.set_item("bp_rp", st.bp_rp_milli as f32 / 1000.0)?;
        } else {
            entry.set_item("bp_rp", py.None())?;
        }
        entry.set_item("residual_px", res_px)?;
        entry.set_item("weight", weight as f32)?;
        matched_list.append(entry)?;
    }
    dict.set_item("stars", matched_list)?;

    Ok(Some(dict))
}
