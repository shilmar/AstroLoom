use num_complex::Complex;
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use std::sync::Arc;

use super::hdr::percentile_exact_f32;
use super::median_f32;

#[inline(always)]
fn fast_ndtr(x: f32) -> f32 {
    let z = (x.abs() / std::f32::consts::SQRT_2) as f64;
    let p = 0.3275911f64;
    let a1 = 0.254829592f64;
    let a2 = -0.284496736f64;
    let a3 = 1.421413741f64;
    let a4 = -1.453152027f64;
    let a5 = 1.061405429f64;
    let t = 1.0f64 / (1.0f64 + p * z);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let erf_val = 1.0f64 - poly * (-z * z).exp();
    let phi = 0.5f64 * (1.0f64 + erf_val);
    let res = if x >= 0.0 { phi } else { 1.0f64 - phi };
    res as f32
}

fn nafe_neighbourhood_raw(n: usize) -> Vec<f32> {
    let half = (n / 2) as isize;
    let mut sig = [0.0f64; 12];
    for m in 1..=12 {
        sig[m - 1] = 2.0f64.powf((m as f64) / 2.0);
    }
    let norm_factor = 2.0f64 * std::f64::consts::PI
        / sig.iter().map(|&s| s.powf(-0.5)).sum::<f64>();

    let mut kernel = vec![0.0f32; n * n];
    for y in 0..n {
        let dy = (y as isize - half) as f64;
        let dy2 = dy * dy;
        for x in 0..n {
            let dx = (x as isize - half) as f64;
            let r2 = dx * dx + dy2;
            let mut g = 0.0f64;
            for &s in &sig {
                let term = (-0.5 * r2 / (s * s)).exp() / (2.0 * std::f64::consts::PI * s.sqrt());
                g += term;
            }
            kernel[y * n + x] = (g * norm_factor) as f32;
        }
    }
    kernel
}

fn next_fast_len(target: usize) -> usize {
    let mut n = target;
    loop {
        let mut m = n;
        while m % 2 == 0 {
            m /= 2;
        }
        while m % 3 == 0 {
            m /= 3;
        }
        while m % 5 == 0 {
            m /= 5;
        }
        if m == 1 {
            return n;
        }
        n += 1;
    }
}

fn transpose_complex(
    in_mat: &[Complex<f32>],
    out_t: &mut [Complex<f32>],
    rows: usize,
    cols: usize,
) {
    out_t
        .par_chunks_exact_mut(rows)
        .enumerate()
        .for_each(|(c, col_out)| {
            for r in 0..rows {
                col_out[r] = in_mat[r * cols + c];
            }
        });
}

pub struct RealFft2DHelper {
    pub ph: usize,
    pub pw: usize,
    pub spec_w: usize,
    pub r2c: Arc<dyn realfft::RealToComplex<f32>>,
    pub c2r: Arc<dyn realfft::ComplexToReal<f32>>,
    pub c2c_fwd: Arc<dyn rustfft::Fft<f32>>,
    pub c2c_inv: Arc<dyn rustfft::Fft<f32>>,
}

impl RealFft2DHelper {
    pub fn new(ph: usize, pw: usize) -> Self {
        let mut r_planner = RealFftPlanner::<f32>::new();
        let r2c = r_planner.plan_fft_forward(pw);
        let c2r = r_planner.plan_fft_inverse(pw);

        let mut c_planner = FftPlanner::<f32>::new();
        let c2c_fwd = c_planner.plan_fft_forward(ph);
        let c2c_inv = c_planner.plan_fft_inverse(ph);

        let spec_w = pw / 2 + 1;
        Self {
            ph,
            pw,
            spec_w,
            r2c,
            c2r,
            c2c_fwd,
            c2c_inv,
        }
    }

    pub fn forward_2d(&self, in_real: &[f32], out_spec: &mut [Complex<f32>]) {
        let ph = self.ph;
        let pw = self.pw;
        let spec_w = self.spec_w;

        let mut row_spec = vec![Complex::new(0.0f32, 0.0f32); ph * spec_w];
        row_spec
            .par_chunks_exact_mut(spec_w)
            .enumerate()
            .for_each(|(y, r_spec)| {
                let row_real = &in_real[y * pw..(y + 1) * pw];
                let mut r_in = row_real.to_vec();
                let mut scratch = vec![Complex::new(0.0f32, 0.0f32); self.r2c.get_scratch_len()];
                let _ = self.r2c.process_with_scratch(&mut r_in, r_spec, &mut scratch);
            });

        let mut spec_t = vec![Complex::new(0.0f32, 0.0f32); spec_w * ph];
        transpose_complex(&row_spec, &mut spec_t, ph, spec_w);

        spec_t
            .par_chunks_exact_mut(ph)
            .for_each(|col| {
                let mut scratch = vec![Complex::new(0.0f32, 0.0f32); self.c2c_fwd.get_inplace_scratch_len()];
                self.c2c_fwd.process_with_scratch(col, &mut scratch);
            });

        transpose_complex(&spec_t, out_spec, spec_w, ph);
    }

    pub fn inverse_2d(&self, in_spec: &[Complex<f32>], out_real: &mut [f32]) {
        let ph = self.ph;
        let pw = self.pw;
        let spec_w = self.spec_w;

        let mut spec_t = vec![Complex::new(0.0f32, 0.0f32); spec_w * ph];
        transpose_complex(in_spec, &mut spec_t, ph, spec_w);

        spec_t
            .par_chunks_exact_mut(ph)
            .for_each(|col| {
                let mut scratch = vec![Complex::new(0.0f32, 0.0f32); self.c2c_inv.get_inplace_scratch_len()];
                self.c2c_inv.process_with_scratch(col, &mut scratch);
            });

        let mut row_spec = vec![Complex::new(0.0f32, 0.0f32); ph * spec_w];
        transpose_complex(&spec_t, &mut row_spec, spec_w, ph);

        let inv_norm = 1.0f32 / ((ph * pw) as f32);
        out_real
            .par_chunks_exact_mut(pw)
            .enumerate()
            .for_each(|(y, r_real)| {
                let r_spec = &row_spec[y * spec_w..(y + 1) * spec_w];
                let mut s_in = r_spec.to_vec();
                let mut scratch = vec![Complex::new(0.0f32, 0.0f32); self.c2r.get_scratch_len()];
                let _ = self.c2r.process_with_scratch(&mut s_in, r_real, &mut scratch);
                for v in r_real.iter_mut() {
                    *v *= inv_norm;
                }
            });
    }

    pub fn convolve_with_kernel(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        n: usize,
        kf: &[Complex<f32>],
        out_cropped: &mut [f32],
    ) {
        let ph = self.ph;
        let pw = self.pw;
        let spec_w = self.spec_w;
        let half = n / 2;

        let mut padded_x = vec![0.0f32; ph * pw];
        for y in 0..h {
            padded_x[y * pw..y * pw + w].copy_from_slice(&x[y * w..(y + 1) * w]);
        }

        let mut spec_x = vec![Complex::new(0.0f32, 0.0f32); ph * spec_w];
        self.forward_2d(&padded_x, &mut spec_x);

        spec_x.par_iter_mut().zip(kf.par_iter()).for_each(|(sx, k)| {
            *sx = *sx * *k;
        });

        let mut out_padded = vec![0.0f32; ph * pw];
        self.inverse_2d(&spec_x, &mut out_padded);

        out_cropped
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(y, row_out)| {
                let src_y = half + y;
                let src_offset = src_y * pw + half;
                row_out.copy_from_slice(&out_padded[src_offset..src_offset + w]);
            });
    }
}

/// 18. apply_nafe_rs:
/// Ecualización difusa adaptativa al ruido (NAFE - Miloslav Druckmüller 2013) en Rust con Rayon.
/// Evalúa el vecindario difuso multiescala mediante FFT 2D en memoria realfft/rustfft,
/// paraleliza la convolución de niveles a lo largo de los núcleos de CPU y aplica modo Overlay.
#[pyfunction]
#[pyo3(signature = (
    image,
    center_x,
    center_y,
    radius_min = 10.0,
    sigma = None,
    n = 129,
    level_step = 0.01,
    blend_ratio = 0.79,
    use_overlay = true
))]
pub fn apply_nafe_rs<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, f32>,
    center_x: f32,
    center_y: f32,
    radius_min: f32,
    sigma: Option<f32>,
    n: usize,
    level_step: f32,
    blend_ratio: f32,
    use_overlay: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    if n % 2 != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n debe ser impar",
        ));
    }

    let shape = image.shape().to_vec();
    let is_3d = shape.len() == 3;
    let (h, w) = if is_3d {
        (shape[0], shape[1])
    } else if shape.len() == 2 {
        (shape[0], shape[1])
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "image must have 2 or 3 dimensions",
        ));
    };

    let img_slice = image.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be C-contiguous"))?;

    let out_vec = py.allow_threads(|| {
        let n_pixels = h * w;
        let mut lum = vec![0.0f32; n_pixels];
        if is_3d {
            lum.par_chunks_exact_mut(w)
                .enumerate()
                .for_each(|(y, lum_row)| {
                    let base_idx = y * w * 3;
                    for x in 0..w {
                        let idx3 = base_idx + x * 3;
                        let r = img_slice[idx3].clamp(0.0, 1.0);
                        let g = img_slice[idx3 + 1].clamp(0.0, 1.0);
                        let b = img_slice[idx3 + 2].clamp(0.0, 1.0);
                        lum_row[x] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    }
                });
        } else {
            for i in 0..n_pixels {
                lum[i] = img_slice[i].clamp(0.0, 1.0);
            }
        }

        let min_r = (radius_min - 2.5).max(0.0);
        let min_r_sq = min_r * min_r;
        let mut m = vec![false; n_pixels];
        let mut dist = vec![0.0f32; n_pixels];

        dist.par_chunks_exact_mut(w)
            .zip(m.par_chunks_exact_mut(w))
            .enumerate()
            .for_each(|(y, (d_row, m_row))| {
                let dy = y as f32 - center_y;
                let dy2 = dy * dy;
                for x in 0..w {
                    let dx = x as f32 - center_x;
                    let d = (dx * dx + dy2).sqrt();
                    d_row[x] = d;
                    m_row[x] = (dx * dx + dy2) >= min_r_sq;
                }
            });

        let any_m = m.iter().any(|&v| v);
        if !any_m {
            return if is_3d {
                img_slice.to_vec()
            } else {
                lum
            };
        }

        let sigma_val = match sigma {
            Some(s) if s > 0.0 => s,
            _ => {
                let mut dist_samples: Vec<f32> = dist
                    .iter()
                    .zip(m.iter())
                    .filter_map(|(&d, &is_m)| if is_m { Some(d) } else { None })
                    .collect();
                if dist_samples.is_empty() {
                    0.035
                } else {
                    let far_r = percentile_exact_f32(&mut dist_samples, 85.0);
                    let mut sky_samples: Vec<f32> = lum
                        .iter()
                        .zip(dist.iter())
                        .zip(m.iter())
                        .filter_map(|((&l, &d), &is_m)| if is_m && d >= far_r { Some(l) } else { None })
                        .collect();
                    if sky_samples.len() > 50 {
                        let med_sky = median_f32(&mut sky_samples);
                        let mut diffs: Vec<f32> = sky_samples.iter().map(|&v| (v - med_sky).abs()).collect();
                        let mad_sky = 1.4826f32 * median_f32(&mut diffs);
                        (12.0f32 * mad_sky).max(0.02)
                    } else {
                        0.035
                    }
                }
            }
        };

        let mut a0 = vec![0.0f32; n_pixels];
        let mut mf = vec![0.0f32; n_pixels];
        for i in 0..n_pixels {
            if m[i] {
                a0[i] = lum[i];
                mf[i] = 1.0;
            }
        }

        let ph = next_fast_len(h + n - 1);
        let pw = next_fast_len(w + n - 1);
        let fft_helper = RealFft2DHelper::new(ph, pw);

        let kernel_spatial = nafe_neighbourhood_raw(n);
        let mut padded_k = vec![0.0f32; ph * pw];
        for y in 0..n {
            padded_k[y * pw..y * pw + n].copy_from_slice(&kernel_spatial[y * n..(y + 1) * n]);
        }
        let mut kf = vec![Complex::new(0.0f32, 0.0f32); ph * fft_helper.spec_w];
        fft_helper.forward_2d(&padded_k, &mut kf);

        let mut norm = vec![0.0f32; n_pixels];
        fft_helper.convolve_with_kernel(&mf, h, w, n, &kf, &mut norm);
        for v in norm.iter_mut() {
            *v = v.max(1e-12);
        }

        let mut v_vals = Vec::new();
        for i in 0..n_pixels {
            if m[i] {
                v_vals.push(a0[i] as f64);
            }
        }
        let lo = v_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut t = vec![lo - sigma_val as f64];
        while t.last().copied().unwrap() < hi || t.len() < 2 {
            let last = *t.last().unwrap();
            let step = (sigma_val as f64 / 2.0).max((level_step as f64) * (last - lo));
            t.push(last + step);
        }

        let n_intervals = t.len() - 1;
        let mut bucket_pixels: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_intervals];

        for i in 0..n_pixels {
            if m[i] {
                let val = a0[i] as f64;
                let k = t.partition_point(|&tv| tv <= val).saturating_sub(1).min(n_intervals - 1);
                bucket_pixels[k].push((i, val));
            }
        }

        let mut needed_levels = vec![false; t.len()];
        for (k, bucket) in bucket_pixels.iter().enumerate() {
            if !bucket.is_empty() {
                needed_levels[k] = true;
                needed_levels[k + 1] = true;
            }
        }

        let needed_indices: Vec<usize> = (0..t.len()).filter(|&j| needed_levels[j]).collect();

        let computed_levels: Vec<(usize, Vec<f32>)> = needed_indices
            .into_par_iter()
            .map(|j| {
                let tj = t[j] as f32;
                let mut tmp = vec![0.0f32; n_pixels];
                for i in 0..n_pixels {
                    if m[i] {
                        let z = (tj - a0[i]) / sigma_val;
                        tmp[i] = fast_ndtr(z);
                    }
                }
                let mut conv_out = vec![0.0f32; n_pixels];
                fft_helper.convolve_with_kernel(&tmp, h, w, n, &kf, &mut conv_out);
                for i in 0..n_pixels {
                    conv_out[i] /= norm[i];
                }
                (j, conv_out)
            })
            .collect();

        let mut level_lookup: Vec<Option<Vec<f32>>> = (0..t.len()).map(|_| None).collect();
        for (j, conv_res) in computed_levels {
            level_lookup[j] = Some(conv_res);
        }

        let mut out = vec![0.5f32; n_pixels];
        for k in 0..n_intervals {
            if !bucket_pixels[k].is_empty() {
                let prev = level_lookup[k].as_ref().unwrap();
                let cur = level_lookup[k + 1].as_ref().unwrap();
                let tj0 = t[k];
                let tj1 = t[k + 1];
                let inv_dt = 1.0 / (tj1 - tj0).max(1e-12);

                for &(p, vp) in &bucket_pixels[k] {
                    let f = ((vp - tj0) * inv_dt) as f32;
                    let val = prev[p] + f * (cur[p] - prev[p]);
                    out[p] = if val.is_finite() { val } else { 0.5 };
                }
            }
        }

        for v in out.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }

        let blend = blend_ratio.clamp(0.0, 1.0);
        let total_out_size = if is_3d { n_pixels * 3 } else { n_pixels };
        let mut final_out = vec![0.0f32; total_out_size];

        if use_overlay {
            if is_3d {
                final_out
                    .par_chunks_exact_mut(w * 3)
                    .enumerate()
                    .for_each(|(y, row_out)| {
                        let base_pix = y * w;
                        let base_src = y * w * 3;
                        for x in 0..w {
                            let pix_idx = base_pix + x;
                            let src_idx = base_src + x * 3;
                            let eq = out[pix_idx];
                            let is_valid = m[pix_idx];

                            for c in 0..3 {
                                if !is_valid {
                                    row_out[x * 3 + c] = 0.0;
                                } else {
                                    let orig = img_slice[src_idx + c].clamp(0.0, 1.0);
                                    let ov = if orig < 0.5 {
                                        2.0 * orig * eq
                                    } else {
                                        1.0 - 2.0 * (1.0 - orig) * (1.0 - eq)
                                    };
                                    let fin = (1.0 - blend) * orig + blend * ov;
                                    row_out[x * 3 + c] = fin.clamp(0.0, 1.0);
                                }
                            }
                        }
                    });
            } else {
                final_out
                    .par_chunks_exact_mut(w)
                    .enumerate()
                    .for_each(|(y, row_out)| {
                        let base_pix = y * w;
                        for x in 0..w {
                            let pix_idx = base_pix + x;
                            let eq = out[pix_idx];
                            let is_valid = m[pix_idx];
                            if !is_valid {
                                row_out[x] = 0.0;
                            } else {
                                let orig = img_slice[pix_idx].clamp(0.0, 1.0);
                                let ov = if orig < 0.5 {
                                    2.0 * orig * eq
                                } else {
                                    1.0 - 2.0 * (1.0 - orig) * (1.0 - eq)
                                };
                                let fin = (1.0 - blend) * orig + blend * ov;
                                row_out[x] = fin.clamp(0.0, 1.0);
                            }
                        }
                    });
            }
        } else {
            if is_3d {
                final_out
                    .par_chunks_exact_mut(w * 3)
                    .enumerate()
                    .for_each(|(y, row_out)| {
                        let base_pix = y * w;
                        let base_src = y * w * 3;
                        for x in 0..w {
                            let pix_idx = base_pix + x;
                            let src_idx = base_src + x * 3;
                            let eq = out[pix_idx];
                            let is_valid = m[pix_idx];

                            for c in 0..3 {
                                if !is_valid {
                                    row_out[x * 3 + c] = 0.0;
                                } else {
                                    let orig = img_slice[src_idx + c].clamp(0.0, 1.0);
                                    let fin = (1.0 - blend) * orig + blend * eq;
                                    row_out[x * 3 + c] = fin.clamp(0.0, 1.0);
                                }
                            }
                        }
                    });
            } else {
                final_out
                    .par_chunks_exact_mut(w)
                    .enumerate()
                    .for_each(|(y, row_out)| {
                        let base_pix = y * w;
                        for x in 0..w {
                            let pix_idx = base_pix + x;
                            let eq = out[pix_idx];
                            let is_valid = m[pix_idx];
                            if !is_valid {
                                row_out[x] = 0.0;
                            } else {
                                let orig = img_slice[pix_idx].clamp(0.0, 1.0);
                                let fin = (1.0 - blend) * orig + blend * eq;
                                row_out[x] = fin.clamp(0.0, 1.0);
                            }
                        }
                    });
            }
        }

        final_out
    });

    let arr = ArrayD::from_shape_vec(IxDyn(&shape), out_vec).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create ArrayD: {}", e))
    })?;
    Ok(arr.into_pyarray(py))
}
