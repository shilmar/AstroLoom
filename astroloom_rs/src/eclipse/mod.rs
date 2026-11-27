mod detail;
mod hdr;
mod mgn;
mod nafe;
mod nrgf;
mod wow;

pub use detail::{
    accumulate_steerable_g2_rs, compute_larson_sekanina_rs, compute_red_excess_mask_rs,
    display_detail_rings_rs, polar_circular_median_filter_rs,
};
pub use hdr::{
    apply_asinh_stretch_rs, balance_coronal_white_balance_rs, fuse_hdr_bracketing_solar_rs,
    optimize_center_circular_flux_rs,
};
pub use mgn::{
    blend_mgn_prominences_and_mask_rs, combine_mgn_scales_rs, compute_mgn_diff_sq_rs,
    compute_mgn_prominence_mask_rs, compute_mgn_scaled_and_chroma_rs, create_lunar_mask_rs,
    finalize_mgn_lum_rs,
};
pub use nafe::apply_nafe_rs;
pub use nrgf::{apply_nrgf_continuous_rs, compute_nrgf_profiles_and_detail_rs};
pub use wow::{apply_wow_edge_aware_rs, b3_normalized_rs};

// ============================================================================
// FILTROS CORONALES DE ECLIPSES HDR (core/eclipse.py) — PRIORIDAD 5
// ============================================================================

#[inline(always)]
pub(super) fn median_f32(data: &mut [f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len();
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
pub(super) fn reflect101(mut idx: isize, len: isize) -> usize {
    if len <= 1 {
        return 0;
    }
    while idx < 0 || idx >= len {
        if idx < 0 {
            idx = -idx;
        } else if idx >= len {
            idx = 2 * len - 2 - idx;
        }
    }
    idx as usize
}

pub(super) fn gaussian_blur_1d(src: &[f32], dst: &mut [f32], ksize: usize, sigma: f32) {
    let n = src.len();
    if n == 0 {
        return;
    }
    if n == 1 || ksize <= 1 || sigma <= 0.0 {
        dst.copy_from_slice(src);
        return;
    }

    let r = (ksize / 2) as isize;
    let two_sig_sq = 2.0 * sigma * sigma;

    // Precomputar pesos gaussianos normalizados
    let mut kernel = vec![0.0f32; ksize];
    let mut k_sum = 0.0f32;
    for i in 0..ksize {
        let d = (i as isize - r) as f32;
        let w = (-d * d / two_sig_sq).exp();
        kernel[i] = w;
        k_sum += w;
    }
    let inv_sum = 1.0 / k_sum;
    for w in kernel.iter_mut() {
        *w *= inv_sum;
    }

    let len_isize = n as isize;
    for i in 0..n {
        let mut sum = 0.0f32;
        for (k_idx, &w) in kernel.iter().enumerate() {
            let offset = k_idx as isize - r;
            let pos = reflect101(i as isize + offset, len_isize);
            sum += src[pos] * w;
        }
        dst[i] = sum;
    }
}

