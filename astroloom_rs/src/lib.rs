pub mod ai_preprocessing;
pub mod astrometry;
pub mod background;
pub mod deconv;
pub mod developer;
pub mod eclipse;
pub mod fits_io;
pub mod meteors;
pub mod panorama;
pub mod photometry;
pub mod project_codec;
pub mod psd_writer;
pub mod renderer;
pub mod spatial_filters;
pub mod stacking;
pub mod streaming_io;
pub mod system_info;

use pyo3::prelude::*;

#[pymodule]
fn astroloom_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 1. Kernels de Apilado Estadístico y TPS (core/stacking.py)
    m.add_function(wrap_pyfunction!(stacking::kappa_sigma_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(stacking::percentile_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(stacking::median_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(stacking::compute_tps_grid_rs, m)?)?;

    // 2. Motor RBF y Neutralización de Fondo (core/background_extraction.py)
    m.add_function(wrap_pyfunction!(background::fit_and_eval_rbf_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(background::apply_bg_model_fused, m)?)?;

    // 3. Motor del Revelador y Capas de Ajuste - Prioridad 1 (core/mask_layers.py y core/stretch.py)
    m.add_function(wrap_pyfunction!(developer::apply_layer_adjustments_fused, m)?)?;
    m.add_function(wrap_pyfunction!(developer::screen_blend, m)?)?;
    m.add_function(wrap_pyfunction!(developer::auto_stretch_adaptive_starless_rs, m)?)?;

    // 4. Estirado Hiperbólico Generalizado y Efectos Ópticos - Prioridad 2 (core/ghs.py, core/orton.py, core/star_diffusion.py)
    m.add_function(wrap_pyfunction!(developer::compute_ghs_lut_rs, m)?)?;
    m.add_function(wrap_pyfunction!(developer::apply_ghs_stretch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(developer::compute_orton_glow_source_rs, m)?)?;
    m.add_function(wrap_pyfunction!(developer::blend_orton_and_diffusion_rs, m)?)?;

    // 5. Deconvolución Óptica Regularizada y Modelado PSF - Prioridad 3 (core/deconvolution.py)
    m.add_function(wrap_pyfunction!(deconv::generate_moffat_psf_rs, m)?)?;
    m.add_function(wrap_pyfunction!(deconv::generate_gaussian_psf_rs, m)?)?;
    m.add_function(wrap_pyfunction!(deconv::richardson_lucy_deconv_rs, m)?)?;
    m.add_function(wrap_pyfunction!(deconv::reconstruct_deconv_color_rs, m)?)?;

    // 6. Detección y Fusión de Lluvias de Meteoros - Prioridad 4 (core/meteors.py)
    m.add_function(wrap_pyfunction!(meteors::meteor_positive_diff_and_stats_rs, m)?)?;
    m.add_function(wrap_pyfunction!(meteors::compute_meteor_capsule_mask_rs, m)?)?;
    m.add_function(wrap_pyfunction!(meteors::estimate_channel_percentile_rs, m)?)?;
    m.add_function(wrap_pyfunction!(meteors::blend_meteor_capsule_lighten_rs, m)?)?;

    // 7. Filtros Coronales de Eclipses HDR - Prioridad 5 (core/eclipse.py)
    m.add_function(wrap_pyfunction!(eclipse::compute_nrgf_profiles_and_detail_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::apply_nrgf_continuous_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::create_lunar_mask_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::compute_mgn_diff_sq_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::combine_mgn_scales_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::finalize_mgn_lum_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::compute_mgn_scaled_and_chroma_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::compute_mgn_prominence_mask_rs, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse::blend_mgn_prominences_and_mask_rs, m)?)?;

    // 8. Alineación por Asterismos y Triángulos Estelares - Prioridad 6 (core/panorama.py)
    m.add_function(wrap_pyfunction!(panorama::build_star_asterisms_rs, m)?)?;
    m.add_function(wrap_pyfunction!(panorama::match_star_asterisms_rs, m)?)?;

    // 9. Fotometría Gaia (PCC), Paletas Dual-Band y Reparación Estelar - Prioridad 7 (core/dualband.py y core/pcc.py)
    m.add_function(wrap_pyfunction!(photometry::fit_theil_sen_pcc_rs, m)?)?;
    m.add_function(wrap_pyfunction!(photometry::apply_pcc_gains_rs, m)?)?;
    m.add_function(wrap_pyfunction!(photometry::apply_dualband_palette_rs, m)?)?;
    m.add_function(wrap_pyfunction!(photometry::unpurple_rs, m)?)?;
    m.add_function(wrap_pyfunction!(photometry::unclip_stars_rs, m)?)?;
    m.add_function(wrap_pyfunction!(photometry::apply_dualband_pipeline_fused, m)?)?;

    // 10. Supresión Banding CMOS, SDF Anti-Ghosting y Anti-Halo - Prioridad 8 (core/banding.py, core/masking.py y core/stacking.py)
    m.add_function(wrap_pyfunction!(spatial_filters::suppress_cmos_banding_rs, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_filters::compose_sky_ground_sdf_rs, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_filters::suppress_horizon_halos_rs, m)?)?;

    // 11. Astrometría por Triángulos, Cribado 3x3 y Mallas TPS - Prioridad 9 (core/panorama.py, core/frame_culling.py y core/stacking.py)
    m.add_function(wrap_pyfunction!(astrometry::find_max_inscribed_rect_rs, m)?)?;
    m.add_function(wrap_pyfunction!(astrometry::match_star_triangles_ransac_rs, m)?)?;
    m.add_function(wrap_pyfunction!(astrometry::compute_tps_remap_grid_rs, m)?)?;
    m.add_function(wrap_pyfunction!(astrometry::compute_3x3_optical_diagnosis_rs, m)?)?;
    m.add_function(wrap_pyfunction!(astrometry::analyze_frame_quality_3x3_rs, m)?)?;

    // 12. Calibración CFA uint16, Startrails e ICC Bradford - Prioridad 10 (core/stacking.py, core/startrails.py y core/icc_manager.py)
    m.add_function(wrap_pyfunction!(streaming_io::calibrate_raw_cfa_inplace_rs, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_io::accumulate_startrail_step_rs, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_io::recover_star_trail_colors_rs, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_io::convert_colorspace_bradford_rs, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_io::convert_colorspace_u16_bradford_rs, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_io::scan_raw_batch_metadata_rs, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_io::scan_raw_metadata_rs, m)?)?;

    // 13. Poda de Librerías: FITS I/O, Escritor PSD Multicapa y Monitor de RAM - Prioridad 11
    m.add_function(wrap_pyfunction!(system_info::get_system_memory, m)?)?;
    m.add_function(wrap_pyfunction!(fits_io::read_fits_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fits_io::get_fits_dimensions_rs, m)?)?;
    m.add_function(wrap_pyfunction!(psd_writer::write_psd_file, m)?)?;

    // 14. Puente de Renderizado Directo a Qt - Prioridad 12 (gui/canvas.py)
    m.add_function(wrap_pyfunction!(renderer::render_canvas_frame_rs, m)?)?;

    // 15. Tiling ONNX y Filtro Guiado - Prioridad 13 / Fase 16 (core/starnet_onnx.py, core/ai_horizon.py, core/ai_denoise.py)
    m.add_function(wrap_pyfunction!(ai_preprocessing::fast_guided_filter_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::compute_hann_2d_window_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::get_tile_coords_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::accumulate_tile_hann_inplace_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::reconstruct_hann_tiling_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::recompose_hann_tiles_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::place_tile_cropped_inplace_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::recompose_cropped_tiles_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::prepare_starnet_tiles_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ai_preprocessing::assemble_starnet_tiles_rs, m)?)?;

    // 16. Serialización Paralela de Proyectos .devloom - Prioridad 14 (core/project_io.py)
    m.add_function(wrap_pyfunction!(project_codec::write_devloom_project_rs, m)?)?;
    m.add_function(wrap_pyfunction!(project_codec::read_devloom_project_rs, m)?)?;

    Ok(())
}
