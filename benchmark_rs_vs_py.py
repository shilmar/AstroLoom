import os
import sys
import glob
import time
import argparse
import tempfile
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import astroloom_rs
import core.stacking as st
import core.background_extraction as be
import core.mask_layers as ml
import core.stretch as stretch
import core.ghs as ghs
import core.orton as orton
import core.star_diffusion as sd
import core.deconvolution as deconv
import core.meteors as meteors
import core.eclipse as eclipse
import core.panorama as pano
import core.dualband as db
import core.pcc as pcc
import core.banding as banding
import core.masking as masking
import core.icc_manager as icc
import core.frame_culling as fc
import core.project_io as pio
import core.starnet_onnx as s_onnx

# Global registry of benchmark results
BENCH_RESULTS = []

def record_result(category, name, t_py, t_rs, diff=None, extra=""):
    speedup = t_py / max(t_rs, 1e-6)
    diff_str = f"{diff:.2e}" if diff is not None else "N/A"
    BENCH_RESULTS.append({
        "category": category,
        "name": name,
        "t_py_ms": t_py,
        "t_rs_ms": t_rs,
        "speedup": speedup,
        "diff": diff_str,
        "extra": extra
    })
    print(f"  {name:38s}: Py {t_py:8.1f} ms | Rs {t_rs:8.1f} ms | {speedup:6.2f}x | Diff: {diff_str} {extra}")

def load_real_or_synthetic_24mp():
    """Loads real 24.5 MP astrophotography image if available, else generates reproducible synthetic data."""
    real_path = "imagenes/resultado_dual_32bit.tiff"
    if os.path.exists(real_path):
        import tifffile
        img = tifffile.imread(real_path)
        print(f"  [Dato Real] Imagen apilada 32-bit cargada: {real_path} ({img.shape[1]}x{img.shape[0]}x{img.shape[2]})")
        return img
    else:
        print("  [Dato Sintético] Generando imagen reproducible de 24.5 MP (4040x6064x3)...")
        np.random.seed(42)
        return np.random.uniform(0.05, 0.45, size=(4040, 6064, 3)).astype(np.float32)

def bench_stacking_kernels():
    print("\n=== 1. STACKING: KERNELS DE APILADO ESTADÍSTICO ===")
    n_frames = 35
    total_items = 200 * 6000 * 3  # 3.6 MP (franja representativa de 35 lights)
    np.random.seed(42)
    data = np.random.uniform(0.05, 0.25, size=(n_frames, total_items)).astype(np.float32)
    data[0, ::10] = 0.95
    data[-1, :500000] = -1.0
    weights = np.random.uniform(0.8, 1.2, size=n_frames).astype(np.float32)

    # Warmup Numba
    _ = st._kappa_sigma_kernel_numba(data[:, :1000], 2.2, weights)
    _ = st._median_kernel_numba(data[:, :1000])
    _ = st._percentile_kernel_numba(data[:, :1000], 15.0)

    # A. Kappa-Sigma Ponderado
    t0 = time.perf_counter()
    out_py_ks = st._kappa_sigma_kernel_numba(data, 2.2, weights)
    t_py_ks = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out_rs_ks = astroloom_rs.kappa_sigma_kernel(data, 2.2, weights)
    t_rs_ks = (time.perf_counter() - t0) * 1000
    diff_ks = float(np.max(np.abs(out_py_ks - out_rs_ks)))
    record_result("Stacking", "Kappa-Sigma Ponderado (35 frames)", t_py_ks, t_rs_ks, diff_ks)

    # B. Mediana Quickselect
    t0 = time.perf_counter()
    out_py_med = st._median_kernel_numba(data)
    t_py_med = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out_rs_med = astroloom_rs.median_kernel(data)
    t_rs_med = (time.perf_counter() - t0) * 1000
    diff_med = float(np.max(np.abs(out_py_med - out_rs_med)))
    record_result("Stacking", "Mediana Quickselect (35 frames)", t_py_med, t_rs_med, diff_med)

    # C. Percentil 15.0
    t0 = time.perf_counter()
    out_py_p = st._percentile_kernel_numba(data, 15.0)
    t_py_p = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out_rs_p = astroloom_rs.percentile_kernel(data, 15.0)
    t_rs_p = (time.perf_counter() - t0) * 1000
    diff_p = float(np.max(np.abs(out_py_p - out_rs_p)))
    record_result("Stacking", "Percentil 15.0 (35 frames)", t_py_p, t_rs_p, diff_p)

def bench_background_extraction(img_24mp):
    print("\n=== 2. FONDO: EXTRACCIÓN RBF Y NEUTRALIZACIÓN FUSIONADA (24.5 MP) ===")
    h, w = img_24mp.shape[:2]
    sky_mask = np.ones((h, w), dtype=np.float32)
    sky_mask[int(h * 0.75):, :] = 0.0
    pts = be.generate_sky_sample_points(h, w, sky_mask=sky_mask, img_rgb=img_24mp, grid_cols=6, grid_rows=5)

    # Python SciPy RBF + NumPy
    orig_has_rust = be.HAS_RUST
    try:
        be.HAS_RUST = False
        t0 = time.perf_counter()
        corr_py, bg_py = be.extract_background_rbf(img_24mp, pts, sky_mask=sky_mask, mode="subtraction")
        t_py = (time.perf_counter() - t0) * 1000

        be.HAS_RUST = True
        t0 = time.perf_counter()
        corr_rs, bg_rs = be.extract_background_rbf(img_24mp, pts, sky_mask=sky_mask, mode="subtraction")
        t_rs = (time.perf_counter() - t0) * 1000
    finally:
        be.HAS_RUST = orig_has_rust

    diff_corr = float(np.max(np.abs(corr_py - corr_rs)))
    record_result("Fondo RBF", "Extracción RBF + Fusión 24.5 MP", t_py, t_rs, diff_corr)

def bench_developer_and_layers(img_24mp):
    print("\n=== 3. REVELADOR: CAPAS DE AJUSTE, CURVAS Y SCREEN BLEND (24.5 MP) ===")
    h, w = img_24mp.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    layer = ml.MaskLayer(
        name="Bench Layer",
        opacity=0.85,
        exposure=0.4,
        contrast=0.15,
        shadows=-0.25,
        highlights=0.3,
        saturation=1.35,
        temp=0.12,
        tint=-0.08,
        curve_points=[(0.0, 0.0), (0.3, 0.25), (0.7, 0.8), (1.0, 1.0)]
    )

    orig_has_rust = ml.HAS_RUST
    try:
        # Numba / Python fallback
        ml.HAS_RUST = False
        t0 = time.perf_counter()
        res_py = ml.apply_local_adjustments_to_image(img_24mp, mask, layer)
        t_py = (time.perf_counter() - t0) * 1000

        # Rust Fused Rayon
        ml.HAS_RUST = True
        t0 = time.perf_counter()
        res_rs = ml.apply_local_adjustments_to_image(img_24mp, mask, layer)
        t_rs = (time.perf_counter() - t0) * 1000
    finally:
        ml.HAS_RUST = orig_has_rust

    diff = float(np.max(np.abs(res_rs - res_py)))
    record_result("Revelador", "Ajustes de Capa Fused 24.5 MP", t_py, t_rs, diff)

    # Screen Blend
    stars = img_24mp * 0.5
    t0 = time.perf_counter()
    s_py = 1.0 - (1.0 - img_24mp) * (1.0 - np.clip(stars * 0.75, 0.0, 1.0))
    t_py_sb = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    s_rs = astroloom_rs.screen_blend(img_24mp, stars, 0.75)
    t_rs_sb = (time.perf_counter() - t0) * 1000
    diff_sb = float(np.max(np.abs(s_py - s_rs)))
    record_result("Revelador", "Screen Blend 24.5 MP", t_py_sb, t_rs_sb, diff_sb)

def bench_stretch_and_ghs(img_24mp):
    print("\n=== 4. ESTIRADO: AUTO-STRETCH ADAPTATIVO Y GHS LUMA/RGB (24.5 MP) ===")
    h, w = img_24mp.shape[:2]
    sky_mask = np.ones((h, w), dtype=np.float32)
    sky_mask[int(h * 0.7):, :] = 0.0

    # A. Auto-stretch adaptativo
    orig_has_rust = stretch.HAS_RUST
    try:
        stretch.HAS_RUST = False
        t0 = time.perf_counter()
        s_py = stretch.auto_stretch_adaptive_starless(img_24mp, sky_mask=sky_mask, target_bg=0.15, saturation_boost=1.5, unlinked=True)
        t_py_as = (time.perf_counter() - t0) * 1000

        stretch.HAS_RUST = True
        t0 = time.perf_counter()
        s_rs = stretch.auto_stretch_adaptive_starless(img_24mp, sky_mask=sky_mask, target_bg=0.15, saturation_boost=1.5, unlinked=True)
        t_rs_as = (time.perf_counter() - t0) * 1000
    finally:
        stretch.HAS_RUST = orig_has_rust

    diff_as = float(np.max(np.abs(s_rs - s_py)))
    record_result("Estirado", "Auto-Stretch Starless 24.5 MP", t_py_as, t_rs_as, diff_as)

    # B. GHS Stretch (Luma)
    orig_has_rust = ghs.HAS_RUST
    try:
        ghs.HAS_RUST = False
        t0 = time.perf_counter()
        ghs_py = ghs.apply_ghs_stretch(img_24mp, sp=0.25, d=5.0, b=2.0, color_mode="luma")
        t_py_ghs = (time.perf_counter() - t0) * 1000

        ghs.HAS_RUST = True
        t0 = time.perf_counter()
        ghs_rs = ghs.apply_ghs_stretch(img_24mp, sp=0.25, d=5.0, b=2.0, color_mode="luma")
        t_rs_ghs = (time.perf_counter() - t0) * 1000
    finally:
        ghs.HAS_RUST = orig_has_rust

    diff_ghs = float(np.max(np.abs(ghs_rs - ghs_py)))
    record_result("Estirado", "GHS Stretch Luma LUT 24.5 MP", t_py_ghs, t_rs_ghs, diff_ghs)

def bench_optical_effects(img_24mp):
    print("\n=== 5. EFECTOS ÓPTICOS: ORTON ASTRONÓMICO Y DIFUSIÓN ESTELAR ===")
    # Usar recorte de 6 MP (2000x3000) representativo para efectos
    sub = img_24mp[:2000, :3000].copy()

    # A. Resplandor Orton
    orig_has_rust = orton.HAS_RUST
    try:
        orton.HAS_RUST = False
        t0 = time.perf_counter()
        o_py = orton.apply_astronomical_orton(sub, strength=0.6, radius=15.0, threshold=0.1, tint_type="warm_gold")
        t_py_ort = (time.perf_counter() - t0) * 1000

        orton.HAS_RUST = True
        t0 = time.perf_counter()
        o_rs = orton.apply_astronomical_orton(sub, strength=0.6, radius=15.0, threshold=0.1, tint_type="warm_gold")
        t_rs_ort = (time.perf_counter() - t0) * 1000
    finally:
        orton.HAS_RUST = orig_has_rust

    diff_ort = float(np.max(np.abs(o_rs - o_py)))
    record_result("Efectos Ópticos", "Orton Glow 6.0 MP", t_py_ort, t_rs_ort, diff_ort)

    # B. Star Diffusion
    orig_has_rust = sd.HAS_RUST
    try:
        sd.HAS_RUST = False
        t0 = time.perf_counter()
        sd_py = sd.apply_star_diffusion(sub, strength=0.5, radius=15.0, threshold=0.4, halo_mode="screen")
        t_py_sd = (time.perf_counter() - t0) * 1000

        sd.HAS_RUST = True
        t0 = time.perf_counter()
        sd_rs = sd.apply_star_diffusion(sub, strength=0.5, radius=15.0, threshold=0.4, halo_mode="screen")
        t_rs_sd = (time.perf_counter() - t0) * 1000
    finally:
        sd.HAS_RUST = orig_has_rust

    diff_sd = float(np.max(np.abs(sd_rs - sd_py)))
    record_result("Efectos Ópticos", "Star Diffusion 6.0 MP", t_py_sd, t_rs_sd, diff_sd)

def bench_deconvolution(img_24mp):
    print("\n=== 6. DECONVOLUCIÓN: RICHARDSON-LUCY Y MODELADO PSF (1.0 MP) ===")
    roi = img_24mp[:1000, :1000].copy()
    psf = deconv.generate_moffat_psf(fwhm=3.0, beta=4.5)

    orig_has_rust = deconv.HAS_RUST
    try:
        deconv.HAS_RUST = False
        t0 = time.perf_counter()
        dec_py = deconv.richardson_lucy_deconvolve(roi, psf, iterations=10, deringing=0.01, damping=0.2, use_gpu=False)
        t_py = (time.perf_counter() - t0) * 1000

        deconv.HAS_RUST = True
        t0 = time.perf_counter()
        dec_rs = deconv.richardson_lucy_deconvolve(roi, psf, iterations=10, deringing=0.01, damping=0.2, use_gpu=False)
        t_rs = (time.perf_counter() - t0) * 1000
    finally:
        deconv.HAS_RUST = orig_has_rust

    diff = float(np.max(np.abs(dec_rs - dec_py)))
    record_result("Deconvolución", "Richardson-Lucy (10 iters, 1 MP)", t_py, t_rs, diff)

def bench_meteors(img_24mp):
    print("\n=== 7. METEOROS: DETECCIÓN DE TRAZAS Y FUSIÓN LIGHTEN (24.5 MP) ===")
    curr = img_24mp[..., 0].copy()
    bg = curr * 0.95
    curr[1000:1500, 2000:3000] += 0.35

    # A. Diferencia positiva y estadísticas robustas MAD
    t0 = time.perf_counter()
    diff_py, mean_py, std_py = meteors._numpy_positive_diff_and_stats(curr, bg)
    sample = (curr - bg)[::4, ::4]
    med_py = float(np.median(sample))
    mad_py = float(np.median(np.abs(sample - med_py)))
    t_py = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    diff_rs, mean_rs, std_rs, med_rs, mad_rs = astroloom_rs.meteor_positive_diff_and_stats_rs(curr, bg)
    t_rs = (time.perf_counter() - t0) * 1000

    diff_err = float(np.max(np.abs(diff_rs - diff_py)))
    record_result("Meteoros", "Detección y Stats Robustas 24.5 MP", t_py, t_rs, diff_err)

    # B. Fusión de cápsula Lighten
    base = img_24mp.copy()
    warped_patch = base.copy()
    warped_patch[1000:1500, 2000:3000] += 0.4
    warped_mask = np.zeros((base.shape[0], base.shape[1]), dtype=np.float32)
    warped_mask[1000:1500, 2000:3000] = 0.85

    t0 = time.perf_counter()
    expected = base.copy()
    excess = np.maximum(0.0, warped_patch - expected)
    expected += warped_mask[..., None] * excess
    t_py_blend = (time.perf_counter() - t0) * 1000

    out_rs = base.copy()
    t0 = time.perf_counter()
    astroloom_rs.blend_meteor_capsule_lighten_rs(out_rs, warped_patch, warped_mask)
    t_rs_blend = (time.perf_counter() - t0) * 1000

    diff_blend = float(np.max(np.abs(out_rs - expected)))
    record_result("Meteoros", "Fusión Lighten en Cápsula 24.5 MP", t_py_blend, t_rs_blend, diff_blend)

def bench_eclipse():
    print("\n=== 8. ECLIPSES HDR: FILTROS CORONALES NRGF Y MGN (4.0 MP) ===")
    h, w = 2000, 2000
    cx, cy = 1000.0, 1000.0
    y, x = np.indices((h, w), dtype=np.float32)
    dist = np.hypot(x - cx, y - cy)
    np.random.seed(42)
    corona = (np.exp(-dist / 250.0) + np.random.normal(0, 0.01, (h, w))).clip(0.0, 1.0).astype(np.float32)

    # A. NRGF
    orig_has_rust = eclipse.HAS_RUST
    try:
        eclipse.HAS_RUST = False
        t0 = time.perf_counter()
        nrgf_py = eclipse.apply_nrgf(corona, cx, cy, radius_min=150.0)
        t_py_nrgf = (time.perf_counter() - t0) * 1000

        eclipse.HAS_RUST = True
        t0 = time.perf_counter()
        nrgf_rs = eclipse.apply_nrgf(corona, cx, cy, radius_min=150.0)
        t_rs_nrgf = (time.perf_counter() - t0) * 1000
    finally:
        eclipse.HAS_RUST = orig_has_rust

    diff_nrgf = float(np.max(np.abs(nrgf_rs - nrgf_py)))
    record_result("Eclipses", "Filtro Coronal NRGF 4.0 MP", t_py_nrgf, t_rs_nrgf, diff_nrgf)

    # B. MGN
    try:
        eclipse.HAS_RUST = False
        t0 = time.perf_counter()
        mgn_py = eclipse.apply_mgn(corona, sigmas=(3.0, 7.0, 15.0), lunar_mask_params=(cx, cy, 150.0))
        t_py_mgn = (time.perf_counter() - t0) * 1000

        eclipse.HAS_RUST = True
        t0 = time.perf_counter()
        mgn_rs = eclipse.apply_mgn(corona, sigmas=(3.0, 7.0, 15.0), lunar_mask_params=(cx, cy, 150.0))
        t_rs_mgn = (time.perf_counter() - t0) * 1000
    finally:
        eclipse.HAS_RUST = orig_has_rust

    diff_mgn = float(np.max(np.abs(mgn_rs - mgn_py)))
    record_result("Eclipses", "Filtro Coronal MGN 4.0 MP", t_py_mgn, t_rs_mgn, diff_mgn)

def bench_astrometry_and_panoramas():
    print("\n=== 9. ASTROMETRÍA: TRIÁNGULOS RANSAC, ASTERISMOS Y MALLAS TPS ===")
    # A. Triángulos Estelares RANSAC (100 estrellas + 25 falsas)
    np.random.seed(42)
    N_stars = 100
    pts1_syn = np.random.uniform(50, 3950, (N_stars, 2)).astype(np.float32)
    theta = np.radians(8.5)
    c, s = np.cos(theta), np.sin(theta)
    scale = 1.015
    tx, ty = 85.0, -45.0
    M_gt = np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    M_inv = np.linalg.inv(M_gt)
    p1_hom = np.hstack([pts1_syn, np.ones((N_stars, 1), dtype=np.float32)])
    p2_proj = (M_inv @ p1_hom.T).T
    pts2_syn = (p2_proj[:, :2] / p2_proj[:, 2:3]).astype(np.float32)
    p1_full = np.vstack([pts1_syn, np.random.uniform(50, 3950, (25, 2)).astype(np.float32)])
    p2_full = np.vstack([pts2_syn, np.random.uniform(50, 3950, (25, 2)).astype(np.float32)])

    orig_has_rust = pano.HAS_RUST
    try:
        pano.HAS_RUST = False
        t0 = time.perf_counter()
        H_py, inliers_py = pano.match_star_triangles(p1_full, p2_full, tolerance=0.035, max_ransac_iters=5000, inlier_thresh=4.0)
        t_py_tri = (time.perf_counter() - t0) * 1000

        pano.HAS_RUST = True
        t0 = time.perf_counter()
        H_rs, inliers_rs = pano.match_star_triangles(p1_full, p2_full, tolerance=0.035, max_ransac_iters=5000, inlier_thresh=4.0)
        t_rs_tri = (time.perf_counter() - t0) * 1000
    finally:
        pano.HAS_RUST = orig_has_rust

    record_result("Astrometría", "Triángulos Estelares RANSAC", t_py_tri, t_rs_tri, None, f"[Coincidencias: {np.sum(inliers_rs)}/{len(p1_full)} estrellas]")

    # B. Asterismos Estelares (120 estrellas)
    pts1 = np.random.uniform(50, 4000, (120, 2)).astype(np.float32)
    theta = np.deg2rad(2.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    pts2 = (pts1 @ R.T) + np.array([50.0, -30.0], dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(10):
        py_f1, py_q1 = pano.build_star_asterisms(pts1, k_neighbors=7)
        py_f2, py_q2 = pano.build_star_asterisms(pts2, k_neighbors=7)
        _ = pano.match_star_asterisms(py_f1, py_q1, pts1, py_f2, py_q2, pts2, 0.020, 2)
    t_py_ast = (time.perf_counter() - t0) / 10.0 * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        rs_f1, rs_q1 = astroloom_rs.build_star_asterisms_rs(pts1, 7)
        rs_f2, rs_q2 = astroloom_rs.build_star_asterisms_rs(pts2, 7)
        _ = astroloom_rs.match_star_asterisms_rs(rs_f1, rs_q1, rs_f2, rs_q2, 0.020, 2)
    t_rs_ast = (time.perf_counter() - t0) / 10.0 * 1000

    record_result("Astrometría", "Asterismos (120 estrellas, k=7)", t_py_ast, t_rs_ast)

    # C. Deformación TPS Grid (24x24)
    from scipy.interpolate import RBFInterpolator
    w, h, grid_size = 6000, 4000, 24
    N = 36
    pts = np.column_stack([np.random.uniform(100, w-100, N), np.random.uniform(100, h-100, N)]).astype(np.float32)
    res = np.random.uniform(-3.0, 3.0, (N, 2)).astype(np.float32)
    gx = np.linspace(0, w - 1, grid_size, dtype=np.float32)
    gy = np.linspace(0, h - 1, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    t0 = time.perf_counter()
    for _ in range(25):
        rbf = RBFInterpolator(pts, res, kernel='thin_plate_spline', smoothing=2.0)
        out_py_tps = rbf(grid_pts).reshape(grid_size, grid_size, 2).astype(np.float32)
    t_py_tps = (time.perf_counter() - t0) / 25.0 * 1000

    t0 = time.perf_counter()
    for _ in range(25):
        out_rs_tps = astroloom_rs.compute_tps_grid_rs(pts, res, grid_size, grid_size, float(w), float(h), 2.0)
    t_rs_tps = (time.perf_counter() - t0) / 25.0 * 1000

    diff_tps = float(np.max(np.abs(out_py_tps - out_rs_tps)))
    record_result("Astrometría", "TPS Grid Interpolation (24x24)", t_py_tps, t_rs_tps, diff_tps)

def bench_photometry_and_dualband(img_24mp):
    print("\n=== 10. FOTOMETRÍA: REGRESIÓN THEIL-SEN PCC Y PALETAS DUAL-BAND ===")
    # A. Regresión Theil-Sen Gaia BP-RP (50 estrellas de calibración)
    np.random.seed(42)
    stars = []
    catalog = []
    for i in range(50):
        fg = float(0.15 + i * 0.015)
        stars.append({"flux_r": fg * (0.8 + 0.1 * np.sin(i)), "flux_g": fg, "flux_b": fg * (1.1 - 0.08 * np.cos(i))})
        catalog.append(0.5 + 0.03 * i)
    cat_arr = np.array(catalog, dtype=np.float64)

    orig_has_rust = pcc.HAS_RUST
    try:
        pcc.HAS_RUST = False
        t0 = time.perf_counter()
        for _ in range(10):
            kr_py, _, kb_py = pcc.fit_theil_sen_pcc(stars, bp_rp_catalog=cat_arr)
        t_py_pcc = (time.perf_counter() - t0) / 10.0 * 1000

        pcc.HAS_RUST = True
        t0 = time.perf_counter()
        for _ in range(10):
            kr_rs, _, kb_rs = pcc.fit_theil_sen_pcc(stars, bp_rp_catalog=cat_arr)
        t_rs_pcc = (time.perf_counter() - t0) / 10.0 * 1000
    finally:
        pcc.HAS_RUST = orig_has_rust

    diff_pcc = float(abs(kr_py - kr_rs) + abs(kb_py - kb_rs))
    record_result("Fotometría", "Theil-Sen Gaia PCC (50 estrellas)", t_py_pcc, t_rs_pcc, diff_pcc)

    # B. Paleta Dual-Band HOO (6.0 MP)
    sub = img_24mp[:2000, :3000].copy()
    orig_has_rust = db.HAS_RUST
    try:
        db.HAS_RUST = False
        t0 = time.perf_counter()
        db_py = db.apply_dualband_palette(sub, palette="HOO", boost_oiii=1.2, desat_cores=True)
        t_py_db = (time.perf_counter() - t0) * 1000

        db.HAS_RUST = True
        t0 = time.perf_counter()
        db_rs = db.apply_dualband_palette(sub, palette="HOO", boost_oiii=1.2, desat_cores=True)
        t_rs_db = (time.perf_counter() - t0) * 1000
    finally:
        db.HAS_RUST = orig_has_rust

    diff_db = float(np.max(np.abs(db_rs - db_py)))
    record_result("Fotometría", "Paleta Dual-Band HOO (6.0 MP)", t_py_db, t_rs_db, diff_db)

def bench_spatial_filters(img_24mp):
    print("\n=== 11. FILTROS ESPACIALES: SUPRESIÓN BANDING CMOS Y SDF ANTI-GHOSTING ===")
    mono = img_24mp[..., 0].copy()

    # A. Supresión Banding CMOS
    orig_has_rust = banding.HAS_RUST
    try:
        banding.HAS_RUST = False
        t0 = time.perf_counter()
        b_py = banding.suppress_cmos_banding(mono, direction="horizontal", amount=1.0)
        t_py_b = (time.perf_counter() - t0) * 1000

        banding.HAS_RUST = True
        t0 = time.perf_counter()
        b_rs = banding.suppress_cmos_banding(mono, direction="horizontal", amount=1.0)
        t_rs_b = (time.perf_counter() - t0) * 1000
    finally:
        banding.HAS_RUST = orig_has_rust

    diff_b = float(np.max(np.abs(b_rs - b_py)))
    record_result("Filtros Espaciales", "Supresión Banding CMOS 24.5 MP", t_py_b, t_rs_b, diff_b)

    # B. SDF Anti-Ghosting Cielo/Suelo (6.0 MP)
    sub = img_24mp[:2000, :3000].copy()
    h, w = sub.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:h//2, :] = 1.0
    ground = sub * 0.4

    orig_has_rust = st.HAS_RUST
    try:
        st.HAS_RUST = False
        t0 = time.perf_counter()
        c_py = st.compose_sky_ground_with_anti_ghosting(sub, ground, mask, enable_anti_ghosting=True, boundary_radius=20, tree_sensitivity=0.6)
        t_py_sdf = (time.perf_counter() - t0) * 1000

        st.HAS_RUST = True
        t0 = time.perf_counter()
        c_rs = st.compose_sky_ground_with_anti_ghosting(sub, ground, mask, enable_anti_ghosting=True, boundary_radius=20, tree_sensitivity=0.6)
        t_rs_sdf = (time.perf_counter() - t0) * 1000
    finally:
        st.HAS_RUST = orig_has_rust

    diff_sdf = float(np.max(np.abs(c_rs - c_py)))
    record_result("Filtros Espaciales", "SDF Anti-Ghosting 6.0 MP", t_py_sdf, t_rs_sdf, diff_sdf)

def bench_calibration_and_icc(img_24mp):
    print("\n=== 12. CALIBRACIÓN: CFA IN-PLACE UINT16 Y GESTIÓN ICC BRADFORD ===")
    h, w = img_24mp.shape[:2]

    # A. Calibración CFA In-Place uint16
    np.random.seed(42)
    cfa_raw = np.random.randint(1000, 15000, (h, w), dtype=np.uint16)
    dark = np.random.uniform(200, 600, (h, w)).astype(np.float32)
    flat = np.random.uniform(0.7, 1.2, (h, w)).astype(np.float32)

    cfa_rs = cfa_raw.copy()
    t0 = time.perf_counter()
    astroloom_rs.calibrate_raw_cfa_inplace_rs(cfa_rs, master_dark=dark, master_flat=flat, black_level=512.0, dark_pedestal=300.0)
    t_rs_cfa = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sig = (cfa_raw.astype(np.float32) - (dark - 300.0)) - 512.0
    corr = sig / np.where(flat > 1e-4, flat, 1.0) + 512.0
    cfa_py = np.clip(np.round(corr), 0, 65535).astype(np.uint16)
    t_py_cfa = (time.perf_counter() - t0) * 1000

    diff_cfa = float(np.max(np.abs(cfa_rs.astype(np.int32) - cfa_py.astype(np.int32))))
    record_result("Calibración", "Calibración CFA In-Place 24.5 MP", t_py_cfa, t_rs_cfa, diff_cfa)

    # B. Conversión ICC Bradford (sRGB -> ProPhoto RGB) 24.5 MP
    orig_has_rust = icc.HAS_RUST
    try:
        icc.HAS_RUST = False
        t0 = time.perf_counter()
        conv_py = icc.convert_colorspace(img_24mp, src_space="srgb", dst_space="prophoto")
        t_py_icc = (time.perf_counter() - t0) * 1000

        icc.HAS_RUST = True
        t0 = time.perf_counter()
        conv_rs = icc.convert_colorspace(img_24mp, src_space="srgb", dst_space="prophoto")
        t_rs_icc = (time.perf_counter() - t0) * 1000
    finally:
        icc.HAS_RUST = orig_has_rust

    diff_icc = float(np.max(np.abs(conv_rs - conv_py)))
    record_result("Calibración", "Conversión ICC Bradford 24.5 MP", t_py_icc, t_rs_icc, diff_icc)

    # C. Startrails Acumulación In-Place
    acc_py = np.full((h, w, 3), 0.1, dtype=np.float32)
    acc_rs = acc_py.copy()
    frame = np.full((h, w, 3), 0.3, dtype=np.float32)
    prev_d = np.full((h, w, 3), 0.25, dtype=np.float32)
    curr_d = np.full((h, w, 3), 0.28, dtype=np.float32)

    t0 = time.perf_counter()
    bridge = np.minimum(prev_d, curr_d)
    bridge[bridge < 0.005] = 0.0
    bridge *= 0.9
    np.maximum(acc_py, bridge, out=acc_py)
    np.maximum(acc_py, frame * 0.8, out=acc_py)
    t_py_st = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    astroloom_rs.accumulate_startrail_step_rs(acc_rs, frame, weight=0.8, prev_dilated=prev_d, curr_dilated=curr_d, w_bridge=0.9)
    t_rs_st = (time.perf_counter() - t0) * 1000

    diff_st = float(np.max(np.abs(acc_py - acc_rs)))
    record_result("Calibración", "Startrails Accumulation 24.5 MP", t_py_st, t_rs_st, diff_st)

def bench_ai_tiling_and_guided(img_24mp):
    print("\n=== 13. TILING AI Y FILTRO GUIADO: FAST GUIDED FILTER Y VENTANAS HANN ===")
    import cv2
    sub_mono = img_24mp[:2000, :3000, 0].copy()

    # A. Fast Guided Filter (6.0 MP)
    def guided_filter_py(I, p, r, eps):
        ksize = (2 * r + 1, 2 * r + 1)
        mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
        return mean_a * I + mean_b

    t0 = time.perf_counter()
    q_py = guided_filter_py(sub_mono, sub_mono, 10, 1e-3)
    t_py_gf = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    q_rs = astroloom_rs.fast_guided_filter_rs(sub_mono, sub_mono, radius=10, eps=1e-3)
    t_rs_gf = (time.perf_counter() - t0) * 1000

    diff_gf = float(np.max(np.abs(q_rs - q_py)))
    record_result("AI Tiling", "Fast Guided Filter 6.0 MP", t_py_gf, t_rs_gf, diff_gf)

    # B. StarNet Tiling & Blending Hann (6.0 MP)
    sub_rgb = img_24mp[:2000, :3000].copy()
    h, w = sub_rgb.shape[:2]
    tile_size = 512
    stride = 384

    # Rust Prepare & Assemble
    t0 = time.perf_counter()
    tiles_rs, coords_rs, bps_rs, ms_rs = astroloom_rs.prepare_starnet_tiles_rs(sub_rgb, tile_size=tile_size, stride=stride, is_linear=True)
    recon_rs = astroloom_rs.assemble_starnet_tiles_rs(
        tiles_rs, coords_rs, orig_h=h, orig_w=w, stride=stride, tile_size=tile_size,
        is_linear=True, bps=bps_rs, ms=ms_rs, orig_img=sub_rgb, use_hann=True
    )
    t_rs_tiling = (time.perf_counter() - t0) * 1000

    # Python Prepare
    t0 = time.perf_counter()
    stretched = np.zeros_like(sub_rgb)
    for c in range(3):
        bp, m = s_onnx._calc_channel_mtf(sub_rgb[..., c], target_bg=0.25, shadows_factor=-2.80)
        c_norm = np.clip((sub_rgb[..., c] - bp) / max(1e-6, 1.0 - bp), 0.0, 1.0)
        stretched[..., c] = np.clip(s_onnx._mtf_curve(c_norm, m), 0.0, 1.0)
    y_coords = s_onnx._get_tile_coords(h, tile_size, stride)
    x_coords = s_onnx._get_tile_coords(w, tile_size, stride)
    tiles_py = [stretched[y : y + tile_size, x : x + tile_size] for y in y_coords for x in x_coords]
    t_py_tiling = (time.perf_counter() - t0) * 1000

    record_result("AI Tiling", f"StarNet Tiling Prep+Blend ({len(coords_rs)} tiles)", t_py_tiling, t_rs_tiling)

def bench_raw_streaming_io():
    print("\n=== 14. STREAMING I/O: ESCANEO DE METADATOS RAW CONCURRENTE ===")
    nefs = glob.glob("fotos_prueba/*.NEF")
    if not nefs:
        nefs = glob.glob("bias/horizontal/**/*.NEF", recursive=True)

    if nefs:
        test_nefs = nefs[:10]
        orig_has_rust = st.HAS_RUST
        try:
            # Python rawpy sequential loop
            st.HAS_RUST = False
            t0 = time.perf_counter()
            meta_py = st.scan_raw_batch_metadata(test_nefs)
            t_py_scan = (time.perf_counter() - t0) * 1000

            # Rust Rayon parallel binary TIFF parser
            st.HAS_RUST = True
            t0 = time.perf_counter()
            meta_rs = st.scan_raw_batch_metadata(test_nefs)
            t_rs_scan = (time.perf_counter() - t0) * 1000
        finally:
            st.HAS_RUST = orig_has_rust

        record_result("Streaming I/O", f"Escaneo Batch RAW ({len(test_nefs)} tomas Nikon)", t_py_scan, t_rs_scan, None, f"[Archivos reales]")
    else:
        print("  [SKIP] No se encontraron archivos NEF de prueba locales.")

def bench_project_serialization():
    print("\n=== 15. PROYECTOS: SERIALIZACIÓN PARALELA .DEVLOOM ===")
    real_devloom = "imagenes/resultado_dual_32bit.devloom"
    if os.path.exists(real_devloom):
        # Medir lectura de proyecto real 298 MB
        pio.HAS_RUST_PROJECT_IO = False
        t0 = time.perf_counter()
        _ = pio.load_devloom_project(real_devloom)
        t_py_load = (time.perf_counter() - t0) * 1000

        pio.HAS_RUST_PROJECT_IO = True
        t0 = time.perf_counter()
        _ = pio.load_devloom_project(real_devloom)
        t_rs_load = (time.perf_counter() - t0) * 1000

        record_result("Proyectos", "Carga Proyecto Real (298 MB)", t_py_load, t_rs_load, None, "[Dato real en disco]")
    else:
        # Medir guardado y carga sintético de sesión completa
        h, w = 600, 600
        rng = np.random.RandomState(42)
        sl = rng.rand(h, w, 3).astype(np.float32)
        st_img = rng.rand(h, w, 3).astype(np.float32)
        mask = rng.rand(h, w).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            f_rs = os.path.join(tmpdir, "bench_rs.devloom")
            pio.HAS_RUST_PROJECT_IO = False
            t0 = time.perf_counter()
            pio.save_devloom_project(f_rs, {"test": True}, [], mask, starnet_data={"image_starless": sl, "image_stars": st_img})
            _ = pio.load_devloom_project(f_rs)
            t_py = (time.perf_counter() - t0) * 1000

            pio.HAS_RUST_PROJECT_IO = True
            t0 = time.perf_counter()
            pio.save_devloom_project(f_rs, {"test": True}, [], mask, starnet_data={"image_starless": sl, "image_stars": st_img})
            _ = pio.load_devloom_project(f_rs)
            t_rs = (time.perf_counter() - t0) * 1000

            record_result("Proyectos", "Guardado/Carga Sesión Completa", t_py, t_rs)

def bench_canvas_renderer(img_24mp):
    print("\n=== 16. RENDERIZADO: MOTOR DIRECTO BGRA8888 Y ALERTAS DE CLIPPING (24.5 MP) ===")
    import cv2
    # Rust Rayon Fused Renderer
    t0 = time.perf_counter()
    buf_rs = astroloom_rs.render_canvas_frame_rs(img_24mp, show_shadows=True, show_highlights=True)
    t_rs = (time.perf_counter() - t0) * 1000

    # Python / OpenCV Canvas pipeline
    t0 = time.perf_counter()
    disp = cv2.convertScaleAbs(np.clip(img_24mp, 0.0, 1.0), alpha=255.0)
    bgra_py = cv2.cvtColor(disp, cv2.COLOR_RGB2BGRA)
    shadows = np.all(disp == 0, axis=-1)
    highlights = np.any(disp == 255, axis=-1)
    bgra_py[shadows] = [255, 112, 0, 255]
    bgra_py[highlights] = [68, 23, 255, 255]
    t_py = (time.perf_counter() - t0) * 1000

    record_result("Renderizado", "Buffer BGRA8888 + Clipping 24.5 MP", t_py, t_rs)

def print_summary_table():
    print("\n" + "=" * 92)
    print(f"{'RESUMEN GLOBAL DEL BENCHMARK: RUST vs PYTHON / NUMPY / NUMBA / SCIPY':^92}")
    print("=" * 92)
    print(f"{'Categoría':<16} | {'Operación Evaluada':<35} | {'Python (ms)':>11} | {'Rust (ms)':>9} | {'Speedup':>9}")
    print("-" * 92)
    total_py = 0.0
    total_rs = 0.0
    for r in BENCH_RESULTS:
        total_py += r["t_py_ms"]
        total_rs += r["t_rs_ms"]
        print(f"{r['category']:<16} | {r['name']:<35} | {r['t_py_ms']:11.1f} | {r['t_rs_ms']:9.1f} | {r['speedup']:8.2f}x")
    print("-" * 92)
    total_speedup = total_py / max(total_rs, 1e-6)
    print(f"{'TIEMPO TOTAL':<16} | {'Suma acumulada del conjunto':<35} | {total_py:11.1f} | {total_rs:9.1f} | {total_speedup:8.2f}x")
    print("=" * 92 + "\n")

def main():
    parser = argparse.ArgumentParser(description="AstroLoom: Benchmark Exhaustivo Rust vs Python")
    parser.add_argument("--section", choices=[
        "all", "stacking", "background", "developer", "stretch", "optical",
        "deconv", "meteors", "eclipse", "astrometry", "photometry",
        "spatial", "calibration", "ai", "raw", "project", "renderer"
    ], default="all", help="Ejecutar una sección específica o todas")
    args = parser.parse_args()

    print("=" * 70)
    print(" AstroLoom: Benchmark Comparativo de Alto Rendimiento (Rust vs Python)")
    print("=" * 70)

    # Iniciar cronometraje global
    t_global_start = time.perf_counter()

    # Carga de datos base (imagen astronómica apilada de 24.5 MP)
    img_24mp = load_real_or_synthetic_24mp()

    sec = args.section
    if sec in ("all", "stacking"):
        bench_stacking_kernels()
    if sec in ("all", "background"):
        bench_background_extraction(img_24mp)
    if sec in ("all", "developer"):
        bench_developer_and_layers(img_24mp)
    if sec in ("all", "stretch"):
        bench_stretch_and_ghs(img_24mp)
    if sec in ("all", "optical"):
        bench_optical_effects(img_24mp)
    if sec in ("all", "deconv"):
        bench_deconvolution(img_24mp)
    if sec in ("all", "meteors"):
        bench_meteors(img_24mp)
    if sec in ("all", "eclipse"):
        bench_eclipse()
    if sec in ("all", "astrometry"):
        bench_astrometry_and_panoramas()
    if sec in ("all", "photometry"):
        bench_photometry_and_dualband(img_24mp)
    if sec in ("all", "spatial"):
        bench_spatial_filters(img_24mp)
    if sec in ("all", "calibration"):
        bench_calibration_and_icc(img_24mp)
    if sec in ("all", "ai"):
        bench_ai_tiling_and_guided(img_24mp)
    if sec in ("all", "raw"):
        bench_raw_streaming_io()
    if sec in ("all", "project"):
        bench_project_serialization()
    if sec in ("all", "renderer"):
        bench_canvas_renderer(img_24mp)

    print_summary_table()
    total_sec = time.perf_counter() - t_global_start
    print(f"Benchmark completado en {total_sec:.2f} segundos.")

if __name__ == "__main__":
    main()
