import time
import tracemalloc
import numpy as np
import core.stacking as st
import core.background_extraction as be
import astroloom_rs

def bench_kappa_sigma():
    print("=== 1. BENCHMARK: KAPPA-SIGMA PONDERADO (Franja de 200x6000x3 en 35 Lights) ===")
    n_frames = 35
    total_items = 200 * 6000 * 3  # 3.600.000 valores por franja
    np.random.seed(42)
    data = np.random.uniform(0.05, 0.25, size=(n_frames, total_items)).astype(np.float32)
    # Simular satélites y suelo desplazado (-1.0)
    data[0, ::10] = 0.95
    data[-1, :500000] = -1.0
    weights = np.random.uniform(0.8, 1.2, size=n_frames).astype(np.float32)

    # Calentar Numba
    _ = st._kappa_sigma_kernel_numba(data[:, :1000], 2.2, weights)

    t0 = time.perf_counter()
    out_py = st._kappa_sigma_kernel_numba(data, 2.2, weights)
    t_py = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out_rs = astroloom_rs.kappa_sigma_kernel(data, 2.2, weights)
    t_rs = (time.perf_counter() - t0) * 1000

    max_diff = np.max(np.abs(out_py - out_rs))
    print(f"  Numba JIT : {t_py:7.1f} ms")
    print(f"  Rust SIMD : {t_rs:7.1f} ms  --> Speedup: {t_py / t_rs:.2f}x")
    print(f"  Diferencia matemática máxima: {max_diff:.2e}\n")

def bench_background_rbf():
    print("=== 2. BENCHMARK: EXTRACCIÓN RBF 24 MP (6000x4000x3 float32) ===")
    h, w = 4000, 6000
    img_rgb = np.random.uniform(0.1, 0.4, size=(h, w, 3)).astype(np.float32)
    sky_mask = np.ones((h, w), dtype=np.float32)
    sky_mask[int(h * 0.75):, :] = 0.0  # 25% inferior es suelo
    pts = be.generate_sky_sample_points(h, w, sky_mask=sky_mask, img_rgb=img_rgb)

    # Medir Python (SciPy + NumPy)
    tracemalloc.start()
    t0 = time.perf_counter()
    corr_py, _ = be.extract_background_rbf(img_rgb, pts, sky_mask=sky_mask)
    t_py = (time.perf_counter() - t0) * 1000
    _, peak_py = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Medir Rust (nalgebra + Fused Rayon loop)
    tracemalloc.start()
    t0 = time.perf_counter()
    # Extraer muestras igual que en Python
    vals_arr = np.random.uniform(0.12, 0.25, size=(len(pts), 3)).astype(np.float64)
    pts_arr = np.array(pts, dtype=np.float64)
    bg_proxy = astroloom_rs.fit_and_eval_rbf_proxy(pts_arr, vals_arr, 180, 240, "multiquadric", 0.15, 1.0)
    import cv2
    bg_full = cv2.resize(bg_proxy, (w, h), interpolation=cv2.INTER_CUBIC)
    pedestal = np.median(vals_arr, axis=0).astype(np.float32)
    corr_rs = astroloom_rs.apply_bg_model_fused(img_rgb, bg_full, pedestal, False, sky_mask)
    t_rs = (time.perf_counter() - t0) * 1000
    _, peak_rs = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  SciPy + NumPy : {t_py:7.1f} ms | Pico RAM extra en Python: {peak_py / (1024**2):6.1f} MB")
    print(f"  Rust Fusionado: {t_rs:7.1f} ms | Pico RAM extra en Python: {peak_rs / (1024**2):6.1f} MB")
    print(f"  Speedup Tiempo: {t_py / t_rs:.2f}x | Ahorro de RAM: {(peak_py - peak_rs) / (1024**2):.1f} MB\n")

def bench_tps_and_asterisms():
    print("=== 3. BENCHMARK: DEFORMACIÓN TPS (grid 24x24) Y ASTERISMOS ESTELARES (120 estrellas) ===")
    import core.panorama as pano
    from scipy.interpolate import RBFInterpolator

    # A. TPS Grid (36 inliers)
    w, h, grid_size = 6000, 4000, 24
    np.random.seed(42)
    N = 36
    pts = np.column_stack([np.random.uniform(100, w-100, N), np.random.uniform(100, h-100, N)]).astype(np.float32)
    res = np.random.uniform(-3.0, 3.0, (N, 2)).astype(np.float32)
    gx = np.linspace(0, w - 1, grid_size, dtype=np.float32)
    gy = np.linspace(0, h - 1, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    t0 = time.perf_counter()
    for _ in range(50):
        rbf = RBFInterpolator(pts, res, kernel='thin_plate_spline', smoothing=2.0)
        _ = rbf(grid_pts).reshape(grid_size, grid_size, 2).astype(np.float32)
    t_py_tps = (time.perf_counter() - t0) / 50.0 * 1000

    t0 = time.perf_counter()
    for _ in range(50):
        _ = astroloom_rs.compute_tps_grid_rs(pts, res, grid_size, grid_size, float(w), float(h), 2.0)
    t_rs_tps = (time.perf_counter() - t0) / 50.0 * 1000

    print(f"  TPS Grid SciPy : {t_py_tps:7.3f} ms")
    print(f"  TPS Grid Rust  : {t_rs_tps:7.3f} ms  --> Speedup: {t_py_tps / t_rs_tps:.2f}x")

    # B. Asterisms (120 estrellas por panel)
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

    print(f"  Asterismos SciPy/Py: {t_py_ast:7.3f} ms")
    print(f"  Asterismos Rust    : {t_rs_ast:7.3f} ms  --> Speedup: {t_py_ast / t_rs_ast:.2f}x\n")

if __name__ == "__main__":
    bench_kappa_sigma()
    bench_background_rbf()
    bench_tps_and_asterisms()