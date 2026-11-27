"""
Verificación automatizada de CI y desarrollo local para AstroLoom.
Comprueba:
1. Compilación de sintaxis (py_compile) de todos los módulos en core/ y gui/.
2. Núcleo nativo astroloom_rs y algoritmos clave (eclipse, startrails, dualband, histogram, onnx_runtime).
3. Exportación plana (gui/export_utils) en TIFF 16/32-bit, PNG 16-bit y JPEG.
4. Arranque de MainWindow en modo offscreen, precarga de pestañas y ciclo Undo/Redo.
"""

from __future__ import annotations

import os
import py_compile
import sys
import tempfile

import cv2
import numpy as np

# Asegurar que la raíz del repo esté en sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def check_syntax() -> None:
    count = 0
    for root_dir in ("core", "gui"):
        base = os.path.join(REPO_ROOT, root_dir)
        for r, _, fs in os.walk(base):
            for f in fs:
                if f.endswith(".py"):
                    py_compile.compile(os.path.join(r, f), doraise=True)
                    count += 1
    print(f"[OK] py_compile ({count} archivos)")


def check_core_rust() -> None:
    import astroloom_rs
    from core import dualband, eclipse, histogram, onnx_runtime
    from core.startrails import generate_startrail

    # 1. Eclipse HDR & filtros coronales (todos los submódulos de eclipse/)
    h, w = 64, 64
    cy, cx, rad = 32.0, 32.0, 10.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    corona = np.where(rr >= rad, np.exp(-(rr - rad) / 14.0), 0.0).astype(np.float32)
    rgb = np.stack([corona * 0.9, corona * 0.85, corona * 0.8], axis=-1)

    for fn_name, out in [
        ("asinh", eclipse.apply_asinh_stretch(rgb, stretch_factor=10.0, color_preserve=0.5)),
        ("nrgf", eclipse.apply_nrgf(rgb, cx, cy, radius_min=rad)),
        ("mgn", eclipse.apply_mgn(rgb, sigmas=(1.5,), lunar_mask_params=(cx, cy, rad))),
        ("g2", eclipse.apply_steerable_g2(rgb, cx, cy, radius_min=rad, sigmas=(1.5,))),
        ("larson", eclipse.apply_larson_sekanina(rgb, cx, cy, radius_min=rad)),
        ("swap", eclipse.apply_swap_filter(rgb, cx, cy, radius_min=rad)),
        ("wow", eclipse.apply_wow_edge_aware(rgb, cx, cy, radius_min=rad, n_scales=3)),
        ("nafe", eclipse.apply_nafe(rgb, cx, cy, radius_min=rad, n=33)),
    ]:
        assert out.shape == rgb.shape and np.isfinite(out).all(), f"eclipse.{fn_name} failed"

    # 2. Paletas Dual-Band distintas entre HOO, Foraxx y Cannistra
    test_neb = np.zeros((32, 32, 3), dtype=np.float32)
    test_neb[..., 0] = np.linspace(0.05, 0.8, 32, dtype=np.float32)[:, None]
    test_neb[..., 1] = np.linspace(0.05, 0.6, 32, dtype=np.float32)[None, :]
    test_neb[..., 2] = np.linspace(0.05, 0.5, 32, dtype=np.float32)[None, :]
    p_hoo = dualband.apply_dualband_palette(test_neb, palette="HOO")
    p_foraxx = dualband.apply_dualband_palette(test_neb, palette="Foraxx")
    p_can = dualband.apply_dualband_palette(test_neb, palette="Cannistra")
    assert not np.allclose(p_hoo, p_foraxx, atol=1e-3), "HOO and Foraxx should differ"
    assert not np.allclose(p_hoo, p_can, atol=1e-3), "HOO and Cannistra should differ"

    # 3. Histograma clipping por canal
    clip_img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    clip_img[0, 0, 1] = 0.0  # solo canal G recortado a negro
    _, _, _, _, has_shadow_clip, _ = histogram.compute_live_histogram(clip_img)
    assert has_shadow_clip, "Shadow clipping must detect single-channel zero"

    # 4. Startrails
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(3):
            img = np.zeros((48, 48, 3), dtype=np.uint8)
            img[16 + i * 2, 16 + i * 2] = (220, 200, 180)
            p = os.path.join(td, f"f_{i}.png")
            cv2.imwrite(p, img)
            paths.append(p)
        st_img, _ = generate_startrail(paths, mode="comet", suppress_streaks=True, fill_gaps=True)
        assert st_img.shape == (48, 48, 3) and np.isfinite(st_img).all()

    # 5. ONNX runtime listeners
    ev: list[str] = []
    onnx_runtime.add_gpu_fallback_listener(ev.append)
    onnx_runtime.notify_gpu_fallback("ci_check")
    assert "ci_check" in ev

    assert hasattr(astroloom_rs, "scan_raw_metadata_rs")
    print("[OK] Core & astroloom_rs smoke tests")


def check_gui_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from gui.export_utils import save_flat_image
    from gui.main_window import MainWindow

    img = np.linspace(0.0, 1.0, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    with tempfile.TemporaryDirectory() as td:
        for fmt, ext in [("tiff16", ".tif"), ("tiff32", "_32.tif"), ("png", ".png"), ("jpg", ".jpg")]:
            p = os.path.join(td, f"out{ext}")
            save_flat_image(p, img, fmt)
            assert os.path.isfile(p) and os.path.getsize(p) > 0

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    assert set(w._loaded_tabs.keys()) == {0, 1}
    while w._preload_queue:
        w._preload_next_tab()
    assert set(w._loaded_tabs.keys()) == set(range(8))

    dev = w.tab_developer
    dev.undo_stack.append(dev._capture_state())
    dev.undo()
    assert len(dev.redo_stack) == 1
    dev.redo()
    assert len(dev.undo_stack) == 1
    w.close()
    print("[OK] GUI offscreen, export_utils & undo/redo")


if __name__ == "__main__":
    check_syntax()
    check_core_rust()
    check_gui_offscreen()
    print("ALL CI CHECKS PASSED")
