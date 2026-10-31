# tests/test_meteor_fix.py
import os
import sys
import shutil
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
import tifffile
import cv2
from PySide6.QtWidgets import QApplication

def run_test():
    app = QApplication.instance() or QApplication(sys.argv)
    from gui.tab_meteors import MeteorTab, MeteorWorker

    tab = MeteorTab()
    print("MeteorTab instantiated successfully.")

    tmp_dir = os.path.abspath("tmp_test_meteors")
    os.makedirs(tmp_dir, exist_ok=True)

    f1 = os.path.join(tmp_dir, "frame_001.tif")
    f2 = os.path.join(tmp_dir, "frame_002.tif")
    f3 = os.path.join(tmp_dir, "frame_003.tif")

    base = np.zeros((400, 600, 3), dtype=np.uint16)
    for (sy, sx) in [(50, 100), (120, 300), (250, 450), (80, 520), (320, 150), (200, 200), (350, 350)]:
        base[max(0, sy-2):min(400, sy+3), max(0, sx-2):min(600, sx+3)] = 40000

    tifffile.imwrite(f1, base)

    # Frame 2 has a meteor streak
    f2_img = base.copy()
    cv2.line(f2_img, (150, 150), (280, 280), (62000, 60000, 58000), 2)
    tifffile.imwrite(f2, f2_img)

    tifffile.imwrite(f3, base)

    # 1. Test Ingestion and Table Population
    tab.add_series_files([f1, f2, f3])
    assert tab.table_files.rowCount() == 3, f"Expected 3 rows, got {tab.table_files.rowCount()}"

    for r in range(3):
        ref_txt = tab.table_files.item(r, 0).text()
        fn_txt = tab.table_files.item(r, 1).text()
        time_txt = tab.table_files.item(r, 2).text()
        print(f"Row {r}: REF={ascii(ref_txt)}, File={ascii(fn_txt)}, Time={ascii(time_txt)}")
        assert len(fn_txt) > 0, f"Filename empty on row {r}"
        assert len(time_txt) > 0, f"Timestamp empty on row {r}"
        if r == 0:
            assert ref_txt == "★ REF"
        else:
            assert ref_txt == ""

    print("Table columns verified: NOT blank!")

    # 2. Test Parallel Scanning with Numba and ThreadPoolExecutor
    worker = MeteorWorker("scan_meteors", {
        "files": [f1, f2, f3],
        "ref_idx": 0,
        "sensitivity_sigma": 3.0,
        "min_length_px": 15.0,
        "max_length_px": 2000.0,
        "filter_airplanes": True,
        "radiant_tolerance_deg": 10.0
    })

    scan_result = None
    def on_finished(res, task_type):
        nonlocal scan_result
        scan_result = res

    worker.finished.connect(on_finished)
    worker.error.connect(lambda err: print("WORKER ERROR:", err))
    t0 = time.perf_counter()
    worker.run()
    t1 = time.perf_counter()
    print(f"Parallel scan finished in {(t1 - t0)*1000:.1f} ms")

    assert scan_result is not None, "Scan worker did not return result"
    cands = scan_result["candidates"]
    print(f"Detected {len(cands)} candidates in parallel scan")
    assert len(cands) >= 1, "Expected at least 1 candidate detected from synthetic streak"
    print(f"Candidate #1: file={cands[0].filename}, length={cands[0].length_px:.1f}px, continuity={cands[0].line_continuity*100:.1f}%, peak_flux={cands[0].peak_flux:.3f}")
    assert cands[0].line_continuity >= 0.70, f"Expected high continuity, got {cands[0].line_continuity}"

    # Populate culling table in tab
    tab.candidates = cands
    tab.homographies_map = scan_result["homographies"]
    tab.radiant_point = scan_result["radiant_pt"]
    tab._refresh_cull_table()
    assert tab.table_cull.rowCount() == len(cands), "Culling table row count mismatch"
    print("Culling table populated successfully!")

    # 3. Test Fusion & View Toggle
    assert hasattr(tab, "btn_view_composite"), "btn_view_composite missing"
    assert not tab.btn_view_composite.isEnabled(), "btn_view_composite should be initially disabled"

    fuse_worker = MeteorWorker("fuse_meteors", {
        "base_img": base.astype(np.float32) / 65535.0,
        "candidates": tab.candidates,
        "homographies": tab.homographies_map,
        "boost_flux": 1.0,
        "include_sporadics": tab.chk_include_sporadics.isChecked(),
        "ground_mask": None
    })
    fuse_result = None
    fuse_worker.finished.connect(lambda res, t: setattr(tab, "dummy_fused", res))
    fuse_worker.run()
    assert hasattr(tab, "dummy_fused"), "Fusion failed"

    tab._on_worker_finished(tab.dummy_fused, "fuse_meteors")
    assert tab.btn_view_composite.isEnabled(), "btn_view_composite should be enabled after fusion"
    assert tab.btn_view_composite.isChecked(), "btn_view_composite should be checked after fusion"
    assert tab.composite_result is not None, "composite_result is None"
    assert "Compuesto final" in tab.lbl_view_status.text()
    print("Fusion finished and auto-activated composite view!")

    # Test that selecting row 0 keeps composite view in canvas
    tab.table_cull.selectRow(0)
    assert tab.btn_view_composite.isChecked(), "btn_view_composite should remain checked"
    assert tab.current_preview_img is tab.composite_result, "Canvas should keep showing composite result"
    print("Cull row selection preserved composite view without overwriting!")

    # Test toggling view to single frame and back
    tab.btn_view_composite.setChecked(False)
    assert "Toma" in tab.lbl_view_status.text(), "Status should show individual frame"
    tab.btn_view_composite.setChecked(True)
    assert "Compuesto final" in tab.lbl_view_status.text(), "Status should return to composite"
    print("View toggle between frame and composite verified!")

    # Test highlight all toggle
    assert hasattr(tab, "chk_highlight_all"), "chk_highlight_all missing"
    tab.chk_highlight_all.setChecked(True)
    assert len(tab.canvas.highlight_streaks_multi) == len(tab.candidates), "Multi reticles count mismatch"
    tab.chk_highlight_all.setChecked(False)
    assert len(tab.canvas.highlight_streaks_multi) == 0, "Multi reticles not cleared"
    print("Highlight all streaks toggle verified!")

    # 4. Test Airplane Beacon & Navigation Light Discrimination
    from core.meteors import detect_streaks_in_frame, refine_satellite_classifications
    h_p, w_p = 350, 350
    bg_p = np.zeros((h_p, w_p, 3), dtype=np.float32)
    bg_p[..., 0], bg_p[..., 1], bg_p[..., 2] = 0.035, 0.045, 0.075

    plane_frame = bg_p.copy()
    cv2.line(plane_frame, (100, 80), (140, 280), (0.035, 0.125, 0.075), 2)
    for py, px in zip(np.linspace(80, 280, 10), np.linspace(104, 144, 10)):
        cv2.circle(plane_frame, (int(round(px)), int(round(py))), 3, (0.135, 0.045, 0.075), -1)

    plane_cands = detect_streaks_in_frame(
        img_curr=plane_frame, img_bg=bg_p, filepath="airplane_test.tif", frame_idx=0,
        sensitivity_sigma=3.0, min_length_px=20.0
    )
    assert len(plane_cands) >= 1, "Airplane streak was not detected"
    assert plane_cands[0].is_airplane == True, "Airplane candidate was not marked as is_airplane"
    assert plane_cands[0].is_meteor == False, "Airplane candidate should be excluded from meteors"
    print("Airplane beacon discrimination verified!")

    # 5. Test Layered PSD Export
    from core.meteors import export_meteor_layered_psd
    psd_path = os.path.join(tmp_dir, "test_output.psd")
    psd_ok = export_meteor_layered_psd(
        output_path=psd_path,
        base_image=base.astype(np.float32) / 65535.0,
        candidates=tab.candidates,
        homographies_map=tab.homographies_map,
        boost_flux=1.0
    )
    assert psd_ok == True, "export_meteor_layered_psd returned False"
    assert os.path.exists(psd_path), "PSD output file does not exist"
    assert os.path.getsize(psd_path) > 1000, "PSD output file is empty or corrupted"
    print("Multilayer PSD export verified!")

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    run_test()
