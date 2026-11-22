import os
import sys
import pytest
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_crop_preserves_and_crops_ai_denoise_caches(qapp):
    from gui.tab_developer import DeveloperTab

    tab = DeveloperTab()
    # Create fake 100x200 image
    h, w = 100, 200
    tab.image_32bit = np.ones((h, w, 3), dtype=np.float32) * 0.5
    tab.image_starless = np.ones((h, w, 3), dtype=np.float32) * 0.4
    tab.image_stars = np.ones((h, w, 3), dtype=np.float32) * 0.1
    tab.current_mask = np.ones((h, w), dtype=np.float32)

    # Fake NAFNet caches
    tab._cache_starless_ai = np.ones((h, w, 3), dtype=np.float32) * 0.45
    tab._cache_ground_ai = np.ones((h, w, 3), dtype=np.float32) * 0.55
    tab.starless_ai_denoise_enabled = True
    tab.denoise_strength = 1.0

    # Set crop rect: cut 20% off top and 30% off bottom (norm: x=0, y=0.2, w=1.0, h=0.5 -> new h=50, w=200)
    tab.canvas.crop_rect_norm = QRectF(0.0, 0.2, 1.0, 0.5)
    tab.canvas.crop_angle = 0.0

    # Apply crop
    tab._apply_crop()

    new_h, new_w = tab.image_32bit.shape[:2]
    assert (new_h, new_w) == (50, 200)
    assert tab.image_starless.shape[:2] == (50, 200)
    assert tab._cache_starless_ai is not None
    assert tab._cache_starless_ai.shape[:2] == (50, 200)
    assert tab._cache_ground_ai is not None
    assert tab._cache_ground_ai.shape[:2] == (50, 200)

    # Test _process_starless_base: aspect ratios match, no squishing
    proc = tab._process_starless_base(tab.image_starless, is_full_res=True)
    assert proc.shape[:2] == (50, 200)
    assert tab._cache_starless_ai is not None

    # Test undo
    tab.undo()
    assert tab.image_32bit.shape[:2] == (100, 200)
    assert tab._cache_starless_ai.shape[:2] == (100, 200)
    assert tab._cache_ground_ai.shape[:2] == (100, 200)

    # Test redo
    tab.redo()
    assert tab.image_32bit.shape[:2] == (50, 200)
    assert tab._cache_starless_ai.shape[:2] == (50, 200)
    assert tab._cache_ground_ai.shape[:2] == (50, 200)

def test_mismatched_cache_safeguard(qapp):
    from gui.tab_developer import DeveloperTab

    tab = DeveloperTab()
    h, w = 50, 200
    tab.image_32bit = np.ones((h, w, 3), dtype=np.float32) * 0.5
    tab.image_starless = np.ones((h, w, 3), dtype=np.float32) * 0.4
    # Outdated cache with completely different aspect ratio (100x100 instead of 50x200)
    tab._cache_starless_ai = np.ones((100, 100, 3), dtype=np.float32) * 0.45
    tab.starless_ai_denoise_enabled = True
    tab.denoise_strength = 1.0

    # _process_starless_base should detect aspect ratio mismatch and invalidate cache rather than squishing
    proc = tab._process_starless_base(tab.image_starless, is_full_res=True)
    assert tab._cache_starless_ai is None
    assert proc.shape[:2] == (50, 200)
