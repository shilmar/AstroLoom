import os
import sys
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PySide6.QtWidgets import QApplication, QMessageBox

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_ai_workers_starnet_success_and_error(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    # Create dummy base image
    dummy_base = np.full((100, 100, 3), 0.5, dtype=np.float32)
    tab.image_32bit = dummy_base

    starless_dummy = np.full((100, 100, 3), 0.2, dtype=np.float32)
    stars_dummy = np.full((100, 100, 3), 0.8, dtype=np.float32)

    tab.on_starnet_success(starless_dummy, stars_dummy)

    assert tab.image_starless is not None
    assert tab.image_stars is not None
    assert tab.image_starless.shape == (100, 100, 3)
    assert tab.image_stars.shape == (100, 100, 3)
    assert tab.combo_layer.isEnabled()
    assert tab.slider_stars.isEnabled()

    # Test error handling
    with patch.object(QMessageBox, "warning"):
        tab.on_starnet_error("Test mock error")
    # Verify no unhandled exception and tab remains responsive
    assert tab.image_32bit is not None

def test_ai_workers_bg_extraction_success(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    tab.image_32bit = np.full((50, 50, 3), 0.4, dtype=np.float32)
    corrected = np.full((50, 50, 3), 0.35, dtype=np.float32)

    tab.on_bg_extraction_success(corrected)
    assert np.allclose(tab.image_32bit, 0.35)

def test_ai_workers_deconv_psf(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    tab.deconv_fwhm = 3.0
    tab.deconv_model = "moffat"
    tab.deconv_beta = 4.5

    psf = tab._get_current_deconv_psf()
    assert psf is not None
    assert psf.ndim == 2
    assert psf.shape[0] > 0
    assert abs(np.sum(psf) - 1.0) < 1e-2

def test_ai_workers_sync_controls(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    tab.sync_controls(bp=0.0123, m=0.2345)
    assert abs(tab.panel_stretch.bp_val - 0.0123) < 1e-4
    assert abs(tab.panel_stretch.mtf_val - 0.2345) < 1e-4
