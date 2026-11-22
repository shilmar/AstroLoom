# tests/test_developer_render_engine.py
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_render_engine_inheritance(qapp):
    from gui.tab_developer import DeveloperTab
    from gui.developer.render_engine import RenderMixin

    tab = DeveloperTab()
    assert isinstance(tab, RenderMixin)

def test_render_engine_proxy_generation_and_cache(qapp):
    from gui.tab_developer import DeveloperTab

    tab = DeveloperTab()
    dummy = np.full((120, 160, 3), 0.25, dtype=np.float32)
    tab.image_32bit = dummy

    tab._generate_preview_proxy()
    assert tab.preview_proxy is not None
    assert tab.draft_proxy is not None
    assert tab.pyramid_image is not None
    assert tab.preview_proxy.shape[2] == 3

    # Invalidate cache test
    tab._invalidate_pipeline_caches()
    assert tab._cached_rendered_image is None
    assert tab._last_rendered_preview is None

def test_render_engine_with_starless_and_mask(qapp):
    from gui.tab_developer import DeveloperTab

    tab = DeveloperTab()
    tab.image_32bit = np.full((100, 100, 3), 0.3, dtype=np.float32)
    tab.image_starless = np.full((100, 100, 3), 0.2, dtype=np.float32)
    tab.image_stars = np.full((100, 100, 3), 0.8, dtype=np.float32)
    tab.current_mask = np.full((100, 100), 1.0, dtype=np.float32)

    tab._generate_preview_proxy()
    assert tab.proxy_starless is not None
    assert tab.proxy_stars is not None
    assert tab.proxy_mask is not None
    assert tab.proxy_starless.shape[:2] == tab.preview_proxy.shape[:2]
    assert tab.proxy_mask.shape[:2] == tab.preview_proxy.shape[:2]

def test_render_engine_zoom_and_pan(qapp):
    from gui.tab_developer import DeveloperTab

    tab = DeveloperTab()
    tab.image_32bit = np.full((200, 200, 3), 0.5, dtype=np.float32)
    tab._generate_preview_proxy()

    # Toggle Zoom 1:1 on
    tab.on_canvas_zoom_toggled(True, 0.5, 0.5)
    assert tab.is_zoomed_100 is True

    # Pan movement
    tab.on_canvas_pan_started()
    tab.on_canvas_pan_moved(10, -10)
    tab.on_canvas_pan_finished()

    # Toggle Zoom off
    tab.on_canvas_zoom_toggled(False, 0.5, 0.5)
    assert tab.is_zoomed_100 is False

def test_render_engine_update_stretch_preview(qapp):
    from gui.tab_developer import DeveloperTab

    tab = DeveloperTab()
    tab.image_32bit = np.full((80, 80, 3), 0.4, dtype=np.float32)
    tab._generate_preview_proxy()

    # Call update_stretch_preview immediate
    tab.update_stretch_preview(immediate=True)
    assert tab._last_rendered_preview is not None
