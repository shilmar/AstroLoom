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


def test_phase1_zoom_crop_and_draft_cache(qapp):
    from gui.tab_developer import DeveloperTab
    from core.mask_layers import MaskLayer, apply_all_mask_layers, clear_subcomposite_cache, MASK_TYPE_LUMA, MASK_TYPE_IMAGE_BASE

    tab = DeveloperTab()
    tab.image_32bit = np.full((300, 300, 3), 0.4, dtype=np.float32)
    tab.image_starless = np.full((300, 300, 3), 0.35, dtype=np.float32)
    tab.image_stars = np.full((300, 300, 3), 0.1, dtype=np.float32)
    tab.current_mask = np.ones((300, 300), dtype=np.float32)
    tab.clarity_starless_val = 0.2
    tab._generate_preview_proxy()

    # 1. Zoom 1:1 reuses crop buffer identity and caches _process_starless_base
    tab.mask_layers_widget.layers = [
        MaskLayer(name="Estrellas", layer_kind="image", mask_type="image_stars"),
        MaskLayer(name="Starless", layer_kind="image", mask_type="image_starless"),
    ]
    tab.on_canvas_zoom_toggled(True, 0.5, 0.5)
    b1, m1 = tab._compose_active_base(for_export=False)
    b2, m2 = tab._compose_active_base(for_export=False)
    assert id(b1) == id(b2)
    assert id(m1) == id(m2)

    # 2. Draft scrubbing populates buffer_cache under _draft without polluting full-quality slots
    tab.on_canvas_zoom_toggled(False, 0.5, 0.5)
    tab._is_scrubbing_heavy_slider = True
    base_d, mask_d = tab._compose_active_base(for_export=False)
    out_d = tab._apply_pipeline_on_image(base_d, precomputed_mask=mask_d, is_preview=True, asynchronous=False)
    assert out_d is not None
    assert any(k[0] == "spatial_draft" for k in tab.buffer_cache._cache.keys())
    tab._is_scrubbing_heavy_slider = False

    # 3. Subcomposite cache works with crop_rect and invalidates on luma_min change
    clear_subcomposite_cache()
    base = np.linspace(0.0, 1.0, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
    l_bot = MaskLayer(name="Base", layer_kind="image", mask_type=MASK_TYPE_IMAGE_BASE)
    l_mid = MaskLayer(name="Luma", mask_type=MASK_TYPE_LUMA, luma_min=0.2, luma_max=0.8, exposure=0.5)
    l_top = MaskLayer(name="Top", mask_type=MASK_TYPE_LUMA, luma_min=0.5, luma_max=1.0, highlights=-0.2)
    r1 = apply_all_mask_layers(base, [l_top, l_mid, l_bot], crop_rect=(0, 0, 64, 64), full_image_shape=(128, 128), fast_draft=False)
    l_mid.luma_min = 0.6
    r2 = apply_all_mask_layers(base, [l_top, l_mid, l_bot], crop_rect=(0, 0, 64, 64), full_image_shape=(128, 128), fast_draft=True)
    assert not np.allclose(r1, r2)

    # 4. Split View 'Antes' uses raw image_32bit/preview_proxy, never stretched starless
    tab.image_32bit = np.full((300, 300, 3), 0.1, dtype=np.float32)
    tab.image_starless = np.full((300, 300, 3), 0.9, dtype=np.float32)
    tab.starless_is_stretched = True
    tab._generate_preview_proxy()
    before_fit = tab._compute_before_image()
    assert before_fit is not None and float(before_fit.mean()) < 0.7


def test_phase3_refined_mask_cache_and_roi_spatial(qapp):
    from core.mask_layers import (
        MaskLayer, compute_layer_mask, apply_local_adjustments_to_image,
        clear_subcomposite_cache, _REFINED_MASK_CACHE, MASK_TYPE_BRUSH
    )
    clear_subcomposite_cache()
    img = np.random.default_rng(42).uniform(0.1, 0.9, (256, 256, 3)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[80:140, 80:140] = 1.0  # ~5.5% ROI

    # 1. Refined mask cache hit
    l = MaskLayer(name="Refined", mask_type=MASK_TYPE_BRUSH, refine_edges=True, refine_strength=0.6)
    l.set_brush_mask(mask)
    m1 = compute_layer_mask(l, img)
    assert len(_REFINED_MASK_CACHE) == 1
    m2 = compute_layer_mask(l, img)
    assert np.array_equal(m1, m2)

    # 2. ROI spatial filter modifies inside ROI and leaves outside untouched
    l_fx = MaskLayer(name="LocalFX", mask_type=MASK_TYPE_BRUSH, clarity=0.6, texture_clarity=0.4, sharp_amount=60.0)
    out = apply_local_adjustments_to_image(img, mask, l_fx)
    assert np.array_equal(out[:20, :20], img[:20, :20])
    assert not np.allclose(out[90:130, 90:130], img[90:130, 90:130])



