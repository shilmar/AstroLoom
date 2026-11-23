import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PySide6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_layer_sync_stars_intensity(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    # Test sync from MaskLayersWidget -> DeveloperTab slider
    tab._on_mask_widget_stars_intensity(75)
    assert tab.slider_stars.value() == 75
    assert tab.star_intensity == 0.75

    # Test setting slider directly triggers on_stars_slider_changed
    tab.slider_stars.setValue(40)
    assert tab.star_intensity == 0.40
    assert tab.slider_stars.value() == 40

def test_layer_sync_starless_tone(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    tab._on_mask_widget_sl_clarity(25)
    assert tab.slider_clarity.value() == 25

    tab._on_mask_widget_sl_dehaze(15)
    assert tab.slider_dehaze.value() == 15

    tab._on_mask_widget_sl_contrast(30)
    assert tab.slider_contrast_starless.value() == 30

def test_layer_sync_ground_tone(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    tab._on_mask_widget_gnd_ev(50)
    assert tab.slider_gnd_ev.value() == 50

    tab._on_mask_widget_gnd_sh(40)
    assert tab.slider_gnd_sh.value() == 40

    tab._on_mask_widget_gnd_bp(20)
    assert tab.slider_gnd_bp.value() == 20

    tab.reset_gnd_tone()
    assert tab.slider_gnd_ev.value() == 0
    assert tab.slider_gnd_sh.value() == 0
    assert tab.slider_gnd_bp.value() == 0

def test_layer_sync_starnet_visibility(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    # Visibility false -> combo_layer index 1 (solo fondo)
    tab._on_mask_layer_starnet_visibility(False)
    assert tab.view_layer_mode == 1
    assert tab.combo_layer.currentIndex() == 1

    # Visibility true -> combo_layer index 0 (compuesta)
    tab._on_mask_layer_starnet_visibility(True)
    assert tab.view_layer_mode == 0
    assert tab.combo_layer.currentIndex() == 0

def test_layer_sync_reset_stars(qapp):
    from gui.tab_developer import DeveloperTab
    tab = DeveloperTab()

    tab.slider_stars.setValue(35)
    assert tab.slider_stars.value() == 35

    tab.reset_stars_intensity()
    assert tab.slider_stars.value() == 100
    assert tab.star_intensity == 1.0


def test_ensure_starnet_layers_stack_order(qapp):
    from gui.widgets.mask_layers_widget import MaskLayersWidget
    from core.mask_layers import (
        MASK_TYPE_IMAGE_STARS, MASK_TYPE_IMAGE_GROUND,
        MASK_TYPE_IMAGE_STARLESS, MASK_TYPE_IMAGE_BASE, MASK_TYPE_BRUSH
    )

    w = MaskLayersWidget()
    # 1. Base layer added first
    w.ensure_base_layer()
    assert len(w.layers) == 1
    assert w.layers[0].mask_type == MASK_TYPE_IMAGE_BASE

    # 2. Add user adjustment layer
    brush_layer = w.add_new_layer(MASK_TYPE_BRUSH)
    assert len(w.layers) == 2
    assert w.layers[0].id == brush_layer.id
    assert w.layers[1].mask_type == MASK_TYPE_IMAGE_BASE

    # 3. Trigger StarNet separation with ground
    w.ensure_starnet_layers(has_ground=True)
    # Order must be: [Brush (Top), Stars, Ground, Starless (Bottom)]
    assert len(w.layers) == 4
    assert w.layers[0].mask_type == MASK_TYPE_BRUSH
    assert w.layers[1].mask_type == MASK_TYPE_IMAGE_STARS
    assert w.layers[2].mask_type == MASK_TYPE_IMAGE_GROUND
    assert w.layers[3].mask_type == MASK_TYPE_IMAGE_STARLESS

    # 4. Remove ground
    w.ensure_starnet_layers(has_ground=False)
    assert len(w.layers) == 3
    assert w.layers[0].mask_type == MASK_TYPE_BRUSH
    assert w.layers[1].mask_type == MASK_TYPE_IMAGE_STARS
    assert w.layers[2].mask_type == MASK_TYPE_IMAGE_STARLESS


def test_layer_curve_isolation_between_layers(qapp):
    import numpy as np
    from gui.widgets.mask_layers_widget import MaskLayersWidget

    w = MaskLayersWidget()
    w.ensure_starnet_layers(has_ground=False)
    # layers: [Stars (0), Starless (1)]
    assert len(w.layers) == 2
    stars_l = w.layers[0]
    starless_l = w.layers[1]

    # Select Stars
    w.select_layer_by_id(stars_l.id)
    assert w.get_selected_layer().id == stars_l.id
    curve_w = w.grp_local_adj.layer_curve_widget

    # Apply custom curve on Stars
    custom_pts = [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)]
    curve_w.set_state({"rgb": custom_pts})
    w.grp_local_adj._on_layer_curve_changed()

    assert stars_l.curve_points == custom_pts
    assert not stars_l.is_curve_identity()
    assert stars_l.curve_lut is not None

    # Switch to Starless (which has neutral curve)
    w.select_layer_by_id(starless_l.id)
    assert w.get_selected_layer().id == starless_l.id
    # Starless must remain identity and not inherit Stars' curve
    assert starless_l.is_curve_identity()
    pts = curve_w.points
    assert len(pts) == 2
    assert pts[0] == (0.0, 0.0)
    assert pts[1] == (1.0, 1.0)

    # Switch back to Stars
    w.select_layer_by_id(stars_l.id)
    assert w.get_selected_layer().id == stars_l.id
    assert curve_w.points == custom_pts

