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
