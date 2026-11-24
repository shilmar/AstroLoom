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

def test_crop_preserves_and_crops_raster_layers_and_masks(qapp):
    from gui.tab_developer import DeveloperTab
    from core.mask_layers import MASK_TYPE_GRADIENT_RADIAL

    tab = DeveloperTab()
    h, w = 100, 200
    tab.image_32bit = np.ones((h, w, 3), dtype=np.float32) * 0.5

    # 1. Capa ráster (simulando filtro coronal de eclipse)
    coronal_filter = np.ones((h, w, 3), dtype=np.float32) * 0.8
    tab.mask_layers_widget.add_raster_layer("Filtro Coronal MGN", coronal_filter, blend_mode="screen", opacity=0.5)
    raster_layer = tab.mask_layers_widget.layers[0]
    assert raster_layer.image_data.shape[:2] == (100, 200)

    # 2. Capa con máscara de pincel
    brush_layer = tab.mask_layers_widget.add_new_layer()
    brush_layer.name = "Máscara Pincel"
    brush_layer.set_brush_mask(np.ones((h, w), dtype=np.float32) * 0.9)

    # 3. Capa con gradiente radial
    grad_layer = tab.mask_layers_widget.add_new_layer(MASK_TYPE_GRADIENT_RADIAL)
    grad_layer.name = "Gradiente Radial"
    grad_layer.radial_center = (0.5, 0.5)
    grad_layer.radial_radius_x = 0.25
    grad_layer.radial_radius_y = 0.25

    # 4. Motas de polvo: una dentro (y=0.45), otra fuera (y=0.05)
    tab.canvas.set_dust_spots([
        {"x": 0.5, "y": 0.45, "radius": 15, "strength": 1.0},
        {"x": 0.5, "y": 0.05, "radius": 15, "strength": 1.0},
    ])

    # Marco de recorte: x=0, y=0.2, w=1.0, h=0.5 -> 50x200
    tab.canvas.crop_rect_norm = QRectF(0.0, 0.2, 1.0, 0.5)
    tab.canvas.crop_angle = 0.0

    # Aplicar recorte
    tab._apply_crop()

    # Verificar que la imagen base y las capas ráster se han recortado
    assert tab.image_32bit.shape[:2] == (50, 200)
    assert raster_layer.image_data.shape[:2] == (50, 200)
    assert brush_layer.brush_mask.shape[:2] == (50, 200)
    # En el recorte y=0.2 a 0.7 (altura 0.5), y=0.5 pasa a ser (0.5 - 0.2) / 0.5 = 0.6
    assert abs(grad_layer.radial_center[0] - 0.5) < 1e-4
    assert abs(grad_layer.radial_center[1] - 0.6) < 1e-4

    # Motas de polvo: la de y=0.05 se descarta, la de y=0.45 pasa a (0.45 - 0.2)/0.5 = 0.5
    spots = tab.canvas.get_dust_spots()
    assert len(spots) == 1
    assert abs(spots[0]["y"] - 0.5) < 1e-4

    # Deshacer (Undo)
    tab.undo()
    assert tab.image_32bit.shape[:2] == (100, 200)
    # Recuperar capas desde el widget
    r_undo = next(l for l in tab.mask_layers_widget.layers if l.name == "Filtro Coronal MGN")
    b_undo = next(l for l in tab.mask_layers_widget.layers if l.name == "Máscara Pincel")
    g_undo = next(l for l in tab.mask_layers_widget.layers if l.name == "Gradiente Radial")
    assert r_undo.image_data.shape[:2] == (100, 200)
    assert b_undo.brush_mask.shape[:2] == (100, 200)
    assert abs(g_undo.radial_center[1] - 0.5) < 1e-4
    assert len(tab.canvas.get_dust_spots()) == 2

    # Rehacer (Redo)
    tab.redo()
    assert tab.image_32bit.shape[:2] == (50, 200)
    r_redo = next(l for l in tab.mask_layers_widget.layers if l.name == "Filtro Coronal MGN")
    b_redo = next(l for l in tab.mask_layers_widget.layers if l.name == "Máscara Pincel")
    g_redo = next(l for l in tab.mask_layers_widget.layers if l.name == "Gradiente Radial")
    assert r_redo.image_data.shape[:2] == (50, 200)
    assert b_redo.brush_mask.shape[:2] == (50, 200)
    assert abs(g_redo.radial_center[1] - 0.6) < 1e-4
    assert len(tab.canvas.get_dust_spots()) == 1

