# gui/widgets/mask_layers/__init__.py
"""
Paquete modular de sub-widgets y utilidades para la gestión de capas de máscaras (MaskLayers).
"""
from gui.widgets.mask_layers.layer_thumbnails import (
    THUMB_SPECS,
    LayerListResizeHandle,
    get_image_aspect_ratio,
    get_thumbnail_dimensions,
    get_layer_thumbnail_pixmap,
    get_raster_mask_thumbnail_pixmap,
    invalidate_layer_thumbnail_cache,
)
from gui.widgets.mask_layers.mask_definition_panels import MaskDefinitionWidget
from gui.widgets.mask_layers.special_layers_panels import (
    RasterTransformPanel,
    GroundSpecialPanel,
    StarsSpecialPanel,
    StarlessSpecialPanel,
)
from gui.widgets.mask_layers.local_adjustments_panel import LocalAdjustmentsPanel
from gui.widgets.mask_layers.common import make_resettable_label, make_box_qss

__all__ = [
    "THUMB_SPECS",
    "LayerListResizeHandle",
    "get_image_aspect_ratio",
    "get_thumbnail_dimensions",
    "get_layer_thumbnail_pixmap",
    "get_raster_mask_thumbnail_pixmap",
    "invalidate_layer_thumbnail_cache",
    "MaskDefinitionWidget",
    "RasterTransformPanel",
    "GroundSpecialPanel",
    "StarsSpecialPanel",
    "StarlessSpecialPanel",
    "LocalAdjustmentsPanel",
    "make_resettable_label",
    "make_box_qss",
]
