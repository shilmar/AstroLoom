"""gui.developer.panels package - Individual collapsible tool panels."""
from gui.developer.panels.panel_color_wb import ColorWbPanel
from gui.developer.panels.panel_deconvolution import DeconvolutionPanel
from gui.developer.panels.panel_light_pollution import LightPollutionPanel
from gui.developer.panels.panel_stretching import StretchingPanel
from gui.developer.panels.panel_effects import EffectsPanel
from gui.developer.panels.panel_presets import PresetsPanel
from gui.developer.panels.panel_ground_sky import GroundSkyPanel
from gui.developer.panels.panel_starnet import StarNetPanel

__all__ = [
    "ColorWbPanel",
    "DeconvolutionPanel",
    "LightPollutionPanel",
    "StretchingPanel",
    "EffectsPanel",
    "PresetsPanel",
    "GroundSkyPanel",
    "StarNetPanel",
]
