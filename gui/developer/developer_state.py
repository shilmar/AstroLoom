"""Centralized parametric state for AstroLoom Developer / Editor suite."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class DeveloperState:
    """Dataclass holding all develop parameters for non-destructive rendering, undo/redo, and presets."""

    # Stretch parameters
    stretch_mode: str = "mtf"
    ghs_color_mode: str = "luma"
    ghs_sp_val: float = 0.02
    ghs_d_val: float = 5.0
    ghs_b_val: float = 2.0
    ghs_hp_val: float = 0.0
    ghs_lp_val: float = 0.0
    ghs_bp_val: float = 0.0
    bp_val: float = 0.0
    mtf_val: float = 0.15
    contrast_val: int = 0

    # Color & White Balance
    temp_val: int = 0
    tint_val: int = 0
    sat_sky_val: int = 100
    sat_gnd_val: int = 100
    vibrance_val: int = 0

    # Light Pollution / Gradients
    lp_val: int = 0
    lp_cx: float = 0.50
    lp_cy: float = 0.90
    lp_w: float = 0.60
    lp_h: float = 0.45
    lp_tint_idx: int = 0
    lp_custom_tint: Tuple[float, float, float] = (1.0, 0.68, 0.22)
    lp_tint_strength: int = 50
    lp_mode_idx: int = 0
    lp_legacy_1d: bool = False

    # Starless Adjustments
    stars_val: int = 100
    contrast_starless_val: int = 0
    clarity_val: int = 0
    dehaze_val: int = 0
    ond_est_val: int = 0
    ond_bg_val: int = 0
    denoise_val: int = 0
    denoise_color_val: int = 50
    denoise_mask_val: int = 50
    starless_is_stretched: bool = False

    # Orton Effect
    orton_strength: int = 0
    orton_radius: int = 40
    orton_threshold: int = 5
    orton_tint_idx: int = 0
    orton_mode_idx: int = 0

    # Star Diffusion
    star_diff_strength: int = 0
    star_diff_radius: int = 20
    star_diff_threshold: int = 45
    star_diff_color: int = 50
    star_diff_mode_idx: int = 0

    # Richardson-Lucy Deconvolution
    deconv_iterations: int = 0
    deconv_fwhm: int = 30
    deconv_deringing: int = 8
    deconv_damping: int = 25
    deconv_aspect: int = 100
    deconv_angle: int = 0
    deconv_model_idx: int = 0
    deconv_scope_idx: int = 0
    deconv_luma_only: bool = True

    # Ground Tone
    gnd_ev_val: int = 0
    gnd_sh_val: int = 0
    gnd_bp_val: int = 0

    # Layer & Canvas view mode
    layer_idx: int = 0

    # Complex sub-states (Curves & Mask Layers)
    curves: Optional[Dict[str, Any]] = None
    mask_layers: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state into a dictionary compatible with history snapshots and presets."""
        res = asdict(self)
        # Ensure tuple conversion for custom tint
        if isinstance(res.get("lp_custom_tint"), list):
            res["lp_custom_tint"] = tuple(res["lp_custom_tint"])
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeveloperState:
        """Create DeveloperState instance from dictionary."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "lp_custom_tint" in filtered and isinstance(filtered["lp_custom_tint"], list):
            filtered["lp_custom_tint"] = tuple(filtered["lp_custom_tint"])
        return cls(**filtered)

    def is_different_from(self, other: Optional[DeveloperState | Dict[str, Any]]) -> bool:
        """Check if this state differs significantly from another state or dict snapshot."""
        if other is None:
            return True
        if isinstance(other, DeveloperState):
            other_dict = other.to_dict()
        else:
            other_dict = other

        current_dict = self.to_dict()
        for k, v1 in current_dict.items():
            v2 = other_dict.get(k)
            if k in ("curves", "mask_layers"):
                if v1 != v2:
                    return True
            elif isinstance(v1, float) and v2 is not None:
                if abs(v1 - float(v2)) > 1e-4:
                    return True
            elif v1 != v2:
                return True
        return False
