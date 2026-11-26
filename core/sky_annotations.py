# core/sky_annotations.py
"""
core/sky_annotations.py - Proyección y renderizado de constelaciones y estrellas nombradas.

Proporciona:
1. Carga en caché de `core/data/sky_annotations.json` (88 constelaciones IAU + 325 estrellas).
2. Proyección geodésica subdividida desde ICRF a píxeles de imagen (rectilínea o proyector genérico).
3. Renderizado de capa RGBA transparente con líneas de constelaciones y etiquetas sin solapamientos.
"""
from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _data_file_path() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        cand = os.path.join(sys._MEIPASS, "core", "data", "sky_annotations.json")
        if os.path.isfile(cand):
            return cand
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "sky_annotations.json")
    )


@lru_cache(maxsize=1)
def load_sky_annotations() -> Dict[str, Any]:
    """Carga y cachea el catálogo ligero de constelaciones y estrellas nombradas."""
    path = _data_file_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def radec_to_icrf(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Convierte (RA, Dec) en grados a vector unitario 3D en ICRF."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    c = math.cos(dec)
    return np.array([c * math.cos(ra), c * math.sin(ra), math.sin(dec)], dtype=np.float64)


def project_icrf_to_rectilinear(
    u: np.ndarray,
    r_cam_to_icrf: np.ndarray,
    width: int,
    height: int,
    focal_px: float,
    k1: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Proyecta un vector unitario ICRF a coordenadas de píxel en una toma rectilínea."""
    d_cam = r_cam_to_icrf.T @ u
    dz = float(d_cam[2])
    if dz <= 0.20:
        return None
    tx = float(d_cam[0]) / dz
    ty = float(d_cam[1]) / dz
    r2 = tx * tx + ty * ty
    # Evitar inversión del polinomio radial (fold-back cuando k1 < 0) fuera del FOV real del sensor
    max_r2 = (((0.5 * width) / focal_px) ** 2 + ((0.5 * height) / focal_px) ** 2) * 1.35
    if r2 > max_r2 or (k1 < 0.0 and (1.0 + 3.0 * k1 * r2) <= 0.25):
        return None
    dist = 1.0 + k1 * r2
    px = width * 0.5 + focal_px * tx * dist
    py = height * 0.5 + focal_px * ty * dist
    if px < -width * 0.25 or px > width * 1.25 or py < -height * 0.25 or py > height * 1.25:
        return None
    return (px, py)


def _is_sky_pixel(
    px: float,
    py: float,
    width: int,
    height: int,
    sky_mask: Optional[np.ndarray],
) -> bool:
    if not (0.0 <= px < width and 0.0 <= py < height):
        return False
    if sky_mask is None:
        return True
    mh, mw = sky_mask.shape[:2]
    ix = min(mw - 1, max(0, int(round(px * mw / max(1, width)))))
    iy = min(mh - 1, max(0, int(round(py * mh / max(1, height)))))
    return float(sky_mask[iy, ix]) > 0.5


def project_sky_annotations(
    projector: Callable[[np.ndarray], Optional[Tuple[float, float]]],
    width: int,
    height: int,
    sky_mask: Optional[np.ndarray] = None,
    max_star_mag: float = 4.0,
    max_dso_mag: float = 9.5,
    show_constellations: bool = True,
    show_constellation_names: bool = True,
    show_star_names: bool = True,
    show_dso: bool = True,
    subdivide_steps: int = 6,
) -> Tuple[List[List[Tuple[float, float]]], List[Dict[str, Any]]]:
    """
    Proyecta constelaciones y estrellas nombradas usando cualquier función `projector(u_icrf) -> (x, y)`.
    Subdivide aristas sobre la esfera celeste para respetar distorsión radial y proyecciones curvas.
    """
    data = load_sky_annotations()
    hip_coords = data["hip_coords"]

    polylines: List[List[Tuple[float, float]]] = []
    labels: List[Dict[str, Any]] = []

    if show_constellations or show_constellation_names:
        for con in data["constellations"]:
            if show_constellations:
                for hip_a, hip_b in con["pairs"]:
                    ca = hip_coords.get(str(hip_a))
                    cb = hip_coords.get(str(hip_b))
                    if ca is None or cb is None:
                        continue
                    ua = radec_to_icrf(ca[0], ca[1])
                    ub = radec_to_icrf(cb[0], cb[1])
                    pts: List[Tuple[float, float]] = []
                    steps = max(1, int(subdivide_steps))
                    for s in range(steps + 1):
                        t = s / float(steps)
                        u_mid = (1.0 - t) * ua + t * ub
                        nrm = float(np.linalg.norm(u_mid))
                        if nrm < 1e-9:
                            continue
                        u_mid /= nrm
                        pt = projector(u_mid)
                        if pt is not None and _is_sky_pixel(pt[0], pt[1], width, height, sky_mask):
                            pts.append(pt)
                        else:
                            if len(pts) >= 2:
                                polylines.append(pts)
                            pts = []
                    if len(pts) >= 2:
                        polylines.append(pts)

            if show_constellation_names and "center" in con:
                uc = radec_to_icrf(con["center"][0], con["center"][1])
                pt_c = projector(uc)
                if pt_c is not None and _is_sky_pixel(pt_c[0], pt_c[1], width, height, sky_mask):
                    labels.append({
                        "x": pt_c[0],
                        "y": pt_c[1],
                        "text": con["name"].upper(),
                        "kind": "constellation",
                        "mag": 0.0,
                    })

    if show_star_names:
        for ra_d, dec_d, mag, name, _hip in data["named_stars"]:
            if mag > max_star_mag:
                break
            us = radec_to_icrf(ra_d, dec_d)
            pt_s = projector(us)
            if pt_s is not None and _is_sky_pixel(pt_s[0], pt_s[1], width, height, sky_mask):
                labels.append({
                    "x": pt_s[0],
                    "y": pt_s[1],
                    "text": name,
                    "kind": "star",
                    "mag": float(mag),
                })

    if show_dso:
        for ra_d, dec_d, mag, name in data.get("dso", []):
            if mag > max_dso_mag:
                break
            ud = radec_to_icrf(ra_d, dec_d)
            pt_d = projector(ud)
            if pt_d is not None and _is_sky_pixel(pt_d[0], pt_d[1], width, height, sky_mask):
                labels.append({
                    "x": pt_d[0],
                    "y": pt_d[1],
                    "text": name,
                    "kind": "dso",
                    "mag": float(mag),
                })

    return polylines, labels


def build_rectilinear_annotations(
    sol: Dict[str, Any],
    width: int,
    height: int,
    sky_mask: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Tuple[List[List[Tuple[float, float]]], List[Dict[str, Any]]]:
    """Proyecta constelaciones y estrellas usando la solución rectilínea de `solve_frame()`."""
    r_icrf = np.asarray(sol["rotation_icrf"], dtype=np.float64)
    focal_px = float(sol["focal_px"])
    k1 = float(sol.get("k1", 0.0))

    def _proj(u: np.ndarray) -> Optional[Tuple[float, float]]:
        return project_icrf_to_rectilinear(u, r_icrf, width, height, focal_px, k1)

    return project_sky_annotations(_proj, width, height, sky_mask=sky_mask, **kwargs)


def _load_ui_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fname in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(fname, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def render_sky_annotations_rgba(
    width: int,
    height: int,
    polylines: List[List[Tuple[float, float]]],
    labels: List[Dict[str, Any]],
    line_color: Tuple[int, int, int, int] = (90, 180, 255, 170),
    star_color: Tuple[int, int, int, int] = (255, 230, 150, 235),
    constellation_color: Tuple[int, int, int, int] = (125, 205, 255, 225),
    dso_color: Tuple[int, int, int, int] = (155, 240, 195, 225),
) -> np.ndarray:
    """Renderiza líneas y etiquetas sobre un lienzo RGBA uint8 transparente (sin solapamientos)."""
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    ref_dim = max(width, height)
    line_w = max(1, int(round(ref_dim / 950.0)))
    star_font_sz = max(12, int(round(ref_dim / 95.0)))
    con_font_sz = max(14, int(round(ref_dim / 75.0)))

    font_star = _load_ui_font(star_font_sz)
    font_con = _load_ui_font(con_font_sz)

    for pts in polylines:
        if len(pts) >= 2:
            draw.line(pts, fill=line_color, width=line_w, joint="curve")

    # Priorizar constelaciones, estrellas brillantes y objetos Messier/NGC para evitar solapamientos
    kind_prio = {"constellation": 0, "star": 1, "dso": 2}
    ordered = sorted(
        labels,
        key=lambda l: (kind_prio.get(l["kind"], 3), l.get("mag", 9.9)),
    )
    placed_boxes: List[Tuple[float, float, float, float]] = []
    marker_r = max(3, int(round(ref_dim / 320.0)))

    for lbl in ordered:
        kind = lbl["kind"]
        is_con = kind == "constellation"
        is_dso = kind == "dso"
        font = font_con if is_con else font_star
        color = constellation_color if is_con else (dso_color if is_dso else star_color)
        text = lbl["text"]
        x, y = float(lbl["x"]), float(lbl["y"])

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x - tw * 0.5 if is_con else x + marker_r + 4.0
        ty = y - th * 0.5
        pad = 4.0
        box = (tx - pad, ty - pad, tx + tw + pad, ty + th + pad)

        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            continue
        if any(
            not (box[2] < pb[0] or box[0] > pb[2] or box[3] < pb[1] or box[1] > pb[3])
            for pb in placed_boxes
        ):
            continue
        placed_boxes.append(box)

        if is_dso:
            dr = marker_r * 1.4
            draw.polygon(
                [(x, y - dr), (x + dr, y), (x, y + dr), (x - dr, y)],
                outline=color,
                width=max(1, line_w),
            )
        elif not is_con:
            draw.ellipse(
                (x - marker_r, y - marker_r, x + marker_r, y + marker_r),
                outline=color,
                width=max(1, line_w),
            )
        # Sombra sutil para legibilidad sobre Vía Láctea brillante
        draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0, 190))
        draw.text((tx, ty), text, font=font, fill=color)

    return np.asarray(canvas, dtype=np.uint8)


def generate_sky_annotation_layer_data(
    image: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    filepath: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Resuelve la astrometría de `image` (rectilínea) y genera `(rgb_f32, alpha_f32, sol)`
    listos para insertar como capa `MASK_TYPE_IMAGE_RASTER` en el Revelador.
    """
    import cv2
    from core.plate_solver import get_best_available_catalog, get_catalog_path, solve_frame

    if image is None or image.ndim < 2:
        return None
    full_h, full_w = image.shape[:2]
    max_dim = max(full_w, full_h)
    scale_work = min(1.0, 1400.0 / float(max(1, max_dim)))
    if scale_work < 0.99:
        ww = max(64, int(round(full_w * scale_work)))
        wh = max(64, int(round(full_h * scale_work)))
        work_img = cv2.resize(image, (ww, wh), interpolation=cv2.INTER_AREA)
        work_mask = cv2.resize(sky_mask, (ww, wh), interpolation=cv2.INTER_LINEAR) if sky_mask is not None else None
        scale_to_full = full_w / float(ww)
    else:
        work_img = image
        work_mask = sky_mask
        scale_to_full = 1.0

    focal_px_hint = None
    if filepath and os.path.isfile(filepath):
        try:
            from core.panorama import extract_panel_metadata
            meta = extract_panel_metadata(filepath, 0)
            if meta.focal_length_mm and meta.focal_length_mm > 1.0:
                focal_px_hint = (float(meta.focal_length_mm) * (max_dim / 36.0)) / scale_to_full
        except Exception:
            pass

    sol = None
    tried = []
    for prof in ("lite", "standard", "deep"):
        cat = get_catalog_path(prof)
        if not cat or cat in tried:
            continue
        tried.append(cat)
        sol = solve_frame(
            work_img,
            sky_mask=work_mask,
            focal_px_hint=focal_px_hint,
            catalog_path=cat,
            scale_to_full=scale_to_full,
        )
        if sol is not None:
            break

    if sol is None:
        cat = get_best_available_catalog("lite")
        if cat and cat not in tried:
            sol = solve_frame(
                work_img,
                sky_mask=work_mask,
                focal_px_hint=focal_px_hint,
                catalog_path=cat,
                scale_to_full=scale_to_full,
            )

    if sol is None:
        return None

    polylines, labels = build_rectilinear_annotations(sol, full_w, full_h, sky_mask=sky_mask, **kwargs)
    rgba = render_sky_annotations_rgba(full_w, full_h, polylines, labels)
    rgb_f32 = np.ascontiguousarray(rgba[..., :3].astype(np.float32) * (1.0 / 255.0))
    alpha_f32 = np.ascontiguousarray(rgba[..., 3].astype(np.float32) * (1.0 / 255.0))
    return rgb_f32, alpha_f32, sol
