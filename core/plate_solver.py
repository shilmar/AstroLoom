# core/plate_solver.py
"""
core/plate_solver.py - Plate Solving local offline (Quads 4D + catálogo ALCAT001).

Proporciona:
1. Gestión y descarga verificada (SHA-256) de catálogos estelares (lite/standard/deep).
2. Detección subpíxel en float32 con máscara de cielo y resolución astrométrica en astroloom_rs.
3. Generación de pares de anclaje celestes (`StarAnchor`) y rotaciones relativas `SO(3)` entre paneles.
4. Caché en memoria por archivo/parámetros para reutilizar soluciones entre borrador y alta resolución.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import sys
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import astroloom_rs
    HAS_RUST_PLATE_SOLVER = hasattr(astroloom_rs, "plate_solve_stars_rs")
except ImportError:
    HAS_RUST_PLATE_SOLVER = False


CATALOG_PROFILES: Dict[str, Dict[str, Any]] = {
    "lite": {
        "id": "lite",
        "name": "Lite (Hipparcos)",
        "recommended_fov_mm": "14-24 mm (FF)",
        "mag_limit": 8.5,
        "filename": "alcat_lite_hip_v1.bin",
        "url": "https://github.com/shilmar/AstroLoom/releases/download/catalogs-v1/alcat_lite_hip_v1.bin.gz",
        "sha256": "",
    },
    "standard": {
        "id": "standard",
        "name": "Standard (Tycho-2)",
        "recommended_fov_mm": "20-50 mm (FF)",
        "mag_limit": 11.5,
        "filename": "alcat_std_tycho2_v1.bin",
        "url": "https://github.com/shilmar/AstroLoom/releases/download/catalogs-v1/alcat_std_tycho2_v1.bin.gz",
        "sha256": "",
    },
    "deep": {
        "id": "deep",
        "name": "Deep (Gaia DR3)",
        "recommended_fov_mm": "50-135 mm (FF)",
        "mag_limit": 13.5,
        "filename": "alcat_deep_gaia_v1.bin",
        "url": "https://github.com/shilmar/AstroLoom/releases/download/catalogs-v1/alcat_deep_gaia_v1.bin.gz",
        "sha256": "",
    },
}

# Caché en memoria: (filepath_abs, mtime, catalog_path, focal_hint_rounded) -> solution dict
_SOLVE_CACHE: Dict[Tuple[str, float, str, int], Optional[Dict[str, Any]]] = {}


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_catalogs_dir() -> str:
    return os.path.join(_app_root(), "models", "catalogs")


def get_catalog_path(profile: str = "lite") -> Optional[str]:
    """Localiza el archivo binario ALCAT001 del perfil solicitado (o el mejor disponible)."""
    info = CATALOG_PROFILES.get(profile)
    if info is None:
        return None
    cand = os.path.join(get_catalogs_dir(), info["filename"])
    if os.path.isfile(cand):
        return os.path.abspath(cand)
    return None


def get_best_available_catalog(preferred: str = "lite") -> Optional[str]:
    """Devuelve el catálogo preferido si existe, o cualquier otro perfil instalado."""
    order = [preferred] + [p for p in ("standard", "lite", "deep") if p != preferred]
    for prof in order:
        path = get_catalog_path(prof)
        if path:
            return path
    return None


def download_catalog(
    profile: str = "lite",
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    """Descarga y verifica (SHA-256) un perfil de catálogo ALCAT001 en models/catalogs/."""
    if profile not in CATALOG_PROFILES:
        raise ValueError(f"Perfil de catálogo desconocido: {profile}")
    info = CATALOG_PROFILES[profile]
    cat_dir = get_catalogs_dir()
    os.makedirs(cat_dir, exist_ok=True)
    target_bin = os.path.join(cat_dir, info["filename"])
    tmp_dl = target_bin + ".dl.tmp"
    tmp_bin = target_bin + ".bin.tmp"

    req = urllib.request.Request(
        info["url"],
        headers={"User-Agent": "AstroLoom/1.0 CatalogDownloader"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            total_mb = total / (1024 * 1024) if total > 0 else 0.0
            downloaded = 0
            with open(tmp_dl, "wb") as out:
                while True:
                    if cancel_check and cancel_check():
                        raise InterruptedError("Descarga de catálogo cancelada por el usuario.")
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        pct = int((downloaded / total) * 100) if total > 0 else 0
                        progress_callback(pct, downloaded / (1024 * 1024), total_mb)

        # Descomprimir si es .gz o mover directamente si ya es .bin
        if info["url"].endswith(".gz"):
            with gzip.open(tmp_dl, "rb") as gz_in, open(tmp_bin, "wb") as raw_out:
                while True:
                    block = gz_in.read(1024 * 1024)
                    if not block:
                        break
                    raw_out.write(block)
            os.remove(tmp_dl)
        else:
            os.replace(tmp_dl, tmp_bin)

        expected_sha = info.get("sha256", "").strip().lower()
        if expected_sha and expected_sha != "0" * 64:
            h = hashlib.sha256()
            with open(tmp_bin, "rb") as f:
                for blk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(blk)
            got = h.hexdigest().lower()
            if got != expected_sha:
                raise ValueError(f"Checksum SHA-256 inválido para {profile}: esperado {expected_sha}, recibido {got}")

        os.replace(tmp_bin, target_bin)
        return target_bin
    finally:
        for p in (tmp_dl, tmp_bin):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def clear_solve_cache() -> None:
    _SOLVE_CACHE.clear()


def _to_linear_green_f32(img: np.ndarray) -> np.ndarray:
    """Extrae canal verde (o luminancia) en float32 [0, 1] contiguo para detección estelar."""
    if img.ndim == 3:
        g = img[..., 1] if img.shape[2] >= 2 else img[..., 0]
    else:
        g = img
    if g.dtype == np.uint8:
        return np.ascontiguousarray(g.astype(np.float32) * (1.0 / 255.0))
    if g.dtype == np.uint16:
        return np.ascontiguousarray(g.astype(np.float32) * (1.0 / 65535.0))
    return np.ascontiguousarray(g, dtype=np.float32)


def solve_frame(
    image: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    focal_px_hint: Optional[float] = None,
    catalog_path: Optional[str] = None,
    scale_to_full: float = 1.0,
    max_stars: int = 180,
    min_separation: int = 8,
    filepath: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Ejecuta plate solving local offline sobre un frame (lineal float32 o uint8).
    Si `scale_to_full != 1.0`, escala coordenadas (x, y) y `focal_px` a resolución nativa.
    """
    if not HAS_RUST_PLATE_SOLVER:
        return None

    cat = catalog_path or get_best_available_catalog("lite")
    if not cat or not os.path.isfile(cat):
        return None

    cache_key = None
    if filepath and os.path.isfile(filepath):
        try:
            mtime = os.path.getmtime(filepath)
            f_key = int(round((focal_px_hint or 0.0) * scale_to_full))
            cache_key = (os.path.abspath(filepath), mtime, os.path.abspath(cat), f_key)
            if cache_key in _SOLVE_CACHE:
                return _SOLVE_CACHE[cache_key]
        except OSError:
            cache_key = None

    gray = _to_linear_green_f32(image)
    h, w = gray.shape[:2]

    mask_f32 = None
    if sky_mask is not None:
        m = np.ascontiguousarray(sky_mask, dtype=np.float32)
        if m.shape[:2] == (h, w):
            mask_f32 = m

    stars = astroloom_rs.detect_plate_stars_rs(
        gray, mask_f32, int(max_stars), int(min_separation), 4.5
    )
    if len(stars) < 8:
        if cache_key is not None:
            _SOLVE_CACHE[cache_key] = None
        return None

    sol = astroloom_rs.plate_solve_stars_rs(
        stars,
        int(w),
        int(h),
        cat,
        float(focal_px_hint) if (focal_px_hint is not None and focal_px_hint > 10.0) else None,
        0.40,
        0.025,
    )

    if sol is not None and abs(scale_to_full - 1.0) > 1e-4:
        sol = dict(sol)
        sol["focal_px"] = float(sol["focal_px"]) * scale_to_full
        sol["scale_arcsec_px"] = float(sol["scale_arcsec_px"]) / scale_to_full
        sol["rms_px"] = float(sol["rms_px"]) * scale_to_full
        scaled_stars = []
        for s in sol["stars"]:
            sc = dict(s)
            sc["x"] = float(s["x"]) * scale_to_full
            sc["y"] = float(s["y"]) * scale_to_full
            sc["residual_px"] = float(s["residual_px"]) * scale_to_full
            scaled_stars.append(sc)
        sol["stars"] = scaled_stars

    if cache_key is not None:
        _SOLVE_CACHE[cache_key] = sol
    return sol


def build_star_anchors_between_frames(
    sol_a: Dict[str, Any],
    sol_b: Dict[str, Any],
    frame_a: str,
    frame_b: str,
) -> Dict[str, Any]:
    """
    Cruza `star_id` entre dos frames resueltos y emite el contrato `StarAnchor`
    más la rotación relativa `R_b_to_a` en SO(3) (ray_a ≈ R_b_to_a @ ray_b).
    """
    map_a = {int(s["star_id"]): s for s in sol_a.get("stars", [])}
    anchors: List[Dict[str, Any]] = []

    for sb in sol_b.get("stars", []):
        sid = int(sb["star_id"])
        sa = map_a.get(sid)
        if sa is None:
            continue
        w_pair = float(min(sa.get("weight", 1.0), sb.get("weight", 1.0)))
        anchors.append({
            "star_id": sid,
            "x_a": round(float(sa["x"]), 2),
            "y_a": round(float(sa["y"]), 2),
            "x_b": round(float(sb["x"]), 2),
            "y_b": round(float(sb["y"]), 2),
            "weight": round(w_pair, 4),
            "source": "plate_solve",
        })

    anchors.sort(key=lambda a: a["weight"], reverse=True)

    r_a = np.asarray(sol_a["rotation_icrf"], dtype=np.float64)
    r_b = np.asarray(sol_b["rotation_icrf"], dtype=np.float64)
    r_b_to_a = r_a.T @ r_b

    return {
        "frame_a": frame_a,
        "frame_b": frame_b,
        "sky_only": True,
        "anchors": anchors,
        "rotation_b_to_a": r_b_to_a,
    }
