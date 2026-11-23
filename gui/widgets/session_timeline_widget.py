# gui/widgets/session_timeline_widget.py
"""
gui/widgets/session_timeline_widget.py - Línea de tiempo interactiva (Timeline) para series fotográficas.

Inspirado en la maquetación temporal de EclipZe:
- Banda superior de fases / cobertura de sesión (Eclipse C1-C4, Startrails, Meteoros).
- Detección visual de discontinuidades (Gaps temporales) en rojo rayado con hitos navegables.
- Pista de tomas con marcadores individuales (♦/•) coloreados por estado de cribado (incluida/excluida/REF).
- Regla de tiempo continua inmune al cruce de medianoche (timestamps continuos en epoch).
- Aguja de tiempo (Playhead) interactiva con scrubbing, zoom/pan y tooltips enriquecidos.
"""
import os
import math
from typing import List, Set, Dict, Any, Optional, Tuple
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtWidgets import QWidget, QToolTip, QMenu
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPolygonF,
    QLinearGradient, QMouseEvent, QWheelEvent, QPaintEvent
)

from gui.theme import (
    BG_CANVAS, BG_PANEL, BG_CARD, BORDER_BASE, TEXT_PRIMARY,
    TEXT_SECONDARY, TEXT_MUTED, ACCENT_COBALT, ACCENT_RED
)


class SessionTimelineWidget(QWidget):
    """
    Widget visual de timeline astronómico para inspección de series, cribado y detección de huecos.
    Soporta modos especializados: 'eclipse', 'startrails', 'meteors', 'general'.
    """
    frame_selected = Signal(int)             # Índice del fotograma seleccionado (0..N-1)
    gap_selected = Signal(int, int)          # Salto detectado (from_idx, to_idx)
    zoom_changed = Signal(bool)              # True si está con zoom activo, False si vista completa
    auto_select_requested = Signal(float)    # Intervalo en segundos para auto-seleccionar tomas

    def __init__(self, mode: str = "general", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mode = mode.lower()  # "general", "startrails", "meteors", "eclipse"
        self.setFixedHeight(108)
        self.setMinimumWidth(320)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

        self.files_list: List[str] = []
        self.excluded_files: Set[str] = set()
        self.ref_idx: int = 0
        self.current_idx: int = 0
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

        # Datos temporales derivados
        self._items: List[Dict[str, Any]] = []
        self._gaps: List[Dict[str, Any]] = []
        self._hdr_bursts: List[Dict[str, Any]] = []
        self._min_time: float = 0.0
        self._max_time: float = 1.0
        self._median_exp: float = 15.0
        self._median_interval: float = 1.0

        # Zoom y Pan temporal
        self._zoom_range: Optional[Tuple[float, float]] = None
        self._is_dragging: bool = False
        self._drag_start_x: float = 0.0
        self._is_panning: bool = False
        self._pan_start_x: float = 0.0
        self._pan_start_bounds: Optional[Tuple[float, float]] = None
        self._has_panned: bool = False

        # Hover y feedback
        self._hover_idx: Optional[int] = None
        self._hover_gap: Optional[Dict[str, Any]] = None

        # Configuración específica para modo Eclipse
        self._eclipse_contacts: Dict[str, float] = {}  # {"C1": t1, "C2": t2, "MAX": t3, "C3": t4, "C4": t5}

    # =========================================================================
    # API PÚBLICA DE CONFIGURACIÓN
    # =========================================================================

    def set_data(
        self,
        files: List[str],
        metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None,
        excluded_files: Optional[Set[str]] = None,
        ref_idx: int = 0,
        current_idx: int = 0
    ) -> None:
        """Carga la serie completa de fotos y reconstruye la línea de tiempo."""
        self.files_list = list(files)
        if metadata_cache is not None:
            self._metadata_cache = metadata_cache
        self.excluded_files = set(excluded_files) if excluded_files else set()
        self.ref_idx = ref_idx
        self.current_idx = max(0, min(current_idx, len(self.files_list) - 1)) if self.files_list else 0
        self._recompute_timeline()
        self.update()

    def set_current_index(self, idx: int) -> None:
        """Actualiza la posición del cabezal/aguja de tiempo sin disparar señales recursivas."""
        if 0 <= idx < len(self.files_list) and idx != self.current_idx:
            self.current_idx = idx
            self.update()

    def set_excluded_files(self, excluded: Set[str]) -> None:
        """Actualiza el conjunto de archivos descartados en el cribado."""
        self.excluded_files = set(excluded)
        for it in self._items:
            it["is_excluded"] = (it["path"] in self.excluded_files)
        self.update()

    def set_reference_index(self, idx: int) -> None:
        """Actualiza la toma de referencia."""
        self.ref_idx = idx
        for it in self._items:
            it["is_ref"] = (it["idx"] == self.ref_idx)
        self.update()

    def update_item_metadata(self, path: str, meta: Dict[str, Any]) -> None:
        """Actualiza reactivamente los metadatos de una toma analizada en background."""
        self._metadata_cache[path] = meta
        ts = meta.get("timestamp")
        exp = meta.get("exp")
        updated = False
        for it in self._items:
            if it["path"] == path:
                if ts is not None and isinstance(ts, (int, float)) and ts > 0:
                    it["t"] = float(ts)
                if exp is not None and isinstance(exp, (int, float)) and exp > 0:
                    it["exp"] = float(exp)
                it["time_str"] = meta.get("time_str", it.get("time_str", "—"))
                it["datetime_str"] = meta.get("datetime_str", it.get("datetime_str", ""))
                updated = True
                break
        if updated:
            times = [it["t"] for it in self._items if it.get("t") is not None]
            if times:
                self._min_time = min(times)
                self._max_time = max(times)
                if self._max_time - self._min_time < 1.0:
                    self._max_time = self._min_time + 10.0
            valid_exps = [it["exp"] for it in self._items if it.get("exp", 0) > 0]
            if valid_exps:
                self._median_exp = sorted(valid_exps)[len(valid_exps) // 2]
            if self.mode == "eclipse":
                self._hdr_bursts = self._detect_hdr_bursts()
            self._recompute_gaps()
            self.update()

    def set_eclipse_contacts(self, contacts: Dict[str, float]) -> None:
        """Configura los instantes clave del eclipse (C1, C2, MAX, C3, C4 en epoch timestamp)."""
        self._eclipse_contacts = dict(contacts)
        self.update()

    def get_gaps(self) -> List[Dict[str, Any]]:
        """Retorna la lista de discontinuidades temporales detectadas."""
        return list(self._gaps)

    def get_hdr_bursts(self) -> List[Dict[str, Any]]:
        """Retorna la lista de ráfagas de bracketing HDR detectadas."""
        return list(self._hdr_bursts)

    def get_continuity_stats(self) -> Dict[str, Any]:
        """Calcula métricas de continuidad para Startrails / Sesión nocturna."""
        if not self._items or len(self._items) < 2:
            return {
                "continuity_pct": 100.0,
                "capture_gaps_count": 0,
                "lost_seconds": 0.0,
                "discarded_count": len(self.excluded_files)
            }
        total_span = max(1.0, self._max_time - self._min_time)
        lost_seconds = sum(g["gap_seconds"] for g in self._gaps)
        continuity_pct = max(0.0, min(100.0, 100.0 * (1.0 - (lost_seconds / total_span))))
        return {
            "continuity_pct": continuity_pct,
            "capture_gaps_count": len(self._gaps),
            "lost_seconds": lost_seconds,
            "discarded_count": len(self.excluded_files)
        }

    def jump_to_next_gap(self) -> bool:
        """Salta a la siguiente discontinuidad posterior a la toma actual."""
        if not self._gaps or not self.files_list:
            return False
        cur_t = self._items[self.current_idx]["t"] if (0 <= self.current_idx < len(self._items)) else 0.0
        for g in self._gaps:
            if g["from_t"] > cur_t + 0.1 or (g["from_idx"] > self.current_idx and abs(g["from_t"] - cur_t) <= 0.1):
                self.current_idx = g["from_idx"]
                self.frame_selected.emit(self.current_idx)
                self.gap_selected.emit(g["from_idx"], g["to_idx"])
                self.update()
                return True
        # Si no hay posteriores, salta al primero (cíclico)
        g = self._gaps[0]
        self.current_idx = g["from_idx"]
        self.frame_selected.emit(self.current_idx)
        self.gap_selected.emit(g["from_idx"], g["to_idx"])
        self.update()
        return True

    def jump_to_prev_gap(self) -> bool:
        """Salta a la discontinuidad anterior."""
        if not self._gaps or not self.files_list:
            return False
        cur_t = self._items[self.current_idx]["t"] if (0 <= self.current_idx < len(self._items)) else 0.0
        for g in reversed(self._gaps):
            if g["from_t"] < cur_t - 0.1 or (g["from_idx"] < self.current_idx and abs(g["from_t"] - cur_t) <= 0.1):
                self.current_idx = g["from_idx"]
                self.frame_selected.emit(self.current_idx)
                self.gap_selected.emit(g["from_idx"], g["to_idx"])
                self.update()
                return True
        # Cíclico al último
        g = self._gaps[-1]
        self.current_idx = g["from_idx"]
        self.frame_selected.emit(self.current_idx)
        self.gap_selected.emit(g["from_idx"], g["to_idx"])
        self.update()
        return True

    def jump_to_next_kept(self) -> bool:
        """Salta a la siguiente toma conservada (no descartada). Especial para modo meteoros."""
        if not self._items:
            return False
        for i in range(self.current_idx + 1, len(self._items)):
            if not self._items[i]["is_excluded"]:
                self.current_idx = i
                self.frame_selected.emit(self.current_idx)
                self.update()
                return True
        for i in range(0, self.current_idx):
            if not self._items[i]["is_excluded"]:
                self.current_idx = i
                self.frame_selected.emit(self.current_idx)
                self.update()
                return True
        return False

    def jump_to_prev_kept(self) -> bool:
        """Salta a la toma conservada anterior."""
        if not self._items:
            return False
        for i in range(self.current_idx - 1, -1, -1):
            if not self._items[i]["is_excluded"]:
                self.current_idx = i
                self.frame_selected.emit(self.current_idx)
                self.update()
                return True
        for i in range(len(self._items) - 1, self.current_idx, -1):
            if not self._items[i]["is_excluded"]:
                self.current_idx = i
                self.frame_selected.emit(self.current_idx)
                self.update()
                return True
        return False

    def zoom_to_totality(self, padding_s: float = 60.0) -> bool:
        """Encuadra la ventana temporal en la fase de totalidad [C2 - 60s, C3 + 60s]."""
        contacts = dict(self._eclipse_contacts) if self._eclipse_contacts else self._auto_estimate_contacts()
        c2 = contacts.get("C2")
        c3 = contacts.get("C3")
        c_max = contacts.get("MAX")
        if c2 and c3 and c2 < c3:
            t_start = max(self._min_time, c2 - padding_s)
            t_end = min(self._max_time, c3 + padding_s)
        elif c_max:
            t_start = max(self._min_time, c_max - padding_s * 1.5)
            t_end = min(self._max_time, c_max + padding_s * 1.5)
        else:
            return False
        self._zoom_range = (t_start, t_end)
        self.zoom_changed.emit(True)
        self.update()
        return True

    def reset_zoom(self) -> None:
        """Restaura la visualización completa de toda la sesión (Vista Completa)."""
        if self._zoom_range is not None:
            self._zoom_range = None
            self.zoom_changed.emit(False)
            self.update()

    # =========================================================================
    # CÁLCULOS INTERNOS DE TIEMPO Y GAPS
    # =========================================================================

    def _recompute_timeline(self) -> None:
        """Procesa metadatos, normaliza tiempos y detecta saltos anómalos."""
        self._items.clear()
        self._gaps.clear()

        if not self.files_list:
            self._min_time = 0.0
            self._max_time = 1.0
            self._hdr_bursts = []
            return

        has_any_ts = False
        parsed = []
        for i, p in enumerate(self.files_list):
            meta = self._metadata_cache.get(p, {})
            ts = meta.get("timestamp")
            if ts is not None and isinstance(ts, (int, float)) and ts > 0:
                has_any_ts = True
                parsed.append((i, p, float(ts), meta))
            else:
                parsed.append((i, p, None, meta))

        # Fallback si no hay metadatos: espaciado sintético equidistante
        if not has_any_ts:
            for i, p, _, meta in parsed:
                self._items.append({
                    "idx": i,
                    "path": p,
                    "t": float(i * 30.0),
                    "exp": float(meta.get("exp") or 15.0),
                    "time_str": meta.get("time_str", f"#{i+1}"),
                    "datetime_str": meta.get("datetime_str", ""),
                    "is_excluded": (p in self.excluded_files),
                    "is_ref": (i == self.ref_idx)
                })
            self._min_time = 0.0
            self._max_time = max(1.0, float((len(self.files_list) - 1) * 30.0))
            self._median_exp = 15.0
            self._hdr_bursts = []
            return

        # Rellenar tomas intermedias sin timestamp interpolando
        t_values = [p[2] for p in parsed if p[2] is not None]
        base_t = min(t_values) if t_values else 0.0
        last_t = base_t

        for i, p, ts, meta in parsed:
            if ts is not None:
                cur_t = ts
                last_t = ts
            else:
                last_t += 30.0  # incremento estimado si falta un EXIF puntual
                cur_t = last_t

            self._items.append({
                "idx": i,
                "path": p,
                "t": cur_t,
                "exp": float(meta.get("exp") or 0.0),
                "time_str": meta.get("time_str", "—"),
                "datetime_str": meta.get("datetime_str", ""),
                "is_excluded": (p in self.excluded_files),
                "is_ref": (i == self.ref_idx),
                "phase_label": meta.get("phase_label", ""),
                "obscuration_pct": meta.get("obscuration_pct", None),
                "is_ingress": meta.get("is_ingress", True),
            })

        times = [it["t"] for it in self._items]
        self._min_time = min(times)
        self._max_time = max(times)
        if self._max_time - self._min_time < 1.0:
            self._max_time = self._min_time + 10.0

        if self.mode == "eclipse":
            if not self._eclipse_contacts:
                self._eclipse_contacts = self._auto_estimate_contacts()
            self._hdr_bursts = self._detect_hdr_bursts()
        else:
            self._hdr_bursts = []

        self._recompute_gaps()

    def _calc_bar_height(self, exp_s: Optional[float]) -> float:
        """Calcula la altura del tallo vertical codificando logarítmicamente el tiempo de exposición (EV).
        Escala al 82% de la altura de la pista (72px * 0.82 = 59.04px):
        <= 1/4000s -> 10px, ~1/1000s -> ~17.5px, ~1/25s -> ~37.7px, ~1s -> ~55.3px, >= 2s -> 59.04px.
        Fallback a 14px si exp_s no está disponible o es nulo.
        """
        if exp_s is None or exp_s <= 0.0:
            return 14.0
        ev = math.log2(exp_s)
        # Rango logarítmico EV [-12 EV a +1 EV] (1/4000s a 2s)
        norm = (ev - (-12.0)) / (1.0 - (-12.0))
        norm = max(0.0, min(1.0, norm))
        track_h = 72.0
        max_h = track_h * 0.82
        min_h = 10.0
        return min_h + (norm * (max_h - min_h))

    def _detect_hdr_bursts(self) -> List[Dict[str, Any]]:
        """Detecta ráfagas de bracketing HDR (tomas consecutivas con dt < 2.0s y distinta exposición)."""
        if len(self._items) < 2:
            return []

        chrono = sorted(self._items, key=lambda it: it["t"])
        bursts: List[List[Dict[str, Any]]] = []
        cur_group: List[Dict[str, Any]] = []

        for i in range(len(chrono) - 1):
            it_a = chrono[i]
            it_b = chrono[i + 1]
            dt = it_b["t"] - it_a["t"]
            exp_a = float(it_a.get("exp") or 0.0)
            exp_b = float(it_b.get("exp") or 0.0)

            # Condición de bracketing: separación temporal < 2.0s y distinta exposición válida
            is_bracket_step = (0.0 <= dt < 2.0) and (abs(exp_b - exp_a) > 1e-5) and (exp_a > 0.0) and (exp_b > 0.0)
            if is_bracket_step:
                if not cur_group:
                    cur_group = [it_a, it_b]
                else:
                    cur_group.append(it_b)
            else:
                if len(cur_group) >= 2:
                    bursts.append(cur_group)
                cur_group = []

        if len(cur_group) >= 2:
            bursts.append(cur_group)

        result = []
        for g in bursts:
            result.append({
                "t_start": g[0]["t"],
                "t_end": g[-1]["t"],
                "items": g,
                "indices": {it["idx"] for it in g},
                "count": len(g)
            })
        return result

    def _recompute_gaps(self) -> None:
        """Detecta discontinuidades temporales (gaps) ordenando los fotogramas cronológicamente por timestamp."""
        self._gaps.clear()
        if len(self._items) < 2:
            return

        # Mediana de exposición
        valid_exps = [it["exp"] for it in self._items if it.get("exp", 0) > 0]
        self._median_exp = sorted(valid_exps)[len(valid_exps) // 2] if valid_exps else 15.0

        # Ordenar cronológicamente por timestamp para calcular gaps físicos reales entre tomas consecutivas en el tiempo
        chrono_items = sorted(self._items, key=lambda it: it["t"])

        deltas = []
        dead_intervals = []
        for i in range(len(chrono_items) - 1):
            dt = chrono_items[i + 1]["t"] - chrono_items[i]["t"]
            exp_val = chrono_items[i]["exp"] if chrono_items[i]["exp"] > 0 else self._median_exp
            idle = max(0.0, dt - exp_val)
            if dt > 0:
                deltas.append(dt)
            if idle > 0:
                dead_intervals.append(idle)

        if len(deltas) >= 3:
            s_deltas = sorted(deltas)
            median_dt = s_deltas[len(s_deltas) // 2]
            s_dead = sorted(dead_intervals) if dead_intervals else [1.0]
            self._median_interval = s_dead[len(s_dead) // 2]

            # Umbral según modo:
            if self.mode == "startrails":
                threshold = max(8.0, self._median_exp + self._median_interval + 2.5)
            else:
                threshold = max(60.0, median_dt * 2.5)

            for i in range(len(chrono_items) - 1):
                dt = chrono_items[i + 1]["t"] - chrono_items[i]["t"]
                if dt > threshold:
                    self._gaps.append({
                        "from_idx": chrono_items[i]["idx"],
                        "to_idx": chrono_items[i + 1]["idx"],
                        "from_t": chrono_items[i]["t"],
                        "to_t": chrono_items[i + 1]["t"],
                        "gap_seconds": dt,
                        "from_path": chrono_items[i]["path"],
                        "to_path": chrono_items[i + 1]["path"],
                        "gap_type": "capture",
                        "missing_frames_est": max(1, int(round((dt - median_dt) / max(1.0, median_dt))))
                    })

    def _auto_estimate_contacts(self) -> Dict[str, float]:
        """Auto-estima los contactos astronómicos a partir de metadatos o timestamps de los fotogramas."""
        if not self._items:
            return {}
        valid = [it for it in self._items if it.get("t") is not None]
        if not valid:
            return {}
        t_c1 = self._min_time
        t_c4 = self._max_time

        tot_items = [it for it in valid if "TOTALIDAD" in (it.get("phase_label") or "").upper() or (it.get("obscuration_pct") or 0.0) >= 98.0]
        c2_items = [it for it in valid if (("DIAMANTE" in (it.get("phase_label") or "").upper() or "BAILY" in (it.get("phase_label") or "").upper()) and it.get("is_ingress", True)) or ("(C2)" in (it.get("phase_label") or "").upper())]
        c3_items = [it for it in valid if (("DIAMANTE" in (it.get("phase_label") or "").upper() or "BAILY" in (it.get("phase_label") or "").upper()) and not it.get("is_ingress", True)) or ("(C3)" in (it.get("phase_label") or "").upper())]

        contacts = {"C1": t_c1, "C4": t_c4}
        if tot_items or c2_items or c3_items:
            t_c2 = c2_items[0]["t"] if c2_items else tot_items[0]["t"]
            t_c3 = c3_items[-1]["t"] if c3_items else tot_items[-1]["t"]
            t_max = tot_items[len(tot_items) // 2]["t"] if tot_items else (t_c2 + t_c3) / 2.0
            contacts["C2"] = t_c2
            contacts["MAX"] = t_max
            contacts["C3"] = t_c3
        else:
            best = max(valid, key=lambda it: it.get("obscuration_pct") or 0.0)
            if (best.get("obscuration_pct") or 0.0) > 0:
                contacts["MAX"] = best["t"]
            else:
                contacts["MAX"] = (t_c1 + t_c4) / 2.0

        return contacts

    # =========================================================================
    # TRANSFORMACIÓN DE COORDENADAS (TIEMPO <-> PÍXELES)
    # =========================================================================

    def _get_active_bounds(self) -> Tuple[float, float]:
        if self._zoom_range is not None:
            return self._zoom_range
        return self._min_time, self._max_time

    def _time_to_x(self, t: float, width: float, margin: float = 12.0) -> float:
        t_start, t_end = self._get_active_bounds()
        span = max(0.001, t_end - t_start)
        usable_w = width - (2 * margin)
        ratio = (t - t_start) / span
        return margin + (ratio * usable_w)

    def _x_to_time(self, x: float, width: float, margin: float = 12.0) -> float:
        t_start, t_end = self._get_active_bounds()
        usable_w = max(1.0, width - (2 * margin))
        ratio = max(0.0, min(1.0, (x - margin) / usable_w))
        return t_start + (ratio * (t_end - t_start))

    def _find_closest_frame(self, t: float) -> int:
        if not self._items:
            return 0
        best_idx = 0
        min_diff = float("inf")
        for it in self._items:
            diff = abs(it["t"] - t)
            if diff < min_diff:
                min_diff = diff
                best_idx = it["idx"]
        return best_idx

    # =========================================================================
    # RENDERIZADO CON QPAINTER
    # =========================================================================

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        w = float(self.width())
        h = float(self.height())
        margin = 14.0
        usable_w = max(1.0, w - (2 * margin))

        # 1. Fondo contenedor
        painter.fillRect(QRectF(0, 0, w, h), QColor(BG_CANVAS))
        pen_border = QPen(QColor(BORDER_BASE), 1)
        painter.setPen(pen_border)
        painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 4, 4)

        if not self._items:
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "Sin tomas cargadas para la línea de tiempo")
            return

        t_start, t_end = self._get_active_bounds()

        # Fondos estructurados de las 3 Franjas Horizontales (108px de alto total, +58%):
        # Franja 1 (Cabecera de Fases / Cobertura): Y = 0..18, alto 18px, fondo #14171D
        # Franja 2 (Pista Principal de Tomas / Gaps): Y = 18..90, alto 72px, fondo #0F1115
        # Franja 3 (Regla Temporal): Y = 90..108, alto 18px, fondo #181B22
        painter.fillRect(QRectF(margin, 0.0, usable_w, 18.0), QColor("#14171d"))
        painter.fillRect(QRectF(margin, 18.0, usable_w, 72.0), QColor("#0f1115"))
        painter.fillRect(QRectF(margin, 90.0, usable_w, 18.0), QColor("#181b22"))

        # Líneas divisorias horizontales finas (#2A2F3A)
        painter.setPen(QPen(QColor("#2a2f3a"), 1.0))
        painter.drawLine(QPointF(margin, 18.0), QPointF(w - margin, 18.0))
        painter.drawLine(QPointF(margin, 90.0), QPointF(w - margin, 90.0))

        # 2. FRANJA 1: Fases o Cobertura de Sesión (Y: 0..18)
        self._draw_phase_band(painter, 0.0, 18.0, w, margin)

        # 3. DISCONTINUIDADES / GAPS TEMPORALES (Franja 2: Y 18..90, alto 72px)
        self._draw_gaps(painter, 18.0, 72.0, w, margin)

        # 4. FRANJA 2: Pista de tomas (centro en Y: 54.0 para modos continuos)
        y_track = 54.0
        self._draw_frame_markers(painter, y_track, w, margin)

        # 5. FRANJA 3: Regla horaria inferior (Y: 90..108, alto 18px)
        self._draw_time_ruler(painter, 90.0, w, margin, t_start, t_end)

        # 6. AGUJA DE TIEMPO (PLAYHEAD) (2px cian con tirador triangular 10x8)
        self._draw_playhead(painter, h, w, margin)

        # 7. MINIMAPA / SCROLLBAR FINO EN ZOOM ACTIVO
        self._draw_minimap(painter, h, w, margin)

    def _draw_phase_band(self, painter: QPainter, y: float, h: float, w: float, margin: float) -> None:
        """Dibuja la Franja 1: fases (Eclipse) o cobertura de sesión (Startrails/Meteoros/General)."""
        usable_w = max(1.0, w - (2 * margin))

        if self.mode == "eclipse":
            contacts = dict(self._eclipse_contacts) if self._eclipse_contacts else self._auto_estimate_contacts()
            c1 = contacts.get("C1", self._min_time)
            c2 = contacts.get("C2")
            c_max = contacts.get("MAX")
            c3 = contacts.get("C3")
            c4 = contacts.get("C4", self._max_time)

            x_c1 = max(margin, min(w - margin, self._time_to_x(c1, w, margin)))
            x_c4 = max(margin, min(w - margin, self._time_to_x(c4, w, margin)))

            total_h = float(self.height())
            # Trazar líneas divisorias verticales de 1px (#2A2F3A) desde cada frontera atravesando las 3 franjas
            pen_div = QPen(QColor("#2a2f3a"), 1.0)
            painter.setPen(pen_div)
            if margin <= x_c1 <= w - margin:
                painter.drawLine(QPointF(x_c1, 0.0), QPointF(x_c1, total_h))
            if margin <= x_c4 <= w - margin:
                painter.drawLine(QPointF(x_c4, 0.0), QPointF(x_c4, total_h))

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))

            if c2 and c3 and c2 < c3:
                x_c2 = max(margin, min(w - margin, self._time_to_x(c2, w, margin)))
                x_c3 = max(margin, min(w - margin, self._time_to_x(c3, w, margin)))

                # Líneas divisorias en C2 y C3 atravesando las tres franjas
                painter.setPen(pen_div)
                if margin <= x_c2 <= w - margin:
                    painter.drawLine(QPointF(x_c2, 0.0), QPointF(x_c2, total_h))
                if margin <= x_c3 <= w - margin:
                    painter.drawLine(QPointF(x_c3, 0.0), QPointF(x_c3, total_h))

                # 1. Fase Parcial C1 a C2 (Entrada) - Borde inferior azul (#2D7FF9)
                w_in = max(1.0, x_c2 - x_c1)
                rect_in = QRectF(x_c1, y, w_in, h - 2.0)
                painter.fillRect(QRectF(x_c1, y + h - 2.0, w_in, 2.0), QColor("#2d7ff9"))
                dur_in_m = int(max(0.0, c2 - c1) / 60.0)
                dur_in_str = f" ({dur_in_m}m)" if dur_in_m > 0 else ""
                if w_in > 70:
                    painter.setPen(QColor("#94a3b8"))
                    painter.drawText(rect_in, Qt.AlignCenter, f"C1-C2 ENTRADA{dur_in_str}")
                elif w_in > 40:
                    painter.setPen(QColor("#94a3b8"))
                    painter.drawText(rect_in, Qt.AlignCenter, "C1-C2")

                # Contacto C2 (Diamante / Perlas C2) en Franja 1
                painter.fillRect(QRectF(x_c2 - 1.0, y, 2.0, h), QColor("#facc15"))

                # 2. Totalidad C2 a C3 (Corona Solar) - Borde inferior violeta (#8B5CF6) y sutil tinte
                w_tot = max(2.0, x_c3 - x_c2)
                rect_tot = QRectF(x_c2, y, w_tot, h - 2.0)
                painter.fillRect(QRectF(x_c2, y, w_tot, h - 2.0), QColor(139, 92, 246, 20))
                painter.fillRect(QRectF(x_c2, y + h - 2.0, w_tot, 2.0), QColor("#8b5cf6"))
                tot_s = int(max(0.0, c3 - c2))
                tot_m, tot_sec = tot_s // 60, tot_s % 60
                dur_tot_str = f" ({tot_m}m {tot_sec}s)" if tot_m > 0 else f" ({tot_sec}s)"
                if w_tot > 65:
                    painter.setPen(QColor("#c084fc"))
                    painter.drawText(rect_tot, Qt.AlignCenter, f"TOTALIDAD{dur_tot_str}")
                elif w_tot > 36:
                    painter.setPen(QColor("#c084fc"))
                    painter.drawText(rect_tot, Qt.AlignCenter, "TOTALIDAD")

                # Contacto C3 (Perlas / Diamante C3) en Franja 1
                painter.fillRect(QRectF(x_c3 - 1.0, y, 2.0, h), QColor("#facc15"))

                # 3. Fase Parcial C3 a C4 (Salida) - Borde inferior terracota/naranja (#EA580C)
                w_out = max(1.0, x_c4 - x_c3)
                rect_out = QRectF(x_c3, y, w_out, h - 2.0)
                painter.fillRect(QRectF(x_c3, y + h - 2.0, w_out, 2.0), QColor("#ea580c"))
                dur_out_m = int(max(0.0, c4 - c3) / 60.0)
                dur_out_str = f" ({dur_out_m}m)" if dur_out_m > 0 else ""
                if w_out > 70:
                    painter.setPen(QColor("#94a3b8"))
                    painter.drawText(rect_out, Qt.AlignCenter, f"C3-C4 SALIDA{dur_out_str}")
                elif w_out > 40:
                    painter.setPen(QColor("#94a3b8"))
                    painter.drawText(rect_out, Qt.AlignCenter, "C3-C4")
            else:
                # Sin totalidad definida (eclipse parcial)
                t_mid = c_max if c_max else (c1 + c4) / 2.0
                x_mid = max(margin, min(w - margin, self._time_to_x(t_mid, w, margin)))
                if margin <= x_mid <= w - margin:
                    painter.setPen(pen_div)
                    painter.drawLine(QPointF(x_mid, 0.0), QPointF(x_mid, total_h))

                w_in = max(1.0, x_mid - x_c1)
                painter.fillRect(QRectF(x_c1, y + h - 2.0, w_in, 2.0), QColor("#2d7ff9"))
                if w_in > 40:
                    painter.setPen(QColor("#94a3b8"))
                    painter.drawText(QRectF(x_c1, y, w_in, h - 2.0), Qt.AlignCenter, "C1-MAX")

                painter.fillRect(QRectF(x_mid - 1.0, y, 2.0, h), QColor("#facc15"))

                w_out = max(1.0, x_c4 - x_mid)
                painter.fillRect(QRectF(x_mid, y + h - 2.0, w_out, 2.0), QColor("#ea580c"))
                if w_out > 40:
                    painter.setPen(QColor("#94a3b8"))
                    painter.drawText(QRectF(x_mid, y, w_out, h - 2.0), Qt.AlignCenter, "MAX-C4")
        else:
            # Modo Astrofotografía General / Startrails / Meteoros: Barra de Cobertura Nocturna
            span_s = max(0.0, self._max_time - self._min_time)
            dur_h = span_s / 3600.0
            if dur_h >= 1.0:
                span_str = f" ({dur_h:.1f}h)"
            elif span_s >= 60.0:
                span_str = f" ({int(span_s / 60.0)}m)"
            else:
                span_str = ""

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor("#64748b"))
            title_text = "COBERTURA DE SESIÓN" + span_str
            painter.drawText(QRectF(margin + 6.0, y, usable_w - 12.0, h), Qt.AlignVCenter | Qt.AlignLeft, title_text)

    def _draw_gaps(self, painter: QPainter, y_top: float, h_total: float, w: float, margin: float) -> None:
        """Dibuja discontinuidades temporales con trama diagonal ámbar y regla anti-solapamiento."""
        for g in self._gaps:
            x1 = self._time_to_x(g["from_t"], w, margin)
            x2 = self._time_to_x(g["to_t"], w, margin)
            gap_w = max(3.0, x2 - x1)
            gap_rect = QRectF(x1, y_top, gap_w, h_total)

            # 1. Fondo sombreado ámbar tenue con trama rayada diagonal
            painter.fillRect(gap_rect, QColor(245, 158, 11, 28))
            painter.fillRect(gap_rect, QBrush(QColor(245, 158, 11, 46), Qt.BDiagPattern))

            # 2. Bordes delimitadores laterales finos (#D97706)
            painter.setPen(QPen(QColor("#d97706"), 1.0))
            painter.drawLine(QPointF(x1, y_top), QPointF(x1, y_top + h_total))
            painter.drawLine(QPointF(x2, y_top), QPointF(x2, y_top + h_total))

            # 3. Marcador triangular inferior para gaps en Startrails
            if self.mode == "startrails":
                tri = QPolygonF([
                    QPointF(x1, y_top + h_total),
                    QPointF(x1 + 6.0, y_top + h_total),
                    QPointF(x1 + 3.0, y_top + h_total - 5.0)
                ])
                painter.setBrush(QColor("#d97706"))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(tri)

            # 4. Etiqueta compacta interna (6m, 20s): SÓLO si gap_w >= 38.0 (anti-solapamiento)
            if gap_w >= 38.0:
                dt_s = g["gap_seconds"]
                if dt_s >= 3600:
                    lbl = f"{dt_s / 3600.0:.1f}h"
                elif dt_s >= 60:
                    lbl = f"{dt_s / 60.0:.0f}m"
                else:
                    lbl = f"{dt_s:.0f}s"
                painter.setPen(QColor("#fbbf24"))
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                painter.drawText(gap_rect, Qt.AlignCenter, lbl)

    def _draw_frame_markers(self, painter: QPainter, y: float, w: float, margin: float) -> None:
        """Dibuja los marcadores de las tomas según el modo activo."""
        t_start, t_end = self._get_active_bounds()

        if self.mode == "startrails":
            # Línea base continua
            painter.setPen(QPen(QColor("#334155"), 1.5))
            painter.drawLine(QPointF(margin, y), QPointF(w - margin, y))

            for it in self._items:
                t = it["t"]
                exp_s = it.get("exp") or self._median_exp or 15.0
                if t + exp_s < t_start or t > t_end:
                    continue

                x1 = self._time_to_x(t, w, margin)
                x2 = self._time_to_x(t + exp_s, w, margin)
                block_w = max(2.5, x2 - x1)
                is_cur = (it["idx"] == self.current_idx)
                is_ref = it.get("is_ref", False)
                is_exc = it.get("is_excluded", False)

                bar_rect = QRectF(x1, y - 3.5, block_w, 7.0)

                if is_exc:
                    # Toma descartada: Ámbar (#f59e0b) avisando de discontinuidad en trazos
                    painter.setBrush(QColor("#f59e0b"))
                    painter.setPen(QPen(QColor("#d97706"), 1))
                    painter.drawRect(bar_rect)
                elif is_ref:
                    # Toma de referencia: Dorado
                    painter.setBrush(QColor("#fbbf24"))
                    painter.setPen(QPen(QColor("#f59e0b"), 1))
                    painter.drawRect(bar_rect)
                else:
                    # Toma normal incluida: Cian / Cobalt
                    painter.setBrush(QColor("#38bdf8"))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(bar_rect)

                if is_cur:
                    painter.setPen(QPen(QColor("#ffffff"), 1.5))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(QRectF(x1 - 1, y - 4.5, block_w + 2, 9.0))

        elif self.mode == "meteors":
            # Línea base continua
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.drawLine(QPointF(margin, y), QPointF(w - margin, y))

            for it in self._items:
                t = it["t"]
                if t < t_start or t > t_end:
                    continue

                x = self._time_to_x(t, w, margin)
                is_cur = (it["idx"] == self.current_idx)
                is_exc = it.get("is_excluded", False)
                is_ref = it.get("is_ref", False)

                if is_exc:
                    # Descartada: Puntito muy tenue gris (#374151, radio 1.5px)
                    painter.setPen(QColor("#374151"))
                    painter.setBrush(QColor("#374151"))
                    painter.drawEllipse(QPointF(x, y), 1.5, 1.5)
                else:
                    # INCLUIDA / CONSERVAR (Meteoro detectado): Rombo / Pin prominente Verde Esmeralda (#10B981)
                    col_fill = QColor("#10b981") if not is_cur else QColor("#34d399")
                    painter.setPen(QPen(QColor("#ffffff") if is_cur else QColor("#059669"), 1))
                    painter.setBrush(col_fill)
                    # Diamante que sobresale 4px por encima y debajo (alto 12px, ancho 8px)
                    poly = QPolygonF([
                        QPointF(x, y - 6.0),
                        QPointF(x + 4.0, y),
                        QPointF(x, y + 6.0),
                        QPointF(x - 4.0, y)
                    ])
                    painter.drawPolygon(poly)

                if is_cur and is_exc:
                    painter.setPen(QColor("#ffffff"))
                    painter.setBrush(QColor("#ffffff"))
                    painter.drawEllipse(QPointF(x, y), 2.5, 2.5)

        elif self.mode == "eclipse":
            y_base = 88.0
            # Línea base continua sutil (Antialiasing desactivado para nitidez de 1px)
            painter.setRenderHint(QPainter.Antialiasing, False)
            painter.setPen(QPen(QColor("#27272a"), 1.0))
            painter.drawLine(QPointF(margin, y_base), QPointF(w - margin, y_base))

            # Conectores de ráfagas HDR (subrayado de 2px en #6366F1 en la base del grupo)
            y_under = 89.0
            painter.setPen(QPen(QColor("#6366f1"), 2.0))
            for burst in self._hdr_bursts:
                b_t1 = burst["t_start"]
                b_t2 = burst["t_end"]
                if b_t2 < t_start or b_t1 > t_end:
                    continue
                x1 = self._time_to_x(b_t1, w, margin)
                x2 = self._time_to_x(b_t2, w, margin)
                left_x = min(x1, x2) - 2.0
                right_x = max(x1, x2) + 2.0
                painter.drawLine(QPointF(left_x, y_under), QPointF(right_x, y_under))

            # Clasificación de tomas visibles para Z-order en dos pasadas
            exc_items = []
            col_items = []
            cur_item = None

            for it in self._items:
                t = it["t"]
                if t < t_start or t > t_end:
                    continue

                x = self._time_to_x(t, w, margin)
                stem_h = self._calc_bar_height(it.get("exp"))
                y_top = y_base - stem_h
                data = (it, x, y_top)

                if it["idx"] == self.current_idx:
                    cur_item = data

                if it.get("is_excluded", False):
                    exc_items.append(data)
                else:
                    col_items.append(data)

            # =================================================================
            # PASADA 1: Tomas Disponibles (gris pizarra #64748B)
            # =================================================================
            # 1.1. Tallos de 1px con Antialiasing = False (máxima nitidez pixel-perfect)
            painter.setRenderHint(QPainter.Antialiasing, False)
            painter.setPen(QPen(QColor("#334155"), 1.0))
            for it, x, y_top in exc_items:
                rx = round(x)
                painter.drawLine(QPointF(rx, y_base), QPointF(rx, y_top))

            # 1.2. Cabezas circulares con Antialiasing = True (radio 2.2px, diámetro 4.4px)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#64748b"))
            for it, x, y_top in exc_items:
                painter.drawEllipse(QPointF(x, y_top), 2.2, 2.2)

            # =================================================================
            # PASADA 2: Tomas En Collage / Fase (encima para no quedar tapadas)
            # =================================================================
            # 2.1. Tallos de 1px con Antialiasing = False
            painter.setRenderHint(QPainter.Antialiasing, False)
            for it, x, y_top in col_items:
                p_label = it.get("phase_label", "").upper()
                if "TOTALIDAD" in p_label:
                    col_stem = QColor(168, 85, 247, 180)
                elif "DIAMANTE" in p_label or "BAILY" in p_label:
                    col_stem = QColor(245, 158, 11, 180)
                elif it.get("is_ref", False):
                    col_stem = QColor(217, 119, 6, 200)
                else:
                    col_stem = QColor(14, 165, 233, 180)
                painter.setPen(QPen(col_stem, 1.0))
                rx = round(x)
                painter.drawLine(QPointF(rx, y_base), QPointF(rx, y_top))

            # 2.2. Halos exteriores translúcidos (radio 5.5px, Antialiasing = True)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            for it, x, y_top in col_items:
                p_label = it.get("phase_label", "").upper()
                if "TOTALIDAD" in p_label:
                    halo_col = QColor(192, 132, 252, 64)   # rgba púrpura 0.25
                elif "DIAMANTE" in p_label or "BAILY" in p_label or it.get("is_ref", False):
                    halo_col = QColor(251, 191, 36, 64)    # rgba dorado 0.25
                else:
                    halo_col = QColor(56, 189, 248, 64)    # rgba(56, 189, 248, 0.25)
                painter.setBrush(halo_col)
                painter.drawEllipse(QPointF(x, y_top), 5.5, 5.5)

            # 2.3. Círculos de las cabezas (radio 3.2px, diámetro 6.4px en #38BDF8)
            for it, x, y_top in col_items:
                p_label = it.get("phase_label", "").upper()
                if "TOTALIDAD" in p_label:
                    painter.setPen(QPen(QColor("#9333ea"), 1.0))
                    painter.setBrush(QColor("#c084fc"))
                elif "DIAMANTE" in p_label or "BAILY" in p_label:
                    painter.setPen(QPen(QColor("#f59e0b"), 1.0))
                    painter.setBrush(QColor("#fde047"))
                elif it.get("is_ref", False):
                    painter.setPen(QPen(QColor("#f59e0b"), 1.0))
                    painter.setBrush(QColor("#fbbf24"))
                else:
                    painter.setPen(QPen(QColor("#0284c7"), 1.0))
                    painter.setBrush(QColor("#38bdf8"))
                painter.drawEllipse(QPointF(x, y_top), 3.2, 3.2)

            # =================================================================
            # REMATE: Fotograma Activo Seleccionado (en la cima de todo el Z-order)
            # =================================================================
            if cur_item is not None:
                it, x, y_top = cur_item
                # Tallo destacado de 1.5px
                painter.setRenderHint(QPainter.Antialiasing, False)
                painter.setPen(QPen(QColor("#38bdf8"), 1.5))
                rx = round(x)
                painter.drawLine(QPointF(rx, y_base), QPointF(rx, y_top))

                # Halo activo más amplio y cabeza blanca brillante con borde cian
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(56, 189, 248, 90))
                painter.drawEllipse(QPointF(x, y_top), 7.0, 7.0)

                painter.setPen(QPen(QColor("#38bdf8"), 1.5))
                painter.setBrush(QColor("#ffffff"))
                painter.drawEllipse(QPointF(x, y_top), 4.0, 4.0)

            # Restaurar Antialiasing a True para el resto de elementos
            painter.setRenderHint(QPainter.Antialiasing, True)
        else:
            # Modo general (Stacker, Panorama)
            painter.setPen(QPen(QColor("#27272a"), 1.0, Qt.DashLine))
            painter.drawLine(QPointF(margin, y), QPointF(w - margin, y))

            for it in self._items:
                t = it["t"]
                if t < t_start or t > t_end:
                    continue
                x = self._time_to_x(t, w, margin)
                is_cur = (it["idx"] == self.current_idx)
                is_ref = it.get("is_ref", False)
                is_exc = it.get("is_excluded", False)

                if is_exc:
                    # Toma descartada: Gris pizarra sobrio (#475569) en vez de rojo chillón
                    color = QColor("#475569") if not is_cur else QColor("#ffffff")
                    painter.setPen(color)
                    painter.setBrush(color)
                    painter.drawEllipse(QPointF(x, y), 2.0, 2.0)
                elif is_ref:
                    painter.setPen(QColor("#f59e0b"))
                    painter.setBrush(QColor("#fbbf24"))
                    poly = QPolygonF([
                        QPointF(x, y - 4.5),
                        QPointF(x + 3.5, y),
                        QPointF(x, y + 4.5),
                        QPointF(x - 3.5, y)
                    ])
                    painter.drawPolygon(poly)
                else:
                    color = QColor("#38bdf8") if not is_cur else QColor("#ffffff")
                    painter.setPen(color)
                    painter.setBrush(color)
                    painter.drawEllipse(QPointF(x, y), 2.5, 2.5)

    def _draw_time_ruler(self, painter: QPainter, y: float, w: float, margin: float, t_start: float, t_end: float) -> None:
        """Dibuja la Franja 3: escala horaria continua inferior con anti-colisión de hitos."""
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(TEXT_MUTED))

        num_ticks = max(3, int((w - 2 * margin) / 85.0))
        span = t_end - t_start

        for i in range(num_ticks + 1):
            ratio = i / float(num_ticks)
            t = t_start + (ratio * span)
            x = margin + (ratio * (w - 2 * margin))

            # Marcas de tiempo de 4px en #64748B
            painter.setPen(QColor("#64748b"))
            painter.drawLine(QPointF(x, y), QPointF(x, y + 4.0))

            try:
                dt = datetime.fromtimestamp(t)
                time_lbl = dt.strftime("%H:%M")
            except Exception:
                mins = int(t / 60.0)
                time_lbl = f"+{mins}m"

            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(QRectF(x - 22.0, y + 4.0, 44.0, 11.0), Qt.AlignCenter, time_lbl)

        # Hitos específicos en Eclipse con Anti-Colisión
        if self.mode == "eclipse":
            contacts = dict(self._eclipse_contacts) if self._eclipse_contacts else self._auto_estimate_contacts()
            c1_t = contacts.get("C1")
            c2_t = contacts.get("C2")
            max_t = contacts.get("MAX")
            c3_t = contacts.get("C3")
            c4_t = contacts.get("C4")

            c1_x = self._time_to_x(c1_t, w, margin) if (c1_t and t_start <= c1_t <= t_end) else None
            c2_x = self._time_to_x(c2_t, w, margin) if (c2_t and t_start <= c2_t <= t_end) else None
            max_x = self._time_to_x(max_t, w, margin) if (max_t and t_start <= max_t <= t_end) else None
            c3_x = self._time_to_x(c3_t, w, margin) if (c3_t and t_start <= c3_t <= t_end) else None
            c4_x = self._time_to_x(c4_t, w, margin) if (c4_t and t_start <= c4_t <= t_end) else None

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))

            # Contactos C1 y C4 en la regla inferior
            if c1_x is not None:
                painter.setPen(QColor("#38bdf8"))
                painter.drawText(QRectF(c1_x - 14.0, y + 3.0, 28.0, 11.0), Qt.AlignCenter, "C1")
            if c4_x is not None:
                painter.setPen(QColor("#ea580c"))
                painter.drawText(QRectF(c4_x - 14.0, y + 3.0, 28.0, 11.0), Qt.AlignCenter, "C4")

            # Contactos C2 y C3 en la regla inferior (Franja 3)
            if c2_x is not None:
                painter.setPen(QColor("#fbbf24"))
                painter.drawText(QRectF(c2_x - 14.0, y + 3.0, 28.0, 11.0), Qt.AlignCenter, "C2")
            if c3_x is not None:
                painter.setPen(QColor("#fbbf24"))
                painter.drawText(QRectF(c3_x - 14.0, y + 3.0, 28.0, 11.0), Qt.AlignCenter, "C3")

            # Contacto MAX: línea dorada vertical fina (#FACC15) desde Franja 1 hasta Franja 3
            if max_x is not None:
                painter.setPen(QPen(QColor("#facc15"), 1.0, Qt.DashLine))
                painter.drawLine(QPointF(max_x, 0.0), QPointF(max_x, y))

                # Regla anti-colisión: mostrar texto MAX en Franja 1 SÓLO si C2 y C3 distan >= 32px
                tot_dist = (c3_x - c2_x) if (c2_x is not None and c3_x is not None) else 100.0
                if tot_dist >= 32.0:
                    painter.setPen(QColor("#facc15"))
                    painter.drawText(QRectF(max_x - 16.0, 1.0, 32.0, 13.0), Qt.AlignCenter, "MAX")

    def _draw_playhead(self, painter: QPainter, h: float, w: float, margin: float) -> None:
        """Dibuja la aguja de tiempo del fotograma actual con tirador triangular de 10x8px y línea de 2px."""
        if not self._items or not (0 <= self.current_idx < len(self._items)):
            return

        cur_t = self._items[self.current_idx]["t"]
        x = self._time_to_x(cur_t, w, margin)

        # Línea vertical de 2px en cian (#38BDF8) atravesando hasta la regla
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawLine(QPointF(x, 0.0), QPointF(x, h - 3.0))

        # Tirador superior triangular de 10x8px en la parte superior de Franja 1
        tri = QPolygonF([
            QPointF(x - 5.0, 0.0),
            QPointF(x + 5.0, 0.0),
            QPointF(x, 8.0)
        ])
        painter.setBrush(QColor("#38bdf8"))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(tri)

        if self.mode == "eclipse":
            # Círculo base brillante sobre la línea de base (Y: 88.0) como ancla de la toma activa
            y_base = 88.0
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.setBrush(QColor("#22d3ee"))
            painter.drawEllipse(QPointF(x, y_base), 4.5, 4.5)
        else:
            # Halo en el punto de la toma activa sobre la pista (centro Franja 2: Y = 54.0)
            y_track = 54.0
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.setBrush(QColor("#0284c7"))
            painter.drawEllipse(QPointF(x, y_track), 4.5, 4.5)

    def _draw_minimap(self, painter: QPainter, h: float, w: float, margin: float) -> None:
        """Dibuja un minimapa/scrollbar horizontal fino (2.5px) cuando zoom_level > 1.0."""
        if self._zoom_range is None:
            return

        y_bar = h - 3.0
        h_bar = 2.5
        usable_w = max(1.0, w - (2 * margin))
        full_span = max(1.0, self._max_time - self._min_time)
        z_start, z_end = self._zoom_range

        # Fondo del canal
        painter.fillRect(QRectF(margin, y_bar, usable_w, h_bar), QColor("#1e293b"))

        # Pulgar (thumb) visible
        r1 = max(0.0, min(1.0, (z_start - self._min_time) / full_span))
        r2 = max(0.0, min(1.0, (z_end - self._min_time) / full_span))
        thumb_x = margin + (r1 * usable_w)
        thumb_w = max(6.0, (r2 - r1) * usable_w)

        painter.fillRect(QRectF(thumb_x, y_bar, thumb_w, h_bar), QColor("#38bdf8"))

    # =========================================================================
    # EVENTOS DE RATÓN E INTERACCIÓN
    # =========================================================================

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        x = pos.x()
        y = pos.y()
        h = float(self.height())

        if event.button() == Qt.LeftButton:
            if self._zoom_range is not None and y >= h - 8.0:
                self._handle_minimap_click(x)
            else:
                self._is_dragging = True
                self._handle_scrub(x)
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._is_panning = True
            self._pan_start_x = x
            self._pan_start_bounds = self._get_active_bounds()
            self._has_panned = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        x = pos.x()
        if self._is_dragging:
            self._handle_scrub(x)
        elif self._is_panning and self._zoom_range is not None and self._pan_start_bounds:
            dx = x - self._pan_start_x
            if abs(dx) > 2.0:
                self._has_panned = True
            w = float(self.width())
            margin = 14.0
            usable_w = max(1.0, w - (2 * margin))
            span = self._pan_start_bounds[1] - self._pan_start_bounds[0]
            dt = -(dx / usable_w) * span

            orig_start, orig_end = self._pan_start_bounds
            new_start = orig_start + dt
            new_end = orig_end + dt

            if new_start < self._min_time:
                new_start = self._min_time
                new_end = new_start + span
            elif new_end > self._max_time:
                new_end = self._max_time
                new_start = new_end - span

            self._zoom_range = (new_start, new_end)
            self.update()
        else:
            self._update_hover_tooltip(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            had_panned = self._has_panned
            self._is_panning = False
            self._pan_start_bounds = None
            if event.button() == Qt.RightButton and not had_panned:
                self._show_context_menu(event.globalPosition().toPoint())

    def _handle_minimap_click(self, x: float) -> None:
        if self._zoom_range is None:
            return
        w = float(self.width())
        margin = 14.0
        usable_w = max(1.0, w - (2 * margin))
        ratio = max(0.0, min(1.0, (x - margin) / usable_w))
        full_span = max(1.0, self._max_time - self._min_time)
        center_t = self._min_time + (ratio * full_span)
        span = self._zoom_range[1] - self._zoom_range[0]

        new_start = max(self._min_time, center_t - (span / 2.0))
        new_end = new_start + span
        if new_end > self._max_time:
            new_end = self._max_time
            new_start = max(self._min_time, new_end - span)
        self._zoom_range = (new_start, new_end)
        self.update()

    def _show_context_menu(self, global_pos) -> None:
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e293b;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }
            QMenu::item:selected {
                background-color: #3b82f6;
            }
        """)

        if self.mode == "eclipse":
            act_tot = menu.addAction("🔍 Zoom a la Totalidad (C2 a C3)")
            act_tot.triggered.connect(lambda: self.zoom_to_totality())

            sub_auto = menu.addMenu("Auto-seleccionar cada N min en tramo libre...")
            sub_auto.addAction("Cada 2 minutos", lambda: self.auto_select_requested.emit(120.0))
            sub_auto.addAction("Cada 5 minutos", lambda: self.auto_select_requested.emit(300.0))
            sub_auto.addAction("Cada 10 minutos", lambda: self.auto_select_requested.emit(600.0))
            menu.addSeparator()

        if self._zoom_range is not None:
            act_reset = menu.addAction("Restaurar vista completa (1.0x)")
            act_reset.triggered.connect(self.reset_zoom)
        else:
            act_full = menu.addAction("Vista completa (1.0x)")
            act_full.setEnabled(False)

        menu.exec(global_pos)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom temporal centrado en el cursor del ratón (zoom_level >= 1.0)."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        x_cursor = event.position().x()
        w = float(self.width())
        margin = 14.0
        t_cursor = self._x_to_time(x_cursor, w, margin)

        t_start, t_end = self._get_active_bounds()
        current_span = t_end - t_start
        full_span = max(1.0, self._max_time - self._min_time)

        factor = 0.8 if delta > 0 else 1.25
        new_span = current_span * factor

        # Límite mínimo: 3 segundos o 50x
        min_span = max(3.0, full_span / 50.0)
        new_span = max(min_span, min(full_span, new_span))

        if new_span >= full_span * 0.98:
            self._zoom_range = None
            self.zoom_changed.emit(False)
        else:
            ratio = (t_cursor - t_start) / max(0.001, current_span)
            new_start = t_cursor - (ratio * new_span)
            new_end = new_start + new_span

            # Clamping
            if new_start < self._min_time:
                new_start = self._min_time
                new_end = min(self._max_time, new_start + new_span)
            elif new_end > self._max_time:
                new_end = self._max_time
                new_start = max(self._min_time, new_end - new_span)

            self._zoom_range = (new_start, new_end)
            self.zoom_changed.emit(True)

        self.update()
        event.accept()

    def _handle_scrub(self, x: float) -> None:
        t = self._x_to_time(x, float(self.width()), 14.0)
        idx = self._find_closest_frame(t)
        if idx != self.current_idx:
            self.current_idx = idx
            self.frame_selected.emit(self.current_idx)
            self.update()

    def _update_hover_tooltip(self, event: QMouseEvent) -> None:
        """Muestra tooltip con información del fotograma, fase o discontinuidad bajo el cursor."""
        x = event.position().x()
        y_pos = event.position().y()
        w = float(self.width())
        margin = 14.0
        t = self._x_to_time(x, w, margin)

        # 1. Banda superior de fases (Eclipse)
        if y_pos <= 18.0 and self.mode == "eclipse":
            contacts = dict(self._eclipse_contacts) if self._eclipse_contacts else self._auto_estimate_contacts()
            c1 = contacts.get("C1", self._min_time)
            c2 = contacts.get("C2")
            c3 = contacts.get("C3")
            c4 = contacts.get("C4", self._max_time)

            if t < c1:
                tip = "<b>Antes de C1</b><br>Sesión previa al inicio del eclipse."
            elif c2 and c3 and c1 <= t < c2:
                tip = "🟦 <b>Fase Parcial C1-C2 (Entrada)</b><br>La Luna cubre progresivamente el disco solar."
            elif c2 and c3 and abs(t - c2) <= 15:
                tip = "💎 <b>Segundo Contacto (C2)</b><br>Anillo de Diamante y Perlas de Baily de entrada."
            elif c2 and c3 and c2 <= t <= c3:
                tip = "🟪 <b>Totalidad (C2 a C3)</b><br>Disco solar 100% oculto. Corona solar y cromosfera visibles."
            elif c2 and c3 and abs(t - c3) <= 15:
                tip = "💎 <b>Tercer Contacto (C3)</b><br>Perlas de Baily y Anillo de Diamante de salida."
            elif c2 and c3 and c3 < t <= c4:
                tip = "🟧 <b>Fase Parcial C3-C4 (Salida)</b><br>La Luna se retira progresivamente del disco solar."
            elif not (c2 and c3) and t <= (c1 + c4) / 2.0:
                tip = "🟦 <b>Fase Parcial C1-MAX (Entrada)</b><br>Creciente hacia el máximo del eclipse."
            elif not (c2 and c3) and t > (c1 + c4) / 2.0:
                tip = "🟧 <b>Fase Parcial MAX-C4 (Salida)</b><br>Decreciente hacia el fin del eclipse."
            else:
                tip = "<b>Fin del eclipse (Post-C4)</b>"
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
            return

        # 2. Comprobar si está sobre un Gap
        for g in self._gaps:
            x1 = self._time_to_x(g["from_t"], w, margin)
            x2 = self._time_to_x(g["to_t"], w, margin)
            if x1 <= x <= x2:
                mins = g["gap_seconds"] / 60.0
                f_from = os.path.basename(g["from_path"])
                f_to = os.path.basename(g["to_path"])
                tip = (
                    f"⚠️ <b>Hueco de {g['gap_seconds']:.1f}s</b> ({mins:.1f} min)<br>"
                    f"Entre: <i>{f_from}</i> y <i>{f_to}</i><br>"
                    f"Tomas faltantes estimadas: ~<b>{g['missing_frames_est']}</b><br>"
                    f"<i>Haz clic para saltar al fotograma previo al corte</i>"
                )
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                return

        # 3. Comprobar toma cercana
        idx = self._find_closest_frame(t)
        if 0 <= idx < len(self._items):
            it = self._items[idx]
            fname = os.path.basename(it["path"])
            if self.mode == "eclipse":
                status = "<font color='#94a3b8'>DISPONIBLE</font>" if it["is_excluded"] else "<font color='#38bdf8'>EN COLLAGE</font>"
                p_text = it.get("phase_label")
                phase_line = f"<br>Fase: <b>{p_text}</b>" if p_text and p_text != "Pendiente" else ""
            elif self.mode == "meteors":
                status = "<font color='#64748b'>DESCARTADA</font>" if it["is_excluded"] else "<font color='#10b981'><b>METEORO / CONSERVAR</b></font>"
                phase_line = ""
            elif self.mode == "startrails":
                status = "<font color='#f59e0b'>DESCARTADA (Discontinuidad)</font>" if it["is_excluded"] else "<font color='#38bdf8'>INCLUIDA</font>"
                phase_line = ""
            else:
                status = "<font color='#ef4444'>DESCARTADA</font>" if it["is_excluded"] else "<font color='#4ade80'>INCLUIDA</font>"
                phase_line = ""

            exp_val = it.get("exp")
            if exp_val and exp_val > 0:
                if exp_val < 0.3:
                    inv = round(1.0 / exp_val)
                    exp_str = f"1/{inv}s"
                elif exp_val < 1.0:
                    exp_str = f"{exp_val:.2f}s"
                else:
                    exp_str = f"{exp_val:.1f}s"
                exp_line = f"<br>Exposición: <b>{exp_str}</b>"
            else:
                exp_line = ""

            hdr_line = ""
            if self.mode == "eclipse" and any(it["idx"] in b.get("indices", set()) for b in self._hdr_bursts):
                hdr_line = "<br><font color='#818cf8'>⚡ Ráfaga HDR (Bracketing)</font>"

            dt_label = it["datetime_str"] or it["time_str"]
            tip = (
                f"<b>#{idx + 1}: {fname}</b> ({status}){phase_line}{exp_line}{hdr_line}<br>"
                f"Hora: {dt_label}<br>"
                f"<i>Arrastra para explorar la serie temporal</i>"
            )
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)


# Alias para máxima retrocompatibilidad y ergonomía
BlinkTimelineWidget = SessionTimelineWidget

__all__ = ["SessionTimelineWidget", "BlinkTimelineWidget"]
