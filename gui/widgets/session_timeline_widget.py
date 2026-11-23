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
from PySide6.QtWidgets import QWidget, QToolTip
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
    """
    frame_selected = Signal(int)             # Índice del fotograma seleccionado (0..N-1)
    gap_selected = Signal(int, int)          # Salto detectado (from_idx, to_idx)
    zoom_changed = Signal(bool)              # True si está con zoom activo, False si vista completa

    def __init__(self, mode: str = "general", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mode = mode.lower()  # "general", "startrails", "meteors", "eclipse"
        self.setFixedHeight(72)
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
        self._min_time: float = 0.0
        self._max_time: float = 1.0

        # Zoom y Pan temporal
        self._zoom_range: Optional[Tuple[float, float]] = None
        self._is_dragging: bool = False
        self._drag_start_x: float = 0.0

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
        updated = False
        for it in self._items:
            if it["path"] == path:
                if ts is not None and isinstance(ts, (int, float)) and ts > 0:
                    it["t"] = float(ts)
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
            self.update()

    def set_eclipse_contacts(self, contacts: Dict[str, float]) -> None:
        """Configura los instantes clave del eclipse (C1, C2, MAX, C3, C4 en epoch timestamp)."""
        self._eclipse_contacts = dict(contacts)
        self.update()

    def get_gaps(self) -> List[Dict[str, Any]]:
        """Retorna la lista de discontinuidades temporales detectadas."""
        return list(self._gaps)

    def jump_to_next_gap(self) -> bool:
        """Salta a la siguiente discontinuidad posterior a la toma actual."""
        if not self._gaps or not self.files_list:
            return False
        for g in self._gaps:
            if g["from_idx"] > self.current_idx:
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
        for g in reversed(self._gaps):
            if g["from_idx"] < self.current_idx:
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
                    "t": float(i),
                    "time_str": meta.get("time_str", f"#{i+1}"),
                    "datetime_str": meta.get("datetime_str", ""),
                    "is_excluded": (p in self.excluded_files),
                    "is_ref": (i == self.ref_idx)
                })
            self._min_time = 0.0
            self._max_time = max(1.0, float(len(self.files_list) - 1))
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

        if self.mode == "eclipse" and not self._eclipse_contacts:
            self._eclipse_contacts = self._auto_estimate_contacts()

        # Detección de discontinuidades (Gaps)
        deltas = []
        for i in range(len(self._items) - 1):
            dt = self._items[i + 1]["t"] - self._items[i]["t"]
            if dt > 0:
                deltas.append(dt)

        if len(deltas) >= 3:
            s_deltas = sorted(deltas)
            median_dt = s_deltas[len(s_deltas) // 2]
            # Un gap se define si el salto excede 2.5x la mediana y supera al menos 60 segundos
            threshold = max(60.0, median_dt * 2.5)

            for i in range(len(self._items) - 1):
                dt = self._items[i + 1]["t"] - self._items[i]["t"]
                if dt > threshold:
                    self._gaps.append({
                        "from_idx": i,
                        "to_idx": i + 1,
                        "from_t": self._items[i]["t"],
                        "to_t": self._items[i + 1]["t"],
                        "gap_seconds": dt,
                        "from_path": self._items[i]["path"],
                        "to_path": self._items[i + 1]["path"],
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

        # 1. Fondo contenedor
        painter.fillRect(QRectF(0, 0, w, h), QColor(BG_CANVAS))
        pen_border = QPen(QColor(BORDER_BASE), 1)
        painter.setPen(pen_border)
        painter.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 5, 5)

        if not self._items:
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "Sin tomas cargadas para la línea de tiempo")
            return

        t_start, t_end = self._get_active_bounds()

        # 2. BANDA SUPERIOR: Fases o Cobertura de Sesión (Y: 4..16)
        y_band_top = 4.0
        h_band = 13.0
        self._draw_phase_band(painter, y_band_top, h_band, w, margin)

        # 3. DISCONTINUIDADES / GAPS TEMPORALES (Franja sombreada vertical)
        self._draw_gaps(painter, y_band_top, h - 22.0, w, margin)

        # 4. PISTA DE TOMAS (Línea central y marcadores ♦/•) (Y: 28..46)
        y_track = 36.0
        painter.setPen(QPen(QColor("#27272a"), 1, Qt.DashLine))
        painter.drawLine(QPointF(margin, y_track), QPointF(w - margin, y_track))

        self._draw_frame_markers(painter, y_track, w, margin)

        # 5. REGLA HORARIA INFERIOR (Marcas de tiempo C1..C4 o reloj continuo) (Y: 52..68)
        self._draw_time_ruler(painter, h - 18.0, w, margin, t_start, t_end)

        # 6. AGUJA DE TIEMPO (PLAYHEAD) DEL FOTOGRAMA ACTIVO
        self._draw_playhead(painter, h, w, margin)

    def _draw_phase_band(self, painter: QPainter, y: float, h: float, w: float, margin: float) -> None:
        """Dibuja la franja superior de fases (Eclipse) o cobertura de sesión (Startrails/Meteoros)."""
        usable_w = w - (2 * margin)

        if self.mode == "eclipse":
            contacts = dict(self._eclipse_contacts) if self._eclipse_contacts else self._auto_estimate_contacts()
            c1 = contacts.get("C1", self._min_time)
            c2 = contacts.get("C2")
            c_max = contacts.get("MAX")
            c3 = contacts.get("C3")
            c4 = contacts.get("C4", self._max_time)

            # Fondo general no eclipsado
            painter.fillRect(QRectF(margin, y, usable_w, h), QColor("#18181b"))

            x_c1 = max(margin, min(w - margin, self._time_to_x(c1, w, margin)))
            x_c4 = max(margin, min(w - margin, self._time_to_x(c4, w, margin)))

            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))

            if c2 and c3 and c2 < c3:
                x_c2 = max(margin, min(w - margin, self._time_to_x(c2, w, margin)))
                x_c3 = max(margin, min(w - margin, self._time_to_x(c3, w, margin)))

                # 1. Fase Parcial C1 a C2 (Entrada) - Azul Cobalto (#0284c7)
                w_in = max(1.0, x_c2 - x_c1)
                rect_in = QRectF(x_c1, y, w_in, h)
                painter.fillRect(rect_in, QColor("#0284c7"))
                if w_in > 48:
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(rect_in, Qt.AlignCenter, "C1-C2 ENTRADA")

                # 2. Contacto C2 (Diamante / Perlas C2) - Línea dorada (#fbbf24)
                painter.fillRect(QRectF(x_c2 - 1.5, y, 3.0, h), QColor("#fbbf24"))

                # 3. Totalidad C2 a C3 (Corona Solar) - Púrpura profundo (#9333ea)
                w_tot = max(2.0, x_c3 - x_c2)
                rect_tot = QRectF(x_c2, y, w_tot, h)
                painter.fillRect(rect_tot, QColor("#9333ea"))
                if w_tot > 44:
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(rect_tot, Qt.AlignCenter, "TOTALIDAD")

                # 4. Contacto C3 (Perlas / Diamante C3) - Línea dorada (#fbbf24)
                painter.fillRect(QRectF(x_c3 - 1.5, y, 3.0, h), QColor("#fbbf24"))

                # 5. Fase Parcial C3 a C4 (Salida) - Naranja Solar (#ea580c)
                w_out = max(1.0, x_c4 - x_c3)
                rect_out = QRectF(x_c3, y, w_out, h)
                painter.fillRect(rect_out, QColor("#ea580c"))
                if w_out > 48:
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(rect_out, Qt.AlignCenter, "C3-C4 SALIDA")
            else:
                # Sin totalidad definida (eclipse parcial)
                t_mid = c_max if c_max else (c1 + c4) / 2.0
                x_mid = max(margin, min(w - margin, self._time_to_x(t_mid, w, margin)))

                # Entrada (C1 a Mid) - Azul (#0284c7)
                w_in = max(1.0, x_mid - x_c1)
                rect_in = QRectF(x_c1, y, w_in, h)
                painter.fillRect(rect_in, QColor("#0284c7"))
                if w_in > 48:
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(rect_in, Qt.AlignCenter, "C1-MAX ENTRADA")

                # Máximo
                painter.fillRect(QRectF(x_mid - 1.5, y, 3.0, h), QColor("#f59e0b"))

                # Salida (Mid a C4) - Naranja (#ea580c)
                w_out = max(1.0, x_c4 - x_mid)
                rect_out = QRectF(x_mid, y, w_out, h)
                painter.fillRect(rect_out, QColor("#ea580c"))
                if w_out > 48:
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(rect_out, Qt.AlignCenter, "MAX-C4 SALIDA")

            # Borde sutil de la banda
            painter.setPen(QColor("#334155"))
            painter.drawRect(QRectF(margin, y, usable_w, h))
        else:
            # Modo Astrofotografía General / Startrails: Barra de Cobertura Nocturna
            grad = QLinearGradient(margin, y, w - margin, y)
            grad.setColorAt(0.0, QColor("#1e293b"))
            grad.setColorAt(0.5, QColor("#0f172a"))
            grad.setColorAt(1.0, QColor("#1e293b"))
            painter.fillRect(QRectF(margin, y, usable_w, h), grad)

            # Borde sutil de la banda
            painter.setPen(QColor("#334155"))
            painter.drawRect(QRectF(margin, y, usable_w, h))

    def _draw_gaps(self, painter: QPainter, y_top: float, h_total: float, w: float, margin: float) -> None:
        """Dibuja las zonas de discontinuidad temporal en rojo translúcido con icono de advertencia."""
        for g in self._gaps:
            x1 = self._time_to_x(g["from_t"], w, margin)
            x2 = self._time_to_x(g["to_t"], w, margin)
            gap_w = max(4.0, x2 - x1)

            # Fondo rayado o translúcido rojo
            gap_rect = QRectF(x1, y_top, gap_w, h_total)
            painter.fillRect(gap_rect, QColor(220, 38, 38, 35))

            # Bordes delimitadores verticales del salto
            painter.setPen(QPen(QColor(ACCENT_RED), 1, Qt.DashLine))
            painter.drawLine(QPointF(x1, y_top), QPointF(x1, y_top + h_total))
            painter.drawLine(QPointF(x2, y_top), QPointF(x2, y_top + h_total))

            # Si el gap es suficientemente ancho, pintar texto explicativo
            dt_min = g["gap_seconds"] / 60.0
            gap_label = f"⚠️ GAP {dt_min:.0f}m" if dt_min >= 1.0 else f"⚠️ GAP {g['gap_seconds']:.0f}s"
            if gap_w > 42.0:
                painter.setPen(QColor("#fca5a5"))
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                painter.drawText(gap_rect, Qt.AlignCenter, gap_label)

    def _draw_frame_markers(self, painter: QPainter, y: float, w: float, margin: float) -> None:
        """Dibuja los puntos/rombos (♦/•) de cada toma."""
        t_start, t_end = self._get_active_bounds()

        for it in self._items:
            t = it["t"]
            if t < t_start or t > t_end:
                continue

            x = self._time_to_x(t, w, margin)
            is_cur = (it["idx"] == self.current_idx)
            is_ref = it.get("is_ref", False)
            is_exc = it.get("is_excluded", False)

            if is_exc:
                if self.mode == "eclipse":
                    # En modo eclipse, una toma no seleccionada para el collage es una toma válida disponible
                    color = QColor("#64748b") if not is_cur else QColor("#ffffff")
                    painter.setPen(color)
                    painter.setBrush(color)
                    painter.drawEllipse(QPointF(x, y), 2.0, 2.0)
                else:
                    # Toma descartada en cribado: cruz o punto rojo
                    painter.setPen(QColor("#ef4444"))
                    painter.setBrush(QColor("#ef4444"))
                    painter.drawEllipse(QPointF(x, y), 2.0, 2.0)
            elif is_ref:
                # Toma de referencia: Rombo dorado (♦)
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
                # Toma activa: en eclipse es una toma seleccionada para el collage (Cian brillante)
                if self.mode == "eclipse":
                    p_label = it.get("phase_label", "").upper()
                    if "DIAMANTE" in p_label or "BAILY" in p_label:
                        # Fenómeno de contacto singular: Rombo dorado brillante (♦)
                        painter.setPen(QColor("#f59e0b"))
                        painter.setBrush(QColor("#fbbf24"))
                        poly = QPolygonF([
                            QPointF(x, y - 4.5),
                            QPointF(x + 3.5, y),
                            QPointF(x, y + 4.5),
                            QPointF(x - 3.5, y)
                        ])
                        painter.drawPolygon(poly)
                    elif "TOTALIDAD" in p_label:
                        painter.setPen(QColor("#ffffff") if is_cur else QColor("#c084fc"))
                        painter.setBrush(QColor("#9333ea"))
                        painter.drawEllipse(QPointF(x, y), 3.0, 3.0)
                    else:
                        color = QColor("#38bdf8") if not is_cur else QColor("#ffffff")
                        painter.setPen(color)
                        painter.setBrush(color)
                        painter.drawEllipse(QPointF(x, y), 2.8, 2.8)
                else:
                    color = QColor(ACCENT_COBALT) if not is_cur else QColor("#ffffff")
                    painter.setPen(color)
                    painter.setBrush(color)
                    painter.drawEllipse(QPointF(x, y), 2.2, 2.2)

    def _draw_time_ruler(self, painter: QPainter, y: float, w: float, margin: float, t_start: float, t_end: float) -> None:
        """Dibuja la escala horaria continua en la parte inferior con hitos C1..C4 en eclipse."""
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(TEXT_MUTED))

        # Determinar número de divisiones legibles según ancho
        num_ticks = max(3, int((w - 2 * margin) / 85.0))
        span = t_end - t_start

        for i in range(num_ticks + 1):
            ratio = i / float(num_ticks)
            t = t_start + (ratio * span)
            x = margin + (ratio * (w - 2 * margin))

            # Marca de graduación
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawLine(QPointF(x, y), QPointF(x, y + 3))

            # Formatear hora (si es epoch datetime, convertir; si es sintético, mostrar relativo)
            try:
                dt = datetime.fromtimestamp(t)
                time_lbl = dt.strftime("%H:%M")
            except Exception:
                mins = int(t / 60.0)
                time_lbl = f"+{mins}m"

            painter.drawText(QRectF(x - 25, y + 3, 50, 12), Qt.AlignCenter, time_lbl)

        # Si estamos en modo eclipse, superponer los hitos clave C1..C4 con sus colores
        if self.mode == "eclipse":
            contacts = dict(self._eclipse_contacts) if self._eclipse_contacts else self._auto_estimate_contacts()
            c_colors = {
                "C1": QColor("#38bdf8"),
                "C2": QColor("#fbbf24"),
                "MAX": QColor("#c084fc"),
                "C3": QColor("#fbbf24"),
                "C4": QColor("#fb923c")
            }
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            for c_name, c_t in contacts.items():
                if t_start <= c_t <= t_end:
                    cx = self._time_to_x(c_t, w, margin)
                    col = c_colors.get(c_name, QColor("#ffffff"))
                    painter.setPen(col)
                    painter.drawText(QRectF(cx - 15, y - 8, 30, 10), Qt.AlignCenter, c_name)

    def _draw_playhead(self, painter: QPainter, h: float, w: float, margin: float) -> None:
        """Dibuja la aguja de tiempo del fotograma actual."""
        if not self._items or not (0 <= self.current_idx < len(self._items)):
            return

        cur_t = self._items[self.current_idx]["t"]
        x = self._time_to_x(cur_t, w, margin)

        # Línea vertical brillante con brillo cyan
        painter.setPen(QPen(QColor("#38bdf8"), 1.5))
        painter.drawLine(QPointF(x, 2.0), QPointF(x, h - 16.0))

        # Cabeza indicadora triangular superior
        tri = QPolygonF([
            QPointF(x - 4.0, 2.0),
            QPointF(x + 4.0, 2.0),
            QPointF(x, 7.0)
        ])
        painter.setBrush(QColor("#38bdf8"))
        painter.drawPolygon(tri)

        # Halo en el punto de la toma activa
        y_track = 36.0
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QColor("#0284c7"))
        painter.drawEllipse(QPointF(x, y_track), 4.5, 4.5)

    # =========================================================================
    # EVENTOS DE RATÓN E INTERACCIÓN
    # =========================================================================

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._handle_scrub(event.position().x())
        elif event.button() == Qt.RightButton:
            # Clic derecho restaura zoom
            self.reset_zoom()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        if self._is_dragging:
            self._handle_scrub(x)
        else:
            self._update_hover_tooltip(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._is_dragging = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Soporte de zoom temporal continuo con Ctrl + Rueda."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            t_center = self._x_to_time(event.position().x(), float(self.width()), 14.0)
            t_start, t_end = self._get_active_bounds()
            current_span = t_end - t_start

            factor = 0.75 if delta > 0 else 1.33
            new_span = max(10.0, min(self._max_time - self._min_time, current_span * factor))

            ratio = (t_center - t_start) / max(0.001, current_span)
            new_start = max(self._min_time, t_center - (ratio * new_span))
            new_end = min(self._max_time, new_start + new_span)

            if new_end - new_start >= (self._max_time - self._min_time) * 0.98:
                self._zoom_range = None
                self.zoom_changed.emit(False)
            else:
                self._zoom_range = (new_start, new_end)
                self.zoom_changed.emit(True)

            self.update()
            event.accept()
        else:
            super().wheelEvent(event)

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

        # 1. Comprobar si está sobre la banda superior de fases
        if y_pos <= 20.0 and self.mode == "eclipse":
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
                    f"⚠️ <b>DISCONTINUIDAD TEMPORAL (GAP)</b><br>"
                    f"Duración: <b>{mins:.1f} min</b> ({g['gap_seconds']:.0f}s)<br>"
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
            else:
                status = "<font color='#ef4444'>DESCARTADA</font>" if it["is_excluded"] else "<font color='#4ade80'>INCLUIDA</font>"
                phase_line = ""
            dt_label = it["datetime_str"] or it["time_str"]
            tip = (
                f"<b>#{idx + 1}: {fname}</b> ({status}){phase_line}<br>"
                f"Hora: {dt_label}<br>"
                f"<i>Arrastra para explorar la serie temporal</i>"
            )
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
