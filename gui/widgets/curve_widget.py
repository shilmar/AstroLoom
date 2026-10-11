"""
gui/curve_widget.py - Widget interactivo de curvas tonales con histograma logarítmico.
Genera tablas de consulta (LUT) de alta precisión con interpolación monótona PCHIP.
"""
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Resolución fina de la LUT (1024 pasos elimina el banding en 16-bit / 32-bit)
LUT_RESOLUTION = 256


class CurveWidget(QWidget):
    """
    Controlador gráfico de curva de transferencia tonal.
    Permite modelar curvas de contraste en espacio continuo con visualización
    de histograma astronómico logarítmico en segundo plano e indicadores
    interactivos de recorte (clipping) de sombras y luces estilo Lightroom.
    """
    curveChanged = Signal()
    clippingChanged = Signal(bool, bool)  # (clip_shadows, clip_highlights)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(220, 180)
        self.setFixedHeight(190)
        self.setMouseTracking(True)

        # Puntos de control normalizados [(x, y)] en rango [0.0, 1.0]
        self.points: List[Tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self.selected_idx: Optional[int] = None
        self.dragging: bool = False

        # LUT normalizada float32 [0.0, 1.0]
        self.lut: np.ndarray = np.linspace(0.0, 1.0, LUT_RESOLUTION, dtype=np.float32)

        # Histograma logarítmico de luminancia normalizado (256 bins para pintado rápido)
        self.hist_data: Optional[np.ndarray] = None

        # Estado de los indicadores de recorte de sombras y luces
        self.clip_shadows_active: bool = False
        self.clip_highlights_active: bool = False
        self.has_shadow_clip: bool = False
        self.has_highlight_clip: bool = False

    def reset_curve(self) -> None:
        """Restablece la curva a la función identidad f(x) = x."""
        self.points = [(0.0, 0.0), (1.0, 1.0)]
        self.selected_idx = None
        self._update_lut()
        self.update()
        self.curveChanged.emit()

    def set_histogram_from_image(self, img_rgb: Optional[np.ndarray]) -> None:
        """
        Calcula el histograma de luminancia con compresión logarítmica
        a partir de un submuestreo de la imagen suministrada usando OpenCV.
        Detecta si existen valores extremos recortados (sombras a 0 o luces a 255).
        """
        if img_rgb is None or img_rgb.size == 0:
            self.hist_data = None
            self.has_shadow_clip = False
            self.has_highlight_clip = False
            self.update()
            return

        # Submuestreo uniforme con paso 4 para cálculo directo en C++
        sample = img_rgb[::4, ::4]
        if sample.dtype != np.float32:
            sample = sample.astype(np.float32)

        # Conversión rápida de luminancia ITU-R BT.709
        gray = cv2.cvtColor(np.clip(sample, 0.0, 1.0), cv2.COLOR_RGB2GRAY)
        u8 = (gray * 255.0).astype(np.uint8)

        hist = cv2.calcHist([u8], [0], None, [256], [0, 256]).ravel()

        # Detección de presencia de píxeles recortados en el histograma
        self.has_shadow_clip = bool(hist[0] > 0)
        self.has_highlight_clip = bool(hist[255] > 0)

        hist_log = np.log1p(hist)
        max_v = float(np.max(hist_log))
        if max_v > 1e-5:
            self.hist_data = (hist_log / max_v).astype(np.float32)
        else:
            self.hist_data = None
        self.update()

    def toggle_shadow_clipping(self) -> None:
        """Alterna el indicador de recorte de sombras (negros empastados)."""
        self.clip_shadows_active = not self.clip_shadows_active
        self.clippingChanged.emit(self.clip_shadows_active, self.clip_highlights_active)
        self.update()

    def toggle_highlight_clipping(self) -> None:
        """Alterna el indicador de recorte de luces (blancos quemados)."""
        self.clip_highlights_active = not self.clip_highlights_active
        self.clippingChanged.emit(self.clip_shadows_active, self.clip_highlights_active)
        self.update()

    def toggle_both_clipping(self) -> None:
        """Alterna simultáneamente los indicadores de sombras y luces (atajo J)."""
        new_state = not (self.clip_shadows_active or self.clip_highlights_active)
        self.clip_shadows_active = new_state
        self.clip_highlights_active = new_state
        self.clippingChanged.emit(self.clip_shadows_active, self.clip_highlights_active)
        self.update()

    def set_clipping_state(self, shadows: bool, highlights: bool) -> None:
        """Establece explícitamente el estado de los indicadores de recorte."""
        if self.clip_shadows_active != shadows or self.clip_highlights_active != highlights:
            self.clip_shadows_active = shadows
            self.clip_highlights_active = highlights
            self.clippingChanged.emit(self.clip_shadows_active, self.clip_highlights_active)
            self.update()

    def _update_lut(self) -> None:
        """Calcula la LUT interpolada garantizando monotonicidad estricta en X."""
        pts = sorted(self.points, key=lambda p: p[0])
        
        # Eliminar duplicados o nodos infinitesimalmente cercanos
        clean_pts = []
        for p in pts:
            if not clean_pts or abs(p[0] - clean_pts[-1][0]) > 1e-4:
                clean_pts.append(p)

        xs = [p[0] for p in clean_pts]
        ys = [p[1] for p in clean_pts]

        if xs[0] > 0.001:
            xs.insert(0, 0.0)
            ys.insert(0, ys[0])
        else:
            xs[0] = 0.0

        if xs[-1] < 0.999:
            xs.append(1.0)
            ys.append(ys[-1])
        else:
            xs[-1] = 1.0

        grid_x = np.linspace(0.0, 1.0, LUT_RESOLUTION, dtype=np.float32)

        if len(xs) == 2 or not HAS_SCIPY:
            self.lut = np.interp(grid_x, xs, ys).astype(np.float32)
        else:
            try:
                pchip = PchipInterpolator(xs, ys)
                self.lut = np.clip(pchip(grid_x), 0.0, 1.0).astype(np.float32)
            except Exception:
                self.lut = np.interp(grid_x, xs, ys).astype(np.float32)

        self.lut = np.clip(self.lut, 0.0, 1.0)

    def get_lut(self) -> np.ndarray:
        """Retorna la Look-Up Table calculada."""
        return self.lut

    def is_identity(self) -> bool:
        """Comprueba si la curva actual equivale a la transformación nula."""
        return len(self.points) == 2 and self.points[0] == (0.0, 0.0) and self.points[1] == (1.0, 1.0)

    # -------------------------------------------------------------
    # Renderizado Gráfico
    # -------------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 12
        inner_w = w - 2 * margin
        inner_h = h - 2 * margin

        # Fondo del recuadro
        painter.fillRect(margin, margin, inner_w, inner_h, QColor("#1e1e24"))

        # Cuadrícula en cuartos
        grid_pen = QPen(QColor("#2f343f"), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            x = margin + inner_w * (i / 4.0)
            y = margin + inner_h * (i / 4.0)
            painter.drawLine(int(x), margin, int(x), margin + inner_h)
            painter.drawLine(margin, int(y), margin + inner_w, int(y))

        # Diagonal neutra
        painter.setPen(QPen(QColor("#3a3f4b"), 1, Qt.DotLine))
        painter.drawLine(margin, margin + inner_h, margin + inner_w, margin)

        # Histograma de luminancia
        if self.hist_data is not None:
            painter.setBrush(QBrush(QColor(100, 140, 180, 55)))
            painter.setPen(Qt.NoPen)
            hist_path = QPainterPath()
            hist_path.moveTo(margin, margin + inner_h)
            for i, val in enumerate(self.hist_data):
                px = margin + (i / 255.0) * inner_w
                py = margin + inner_h - (val * inner_h * 0.90)
                hist_path.lineTo(px, py)
            hist_path.lineTo(margin + inner_w, margin + inner_h)
            hist_path.closeSubpath()
            painter.drawPath(hist_path)

        # Curva interpolada
        curve_pen = QPen(QColor("#88c0d0"), 2)
        painter.setPen(curve_pen)
        painter.setBrush(Qt.NoBrush)
        curve_path = QPainterPath()
        
        # Muestreo a pantalla regular
        step = max(1, len(self.lut) // 256)
        for i in range(0, len(self.lut), step):
            val = self.lut[i]
            px = margin + (i / float(len(self.lut) - 1)) * inner_w
            py = margin + inner_h - (val * inner_h)
            if i == 0:
                curve_path.moveTo(px, py)
            else:
                curve_path.lineTo(px, py)
        painter.drawPath(curve_path)

        # Nodos de control
        for i, (px, py) in enumerate(self.points):
            cx = margin + px * inner_w
            cy = margin + inner_h - py * inner_h
            if i == self.selected_idx:
                painter.setBrush(QBrush(QColor("#ebcb8b")))
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawEllipse(QPointF(cx, cy), 5.0, 5.0)
            else:
                painter.setBrush(QBrush(QColor("#eceff4")))
                painter.setPen(QPen(QColor("#2e3440"), 1))
                painter.drawEllipse(QPointF(cx, cy), 4.0, 4.0)

        # Marco exterior
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#4c566a"), 1.5))
        painter.drawRect(margin, margin, inner_w, inner_h)

        # --- Botones Indicadores de Recorte (Lightroom / Camera Raw style) ---
        # 1. Indicador de Sombras (Esquina Superior Izquierda)
        tl_x = margin + 3
        tl_y = margin + 3
        tl_w, tl_h = 16, 14
        tl_box = QRectF(tl_x, tl_y, tl_w, tl_h)

        if self.clip_shadows_active:
            bg_sh = QColor(10, 36, 68, 230)
            border_sh = QColor(0, 112, 255)
            icon_sh = QColor(0, 160, 255)
        elif self.has_shadow_clip:
            bg_sh = QColor(20, 26, 36, 180)
            border_sh = QColor(40, 80, 130)
            icon_sh = QColor(64, 140, 220)
        else:
            bg_sh = QColor(25, 28, 34, 160)
            border_sh = QColor(55, 62, 74)
            icon_sh = QColor(85, 95, 110)

        painter.setBrush(QBrush(bg_sh))
        painter.setPen(QPen(border_sh, 1))
        painter.drawRoundedRect(tl_box, 2.0, 2.0)

        tri_sh = QPainterPath()
        tri_sh.moveTo(tl_x + 8.0, tl_y + 3.0)
        tri_sh.lineTo(tl_x + 3.5, tl_y + 11.0)
        tri_sh.lineTo(tl_x + 12.5, tl_y + 11.0)
        tri_sh.closeSubpath()
        painter.setBrush(QBrush(icon_sh))
        painter.setPen(Qt.NoPen)
        painter.drawPath(tri_sh)

        # 2. Indicador de Luces (Esquina Superior Derecha)
        tr_x = margin + inner_w - 19
        tr_y = margin + 3
        tr_w, tr_h = 16, 14
        tr_box = QRectF(tr_x, tr_y, tr_w, tr_h)

        if self.clip_highlights_active:
            bg_hl = QColor(68, 16, 26, 230)
            border_hl = QColor(255, 23, 68)
            icon_hl = QColor(255, 64, 96)
        elif self.has_highlight_clip:
            bg_hl = QColor(36, 20, 26, 180)
            border_hl = QColor(130, 40, 60)
            icon_hl = QColor(220, 64, 90)
        else:
            bg_hl = QColor(25, 28, 34, 160)
            border_hl = QColor(55, 62, 74)
            icon_hl = QColor(85, 95, 110)

        painter.setBrush(QBrush(bg_hl))
        painter.setPen(QPen(border_hl, 1))
        painter.drawRoundedRect(tr_box, 2.0, 2.0)

        tri_hl = QPainterPath()
        tri_hl.moveTo(tr_x + 8.0, tr_y + 3.0)
        tri_hl.lineTo(tr_x + 3.5, tr_y + 11.0)
        tri_hl.lineTo(tr_x + 12.5, tr_y + 11.0)
        tri_hl.closeSubpath()
        painter.setBrush(QBrush(icon_hl))
        painter.setPen(Qt.NoPen)
        painter.drawPath(tri_hl)

    # -------------------------------------------------------------
    # Gestión de Eventos de Ratón
    # -------------------------------------------------------------
    def _to_norm(self, pos) -> Tuple[float, float]:
        margin = 12
        inner_w = max(1, self.width() - 2 * margin)
        inner_h = max(1, self.height() - 2 * margin)
        nx = np.clip((pos.x() - margin) / inner_w, 0.0, 1.0)
        ny = np.clip(1.0 - (pos.y() - margin) / inner_h, 0.0, 1.0)
        return float(nx), float(ny)

    def mousePressEvent(self, event) -> None:
        pos = event.position().toPoint()
        margin = 12
        inner_w = self.width() - 2 * margin
        inner_h = self.height() - 2 * margin

        tl_rect = QRect(margin + 1, margin + 1, 20, 18)
        tr_rect = QRect(margin + inner_w - 21, margin + 1, 20, 18)

        if event.button() == Qt.LeftButton:
            if tl_rect.contains(pos):
                self.toggle_shadow_clipping()
                return
            elif tr_rect.contains(pos):
                self.toggle_highlight_clipping()
                return

        # Si el usuario hace clic derecho en el botón de recorte, ignorar para proteger puntos
        if tl_rect.contains(pos) or tr_rect.contains(pos):
            return

        mx, my = self._to_norm(pos)

        clicked_idx = None
        for i, (px, py) in enumerate(self.points):
            cx = margin + px * inner_w
            cy = margin + inner_h - py * inner_h
            dist = np.hypot(pos.x() - cx, pos.y() - cy)
            if dist <= 9.0:
                clicked_idx = i
                break

        if event.button() == Qt.RightButton:
            if clicked_idx is not None and len(self.points) > 2:
                # Proteger los anclajes absolutos de inicio y fin
                if clicked_idx not in (0, len(self.points) - 1):
                    del self.points[clicked_idx]
                    self.selected_idx = None
                    self._update_lut()
                    self.update()
                    self.curveChanged.emit()
            return

        if event.button() == Qt.LeftButton:
            if clicked_idx is not None:
                self.selected_idx = clicked_idx
            else:
                new_pt = (mx, my)
                self.points.append(new_pt)
                self.points.sort(key=lambda p: p[0])
                self.selected_idx = self.points.index(new_pt)
            
            self.dragging = True
            self._update_lut()
            self.update()
            self.curveChanged.emit()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        margin = 12
        inner_w = self.width() - 2 * margin
        tl_rect = QRect(margin + 1, margin + 1, 20, 18)
        tr_rect = QRect(margin + inner_w - 21, margin + 1, 20, 18)

        if not self.dragging:
            if tl_rect.contains(pos):
                state_str = "Activo" if self.clip_shadows_active else "Inactivo"
                clip_status = " (Recorte detectado)" if self.has_shadow_clip else ""
                self.setToolTip(f"Aviso de Recorte de Sombras (Azul) [{state_str}]{clip_status}\nClic para alternar (Atajo: J)")
                self.setCursor(Qt.PointingHandCursor)
            elif tr_rect.contains(pos):
                state_str = "Activo" if self.clip_highlights_active else "Inactivo"
                clip_status = " (Recorte detectado)" if self.has_highlight_clip else ""
                self.setToolTip(f"Aviso de Recorte de Luces (Rojo) [{state_str}]{clip_status}\nClic para alternar (Atajo: J)")
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setToolTip("Curva de Tonos e Histograma Logarítmico")
                self.setCursor(Qt.ArrowCursor)

        if self.dragging and self.selected_idx is not None:
            mx, my = self._to_norm(pos)

            idx = self.selected_idx
            if idx == 0:
                mx = 0.0
            elif idx == len(self.points) - 1:
                mx = 1.0
            else:
                min_x = self.points[idx - 1][0] + 0.01 if idx > 0 else 0.0
                max_x = self.points[idx + 1][0] - 0.01 if idx < len(self.points) - 1 else 1.0
                mx = max(min_x, min(max_x, mx))

            self.points[idx] = (mx, my)
            self._update_lut()
            self.update()
            self.curveChanged.emit()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def leaveEvent(self, event) -> None:
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)