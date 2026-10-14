"""
core/project_manager.py - Gestor de proyectos y sesiones de apilado (.mwstack).
Guarda metadatos en JSON y máscaras continuas de 16 bits asociadas.
"""
from typing import Any, Dict, Optional, Tuple
import json
import os
import cv2
import numpy as np
from core.config_manager import load_config


class ProjectManager:
    """Administra la serialización y deserialización de sesiones de trabajo de AstroLoom."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Restablece el estado interno a los valores iniciales de sesión."""
        self.project_path: Optional[str] = None
        self.data: Dict[str, Any] = {
            "version": "1.2",
            "mode": "fixed_tripod",
            "ground_mode_idx": 0,
            "mask_path": "",
            "light_frames": [],
            "dark_frames": [],
            "ground_frames": [],
            "parameters": {
                "kappa": 2.2,
                "lp_method_idx": 0,
                "lp_strength": 50
            }
        }

    def save(self, filepath: str, mask_array: Optional[np.ndarray] = None) -> bool:
        """
        Guarda la sesión en un archivo JSON (.mwstack).
        Si se suministra mask_array, la exporta como PNG de 16 bits en la subcarpeta
        de máscaras configurada ('masks_dir') o en 'mascaras/' relativa al proyecto.
        """
        self.project_path = os.path.abspath(filepath)
        project_dir = os.path.dirname(self.project_path)
        base_name = os.path.splitext(os.path.basename(self.project_path))[0]

        if mask_array is not None:
            # Determinamos la carpeta destino de las máscaras
            cfg = load_config()
            target_masks_dir = cfg.get("masks_dir", os.path.join(project_dir, "mascaras"))
            os.makedirs(target_masks_dir, exist_ok=True)

            mask_filename = f"{base_name}_mask.png"
            mask_full_path = os.path.join(target_masks_dir, mask_filename)

            u16_mask = (np.clip(mask_array, 0.0, 1.0) * 65535.0).astype(np.uint16)
            cv2.imwrite(mask_full_path, u16_mask)

            # Si la máscara está dentro de la misma jerarquía del proyecto, guardar ruta relativa
            try:
                rel_mask_path = os.path.relpath(mask_full_path, project_dir)
                self.data["mask_path"] = rel_mask_path.replace("\\", "/")
            except ValueError:
                self.data["mask_path"] = mask_full_path
        else:
            self.data["mask_path"] = ""

        with open(self.project_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        return True

    def load(self, filepath: str) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Carga la sesión desde el archivo indicado y restaura la máscara asociada si existe.
        
        Retorna:
            (data, mask_data): Diccionario de sesión y array float32 de la máscara (o None).
        """
        self.project_path = os.path.abspath(filepath)
        project_dir = os.path.dirname(self.project_path)

        with open(self.project_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        mask_data = None
        raw_mask_path = self.data.get("mask_path", "")

        if raw_mask_path:
            cfg = load_config()
            global_masks_dir = cfg.get("masks_dir", "")
            mask_basename = os.path.basename(raw_mask_path)

            # Búsqueda en orden de prioridad para máxima portabilidad
            potential_paths = [
                os.path.join(project_dir, raw_mask_path),                  # Ruta relativa guardada
                os.path.join(project_dir, "mascaras", mask_basename),       # Subcarpeta local máscaras
                os.path.join(project_dir, mask_basename),                   # Mismo directorio que la sesión
                os.path.join(global_masks_dir, mask_basename) if global_masks_dir else "", # Carpeta global
                raw_mask_path                                              # Ruta absoluta directa
            ]

            valid_path = next((p for p in potential_paths if p and os.path.exists(p)), None)

            if valid_path:
                raw = cv2.imread(valid_path, cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    if raw.ndim == 3:
                        raw = raw[..., 0]
                    max_v = 65535.0 if raw.dtype == np.uint16 else 255.0
                    mask_data = np.clip(raw.astype(np.float32) / max_v, 0.0, 1.0)

        return self.data, mask_data