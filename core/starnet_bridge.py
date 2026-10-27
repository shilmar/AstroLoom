# core/starnet_bridge.py
"""
core/starnet_bridge.py - Puente de invocación para StarNetv2 CLI en modo lineal.
"""
from typing import Callable, Optional, Tuple
import os
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import tifffile

from core.config_manager import get_config_val


def find_starnet_executable(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Localiza el ejecutable de StarNet (starnet2.exe o starnet++.exe) por orden de prioridad:
    1. Ruta personalizada pasada por parámetro.
    2. Ruta guardada en la configuración global.
    3. Carpetas locales del proyecto o directorio raíz.
    4. PATH del sistema operativo.
    """
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    saved_path = get_config_val("starnet_exe", "")
    if saved_path and os.path.isfile(saved_path):
        return os.path.abspath(saved_path)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(root, "starnet++", "starnet++.exe"),
        os.path.join(root, "starnet2", "starnet2.exe"),
        os.path.join(root, "starnet", "starnet++.exe"),
        os.path.join(root, "starnet++.exe"),
        os.path.join(root, "starnet2.exe"),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    which_path = shutil.which("starnet2") or shutil.which("starnet++")
    if which_path:
        return os.path.abspath(which_path)

    return None


def run_starnet(
    img_rgb: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    stride: int = 256,
    starnet_exe: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta StarNetv2 en modo LINEAL sobre img_rgb (float32 [0.0, 1.0]).
    
    Prioriza inferencia nativa DirectML ONNX (sin binarios ni I/O a disco).
    Si se especifica un binario CLI personalizado o no está disponible ONNX,
    utiliza el ejecutable de StarNet.
    Retorna:
        starless_rgb (np.ndarray): Imagen sin estrellas con el suelo intacto.
        stars_rgb (np.ndarray): Capa aislada de estrellas en espacio lineal.
    """
    from core.starnet_onnx import is_starnet_onnx_available, run_starnet_onnx

    # 1. Prioridad: Inferencia nativa ONNX DirectML
    if not (starnet_exe and os.path.isfile(starnet_exe)) and is_starnet_onnx_available():
        return run_starnet_onnx(
            img_rgb=img_rgb,
            sky_mask=sky_mask,
            stride=stride,
            is_linear=True,
            progress_callback=log_callback,
            cancel_check=cancel_check
        )

    # 2. Fallback: Ejecutable CLI externo
    exe = find_starnet_executable(starnet_exe)
    if not exe or not os.path.isfile(exe):
        raise FileNotFoundError(
            "No se encontró ni el modelo ONNX de StarNet ('models/starnet2.onnx') "
            "ni el ejecutable de StarNet++ ('starnet2.exe')."
        )

    def log(msg: str):
        if log_callback:
            log_callback(msg)

    starnet_dir = os.path.dirname(exe)
    temp_dir = tempfile.mkdtemp(prefix="starnet_run_")

    try:
        h, w = img_rgb.shape[:2]
        log(f"Preparando fotograma lineal para StarNet++ ({w}x{h} px)...")

        # 1. Proteger suelo con tono neutro del fondo del cielo
        input_data = np.clip(img_rgb.copy(), 0.0, 1.0)
        mask_3d = None
        if sky_mask is not None:
            if sky_mask.shape[:2] != (h, w):
                mask_aligned = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_aligned = sky_mask

            sky_pixels = input_data[mask_aligned > 0.5]
            bg_color = np.median(sky_pixels, axis=0) if len(sky_pixels) > 0 else np.array([0.005, 0.005, 0.005], dtype=np.float32)

            mask_3d = mask_aligned[..., np.newaxis]
            input_data = (input_data * mask_3d) + (bg_color * (1.0 - mask_3d))

        in_tif = os.path.join(temp_dir, "starnet_in.tif")
        out_tif = os.path.join(temp_dir, "starnet_out.tif")

        # Guardar en TIFF 16-bit
        u16_input = (np.clip(input_data, 0.0, 1.0) * 65535.0).astype(np.uint16)
        tifffile.imwrite(in_tif, u16_input, photometric='rgb')

        cmd = [
            exe,
            "-i", in_tif,
            "-o", out_tif,
            "-s", str(stride),
            "--linear"
        ]
        log(f"Ejecutando StarNetv2 con flag --linear (stride={stride})...")

        proc = subprocess.Popen(
            cmd,
            cwd=starnet_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False
        )

        if proc.stdout:
            for line in proc.stdout:
                txt = line.strip()
                if txt:
                    log(f"[StarNet] {txt}")

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"StarNet finalizó con código de error {proc.returncode}")

        if not os.path.exists(out_tif):
            raise FileNotFoundError("StarNet concluyó pero no generó el archivo de salida.")

        log("Cargando resultado lineal y descomponiendo capas...")
        raw_out = tifffile.imread(out_tif).astype(np.float32) / 65535.0
        if raw_out.ndim == 2:
            raw_out = np.stack([raw_out] * 3, axis=-1)

        # 2. Restaurar suelo real intacto en la capa Starless
        if mask_3d is not None:
            starless_final = (raw_out * mask_3d) + (img_rgb * (1.0 - mask_3d))
        else:
            starless_final = raw_out

        starless_final = np.clip(starless_final, 0.0, 1.0).astype(np.float32)

        # 3. Capa Estrellas = max(0.0, Original - Starless)
        stars_only = np.clip(img_rgb - starless_final, 0.0, 1.0).astype(np.float32)

        log("Desacoplo de estrellas completado limpiamente.")
        return starless_final, stars_only

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)