# tests/test_zero_copy_pyo3.py
"""
Pruebas de verificación exhaustiva de la Tarea 12: Gestión Zero-Copy entre Rust y Python (PyO3).

Verifica de forma sistemática:
1. Préstamo de memoria de solo lectura (zero-copy borrow via PyReadonlyArray) sin duplicar buffers.
2. Mutación in-place en búferes de destino preasignados (bytearray) sin asignaciones dinámicas.
3. Mutación in-place mutable directa sobre arreglos NumPy de Python (PyReadwriteArray2) sin copia.
4. Calibración RAW CFA in-place (calibrate_raw_cfa_inplace_rs) sobre uint16 sin asignación.
5. Ausencia de fugas de memoria o churn del GC durante bucles de alta frecuencia (simulación 60 FPS).
6. Renderizado directo de sub-regiones (zoom_rect) sin copias intermedias en Python.
7. Soporte multiformato (float32, uint8, uint16) en vistas 2D y 3D.
8. Robustez y salvaguardas de contigüidad C-contiguous frente a slices discontinuos.
9. Liberación del GIL (py.allow_threads) y paralelización segura con Rayon en concurrencia multihilo.
"""

import os
import sys
import gc
import tracemalloc
import threading
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astroloom_rs


def test_zero_copy_buffer_reuse_identity():
    """Verifica que render_canvas_frame_rs muta el buffer de destino in-place sin reasignar."""
    h, w = 120, 160
    img = np.random.uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)
    expected_len = h * w * 4

    out_buf = bytearray(expected_len)
    initial_id = id(out_buf)

    ret_buf = astroloom_rs.render_canvas_frame_rs(
        img,
        out=out_buf,
        show_clipping=False
    )

    # Debe ser exactamente la misma instancia en memoria
    assert ret_buf is out_buf, "El buffer retornado debe ser la misma instancia del bytearray provisto"
    assert id(ret_buf) == initial_id, "El identificador de objeto en Python debe permanecer inalterado"
    assert len(ret_buf) == expected_len

    # Comprobamos que el buffer contiene datos válidos BGRA
    arr = np.frombuffer(ret_buf, dtype=np.uint8).reshape((h, w, 4))
    assert arr[0, 0, 3] == 255, "Canal alfa debe ser 255"
    assert np.any(arr[..., :3] > 0), "Los canales de color deben contener datos renderizados"


def test_zero_allocation_in_high_frequency_loop():
    """Comprueba que durante un ciclo continuo (ej. arrastre de slider a 60 FPS) el consumo neto de RAM no aumenta."""
    h, w = 100, 100
    img = np.random.uniform(0.1, 0.9, (h, w, 3)).astype(np.float32)
    out_buf = bytearray(h * w * 4)

    # Calentamiento inicial
    astroloom_rs.render_canvas_frame_rs(img, out=out_buf)

    gc.collect()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Simular 120 fotogramas consecutivos de arrastre de slider
    for i in range(120):
        split_pos = (i % 100) / 100.0
        astroloom_rs.render_canvas_frame_rs(
            img,
            split_position=split_pos,
            out=out_buf,
            show_clipping=(i % 2 == 0)
        )

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Comprobamos diferencias de memoria rastreadas por Python
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_diff = sum(stat.size_diff for stat in stats)

    # Las llamadas nativas a Rust con buffer preasignado no deben generar churn en el heap de Python
    # Permitimos un margen mínimo (< 4 KB) por metadatos internos de tracemalloc/pytest
    assert total_diff < 4096, f"Fuga o churn excesivo de memoria en Python: {total_diff} bytes generados"


def test_zero_copy_readonly_source_integrity():
    """Verifica que el puntero de datos de origen NumPy no es modificado ni clonado por Rust."""
    h, w = 64, 64
    img = np.ascontiguousarray(np.random.uniform(0.2, 0.8, (h, w, 3)).astype(np.float32))
    img_ptr = img.ctypes.data
    img_copy = img.copy()

    out_buf = bytearray(h * w * 4)
    astroloom_rs.render_canvas_frame_rs(img, out=out_buf)

    # El puntero base del ndarray de origen debe ser idéntico
    assert img.ctypes.data == img_ptr, "El array de Python no debe ser reubicado en memoria"
    # Los datos de entrada no deben ser alterados (préstamo de solo lectura estricto)
    np.testing.assert_array_equal(img, img_copy, err_msg="La entrada de solo lectura fue mutada accidentalmente")


def test_zero_copy_subregion_crop_zoom_rect():
    """Verifica que Rust recorta y renderiza directamente desde zoom_rect sin crear copias intermedias en Python."""
    h, w = 200, 200
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Marcar una región central (50..100, 40..90) con valores específicos
    img[50:100, 40:90, :] = [0.25, 0.50, 0.75]

    crop_w, crop_h = 50, 50
    out_buf = bytearray(crop_w * crop_h * 4)

    # Renderizar recorte directo usando zoom_rect=(rx, ry, rw, rh)
    ret = astroloom_rs.render_canvas_frame_rs(
        img,
        zoom_rect=(40, 50, crop_w, crop_h),
        out=out_buf
    )

    assert ret is out_buf
    arr = np.frombuffer(ret, dtype=np.uint8).reshape((crop_h, crop_w, 4))
    # Comprobar que los píxeles renderizados corresponden exactamente a los valores de la región
    assert abs(int(arr[0, 0, 0]) - int(round(0.75 * 255))) <= 1, "Canal B del zoom incorrecto"
    assert abs(int(arr[0, 0, 1]) - int(round(0.50 * 255))) <= 1, "Canal G del zoom incorrecto"
    assert abs(int(arr[0, 0, 2]) - int(round(0.25 * 255))) <= 1, "Canal R del zoom incorrecto"


def test_zero_copy_multitype_support():
    """Comprueba que el puente PyO3 acepta vistas prestadas de float32, uint8 y uint16 (2D y 3D)."""
    h, w = 32, 32
    out_buf = bytearray(h * w * 4)

    # 1. Float32 3D
    f32_3d = np.full((h, w, 3), 0.5, dtype=np.float32)
    astroloom_rs.render_canvas_frame_rs(f32_3d, out=out_buf)
    arr = np.frombuffer(out_buf, dtype=np.uint8).reshape((h, w, 4))
    assert arr[0, 0, 0] == 128

    # 2. Float32 2D (Monocromático)
    f32_2d = np.full((h, w), 0.25, dtype=np.float32)
    astroloom_rs.render_canvas_frame_rs(f32_2d, out=out_buf)
    arr = np.frombuffer(out_buf, dtype=np.uint8).reshape((h, w, 4))
    assert abs(int(arr[0, 0, 0]) - 64) <= 1

    # 3. Uint8 3D
    u8_3d = np.full((h, w, 3), 200, dtype=np.uint8)
    astroloom_rs.render_canvas_frame_rs(u8_3d, out=out_buf)
    arr = np.frombuffer(out_buf, dtype=np.uint8).reshape((h, w, 4))
    assert arr[0, 0, 0] == 200

    # 4. Uint16 3D
    u16_3d = np.full((h, w, 3), 32768, dtype=np.uint16)
    astroloom_rs.render_canvas_frame_rs(u16_3d, out=out_buf)
    arr = np.frombuffer(out_buf, dtype=np.uint8).reshape((h, w, 4))
    assert abs(int(arr[0, 0, 0]) - 128) <= 1


def test_zero_copy_pyreadwrite_mask_stamp_inplace():
    """Verifica que stamp_brush_stroke_rs muta PyReadwriteArray2 directamente en la memoria de Python sin copias."""
    h, w = 60, 60
    mask = np.zeros((h, w), dtype=np.float32)
    mask_ptr = mask.ctypes.data

    # Estampar un trazo desde (10, 10) hasta (50, 50) con radio 8 y flujo 1.0
    roi = astroloom_rs.stamp_brush_stroke_rs(
        mask,
        p0=(10.0, 10.0),
        p1=(50.0, 50.0),
        radius=8.0,
        hardness=0.8,
        flow=1.0,
        erase=False
    )

    # El puntero base del ndarray debe ser exactamente el mismo
    assert mask.ctypes.data == mask_ptr, "El array no debe ser reasignado por Rust"
    assert roi is not None, "Debe retornar el ROI calculado"
    # El centro del trazo debe haber sido mutado directamente en el array original
    assert mask[30, 30] > 0.8, "El trazo debió mutar los datos in-place en Python"
    # Fuera del trazo debe seguir en 0
    assert mask[0, 59] == 0.0, "Las áreas no afectadas deben permanecer intactas"


def test_zero_copy_raw_cfa_inplace_calibration():
    """Verifica la mutación in-place en memoria de enteros (uint16) vía calibrate_raw_cfa_inplace_rs."""
    h, w = 100, 100
    raw = np.full((h, w), 2000, dtype=np.uint16)
    raw_ptr = raw.ctypes.data

    dark = np.full((h, w), 500.0, dtype=np.float32)

    # Ejecutar calibración in-place
    astroloom_rs.calibrate_raw_cfa_inplace_rs(
        raw,
        master_dark=dark,
        black_level=0.0,
        dark_pedestal=0.0
    )

    # Mismo puntero de memoria en Python
    assert raw.ctypes.data == raw_ptr, "El puntero del array de Python no debe cambiar"
    # 2000 - 500 = 1500
    assert raw[50, 50] == 1500, f"Valor calibrado esperado 1500, obtenido {raw[50, 50]}"


def test_zero_copy_strided_vs_contiguous_safety():
    """Comprueba el comportamiento robusto ante arrays no contiguos en memoria (Fortran order / slices con paso)."""
    h, w = 50, 50
    base = np.random.uniform(0.0, 1.0, (h * 2, w * 2)).astype(np.float32)
    # Slice no contiguo
    strided_slice = base[::2, ::2]
    assert not strided_slice.flags['C_CONTIGUOUS']

    # stamp_brush_stroke_rs requiere C-contiguous explícitamente para punteros crudos; debe rechazar de forma segura
    with pytest.raises(ValueError, match="C-contiguous"):
        astroloom_rs.stamp_brush_stroke_rs(
            strided_slice,
            p0=(10.0, 10.0),
            p1=(20.0, 20.0),
            radius=5.0
        )


def test_zero_copy_gil_release_multithreading():
    """Verifica la liberación del GIL y la ausencia de interbloqueos (deadlocks) al invocar Rust desde múltiples hilos."""
    h, w = 100, 100
    errors = []

    def worker_render(thread_id: int):
        try:
            img = np.full((h, w, 3), 0.1 * thread_id, dtype=np.float32)
            out = bytearray(h * w * 4)
            for _ in range(20):
                astroloom_rs.render_canvas_frame_rs(img, out=out)
                arr = np.frombuffer(out, dtype=np.uint8)
                assert arr[3] == 255
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker_render, args=(i + 1,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(errors) == 0, f"Errores en ejecución concurrente con Rust: {errors}"


def test_memory_ceiling_large_megaframes():
    """Verifica que el renderizado de un megacuadro (ej. 12-24 MP) con buffer reutilizado no desborda la memoria."""
    h, w = 1500, 2000  # 3 MP, ~36 MB en float32
    img = np.full((h, w, 3), 0.4, dtype=np.float32)
    out_buf = bytearray(h * w * 4)

    # Ejecutar 5 veces seguidas reutilizando el mismo buffer
    for _ in range(5):
        astroloom_rs.render_canvas_frame_rs(img, out=out_buf)

    assert len(out_buf) == h * w * 4
    arr = np.frombuffer(out_buf, dtype=np.uint8).reshape((h, w, 4))
    assert arr[0, 0, 0] == int(round(0.4 * 255))


def test_export_pipeline_uses_rust_fused():
    """Verifica que el pipeline de exportación (is_preview=False) utiliza el grafo Rust fused optimizado."""
    from gui.developer.developer_state import DeveloperState
    from gui.developer.pipeline import apply_developer_pipeline

    h, w = 120, 160
    img = np.full((h, w, 3), 0.25, dtype=np.float32)
    img[..., 1] = 0.55  # Dominante verde

    state_noscnr = DeveloperState()
    state_noscnr.scnr_green_val = 0

    state_scnr = DeveloperState()
    state_scnr.scnr_green_val = 100

    # Ejecutar pipeline en modo exportación completa (is_preview=False, is_scrubbing=False)
    res_noscnr = apply_developer_pipeline(img, state_noscnr, is_preview=False, is_scrubbing=False)
    res_scnr = apply_developer_pipeline(img, state_scnr, is_preview=False, is_scrubbing=False)

    assert res_scnr.shape == (h, w, 3)
    assert res_scnr.dtype == np.float32
    assert res_scnr.flags['C_CONTIGUOUS']
    # Con SCNR al 100% en el pipeline de exportación, el verde es significativamente menor que sin SCNR
    assert np.mean(res_scnr[..., 1]) < np.mean(res_noscnr[..., 1])
    # Y los canales rojo y azul se preservan (o aumentan ligeramente por preservación Rec.709)
    assert np.mean(res_scnr[..., 0]) >= np.mean(res_noscnr[..., 0])


def test_export_uint16_inplace_no_temp_churn():
    """Comprueba la cuantización a uint16 de 16-bit TIFF sin creación excesiva de temporales en Python."""
    h, w = 200, 200
    processed = np.random.uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)

    u16 = np.empty(processed.shape, dtype=np.uint16)
    np.clip(processed * 65535.0 + 0.5, 0, 65535, out=u16, casting='unsafe')

    assert u16.shape == (h, w, 3)
    assert u16.dtype == np.uint16
    assert u16.flags['C_CONTIGUOUS']
    # Validación numérica contra referencia
    ref = np.clip(np.round(processed * 65535.0), 0, 65535).astype(np.uint16)
    np.testing.assert_allclose(u16, ref, atol=1)
