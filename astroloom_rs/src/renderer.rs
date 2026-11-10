use numpy::ndarray::{ArrayView2, ArrayView3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use rayon::prelude::*;

// ============================================================================
// FASE 12: PUENTE DE RENDERIZADO DIRECTO A QT (gui/canvas.py)
// Renderizado nativo ultrarrápido a 60+ FPS sin copias intermedias en NumPy.
// Formato de salida: BGRA8888 continuo compatible con QImage.Format_RGB32 / Format_ARGB32.
// ============================================================================

enum ImageSource<'a> {
    F32_3D(PyReadonlyArray3<'a, f32>),
    F32_2D(PyReadonlyArray2<'a, f32>),
    U8_3D(PyReadonlyArray3<'a, u8>),
    U8_2D(PyReadonlyArray2<'a, u8>),
    U16_3D(PyReadonlyArray3<'a, u16>),
    U16_2D(PyReadonlyArray2<'a, u16>),
}

impl<'a> ImageSource<'a> {
    #[inline(always)]
    fn dim(&self) -> (usize, usize) {
        match self {
            ImageSource::F32_3D(a) => {
                let d = a.as_array().dim();
                (d.0, d.1)
            }
            ImageSource::F32_2D(a) => {
                let d = a.as_array().dim();
                (d.0, d.1)
            }
            ImageSource::U8_3D(a) => {
                let d = a.as_array().dim();
                (d.0, d.1)
            }
            ImageSource::U8_2D(a) => {
                let d = a.as_array().dim();
                (d.0, d.1)
            }
            ImageSource::U16_3D(a) => {
                let d = a.as_array().dim();
                (d.0, d.1)
            }
            ImageSource::U16_2D(a) => {
                let d = a.as_array().dim();
                (d.0, d.1)
            }
        }
    }

    #[inline(always)]
    fn to_view(&'a self) -> ImageView<'a> {
        match self {
            ImageSource::F32_3D(a) => ImageView::F32_3D(a.as_array()),
            ImageSource::F32_2D(a) => ImageView::F32_2D(a.as_array()),
            ImageSource::U8_3D(a) => ImageView::U8_3D(a.as_array()),
            ImageSource::U8_2D(a) => ImageView::U8_2D(a.as_array()),
            ImageSource::U16_3D(a) => ImageView::U16_3D(a.as_array()),
            ImageSource::U16_2D(a) => ImageView::U16_2D(a.as_array()),
        }
    }
}

enum ImageView<'a> {
    F32_3D(ArrayView3<'a, f32>),
    F32_2D(ArrayView2<'a, f32>),
    U8_3D(ArrayView3<'a, u8>),
    U8_2D(ArrayView2<'a, u8>),
    U16_3D(ArrayView3<'a, u16>),
    U16_2D(ArrayView2<'a, u16>),
}

impl<'a> ImageView<'a> {
    #[inline(always)]
    fn get_pixel_u8(&self, y: usize, x: usize, lut: Option<&[u8]>) -> (u8, u8, u8) {
        match self {
            ImageView::F32_3D(v) => {
                let c = v.dim().2;
                let r = v[[y, x, 0]];
                let g = if c >= 3 { v[[y, x, 1]] } else { r };
                let b = if c >= 3 { v[[y, x, 2]] } else { r };
                (map_f32(r, lut), map_f32(g, lut), map_f32(b, lut))
            }
            ImageView::F32_2D(v) => {
                let val = map_f32(v[[y, x]], lut);
                (val, val, val)
            }
            ImageView::U8_3D(v) => {
                let c = v.dim().2;
                let r = v[[y, x, 0]];
                let g = if c >= 3 { v[[y, x, 1]] } else { r };
                let b = if c >= 3 { v[[y, x, 2]] } else { r };
                (map_u8(r, lut), map_u8(g, lut), map_u8(b, lut))
            }
            ImageView::U8_2D(v) => {
                let val = map_u8(v[[y, x]], lut);
                (val, val, val)
            }
            ImageView::U16_3D(v) => {
                let c = v.dim().2;
                let r = v[[y, x, 0]];
                let g = if c >= 3 { v[[y, x, 1]] } else { r };
                let b = if c >= 3 { v[[y, x, 2]] } else { r };
                (map_u16(r, lut), map_u16(g, lut), map_u16(b, lut))
            }
            ImageView::U16_2D(v) => {
                let val = map_u16(v[[y, x]], lut);
                (val, val, val)
            }
        }
    }
}

fn extract_image_source<'py>(obj: &'py Bound<'py, PyAny>) -> PyResult<ImageSource<'py>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray3<f32>>() {
        Ok(ImageSource::F32_3D(a))
    } else if let Ok(a) = obj.extract::<PyReadonlyArray2<f32>>() {
        Ok(ImageSource::F32_2D(a))
    } else if let Ok(a) = obj.extract::<PyReadonlyArray3<u8>>() {
        Ok(ImageSource::U8_3D(a))
    } else if let Ok(a) = obj.extract::<PyReadonlyArray2<u8>>() {
        Ok(ImageSource::U8_2D(a))
    } else if let Ok(a) = obj.extract::<PyReadonlyArray3<u16>>() {
        Ok(ImageSource::U16_3D(a))
    } else if let Ok(a) = obj.extract::<PyReadonlyArray2<u16>>() {
        Ok(ImageSource::U16_2D(a))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input image must be a 2D or 3D numpy array of float32, uint8, or uint16",
        ))
    }
}

#[inline(always)]
fn map_f32(v: f32, lut: Option<&[u8]>) -> u8 {
    if let Some(lut_slice) = lut {
        let n = lut_slice.len();
        if n > 0 {
            let max_idx = (n - 1) as f32;
            let idx = (v.clamp(0.0, 1.0) * max_idx + 0.5) as usize;
            return lut_slice[idx.min(n - 1)];
        }
    }
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline(always)]
fn map_u8(v: u8, lut: Option<&[u8]>) -> u8 {
    if let Some(lut_slice) = lut {
        if lut_slice.len() == 256 {
            return lut_slice[v as usize];
        } else if !lut_slice.is_empty() {
            let n = lut_slice.len();
            let idx = (v as usize * (n - 1) + 127) / 255;
            return lut_slice[idx.min(n - 1)];
        }
    }
    v
}

#[inline(always)]
fn map_u16(v: u16, lut: Option<&[u8]>) -> u8 {
    if let Some(lut_slice) = lut {
        if lut_slice.len() == 65536 {
            return lut_slice[v as usize];
        } else if !lut_slice.is_empty() {
            let n = lut_slice.len();
            let idx = (v as usize * (n - 1) + 32767) / 65535;
            return lut_slice[idx.min(n - 1)];
        }
    }
    (v >> 8) as u8
}

#[inline(always)]
fn write_pixel(
    r: u8,
    g: u8,
    b: u8,
    clip_shadows: bool,
    clip_highlights: bool,
    out: &mut [u8],
) {
    if clip_shadows && r == 0 && g == 0 && b == 0 {
        // Sombras empastadas: #0070ff (B=255, G=112, R=0, A=255)
        out[0] = 255;
        out[1] = 112;
        out[2] = 0;
        out[3] = 255;
    } else if clip_highlights && (r == 255 || g == 255 || b == 255) {
        // Luces quemadas: #ff1744 (B=68, G=23, R=255, A=255)
        out[0] = 68;
        out[1] = 23;
        out[2] = 255;
        out[3] = 255;
    } else {
        // Color estándar BGRA
        out[0] = b;
        out[1] = g;
        out[2] = r;
        out[3] = 255;
    }
}

/// Renderiza un fotograma de lienzo directamente a un buffer contiguo BGRA8888
/// (compatible nativamente con QImage.Format_RGB32 / Format_ARGB32) sin conversiones NumPy.
#[pyfunction]
#[pyo3(signature = (
    primary_img,
    secondary_img=None,
    split_position=0.5,
    split_orientation="vertical",
    show_clipping=false,
    show_shadows=None,
    show_highlights=None,
    lut=None,
    lut_secondary=None,
    out=None,
    zoom_rect=None,
    is_vertical=None
))]
pub fn render_canvas_frame_rs<'py>(
    py: Python<'py>,
    primary_img: &Bound<'py, PyAny>,
    secondary_img: Option<&Bound<'py, PyAny>>,
    split_position: f32,
    split_orientation: &str,
    show_clipping: bool,
    show_shadows: Option<bool>,
    show_highlights: Option<bool>,
    lut: Option<PyReadonlyArray1<'py, u8>>,
    lut_secondary: Option<PyReadonlyArray1<'py, u8>>,
    out: Option<&Bound<'py, PyByteArray>>,
    zoom_rect: Option<(usize, usize, usize, usize)>,
    is_vertical: Option<bool>,
) -> PyResult<Bound<'py, PyByteArray>> {
    let p_src = extract_image_source(primary_img)?;
    let (h, w) = p_src.dim();

    if h == 0 || w == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Primary image dimensions must be greater than zero",
        ));
    }

    let (zx, zy, out_w, out_h) = if let Some((rx, ry, rw, rh)) = zoom_rect {
        if rw == 0 || rh == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "zoom_rect width and height must be greater than zero",
            ));
        }
        if rx + rw > w || ry + rh > h {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "zoom_rect ({}, {}, {}, {}) exceeds image dimensions ({}, {})",
                rx, ry, rw, rh, w, h
            )));
        }
        (rx, ry, rw, rh)
    } else {
        (0, 0, w, h)
    };

    let s_src = if let Some(sec) = secondary_img {
        let s = extract_image_source(sec)?;
        let (sh, sw) = s.dim();
        if sh != h || sw != w {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Secondary image dimensions ({}, {}) do not match primary ({}, {})",
                sw, sh, w, h
            )));
        }
        Some(s)
    } else {
        None
    };

    let p_view = p_src.to_view();
    let s_view = s_src.as_ref().map(|s| s.to_view());

    let clip_shadows = show_shadows.unwrap_or(show_clipping);
    let clip_highlights = show_highlights.unwrap_or(show_clipping);

    let lut_slice = lut.as_ref().map(|l| l.as_slice()).transpose()?;
    let lut_sec_slice = lut_secondary
        .as_ref()
        .map(|l| l.as_slice())
        .transpose()?
        .or(lut_slice);

    let is_horizontal = if let Some(v) = is_vertical {
        !v
    } else {
        split_orientation.eq_ignore_ascii_case("horizontal")
    };
    let split_pos = split_position.clamp(0.0, 1.0);
    let split_x = ((split_pos * w as f32).round() as usize).min(w);
    let split_y = ((split_pos * h as f32).round() as usize).min(h);

    let total_bytes = out_h * out_w * 4;

    let process_buffer = |dest: &mut [u8]| {
        dest.par_chunks_exact_mut(out_w * 4)
            .enumerate()
            .for_each(|(y_rel, row_bytes)| {
                let y = zy + y_rel;
                let is_sec_row = s_view.is_some() && is_horizontal && y < split_y;

                for x_rel in 0..out_w {
                    let x = zx + x_rel;
                    let px_idx = x_rel * 4;
                    let px_out = &mut row_bytes[px_idx..px_idx + 4];

                    let is_sec_col = s_view.is_some() && !is_horizontal && x < split_x;

                    if is_sec_row || is_sec_col {
                        // Zona "Antes": imagen secundaria sin alertas de recorte
                        let (r, g, b) = s_view
                            .as_ref()
                            .unwrap()
                            .get_pixel_u8(y, x, lut_sec_slice);
                        px_out[0] = b;
                        px_out[1] = g;
                        px_out[2] = r;
                        px_out[3] = 255;
                    } else {
                        // Zona "Después" / Imagen principal: con soporte de alertas de recorte
                        let (r, g, b) = p_view.get_pixel_u8(y, x, lut_slice);
                        write_pixel(r, g, b, clip_shadows, clip_highlights, px_out);
                    }
                }
            });
    };

    let byte_array = match out {
        Some(ba) if ba.len() >= total_bytes => {
            let ba_slice = unsafe { ba.as_bytes_mut() };
            py.allow_threads(|| {
                process_buffer(&mut ba_slice[..total_bytes]);
            });
            ba.clone()
        }
        _ => {
            let mut buf = vec![0u8; total_bytes];
            py.allow_threads(|| {
                process_buffer(&mut buf);
            });
            PyByteArray::new(py, &buf)
        }
    };

    Ok(byte_array)
}
