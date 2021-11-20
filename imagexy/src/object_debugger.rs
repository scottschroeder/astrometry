use crate::Point;
use num_traits::NumCast;

use crate::image_extractor::LumaImage;
use image::{ImageBuffer, Pixel, Primitive, Rgba};

#[derive(Debug, Clone, Copy)]
pub(crate) enum BasicColor {
    Red,
    Green,
    Blue,
}

impl BasicColor {
    fn rgba(self) -> Rgba<u8> {
        match self {
            BasicColor::Red => Rgba::from_channels(255, 0, 0, 255),
            BasicColor::Green => Rgba::from_channels(0, 255, 0, 255),
            BasicColor::Blue => Rgba::from_channels(0, 0, 255, 255),
        }
    }
}

pub(crate) struct ImageDebugger<'a, T: Primitive + 'static> {
    inner: &'a LumaImage<T>,
    marks: Vec<(Point, BasicColor)>,
    cross_size: Option<usize>,
}

impl<'a, T: Primitive + 'static> ImageDebugger<'a, T> {
    pub(crate) fn new(img: &'a LumaImage<T>) -> ImageDebugger<'a, T> {
        ImageDebugger {
            inner: img,
            marks: vec![],
            cross_size: None,
        }
    }

    pub(crate) fn add_mark(&mut self, p: Point, c: BasicColor) {
        self.marks.push((p, c))
    }

    pub(crate) fn set_cross_size(&mut self, size: usize) {
        self.cross_size = Some(size)
    }

    pub(crate) fn save<P: AsRef<std::path::Path>>(&self, name: P) -> anyhow::Result<()> {
        let p = name.as_ref();
        let p = if p.extension().is_none() {
            p.with_extension("png")
        } else {
            p.to_owned()
        };
        let img = self.create_printable_image();
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        img.save(p)?;
        Ok(())
    }

    fn auto_size_cross(&self) -> usize {
        let scale = std::cmp::min(self.inner.height(), self.inner.height()) as usize;
        scale / 100
    }

    fn create_printable_image(&self) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let (width, height) = self.inner.dimensions();
        let mut out = ImageBuffer::new(width, height);

        let cross_size = self.cross_size.unwrap_or_else(|| self.auto_size_cross());
        let crosses = self
            .marks
            .iter()
            .map(|(p, c)| Cross {
                x: (p.x + 0.5) as i32,
                y: (p.y + 0.5) as i32,
                color: *c,
                length: cross_size as i32,
            })
            .collect::<Vec<_>>();

        for y in 0..height {
            for x in 0..width {
                let mark = crosses
                    .iter()
                    .filter(|c| c.is_cross(x as i32, y as i32))
                    .map(|c| c.color)
                    .next();
                let p = if let Some(c) = mark {
                    c.rgba()
                } else {
                    let p = self.inner.get_pixel(x, y).0[0];
                    let flux: u8 = NumCast::from(p).unwrap_or(255);
                    Rgba::from_channels(flux, flux, flux, 255)
                };
                out.put_pixel(x, y, p);
            }
        }
        out
    }
}

struct Cross {
    x: i32,
    y: i32,
    color: BasicColor,
    length: i32,
}

impl Cross {
    fn is_cross(&self, x: i32, y: i32) -> bool {
        if x == self.x && (y >= self.y - self.length && y <= self.y + self.length) {
            true
        } else {
            y == self.y && (x >= self.x - self.length && x <= self.x + self.length)
        }
    }
}
