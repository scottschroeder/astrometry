use crate::Point;
use num_traits::NumCast;

use super::LumaImage;
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

pub(crate) struct ObjectImageDebugger<'a, T: Primitive + 'static> {
    inner: &'a LumaImage<T>,
    marks: Vec<(Point, BasicColor)>,
}

impl<'a, T: Primitive + 'static> ObjectImageDebugger<'a, T> {
    pub(crate) fn new(img: &'a LumaImage<T>) -> ObjectImageDebugger<'a, T> {
        ObjectImageDebugger {
            inner: img,
            marks: vec![],
        }
    }

    pub(crate) fn add_mark(&mut self, p: Point, c: BasicColor) {
        self.marks.push((p, c))
    }

    pub(crate) fn save(&self, name: &str) -> anyhow::Result<()> {
        let img = self.create_printable_image();

        img.save(format!("{}.png", name))?;
        Ok(())
    }

    fn create_printable_image(&self) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let (width, height) = self.inner.dimensions();
        let mut out = ImageBuffer::new(width, height);

        let crosses = self
            .marks
            .iter()
            .map(|(p, c)| Cross {
                x: (p.x + 0.5) as i32,
                y: (p.y + 0.5) as i32,
                color: *c,
                length: 0,
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
        } else if y == self.y && (x >= self.x - self.length && x <= self.x + self.length) {
            true
        } else {
            false
        }
    }
}
