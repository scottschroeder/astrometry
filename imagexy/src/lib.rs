#![allow(dead_code)] // TODO remove

use image::{GenericImage, GenericImageView, Pixel};

use crate::image_extractor::ImageExtractor;

mod image_extractor;
mod lanczos;
mod loader;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Point {
        Point { x, y }
    }
}

impl<X: Into<f32>, Y: Into<f32>> From<(X, Y)> for Point {
    fn from((x, y): (X, Y)) -> Self {
        Point {
            x: x.into(),
            y: y.into(),
        }
    }
}

pub fn demo(p: &str) -> anyhow::Result<()> {
    let mut d = loader::load_from_path(p)?;
    log::info!("img: {:?}x{:?}", d.width(), d.height());
    let mut g = image::imageops::grayscale(&d.view(0, 0, d.width(), d.height()));
    // let g = d.grayscale();
    log::info!("img: {:?}", g.dimensions());
    g.save("out.jpg")?;
    let extractor = ImageExtractor {
        invert: false,
        dpsf: 1.0,
        // lanczos_window: Some(2.5),
        ..Default::default()
    };
    let s = extractor.run(&mut g)?;

    for star in s {
        let x = (star.point.x).floor() as i32;
        let y = (star.point.y).floor() as i32;

        let hair = 1u32;

        let ilo = std::cmp::max(0, x - hair as i32) as u32;
        let ihi = std::cmp::min(d.width() - 1, x as u32 + 1 + hair);
        let jlo = std::cmp::max(0, y - hair as i32) as u32;
        let jhi = std::cmp::min(d.height() - 1, y as u32 + 1 + hair);

        for ix in ilo..ihi {
            d.put_pixel(ix, y as u32, image::Rgba::from_channels(0, 255, 0, 255));
        }
        for iy in jlo..jhi {
            d.put_pixel(x as u32, iy, image::Rgba::from_channels(0, 255, 0, 255));
        }

        // let p = d.get_pixel_mut(x, y);
        // p.0[1] = 255;
    }

    d.save("stars.png")?;

    Ok(())
}

pub fn gaussian(p: &str, x: u32, y: u32) -> anyhow::Result<()> {
    let d = loader::load_from_path(p)?;
    log::info!("img: {:?}x{:?}", d.width(), d.height());
    let g = image::imageops::grayscale(&d.view(0, 0, d.width(), d.height()));
    let p = image_extractor::max_gaussian(&g, 1.0, x, y);
    log::info!("final point: {:?}", p);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
