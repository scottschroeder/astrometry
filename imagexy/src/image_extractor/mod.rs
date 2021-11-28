pub(crate) use self::gaussian::max_gaussian;
use crate::{image_extractor::peaks::dallpeaks, lanczos::lanczos_resample, ImageDebugger, Point};
use anyhow::Result;
use image::{ImageBuffer, Luma, Pixel, Primitive};
use num_traits::NumCast;
use std::{collections::HashMap, fmt::Debug, path::PathBuf};
use thiserror::Error;

mod flatten;
mod gaussian;
mod label;
mod peaks;

const SIMPLEXY_DEFAULT_DPSF: f64 = 1.0;
const SIMPLEXY_DEFAULT_PLIM: f64 = 8.0;
const SIMPLEXY_DEFAULT_DLIM: f64 = 1.0;
const SIMPLEXY_DEFAULT_SADDLE: f64 = 5.0;
const SIMPLEXY_DEFAULT_MAXPER: u32 = 1000;
const SIMPLEXY_DEFAULT_MAXSIZE: u32 = 2000;
const SIMPLEXY_DEFAULT_HALFBOX: u32 = 100;
const SIMPLEXY_DEFAULT_MAXNPEAKS: usize = 100000;

const SIMPLEXY_U8_DEFAULT_PLIM: f64 = 4.0;
const SIMPLEXY_U8_DEFAULT_SADDLE: f64 = 2.0;
const DSIGMA_DEFAULT_GRIDSIZE: u32 = 20;

#[derive(Error, Debug)]
pub enum ExtractionError {
    #[error(
        "no significant pixels, significace threshold = {limit}, max value in image = {max_value}"
    )]
    NoSignificantPixels { limit: f64, max_value: f64 },
    #[error("ran out of labels for objects")]
    ExhaustedLabels,
    #[error("unable to find peak")]
    NoPeak,
    #[error("peak value was outside of possible range")]
    PeakOutsideRange,
    #[error("peak was NaN or Inf")]
    PeakBadFloat,
}

#[derive(Debug, Default)]
pub struct ImageDebugConfig {
    pub background_subtracted_image: Option<PathBuf>,
    pub smooth_background_subtracted_image: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(crate) struct Source {
    pub point: Point,
    pub flux: f64,
    pub background: f64,
}

/// Settings for how the star xy points should be extracted from
/// an image.
pub struct ImageExtractor {
    /// gaussian psf width (sigma, not FWHM)
    pub dpsf: f64,
    /// significance to keep
    pub plim: f64,
    /// closest two peaks can be
    pub dlim: f64,
    /// saddle difference (in sig)
    pub saddle: f64,
    /// maximum number of peaks per object
    pub maxper: u32,
    /// maximum number of peaks total
    pub maxnpeaks: usize,
    /// maximum size for extended objects
    pub maxsize: u32,
    /// size for sliding sky estimation box
    pub halfbox: u32,

    // don't do background subtraction.
    pub nobgsub: bool,

    // global background.
    pub globalbg: f64,

    // invert the image before processing (for black-on-white images)
    pub invert: bool,

    // should we perform lanczos resampling for flux
    // what window size should we resample
    pub lanczos_window: Option<f64>,

    // If set, the given sigma value will be used;
    // otherwise a value will be estimated.
    pub sigma: Option<f64>,

    pub debug: ImageDebugConfig,
}
impl Default for ImageExtractor {
    fn default() -> Self {
        Self {
            dpsf: SIMPLEXY_DEFAULT_DPSF,
            plim: SIMPLEXY_DEFAULT_PLIM,
            dlim: SIMPLEXY_DEFAULT_DLIM,
            saddle: SIMPLEXY_DEFAULT_SADDLE,
            maxper: SIMPLEXY_DEFAULT_MAXPER,
            maxnpeaks: SIMPLEXY_DEFAULT_MAXNPEAKS,
            maxsize: SIMPLEXY_DEFAULT_MAXSIZE,
            halfbox: SIMPLEXY_DEFAULT_HALFBOX,
            nobgsub: false,
            globalbg: 0.0,
            invert: false,
            lanczos_window: None,
            sigma: None,
            debug: ImageDebugConfig::default(),
        }
    }
}

pub(crate) type LumaImage<T> = image::ImageBuffer<image::Luma<T>, Vec<T>>;
type Grayscale = LumaImage<u8>;

pub fn invert(img: &mut Grayscale) {
    image::imageops::invert(img);
}

fn dump_colored_objects(img: &ImageBuffer<Luma<i32>, Vec<i32>>, name: &str) -> Result<()> {
    let mut colors = HashMap::new();
    colors.insert(-1i32, image::Rgb::from_channels(0, 0, 0, 0));

    let mut get_color = |obj: i32| {
        colors
            .entry(obj)
            .or_insert_with(|| {
                let [r, g, b] = random_color::RandomColor::new().to_rgb_array();
                image::Rgb::from_channels(r, g, b, 0)
            })
            .clone()
    };

    let pretty = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let obj = img.get_pixel(x, y).0[0];
        get_color(obj)
    });
    pretty.save(format!("{}.jpg", name))?;
    Ok(())
}

fn log_image<T: Primitive + 'static, P: AsRef<std::path::Path>>(
    img: &LumaImage<T>,
    path: Option<P>,
) -> anyhow::Result<()> {
    if let Some(p) = path {
        ImageDebugger::new(img).save(p)
    } else {
        Ok(())
    }
}

impl ImageExtractor {
    pub(crate) fn run(&self, img: &mut Grayscale) -> Result<Vec<Source>> {
        if self.invert {
            invert(img)
        }

        let flattened_bg = if self.nobgsub {
            todo!("dont do background smoothing")
        } else {
            flatten::flatten(&img, self.halfbox)?
        };
        log_image(
            &flattened_bg,
            self.debug.background_subtracted_image.as_ref(),
        )?;

        let smooth = self.smoothing(&flattened_bg)?;
        log_image(
            &smooth,
            self.debug.smooth_background_subtracted_image.as_ref(),
        )?;

        let sigma = dsigma(&smooth, 5, None);
        log::debug!("image sigma: {}", sigma);

        let limit =
            (sigma / (std::f64::consts::FRAC_2_SQRT_PI * self.dpsf)) * self.plim + self.globalbg;

        let mask = dmask(&smooth, limit, self.dpsf)?;

        let masked = luma_apply_mask(img, &mask);
        masked.save("mask.jpg")?;

        let obj = label::dfind(&mask)?;
        dump_colored_objects(&obj, "objects")?;
        let peaks = dallpeaks(
            &flattened_bg,
            &obj,
            self.dpsf,
            sigma,
            self.dlim,
            self.saddle,
            self.maxper as usize,
            self.maxnpeaks,
            sigma,
            self.maxsize as usize,
        )?;
        log::info!("image resulted in {} sources", peaks.len());

        let sources = if let Some(window) = self.lanczos_window {
            self.lanczos_levels(img, &flattened_bg, &peaks, window)
        } else {
            self.levels(img, &flattened_bg, &peaks)
        };
        Ok(sources)
    }

    fn levels<T: Primitive + Into<f64> + 'static>(
        &self,
        img: &LumaImage<T>,
        flat: &LumaImage<T>,
        peaks: &[Point],
    ) -> Vec<Source> {
        let mut sources = Vec::with_capacity(peaks.len());
        for point in peaks.iter().cloned() {
            let ix = (point.x + 0.5) as u32;
            let iy = (point.y + 0.5) as u32;
            let mut flux: f64 = flat.get_pixel(ix, iy).0[0].into();
            let mut background: f64 = img.get_pixel(ix, iy).0[0].into() - flux;
            flux -= self.globalbg;
            background += self.globalbg;

            sources.push(Source {
                point,
                flux,
                background,
            })
        }

        sources
    }

    fn lanczos_levels<T: Primitive + Into<f64> + Into<f64> + 'static>(
        &self,
        img: &LumaImage<T>,
        flat: &LumaImage<T>,
        peaks: &[Point],
        window: f64,
    ) -> Vec<Source> {
        let mut sources = Vec::with_capacity(peaks.len());
        for point in peaks.iter().cloned() {
            let mut flux =
                lanczos_resample(flat, point.x as f64, point.y as f64, window, false) as f64;
            let mut background =
                lanczos_resample(img, point.x as f64, point.y as f64, window, false) as f64 - flux;
            flux -= self.globalbg;
            background += self.globalbg;

            sources.push(Source {
                point,
                flux,
                background,
            })
        }

        sources
    }

    fn smoothing(&self, img: &Grayscale) -> Result<Grayscale> {
        log::trace!("smoothing image");
        let (smooth, max_pixel) = dsmooth2(img, self.dpsf);
        let export_smooth = grayf2u8(&smooth, max_pixel);
        Ok(export_smooth)
    }
}

fn grayf2u8(img: &LumaImage<f64>, max_pixel: f64) -> Grayscale {
    let (width, height) = img.dimensions();
    let mut out = Grayscale::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x, y).0[0];
            let b = p / max_pixel * 255.0;
            out.put_pixel(x, y, Pixel::from_channels(b as u8, 255, 255, 255));
        }
    }

    out
}

fn luma_apply_mask<T: Primitive + 'static>(
    img: &ImageBuffer<Luma<T>, Vec<T>>,
    mask: &Grayscale,
) -> ImageBuffer<Luma<T>, Vec<T>> {
    assert_eq!(img.dimensions(), mask.dimensions());
    let mut masked = ImageBuffer::new(img.width(), img.height());
    for row in 0..img.height() {
        for col in 0..img.width() {
            let m = mask.get_pixel(col, row).0[0];
            if m > 0 {
                masked.put_pixel(col, row, img.get_pixel(col, row).clone());
            }
        }
    }
    masked
}

fn luma_max<T: Primitive + 'static>(img: &ImageBuffer<Luma<T>, Vec<T>>) -> T {
    img.pixels()
        .map(|p| p.0[0])
        .fold(None, |a, b| {
            Some(if let Some(inner) = a {
                if inner > b {
                    inner
                } else {
                    b
                }
            } else {
                b
            })
        })
        .unwrap_or(T::zero())
}

fn dsmooth2<T: Primitive + 'static>(img: &LumaImage<T>, sigma: f64) -> (LumaImage<f64>, f64) {
    let npix = 2 * ((3.0 * sigma).ceil() as usize + 1);
    let half = (npix / 2) as i32;

    let neghalfinvvar = -1.0 / (2.0 * sigma * sigma);

    let mut total = 0.0;
    let mut kernel_1d = (0..npix)
        .map(|idx| {
            let dx = idx as f64 - 0.5 * (npix as f64 - 1.0);
            let k = (dx * dx * neghalfinvvar).exp();
            total += k;
            k
        })
        .collect::<Vec<_>>();
    let scale = 1.0 / total;
    for k in &mut kernel_1d {
        *k *= scale;
    }

    log::trace!("smoothing image {:?} npix:{}", img.dimensions(), npix);
    let mut smooth = LumaImage::new(img.width(), img.height());

    let width_max = (img.width() - 1) as i32;
    let height_max = (img.height() - 1) as i32;
    let mut max_pixel = 0.0;

    // convolve x
    for row in 0..img.height() {
        for col in 0..img.width() {
            let idx = col as i32;
            let start = std::cmp::max(0, idx - half);
            let end = std::cmp::min(width_max, idx + half);
            let sum = (start..end)
                .map(|sample| {
                    let base_idx = sample - idx + half;
                    let p: f64 = NumCast::from(img.get_pixel(sample as u32, row).channels4().0)
                        .expect("could not convert to float");
                    let o = p * kernel_1d[base_idx as usize];
                    o
                })
                .sum::<f64>();
            smooth.put_pixel(col, row, Pixel::from_channels(sum, 1.0, 1.0, 1.0))
        }
    }

    let mut smoothbuf = Vec::<f64>::with_capacity(img.height() as usize);
    // convolve y
    for col in 0..img.width() {
        for row in 0..img.height() {
            let idx = row as i32;
            let start = std::cmp::max(0, idx - half);
            let end = std::cmp::min(height_max, idx + half);
            let sum = (start..end)
                .map(|sample| {
                    let base_idx = sample - idx + half;
                    // log::trace!("input[{}] * kernel[{}]", sample, base_idx);
                    let p = smooth.get_pixel(col, sample as u32).channels4().0 as f64;
                    let o = p * kernel_1d[base_idx as usize];
                    // log::trace!("p:{} k:{} o:{}", p, kernel_1d[base_idx as usize], o);
                    o
                })
                .sum::<f64>();
            smoothbuf.push(sum);
        }
        for (row, sum) in smoothbuf.iter().enumerate() {
            if *sum > max_pixel {
                max_pixel = *sum;
            }
            smooth.put_pixel(col, row as u32, Pixel::from_channels(*sum, 1.0, 1.0, 1.0))
        }
        smoothbuf.clear()
    }

    (smooth, max_pixel)
}

fn dmask(img: &Grayscale, limit: f64, dpsf: f64) -> Result<Grayscale> {
    let mut mask = Grayscale::new(img.width(), img.height());

    let mut flagged_one = false;
    let box_size = (3.0 * dpsf) as u32;

    let bounding_box = |n: u32, idx: u32| {
        let ilow = if box_size > idx { 0 } else { idx - box_size };
        let ihi = if idx + box_size > n - 1 {
            n - 1
        } else {
            idx + box_size
        };
        (ilow, ihi)
    };

    for row in 0..img.height() {
        let (rlow, rhi) = bounding_box(img.height(), row);
        for col in 0..img.width() {
            let p = img.get_pixel(col, row).channels4().0 as f64;
            if p < limit {
                continue;
            }
            flagged_one = true;
            let (clow, chi) = bounding_box(img.width(), col);
            for r_mask in rlow..rhi {
                for c_mask in clow..chi {
                    mask.put_pixel(c_mask, r_mask, Pixel::from_channels(1, 0, 0, 0));
                }
            }
        }
    }

    if !flagged_one {
        // no mask
        let max_value = luma_max(img) as f64;
        Err(ExtractionError::NoSignificantPixels { limit, max_value }.into())
    } else {
        Ok(mask)
    }
}
fn dsigma(img: &Grayscale, sp: u32, gridsize: Option<u32>) -> f64 {
    if img.dimensions() == (1, 1) {
        return 0.0;
    }

    let gridsize = gridsize.unwrap_or(DSIGMA_DEFAULT_GRIDSIZE);

    let set_step = |dim: u32| {
        let mut d = gridsize;
        if d > dim / 4 {
            d = dim / 4;
        }
        if d == 0 {
            d = 1;
        }
        d
    };

    let dx = set_step(img.width());
    let dy = set_step(img.height());

    let ndiff = (((img.width() - sp + dx - 1) / dx) * ((img.height() - sp + dy - 1) / dy)) as usize;
    if ndiff <= 1 {
        return 0.0;
    }
    let mut diff = Vec::with_capacity(ndiff as usize);
    log::trace!("sampling sigma at {} points", ndiff);
    for jdx in (0..img.height() - sp).step_by(dy as usize) {
        for idx in (0..img.width() - sp).step_by(dx as usize) {
            let lhs = img.get_pixel(idx, jdx).channels4().0 as f64;
            let rhs = img.get_pixel(idx + sp, jdx + sp).channels4().0 as f64;
            diff.push((lhs - rhs).abs());
        }
    }
    assert_eq!(diff.len(), ndiff);

    if ndiff <= 10 {
        return diff.iter().map(|d| d * d).sum::<f64>() / ndiff as f64;
    }

    let mut n_sigma = 0.7f64; // ?
    let mut s = 0.0;
    let mut sorted = diff.to_owned();
    // TODO the point of sorting is to put unused values at the end, so we should
    // throw all the NAN values to the end
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("unable to sort floats"));

    while s == 0.0 {
        let k = (ndiff as f64
            * statrs::function::erf::erf(n_sigma * std::f64::consts::FRAC_1_SQRT_2))
            as usize;

        if k >= ndiff {
            log::error!("failed to estimate image noise setting sigma=1.0, expect the worst.");
            // TODO - try finer grid of sample points...
            return 1.0;
        }
        s = sorted[k] as f64 / (n_sigma * std::f64::consts::SQRT_2); //todo dselip
        log::trace!("n_sigma={}, s={}", n_sigma, s);
        n_sigma += 0.1;
    }
    s as f64
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn default_with_override() {
        let obj = ImageExtractor {
            dlim: 3.3,
            ..Default::default()
        };

        assert_eq!(obj.dlim, 3.3);
        assert_eq!(obj.plim, SIMPLEXY_DEFAULT_PLIM);
    }
}
