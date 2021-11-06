use image::{GenericImageView, ImageBuffer, Luma, Primitive};

use super::{dsmooth2, ExtractionError, LumaImage};
use crate::{
    image_extractor::{
        gaussian::max_gaussian,
        object_debugger::{BasicColor, ObjectImageDebugger},
    },
    Point,
};

struct BoundingBox {
    xmin: usize,
    xmax: usize,
    ymin: usize,
    ymax: usize,
}

impl BoundingBox {
    fn width(&self) -> u32 {
        (self.xmax - self.xmin + 1) as u32
    }
    fn height(&self) -> u32 {
        (self.ymax - self.ymin + 1) as u32
    }
    fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.xmin && x < self.xmax && y >= self.ymin && y < self.ymax
    }
}

struct ObjectIndex<'a> {
    object_image: &'a [i32],
    permutation: permutation::Permutation,
}

impl<'a> ObjectIndex<'a> {
    fn from_image(obj: &'a [i32]) -> ObjectIndex<'a> {
        let p = permutation::sort(obj);
        ObjectIndex {
            object_image: obj,
            permutation: p,
        }
    }

    fn first_object(&self) -> usize {
        (0..self.object_image.len())
            .rev()
            .take_while(|idx| self.get(*idx) != -1)
            .last()
            .unwrap_or(0)
    }

    fn get(&self, idx: usize) -> i32 {
        self.object_image[self.permutation.apply_inv_idx(idx)]
    }

    fn pos(&self, idx: usize) -> usize {
        self.permutation.apply_inv_idx(idx)
    }

    fn len(&self) -> usize {
        self.object_image.len()
    }

    /// Find the points that bound an object
    /// Must provide the first index into the permuation that
    /// describes the object
    fn bounding_box(&self, obj_start: usize, nx: usize) -> (usize, BoundingBox) {
        let mut xmin = None;
        let mut xmax = None;
        let mut ymin = None;
        let mut ymax = None;

        let current = self.get(obj_start);

        let mut next = obj_start;

        for m in (obj_start..self.len()).take_while(|idx| self.get(*idx) == current) {
            let xcurr = self.pos(m) % nx;
            let ycurr = self.pos(m) / nx;
            xmin = minmax_opt(xmin, xcurr, std::cmp::min);
            xmax = minmax_opt(xmax, xcurr, std::cmp::max);
            ymin = minmax_opt(ymin, ycurr, std::cmp::min);
            ymax = minmax_opt(ymax, ycurr, std::cmp::max);
            next = m;
        }

        (
            next + 1,
            BoundingBox {
                xmin: xmin.unwrap(),
                xmax: xmax.unwrap(),
                ymin: ymin.unwrap(),
                ymax: ymax.unwrap(),
            },
        )
    }
}

fn minmax_opt<T: Clone + Ord, F>(opt: Option<T>, val: T, cmp: F) -> Option<T>
where
    F: Fn(T, T) -> T,
{
    Some(if let Some(inner) = opt {
        cmp(inner.clone(), val)
    } else {
        val
    })
}

pub(crate) fn dallpeaks<T: Primitive + Into<f32> + 'static>(
    img: &ImageBuffer<Luma<T>, Vec<T>>,
    object: &ImageBuffer<Luma<i32>, Vec<i32>>,
    dpsf: f32,
    sigma: f32,
    dlim: f32,
    saddle: f32,
    _maxper: usize,
    maxnpeaks: usize,
    minpeak: f32,
    maxsize: usize,
) -> Result<Vec<Point>, ExtractionError> {
    let indx = ObjectIndex::from_image(object.as_raw().as_slice());

    // let mut k = (0..obj_data.len())
    //     .take_while(|k| {
    //         let obj_k = indx.apply_inv_idx(*k);
    //         obj_data[obj_k] == -1
    //     })
    //     .last()
    //     .map(|x| x + 1)
    //     .unwrap_or(0);

    let mut k = indx.first_object();

    // let sorted_start = indx.apply_slice(obj_data);
    // log::trace!("ss: {:?}", &sorted_start[k-1..k+300]);

    let mut peak_points = Vec::new();

    let nx = img.width() as usize;

    while k < indx.len() {
        let current = indx.get(k);
        // log::trace!("k={}, obj={}", k, current);

        let (m, bounding_box) = indx.bounding_box(k, nx);
        k = m;

        let onx = bounding_box.xmax - bounding_box.xmin + 1;
        let ony = bounding_box.ymax - bounding_box.ymin + 1;

        let obj_space = |p: Point| Point {
            x: p.x - bounding_box.xmin as f32,
            y: p.y - bounding_box.ymin as f32,
        };

        if onx < 3 || ony < 3 {
            log::trace!(
                "skipping object {}: too small {}x{} (x {}:{} y {}:{})",
                current,
                onx,
                ony,
                bounding_box.xmin,
                bounding_box.xmax,
                bounding_box.ymin,
                bounding_box.ymax
            );
            continue;
        }

        if onx > maxsize || ony > maxsize {
            log::warn!(
                "skipping object {}: too big {}x{} (x {}:{} y {}:{})",
                current,
                onx,
                ony,
                bounding_box.xmin,
                bounding_box.xmax,
                bounding_box.ymin,
                bounding_box.ymax
            );
            continue;
        }

        if peak_points.len() > maxnpeaks {
            log::warn!(
                "skipping all further objects, already found max number {}",
                maxnpeaks
            );
            break;
        }

        let oimage = copy_object(img, object, &bounding_box, current);
        let mut dbg_img = ObjectImageDebugger::new(&oimage);
        let (simage, _max_f) = dsmooth2(&oimage, dpsf);

        let peaks = super::peaks::dpeaks(&simage, saddle, sigma, dlim, minpeak, maxnpeaks)?;
        if peaks.len() > 1 {
            log::trace!("obj {}: peaks: {:?}", current, peaks);
        }

        for (i, (xci, yci)) in peaks.iter().cloned().enumerate() {
            if (xci == 0 || xci >= onx as u32 - 1) || (yci == 0 || yci >= ony as u32 - 1) {
                log::trace!(
                    "skipping subpeak {}, position ({}, {}) out of bounds x[1:{}] y[1:{}]",
                    i,
                    xci,
                    yci,
                    onx - 1,
                    ony - 1
                );
                continue;
            }
            if peak_points.len() >= maxnpeaks {
                log::debug!(
                    "skipping all further subpeaks: exceeded max number: {}",
                    maxnpeaks
                );
                break;
            }

            let this_xycen = Point {
                x: xci as f32 + bounding_box.xmin as f32,
                y: yci as f32 + bounding_box.ymin as f32,
            };
            let three = simage.view(xci - 1, yci - 1, 3, 3);

            if let Ok(p) = dcen3x3(three) {
                let img_point = Point {
                    x: p.x - 1.0 + this_xycen.x,
                    y: p.y - 1.0 + this_xycen.y,
                };
                dbg_img.add_mark(obj_space(img_point), BasicColor::Red);
                peak_points.push(img_point);
            } else if false
                && (xci > 1 && xci < onx as u32 - 2)
                && (yci > 1 && yci < ony as u32 - 2)
            {
                // TODO remove this whole block
                log::warn!(
                    "o:{} subpeak {} at ({}, {}): searching for 3x3 failed; trying 5x5...",
                    current,
                    i,
                    xci,
                    yci
                );
                let mut fives = LumaImage::new(3, 3);
                /*
                    I think this is what's happening

                    1 0 1 0 1
                    0 0 0 0 0
                    1 0 1 0 1
                    0 0 0 0 0
                    1 0 1 0 1
                */

                for di in -1..1 {
                    let sdi = (xci as i32 + 2 * di) as u32;
                    let ddi = (di + 1) as u32;
                    for dj in -1..1 {
                        let sdj = (yci as i32 + 2 * dj) as u32;
                        let ddj = (dj + 1) as u32;
                        log::trace!("({}, {}) -> ({}, {})", sdi, sdj, ddi, ddj);
                        fives.put_pixel(ddi, ddj, simage.get_pixel(sdi, sdj).clone());
                    }
                }

                match dcen3x3(fives) {
                    Ok(p) => {
                        log::warn!("3x3 saved by the 5x5");
                        let img_point = Point {
                            x: 2.0 * (p.x - 1.0) + this_xycen.x,
                            y: 2.0 * (p.y - 1.0) + this_xycen.y,
                        };
                        dbg_img.add_mark(obj_space(img_point), BasicColor::Blue);
                        peak_points.push(img_point);
                    }
                    Err(_e) => {
                        let p = max_gaussian(&oimage, dpsf, xci, yci);
                        log::trace!("max_gaussian: o:{} -> {:?}", current, p);
                        peak_points.push(Point {
                            x: bounding_box.xmin as f32 + p.x,
                            y: bounding_box.ymin as f32 + p.y,
                        });
                    }
                }
            } else {
                let p = max_gaussian(&oimage, dpsf, xci, yci);
                log::trace!("max_gaussian: o:{} -> {:?}", current, p);
                let img_point = Point {
                    x: bounding_box.xmin as f32 + p.x,
                    y: bounding_box.ymin as f32 + p.y,
                };
                dbg_img.add_mark(obj_space(img_point), BasicColor::Green);
                peak_points.push(img_point);
            }
        }
        dbg_img.save(&format!("object_marked/{}", current)).unwrap();
    }
    Ok(peak_points)
}
fn copy_object<T: Primitive + 'static>(
    img: &ImageBuffer<Luma<T>, Vec<T>>,
    object: &ImageBuffer<Luma<i32>, Vec<i32>>,
    bounding_box: &BoundingBox,
    current: i32,
) -> ImageBuffer<Luma<T>, Vec<T>> {
    let obj_at = |x, y| object.get_pixel(x, y).0[0];
    // let current = obj_at(bounding_box.xmin as u32, bounding_box.ymin as u32);

    let mut oimage = ImageBuffer::new(bounding_box.width(), bounding_box.height());
    for oj in 0..bounding_box.height() {
        for oi in 0..bounding_box.width() {
            let col = oi + bounding_box.xmin as u32;
            let row = oj + bounding_box.ymin as u32;
            if obj_at(col, row) == current {
                oimage.put_pixel(
                    oi as u32,
                    oj as u32,
                    img.get_pixel(col as u32, row as u32).clone(),
                )
            }
        }
    }
    oimage
}

fn dcen3(f0: f32, f1: f32, f2: f32) -> Result<f32, ExtractionError> {
    /*
        f0 = c
        f1 = a + b + c
        f2 = 4a + 2b + c
    */

    let a = 0.5 * (f2 - 2.0 * f1 + f0);
    if a == 0.0 {
        return Err(ExtractionError::NoPeak);
    }
    let b = f1 - a - f0;
    let xcen = -0.5 * b / a;
    if xcen < 0.0 || xcen > 2.0 {
        return Err(ExtractionError::NoPeak);
    }

    Ok(xcen)
}

fn dcen3x3<I: GenericImageView<Pixel = Luma<f32>>>(img: I) -> Result<Point, ExtractionError> {
    let get = |x: u32, y: u32| img.get_pixel(x, y).0[0];
    let mx0 = dcen3(get(0, 0), get(1, 0), get(2, 0))?;
    let mx1 = dcen3(get(0, 1), get(1, 1), get(2, 1))?;
    let mx2 = dcen3(get(0, 2), get(1, 2), get(2, 2))?;

    let my0 = dcen3(get(0, 0), get(0, 1), get(0, 2))?;
    let my1 = dcen3(get(1, 0), get(1, 1), get(1, 2))?;
    let my2 = dcen3(get(2, 0), get(2, 1), get(2, 2))?;

    let bx = (mx0 + mx1 + mx2) / 3.0;
    let mx = (mx2 - mx0) / 2.0;

    let by = (my0 + my1 + my2) / 3.0;
    let my = (my2 - my0) / 2.0;

    let xcen = (mx * (by - my - 1.0) + bx) / (1.0 + mx * my);
    let ycen = (xcen - 1.0) * my + by;

    if xcen < 0.0 || xcen > 2.0 || ycen < 0.0 || ycen > 2.0 {
        return Err(ExtractionError::PeakOutsideRange);
    }

    if !xcen.is_normal() || !ycen.is_normal() {
        return Err(ExtractionError::PeakBadFloat);
    }

    Ok(Point::new(xcen, ycen))
}
