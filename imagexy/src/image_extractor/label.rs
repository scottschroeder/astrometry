use super::ExtractionError;
use image::{ImageBuffer, Luma, Pixel, Primitive};

const INITIAL_GROUPS: usize = 64;
const INITIAL_ON_PIXELS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Label(i32);

impl Label {
    #[inline]
    fn increment(&mut self) {
        self.0 += 1
    }
    #[inline]
    fn to_pixel(self) -> Luma<i32> {
        Luma::from_channels(self.0, 0, 0, 0)
    }
    #[inline]
    fn as_index(self) -> usize {
        assert!(self.0 >= 0, "trying to read negative number as index");
        self.0 as usize
    }
}

impl From<&Luma<i32>> for Label {
    fn from(p: &Luma<i32>) -> Self {
        Label(p.0[0])
    }
}

impl Default for Label {
    fn default() -> Self {
        Self(-1)
    }
}

#[inline]
fn is_zero<T: Primitive + 'static>(img: &ImageBuffer<Luma<T>, Vec<T>>, x: u32, y: u32) -> bool {
    img.get_pixel(x, y).0[0].is_zero()
}

fn collapsing_find_minlabel(label: Label, equivs: &mut Vec<Label>) -> Label {
    let mut min = label;
    let mut label = label;
    // log::trace!("collapsing_find_minlabel [{:?}]: {:?}", label, equivs);
    while equivs[min.as_index()] != min {
        min = equivs[min.as_index()]
    }
    while label != min {
        let next = equivs[label.as_index()];
        equivs[label.as_index()] = min;
        label = next;
    }
    return min;
}

pub(crate) fn dfind<T: Primitive + 'static>(
    img: &ImageBuffer<Luma<T>, Vec<T>>,
) -> Result<ImageBuffer<Luma<i32>, Vec<i32>>, ExtractionError> {
    let mut equivs = Vec::with_capacity(INITIAL_GROUPS);
    let mut max_label = Label(0);
    let mut on_pixels = Vec::with_capacity(INITIAL_ON_PIXELS);

    let mut object = ImageBuffer::new(img.width(), img.height());

    for iy in 0..img.height() {
        for ix in 0..img.width() {
            object.put_pixel(ix, iy, Label::default().to_pixel());
            if is_zero(img, ix, iy) {
                continue;
            }
            on_pixels.push((ix, iy));

            if ix > 0 && !is_zero(img, ix - 1, iy) {
                // Old group
                // TODO won't old pixel always just be the last used value? do we need to look it up?
                object.put_pixel(ix, iy, object.get_pixel(ix - 1, iy).clone());
            } else {
                // New group
                object.put_pixel(ix, iy, max_label.to_pixel());
                equivs.push(max_label);
                max_label.increment();
                if max_label.0 == std::i32::MAX {
                    // TODO does relabelling help?
                    // relabel
                    return Err(ExtractionError::ExhaustedLabels);
                }
            }

            // TODO why aren't we saving this in a variable AGAIN?
            let thislabel = Label::from(object.get_pixel(ix, iy));
            let mut thislabelmin = collapsing_find_minlabel(thislabel, &mut equivs);
            if iy == 0 {
                continue;
            }

            let imin = if ix > 0 { ix - 1 } else { 0 };
            let imax = if ix + 1 > img.width() - 1 {
                img.width() - 1
            } else {
                ix + 1
            };
            for idx in imin..imax {
                if !is_zero(img, idx, iy - 1) {
                    let otherlabel = Label::from(object.get_pixel(idx, iy - 1));
                    let otherlabelmin = collapsing_find_minlabel(otherlabel, &mut equivs);
                    if thislabelmin != otherlabelmin {
                        let oldlabelmin = std::cmp::max(thislabelmin, otherlabelmin);
                        let newlabelmin = std::cmp::min(thislabelmin, otherlabelmin);
                        thislabelmin = newlabelmin;
                        equivs[oldlabelmin.as_index()] = newlabelmin;
                        equivs[thislabel.as_index()] = newlabelmin;
                        object.put_pixel(idx, iy - 1, newlabelmin.to_pixel());
                    }
                }
            }
            object.put_pixel(ix, iy, thislabelmin.to_pixel());
        }
    }
    // relabel
    return Ok(object);
}
