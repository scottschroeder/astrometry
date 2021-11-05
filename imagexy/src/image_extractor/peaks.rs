use image::{Luma, Pixel, Primitive};
use num_traits::NumCast;

use super::{ExtractionError, LumaImage};

pub fn dpeaks<T: Primitive + 'static + PartialOrd>(
    img: &LumaImage<T>,
    saddle: f32,
    sigma: f32,
    dlim: f32,
    minpeak: f32,
    maxnpeaks: usize,
) -> Result<Vec<(u32, u32)>, ExtractionError> {
    // TODO remove/rename
    let smooth = img;

    let get = |x: u32, y: u32| {
        let p: f32 = NumCast::from(smooth.get_pixel(x, y).0[0]).unwrap();
        p
    };

    let mut peaks = Vec::new();

    for j in 1..img.height() - 1 {
        let jst = j - 1;
        let jnd = j + 1;
        for i in 1..img.width() - 1 {
            if get(i, j) < minpeak {
                continue;
            }
            let ist = i - 1;
            let ind = i + 1;
            let mut highest = true;
            for ip in ist..ind {
                for jp in jst..jnd {
                    if get(ip, jp) > get(i, j) {
                        highest = false;
                    }
                }
            }
            if highest {
                peaks.push((i, j));
            }
        }
    }
    let data = smooth.as_raw().as_slice();
    let indx = permutation::sort_by(data, |a, b| a.partial_cmp(b).unwrap().reverse());

    peaks.truncate(maxnpeaks);

    let mut fullxycen = Vec::with_capacity(peaks.len());
    for idx in 0..peaks.len() {
        let x = indx.apply_inv_idx(idx) as u32 % img.width();
        let y = indx.apply_inv_idx(idx) as u32 / img.width();
        fullxycen.push((x, y));
    }

    let mut keep = vec![true; peaks.len()];

    let get_peak = |idx: usize| {
        let (x, y) = fullxycen[idx];
        let p: f32 = NumCast::from(smooth.get_pixel(x, y).0[0]).unwrap();
        p
    };

    let mut mask = LumaImage::new(img.width(), img.height());
    for i in (0..peaks.len()).rev() {
        let peak = get_peak(i);
        let mut level = peak - saddle * sigma;
        if level < sigma {
            level = sigma
        }
        if level > 0.99 * peak {
            level = 0.99 * peak;
        }
        for jp in 0..img.height() {
            for ip in 0..img.width() {
                let p = (get(ip, jp) > level) as u8;
                mask.put_pixel(ip, jp, Luma::from_channels(p, 0, 0, 0))
            }
        }
        let object = super::label::dfind(&mask)?;
        let get_object = |idx: usize| {
            let (x, y) = fullxycen[idx];
            object.get_pixel(x, y).0[0]
        };
        for j in (0..i).rev() {
            if get_object(j) == get_object(i) || get_object(i) == -1 {
                keep[i] = false;
                break;
            }

            let (jx, jy) = fullxycen[j];
            let (ix, iy) = fullxycen[i];
            let dx = jx as f32 - ix as f32;
            let dy = jy as f32 - iy as f32;
            if dx * dx + dy * dy < dlim * dlim {
                keep[i] = false;
                break;
            }
        }
    }

    Ok(fullxycen
        .into_iter()
        .zip(keep)
        .filter_map(|(p, k)| if k { Some(p) } else { None })
        .collect::<Vec<_>>())
}
