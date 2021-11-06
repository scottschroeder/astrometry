use crate::image_extractor::LumaImage;
use image::Primitive;
use std::f64::consts::PI;

fn lanczos(x: f64, window: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else if x > window || x < -window {
        0.0
    } else {
        let xpi = x * PI;
        window * xpi.sin() * (xpi / window).sin() / (xpi * xpi)
    }
}

pub(crate) fn lanczos_resample<T: Primitive + Into<f64> + 'static>(
    img: &LumaImage<T>,
    px: f64,
    py: f64,
    window: f64,
    weighted: bool,
) -> f64 {
    let x0 = std::cmp::max(0, (px - window).floor() as i32) as u32;
    let y0 = std::cmp::max(0, (py - window).floor() as i32) as u32;
    let x1 = std::cmp::min(img.width() - 1, (px + window).ceil() as u32);
    let y1 = std::cmp::min(img.height() - 1, (py + window).ceil() as u32);

    let nx = 1 + x1 - x0;
    let ny = 1 + y1 - y0;

    // // TODO these asserts seem like they will get hit and be very confusing
    // assert!(nx < 12, "window must be less than 12");
    // assert!(ny < 12, "window must be less than 12");

    let ky = (0..ny)
        .map(|dy| lanczos(py - (y0 + dy) as f64, window))
        .collect::<Vec<_>>();
    let kx = (0..nx)
        .map(|dx| lanczos(px - (x0 + dx) as f64, window))
        .collect::<Vec<_>>();

    let mut weight = 0.0;
    let mut sum = 0.0;

    let get_img_pixel = |x: usize, y: usize| {
        let p: f64 = img.get_pixel(x0 + x as u32, y0 + y as u32).0[0].into();
        p
    };

    for (dy, idx_ky) in ky.iter().cloned().enumerate() {
        if idx_ky == 0.0 {
            continue;
        }
        let mut xweight = 0.0;
        let mut xsum = 0.0;
        for (dx, idx_kx) in kx.iter().cloned().enumerate() {
            if idx_kx == 0.0 {
                continue;
            }
            let pix = get_img_pixel(dx, dy);
            if weighted {
                xweight += idx_kx;
            }
            xsum += idx_kx * pix;
        }
        if weighted && xweight == 0.0 {
            continue;
        }
        if weighted {
            weight += idx_ky * xweight;
        }
        sum += idx_ky * xsum;
    }
    if weighted {
        sum / weight
    } else {
        sum
    }
}
