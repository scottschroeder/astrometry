use super::LumaImage;
use crate::Point;
use image::Primitive;

const INITIAL_STEPSIZE: f64 = 0.1;
const N_SIGMA: f64 = 6.0;
const N_STEPS: usize = 100;

#[inline]
fn sign(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

pub(crate) fn max_gaussian<T: Primitive + Into<f64> + 'static>(
    img: &LumaImage<T>,
    sigma: f64,
    x0: u32,
    y0: u32,
) -> Point {
    let mut stepsize = INITIAL_STEPSIZE;
    let iv = 1.0 / (sigma * sigma);
    let hiv = 1.0 / (2.0 * sigma * sigma);
    let n_sigma = N_SIGMA * sigma;

    let mut x = x0 as f64;
    let mut y = y0 as f64;

    let img_max_x = img.width() as i32 - 1;
    let img_max_y = img.height() as i32 - 1;

    loop {
        let mut xdir = 0f64;
        let mut ydir = 0f64;
        let mut xflipped = false;
        let mut yflipped = false;

        for step in 0..N_STEPS {
            let mut v = 0f64;
            let mut gx = 0f64;
            let mut gy = 0f64;

            let ilo = std::cmp::max(0, (x - n_sigma).floor() as i32) as u32;
            let ihi = std::cmp::min(img_max_x, (x + n_sigma).ceil() as i32) as u32;
            let jlo = std::cmp::max(0, (y - n_sigma).floor() as i32) as u32;
            let jhi = std::cmp::min(img_max_y, (y + n_sigma).ceil() as i32) as u32;

            for j in jlo..jhi {
                for i in ilo..ihi {
                    let dx = i as f64 - x;
                    let dy = j as f64 - y;
                    let mut g = (-1.0 * (dx * dx + dy * dy) * hiv).exp();
                    g *= (img.get_pixel(i, j).0[0]).into() * iv;
                    gx += g * -dx;
                    gy += g * -dy;
                    v += g;
                }
            }

            let dx_sign = sign(-gx);
            let dy_sign = sign(-gy);

            log::trace!(
                "stepsize {}, step #{} - x,y = ({},{}), V={}, Gx={}, Gy={}",
                stepsize,
                step,
                x,
                y,
                v,
                gx,
                gy
            );
            if step == 0 {
                xdir = dx_sign;
                ydir = dy_sign;
                if xdir == 0.0 && ydir == 0.0 {
                    break;
                }
            }
            if !xflipped && (dx_sign - xdir).abs() < f64::EPSILON {
                x += dx_sign * stepsize;
            } else {
                xflipped = true;
            }
            if !yflipped && (dy_sign - ydir).abs() < f64::EPSILON {
                y += dy_sign * stepsize;
            } else {
                yflipped = true;
            }
            if xflipped && yflipped {
                break;
            }
        }
        if stepsize <= 0.002 {
            break;
        }
        stepsize /= 10.0;
    }

    Point { x, y }
}
