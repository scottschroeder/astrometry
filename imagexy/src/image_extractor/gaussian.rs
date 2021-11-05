use super::LumaImage;
use crate::Point;
use image::Primitive;

const INITIAL_STEPSIZE: f32 = 0.1;
const N_SIGMA: f32 = 6.0;
const N_STEPS: usize = 100;

#[inline]
fn sign(x: f32) -> f32 {
    if x == 0.0 {
        0.0
    } else if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

struct DebugGrid {
    inner: Vec<f32>,
    stride: usize,
}

impl DebugGrid {
    fn new(w: usize, h: usize) -> DebugGrid {
        DebugGrid {
            inner: vec![0.0; w * h],
            stride: w,
        }
    }

    fn put(&mut self, x: usize, y: usize, v: f32) {
        self.inner[y * self.stride + x] = v
    }
}

impl std::fmt::Debug for DebugGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let w = self.stride;
        let h = self.inner.len() / w;
        write!(f, "DebugGrid ({}x{}) [", w, h)?;
        for (idx, v) in self.inner.iter().enumerate() {
            let col = idx % w;
            let _row = idx / w;
            if col == 0 {
                write!(f, "\n")?;
            }
            write!(f, "\t{:.02}", *v)?;
        }
        writeln!(f, "\n]")
    }
}

pub(crate) fn max_gaussian<T: Primitive + Into<f32> + 'static>(
    img: &LumaImage<T>,
    sigma: f32,
    x0: u32,
    y0: u32,
) -> Point {
    let mut stepsize = INITIAL_STEPSIZE;
    let iv = 1.0 / (sigma * sigma);
    let hiv = 1.0 / (2.0 * sigma * sigma);
    let n_sigma = N_SIGMA * sigma;

    let mut x = x0 as f32;
    let mut y = y0 as f32;

    let img_max_x = img.width() as i32 - 1;
    let img_max_y = img.height() as i32 - 1;

    loop {
        let mut xdir = 0f32;
        let mut ydir = 0f32;
        let mut xflipped = false;
        let mut yflipped = false;

        for step in 0..N_STEPS {
            let mut v = 0f32;
            let mut gx = 0f32;
            let mut gy = 0f32;

            let ilo = std::cmp::max(0, (x - n_sigma).floor() as i32) as u32;
            let ihi = std::cmp::min(img_max_x, (x + n_sigma).ceil() as i32) as u32;
            let jlo = std::cmp::max(0, (y - n_sigma).floor() as i32) as u32;
            let jhi = std::cmp::min(img_max_y, (y + n_sigma).ceil() as i32) as u32;

            // log::trace!("window x[{}:{}] y[{}:{}]", ilo, ihi, jlo, jhi);
            // let mut dbg = DebugGrid::new((ihi - ilo) as usize, (jhi - jlo) as usize);
            for j in jlo..jhi {
                for i in ilo..ihi {
                    let dx = i as f32 - x;
                    let dy = j as f32 - y;
                    let mut g = (-1.0 * (dx * dx + dy * dy) * hiv).exp();
                    g *= (img.get_pixel(i, j).0[0]).into() * iv;
                    // log::trace!("@({},{}) dx={}, dy={}, g={}", i, j, dx, dy, g);
                    // dbg.put((i - ilo) as usize, (j - jlo) as usize, g);
                    gx += g * -dx;
                    gy += g * -dy;
                    v += g;
                }
            }

            // log::debug!("{:?}", dbg);
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
            if !xflipped && dx_sign == xdir {
                x += dx_sign * stepsize;
            } else {
                xflipped = true;
            }
            if !yflipped && dy_sign == ydir {
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
