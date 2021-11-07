use super::Grayscale;
use image::Pixel;
pub(crate) fn flatten(img: &Grayscale, halfbox: u32) -> anyhow::Result<Grayscale> {
    let min_xy = std::cmp::min(img.width(), img.height());
    let mut hbox = halfbox;
    if min_xy < 2 * halfbox + 1 {
        hbox = (((min_xy as f64) - 1.0) / 2.0).floor() as u32;
    }
    // TODO I think tighter filters produces better results, but I'll trust the PhDs
    // halfbox /= 4;
    assert!(min_xy >= 2 * hbox + 1);
    log::trace!("apply median filter");
    // TODO faster implementations exist
    let mut median_filtered = imageproc::filter::median_filter(img, hbox, hbox);
    median_filtered.save("median_filter.jpg")?;

    for (m, i) in median_filtered.pixels_mut().zip(img.pixels()) {
        let img_ch = i.channels4();
        let med_ch = m.channels4();

        let f = if med_ch.0 >= img_ch.0 {
            0
        } else {
            img_ch.0 - med_ch.0
        };

        *m = Pixel::from_channels(f, 255, 255, 255);
    }

    Ok(median_filtered)
}
