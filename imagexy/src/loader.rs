use anyhow::{Context, Result};
use image::DynamicImage;
use std::path;

pub fn load_from_path<P: AsRef<path::Path>>(p: P) -> Result<DynamicImage> {
    let p = p.as_ref();
    Ok(image::open(p).context(format!("could not open: {:?}", p))?)
}
