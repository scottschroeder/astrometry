use std::io::Read; // TODO remove
pub(crate) mod kdtree;

pub fn demo(path: &str) -> anyhow::Result<()> {
    log::debug!("reading index");

    let index_path = std::path::Path::new(path);

    let mut f = std::fs::File::open(index_path)?;
    let mut buffer = vec![];
    let _ = f.read_to_end(&mut buffer);
    let fits = fits_rs::parser::parse(&buffer)
        .map_err(|e| e.map_input(|bad| String::from_utf8_lossy(bad).to_string()))?;
    for hdu in &fits.hdu {
        log::trace!("Fits Header data[{}]\n{}", hdu.data.len(), hdu.header)
    }
    kdtree::fits::demo(fits.hdu.as_slice())?;
    Ok(())
}
