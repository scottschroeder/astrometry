use image::{ImageBuffer, Luma, Primitive};

pub(crate) struct Canvas<T> {
    inner: Vec<T>,
    stride: usize,
}

impl<T> Canvas<T> {
    pub fn width(&self) -> usize {
        self.stride
    }
    pub fn height(&self) -> usize {
        self.inner.len() / self.stride
    }
}

impl<T: Primitive + 'static> From<ImageBuffer<Luma<T>, Vec<T>>> for Canvas<T> {
    fn from(img: ImageBuffer<Luma<T>, Vec<T>>) -> Self {
        let stride = img.width() as usize;
        let inner = img.into_raw();
        Canvas { inner, stride }
    }
}

impl<T: Primitive + 'static> Into<ImageBuffer<Luma<T>, Vec<T>>> for Canvas<T> {
    fn into(self) -> ImageBuffer<Luma<T>, Vec<T>> {
        ImageBuffer::from_raw(self.width() as u32, self.height() as u32, self.inner)
            .expect("invalid container")
    }
}
