use std::fmt;
#[derive(Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }
}

impl<X: Into<f64>, Y: Into<f64>> From<(X, Y)> for Point {
    fn from((x, y): (X, Y)) -> Self {
        Point {
            x: x.into(),
            y: y.into(),
        }
    }
}

impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.05}, {:.05})", self.x, self.y)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.02}, {:.02})", self.x, self.y)
    }
}
