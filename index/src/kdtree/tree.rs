use super::kd_types::KdtMetadata;

pub(crate) struct KDTree<'a, T, D> {
    pub(crate) metadata: KdtMetadata,
    pub(crate) cut: TreeCut<'a, T>,
    pub(crate) data: &'a [D],
}

pub(crate) enum TreeCut<'a, T> {
    BoundingBox(&'a [T]),
    SplitDim(&'a [T])
}

#[derive(Debug)]
pub(crate) struct KDRange {
    pub(crate) range: Vec<(f64, f64)>,
    pub(crate) scale: f64,
    pub(crate) invscale: f64,
}

pub(crate) struct DimSplit {
    pub(crate) dimbits: u8,
    pub(crate) dimmask: u32,
    pub(crate) splitmask: u32,
}