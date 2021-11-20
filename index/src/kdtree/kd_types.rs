pub use super::error::Error;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct KdtType {
    pub(crate) ext: KdtExt,
    pub(crate) tree: KdtTree,
    pub(crate) data: KdtData,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KdtMetadata {
    pub(crate) kdt_type: KdtType,
    pub(crate) has_linear_lr: bool,
    pub(crate) ndata: usize,
    pub(crate) ndim: usize,
    pub(crate) nnodes: usize,
    pub(crate) nbottom: usize,
    pub(crate) ninterior: usize,
    pub(crate) nlevels: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum KdtData {
    Null,
    F64,
    F32,
    U64,
    U32,
    U16,
}

impl FromStr for KdtData {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "double" => KdtData::F64,
            "float" => KdtData::F32,
            "u64" => KdtData::U64,
            "u32" => KdtData::U64,
            "u16" => KdtData::U64,
            _ => return Err(Error::ParseType("KdtData", s.to_string())),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum KdtTree {
    Null,
    F64,
    F32,
    U64,
    U32,
    U16,
}

impl FromStr for KdtTree {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "double" => KdtTree::F64,
            "float" => KdtTree::F32,
            "u64" => KdtTree::U64,
            "u32" => KdtTree::U64,
            "u16" => KdtTree::U64,
            _ => return Err(Error::ParseType("KdtTree", s.to_string())),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum KdtExt {
    Null,
    F64,
    F32,
    U64,
}

impl FromStr for KdtExt {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "double" => KdtExt::F64,
            "float" => KdtExt::F32,
            "u64" => KdtExt::U64,
            _ => return Err(Error::ParseType("KdtExt", s.to_string())),
        })
    }
}
