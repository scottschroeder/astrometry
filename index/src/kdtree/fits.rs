use super::{
    error::Error,
    kd_types::{KdtData, KdtExt, KdtMetadata, KdtTree, KdtType},
    tree::DimSplit,
};
use crate::kdtree::tree::KDRange;
use crate::kdtree::tree::KDTree;
use fits_rs::types::{BinTable, Header, Keyword, TableError, Value, ValueRetrievalError, HDU};
use std::{collections::HashMap, str::FromStr};
use strum::EnumString;
use thiserror::Error as _;

const FITS_ENDIAN_DETECTOR: u32 = 0x01020304;

// names (actually prefixes) of FITS tables.
const KD_STR_HEADER: &str = "kdtree_header";
const KD_STR_LR: &str = "kdtree_lr";
const KD_STR_PERM: &str = "kdtree_perm";
const KD_STR_BB: &str = "kdtree_bb";
const KD_STR_SPLIT: &str = "kdtree_split";
const KD_STR_SPLITDIM: &str = "kdtree_splitdim";
const KD_STR_DATA: &str = "kdtree_data";
const KD_STR_RANGE: &str = "kdtree_range";

trait MustKeyword {
    fn to_keyword(self) -> Keyword;
}

impl MustKeyword for Keyword {
    fn to_keyword(self) -> Keyword {
        self
    }
}

impl MustKeyword for &'static str {
    fn to_keyword(self) -> Keyword {
        Keyword::from_str(self).expect("invalid keyword")
    }
}

struct FitsFileMap<'a> {
    hdu: &'a [HDU<'a>],
    bintables: Vec<Option<BinTable<'a>>>,
}

fn first_column<'a>(b: &'a BinTable<'a>) -> Option<&'a str> {
    b.ttype.as_ref().and_then(|tv| tv.get(0).cloned())
}

impl<'a> FitsFileMap<'a> {
    fn load(hdu: &'a [HDU<'a>]) -> FitsFileMap<'a> {
        let bintables = hdu
            .iter()
            .map(|h| match BinTable::new(&h.header) {
                Ok(t) => Some(t),
                Err(e) => {
                    match e {
                        TableError::IncorrectExtension => {}
                        _ => {
                            log::warn!("could not read header: {:?}\n{}", e, h.header)
                        }
                    };
                    None
                }
            })
            .collect::<Vec<_>>();
        FitsFileMap { hdu, bintables }
    }

    fn find_table_column<'b>(&'b self, name: &str) -> Option<(&'a HDU<'a>, &'b BinTable<'a>)> {
        for (h, bt) in self
            .hdu
            .iter()
            .zip(&self.bintables)
            .filter_map(|(h, b)| b.as_ref().map(|t| (h, t)))
        {
            if let Some(col_name) = first_column(bt) {
                if col_name == name {
                    return Some((h, bt));
                }
            }
        }
        None
    }

    fn read_chunk<'b, T: bytemuck::Pod>(
        &self,
        tablename: &'b str,
        nrows: usize,
        ncol: usize,
    ) -> Result<&'a [T], Error> {
        let itemsize = std::mem::size_of::<T>() * ncol;

        let (h, c) = self
            .find_table_column(tablename)
            .ok_or_else(|| Error::MissingTable(tablename.to_string()))?;


        if nrows != c.rows {
            log::error!(
                "table {} expected {} data items (rows) found {}",
                tablename,
                nrows,
                c.rows
            );
            return Err(Error::InvalidTableDimmensions);
        }
        if itemsize != c.row_width() {
            log::error!(
                "table {} expected {} data size (row width) found {}",
                tablename,
                itemsize,
                c.row_width()
            );
            return Err(Error::InvalidTableDimmensions);
        }
        let expected = itemsize * nrows;

        log::trace!(
            "want: {} bytes, have: {} bytes, rem: {} bytes",
            expected,
            h.data.len(),
            h.data.len() - expected
        );
        if expected > h.data.len() {
            return Err(Error::InvalidTableDimmensions);
        }
        Ok(bytemuck::cast_slice(&h.data[..expected]))
    }
}

// fn fits_table_open2(hdu: &HDU) -> Result<(), Error> {
//     let bt = BinTable::new(&hdu.header).map_err(|e|{

//         match e {
//             TableError::IncorrectExtension => todo!(),
//             TableError::PropertyNotDefined(_, _) => todo!(),
//             TableError::UnexpectedValue(_, _) => todo!(),
//             TableError::InvalidFormString(_, _) => todo!(),
//         }
//     })?;
//     let n_col = header_get_int(&hdu.header, Keyword::TFIELDS)? as usize;
//     let mut tload = QFitsTable {
//         table_type: get_table_type(&hdu.header)?,
//         width: header_get_int(&hdu.header, Keyword::NAXISn(1))? as usize,
//         n_row: header_get_int(&hdu.header, Keyword::NAXISn(2))? as usize,
//         n_col,
//         col: Vec::with_capacity(n_col),
//     };

//     for col_idx in 0..n_col as u16 {
//         let label = header_get_str(&hdu.header, Keyword::TTYPEn(col_idx)).unwrap_or("");
//         let unit = header_get_str(&hdu.header, Keyword::TUNITn(col_idx)).unwrap_or("");
//         let disp = header_get_str(&hdu.header, Keyword::TDISPn(col_idx)).unwrap_or("");
//         let nullval = header_get_str(&hdu.header, Keyword::TNULLn(col_idx)).unwrap_or("");
//         let tform = header_get_str(&hdu.header, Keyword::TFORMn(col_idx))?;
//         let atom = Atom::parse_tform(tload.table_type, tform)?;
//     }

//     Ok(())
// }

#[derive(Debug, Clone, Copy, PartialEq)]
enum TreeStyle {
    Old,
    New,
}

pub fn demo(hdu: &[HDU]) -> anyhow::Result<()> {
    let cache = FitsFileMap::load(hdu);
    read_fits_kdtree::<i32, f32>(&cache, Some("stars"))?;
    Ok(())
}

fn read_fits_kdtree<'a, T: bytemuck::Pod, D: bytemuck::Pod>(
    filemap: &FitsFileMap<'a>,
    treename: Option<&str>,
) -> Result<KDTree<'a, T, D>, Error> {
    let (h, metadata) = find_tree(filemap.hdu, treename)?;

    let lr_data =
        filemap.read_chunk::<u32>(&get_table_name(treename, KD_STR_LR), metadata.nbottom, 1);

    let perm_data =
        filemap.read_chunk::<u32>(&get_table_name(treename, KD_STR_PERM), metadata.ndata, 1);

    let bb_data = filemap.read_chunk::<T>(&get_table_name(treename, KD_STR_BB), 0, 1);

    if bb_data.is_ok() {
        todo!("nothing is implemented for bounding boxes");
    //     let nbb_old = (metadata.nnodes + 1) / 2 - 1;
    //     let nbb_new = metadata.nnodes;

    //     if bb_chunk.nrows == nbb_new {
    //     } else if bb_chunk.nrows == nbb_old {
    //         log::warn!(
    //             "This file contains an old, buggy {} extension, it has {} rather than {} items. This will probably cause problems!",
    //             bb_chunk.tablename, nbb_old, nbb_new
    //         );
    //     } else {
    //         log::error!(
    //             "bounding box table {} should contain either {} (new) or {} (old) bounding boxes, but it has {}",
    //             bb_chunk.tablename, nbb_old, nbb_new, bb_chunk.nrows
    //         );
    //         return Err(Error::InvalidTableDimmensions);
    //     }
    }

    let split_data = filemap.read_chunk::<T>(
        &get_table_name(treename, KD_STR_SPLIT),
        metadata.ninterior,
        1,
    );

    let splitdim_data = filemap.read_chunk::<u8>(
        &get_table_name(treename, KD_STR_SPLITDIM),
        metadata.ninterior,
        1,
    );

    let data_data = filemap.read_chunk::<D>(
        &get_table_name(treename, KD_STR_DATA),
        metadata.ndata,
        metadata.ndim,
    );

    let range_tablename =get_table_name(treename, KD_STR_RANGE);
    let kdr = filemap
        .read_chunk::<f64>(
            range_tablename.as_str(),
            metadata.ndim * 2 + 1,
            1,
        )
        .map(|range| {
            let scale_idx = metadata.ndim * 2;
            let scale = range[scale_idx];

            let kdr = KDRange {
                range: range[..metadata.ndim]
                    .iter()
                    .zip(&range[metadata.ndim..scale_idx])
                    .map(|(a, b)| (*a, *b))
                    .collect::<Vec<_>>(),
                scale,
                invscale: 1.0 / scale,
            };
            log::trace!("range({}): {:?}", range_tablename, kdr);
            kdr
        });

    if split_data.is_ok() {
        if splitdim_data.is_ok() {
            let splitmask = u32::MAX;
        } else {
            let mask = compute_splitbits(metadata.ndim as u32);
        }
    }

    // Ok(KDTree {
    //     metadata,
    //     bb: todo!(),
    //     split: bytemuck::cast_slice(a)
    //     data: bytemuck::cast_slice(data_data),
    // })
    todo!()
}

fn compute_splitbits(ndim: u32) -> DimSplit {
    let mut bits = 0u8;
    let mut val = 1;
    while val < ndim {
        bits += 1;
        val *= 2;
    }

    let dimmask = val - 1;
    DimSplit {
        dimbits: bits,
        dimmask,
        splitmask: !dimmask,
    }
}

fn get_table_name(name: Option<&str>, table: &str) -> String {
    if let Some(name) = name {
        format!("{}_{}", table, name)
    } else {
        format!("{}", table)
    }
}

fn find_tree<'a>(
    hdu: &'a [HDU],
    treename: Option<&str>,
) -> Result<(&'a HDU<'a>, KdtMetadata), Error> {
    if let Some(treename) = treename {
        for h in hdu.iter().skip(1) {
            if let Ok(name) = header_get_str(&h.header, "KDT_NAME") {
                if name == treename {
                    return Ok((h, is_tree_header_ok(&h.header, TreeStyle::New)?));
                }
            }
        }
    }
    Ok((&hdu[0], is_tree_header_ok(&hdu[0].header, TreeStyle::Old)?))
}

fn get_primary_header() {
    todo!()
}

fn header_get_int<K: MustKeyword>(header: &Header, keyword: K) -> Result<i64, Error> {
    let keyword = keyword.to_keyword();
    header
        .integer_value_of(&keyword)
        .map_err(|cause| Error::FitsMissingKey { keyword, cause })
}

fn header_get_bool<K: MustKeyword>(header: &Header, keyword: K) -> Result<bool, Error> {
    let keyword = keyword.to_keyword();
    header
        .value_of(&keyword)
        .and_then(|v| match v {
            Value::Logical(b) => Ok(b),
            Value::CharacterString("t") => Ok(true),
            Value::CharacterString("f") => Ok(false),
            Value::CharacterString("T") => Ok(true),
            Value::CharacterString("F") => Ok(false),
            Value::Integer(1) => Ok(true),
            Value::Integer(0) => Ok(false),
            _ => Err(ValueRetrievalError::NotABool),
        })
        .map_err(|cause| Error::FitsMissingKey { keyword, cause })
}

fn header_get_str<'a, K: MustKeyword>(header: &'a Header, keyword: K) -> Result<&'a str, Error> {
    let keyword = keyword.to_keyword();
    header
        .str_value_of(&keyword)
        .map_err(|cause| Error::FitsMissingKey { keyword, cause })
}

fn is_tree_header_ok(header: &Header, style: TreeStyle) -> Result<KdtMetadata, Error> {
    let (ndim, ndata, nnodes) = match style {
        TreeStyle::Old => (
            header_get_int(header, "NDIM")? as usize,
            header_get_int(header, "NDATA")? as usize,
            header_get_int(header, "NNODES")? as usize,
        ),
        TreeStyle::New => (
            header_get_int(header, "KDT_NDIM")? as usize,
            header_get_int(header, "KDT_NDAT")? as usize,
            header_get_int(header, "KDT_NNOD")? as usize,
        ),
    };

    let kdt_type = KdtType {
        ext: KdtExt::from_str(header_get_str(header, "KDT_EXT")?).unwrap_or(KdtExt::F64),
        tree: KdtTree::from_str(header_get_str(header, "KDT_INT")?)?,
        data: KdtData::from_str(header_get_str(header, "KDT_DATA")?)?,
    };

    fits_check_endian(header)?;

    let has_linear_lr = header_get_bool(header, "KDT_LINL").unwrap_or(false);
    let nlevels = kdtree_nnodes_to_nlevels(nnodes as u32);
    let nbottom = (nnodes + 1) / 2;
    let ninterior = nnodes - nbottom;

    Ok(KdtMetadata {
        kdt_type,
        has_linear_lr,
        ndata,
        ndim,
        nnodes,
        nbottom,
        ninterior,
        nlevels,
    })
}

fn fits_check_endian(header: &Header) -> Result<(), Error> {
    let filestr = header_get_str(header, "ENDIAN")?;
    let localstr = fits_get_endian_string();
    if filestr != localstr.as_str() {
        log::error!(
            "file endian {:?}, this machine endian: {:?}",
            filestr,
            localstr
        );
        return Err(Error::EndianMismatch);
    }
    Ok(())
}

fn fits_get_endian_string() -> String {
    // TODO const or once_cell?
    fits_generate_endian_string::<byteorder::NativeEndian>()
}

fn fits_generate_endian_string<O: byteorder::ByteOrder>() -> String {
    use byteorder::WriteBytesExt;
    let mut wtr = [0u8; 4];
    (&mut wtr[..]).write_u32::<O>(FITS_ENDIAN_DETECTOR).unwrap();
    format!(
        "{:02x}:{:02x}:{:02x}:{:02x}",
        wtr[0], wtr[1], wtr[2], wtr[3],
    )
}

fn kdtree_nnodes_to_nlevels(nnodes: u32) -> u8 {
    leftmost_significant_bit(nnodes + 1)
}

/// How far left is the most significant bit?
/// 0b0001 would be the 0-bit
/// 0b0011 would be the 1-bit
/// 0b1010 would be the 3-bit
#[inline]
fn leftmost_significant_bit(x: u32) -> u8 {
    assert!(
        x > 0,
        "x must be greater than 0 so that there is at least one flipped bit"
    );
    31 - x.leading_zeros() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_big() {
        let s = fits_generate_endian_string::<byteorder::BigEndian>();
        assert_eq!(s.as_str(), "01:02:03:04");
    }
    #[test]
    fn order_little() {
        let s = fits_generate_endian_string::<byteorder::LittleEndian>();
        assert_eq!(s.as_str(), "04:03:02:01");
    }
}
