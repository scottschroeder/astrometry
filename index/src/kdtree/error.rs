use fits_rs::types::{Keyword, ValueRetrievalError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("we do not match the endian-ness of the data")]
    EndianMismatch,
    #[error("could not find table with name: {0}")]
    MissingTable(String),
    #[error("we got a value `{0}` we didn't expect")]
    InvalidValue(String),
    #[error("the actual table size didn't match what we needed")]
    InvalidTableDimmensions,
    #[error("could not parse type `{0}` for `{1}")]
    ParseType(&'static str, String),
    #[error("could not get `{}` from fits header: {:?}", .keyword, .cause)]
    FitsMissingKey {
        keyword: Keyword,
        cause: ValueRetrievalError,
    },
}
