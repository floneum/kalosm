#![warn(missing_docs)]
#![allow(clippy::type_complexity)]
#![doc = include_str!("../README.md")]

#[cfg(any(
    feature = "documents",
    feature = "fs-documents",
    feature = "web-documents",
    feature = "scrape"
))]
pub mod context;
#[cfg(feature = "chunking")]
pub mod search;
#[cfg(feature = "vector-db")]
pub mod vector_db;

pub use kalosm_language_model;
#[cfg(feature = "llama")]
pub use kalosm_llama;
pub use kalosm_sample;
#[cfg(feature = "bert")]
pub use rbert;

/// A prelude of commonly used items in kalosm-language
pub mod prelude {
    #[cfg(any(
        feature = "documents",
        feature = "fs-documents",
        feature = "web-documents",
        feature = "scrape"
    ))]
    pub use crate::context::*;
    #[cfg(feature = "chunking")]
    pub use crate::search::*;
    #[cfg(feature = "vector-db")]
    pub use crate::vector_db::*;
    pub use futures_util::StreamExt as _;
    pub use kalosm_language_model::*;
    #[cfg(feature = "llama")]
    pub use kalosm_llama::{Llama, LlamaBuilder, LlamaSession, LlamaSource};
    pub use kalosm_sample::*;
    pub use kalosm_streams::text_stream::*;
    #[cfg(feature = "bert")]
    pub use rbert::{Bert, BertBuilder, BertSource};
    #[cfg(any(feature = "chunking", feature = "scrape"))]
    pub use scraper::Html;
}
