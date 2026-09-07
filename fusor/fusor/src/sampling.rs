//! Sampling. These are inference-only, have no adjoint, and enter through
//! `Launch::Ext` + an `OpDef` with one declared cost row.
//!
//! Everything here is built out of ordinary facade ops on the caller's graph,
//! so a draw is a lazy device value: [`standard::sample`] resolves nothing and
//! hands back a `U32` token tensor a decode loop can consume directly.

pub(crate) mod mirostat2;
pub(crate) mod row;
pub(crate) mod standard;
pub(crate) mod top_k;

pub use mirostat2::Mirostat2Sampler;
pub use standard::{StandardSamplerParams, sample, sample_async};
pub use top_k::{GpuSampledToken, top_k_pairs};
