//! mDeBERTa-v3 encoder for GLiNER-RelEx.
//!
//! mDeBERTa uses disentangled attention with relative position embeddings,
//! which differs from ModernBERT's RoPE-based attention.

mod attention;
mod config;
mod feed_forward;
mod layer;
mod model;

pub use model::MDebertaModel;
