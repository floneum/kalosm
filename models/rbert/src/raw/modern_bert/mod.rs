//! ModernBERT/Ettin encoder implementation.
//!
//! ModernBERT uses:
//! - RoPE (Rotary Position Embeddings), with separate global/local bases
//! - Alternating global and sliding-window local attention
//! - Pre-normalization with LayerNorm
//! - GeGLU activation in FFN
//! - No token type IDs

mod attention;
mod config;
mod feed_forward;
mod layer;
mod model;

pub use model::ModernBertModel;
