//! ModernBERT/Ettin encoder implementation.
//!
//! ModernBERT uses:
//! - RoPE (Rotary Position Embeddings)
//! - Pre-normalization with RMSNorm
//! - GeGLU activation in FFN
//! - No token type IDs

mod attention;
mod config;
mod feed_forward;
mod layer;
mod model;

pub use config::ModernBertConfig;
pub use model::ModernBertModel;
