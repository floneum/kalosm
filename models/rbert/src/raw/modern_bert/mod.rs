//! ModernBERT/Ettin encoder implementation.
//!
//! ModernBERT uses:
//! - RoPE (Rotary Position Embeddings), with separate global/local bases
//! - Alternating global and sliding-window local attention
//! - Pre-normalization with LayerNorm
//! - GeGLU activation in FFN
//! - No token type IDs
//!
//! Each layer is the shared [`fusor::TransformerBlock`] (fused QKV + RoPE +
//! pre-norm LayerNorm + GeGLU); the sliding-window local attention lives in
//! [`layer`] alongside the block.

mod config;
mod layer;
mod model;

pub use model::ModernBertModel;
