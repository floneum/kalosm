//! Qwen embedding model (encoder-only): separate Q/K/V + optional q/k norm,
//! RoPE, pre-norm RMSNorm, SwiGLU FFN. Built on the shared
//! [`fusor::TransformerBlock`].

mod model;

pub use model::QwenEmbeddingModel;
