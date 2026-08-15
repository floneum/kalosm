//! Inference-time caches: KV, attention masks and rope tables.

pub mod kv;
pub mod mask;
pub mod rope;

pub use kv::{KvCache, TensorCache};
pub use mask::{AttentionMask, MaskCache};
pub use rope::RopeCache;

/// The mask attribute [`AttentionMask::Structural`] carries and
/// [`crate::composite::attention`] consumes, re-exported so a model crate
/// never has to name the IR crate.
pub use fusor2_ir::ir::level1::MaskKind;
