//! Inference-time caches: KV, attention masks and rope tables.

pub(crate) mod kv;
pub(crate) mod mask;
pub(crate) mod rope;

pub use kv::{KvCache, TensorCache};
pub use mask::{AttentionMask, MaskCache};
pub use rope::RopeCache;

/// The mask attribute [`AttentionMask::Structural`] carries and
/// [`crate::composite::attention`] consumes, re-exported so a model crate
/// never has to name the IR crate.
pub use fusor_ir::ir::launch::MaskKind;
