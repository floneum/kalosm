use fusor_tile_ir::{Layout, MemoryLevel, Shape};

mod flash;
mod helpers;
mod mirostat;
mod rms_norm;
mod top_k;
mod types;

pub use flash::{flash_attention, flash_decode_small};
pub use mirostat::mirostat2;
pub use rms_norm::rms_norm_vec4;
pub use top_k::{top_k_chunk, top_k_exactness, top_k_merge};
pub use types::{
    FlashAttentionDims, FlashAttentionMeta, FlashDecodeSmallMeta, MergeTopKMeta, Mirostat2Meta,
    RmsNormVec4Meta, TensorMeta, TopKChunkMeta, TopKExactnessMeta,
};

/// The default rank-1 unit-stride layout used by tile-ir's pre-built kernels
/// for tensors whose offset/stride is encoded in the `Meta` struct itself.
/// Callers feed this into [`fusor_tile_ir::KernelTensorRef`] to attach a runtime
/// binding.
pub fn linear_storage_layout() -> Layout {
    Layout::strided(MemoryLevel::Storage, Shape::new([1]), &[1])
}
