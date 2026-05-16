use fusor_tile_ir::{Layout, MemoryLevel, Shape};

mod flash;
mod gemv;
mod helpers;
mod matmul;
mod mirostat;
mod qdequantize;
mod qgemv;
mod qgemv_paired_q4k;
mod qmatmul;
mod quantized_matrix;
mod rms_norm;
mod top_k;
mod types;

pub use flash::{flash_attention, flash_decode_small};
pub use gemv::gemv;
pub use matmul::{matmul, matmul_with_epilogues};
pub use mirostat::{mirostat2, Mirostat2};
pub use qdequantize::qdequantize;
pub use qgemv::{qgemv, qgemv_with_epilogue, IntoQgemvEpilogues};
pub use qgemv_paired_q4k::{
    qgemv_q4k_paired, qgemv_q4k_paired_cols_per_workgroup, qgemv_q4k_paired_tile_name,
    Q4KPairedGgml,
};
pub use qmatmul::{qmatmul, qmatmul_with_epilogue};
pub use quantized_matrix::{quantized_matrix, quantized_matrix_for};
pub use rms_norm::{rms_norm_vec4, RmsNormVec4};
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
