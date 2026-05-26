use fusor_tile_ir::{Layout, MemoryLevel, Shape};

mod flash;
mod helpers;
mod matmul;
mod mirostat;
mod qdequantize;
mod qgemv;
mod qgemv_paired_q4k;
mod qmatmul;
mod qmatmul_workgroup;
mod quantized_matrix;
mod rms_norm;
mod softmax;
mod top_k;
mod types;

pub use flash::{
    flash_attention, flash_attention_tiled, flash_decode_small, flash_decode_split_partials,
    flash_decode_split_reduce, flash_outputs_per_workgroup, flash_tiled_dispatch_size,
    flash_tiled_outputs_per_workgroup, FlashAttentionTensors,
};
pub use helpers::AccumCast;
pub use matmul::{
    batched_gemv_with_epilogues, batched_matmul_register_with_epilogues,
    batched_matmul_with_epilogues, try_batched_coop_matmul, DenseCoopMatmulTile, DenseMatmulShape,
    DenseMatmulTensors,
};
pub use mirostat::{mirostat2, Mirostat2};
pub use qdequantize::qdequantize;
pub use qgemv::{qgemv_with_epilogue, IntoQgemvEpilogues};
pub use qgemv_paired_q4k::{
    qgemv_q4k_paired, qgemv_q4k_paired_dispatch, Q4KPairedGgml, Q4KPairedShape,
};
pub use qmatmul::qmatmul_with_epilogue;
pub use qmatmul_workgroup::{qgemv_workgroup_with_epilogue, qmatmul_workgroup_with_epilogues};
pub use quantized_matrix::{quantized_matrix, quantized_matrix_for};
pub use rms_norm::{rms_norm_vec4, RmsNormVec4};
pub use softmax::{softmax, softmax_partials, softmax_reduce, softmax_write};
pub use top_k::{top_k_chunk, top_k_exactness, top_k_merge};
pub use types::{
    FlashAttentionDims, FlashAttentionMeta, FlashDecodeSmallMeta, MergeTopKMeta, Mirostat2Meta,
    RmsNormVec4Meta, SoftmaxMeta, TensorMeta, TopKChunkMeta, TopKExactnessMeta,
};

/// The default rank-1 unit-stride layout used by tile-ir's pre-built kernels
/// for tensors whose offset/stride is encoded in the `Meta` struct itself.
/// Callers feed this into [`fusor_tile_ir::KernelTensorRef`] to attach a runtime
/// binding.
pub fn linear_storage_layout() -> Layout {
    Layout::strided(MemoryLevel::Storage, Shape::new([1]), &[1])
}
