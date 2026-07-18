use fusor_tile_ir::{Layout, MemoryLevel, Shape};

mod attention;
mod helpers;
mod matmul;
mod qgemv;
mod qgemv_q4k_ggml;
mod qgemv_q6k;
mod qmatmul;
mod qmatmul_workgroup;
mod quantized_matrix;

pub use attention::{flash_attention_f32, flash_attention_supported, FlashAttentionShape};
pub use helpers::AccumCast;
pub use matmul::{
    fma_matmul_f32,
    coop_tile_entries, merged_split_k_combine, split_k_combine, try_batched_coop_matmul,
    try_batched_coop_matmul_split_k, try_merged_coop_matmul, CoopTileEntry,
    DenseCoopMatmulConfig, DenseCoopMatmulTile, DenseMatmulShape, DenseMatmulTensors,};
pub use qgemv::{qgemv_with_epilogue, IntoQgemvEpilogues};
pub use qmatmul::qmatmul_with_epilogue;
pub use qmatmul_workgroup::{
    qgemv_workgroup_f16_with_epilogue, qgemv_workgroup_storage_f16_with_epilogue,
    qgemv_workgroup_with_epilogue, qmatmul_workgroup_f16_with_epilogues,
    qmatmul_workgroup_storage_f16_with_epilogues, qmatmul_workgroup_with_epilogues,
};
pub use quantized_matrix::{quantized_matrix, quantized_matrix_for};

/// The default rank-1 unit-stride layout used by tile-ir's pre-built kernels
/// for tensors whose offset/stride is encoded in the `Meta` struct itself.
/// Callers feed this into [`fusor_tile_ir::KernelTensorRef`] to attach a runtime
/// binding.
pub fn linear_storage_layout() -> Layout {
    Layout::strided(MemoryLevel::Storage, Shape::new([1]), &[1])
}
