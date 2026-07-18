//! Pre-built kernels for `fusor-tile-ir`.
//!
//! `fusor-tile-ir` contains the IR, lowerer, and generic tile builder. This
//! crate contains concrete kernels: cooperative dense matmul, quantized
//! matmul/GEMV, top-k, and Mirostat sampling.

mod dispatch;
mod grid;
mod kernels;
mod types;

pub use dispatch::{
    qgemv_selected_shape,
    SubgroupConfig,
};
pub use kernels::{
    coop_tile_entries, flash_attention_bwd_supported, flash_attention_dispatch,
    flash_attention_f32, flash_attention_supported, flash_bwd_kv_dispatch, flash_bwd_kv_f32,
    flash_bwd_q_dispatch, flash_bwd_q_f32, flash_lse_dispatch, flash_lse_f32,
    fma_matmul_f32, linear_storage_layout, qgemv_with_epilogue,
    qgemv_workgroup_f16_with_epilogue, qgemv_workgroup_storage_f16_with_epilogue,
    qgemv_workgroup_with_epilogue, qmatmul_with_epilogue, qmatmul_workgroup_f16_with_epilogues,
    qmatmul_workgroup_storage_f16_with_epilogues, qmatmul_workgroup_with_epilogues,
    merged_split_k_combine, quantized_matrix, quantized_matrix_for, split_k_combine,
    try_batched_coop_matmul, try_batched_coop_matmul_split_k, try_merged_coop_matmul,
    AccumCast, CoopTileEntry, DenseCoopMatmulConfig, DenseCoopMatmulTile, DenseMatmulShape,
    DenseMatmulTensors, FlashAttentionLayouts, FlashAttentionShape, FlashBwdLayouts,
    FlashMaskLayout, FlashOperandLayout, FlashRowLayout, IntoQgemvEpilogues,
};
pub use types::{
    cooperative_store_layout_supported, DenseMatmulEpilogues, QmatmulEpilogues, QmatmulExtra,
    UnaryEpilogue, UnaryEpilogueWithExtras,
};
