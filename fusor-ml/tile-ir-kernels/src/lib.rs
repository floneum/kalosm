//! Pre-built kernels for `fusor-tile-ir`.
//!
//! `fusor-tile-ir` itself contains only the IR types, the lowerer, and the
//! generic kernel-builder primitives (`Program::program_grid`,
//! `TileBlock::load`, etc.). Every concrete kernel — qmatmul, qgemv, dense
//! matmul, gemv, qdequantize, flash attention, top-k, rms-norm, mirostat —
//! lives in this crate as free functions over `&mut Program`.

mod dispatch;
mod grid;
mod kernels;
mod program_kernels;
mod program_qgemv;
mod types;

pub use dispatch::{
    q4k_default_large, q4k_default_mid, q4k_default_tall, q4k_large_override, q4k_mid_override,
    q4k_tall_override, q6k_default_large, q6k_default_tall, q6k_large_override, q6k_tall_override,
    qmatmul_path, QgemvShapeQ4K, QgemvShapeQ6K, QmatmulPath,
};
pub use grid::{
    dot4_sum, q4k_ggml_activations, q4k_lane_decomposition, qgemv_grid, store_qgemv_sums,
    Q4KGgmlActivations, Q4KLane, QgemvGrid,
};
pub use kernels::{
    flash_attention, flash_decode_small, linear_storage_layout, mirostat2, rms_norm_vec4,
    top_k_chunk, top_k_exactness, top_k_merge, FlashAttentionDims, FlashAttentionMeta,
    FlashDecodeSmallMeta, MergeTopKMeta, Mirostat2Meta, RmsNormVec4Meta, TensorMeta, TopKChunkMeta,
    TopKExactnessMeta,
};
pub use program_kernels::{
    gemv, matmul, qdequantize, qgemv, qgemv_with_epilogue, qmatmul, qmatmul_dispatch,
    qmatmul_options, qmatmul_perf, qmatmul_tile, quantized_matrix, quantized_matrix_for,
    IntoQgemvEpilogues,
};
pub use program_qgemv::{
    qgemv_perf, qgemv_q4k_dispatch, qgemv_q4k_ggml, qgemv_q4k_paired_2x2, qgemv_q4k_paired_2x4,
    qgemv_q4k_paired_4x1, qgemv_q4k_paired_4x2, qgemv_q4k_paired_4x4, qgemv_q4k_paired_8x1,
    qgemv_q4k_paired_8x2, qgemv_q4k_paired_ggml, qgemv_q6k_dispatch, qgemv_q6k_ggml, qgemv_tile,
};
pub use types::{
    apply_optional_epilogue, PairedEpilogue, PairedEpiloguePreset, QmatmulEpilogues, UnaryEpilogue,
};
