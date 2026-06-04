//! Pre-built kernels for `fusor-tile-ir`.
//!
//! `fusor-tile-ir` contains the IR, lowerer, and generic tile builder. This
//! crate contains concrete kernels: dense matmul/GEMV, quantized matmul/GEMV,
//! dequantization, flash attention, top-k, RMS norm, and Mirostat sampling.

mod dispatch;
mod grid;
mod kernels;
mod types;

pub use dispatch::{
    qgemv_cols_per_workgroup, qgemv_cols_per_workgroup_for_shape,
    qgemv_subgroups_per_workgroup_for_shape,
};
pub use kernels::{
    batched_gemv_with_epilogues, batched_matmul_register_with_epilogues,
    batched_matmul_with_epilogues, flash_attention, flash_attention_tiled, flash_decode_small,
    flash_decode_split_partials, flash_decode_split_reduce, flash_outputs_per_workgroup,
    flash_tiled_dispatch_size, flash_tiled_outputs_per_workgroup, linear_storage_layout, mirostat2,
    qdequantize, qgemv_with_epilogue, qgemv_workgroup_f16_with_epilogue,
    qgemv_workgroup_storage_f16_with_epilogue, qgemv_workgroup_with_epilogue,
    qmatmul_with_epilogue, qmatmul_workgroup_f16_with_epilogues,
    qmatmul_workgroup_storage_f16_with_epilogues, qmatmul_workgroup_with_epilogues,
    quantized_matrix, quantized_matrix_for, rms_norm_vec4, softmax, softmax_partials,
    softmax_reduce, softmax_write, standard_sampler, top_k_chunk, top_k_exactness, top_k_merge,
    try_batched_coop_matmul, AccumCast, DenseCoopMatmulTile, DenseMatmulShape, DenseMatmulTensors,
    FlashAttentionDims, FlashAttentionMeta, FlashAttentionTensors, FlashDecodeSmallMeta,
    IntoQgemvEpilogues, MergeTopKMeta, Mirostat2, Mirostat2Meta, RmsNormVec4, RmsNormVec4Meta,
    SoftmaxMeta, StandardSampler, TensorMeta, TopKChunkMeta, TopKExactnessMeta,
};
pub use types::{
    DenseMatmulEpilogues, QmatmulEpilogues, QmatmulExtra, UnaryEpilogue, UnaryEpilogueWithExtras,
};
