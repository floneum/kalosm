//! Pre-built kernels for `fusor-tile-ir`.
//!
//! `fusor-tile-ir` contains the IR, lowerer, and generic tile builder. This
//! crate contains concrete kernels: dense matmul/GEMV, quantized matmul/GEMV,
//! dequantization, flash attention, top-k, RMS norm, and Mirostat sampling.
//!
//! ```
//! use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, F32};
//! use fusor_tile_ir_kernels::{qgemv, quantized_matrix};
//!
//! let ir = tile::build(|program| {
//!     let a = program.storage_read::<F32, 2>(Shape::new([1, 256]));
//!     let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 128);
//!     let y = program.storage_write::<F32, 2>(Shape::new([1, 128]));
//!     qgemv::<4, 64>(program, &a, &b, &y, 4, 1);
//! });
//! # let _ = ir;
//! ```
//!
//! For runtime-owned bindings, pair kernel constructors with
//! [`fusor_tile_ir::KernelBuilder`]:
//!
//! ```
//! use fusor_tile_ir::{
//!     GgmlQuantFormat, KernelBuilder, KernelTensorRef, Layout, MemoryLevel, Shape, F32,
//! };
//! use fusor_tile_ir_kernels::{qdequantize, quantized_matrix_for};
//!
//! let mut kb = KernelBuilder::<&'static str>::new();
//! let q = quantized_matrix_for(&mut kb, "matrix", GgmlQuantFormat::Q4K, 256, 4);
//! let layout = Layout::contiguous(MemoryLevel::Storage, Shape::new([1024]));
//! let y = kb.write::<F32, 1>(KernelTensorRef::new("output", layout));
//! qdequantize(kb.program(), &q, &y, 1);
//! let (_ir, bindings) = kb.finish();
//! assert_eq!(bindings, ["matrix", "output"]);
//! ```

mod dispatch;
mod grid;
mod kernels;
mod types;

pub use dispatch::{
    qgemv_cols_per_workgroup, qgemv_cols_per_workgroup_for_shape,
    qgemv_subgroups_per_workgroup_for_shape,
};
pub use kernels::{
    flash_attention, flash_decode_small, gemv, linear_storage_layout, matmul,
    matmul_with_epilogues, mirostat2, qdequantize, qgemv, qgemv_q4k_paired_2x2,
    qgemv_q4k_paired_2x4, qgemv_q4k_paired_4x1, qgemv_q4k_paired_4x2, qgemv_q4k_paired_4x4,
    qgemv_q4k_paired_8x1, qgemv_q4k_paired_8x2, qgemv_q4k_paired_ggml, qgemv_with_epilogue,
    qmatmul, qmatmul_with_epilogue, quantized_matrix, quantized_matrix_for, rms_norm_vec4,
    top_k_chunk, top_k_exactness, top_k_merge, FlashAttentionDims, FlashAttentionMeta,
    FlashDecodeSmallMeta, IntoQgemvEpilogues, MergeTopKMeta, Mirostat2, Mirostat2Meta,
    Q4KPairedGgml, RmsNormVec4, RmsNormVec4Meta, TensorMeta, TopKChunkMeta, TopKExactnessMeta,
};
pub use types::{DenseMatmulEpilogues, PairedEpilogue, QmatmulEpilogues, UnaryEpilogue};
