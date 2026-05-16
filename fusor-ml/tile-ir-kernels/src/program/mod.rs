//! Program-level kernel constructors.

mod gemv;
mod matmul;
mod qdequantize;
mod qgemv;
mod qgemv_paired_q4k;
mod qmatmul;
mod quantized_matrix;

pub use gemv::gemv;
pub use matmul::{matmul, matmul_with_epilogues};
pub use qdequantize::qdequantize;
pub use qgemv::{IntoQgemvEpilogues, qgemv, qgemv_with_epilogue};
pub use qgemv_paired_q4k::{
    Q4KPairedGgml, qgemv_q4k_paired_2x2, qgemv_q4k_paired_2x4, qgemv_q4k_paired_4x1,
    qgemv_q4k_paired_4x2, qgemv_q4k_paired_4x4, qgemv_q4k_paired_8x1, qgemv_q4k_paired_8x2,
    qgemv_q4k_paired_ggml,
};
pub use qmatmul::{qmatmul, qmatmul_with_epilogue};
pub use quantized_matrix::{quantized_matrix, quantized_matrix_for};
