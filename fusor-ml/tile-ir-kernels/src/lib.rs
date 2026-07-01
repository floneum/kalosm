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
    qgemv_cols_per_workgroup, qgemv_cols_per_workgroup_for_shape,
    qgemv_subgroups_per_workgroup_for_shape, SubgroupConfig,
};
pub use kernels::{
    linear_storage_layout, qgemv_with_epilogue, qgemv_workgroup_f16_with_epilogue,
    qgemv_workgroup_storage_f16_with_epilogue, qgemv_workgroup_with_epilogue,
    qmatmul_with_epilogue, qmatmul_workgroup_f16_with_epilogues,
    qmatmul_workgroup_storage_f16_with_epilogues, qmatmul_workgroup_with_epilogues,
    quantized_matrix, quantized_matrix_for, try_batched_coop_matmul, AccumCast,
    DenseCoopMatmulConfig, DenseCoopMatmulTile, DenseMatmulShape, DenseMatmulTensors,
    IntoQgemvEpilogues,
};
pub use types::{
    cooperative_store_layout_supported, DenseMatmulEpilogues, QmatmulEpilogues, QmatmulExtra,
    UnaryEpilogue, UnaryEpilogueWithExtras,
};
