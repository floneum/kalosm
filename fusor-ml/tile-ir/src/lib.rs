//! Typed tile IR and lowering for Fusor kernels.
//!
//! Source kernels are built with [`tile::build`]. Each kernel body is a single
//! [`TileProgramOp`]; conveniences such as dense matmul, quantized matmul,
//! quantized GEMV, dequantization, reductions, and softmax are expressed by
//! composing tile program expressions.

mod dispatch;
mod ir;
pub mod kernel_builder;
pub mod kernels;
mod lower;
pub mod quantized;
pub mod tile;

pub use ir::{
    Bool, BufferAccess, BufferDecl, BufferId, BufferRef, CoopOperandRole, ElementType, F32Bits,
    F32Vec4, FlattenedMatrixMap, Im2ColNhwcMap, KernelIr, Layout, LoadSource, LocalDecl, LocalId,
    LocalRef, MemoryLevel, Numeric, Shape, StorageIndexMap, StorageView, Strides, TileBinaryOp,
    TileCompareOp, TileDecl, Expr, TileId, TileIndexedStoreStmt, TileLinearLoadExpr, TileLiteral,
    TileLoadExpr, TileProgramOp, TileReduceOp, TileRef, TileStmt, TileUnaryOp, WorkgroupAxis, F16,
    F32, U32,
};
pub use kernel_builder::{KernelBuilder, KernelTensorRef};
pub use kernels::{
    FlashAttentionDims, FlashAttentionMeta, FlashDecodeSmallMeta, MergeTopKMeta, Mirostat2Meta,
    RmsNormVec4Meta, TensorMeta, TopKChunkMeta, TopKExactnessMeta,
};
pub use lower::{LowerError, NagaKernel};
pub use quantized::{GgmlQuantFormat, QuantizedMatrix};
pub use tile::PairedActivation;

#[cfg(test)]
mod tests;
