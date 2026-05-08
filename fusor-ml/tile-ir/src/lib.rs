//! Typed tile IR and lowering for Fusor kernels.
//!
//! Source kernels are built with [`tile::build`]. The only executable IR
//! operation is [`Op::TileProgram`]; conveniences such as dense matmul,
//! quantized matmul, quantized GEMV, dequantization, reductions, and softmax
//! are expressed by composing tile program expressions.

mod ir;
pub mod kernels;
mod lower;
pub mod quantized;
pub mod tile;

pub use ir::{
    Block, Bool, BufferAccess, BufferDecl, BufferId, BufferRef, CoopOperandRole, DynamicOffset,
    ElementType, F32Bits, F32Vec4, FlattenedMatrixMap, Im2ColNhwcMap, KernelIr, Layout, LocalDecl,
    LocalId, LocalRef, LoopOffset, MemoryLevel, Numeric, Op, Shape, StorageIndexMap, StorageView,
    Strides, TileBinaryOp, TileCompareOp, TileDecl, TileExpr, TileId, TileIndexExpr, TileLevel,
    TileLinearLoadExpr, TileLinearStoreStmt, TileLiteral, TileLoadExpr, TileMaskExpr, TileOrigin,
    TileProgramOp, TileQuantizedLoadExpr, TileReduceOp, TileRef, TileScalarExpr, TileStmt,
    TileUnaryOp, TileVec4LoadExpr, TileVec4StoreStmt, WorkgroupAxis, WorkgroupOffset, F16, F32,
    U32,
};
pub use kernels::{
    FlashAttentionDims, FlashAttentionMeta, FlashDecodeSmallMeta, MergeTopKMeta, Mirostat2Meta,
    RmsNormVec4Meta, TensorMeta, TopKChunkMeta, TopKExactnessMeta,
};
pub use lower::{LowerError, NagaKernel};
pub use quantized::{GgmlQuantFormat, QuantizedMatrix};

#[cfg(test)]
mod tests;
