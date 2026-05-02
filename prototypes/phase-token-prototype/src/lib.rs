//! A prototype typed tile IR.
//!
//! Source kernels are built with [`tile::build`]. The only executable IR
//! operation is [`Op::TileProgram`]; conveniences such as dense matmul,
//! quantized matmul, quantized GEMV, dequantization, reductions, and softmax
//! are expressed by composing tile program expressions.

mod ir;
mod lower;
pub mod tile;

pub use ir::{
    BlockDequantId, Block, BufferAccess, BufferDecl, BufferId, BufferRef, CoopAccDecl, CoopAccId,
    CoopFragmentId, DynamicOffset, ElementType, F32Bits, FlattenedMatrixMap, GgmlQuantFormat,
    Im2ColNhwcMap, KernelIr, Layout, LoopFoldGroup, LoopFoldGroupId, LoopOffset, MemoryLevel,
    Numeric, Op, PinId, QuantizedMatrix, Shape, StorageIndexMap, StorageView, Strides,
    SubgroupStmt, TileBinaryOp, TileCompareOp, TileDecl, TileExpr, TileId, TileIndexExpr, TileLevel,
    TileLiteral, TileLoadExpr, TileMaskExpr, TileOrigin, TileProgramOp, TileQuantizedLoadExpr,
    TileReduceOp, TileRef, TileScalarExpr, TileStoreProgramOp, TileUnaryOp, WorkgroupAxis,
    WorkgroupOffset, F16, F32, U32,
};
pub use lower::{LowerError, NagaKernel};

#[cfg(test)]
mod tests;
