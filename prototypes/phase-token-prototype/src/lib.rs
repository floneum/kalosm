//! A prototype typed tile IR.
//!
//! Source kernels are built with [`tile::build`]. The only executable IR
//! operation is [`Op::TileProgram`]; conveniences such as dense matmul,
//! quantized matmul, quantized GEMV, dequantization, reductions, and softmax
//! are expressed by composing tile program expressions.

mod ir;
mod lower;
pub mod quantized;
pub mod tile;

pub use ir::{
    Block, BufferAccess, BufferDecl, BufferId, BufferRef, CoopOperandRole, DynamicOffset,
    ElementType, F32Bits, FlattenedMatrixMap, Im2ColNhwcMap, KernelIr, Layout, LoopOffset,
    MemoryLevel, Numeric, Op, Shape, StorageIndexMap, StorageView, Strides, TileBinaryOp,
    TileCompareOp, TileDecl, TileExpr, TileId, TileIndexExpr, TileLevel, TileLiteral, TileLoadExpr,
    TileMaskExpr, TileOrigin, TileProgramOp, TileQuantizedLoadExpr, TileReduceOp, TileRef,
    TileScalarExpr, TileStmt, TileUnaryOp, WorkgroupAxis, WorkgroupOffset, F16, F32, U32,
};
pub use lower::{LowerError, NagaKernel};
pub use quantized::{GgmlQuantFormat, QuantizedMatrix};

#[cfg(test)]
mod tests;
