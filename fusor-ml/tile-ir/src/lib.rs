//! Typed tile IR and lowering for Fusor kernels.
//!
//! Source kernels are built with [`tile::build`]. Each kernel body is a single
//! [`TileProgramOp`]; conveniences such as dense matmul, quantized matmul,
//! quantized GEMV, dequantization, reductions, and softmax are expressed by
//! composing tile program expressions.

mod ir;
mod kernel_builder;
mod lower;
mod quantized;
pub mod tile;

pub use ir::{
    AxisGroup, Bool, ElementType, F32Bits, F32Vec4, KernelIr, Layout, MemoryLevel, MultiFlattenMap,
    Numeric, Shape, StorageView, SubAxis, TileBinaryOp, TileCompareOp, TileLiteral, TileReduceOp,
    TileRef, TileUnaryOp, WorkgroupAxis, F16, F32, U32,
};
pub use kernel_builder::{KernelBuilder, KernelTensorRef};
pub use lower::{LowerError, NagaKernel};
pub use quantized::{GgmlQuantFormat, QuantizedMatrix};

#[cfg(test)]
mod tests;
