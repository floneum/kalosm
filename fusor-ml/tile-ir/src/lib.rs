//! Typed tile IR and Naga lowering for Fusor kernels.
//!
//! Use [`tile::build`] when the runtime bindings are managed elsewhere, or
//! [`KernelBuilder`] when a caller also needs the binding list paired with the
//! generated IR. Each kernel body is a single tile program built from
//! per-lane tile expressions and lowered to a validated Naga compute module.
//!
//! ```
//! use fusor_tile_ir::{tile, Shape, ScalarElement};
//!
//! let ir = tile::build(|program| {
//!     let f32 = ScalarElement::F32.element();
//!     let input = program.storage_read(f32, Shape::new([1, 128]));
//!     let output = program.storage_write(f32, Shape::new([1, 128]));
//!
//!     program.program_grid(128, [1, 1, 1], |program| {
//!         let lane = program.lane();
//!         let mask = lane.clone().lt(128u32);
//!         let value = program.load(input.at((0u32, lane.clone())), mask.clone(), 0.0);
//!         program.store(output.at((0u32, lane)), value, mask);
//!     });
//! });
//!
//! let _module = ir.lower_to_naga()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod ir;
mod kernel_builder;
mod lower;
mod quantized;
pub mod tile;

pub use ir::{
    Accumulator, Addr, AxisGroup, Buffer, BufferAccess, BufferDecl, Builtin, CoopMatrixRole,
    CoopSrc, ElementType, Expr, ExprKind, F32Bits, KernelIr, Layout, Local, LocalDecl, MemoryLevel,
    MultiFlattenMap, Node, ReduceKind, ScalarElement, Shape, Source, Stmt, StorageView, SubAxis,
    Tile, TileBinaryOp, TileCompareOp, TileDecl, TileLiteral, TileReduceOp, TileUnaryOp,
    WorkgroupAxis,
};
pub use kernel_builder::{KernelBuilder, KernelTensorRef};
pub use lower::{LowerError, NagaKernel, WgslExtensions};
pub use quantized::{GgmlQuantFormat, QuantizedMatrix};

#[cfg(test)]
mod tests;
