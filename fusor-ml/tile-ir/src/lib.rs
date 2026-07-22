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

mod analysis;
mod ir;
mod kernel_builder;
mod lower;
mod quantized;
pub mod tile;

pub use ir::{
    AxisGroup, Buffer, BufferAccess, BufferDecl, CoopMatrixRole, ElementType, F32Bits, KernelIr,
    Layout, MemoryLevel, MultiFlattenMap, ScalarElement, Shape, StorageView, SubAxis, TileBinaryOp,
    TileCompareOp, TileLiteral, TileReduceOp, TileUnaryOp, WorkgroupAxis,
};
// Raw IR node tree (Accumulator, Addr, Builtin, CoopSrc, Expr, ExprKind, Local,
// LocalDecl, Node, ReduceKind, Source, Stmt, Tile, TileDecl) is intentionally not
// re-exported: consumers only build via `tile`/`KernelBuilder` and lower to an
// opaque `NagaKernel`. Internal code names these through `crate::ir::*`.
pub use analysis::{set_liveness_trace, BarrierSuggestion};
pub use kernel_builder::{KernelBuilder, KernelTensorRef};
pub use lower::{LowerError, NagaKernel};
pub use quantized::{GgmlQuantFormat, QuantizedMatrix};
pub use tile::{ByteArenaToken, CoopMatrixToken, SubgroupToken};

#[cfg(test)]
mod tests;
