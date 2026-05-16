//! Typed tile IR and Naga lowering for Fusor kernels.
//!
//! Use [`tile::build`] when the runtime bindings are managed elsewhere, or
//! [`KernelBuilder`] when a caller also needs the binding list paired with the
//! generated IR. Each kernel body is a single tile program built from
//! per-lane tile expressions and lowered to a validated Naga compute module.
//!
//! ```
//! use fusor_tile_ir::{tile, Shape, F32};
//!
//! let ir = tile::build(|program| {
//!     let input = program.storage_read::<F32, 2>(Shape::new([1, 128]));
//!     let output = program.storage_write::<F32, 2>(Shape::new([1, 128]));
//!
//!     program.program_grid::<128>([1, 1, 1], |program| {
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
    AxisGroup, Bool, CoopElement, CoopMatrixRole, ElementType, F32Bits, FloatElement, KernelIr,
    Layout, MemoryLevel, MultiFlattenMap, Numeric, ScalarElement, ScalarMarker, Shape, StorageView,
    SubAxis, TileBinaryOp, TileCompareOp, TileLiteral, TileReduceOp, TileRef, TileUnaryOp, Vector,
    WorkgroupAxis, F16, F32, U32,
};
pub use kernel_builder::{KernelBuilder, KernelTensorRef};
pub use lower::{LowerError, NagaKernel};
pub use quantized::{GgmlQuantFormat, QuantizedMatrix};

#[cfg(test)]
mod tests;
