//! A tiny prototype for same-lifetime phase handles in a typed IR builder.
//!
//! The experiment here is deliberately small:
//!
//! - [`Phase<'_, '_, 'flow>`] is affine: it is moved through phase-changing API.
//! - [`Synced<'flow>`] is a witness that can only be produced by a barrier.
//! - [`Phase::range_step`] requires a body of the form
//!   `for<'iter> FnOnce(Phase<'_, '_, 'iter>, ...) -> Synced<'iter>`.
//! - [`Phase::partition`] runs its Rust closure once, but emits a structured
//!   [`PartitionOp`] whose child tile views are branded to the partition body.
//! - [`Phase::gemm`] is the higher-level TileLang/Triton-style operation; the
//!   middle-end can expand it into partitioned [`MmaOp`]s.
//! - [`KernelIr`] stores typed tile declarations plus a tree of [`Op`] blocks,
//!   not a flat event log.
//!
//! The loop body takes and returns a witness with the same lifetime, but it
//! cannot return the raw phase handle. It must return `Synced<'iter>`, which
//! only a sync method can create.
//!
//! Using a pending tile does not type check:
//!
//! ```compile_fail
//! use phase_token_prototype::{build, F32, Shape};
//!
//! build(|mut phase| {
//!     let src = phase.storage_tensor::<F32>(Shape::new([32]));
//!     let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
//!     let pending = phase.cooperative_load(tile, &src);
//!     pending.finish();
//! });
//! ```
//!
//! A loop body cannot simply return the phase handle without a barrier:
//!
//! ```compile_fail
//! use phase_token_prototype::build;
//!
//! build(|phase| {
//!     phase.range_step(|phase, _| phase, |phase| phase.finish())
//! });
//! ```
//!
//! A pending load cannot be dropped and still produce a finished kernel:
//!
//! ```compile_fail
//! use phase_token_prototype::{build, F32, Shape};
//!
//! build(|mut phase| {
//!     let src = phase.storage_tensor::<F32>(Shape::new([32]));
//!     let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
//!     let pending = phase.cooperative_load(tile, &src);
//!     drop(pending);
//!     phase.finish()
//! });
//! ```
//!
//! Once a body has produced its final `Synced` witness, it has consumed the
//! phase handle, so it cannot emit more IR afterward:
//!
//! ```compile_fail
//! use phase_token_prototype::{build, F32, Shape};
//!
//! build(|phase| {
//!     phase.range_step(
//!         |phase, _| {
//!             let synced = phase.sync_end();
//!             let _late = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
//!             synced
//!         },
//!         |phase| phase.finish(),
//!     )
//! });
//! ```
//!
//! A phase-branded tile from inside the loop body cannot escape to the
//! continuation after the loop:
//!
//! ```compile_fail
//! use phase_token_prototype::{build, F32, ReadyTile, Shape};
//!
//! build(|phase| {
//!     let mut leaked: Option<ReadyTile<'_, '_, F32>> = None;
//!
//!     phase.range_step(
//!         |mut phase, _| {
//!             let src = phase.storage_tensor::<F32>(Shape::new([32]));
//!             let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
//!             let pending = phase.cooperative_load(tile, &src);
//!             let (ready, phase) = pending.sync_tile();
//!             leaked = Some(ready);
//!             phase.sync_end()
//!         },
//!         |mut phase| {
//!             let out = phase.storage_tensor::<F32>(Shape::new([32]));
//!             phase.store_ready_to_storage(&leaked.unwrap(), &out);
//!             phase.finish()
//!         },
//!     )
//! });
//! ```
//!
//! A tile made ready before a loop can be used inside the loop body:
//!
//! ```
//! use phase_token_prototype::{build, F32, Shape};
//!
//! build(|mut phase| {
//!     let src = phase.storage_tensor::<F32>(Shape::new([32]));
//!     let out = phase.storage_tensor::<F32>(Shape::new([32]));
//!     let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
//!     let pending = phase.cooperative_load(tile, &src);
//!     let (ready, phase) = pending.sync_tile();
//!
//!     phase.range_step(
//!         |mut phase, _| {
//!             phase.store_ready_to_storage(&ready, &out);
//!             phase.sync_end()
//!         },
//!         |phase| phase.finish(),
//!     )
//! });
//! ```
//!
//! A partition child view cannot escape the partition body:
//!
//! ```compile_fail
//! use phase_token_prototype::{build, F32, ReadyTile, Shape, TileLevel};
//!
//! build(|mut phase| {
//!     let src = phase.storage_tensor::<F32>(Shape::new([32]));
//!     let out = phase.storage_tensor::<F32>(Shape::new([16]));
//!     let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
//!     let pending = phase.cooperative_load(tile, &src);
//!     let (ready, mut phase) = pending.sync_tile();
//!     let mut leaked: Option<ReadyTile<'_, '_, F32>> = None;
//!
//!     phase.partition(&ready, TileLevel::Subgroup, Shape::new([16]), |_, child| {
//!         leaked = Some(child);
//!     });
//!
//!     phase.store_ready_to_storage(&leaked.unwrap(), &out);
//!     phase.finish()
//! });
//! ```

mod api;
mod ir;
mod lower;

pub use api::{
    build, Clean, KernelBuilder, KernelDone, Numeric, Pending, Pending2, PendingTile,
    PendingTilePair, Phase, ReadyTile, RegTile, StorageTensor, Synced, UninitTile, F32,
};
pub use ir::{
    BarrierOp, BarrierScope, Block, BufferAccess, BufferDecl, BufferId, BufferRef,
    CooperativeLoadOp, Dim, DynamicOffset, ElementType, FillTileOp, FillValue, GemmOp, GemmTiling,
    GemvOp, KernelIr, Layout, LoopKind, LoopOffset, LoopOp, MemoryLevel, MmaBackend, MmaOp, Op,
    PartitionBinding, PartitionOp, Shape, StorageView, StoreTileOp, Strides, TileDecl, TileId,
    TileLevel, TileOrigin, TileRef, ViewMapping, WorkgroupAxis, WorkgroupOffset,
};
pub use lower::{LowerError, NagaKernel};

#[cfg(test)]
mod tests;
