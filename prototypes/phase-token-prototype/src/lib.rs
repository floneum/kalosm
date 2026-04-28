//! A tiny prototype for same-lifetime phase handles in an IR builder.
//!
//! The experiment here is deliberately small:
//!
//! - [`Phase<'_, '_, 'flow>`] is affine: it is moved through phase-changing API.
//! - [`Synced<'flow>`] is a witness that can only be produced by a barrier.
//! - [`Phase::range_step`] requires a body of the form
//!   `for<'iter> FnOnce(Phase<'_, '_, 'iter>, ...) -> Synced<'iter>`.
//!
//! The loop body takes and returns a witness with the same lifetime, but it
//! cannot return the raw phase handle. It must return `Synced<'iter>`, which
//! only a sync method can create.
//!
//! Reading a pending tile does not type check:
//!
//! ```compile_fail
//! use phase_token_prototype::{build, F32};
//!
//! build(|mut phase| {
//!     let tile = phase.alloc_workgroup::<F32>();
//!     let pending = phase.cooperative_load(tile);
//!     pending.read();
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
//! use phase_token_prototype::{build, F32};
//!
//! build(|mut phase| {
//!     let tile = phase.alloc_workgroup::<F32>();
//!     let pending = phase.cooperative_load(tile);
//!     drop(pending);
//!     phase.finish()
//! });
//! ```
//!
//! Once a body has produced its final `Synced` witness, it has consumed the
//! phase handle, so it cannot emit more IR afterward:
//!
//! ```compile_fail
//! use phase_token_prototype::{build, F32};
//!
//! build(|phase| {
//!     phase.range_step(
//!         |phase, _| {
//!             let synced = phase.sync_end();
//!             let _late = phase.alloc_workgroup::<F32>();
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
//! use phase_token_prototype::{build, F32, ReadyTile};
//!
//! build(|phase| {
//!     let mut leaked: Option<ReadyTile<'_, '_, F32>> = None;
//!
//!     phase.range_step(
//!         |mut phase, _| {
//!             let tile = phase.alloc_workgroup::<F32>();
//!             let pending = phase.cooperative_load(tile);
//!             let (ready, phase) = pending.sync_tile();
//!             leaked = Some(ready);
//!             phase.sync_end()
//!         },
//!         |mut phase| {
//!             phase.read(&leaked.unwrap());
//!             phase.finish()
//!         },
//!     )
//! });
//! ```
//!
//! A tile made ready before a loop can be read inside the loop body:
//!
//! ```
//! use phase_token_prototype::{build, F32};
//!
//! build(|mut phase| {
//!     let tile = phase.alloc_workgroup::<F32>();
//!     let pending = phase.cooperative_load(tile);
//!     let (ready, phase) = pending.sync_tile();
//!
//!     phase.range_step(
//!         |mut phase, _| {
//!             phase.read(&ready);
//!             phase.sync_end()
//!         },
//!         |phase| phase.finish(),
//!     )
//! });
//! ```

mod api;
mod ir;
mod lower;

pub use api::{
    build, Clean, Dim, KernelBuilder, KernelDone, Pending, PendingTile, Phase, ReadyTile, Synced,
    UninitTile, F32,
};
pub use ir::{Event, KernelIr, TileId};
pub use lower::{LowerError, NagaKernel};

#[cfg(test)]
mod tests;
