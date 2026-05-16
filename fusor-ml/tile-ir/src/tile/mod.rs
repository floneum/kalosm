//! Ergonomic builder API for tile programs.
//!
//! A [`Program`] declares storage buffers and workgroup scratch storage. Each
//! [`Program::program_grid`] call receives a [`TileBlock`] whose expressions
//! describe one logical lane in the workgroup.
//!
//! ```
//! use fusor_tile_ir::{tile, Shape, F32};
//!
//! let ir = tile::build(|program| {
//!     let x = program.storage_read::<F32, 2>(Shape::new([1, 64]));
//!     let y = program.storage_write::<F32, 2>(Shape::new([1, 64]));
//!     program.program_grid::<64>([1, 1, 1], |block| {
//!         let lane = block.lane();
//!         let mask = lane.clone().lt(64u32);
//!         let value = block.load(x.at((0u32, lane.clone())), mask.clone(), 0.0);
//!         block.store(y.at((0u32, lane)), value.relu(), mask);
//!     });
//! });
//! # let _ = ir;
//! ```

mod block;
mod coop;
mod grid;
mod program;
pub mod quantized;
mod reduce;
mod storage;
mod value;

pub use block::TileBlock;
pub use coop::{CoopRole, CoopTileLoad};
pub use grid::build;
pub use program::Program;
pub use quantized::QuantizedDot;
pub use storage::{RuntimeElement, Storage, StorageIndex};
pub use value::{range, Address, CoopAcc, CoopFragment, FoldIter, Local, Mask, Tile, Workgroup};
