//! `fusor-autograd` — reverse mode as a Logical -> Logical transform whose output is
//! ingested **together with the forward as one graph with one root set**.
//!
//! *Why Logical*: adjoints are facts about tensor algebra. `d(Contract) =
//! (grad @ Bt, At @ grad)` holds regardless of tile geometry.
//! *Why not rewrite rules*: an adjoint is a directed transformation, not an
//! equality; putting `grad` in the primal's chain is unsound.
//! Forward and backward share a graph so kernel selection can fuse across
//! their boundary and reuse intermediate storage.

#![warn(unreachable_pub)]

mod adjoints;
pub mod backward;
mod contract;
pub mod custom;
mod map_adjoint;
mod rules;
mod structural;
pub mod tape;

pub use adjoints::ADJOINTS;
pub use map_adjoint::map_adjoint;
pub use rules::ADJOINT_RULES;
pub use tape::GraphTape;
