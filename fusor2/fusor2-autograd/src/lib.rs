//! `fusor2-autograd` — reverse mode as an L0 -> L0 transform whose output is
//! ingested **together with the forward as one graph with one root set**.
//!
//! *Why L0*: adjoints are facts about tensor algebra. `d(Contract) =
//! (grad @ Bt, At @ grad)` holds regardless of tile geometry.
//! *Why not rewrite rules*: an adjoint is a directed transformation, not an
//! equality; putting `grad` in the primal's chain is unsound.
//! *Why one graph*: gradient checkpointing is then the extractor's
//! materialization bit. Nobody writes a checkpointing pass and there is no
//! user annotation.
//!
//! Seven [`ADJOINTS`] entries. No `Arc<dyn Fn>` closures, no
//! type-erased downcasts.

pub mod adjoints;
pub mod backward;
pub mod contract;
pub mod custom;
pub mod map_adjoint;
pub mod rules;
pub mod structural;
pub mod tape;

pub use adjoints::ADJOINTS;
pub use backward::Reverse;
pub use map_adjoint::map_adjoint;
pub use rules::ADJOINT_RULES;
pub use tape::GraphTape;
