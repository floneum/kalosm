//! `fusor2-autograd` — reverse mode as an L0 -> L0 transform whose output is
//! ingested together with the forward as one graph with one root set.
//!
//! Adjoints are directed transformations, not equalities, so they are a table
//! ([`ADJOINTS`]) rather than rewrite rules. Gradient checkpointing falls out
//! of the extractor's materialization bit on the combined graph.

pub mod adjoints;
pub mod backward;
pub mod contract;
pub mod custom;
pub mod map_adjoint;
pub mod rules;
pub mod structural;
pub mod tape;

pub use adjoints::ADJOINTS;
pub use map_adjoint::map_adjoint;
pub use rules::ADJOINT_RULES;
pub use tape::GraphTape;
