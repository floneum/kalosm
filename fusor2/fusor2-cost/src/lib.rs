//! `fusor2-cost` — a scalar picosecond roofline parameterized on measured
//! device facts, and the extraction that resolves node selection,
//! materialization and schedule point together against it.
//!
//! The cost is a single scalar. Precision is a verifier property
//! (`NumericContract`), not a cost term.

pub mod tune_cache;
pub mod cache;
pub mod extract;
pub mod facts;
pub mod lower_bound;
pub mod model;
pub mod moves;
pub mod plan;
pub mod realize;
pub mod replay;
pub mod terms;
pub mod verify_plan;

pub use extract::LocalSearch;
pub use model::Roofline;
pub use replay::ReplayMemo;
