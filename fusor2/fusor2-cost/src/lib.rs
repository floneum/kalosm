//! `fusor2-cost` — one scalar picosecond roofline parameterized on *measured*
//! device facts, and the one extraction that resolves node selection,
//! materialization and schedule point together against it.
//!
//! Not a lexicographic tuple: the reference's own unit test shows the tuple
//! gives the wrong verdict, and its own doc concedes dispatches are 0.2% of
//! modelled time while the tuple will pay unbounded bandwidth to remove one.
//!
//! Precision is **not** a cost term — it is a verifier property
//! (`NumericContract`), because a time-only model eliminates f32 everywhere.

pub mod tune_cache;
pub mod cache;
pub mod calibrate;
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

pub use calibrate::Calibrator;
pub use extract::LocalSearch;
pub use model::Roofline;
pub use replay::ReplayMemo;
