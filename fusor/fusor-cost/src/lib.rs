//! `fusor-cost`: one picosecond roofline over measured device facts, and the
//! extraction that resolves selection, materialization and schedule against it.

#![warn(unreachable_pub)]

mod debug;
pub mod extract;
pub mod facts;
pub mod forward;
mod lower_bound;
mod model;
mod moves;
mod nodes;
pub mod plan;
mod quantized;
pub mod realize;
pub mod replay;
mod terms;
pub mod tune_cache;
#[cfg(feature = "compiler-tests")]
mod verify_plan;
pub mod wavefront;

pub use extract::LocalSearch;
pub use model::Roofline;
pub use replay::ReplayMemo;
