//! `fusor2-tile` — everything that reasons about one kernel body and about the
//! schedule-parameter space of one node.
//!
//! Owns the Kernel algorithms (liveness, workgroup arena packing, barrier
//! insertion argmin, all-pairs arena verification, uniformity analysis, and
//! full Kernel type-checking.
//! Also owns the schedule-domain generators and the Launch lowering rules that
//! consult them, because a rule whose legality filter is exact workgroup bytes
//! must live with the function that computes them.

#![warn(unreachable_pub)]

mod arena;
mod barrier;
pub mod domains;
mod liveness;
pub mod planner;
pub mod rules;
mod uniformity;
mod verify_arena;
mod verify_kernel;

pub use planner::Planner;
pub use rules::SCHED_RULES;
pub use verify_kernel::verify_kernel;
