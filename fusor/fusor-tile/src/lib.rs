//! `fusor-tile` — the algorithms over one kernel body (liveness, arena
//! packing, barriers, uniformity, verification) plus the schedule-domain
//! generators and the rules that need exact workgroup bytes.

#![warn(unreachable_pub)]

mod arena;
mod barrier;
pub mod build;
pub mod domains;
mod flags;
mod liveness;
pub mod planner;
pub mod rules;
mod uniformity;
mod verify_arena;
mod verify_kernel;

pub use planner::Planner;
pub use rules::SCHED_RULES;
pub use verify_kernel::verify_kernel;
