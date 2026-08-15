//! `fusor2-tile` — everything that reasons about one kernel body and about the
//! schedule-parameter space of one node.
//!
//! Owns the L2 algorithms (liveness, workgroup arena packing, barrier
//! elision/insertion argmin, all-pairs arena verification, uniformity analysis,
//! full L2 type-check) and the hash-consed L2 term builders both emitters share.
//! Also owns the schedule-domain generators and the L1 lowering rules that
//! consult them, because a rule whose legality filter is exact workgroup bytes
//! must live with the function that computes them.

pub mod arena;
pub mod barrier;
pub mod build;
pub mod domains;
pub mod liveness;
pub mod planner;
pub mod rules;
pub mod uniformity;
pub mod verify_arena;
pub mod verify_l2;

pub use planner::Planner;
pub use rules::SCHED_RULES;
pub use verify_l2::verify_l2;
