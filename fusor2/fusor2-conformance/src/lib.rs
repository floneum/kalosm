//! `fusor2-conformance` — the tests crate, and the only thing that can
//! falsify the design.
//!
//! Op x backward matrix against CPU and GPU; launch-count asserts
//! (`resolves_in::<N>`) for eight named backward shapes, so a non-firing
//! fusion rule is a hard failure rather than a quiet 5-10x regression; a debug
//! ILP extraction oracle that must agree with the local search on small graphs
//! — because a greedy search compared only against itself cannot distinguish
//! "found the optimum" from "made the same mistake twice"; `PlanHash`
//! goldens; and an `--exhaustive` mode.
//!
//! # Not covered here
//!
//! The MSQ1 byte-identical export and the gate that builds the betlang trainer
//! against fusor2 both reach outside this workspace. They live with the
//! trainer, in betlang.

pub mod compare;
pub mod exhaustive;
pub mod goldens;
pub mod harness;
pub mod launch_counts;
pub mod oracle;
pub mod suite;

pub use compare::assert_close;
pub use suite::REGISTRY;
