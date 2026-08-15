//! `fusor2-conformance` — the tests crate, and the only thing that can
//! falsify the design.
//!
//! Op x backward matrix against CPU and GPU; launch-count asserts
//! (`resolves_in::<N>`) for eight named backward shapes, so a non-firing
//! fusion rule is a hard failure rather than a quiet 5-10x regression; a debug
//! ILP extraction oracle that must agree with the local search on small graphs
//! — because a greedy search compared only against itself cannot distinguish
//! "found the optimum" from "made the same mistake as the reference";
//! `PlanHash` goldens; and an `--exhaustive` mode.
//!
//! # Not here: the betlang trainer gate
//!
//! This crate used to carry a second acceptance half — MSQ1 export cases and a
//! gate over betlang's `trainer/` crate — built by `#[path]`-compiling
//! `trainer/src/{arch,export}.rs` verbatim and reading betlang's shipped
//! `assets/magika/source-student-q4.bin`. Both reach outside this workspace,
//! so they did not survive the move into this repository. What they falsified
//! (a byte-identical export from the *shipped* writer, and the trainer
//! building against fusor2 with zero source changes) has no coverage here;
//! it lives with the trainer, in betlang.

pub mod compare;
pub mod exhaustive;
pub mod goldens;
pub mod harness;
pub mod launch_counts;
pub mod oracle;
pub mod suite;

pub use compare::{allclose, assert_close};
pub use harness::Harness;
pub use suite::REGISTRY;
