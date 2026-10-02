//! `fusor-conformance` — the tests crate, and the only thing that can
//! falsify the design.
//!
//! Op x backward matrix against CPU and GPU, in the fuzzing style: every case
//! runs several times with re-sampled shapes, and every resolve races every
//! e-class member of every launch (`FUSOR_VERIFY_MEMBERS`), so a case covers
//! the class of kernels the compiler could emit. Constructor and structural
//! invariant tests complement these numerical comparisons.

pub mod bench;
pub mod compare;
pub mod harness;
pub mod suite;

pub use compare::{allclose, assert_close};
pub use harness::Harness;
pub use suite::REGISTRY;
