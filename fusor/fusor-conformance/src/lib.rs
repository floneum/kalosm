//! `fusor-conformance` — the tests crate, and the only thing that can
//! falsify the design.
//!
//! Op x backward matrix against CPU and GPU, in the fuzzing style: every case
//! runs several times with re-sampled shapes, and every resolve races every
//! e-class member of every launch (`FUSOR_VERIFY_MEMBERS`), so a case covers
//! the *class* of kernels the compiler could emit rather than whichever member
//! extraction happened to pick. There are no structural asserts on rule
//! firings, launch counts or plan hashes — correctness of every candidate
//! kernel at random sizes is the whole contract.

pub mod compare;
pub mod harness;
pub mod suite;

pub use compare::{allclose, assert_close};
pub use harness::Harness;
pub use suite::REGISTRY;
