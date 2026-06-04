//! The conformance suite as a library.
//!
//! Every case has a single definition here, reused by two consumers:
//!   * `cargo test`, via the `#[tokio::test]` wrappers that [`registry`]
//!     generates from its case list (plus the hand-written `#[cfg(test)]`
//!     modules for the panic / vision cases that aren't in the registry), and
//!   * the in-browser WebGPU suite ([`webgpu`]), which the kalosm-chat web app
//!     runs on its `/conformance` route.
//!
//! There is no separate `tests/` directory: the test entry points live next to
//! the case definitions so a case is defined exactly once.
//!
//! Cases the browser actually executes live in always-compiled modules so they
//! build for `wasm32`. The remaining cases rely on the full CPU/GPU differential
//! and only ever run under the native test harness, so they are gated behind
//! `#[cfg(not(target_arch = "wasm32"))]` to keep them out of the browser build.

pub mod registry;
pub mod webgpu;

/// The promoted conformance cases. Always compiled so the browser suite can run
/// the same list the native harness runs. Individual cases that depend on host
/// facilities unavailable in the browser are gated inside the module tree.
pub mod native;
