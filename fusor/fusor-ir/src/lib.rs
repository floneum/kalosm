//! `fusor-ir` — the shared contracts every other fusor crate is written
//! against. Three levels (Logical `tensor`, Launch `nest`, Kernel `tile`), one acyclic
//! append-only e-graph spanning Logical/Launch, one scalar picosecond cost model, one
//! extraction. Almost nothing here decides anything: every type is either a
//! *description* or a *contract* a downstream crate implements.
//!
//! The two things only the IR can own — total inference/verification for the
//! closed `Logical`/`Launch` enums, and the shared rewrite-rule set with its saturation
//! driver — live in [`semantics`], [`verify_l0()`], [`verify_launch()`], [`saturate`]
//! and [`rules`].

#![warn(unreachable_pub)]

pub mod error;

pub mod dtype;
pub mod facts;
pub mod scalar;
pub mod shape;

pub mod ir;

pub mod autograd;
pub mod cost;
pub mod device;
pub mod egraph;
pub mod extract;
pub mod target;

pub mod carrier;
pub mod contract_spec;
pub mod semantics;
mod verify_l0;
pub mod verify_launch;

mod rule_macro;
pub mod rules;
pub mod saturate;

pub use error::{Error, Result};
pub use rules::CORE_RULES;
pub use saturate::Driver;
pub use semantics::CoreSemantics;
pub use verify_l0::verify_l0;
pub use verify_launch::verify_launch;
