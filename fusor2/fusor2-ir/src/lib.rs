//! `fusor2-ir` — the shared contracts every other fusor2 crate is written
//! against. Three levels (L0 `tensor`, L1 `nest`, L2 `tile`), one acyclic
//! append-only e-graph spanning L0/L1, one scalar picosecond cost model, one
//! extraction. Almost nothing here decides anything: every type is either a
//! *description* or a *contract* a downstream crate implements.
//!
//! The two things only the IR can own — total inference/verification for the
//! closed `L0`/`L1` enums, and the shared rewrite-rule set with its saturation
//! driver — live in [`semantics`], [`verify_l0`], [`verify_l1`], [`saturate`]
//! and [`rules`].

pub mod error;

pub mod dtype;
pub mod shape;
pub mod facts;
pub mod scalar;

pub mod ir;

pub mod egraph;
pub mod device;
pub mod cost;
pub mod extract;
pub mod target;
pub mod autograd;

pub mod carrier;
pub mod contract_spec;
pub mod semantics;
pub mod verify_l0;
pub mod verify_l1;

pub mod rule_macro;
pub mod rules;
pub mod saturate;

pub use error::{Error, Result};
pub use rules::CORE_RULES;
pub use saturate::Driver;
pub use semantics::CoreSemantics;
pub use verify_l0::verify_l0;
pub use verify_l1::verify_l1;
