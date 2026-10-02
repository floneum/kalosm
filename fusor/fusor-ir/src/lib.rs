//! `fusor-ir` — the shared contracts every fusor crate is written against:
//! three levels (Logical, Launch, Kernel), one acyclic append-only e-graph,
//! one picosecond cost model, one extraction, and the shared rewrite rules.

#![warn(unreachable_pub)]

pub mod error;

pub mod dtype;
pub mod facts;
pub mod scalar;
pub mod shape;

pub mod ir;
pub mod packing;

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
