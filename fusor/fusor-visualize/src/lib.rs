//! A deterministic memory model, with text and terminal views over the same steps.
//! Addresses are buffer-relative bytes. This is an explanatory model, not a timing
//! model: subgroup assignment, scheduling, cache and banks are explicit assumptions.

pub mod demo;
pub mod kernel;
mod model;
pub mod plan;
pub mod render;
mod simulator;

pub use model::*;
pub use simulator::Simulator;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Error {}

pub(crate) fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}
